// P2 zero-copy inference test for the DML migration (docs/zero-copy-migration.md).
//
// Welds P0 (ORT DML API) and P1 (GL -> D3D11 -> D3D12 interop) together:
//   GL R32F texture 192x784 (flat CHW tiling: 192*4=768B row = 256-aligned,
//   784*192 = 150528 texels = 3*224*224 exactly -> single dense copy)
//     -> glCopyImageSubData -> registered staging D3D11 tex (P1 arch)
//     -> keyed-mutex CopySubresourceRegion -> NTH|KM shared tex
//     -> D3D12 OpenSharedHandle; per frame: shared-fence wait -> CopyTextureRegion
//        into UAV buffer (602112 B, dense)
//     -> OrtDmlApi::CreateGPUAllocationFromD3DResource(buffer)
//     -> OrtValue("DML" memory info) -> Run on a DML session that was created
//        with OUR IDMLDevice + command queue (DML1) -> queue FIFO orders the
//        copy before the inference reads.
//
// Gates:
//   [A0] GL roundtrip of the source texture
//   [A1] GL readback of the staging (registered) texture
//   [C]  zero-copy output vs CPU-input output on the SAME session: MAE < 1e-5
//        (the model itself validates every input byte - strongest check)
//   [T]  timing: zero-copy Run vs CPU-input Run, plus full per-frame chain.
//
// Also dumps the zero-copy output as .npy for the Python-path cross-check
// (p2_ref.py compares against the app's own onnxruntime-directml session).
//
// usage: p2_zerocopy_test.exe [ortdll] [model] [out.npy]
//   defaults: app capi dll, runtime/depth-224.onnx, _p2_zc.npy next to the exe

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <windows.h>
#include <d3d11.h>
#include <d3d11_3.h>
#include <d3d11_4.h>
#include <dxgi1_5.h>
#include <d3d12.h>
#include <DirectML.h>
#include <gl/gl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#ifndef GL_R32F
#define GL_R32F 0x822E
#endif

typedef void (APIENTRY* PFNGLCOPYIMAGESUBDATA)(GLuint, GLenum, GLint, GLint, GLint, GLint,
                                               GLuint, GLenum, GLint, GLint, GLint, GLint,
                                               GLsizei, GLsizei, GLsizei);
typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);
typedef HANDLE(WINAPI* PFNWGLDXOPENDEVICENVPROC)(void*);
typedef BOOL(WINAPI* PFNWGLDXCLOSEDEVICENVPROC)(HANDLE);
typedef HANDLE(WINAPI* PFNWGLDXREGISTEROBJECTNVPROC)(HANDLE, void*, GLuint, GLenum, GLenum);
typedef BOOL(WINAPI* PFNWGLDXUNREGISTEROBJECTNVPROC)(HANDLE, HANDLE);
typedef BOOL(WINAPI* PFNWGLDXLOCKOBJECTSNVPROC)(HANDLE, GLint, HANDLE*);
typedef BOOL(WINAPI* PFNWGLDXUNLOCKOBJECTSNVPROC)(HANDLE, GLint, HANDLE*);

#define HR(x) do { HRESULT _h = (x); if (FAILED(_h)) { \
    fprintf(stderr, "HRESULT 0x%08lX at %s:%d\n", (unsigned long)_h, __FILE__, __LINE__); return 2; } } while (0)
#define GLCK(tag) do { GLenum _e = glGetError(); if (_e) { \
    fprintf(stderr, "GL error 0x%04X after %s (%s:%d)\n", _e, tag, __FILE__, __LINE__); return 2; } } while (0)

static const uint32_t TW = 192, TH = 784;   // 192*4=768B row (256-aligned), 784*192=150528
static const uint64_t TENSOR_BYTES = 602112ull;  // 3*224*224*4

static uint32_t sm32(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return (uint32_t)(x ^ (x >> 31));
}
// Same pattern family as p1 but model-safe: keep +-0 / denormal / smallest-normal
// (bit-exactness stress), but +-FLT_MAX overflows the depth network's activations
// (observed: CPU-input reference output NaN). +-10 is far outside ImageNet range
// yet harmless. Python reference (p2_ref.py) must use the SAME function.
static uint32_t pattern_bits(uint32_t flat) {
  switch (flat) {
    case 0: return 0x00000000u;  // +0
    case 1: return 0x80000000u;  // -0
    case 2: return 0x00000001u;  // denormal
    case 3: return 0x00800000u;  // smallest normal
    case 4: return 0x41200000u;  // +10.0f
    case 5: return 0xC1200000u;  // -10.0f
  }
  return 0x3F800000u | (sm32(0xD2B74407B1CE7E35ull + flat) & 0x007FFFFFu);
}

static double now_ms(void) {
  static LARGE_INTEGER f = {0};
  LARGE_INTEGER c;
  if (!f.QuadPart) QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return c.QuadPart * 1000.0 / f.QuadPart;
}

static int write_npy_f32(const char* path, const float* data, size_t n) {
  // (1,1,224,224) float32 LE; header total length rounded up to a 64-byte
  // multiple (dict itself is 60 chars, so the total is 128, hlen 118).
  const char* dict = "{'descr':'<f4','fortran_order':False,'shape':(1,1,224,224),}";
  const size_t dl = strlen(dict);
  const size_t total = ((10 + dl + 1 + 63) / 64) * 64;
  uint16_t hlen = (uint16_t)(total - 10);
  char hdr[256];
  memset(hdr, ' ', total);
  memcpy(hdr, "\x93NUMPY\x01\x00", 8);
  memcpy(hdr + 8, &hlen, 2);
  memcpy(hdr + 10, dict, dl);
  hdr[total - 1] = '\n';
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  fwrite(hdr, 1, total, f);
  size_t w = fwrite(data, 4, n, f);
  fclose(f);
  return w == n ? 0 : -1;
}

static const OrtApi* g_ort = NULL;
#define OCHK(stmt) do { OrtStatus* _s = (stmt); if (_s) { \
    fprintf(stderr, "ORT error %s:%d code=%d: %s\n", __FILE__, __LINE__, \
            (int)g_ort->GetErrorCode(_s), g_ort->GetErrorMessage(_s)); \
    g_ort->ReleaseStatus(_s); return 2; } } while (0)

int main(int argc, char** argv) {
  const char* ortDllA = argc > 1 ? argv[1] : "E:\\Projects\\Desktop2Stereo\\_dml\\onnxruntime\\capi\\onnxruntime.dll";
  const char* mdlA = argc > 2 ? argv[2] : "E:\\Projects\\Desktop2Stereo\\runtime\\depth-224.onnx";
  const char* npyA = argc > 3 ? argv[3] : "_p2_zc.npy";
  wchar_t ortDllW[512], mdlW[512];
  MultiByteToWideChar(CP_UTF8, 0, ortDllA, -1, ortDllW, 512);
  MultiByteToWideChar(CP_UTF8, 0, mdlA, -1, mdlW, 512);
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("p2_zerocopy_test  tex %ux%u  tensor %llu B\n", TW, TH, (unsigned long long)TENSOR_BYTES);
  printf("  dll=%ls\n  model=%ls\n", ortDllW, mdlW);

  // ---- 1. GL context + source pattern --------------------------------------
  WNDCLASSA wc = {0};
  wc.lpfnWndProc = DefWindowProcA; wc.hInstance = GetModuleHandleA(NULL); wc.lpszClassName = "p2zc";
  RegisterClassA(&wc);
  HWND hwnd = CreateWindowA("p2zc", "p2", WS_OVERLAPPEDWINDOW, 0, 0, 8, 8, NULL, NULL, wc.hInstance, NULL);
  HDC hdc = GetDC(hwnd);
  PIXELFORMATDESCRIPTOR pfd = {0};
  pfd.nSize = sizeof pfd; pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
  pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 24;
  int pf = ChoosePixelFormat(hdc, &pfd);
  if (!pf || !SetPixelFormat(hdc, pf, &pfd) || !wglMakeCurrent(hdc, wglCreateContext(hdc))) {
    fprintf(stderr, "GL init failed\n"); return 1;
  }
  PFNGLCOPYIMAGESUBDATA glCopyImageSubData_ = (PFNGLCOPYIMAGESUBDATA)wglGetProcAddress("glCopyImageSubData");
  PFNWGLDXOPENDEVICENVPROC wglDXOpenDeviceNV_ = (PFNWGLDXOPENDEVICENVPROC)wglGetProcAddress("wglDXOpenDeviceNV");
  PFNWGLDXCLOSEDEVICENVPROC wglDXCloseDeviceNV_ = (PFNWGLDXCLOSEDEVICENVPROC)wglGetProcAddress("wglDXCloseDeviceNV");
  PFNWGLDXREGISTEROBJECTNVPROC wglDXRegisterObjectNV_ = (PFNWGLDXREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXRegisterObjectNV");
  PFNWGLDXUNREGISTEROBJECTNVPROC wglDXUnregisterObjectNV_ = (PFNWGLDXUNREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXUnregisterObjectNV");
  PFNWGLDXLOCKOBJECTSNVPROC wglDXLockObjectsNV_ = (PFNWGLDXLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXLockObjectsNV");
  PFNWGLDXUNLOCKOBJECTSNVPROC wglDXUnlockObjectsNV_ = (PFNWGLDXUNLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXUnlockObjectsNV");
  if (!glCopyImageSubData_ || !wglDXOpenDeviceNV_ || !wglDXRegisterObjectNV_ || !wglDXLockObjectsNV_ ||
      !wglDXUnlockObjectsNV_ || !wglDXUnregisterObjectNV_ || !wglDXCloseDeviceNV_) {
    fprintf(stderr, "GL/interop entry points missing\n"); return 1;
  }

  uint32_t* cpu = (uint32_t*)malloc(TENSOR_BYTES);
  for (uint32_t i = 0; i < TW * TH; i++) cpu[i] = pattern_bits(i);
  GLuint srcTex;
  glGenTextures(1, &srcTex);
  glBindTexture(GL_TEXTURE_2D, srcTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (GLsizei)TW, (GLsizei)TH, 0, GL_RED, GL_FLOAT, cpu);
  GLCK("glTexImage2D src");
  uint32_t* glA = (uint32_t*)malloc(TENSOR_BYTES);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, glA);
  GLCK("glGetTexImage src");
  uint32_t badA0 = 0;
  for (uint32_t i = 0; i < TW * TH; i++) badA0 += (glA[i] != cpu[i]);
  printf("[A0] GL roundtrip: %u / %u differ\n", badA0, TW * TH);
  if (badA0) { printf("P2 RESULT: FAIL (A0)\n"); return 3; }

  // ---- 2. D3D11 + staging + shared texture (P1 architecture) ----------------
  ID3D11Device* d3d11 = NULL; ID3D11DeviceContext* d3d11ctx = NULL;
  HR(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0,
                       D3D11_SDK_VERSION, &d3d11, NULL, &d3d11ctx));
  D3D11_TEXTURE2D_DESC td = {0};
  td.Width = TW; td.Height = TH; td.MipLevels = 1; td.ArraySize = 1;
  td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  ID3D11Texture2D *tStaging = NULL, *tShared = NULL;
  { D3D11_TEXTURE2D_DESC sd = td; sd.MiscFlags = 0;
    HR(d3d11->CreateTexture2D(&sd, NULL, &tStaging)); }
  td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
  HR(d3d11->CreateTexture2D(&td, NULL, &tShared));

  HANDLE dxh = wglDXOpenDeviceNV_(d3d11);
  if (!dxh) { fprintf(stderr, "wglDXOpenDeviceNV failed\n"); return 2; }
  GLuint dstTex; glGenTextures(1, &dstTex);
  HANDLE hObj = wglDXRegisterObjectNV_(dxh, tStaging, dstTex, GL_TEXTURE_2D, 0x0001);
  if (!hObj) { fprintf(stderr, "register staging failed\n"); return 2; }
  if (!wglDXLockObjectsNV_(dxh, 1, &hObj)) { fprintf(stderr, "lock failed\n"); return 2; }
  glCopyImageSubData_(srcTex, GL_TEXTURE_2D, 0, 0, 0, 0, dstTex, GL_TEXTURE_2D, 0, 0, 0, 0,
                      (GLsizei)TW, (GLsizei)TH, 1);
  GLCK("glCopyImageSubData");
  glFinish();
  glBindTexture(GL_TEXTURE_2D, dstTex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, glA);
  GLCK("glGetTexImage staging");
  wglDXUnlockObjectsNV_(dxh, 1, &hObj);
  uint32_t badA1 = 0;
  for (uint32_t i = 0; i < TW * TH; i++) badA1 += (glA[i] != cpu[i]);
  printf("[A1] staging via GL: %u / %u differ\n", badA1, TW * TH);
  if (badA1) { printf("P2 RESULT: FAIL (A1)\n"); return 3; }

  // ---- 3. D3D12 on the same adapter + queue + shared fence ------------------
  ID3D12Device* dev12 = NULL; ID3D12CommandQueue* queue = NULL;
  ID3D12CommandAllocator* alloc12 = NULL; ID3D12Resource *rTex = NULL, *rBuf = NULL;
  { IDXGIDevice* gd = NULL; IDXGIAdapter* adapter = NULL;
    HR(d3d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&gd));
    HR(gd->GetAdapter(&adapter));
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev12);
    adapter->Release(); gd->Release(); HR(hr); }
  D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  HR(dev12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue));
  HR(dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc12));

  HANDLE hTexNT = NULL;
  { IDXGIResource1* r1 = NULL;
    HR(tShared->QueryInterface(__uuidof(IDXGIResource1), (void**)&r1));
    HRESULT hr = r1->CreateSharedHandle(NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &hTexNT);
    r1->Release(); HR(hr); }
  HR(dev12->OpenSharedHandle(hTexNT, __uuidof(ID3D12Resource), (void**)&rTex));

  D3D12_HEAP_PROPERTIES hpDef = {}; hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC bd = {};
  bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = TENSOR_BYTES; bd.Height = 1;
  bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
  bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  HR(dev12->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd,
                                    D3D12_RESOURCE_STATE_COMMON, NULL, __uuidof(ID3D12Resource), (void**)&rBuf));

  // shared fence: D3D11 signals after the staging->shared copy, D3D12 queue waits
  ID3D11Device5* d3d11_5 = NULL; ID3D11DeviceContext4* ctx4 = NULL;
  ID3D11Fence* fence11 = NULL; ID3D12Fence* fence12 = NULL;
  HR(d3d11->QueryInterface(__uuidof(ID3D11Device5), (void**)&d3d11_5));
  HR(d3d11ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&ctx4));
  HR(d3d11_5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&fence11));
  HANDLE hFence = NULL;
  HR(fence11->CreateSharedHandle(NULL, GENERIC_ALL, NULL, &hFence));
  HR(dev12->OpenSharedHandle(hFence, __uuidof(ID3D12Fence), (void**)&fence12));

  // ---- 4. ORT DML session on OUR device + queue (DML1) ----------------------
  HMODULE hOrt = LoadLibraryExW(ortDllW, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hOrt) { fprintf(stderr, "LoadLibrary ORT failed %lu\n", GetLastError()); return 1; }
  const OrtApiBase* (*pGetBase)(void) =
      (const OrtApiBase * (*)(void))(uintptr_t) GetProcAddress(hOrt, "OrtGetApiBase");
  if (!pGetBase) { fprintf(stderr, "no OrtGetApiBase\n"); return 1; }
  g_ort = (const OrtApi*)pGetBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) { fprintf(stderr, "GetApi(%d) NULL\n", ORT_API_VERSION); return 1; }
  const void* epapi = NULL;
  OCHK(g_ort->GetExecutionProviderApi("DML", 1, &epapi));
  const OrtDmlApi* dml = (const OrtDmlApi*)epapi;
  printf("[ort] %s, OrtDmlApi @ %p\n", pGetBase()->GetVersionString(), epapi);

  // IDMLDevice from the app's own DirectML.dll on OUR D3D12 device
  wchar_t dmlDllW[512];
  wcscpy_s(dmlDllW, 512, ortDllW);
  wchar_t* slash = wcsrchr(dmlDllW, L'\\');
  wcscpy_s(slash ? slash + 1 : dmlDllW, 256, L"DirectML.dll");
  HMODULE hDmlDll = LoadLibraryExW(dmlDllW, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hDmlDll) { fprintf(stderr, "LoadLibrary DirectML(%ls) failed %lu\n", dmlDllW, GetLastError()); return 1; }
  typedef HRESULT(WINAPI * PFN_DMLCREATEDEVICE)(IUnknown*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
  PFN_DMLCREATEDEVICE DMLCreateDevice_ = (PFN_DMLCREATEDEVICE)GetProcAddress(hDmlDll, "DMLCreateDevice");
  if (!DMLCreateDevice_) { fprintf(stderr, "no DMLCreateDevice export\n"); return 1; }
  IDMLDevice* dmlDev = NULL;
  HR(DMLCreateDevice_(dev12, DML_CREATE_DEVICE_FLAG_NONE, __uuidof(IDMLDevice), (void**)&dmlDev));
  printf("[dml] IDMLDevice on our D3D12 device (%ls)\n", dmlDllW);

  OrtEnv* env = NULL;
  OCHK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "p2zc", &env));
  OrtSessionOptions* so = NULL;
  OCHK(g_ort->CreateSessionOptions(&so));
  OCHK(dml->SessionOptionsAppendExecutionProvider_DML1(so, dmlDev, queue));  // OUR queue
  OrtSession* session = NULL;
  OCHK(g_ort->CreateSession(env, mdlW, so, &session));
  OrtAllocator* oalloc = NULL;
  OCHK(g_ort->GetAllocatorWithDefaultOptions(&oalloc));
  char *inName = NULL, *outName = NULL;
  OCHK(g_ort->SessionGetInputName(session, 0, oalloc, &inName));
  OCHK(g_ort->SessionGetOutputName(session, 0, oalloc, &outName));
  OrtTypeInfo* ti = NULL;
  OCHK(g_ort->SessionGetInputTypeInfo(session, 0, &ti));
  const OrtTensorTypeAndShapeInfo* tsi = NULL;
  OCHK(g_ort->CastTypeInfoToTensorInfo(ti, &tsi));
  size_t ndim = 0; int64_t dims[8] = {0};
  OCHK(g_ort->GetDimensionsCount(tsi, &ndim));
  OCHK(g_ort->GetDimensions(tsi, dims, ndim));
  size_t count = 1;
  printf("[ort] in \"%s\" dims=[", inName);
  for (size_t i = 0; i < ndim; i++) { count *= (size_t)dims[i]; printf("%lld%s", (long long)dims[i], i + 1 < ndim ? "," : ""); }
  printf("] elems=%zu\n", count);
  if (count * 4ull != TENSOR_BYTES) { fprintf(stderr, "model input size mismatch\n"); return 2; }

  // ---- 5. OrtValues: CPU reference + zero-copy (DML alloc over rBuf) --------
  OrtMemoryInfo *miCpu = NULL, *miDml = NULL;
  OCHK(g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &miCpu));
  OCHK(g_ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &miDml));
  OrtValue *inCpu = NULL, *inZC = NULL;
  OCHK(g_ort->CreateTensorWithDataAsOrtValue(miCpu, cpu, TENSOR_BYTES, dims, ndim,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inCpu));
  void* zcAlloc = NULL;
  OCHK(dml->CreateGPUAllocationFromD3DResource(rBuf, &zcAlloc));
  OCHK(g_ort->CreateTensorWithDataAsOrtValue(miDml, zcAlloc, TENSOR_BYTES, dims, ndim,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inZC));
  printf("[zc] CreateGPUAllocationFromD3DResource -> %p\n", zcAlloc);
  const char* inNames[1] = {inName};
  const char* outNames[1] = {outName};

  // ---- 6. per-frame chain + reference runs ----------------------------------
  IDXGIKeyedMutex* km = NULL;
  HR(tShared->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km));
  ID3D12GraphicsCommandList* cl = NULL;
  HR(dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc12, NULL,
                              __uuidof(ID3D12GraphicsCommandList), (void**)&cl));
  HR(cl->Close());  // lists are created in recording state; close so run_frame can Reset
  ID3D12Fence* doneFence = NULL; HANDLE doneEv = CreateEventA(NULL, FALSE, FALSE, NULL);
  HR(dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&doneFence));
  UINT64 frame = 1;
  OrtValue* outZC = NULL;

  // full chain once, gates on. NOTE: lambda-local failures must return a FAILED
  // HRESULT - a bare `return 2` is a success code and silently swallows errors.
#define HRL(x) do { HRESULT _h = (x); if (FAILED(_h)) { \
    fprintf(stderr, "HRESULT 0x%08lX at %s:%d (frame)\n", (unsigned long)_h, __FILE__, __LINE__); \
    return E_FAIL; } } while (0)
  auto run_frame = [&](void) -> HRESULT {
    HRL(km->AcquireSync(0, INFINITE));
    d3d11ctx->CopySubresourceRegion(tShared, 0, 0, 0, 0, tStaging, 0, NULL);
    HRL(ctx4->Signal(fence11, frame));
    HRL(km->ReleaseSync(0));
    d3d11ctx->Flush();

    HRL(queue->Wait(fence12, frame));          // ordering: D3D11 copy first
    HRL(cl->Reset(alloc12, NULL));
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = rTex; bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cl->ResourceBarrier(1, &bar);
    D3D12_TEXTURE_COPY_LOCATION ls = {}, ld = {};
    ls.pResource = rTex; ls.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; ls.SubresourceIndex = 0;
    ld.pResource = rBuf; ld.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    ld.PlacedFootprint.Offset = 0;
    ld.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
    ld.PlacedFootprint.Footprint.Width = TW; ld.PlacedFootprint.Footprint.Height = TH;
    ld.PlacedFootprint.Footprint.Depth = 1;
    ld.PlacedFootprint.Footprint.RowPitch = TW * 4;  // 768, already 256-aligned
    cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, NULL);
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    cl->ResourceBarrier(1, &bar);
    HRL(cl->Close());
    ID3D12CommandList* exec[] = {cl};
    queue->ExecuteCommandLists(1, exec);      // same queue as DML1 -> FIFO before Run
    frame++;
    return S_OK;
  };

  HR(run_frame());
  OCHK(g_ort->Run(session, NULL, inNames, (const OrtValue* const*)&inZC, 1, outNames, 1, &outZC));
  HR(queue->Signal(doneFence, frame));
  HR(doneFence->SetEventOnCompletion(frame, doneEv));
  WaitForSingleObject(doneEv, 10000);

  // CPU-input reference on the same session
  OrtValue* outRef = NULL;
  OCHK(g_ort->Run(session, NULL, inNames, (const OrtValue* const*)&inCpu, 1, outNames, 1, &outRef));

  float *dZC = NULL, *dRef = NULL;
  OCHK(g_ort->GetTensorMutableData(outZC, (void**)&dZC));
  OCHK(g_ort->GetTensorMutableData(outRef, (void**)&dRef));
  size_t on = 1; OrtTensorTypeAndShapeInfo* osi = NULL;
  OCHK(g_ort->GetTensorTypeAndShape(outZC, &osi));
  OCHK(g_ort->GetDimensionsCount(osi, &ndim));
  OCHK(g_ort->GetDimensions(osi, dims, ndim));
  for (size_t i = 0; i < ndim; i++) on *= (size_t)dims[i];
  g_ort->ReleaseTensorTypeAndShapeInfo(osi);
  double mae = 0, mx = 0; size_t bitEq = 0, nonfinite = 0;
  for (size_t i = 0; i < on; i++) {
    if (!isfinite(dZC[i])) { nonfinite++; continue; }
    double d = fabs((double)dZC[i] - (double)dRef[i]);
    mae += d; if (d > mx) mx = d;
    if (memcmp(&dZC[i], &dRef[i], 4) == 0) bitEq++;
  }
  mae /= (double)on;
  printf("[C] zero-copy vs CPU-input: MAE=%.3g  maxAbs=%.3g  bitEqual=%zu/%zu  nonfinite=%zu\n",
         mae, mx, bitEq, on, nonfinite);
  int passC = (mae < 1e-5) && (nonfinite == 0);

  // ---- 7. timing -------------------------------------------------------------
  for (int w = 0; w < 3; w++) { run_frame(); OCHK(g_ort->Run(session, NULL, inNames, (const OrtValue* const*)&inZC, 1, outNames, 1, &outZC)); }
  double bestChain = 1e9, bestRunZC = 1e9, bestRunCpu = 1e9, t0, a;
  for (int r = 0; r < 10; r++) {
    t0 = now_ms(); HR(run_frame());
    a = now_ms();
    OCHK(g_ort->Run(session, NULL, inNames, (const OrtValue* const*)&inZC, 1, outNames, 1, &outZC));
    double b = now_ms();
    if (b - t0 < bestChain) bestChain = b - t0;
    if (b - a < bestRunZC) bestRunZC = b - a;
  }
  for (int r = 0; r < 10; r++) {
    a = now_ms();
    OCHK(g_ort->Run(session, NULL, inNames, (const OrtValue* const*)&inCpu, 1, outNames, 1, &outRef));
    double b = now_ms();
    if (b - a < bestRunCpu) bestRunCpu = b - a;
  }
  printf("[T] best: chain %.2fms  Run(zero-copy) %.2fms  Run(CPU-input) %.2fms  (P0 measured 4.15)\n",
         bestChain, bestRunZC, bestRunCpu);

  if (write_npy_f32(npyA, dZC, on) == 0) printf("[npy] %s written (%zu floats)\n", npyA, on);
  else printf("[npy] WRITE FAILED\n");

  printf(passC ? "P2 RESULT: PASS\n" : "P2 RESULT: FAIL (C)\n");
  return passC ? 0 : 3;
}
