// P1 interop test for the zero-copy DML migration (docs/zero-copy-migration.md).
//
// Proves the transfer chain on the app's GPU/driver, standalone (no app code):
//   GL R32F source texture (224x672, CHW target layout, deterministic pattern)
//     -> wglDXRegisterObjectNV'd D3D11 texture (NV interop, same call sequence
//        as D2S/gpu/dx_interop.py: register -> lock -> glCopyImageSubData -> unlock)
//     -> [if the NT-shared texture itself is not registerable] D3D11
//        CopySubresourceRegion stagingTex -> sharedTex (same-device GPU copy)
//     -> IDXGIResource1::CreateSharedHandle (NT handle, NTHANDLE|KEYEDMUTEX)
//     -> D3D12 OpenSharedHandle -> CopyTextureRegion -> READBACK heap -> Map
//     -> byte-compare vs the CPU pattern.
//
// First runs a registration matrix (format x misc flags) to pin down which
// D3D11 textures this driver accepts for interop registration, then picks:
//   - direct:  GL copy straight into the NT-shared texture (if registerable)
//   - staging: GL copy into a registerable texture, D3D11 copy into shared
//
// Gates:
//   [A0] GL roundtrip of the source texture (upload sanity)
//   [A1] GL-side readback of the registered texture (copy really landed)
//   [B]  D3D12 readback through the NT handle (cross-API visibility)
//
// usage: p1_interop_test.exe [width height]   (default 224 672)
// exit 0 = PASS (A0+A1+B all byte-identical)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_5.h>
#include <d3d12.h>
#include <gl/gl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#ifndef GL_R32F
#define GL_R32F 0x822E
#endif

typedef void (APIENTRY* PFNGLCOPYIMAGESUBDATA)(GLuint srcName, GLenum srcTarget, GLint srcLevel,
                                               GLint srcX, GLint srcY, GLint srcZ,
                                               GLuint dstName, GLenum dstTarget, GLint dstLevel,
                                               GLint dstX, GLint dstY, GLint dstZ,
                                               GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth);
typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);
typedef HANDLE(WINAPI* PFNWGLDXOPENDEVICENVPROC)(void* dxDevice);
typedef BOOL(WINAPI* PFNWGLDXCLOSEDEVICENVPROC)(HANDLE hDevice);
typedef HANDLE(WINAPI* PFNWGLDXREGISTEROBJECTNVPROC)(HANDLE hDevice, void* dxObject, GLuint name,
                                                     GLenum type, GLenum access);
typedef BOOL(WINAPI* PFNWGLDXUNREGISTEROBJECTNVPROC)(HANDLE hDevice, HANDLE hObject);
typedef BOOL(WINAPI* PFNWGLDXLOCKOBJECTSNVPROC)(HANDLE hDevice, GLint count, HANDLE* objects);
typedef BOOL(WINAPI* PFNWGLDXUNLOCKOBJECTSNVPROC)(HANDLE hDevice, GLint count, HANDLE* objects);

#define HR(x) do { HRESULT _h = (x); if (FAILED(_h)) { \
    fprintf(stderr, "HRESULT 0x%08lX at %s:%d\n", (unsigned long)_h, __FILE__, __LINE__); return 2; } } while (0)
#define GLCK(tag) do { GLenum _e = glGetError(); if (_e) { \
    fprintf(stderr, "GL error 0x%04X after %s (%s:%d)\n", _e, tag, __FILE__, __LINE__); return 2; } } while (0)

static uint32_t sm32(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return (uint32_t)(x ^ (x >> 31));
}

// Deterministic finite floats in [1,2) with per-pixel hash mantissa + specials at row 0.
static uint32_t pattern_bits(uint32_t x, uint32_t y) {
  switch ((x << 16) | y) {
    case (0u << 16) | 0u: return 0x00000000u;  // +0
    case (1u << 16) | 0u: return 0x80000000u;  // -0
    case (2u << 16) | 0u: return 0x00000001u;  // denormal
    case (3u << 16) | 0u: return 0x00800000u;  // smallest normal
    case (4u << 16) | 0u: return 0x7F7FFFFFu;  // FLT_MAX
    case (5u << 16) | 0u: return 0xFF7FFFFFu;  // -FLT_MAX
  }
  return 0x3F800000u | (sm32(((uint64_t)y << 32) | x) & 0x007FFFFFu);
}

static int cmp_rows(const char* tag, const uint32_t* ref, const uint8_t* got, uint32_t w, uint32_t h,
                    uint32_t rowPitch) {
  uint32_t bad = 0, shown = 0;
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      uint32_t v;
      memcpy(&v, got + (size_t)y * rowPitch + (size_t)x * 4, 4);
      if (v != ref[(size_t)y * w + x]) {
        if (shown < 4) {
          uint32_t r;
          memcpy(&r, &ref[(size_t)y * w + x], 4);
          printf("    %s mismatch @(%u,%u): ref bits %08X got %08X\n", tag, x, y, r, v);
          shown++;
        }
        bad++;
      }
    }
  printf("    %s: %u / %u texels differ\n", tag, bad, w * h);
  return bad ? 1 : 0;
}

// ---- globals shared by matrix + main path (set once in main) ----------------
static PFNWGLDXREGISTEROBJECTNVPROC wglDXRegisterObjectNV_;
static PFNWGLDXCLOSEDEVICENVPROC wglDXCloseDeviceNV_;
static PFNWGLDXUNREGISTEROBJECTNVPROC wglDXUnregisterObjectNV_;
static PFNWGLDXLOCKOBJECTSNVPROC wglDXLockObjectsNV_;
static PFNWGLDXUNLOCKOBJECTSNVPROC wglDXUnlockObjectsNV_;

// Try create+register a W x H texture; returns S_OK if registered (and unregistered again),
// else the failing HRESULT (create) or 1 (register failed).
static HRESULT try_combo(ID3D11Device* dev, HANDLE dxh, DXGI_FORMAT fmt, UINT misc, uint32_t w, uint32_t h) {
  D3D11_TEXTURE2D_DESC td = {0};
  td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
  td.Format = fmt; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  td.MiscFlags = misc;
  ID3D11Texture2D* t = NULL;
  HRESULT hr = dev->CreateTexture2D(&td, NULL, &t);
  if (FAILED(hr)) return hr;
  GLuint name; glGenTextures(1, &name);
  HANDLE hObj = wglDXRegisterObjectNV_(dxh, t, name, GL_TEXTURE_2D, 0x0001);
  GLenum e = glGetError();
  if (hObj) {
    wglDXLockObjectsNV_(dxh, 1, &hObj);
    wglDXUnlockObjectsNV_(dxh, 1, &hObj);
    wglDXUnregisterObjectNV_(dxh, hObj);
    glDeleteTextures(1, &name);
    t->Release();
    return S_OK;
  }
  glDeleteTextures(1, &name);
  t->Release();
  return e ? (HRESULT)(0x80000000u | e) : (HRESULT)1;  // 1 = register refused, no GL error
}

int main(int argc, char** argv) {
  const uint32_t W = argc > 2 ? (uint32_t)atoi(argv[1]) : 224;
  const uint32_t H = argc > 2 ? (uint32_t)atoi(argv[2]) : 672;
  if (argc == 2) { fprintf(stderr, "usage: %s [width height]\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("p1_interop_test  %ux%u R32F\n", W, H);

  // ---- 1. hidden window + WGL context --------------------------------------
  WNDCLASSA wc = {0};
  wc.lpfnWndProc = DefWindowProcA; wc.hInstance = GetModuleHandleA(NULL);
  wc.lpszClassName = "p1interop";
  RegisterClassA(&wc);
  HWND hwnd = CreateWindowA("p1interop", "p1", WS_OVERLAPPEDWINDOW, 0, 0, 8, 8,
                            NULL, NULL, wc.hInstance, NULL);
  HDC hdc = GetDC(hwnd);
  PIXELFORMATDESCRIPTOR pfd = {0};
  pfd.nSize = sizeof pfd; pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
  pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 24; pfd.cDepthBits = 0; pfd.cStencilBits = 0;
  int pf = ChoosePixelFormat(hdc, &pfd);
  if (!pf || !SetPixelFormat(hdc, pf, &pfd)) { fprintf(stderr, "pixelformat failed\n"); return 1; }
  HGLRC hgl = wglCreateContext(hdc);
  if (!hgl || !wglMakeCurrent(hdc, hgl)) { fprintf(stderr, "wglCreate/MakeCurrent failed\n"); return 1; }

  PFNWGLGETEXTENSIONSSTRINGEXTPROC wglGetExtStr =
      (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");
  const char* wglExt = wglGetExtStr ? wglGetExtStr() : "";
  printf("[wgl] interop ext: %s\n", strstr(wglExt, "WGL_NV_DX_interop") ? "WGL_NV_DX_interop YES" : "MISSING!");
  if (!strstr(wglExt, "WGL_NV_DX_interop")) return 1;
  PFNGLCOPYIMAGESUBDATA glCopyImageSubData_ = (PFNGLCOPYIMAGESUBDATA)wglGetProcAddress("glCopyImageSubData");
  if (!glCopyImageSubData_) { fprintf(stderr, "glCopyImageSubData not exported\n"); return 1; }
  PFNWGLDXOPENDEVICENVPROC wglDXOpenDeviceNV_ = (PFNWGLDXOPENDEVICENVPROC)wglGetProcAddress("wglDXOpenDeviceNV");
  wglDXCloseDeviceNV_ = (PFNWGLDXCLOSEDEVICENVPROC)wglGetProcAddress("wglDXCloseDeviceNV");
  wglDXRegisterObjectNV_ = (PFNWGLDXREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXRegisterObjectNV");
  wglDXUnregisterObjectNV_ = (PFNWGLDXUNREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXUnregisterObjectNV");
  wglDXLockObjectsNV_ = (PFNWGLDXLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXLockObjectsNV");
  wglDXUnlockObjectsNV_ = (PFNWGLDXUNLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXUnlockObjectsNV");
  if (!wglDXOpenDeviceNV_ || !wglDXCloseDeviceNV_ || !wglDXRegisterObjectNV_ || !wglDXLockObjectsNV_ ||
      !wglDXUnlockObjectsNV_ || !wglDXUnregisterObjectNV_) { fprintf(stderr, "NV interop entry points missing\n"); return 1; }

  // ---- 2. GL source texture with pattern ------------------------------------
  uint32_t* cpu = (uint32_t*)malloc((size_t)W * H * 4);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) cpu[(size_t)y * W + x] = pattern_bits(x, y);
  GLuint srcTex;
  glGenTextures(1, &srcTex);
  glBindTexture(GL_TEXTURE_2D, srcTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (GLsizei)W, (GLsizei)H, 0, GL_RED, GL_FLOAT, cpu);
  GLCK("glTexImage2D src");

  uint32_t* glA = (uint32_t*)malloc((size_t)W * H * 4);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, glA);
  GLCK("glGetTexImage src");
  printf("[A0] source texture GL roundtrip:\n");
  int a0 = cmp_rows("A0", cpu, (uint8_t*)glA, W, H, W * 4);
  if (a0) { printf("P1 RESULT: FAIL (A0)\n"); return 3; }

  // ---- 3. D3D11 device -------------------------------------------------------
  ID3D11Device* d3d11 = NULL; ID3D11DeviceContext* d3d11ctx = NULL;
  HR(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0,
                       D3D11_SDK_VERSION, &d3d11, NULL, &d3d11ctx));
  DXGI_ADAPTER_DESC ad = {0};
  { IDXGIDevice* gd = NULL; IDXGIAdapter* adapter = NULL;
    HR(d3d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&gd));
    HR(gd->GetAdapter(&adapter));
    HR(adapter->GetDesc(&ad));
    adapter->Release(); gd->Release(); }
  printf("[d3d11] adapter LUID %08llX-%08llX\n", (unsigned long long)ad.AdapterLuid.HighPart,
         (unsigned long long)ad.AdapterLuid.LowPart);

  HANDLE dxh = wglDXOpenDeviceNV_(d3d11);
  if (!dxh) { fprintf(stderr, "wglDXOpenDeviceNV failed\n"); return 2; }

  // ---- 3b. NT-shared texture + registration ladder ---------------------------
  // Rung 1/2: register the NT-shared texture directly (rung 2 = fresh interop
  //           session: observed driver state pollution after many reg/unreg
  //           cycles makes the first register of a fresh session unreliable).
  // Rung 3/4: staging texture (registers cleanly), D3D11 CopySubresourceRegion
  //           staging -> shared afterwards (same-device GPU copy, ~0.02ms).
  D3D11_TEXTURE2D_DESC td = {0};
  td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
  td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
  ID3D11Texture2D* tShared = NULL;
  HR(d3d11->CreateTexture2D(&td, NULL, &tShared));

  ID3D11Texture2D* tStaging = NULL;
  GLuint dstTex = 0;
  HANDLE hObj = NULL;
  const char* arch = "";
  for (int attempt = 0; attempt < 4 && !hObj; attempt++) {
    if (attempt) {  // fresh interop session + fresh GL name
      wglDXCloseDeviceNV_(dxh);
      dxh = wglDXOpenDeviceNV_(d3d11);
      if (dstTex) glDeleteTextures(1, &dstTex);
      glGenTextures(1, &dstTex);
      printf("[ladder] rung%d: fresh interop session%s\n", attempt + 1,
             attempt == 2 ? " + plain warm-up" : attempt == 3 ? " + keyedmutex warm-up" : "");
    } else {
      glGenTextures(1, &dstTex);
    }
    if (attempt == 2) {  // warm-up theory A: any prior register/unregister cycle
      try_combo(d3d11, dxh, DXGI_FORMAT_R32_FLOAT, 0, 4, 4);
    }
    if (attempt == 3) {  // warm-up theory B: specifically a KEYEDMUTEX cycle first
      try_combo(d3d11, dxh, DXGI_FORMAT_R32_FLOAT, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, 4, 4);
    }
    hObj = wglDXRegisterObjectNV_(dxh, tShared, dstTex, GL_TEXTURE_2D, 0x0001);
    if (hObj) arch = "DIRECT";
  }
  for (int m = 0; m < 2 && !hObj; m++) {  // staging fallback: plain, then legacy-SHARED
    wglDXCloseDeviceNV_(dxh);
    dxh = wglDXOpenDeviceNV_(d3d11);
    if (dstTex) glDeleteTextures(1, &dstTex);
    glGenTextures(1, &dstTex);
    if (tStaging) { tStaging->Release(); tStaging = NULL; }
    D3D11_TEXTURE2D_DESC sd = td;
    sd.MiscFlags = m ? D3D11_RESOURCE_MISC_SHARED : 0;
    HR(d3d11->CreateTexture2D(&sd, NULL, &tStaging));
    printf("[ladder] rung%d: staging tex (misc=0x%X)\n", 3 + m, sd.MiscFlags);
    hObj = wglDXRegisterObjectNV_(dxh, tStaging, dstTex, GL_TEXTURE_2D, 0x0001);
    if (hObj) arch = m ? "STAGING(shared)" : "STAGING(plain)";
  }
  if (!hObj) { fprintf(stderr, "all register rungs failed\n"); return 2; }
  printf("[arch] %s\n", arch);
  const BOOL direct = (arch[0] == 'D');
  ID3D11Texture2D* tReg = direct ? tShared : tStaging;

  // ---- 4. interop register + copy -------------------------------------------
  if (!wglDXLockObjectsNV_(dxh, 1, &hObj)) { fprintf(stderr, "wglDXLockObjectsNV failed\n"); return 2; }
  glCopyImageSubData_(srcTex, GL_TEXTURE_2D, 0, 0, 0, 0, dstTex, GL_TEXTURE_2D, 0, 0, 0, 0,
                      (GLsizei)W, (GLsizei)H, 1);
  GLCK("glCopyImageSubData");
  glFinish();

  glBindTexture(GL_TEXTURE_2D, dstTex);  // [A1] GL-side readback of the registered texture
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, glA);
  GLCK("glGetTexImage dst");
  wglDXUnlockObjectsNV_(dxh, 1, &hObj);
  printf("[A1] registered texture via GL readback:\n");
  int a1 = cmp_rows("A1", cpu, (uint8_t*)glA, W, H, W * 4);
  if (a1) { printf("P1 RESULT: FAIL (A1: copy into interop texture)\n"); return 3; }

  if (!direct) {  // staging -> shared, GPU copy inside D3D11
    // KEYEDMUTEX resource: D3D11 writes without AcquireSync are silently
    // deferred/dropped (observed: D3D12 read back all zeros). Protocol:
    // AcquireSync(0) -> copy -> Flush -> ReleaseSync(0); D3D12 side ignores
    // the mutex (it can't acquire), ordering comes from fence serialization.
    IDXGIKeyedMutex* km = NULL;
    HR(tShared->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km));
    HR(km->AcquireSync(0, INFINITE));
    d3d11ctx->CopySubresourceRegion(tShared, 0, 0, 0, 0, tStaging, 0, NULL);
    d3d11ctx->Flush();
    HR(km->ReleaseSync(0));
    km->Release();
    printf("[stage] CopySubresourceRegion under keyed mutex\n");
  }
  d3d11ctx->Flush();
  Sleep(50);  // test-only ordering; production shim will use a shared fence

  // ---- 5. NT handle -> D3D12 readback ---------------------------------------
  HANDLE hShared = NULL;
  { IDXGIResource1* r1 = NULL;
    HR(tShared->QueryInterface(__uuidof(IDXGIResource1), (void**)&r1));
    HRESULT hr = r1->CreateSharedHandle(NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &hShared);
    r1->Release();
    if (FAILED(hr) || !hShared) { fprintf(stderr, "CreateSharedHandle failed 0x%08lX\n", (unsigned long)hr); return 2; }
  }

  ID3D12Device* dev12 = NULL; ID3D12CommandQueue* queue = NULL;
  ID3D12CommandAllocator* alloc = NULL; ID3D12GraphicsCommandList* list = NULL;
  ID3D12Resource* r12 = NULL;
  { IDXGIDevice* gd = NULL; IDXGIAdapter* adapter = NULL;
    HR(d3d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&gd));
    HR(gd->GetAdapter(&adapter));
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev12);
    adapter->Release(); gd->Release();
    HR(hr); }
  D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  HR(dev12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue));
  HR(dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc));
  HR(dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&list));
  HR(dev12->OpenSharedHandle(hShared, __uuidof(ID3D12Resource), (void**)&r12));
  printf("[d3d12] OpenSharedHandle OK\n");

  const UINT rowPitch = ((UINT)W * 4 + 255u) & ~255u;
  const UINT bufSize = rowPitch * H;
  ID3D12Resource* rb = NULL;
  D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC bd = {};
  bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bufSize; bd.Height = 1;
  bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
  bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  HR(dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                    NULL, __uuidof(ID3D12Resource), (void**)&rb));

  D3D12_TEXTURE_COPY_LOCATION locSrc = {}, locDst = {};
  locSrc.pResource = r12; locSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; locSrc.SubresourceIndex = 0;
  locDst.pResource = rb; locDst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  locDst.PlacedFootprint.Offset = 0;
  locDst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
  locDst.PlacedFootprint.Footprint.Width = W; locDst.PlacedFootprint.Footprint.Height = H;
  locDst.PlacedFootprint.Footprint.Depth = 1; locDst.PlacedFootprint.Footprint.RowPitch = rowPitch;
  D3D12_RESOURCE_BARRIER bar = {};
  bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  bar.Transition.pResource = r12; bar.Transition.Subresource = 0;
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;  // implicit promotion COMMON->COPY_SOURCE
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  list->ResourceBarrier(1, &bar);
  list->CopyTextureRegion(&locDst, 0, 0, 0, &locSrc, NULL);
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  list->ResourceBarrier(1, &bar);
  HR(list->Close());
  ID3D12CommandList* exec[] = {list};
  queue->ExecuteCommandLists(1, exec);

  ID3D12Fence* fence = NULL; HANDLE ev = CreateEventA(NULL, FALSE, FALSE, NULL);
  HR(dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence));
  HR(queue->Signal(fence, 1));
  HR(fence->SetEventOnCompletion(1, ev));
  WaitForSingleObject(ev, 10000);
  void* mapped = NULL;
  D3D12_RANGE whole = {0, (SIZE_T)bufSize};
  HR(rb->Map(0, &whole, &mapped));
  printf("[B] D3D12 readback via NT handle (RowPitch %u, pad %u/row):\n", rowPitch, rowPitch - W * 4);
  int b = cmp_rows("B", cpu, (uint8_t*)mapped, W, H, rowPitch);
  rb->Unmap(0, NULL);

  printf(b ? "P1 RESULT: FAIL (B: D3D12 visibility)\n" : "P1 RESULT: PASS\n");

  if (!b && !direct) {  // staging-copy cost: 200 acquire+copy+release cycles
    LARGE_INTEGER f, a, z;
    QueryPerformanceFrequency(&f);
    IDXGIKeyedMutex* km = NULL;
    if (SUCCEEDED(tShared->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km))) {
      QueryPerformanceCounter(&a);
      for (int i = 0; i < 200; i++) {
        km->AcquireSync(0, INFINITE);
        d3d11ctx->CopySubresourceRegion(tShared, 0, 0, 0, 0, tStaging, 0, NULL);
        km->ReleaseSync(0);
      }
      d3d11ctx->Flush();
      QueryPerformanceCounter(&z);
      printf("[perf] staging copy: %.1f us/op (acquire+copy+release, flush amortized)\n",
             (z.QuadPart - a.QuadPart) * 1e6 / f.QuadPart / 200.0);
      km->Release();
    }
  }

  // ---- 6. registration matrix (informational only, run last) -----------------
  struct { UINT misc; const char* name; } combos[] = {
    {0, "no-misc"},
    {D3D11_RESOURCE_MISC_SHARED, "SHARED(legacy)"},
    {D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, "KEYEDMUTEX"},
    {D3D11_RESOURCE_MISC_SHARED_NTHANDLE, "NTHANDLE"},
    {D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, "NTH|KM"},
  };
  DXGI_FORMAT fmts[] = {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM};
  printf("[matrix] create+register (R32F / BGRA8), %ux%u:\n", W, H);
  for (int f = 0; f < 2; f++) {
    printf("    %-6s", f ? "BGRA8" : "R32F");
    for (int c = 0; c < 5; c++) {
      HRESULT r = try_combo(d3d11, dxh, fmts[f], combos[c].misc, W, H);
      printf("  %-14s=%s", combos[c].name, SUCCEEDED(r) ? "OK" : "fail");
    }
    printf("\n");
  }
  return b ? 3 : 0;
}
