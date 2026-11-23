// d2s_dml_shim.dll — 零拷贝 DML 深度推理 shim（P3 集成，P0/P1/P2 验证链的产品化封装）。
//
// 链路（P2 已逐位验证）：
//   app 的 _prep_tex（GL R32F flat CHW，192x784 @res224）
//     -> wglDXRegisterObjectNV 的无 misc staging D3D11 纹理（lock/copy/unlock）
//     -> keyed mutex 协议 CopySubresourceRegion -> NTH|KM 共享纹理
//     -> D3D12 OpenSharedHandle；共享 fence Wait -> CopyTextureRegion 进 UAV buffer
//     -> OrtDmlApi::CreateGPUAllocationFromD3DResource(buffer) -> "DML" OrtValue
//     -> ORT Run（session 用 DML1 注入自有 IDMLDevice+DIRECT 队列，与拷贝同队列 FIFO）
//     -> 输出 CPU float[res*res] 拷给调用方
//
// 线程约定：create/run 必须在持有目标 GL 上下文的线程调用（wglDXOpenDeviceNV
// 绑定当前上下文；app 的渲染线程满足）。失败路径：create 返回 NULL 并填 err，
// python 侧回退 glReadPixels 路径。
//
// ABI（ctypes 友好，全部 __cdecl）：
//   void* d2s_shim_create(wchar_t* ort_dll, wchar_t* dml_dll, wchar_t* model,
//                         uint32 res, uint32 tex_w, uint32 tex_h,
//                         int64 luid /*0=默认适配器*/, wchar_t* err, size_t errlen)
//   int    d2s_shim_run(void* h, uint32 src_glo, float* out, size_t out_bytes) // 0=ok
//   void   d2s_shim_times(void* h, double* t_input_ms, double* t_run_ms)
//   void   d2s_shim_destroy(void* h)

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
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <DirectML.h>
#include <gl/gl.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"

typedef void (APIENTRY* PFNGLCOPYIMAGESUBDATA)(GLuint, GLenum, GLint, GLint, GLint, GLint,
                                               GLuint, GLenum, GLint, GLint, GLint, GLint,
                                               GLsizei, GLsizei, GLsizei);
typedef HANDLE(WINAPI* PFNWGLDXOPENDEVICENVPROC)(void*);
typedef BOOL(WINAPI* PFNWGLDXCLOSEDEVICENVPROC)(HANDLE);
typedef HANDLE(WINAPI* PFNWGLDXREGISTEROBJECTNVPROC)(HANDLE, void*, GLuint, GLenum, GLenum);
typedef BOOL(WINAPI* PFNWGLDXUNREGISTEROBJECTNVPROC)(HANDLE, HANDLE);
typedef BOOL(WINAPI* PFNWGLDXLOCKOBJECTSNVPROC)(HANDLE, GLint, HANDLE*);
typedef BOOL(WINAPI* PFNWGLDXUNLOCKOBJECTSNVPROC)(HANDLE, GLint, HANDLE*);
typedef HRESULT(WINAPI* PFN_DMLCREATEDEVICE)(IUnknown*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);

struct Shim {
  // geometry
  uint32_t res, tw, th;
  uint64_t bytes;
  // GL interop（当前线程的 GL 上下文）
  HMODULE gl32 = NULL;
  PFNGLCOPYIMAGESUBDATA glCopyImageSubData = NULL;
  PFNWGLDXOPENDEVICENVPROC wglDXOpenDevice = NULL;
  PFNWGLDXCLOSEDEVICENVPROC wglDXCloseDevice = NULL;
  PFNWGLDXREGISTEROBJECTNVPROC wglDXRegisterObject = NULL;
  PFNWGLDXUNREGISTEROBJECTNVPROC wglDXUnregisterObject = NULL;
  PFNWGLDXLOCKOBJECTSNVPROC wglDXLockObjects = NULL;
  PFNWGLDXUNLOCKOBJECTSNVPROC wglDXUnlockObjects = NULL;
  HANDLE dxh = NULL;
  GLuint dstTex = 0;
  HANDLE hObj = NULL;
  // D3D11
  ID3D11Device* d3d11 = NULL;
  ID3D11DeviceContext* ctx11 = NULL;
  ID3D11DeviceContext4* ctx4 = NULL;
  ID3D11Texture2D *tStaging = NULL, *tShared = NULL;
  IDXGIKeyedMutex* km = NULL;
  ID3D11Fence* fence11 = NULL;
  // D3D12
  ID3D12Device* dev12 = NULL;
  ID3D12CommandQueue* queue = NULL;
  ID3D12CommandAllocator* alloc12 = NULL;
  ID3D12GraphicsCommandList* cl = NULL;
  ID3D12Resource *rTex = NULL, *rBuf = NULL;
  ID3D12Fence* fence12 = NULL;
  UINT64 frame = 1;
  // ORT / DML
  HMODULE hOrt = NULL, hDmlDll = NULL;
  const OrtApi* ort = NULL;
  const OrtDmlApi* dmlapi = NULL;
  OrtEnv* env = NULL;
  OrtSession* session = NULL;
  char *inName = NULL, *outName = NULL;
  OrtAllocator* oalloc = NULL;
  OrtMemoryInfo* miDml = NULL;
  void* zcAlloc = NULL;
  OrtValue* inZC = NULL;
  IDMLDevice* dmlDev = NULL;
  // timing（毫秒，最近一帧）：lock / glCopy+finish / unlock / km+fence+flush / d3d12 wait+exec / run
  double t_input = 0, t_run = 0;
  double t6[6] = {0};

  ~Shim() { destroy(); }

  void destroy() {
    if (ort) {
      if (inZC) ort->ReleaseValue(inZC);
      if (zcAlloc && dmlapi) dmlapi->FreeGPUAllocation(zcAlloc);
      if (miDml) ort->ReleaseMemoryInfo(miDml);
      if (inName && oalloc) oalloc->Free(oalloc, inName);
      if (outName && oalloc) oalloc->Free(oalloc, outName);
      if (session) ort->ReleaseSession(session);
      if (env) ort->ReleaseEnv(env);
    }
    if (dmlDev) dmlDev->Release();
    if (hObj && wglDXUnregisterObject) wglDXUnregisterObject(dxh, hObj);
    if (dxh && wglDXCloseDevice) wglDXCloseDevice(dxh);
    if (dstTex && gl32) glDeleteTextures(1, &dstTex);
    if (rBuf) rBuf->Release();
    if (rTex) rTex->Release();
    if (fence12) fence12->Release();
    if (cl) cl->Release();
    if (alloc12) alloc12->Release();
    if (queue) queue->Release();
    if (dev12) dev12->Release();
    if (fence11) fence11->Release();
    if (km) km->Release();
    if (tShared) tShared->Release();
    if (tStaging) tStaging->Release();
    if (ctx11) ctx11->Release();
    if (d3d11) d3d11->Release();
    if (hDmlDll) FreeLibrary(hDmlDll);
    if (hOrt) FreeLibrary(hOrt);
    memset(this, 0, sizeof *this);
  }
};

static void set_err(wchar_t* err, size_t n, const char* msg) {
  if (err && n) MultiByteToWideChar(CP_UTF8, 0, msg, -1, err, (int)n);
}

static double qpc_ms(void) {
  static LARGE_INTEGER f = {0};
  LARGE_INTEGER c;
  if (!f.QuadPart) QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return c.QuadPart * 1000.0 / f.QuadPart;
}

extern "C" {

__declspec(dllexport) void* __cdecl d2s_shim_create(
    const wchar_t* ort_dll, const wchar_t* dml_dll, const wchar_t* model,
    unsigned int res, unsigned int tex_w, unsigned int tex_h,
    long long luid, wchar_t* err, size_t errlen) {
  Shim* s = NULL;
  try {
    if (!ort_dll || !dml_dll || !model || !res) { set_err(err, errlen, "bad args"); return NULL; }
    if ((uint64_t)tex_w * tex_h != (uint64_t)res * res * 3 || tex_w % 64u) {
      set_err(err, errlen, "prep geometry must be W*H==3*res^2 with W%64==0 (flat CHW, 256-aligned rows)");
      return NULL;
    }
    s = new Shim();
    s->res = res; s->tw = tex_w; s->th = tex_h;
    s->bytes = (uint64_t)res * res * 3 * 4;

    // ---- wgl 入口（需要当前线程已有 GL 上下文）----
    s->gl32 = GetModuleHandleA("opengl32.dll");
    if (!s->gl32) { set_err(err, errlen, "opengl32 not loaded"); delete s; return NULL; }
    s->wglDXOpenDevice = (PFNWGLDXOPENDEVICENVPROC)wglGetProcAddress("wglDXOpenDeviceNV");
    s->wglDXCloseDevice = (PFNWGLDXCLOSEDEVICENVPROC)wglGetProcAddress("wglDXCloseDeviceNV");
    s->wglDXRegisterObject = (PFNWGLDXREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXRegisterObjectNV");
    s->wglDXUnregisterObject = (PFNWGLDXUNREGISTEROBJECTNVPROC)wglGetProcAddress("wglDXUnregisterObjectNV");
    s->wglDXLockObjects = (PFNWGLDXLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXLockObjectsNV");
    s->wglDXUnlockObjects = (PFNWGLDXUNLOCKOBJECTSNVPROC)wglGetProcAddress("wglDXUnlockObjectsNV");
    s->glCopyImageSubData = (PFNGLCOPYIMAGESUBDATA)wglGetProcAddress("glCopyImageSubData");
    if (!s->wglDXOpenDevice || !s->wglDXRegisterObject || !s->wglDXLockObjects ||
        !s->wglDXUnlockObjects || !s->wglDXUnregisterObject || !s->glCopyImageSubData) {
      set_err(err, errlen, "NV interop / glCopyImageSubData unavailable (no GL context or driver)");
      delete s; return NULL;
    }

    // ---- D3D11（按 LUID 选卡；0 = 默认）----
    IDXGIAdapter* adapter = NULL;
    if (luid) {
      IDXGIFactory4* fct = NULL;
      HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&fct);
      if (FAILED(hr)) { set_err(err, errlen, "CreateDXGIFactory1 failed"); delete s; return NULL; }
      for (UINT i = 0; ; i++) {
        IDXGIAdapter1* a = NULL;
        if (fct->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d;
        a->GetDesc1(&d);
        if (d.AdapterLuid.HighPart == (int)(luid >> 32) &&
            d.AdapterLuid.LowPart == (unsigned)(luid & 0xFFFFFFFFULL)) { adapter = a; break; }
        a->Release();
      }
      fct->Release();
      if (!adapter) { set_err(err, errlen, "adapter with given LUID not found"); delete s; return NULL; }
    }
    HRESULT hr = D3D11CreateDevice(adapter, adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
                                   NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0, D3D11_SDK_VERSION,
                                   &s->d3d11, NULL, &s->ctx11);
    if (adapter) adapter->Release();
    if (FAILED(hr)) { set_err(err, errlen, "D3D11CreateDevice failed"); delete s; return NULL; }
    s->ctx11->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&s->ctx4);

    D3D11_TEXTURE2D_DESC td = {0};
    td.Width = tex_w; td.Height = tex_h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    D3D11_TEXTURE2D_DESC sd = td; sd.MiscFlags = 0;
    if (FAILED(s->d3d11->CreateTexture2D(&sd, NULL, &s->tStaging))) {
      set_err(err, errlen, "staging texture create failed"); delete s; return NULL; }
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    if (FAILED(s->d3d11->CreateTexture2D(&td, NULL, &s->tShared))) {
      set_err(err, errlen, "shared texture create failed"); delete s; return NULL; }
    if (FAILED(s->tShared->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&s->km))) {
      set_err(err, errlen, "no keyed mutex on shared texture"); delete s; return NULL; }

    // ---- interop 注册（P1 架构：staging 无 misc，注册 100% 可靠）----
    s->dxh = s->wglDXOpenDevice(s->d3d11);
    if (!s->dxh) { set_err(err, errlen, "wglDXOpenDeviceNV failed (GL ctx / adapter mismatch?)"); delete s; return NULL; }
    glGenTextures(1, &s->dstTex);
    s->hObj = s->wglDXRegisterObject(s->dxh, s->tStaging, s->dstTex, 0x0DE1 /*GL_TEXTURE_2D*/, 0x0001);
    if (!s->hObj) { set_err(err, errlen, "wglDXRegisterObjectNV failed"); delete s; return NULL; }
    if (!s->wglDXLockObjects(s->dxh, 1, &s->hObj) || !s->wglDXUnlockObjects(s->dxh, 1, &s->hObj)) {
      set_err(err, errlen, "interop lock/unlock probe failed"); delete s; return NULL;
    }

    // ---- D3D12 同卡 + 队列 + 共享纹理/缓冲 ----
    { IDXGIDevice* gd = NULL; IDXGIAdapter* ad = NULL;
      if (FAILED(s->d3d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&gd)) ||
          FAILED(gd->GetAdapter(&ad)) ||
          FAILED(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&s->dev12))) {
        if (gd) gd->Release();
        set_err(err, errlen, "D3D12CreateDevice failed"); delete s; return NULL;
      }
      ad->Release(); gd->Release();
    }
    // 高优先级队列：DML Run 是同步等待，游戏占满 GPU 时会被拖到十几~几十 ms
    // （2026-11-22 实测 run 4.0→12~39ms）。高优先级让拷贝/Run 先拿到调度窗口。
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_PRIORITY_HIGH, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    s->dev12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&s->queue);
    s->dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     __uuidof(ID3D12CommandAllocator), (void**)&s->alloc12);
    s->dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s->alloc12, NULL,
                                __uuidof(ID3D12GraphicsCommandList), (void**)&s->cl);
    s->cl->Close();  // 创建即打开态；先 Close 让每帧可以 Reset
    HANDLE hNT = NULL;
    { IDXGIResource1* r1 = NULL;
      if (FAILED(s->tShared->QueryInterface(__uuidof(IDXGIResource1), (void**)&r1)) ||
          FAILED(r1->CreateSharedHandle(NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                        NULL, &hNT))) {
        if (r1) r1->Release();
        set_err(err, errlen, "CreateSharedHandle failed"); delete s; return NULL;
      }
      r1->Release();
    }
    if (FAILED(s->dev12->OpenSharedHandle(hNT, __uuidof(ID3D12Resource), (void**)&s->rTex))) {
      CloseHandle(hNT); set_err(err, errlen, "OpenSharedHandle(tex) failed"); delete s; return NULL; }
    CloseHandle(hNT);
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = s->bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(s->dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_COMMON, NULL,
                                                 __uuidof(ID3D12Resource), (void**)&s->rBuf))) {
      set_err(err, errlen, "UAV buffer create failed"); delete s; return NULL; }

    // 共享 fence：D3D11 拷贝完成 -> D3D12 队列 Wait
    ID3D11Device5* d5 = NULL;
    if (FAILED(s->d3d11->QueryInterface(__uuidof(ID3D11Device5), (void**)&d5)) ||
        FAILED(d5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&s->fence11))) {
      if (d5) d5->Release();
      set_err(err, errlen, "ID3D11Fence create failed (needs D3D11.3+)"); delete s; return NULL; }
    d5->Release();
    HANDLE hF = NULL;
    if (FAILED(s->fence11->CreateSharedHandle(NULL, GENERIC_ALL, NULL, &hF)) ||
        FAILED(s->dev12->OpenSharedHandle(hF, __uuidof(ID3D12Fence), (void**)&s->fence12))) {
      if (hF) CloseHandle(hF);
      set_err(err, errlen, "fence NT share failed"); delete s; return NULL; }
    CloseHandle(hF);

    // ---- ORT：自持会话（Python 拿不到 OrtSession*，别无选择）----
    s->hOrt = LoadLibraryExW(ort_dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!s->hOrt) { set_err(err, errlen, "LoadLibrary onnxruntime.dll failed"); delete s; return NULL; }
    const OrtApiBase* (*pGetBase)(void) =
        (const OrtApiBase * (*)(void))(uintptr_t) GetProcAddress(s->hOrt, "OrtGetApiBase");
    if (!pGetBase) { set_err(err, errlen, "OrtGetApiBase not exported"); delete s; return NULL; }
    s->ort = (const OrtApi*)pGetBase()->GetApi(ORT_API_VERSION);
    if (!s->ort) { set_err(err, errlen, "GetApi rejected (ORT version mismatch)"); delete s; return NULL; }
    const void* epapi = NULL;
    if (s->ort->GetExecutionProviderApi("DML", 1, &epapi) || !epapi) {
      set_err(err, errlen, "GetExecutionProviderApi(DML,1) failed"); delete s; return NULL; }
    s->dmlapi = (const OrtDmlApi*)epapi;

    // 绝对路径（随包）用 ALTERED 搜索（依赖同目录）；裸名走标准搜索（system32）
    s->hDmlDll = LoadLibraryExW(dml_dll, NULL,
                                wcschr(dml_dll, L'\\') ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
    if (!s->hDmlDll) { set_err(err, errlen, "LoadLibrary DirectML.dll failed"); delete s; return NULL; }
    PFN_DMLCREATEDEVICE pDMLCreateDevice = (PFN_DMLCREATEDEVICE)GetProcAddress(s->hDmlDll, "DMLCreateDevice");
    if (!pDMLCreateDevice) { set_err(err, errlen, "DMLCreateDevice not exported"); delete s; return NULL; }
    if (FAILED(pDMLCreateDevice(s->dev12, DML_CREATE_DEVICE_FLAG_NONE,
                                __uuidof(IDMLDevice), (void**)&s->dmlDev))) {
      set_err(err, errlen, "DMLCreateDevice failed"); delete s; return NULL; }

    OrtStatus* st;
    if ((st = s->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "d2s-shim", &s->env))) {
      set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); delete s; return NULL; }
    OrtSessionOptions* so = NULL;
    if ((st = s->ort->CreateSessionOptions(&so)) ||
        (st = s->dmlapi->SessionOptionsAppendExecutionProvider_DML1(so, s->dmlDev, s->queue)) ||
        (st = s->ort->CreateSession(s->env, model, so, &s->session))) {
      if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
      if (so) s->ort->ReleaseSessionOptions(so);
      delete s; return NULL;
    }
    s->ort->ReleaseSessionOptions(so);

    if ((st = s->ort->GetAllocatorWithDefaultOptions(&s->oalloc)) ||
        (st = s->ort->SessionGetInputName(s->session, 0, s->oalloc, &s->inName)) ||
        (st = s->ort->SessionGetOutputName(s->session, 0, s->oalloc, &s->outName))) {
      if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
      delete s; return NULL; }
    { OrtTypeInfo* ti = NULL;
      const OrtTensorTypeAndShapeInfo* tsi = NULL;
      size_t nd = 0; int64_t dims[8] = {0};
      size_t cnt = 1;
      ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      if ((st = s->ort->SessionGetInputTypeInfo(s->session, 0, &ti)) ||
          (st = s->ort->CastTypeInfoToTensorInfo(ti, &tsi)) ||
          (st = s->ort->GetDimensionsCount(tsi, &nd)) ||
          (st = s->ort->GetDimensions(tsi, dims, nd)) ||
          (st = s->ort->GetTensorElementType(tsi, &et))) {
        if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
        if (ti) s->ort->ReleaseTypeInfo(ti);
        delete s; return NULL; }
      for (size_t i = 0; i < nd; i++) cnt *= (size_t)dims[i];
      s->ort->ReleaseTypeInfo(ti);
      if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || cnt * 4 != s->bytes) {
        set_err(err, errlen, "model input is not float[3*res*res]"); delete s; return NULL; }
      if ((st = s->ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &s->miDml)) ||
          (st = s->dmlapi->CreateGPUAllocationFromD3DResource(s->rBuf, &s->zcAlloc)) ||
          (st = s->ort->CreateTensorWithDataAsOrtValue(s->miDml, s->zcAlloc, s->bytes, dims, nd,
                                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s->inZC))) {
        if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
        delete s; return NULL; }
    }
    (void)qpc_ms();
    return s;
  } catch (...) {
    set_err(err, errlen, "unexpected exception in shim_create");
    delete s;
    return NULL;
  }
}

__declspec(dllexport) int __cdecl d2s_shim_run(void* h, unsigned int src_glo,
                                               float* out, size_t out_bytes) {
  Shim* s = (Shim*)h;
  if (!s || !out || out_bytes < (size_t)s->res * s->res * 4) return -1;
  double t0 = qpc_ms();
  double tt = t0;

  // GL：app 的 _prep_tex -> 注册的 staging 纹理（当前线程 GL 上下文）
  if (!s->wglDXLockObjects(s->dxh, 1, &s->hObj)) return 2;
  s->t6[0] = qpc_ms() - tt; tt = qpc_ms();
  while (glGetError() != GL_NO_ERROR) {}  // 排空粘滞错误位，避免旧账记到本次拷贝
  s->glCopyImageSubData(src_glo, GL_TEXTURE_2D, 0, 0, 0, 0,
                        s->dstTex, GL_TEXTURE_2D, 0, 0, 0, 0,
                        (GLsizei)s->tw, (GLsizei)s->th, 1);
  if (glGetError() != GL_NO_ERROR) {
    s->wglDXUnlockObjects(s->dxh, 1, &s->hObj);
    return 3;
  }
  // glFlush 而非 glFinish：interop unlock 会在 GPU 侧插入 GL->D3D11 依赖，
  // CPU 不必等整帧 GL 队列排空（省 ~0.5ms；正确性由 python 首帧实况对账兜底）
  glFlush();
  s->t6[1] = qpc_ms() - tt; tt = qpc_ms();
  s->wglDXUnlockObjects(s->dxh, 1, &s->hObj);
  s->t6[2] = qpc_ms() - tt; tt = qpc_ms();

  // D3D11：keyed mutex 协议拷进共享纹理（不 AcquireSync 写 = 静默丢弃，P1 实锤）+ fence 信号
  if (FAILED(s->km->AcquireSync(0, INFINITE))) return 4;
  s->ctx11->CopySubresourceRegion(s->tShared, 0, 0, 0, 0, s->tStaging, 0, NULL);
  if (!s->ctx4 || FAILED(s->ctx4->Signal(s->fence11, s->frame))) return 5;
  if (FAILED(s->km->ReleaseSync(0))) return 6;
  s->ctx11->Flush();
  s->t6[3] = qpc_ms() - tt; tt = qpc_ms();

  // D3D12：等 fence -> 纹理拷进 UAV buffer（与推理同队列，FIFO 保证先于 Run）
  if (FAILED(s->queue->Wait(s->fence12, s->frame))) return 7;
  // allocator 必须先 Reset（只 Reset list 会每次追加 command buffer 内存不回收）
  if (FAILED(s->alloc12->Reset())) return 8;
  if (FAILED(s->cl->Reset(s->alloc12, NULL))) return 8;
  D3D12_RESOURCE_BARRIER bar = {};
  bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  bar.Transition.pResource = s->rTex; bar.Transition.Subresource = 0;
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  s->cl->ResourceBarrier(1, &bar);
  D3D12_TEXTURE_COPY_LOCATION ls = {}, ld = {};
  ls.pResource = s->rTex; ls.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; ls.SubresourceIndex = 0;
  ld.pResource = s->rBuf; ld.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  ld.PlacedFootprint.Offset = 0;
  ld.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
  ld.PlacedFootprint.Footprint.Width = s->tw;
  ld.PlacedFootprint.Footprint.Height = s->th;
  ld.PlacedFootprint.Footprint.Depth = 1;
  ld.PlacedFootprint.Footprint.RowPitch = s->tw * 4;  // W%64==0 => 已 256 对齐且稠密
  s->cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, NULL);
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  s->cl->ResourceBarrier(1, &bar);
  if (FAILED(s->cl->Close())) return 9;
  ID3D12CommandList* exec[] = {s->cl};
  s->queue->ExecuteCommandLists(1, exec);
  s->frame++;
  double t1 = qpc_ms();
  s->t6[4] = t1 - tt;
  s->t_input = t1 - t0;

  // ORT：零拷贝输入推理，输出 CPU
  const char* inN[1] = {s->inName};
  const char* outN[1] = {s->outName};
  OrtValue* ov = NULL;
  OrtStatus* st = s->ort->Run(s->session, NULL, inN, (const OrtValue* const*)&s->inZC, 1, outN, 1, &ov);
  if (st || !ov) {
    if (st) s->ort->ReleaseStatus(st);
    return 10;
  }
  float* d = NULL;
  if (s->ort->GetTensorMutableData(ov, (void**)&d)) {
    s->ort->ReleaseValue(ov);
    return 11;
  }
  memcpy(out, d, (size_t)s->res * s->res * 4);
  s->ort->ReleaseValue(ov);
  s->t_run = qpc_ms() - t1;
  s->t6[5] = s->t_run;
  return 0;
}

__declspec(dllexport) void __cdecl d2s_shim_times6(void* h, double* out6) {
  Shim* s = (Shim*)h;
  if (s && out6) memcpy(out6, s->t6, sizeof s->t6);
}

__declspec(dllexport) void __cdecl d2s_shim_times(void* h, double* t_input_ms, double* t_run_ms) {
  Shim* s = (Shim*)h;
  if (t_input_ms) *t_input_ms = s ? s->t_input : 0;
  if (t_run_ms) *t_run_ms = s ? s->t_run : 0;
}

__declspec(dllexport) void __cdecl d2s_shim_destroy(void* h) {
  delete (Shim*)h;
}

// ============================================================================
// shim v2（迁移 T2）：D3D12 直入——输入 = d3drender 管线的 prep 纹理指针
// （同 device/queue 注入，无 GL、无 D3D11 KM 段、无 fence——同队列 FIFO
// 保证 prep 拷贝先于推理）。v1 原样保留（GL 生产路径在用）。
//
// ABI：
//   void* d2s_shim2_create(wchar_t* ort_dll, wchar_t* dml_dll, wchar_t* model,
//                          uint32 res, void* dev12, void* queue12,
//                          void* prep_res, uint32 prep_w, uint32 prep_h,
//                          wchar_t* err, size_t errlen)
//   int   d2s_shim2_run(void* h, float* out, size_t out_bytes)   // 0=ok
//   void  d2s_shim2_times6(void* h, double* out6)                // [copy, run]
//   void  d2s_shim2_destroy(void* h)
//
// prep_tex 状态约定：调用方保证 run 时该资源处于 COMMON（d3drender 的
// _pass 语义：RENDER_TARGET 用完记回 COMMON；ECL 边界自动 decay）——
// run 内 COMMON→COPY_SOURCE→COMMON 走隐式状态提升（与 v1 rTex 同款）。
// ============================================================================

struct Shim2 {
  // res = 输入宽 W，rh = 输入高 H（方形模型 rh==res；16:9 宽屏模型 W≠H）
  uint32_t res, rh, tw, th;
  uint64_t bytes;
  // 注入（引用计数不增——d3drender 管线持有）
  ID3D12Device* dev12 = NULL;
  ID3D12CommandQueue* queue = NULL;
  ID3D12Resource* prep = NULL;
  ID3D12CommandAllocator* alloc12 = NULL;
  ID3D12GraphicsCommandList* cl = NULL;
  ID3D12Resource* rBuf = NULL;
  UINT64 frame = 1;
  // ORT / DML
  HMODULE hOrt = NULL, hDmlDll = NULL;
  const OrtApi* ort = NULL;
  const OrtDmlApi* dmlapi = NULL;
  OrtEnv* env = NULL;
  OrtSession* session = NULL;
  char* inName = NULL, * outName = NULL;
  OrtAllocator* oalloc = NULL;
  OrtMemoryInfo* miDml = NULL;
  void* zcAlloc = NULL;
  OrtValue* inZC = NULL;
  IDMLDevice* dmlDev = NULL;
  // GPU 输出（让推理结果留在显存，深度链直接采样，不走 CPU 往返）：
  // 输出绑定到 outBuf，Run 只 Flush 不 Sync → CPU 不等；随后同一队列上
  // 录 outBuf→outTex 拷贝，链的提交天然排在它后面（同队列 FIFO）。
  bool gpuOut = false;
  char gpuErr[256] = {0};   // GPU 输出初始化失败原因（ASCII，供 Python 读）
  // 独立 DML 队列：推理不能挤在渲染队列上（同队列 FIFO 会把推理/链/显示串成
  // 一条线，实测 10.5fps）。两条队列用 fence 握手，推理(N) 与链/显示(N-1) 并行。
  ID3D12CommandQueue* dmlQueue = NULL;
  ID3D12Fence* fIn = NULL;      // 渲染队列 signal：prep 拷贝完成
  ID3D12Fence* fOut = NULL;     // DML 队列 signal：推理完成
  UINT64 vin = 0, vout = 0;
  ID3D12Resource* outBuf = NULL;
  ID3D12Resource* outTex = NULL;
  ID3D12CommandAllocator* alloc12b = NULL;
  ID3D12GraphicsCommandList* cl2 = NULL;
  ID3D12Fence* fence = NULL;
  HANDLE ev = NULL;
  UINT64 fval = 0;
  void* outAlloc = NULL;
  OrtValue* outZC = NULL;
  OrtIoBinding* bind = NULL;
  // timing（毫秒，最近一帧）：copy+exec / run
  double t6[2] = {0};

  ~Shim2() { destroy(); }

  void destroy() {
    if (ort) {
      if (bind) ort->ReleaseIoBinding(bind);
      if (outZC) ort->ReleaseValue(outZC);
      if (outAlloc && dmlapi) dmlapi->FreeGPUAllocation(outAlloc);
      if (inZC) ort->ReleaseValue(inZC);
      if (zcAlloc && dmlapi) dmlapi->FreeGPUAllocation(zcAlloc);
      if (miDml) ort->ReleaseMemoryInfo(miDml);
      if (inName && oalloc) oalloc->Free(oalloc, inName);
      if (outName && oalloc) oalloc->Free(oalloc, outName);
      if (session) ort->ReleaseSession(session);
      if (env) ort->ReleaseEnv(env);
    }
    if (fence) fence->Release();
    if (fOut) fOut->Release();
    if (fIn) fIn->Release();
    if (dmlQueue) dmlQueue->Release();
    if (ev) CloseHandle(ev);
    if (cl2) cl2->Release();
    if (alloc12b) alloc12b->Release();
    if (outTex) outTex->Release();
    if (outBuf) outBuf->Release();
    if (dmlDev) dmlDev->Release();
    if (rBuf) rBuf->Release();
    if (cl) cl->Release();
    if (alloc12) alloc12->Release();
    // 注入的 dev12/queue/prep 不 Release（管线持有）
    if (hDmlDll) FreeLibrary(hDmlDll);
    if (hOrt) FreeLibrary(hOrt);
    memset(this, 0, sizeof *this);
  }
};

static HRESULT shim2_ort_init(Shim2* s, const wchar_t* ort_dll,
                              const wchar_t* dml_dll, const wchar_t* model,
                              wchar_t* err, size_t errlen) {
  s->hOrt = LoadLibraryExW(ort_dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!s->hOrt) { set_err(err, errlen, "LoadLibrary onnxruntime.dll failed"); return E_FAIL; }
  const OrtApiBase* (*pGetBase)(void) =
      (const OrtApiBase * (*)(void))(uintptr_t)GetProcAddress(s->hOrt, "OrtGetApiBase");
  if (!pGetBase) { set_err(err, errlen, "OrtGetApiBase not exported"); return E_FAIL; }
  s->ort = (const OrtApi*)pGetBase()->GetApi(ORT_API_VERSION);
  if (!s->ort) { set_err(err, errlen, "GetApi rejected (ORT version mismatch)"); return E_FAIL; }
  const void* epapi = NULL;
  if (s->ort->GetExecutionProviderApi("DML", 1, &epapi) || !epapi) {
    set_err(err, errlen, "GetExecutionProviderApi(DML,1) failed"); return E_FAIL; }
  s->dmlapi = (const OrtDmlApi*)epapi;
  s->hDmlDll = LoadLibraryExW(dml_dll, NULL,
                              wcschr(dml_dll, L'\\') ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
  if (!s->hDmlDll) { set_err(err, errlen, "LoadLibrary DirectML.dll failed"); return E_FAIL; }
  PFN_DMLCREATEDEVICE pDMLCreateDevice =
      (PFN_DMLCREATEDEVICE)GetProcAddress(s->hDmlDll, "DMLCreateDevice");
  if (!pDMLCreateDevice) { set_err(err, errlen, "DMLCreateDevice not exported"); return E_FAIL; }
  if (FAILED(pDMLCreateDevice(s->dev12, DML_CREATE_DEVICE_FLAG_NONE,
                              __uuidof(IDMLDevice), (void**)&s->dmlDev))) {
    set_err(err, errlen, "DMLCreateDevice failed"); return E_FAIL; }
  // 独立 DML 队列 + 两个握手 fence（推理与渲染并行，顺序由 GPU 端保证）
  {
    D3D12_COMMAND_QUEUE_DESC dq = {};
    dq.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    dq.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    if (FAILED(s->dev12->CreateCommandQueue(&dq, __uuidof(ID3D12CommandQueue),
                                            (void**)&s->dmlQueue)) ||
        FAILED(s->dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                     __uuidof(ID3D12Fence), (void**)&s->fIn)) ||
        FAILED(s->dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                     __uuidof(ID3D12Fence), (void**)&s->fOut))) {
      set_err(err, errlen, "create dml queue/fences failed"); return E_FAIL;
    }
  }
  OrtStatus* st;
  if ((st = s->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "d2s-shim2", &s->env))) {
    set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); return E_FAIL; }
  OrtSessionOptions* so = NULL;
  if ((st = s->ort->CreateSessionOptions(&so)) ||
      (st = s->dmlapi->SessionOptionsAppendExecutionProvider_DML1(so, s->dmlDev, s->queue)) ||
      (st = s->ort->CreateSession(s->env, model, so, &s->session))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
    if (so) s->ort->ReleaseSessionOptions(so);
    return E_FAIL;
  }
  s->ort->ReleaseSessionOptions(so);
  if ((st = s->ort->GetAllocatorWithDefaultOptions(&s->oalloc)) ||
      (st = s->ort->SessionGetInputName(s->session, 0, s->oalloc, &s->inName)) ||
      (st = s->ort->SessionGetOutputName(s->session, 0, s->oalloc, &s->outName))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
    return E_FAIL;
  }
  OrtTypeInfo* ti = NULL;
  const OrtTensorTypeAndShapeInfo* tsi = NULL;
  size_t nd = 0; int64_t dims[8] = {0};
  size_t cnt = 1;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if ((st = s->ort->SessionGetInputTypeInfo(s->session, 0, &ti)) ||
      (st = s->ort->CastTypeInfoToTensorInfo(ti, &tsi)) ||
      (st = s->ort->GetDimensionsCount(tsi, &nd)) ||
      (st = s->ort->GetDimensions(tsi, dims, nd)) ||
      (st = s->ort->GetTensorElementType(tsi, &et))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
    if (ti) s->ort->ReleaseTypeInfo(ti);
    return E_FAIL;
  }
  for (size_t i = 0; i < nd; i++) cnt *= (size_t)dims[i];
  s->ort->ReleaseTypeInfo(ti);
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || cnt * 4 != s->bytes) {
    set_err(err, errlen, "model input is not float[3*res*res]"); return E_FAIL; }
  if ((st = s->ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &s->miDml)) ||
      (st = s->dmlapi->CreateGPUAllocationFromD3DResource(s->rBuf, &s->zcAlloc)) ||
      (st = s->ort->CreateTensorWithDataAsOrtValue(s->miDml, s->zcAlloc, s->bytes, dims, nd,
                                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s->inZC))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st)); s->ort->ReleaseStatus(st); }
    return E_FAIL;
  }
  return S_OK;
}

// GPU 输出初始化：把模型输出绑到一块 D3D12 buffer（DML 显存），再用一条
// 同队列的 buffer→texture 拷贝把结果变成可采样的 R32F 纹理。
// 关键收益：Run 之后 CPU 不等待（ORT 对 GPU 输出只 Flush），深度链能立刻
// 录制，且同队列 FIFO 保证"先推理、后链"——不需要额外 fence 或流水级。
static HRESULT shim2_gpu_out_init(Shim2* s, wchar_t* err, size_t errlen) {
  D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC bd = {};
  bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bd.Width = (UINT64)s->res * s->rh * 4;
  bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
  bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (FAILED(s->dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
        D3D12_RESOURCE_STATE_COMMON, NULL, __uuidof(ID3D12Resource),
        (void**)&s->outBuf))) {
    set_err(err, errlen, "outBuf create failed"); return E_FAIL; }
  D3D12_RESOURCE_DESC td = {};
  td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  td.Width = s->res; td.Height = s->rh; td.DepthOrArraySize = 1;
  td.MipLevels = 1; td.Format = DXGI_FORMAT_R32_FLOAT;
  td.SampleDesc.Count = 1; td.Flags = D3D12_RESOURCE_FLAG_NONE;
  if (FAILED(s->dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
        D3D12_RESOURCE_STATE_COMMON, NULL, __uuidof(ID3D12Resource),
        (void**)&s->outTex))) {
    set_err(err, errlen, "outTex create failed"); return E_FAIL; }
  if (FAILED(s->dev12->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
        (void**)&s->alloc12b)) ||
      FAILED(s->dev12->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, s->alloc12b, NULL,
        __uuidof(ID3D12GraphicsCommandList), (void**)&s->cl2)) ||
      FAILED(s->dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        __uuidof(ID3D12Fence), (void**)&s->fence))) {
    set_err(err, errlen, "gpu-out alloc/list/fence create failed");
    return E_FAIL; }
  s->cl2->Close();
  s->ev = CreateEventW(NULL, FALSE, FALSE, NULL);

  OrtStatus* st = NULL;
  OrtTypeInfo* ti = NULL;
  const OrtTensorTypeAndShapeInfo* tsi = NULL;
  size_t nd = 0; int64_t dims[8] = {0};
  if ((st = s->ort->SessionGetOutputTypeInfo(s->session, 0, &ti)) ||
      (st = s->ort->CastTypeInfoToTensorInfo(ti, &tsi)) ||
      (st = s->ort->GetDimensionsCount(tsi, &nd)) ||
      (st = s->ort->GetDimensions(tsi, dims, nd))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st));
              s->ort->ReleaseStatus(st); }
    if (ti) s->ort->ReleaseTypeInfo(ti);
    return E_FAIL;
  }
  s->ort->ReleaseTypeInfo(ti);
  if ((st = s->dmlapi->CreateGPUAllocationFromD3DResource(s->outBuf,
                                                          &s->outAlloc)) ||
      (st = s->ort->CreateTensorWithDataAsOrtValue(
        s->miDml, s->outAlloc, (size_t)s->res * s->rh * 4, dims, nd,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s->outZC)) ||
      (st = s->ort->CreateIoBinding(s->session, &s->bind)) ||
      (st = s->ort->BindInput(s->bind, s->inName, s->inZC)) ||
      (st = s->ort->BindOutput(s->bind, s->outName, s->outZC))) {
    if (st) { set_err(err, errlen, s->ort->GetErrorMessage(st));
              s->ort->ReleaseStatus(st); }
    return E_FAIL;
  }
  s->gpuOut = true;
  return S_OK;
}

// 非方形创建（16:9 宽屏模型，如 ZipDepth wide 672×384）：res_w×res_h = 模型输入
// 空间尺寸；prep flat 纹理须满足 prep_w*prep_h == 3*res_w*res_h 且行距 256 对齐。
__declspec(dllexport) void* __cdecl d2s_shim2_create_wh(
    const wchar_t* ort_dll, const wchar_t* dml_dll, const wchar_t* model,
    unsigned int res_w, unsigned int res_h,
    void* dev12_ptr, void* queue_ptr, void* prep_res_ptr,
    unsigned int prep_w, unsigned int prep_h, wchar_t* err, size_t errlen) {
  Shim2* s = NULL;
  try {
    if (!ort_dll || !dml_dll || !model || !res_w || !res_h || !dev12_ptr ||
        !queue_ptr || !prep_res_ptr) { set_err(err, errlen, "bad args"); return NULL; }
    if ((uint64_t)prep_w * prep_h != (uint64_t)res_w * res_h * 3 || prep_w % 64u) {
      set_err(err, errlen, "prep geometry must be W*H==3*res_w*res_h with W%64==0 (flat CHW, 256-aligned rows)");
      return NULL;
    }
    s = new Shim2();
    s->res = res_w; s->rh = res_h; s->tw = prep_w; s->th = prep_h;
    s->bytes = (uint64_t)res_w * res_h * 3 * 4;
    s->dev12 = (ID3D12Device*)dev12_ptr;
    s->queue = (ID3D12CommandQueue*)queue_ptr;
    s->prep = (ID3D12Resource*)prep_res_ptr;
    s->dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     __uuidof(ID3D12CommandAllocator), (void**)&s->alloc12);
    s->dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s->alloc12, NULL,
                                __uuidof(ID3D12GraphicsCommandList), (void**)&s->cl);
    s->cl->Close();
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = s->bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(s->dev12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_COMMON, NULL,
                                                 __uuidof(ID3D12Resource), (void**)&s->rBuf))) {
      set_err(err, errlen, "UAV buffer create failed"); delete s; return NULL; }
    if (FAILED(shim2_ort_init(s, ort_dll, dml_dll, model, err, errlen))) {
      delete s; return NULL; }
    // GPU 输出默认【关】（D2S_DML_GPU_OUT=1 才开）：out-copy 命令列表的
    // Close 还返回 E_INVALIDARG（0x80070057），待定位；关掉时走原 CPU 输出
    // 路径（工作线程），行为与之前一致。
    {
      wchar_t envbuf[8] = {0};
      DWORD en = GetEnvironmentVariableW(L"D2S_DML_GPU_OUT", envbuf, 8);
      bool want = (en > 0 && envbuf[0] == L'1');
      if (want && FAILED(shim2_gpu_out_init(s, err, errlen))) {
        s->gpuOut = false;
        size_t i = 0;
        if (err) {
          for (; err[i] && i < 255; ++i) s->gpuErr[i] = (char)err[i];
        }
        s->gpuErr[i] = 0;
      }
      fprintf(stdout, "[shim2] gpu_out want=%d(env_len=%lu) enabled=%d\n",
              (int)want, (unsigned long)en, (int)s->gpuOut);
      fflush(stdout);
    }
    return s;
  } catch (...) {
    set_err(err, errlen, "unexpected exception in shim2_create");
    delete s;
    return NULL;
  }
}

// 方形创建（旧 ABI 保持不变）：等价于 res_h = res_w。
__declspec(dllexport) void* __cdecl d2s_shim2_create(
    const wchar_t* ort_dll, const wchar_t* dml_dll, const wchar_t* model,
    unsigned int res, void* dev12_ptr, void* queue_ptr, void* prep_res_ptr,
    unsigned int prep_w, unsigned int prep_h, wchar_t* err, size_t errlen) {
  return d2s_shim2_create_wh(ort_dll, dml_dll, model, res, res, dev12_ptr,
                             queue_ptr, prep_res_ptr, prep_w, prep_h, err,
                             errlen);
}

__declspec(dllexport) int __cdecl d2s_shim2_run(void* h, float* out,
                                                size_t out_bytes) {
  Shim2* s = (Shim2*)h;
  if (!s || !out || out_bytes < (size_t)s->res * s->rh * 4) return -1;
  double t0 = qpc_ms();
  // prep(COMMON) -> COPY_SOURCE 提升回收 -> UAV buffer（与推理同队列 FIFO）
  // allocator 必须先 Reset（只 Reset list 会每次追加 command buffer 内存不回收）
  if (FAILED(s->alloc12->Reset())) return 8;
  if (FAILED(s->cl->Reset(s->alloc12, NULL))) return 8;
  D3D12_RESOURCE_BARRIER bar = {};
  bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  bar.Transition.pResource = s->prep; bar.Transition.Subresource = 0;
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  s->cl->ResourceBarrier(1, &bar);
  D3D12_TEXTURE_COPY_LOCATION ls = {}, ld = {};
  ls.pResource = s->prep; ls.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  ls.SubresourceIndex = 0;
  ld.pResource = s->rBuf; ld.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  ld.PlacedFootprint.Offset = 0;
  ld.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
  ld.PlacedFootprint.Footprint.Width = s->tw;
  ld.PlacedFootprint.Footprint.Height = s->th;
  ld.PlacedFootprint.Footprint.Depth = 1;
  ld.PlacedFootprint.Footprint.RowPitch = s->tw * 4;
  s->cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, NULL);
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  s->cl->ResourceBarrier(1, &bar);
  if (FAILED(s->cl->Close())) return 9;
  ID3D12CommandList* exec[] = {s->cl};
  s->queue->ExecuteCommandLists(1, exec);
  s->frame++;
  double t1 = qpc_ms();
  s->t6[0] = t1 - t0;

  const char* inN[1] = {s->inName};
  const char* outN[1] = {s->outName};
  OrtValue* ov = NULL;
  OrtStatus* st = s->ort->Run(s->session, NULL, inN, (const OrtValue* const*)&s->inZC,
                              1, outN, 1, &ov);
  if (st || !ov) {
    if (st) s->ort->ReleaseStatus(st);
    return 10;
  }
  float* d = NULL;
  if (s->ort->GetTensorMutableData(ov, (void**)&d)) {
    s->ort->ReleaseValue(ov);
    return 11;
  }
  memcpy(out, d, (size_t)s->res * s->rh * 4);
  s->ort->ReleaseValue(ov);
  s->t6[1] = qpc_ms() - t1;
  return 0;
}

// 只入队、不等待的推理：输出留在显存（gpu_out_init 的 outBuf），随后在
// 同一队列上把 outBuf 拷进 outTex。返回时 CPU 不阻塞；链的提交排在它后面，
// 由同队列 FIFO 保证"先推理后链"，无需 fence/流水级。
__declspec(dllexport) int __cdecl d2s_shim2_run_gpu(void* h) {
  Shim2* s = (Shim2*)h;
  if (!s || !s->gpuOut) return -1;
  // 上一轮的输出拷贝还没跑完 → 忙（调用方跳过这一拍，绝不阻塞）
  if (s->fence->GetCompletedValue() < s->fval) return 3;   // 上次推理未完成
  double t0 = qpc_ms();
  if (FAILED(s->alloc12->Reset())) return 8;
  if (FAILED(s->cl->Reset(s->alloc12, NULL))) return 8;
  D3D12_RESOURCE_BARRIER bar = {};
  bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  bar.Transition.pResource = s->prep;
  bar.Transition.Subresource = 0;
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  s->cl->ResourceBarrier(1, &bar);
  D3D12_TEXTURE_COPY_LOCATION ls = {}, ld = {};
  ls.pResource = s->prep; ls.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  ls.SubresourceIndex = 0;
  ld.pResource = s->rBuf; ld.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  ld.PlacedFootprint.Offset = 0;
  ld.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_FLOAT;
  ld.PlacedFootprint.Footprint.Width = s->tw;
  ld.PlacedFootprint.Footprint.Height = s->th;
  ld.PlacedFootprint.Footprint.Depth = 1;
  ld.PlacedFootprint.Footprint.RowPitch = s->tw * 4;
  s->cl->CopyTextureRegion(&ld, 0, 0, 0, &ls, NULL);
  bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  s->cl->ResourceBarrier(1, &bar);
  {
    HRESULT hr = s->cl->Close();
    if (FAILED(hr)) {
      s->t6[0] = (double)(unsigned)hr;   // 失败时带回 hr（调试）
      return 9;
    }
  }
  ID3D12CommandList* exec[] = {s->cl};
  s->queue->ExecuteCommandLists(1, exec);
  double t1 = qpc_ms();
  s->t6[0] = t1 - t0;

  // 跨队列握手已撤：ORTSession 绑的就是渲染队列，同队列 FIFO 天然有序，
  // 额外 fence 反而把 prep(N+1) 挡在 inference(N) 后面（实测 35→11fps）。
  OrtStatus* st = s->ort->RunWithBinding(s->session, NULL, s->bind);
  if (st) {
    s->ort->ReleaseStatus(st);
    return 10;
  }
  // 结果在 outBuf（显存、连续布局）。纹理化交给渲染侧的一个全屏 PS
  // （StructuredBuffer → R32F 纹理）——不用 CopyTextureRegion，因为
  // res*4=896 的 RowPitch 不满足 placed footprint 的 256 字节对齐。
  s->fval++;
  s->queue->Signal(s->fence, s->fval);
  s->t6[1] = qpc_ms() - t1;
  return 0;
}

// GPU 输出纹理（ID3D12Resource*，引用计数不增；未启用时返回 NULL）
__declspec(dllexport) void* __cdecl d2s_shim2_out_buf(void* h) {
  Shim2* s = (Shim2*)h;
  return (s && s->gpuOut) ? (void*)s->outBuf : NULL;
}

__declspec(dllexport) const char* __cdecl d2s_shim2_gpu_error(void* h) {
  Shim2* s = (Shim2*)h;
  return s ? s->gpuErr : "";
}

__declspec(dllexport) void __cdecl d2s_shim2_times6(void* h, double* out6) {
  Shim2* s = (Shim2*)h;
  if (s && out6) memcpy(out6, s->t6, sizeof s->t6);
}

__declspec(dllexport) void __cdecl d2s_shim2_destroy(void* h) {
  delete (Shim2*)h;
}

}  // extern "C"
