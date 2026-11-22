// P0 feasibility probe for the zero-copy DML migration (docs/zero-copy-migration.md).
//
// Loads the app's exact onnxruntime.dll via LoadLibraryEx (no import lib needed),
// resolves OrtGetApiBase -> GetApi(24) -> GetExecutionProviderApi("DML"),
// dumps the OrtDmlApi function table, then creates a DML session on the app's
// depth model and runs timed CPU-input inferences.
//
// Headers are the official ones from tag v1.24.4 (commit 2d92497), the same
// commit the DLL itself reports in its FileVersion, so struct layouts match.
//
// usage: p0_dml_probe.exe [onnxruntime.dll] [model.onnx] [device_id]
//   defaults: <repo>\_dml\onnxruntime\capi\onnxruntime.dll
//             <repo>\runtime\depth-224.onnx, device 0
// exit 0 = pass (API table complete + inferences ran, outputs finite)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"

static const OrtApi* g_ort = NULL;

#define CHK(stmt) do { OrtStatus* _s = (stmt); if (_s) { \
    fprintf(stderr, "ORT error %s:%d code=%d: %s\n", __FILE__, __LINE__, \
            (int)g_ort->GetErrorCode(_s), g_ort->GetErrorMessage(_s)); \
    g_ort->ReleaseStatus(_s); return 2; } } while (0)

static wchar_t* to_w(const char* s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  wchar_t* w = (wchar_t*)malloc((size_t)n * sizeof(wchar_t));
  MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
  return w;
}

static double now_ms(void) {
  static LARGE_INTEGER f = {0};
  LARGE_INTEGER c;
  if (!f.QuadPart) QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return c.QuadPart * 1000.0 / f.QuadPart;
}

static void dump_dml_api(const OrtDmlApi* d) {
  void* m[8];
  m[0] = (void*)(uintptr_t)d->SessionOptionsAppendExecutionProvider_DML;
  m[1] = (void*)(uintptr_t)d->SessionOptionsAppendExecutionProvider_DML1;
  m[2] = (void*)(uintptr_t)d->CreateGPUAllocationFromD3DResource;
  m[3] = (void*)(uintptr_t)d->FreeGPUAllocation;
  m[4] = (void*)(uintptr_t)d->GetD3D12ResourceFromAllocation;
  m[5] = (void*)(uintptr_t)d->SessionOptionsAppendExecutionProvider_DML2;
  m[6] = (void*)(uintptr_t)d->GetDMLDevice;
  m[7] = (void*)(uintptr_t)d->GetDMLCommandQueue;
  const char* n[8] = {
    "SessionOptionsAppendExecutionProvider_DML",
    "SessionOptionsAppendExecutionProvider_DML1",
    "CreateGPUAllocationFromD3DResource",
    "FreeGPUAllocation",
    "GetD3D12ResourceFromAllocation",
    "SessionOptionsAppendExecutionProvider_DML2",
    "GetDMLDevice",
    "GetDMLCommandQueue",
  };
  printf("[A] OrtDmlApi function table (%d members):\n", 8);
  for (int i = 0; i < 8; i++) printf("    %-46s %p %s\n", n[i], m[i], m[i] ? "" : "<== NULL!");
}

int main(int argc, char** argv) {
  const char* dll_a  = argc > 1 ? argv[1] : "E:\\Projects\\Desktop2Stereo\\_dml\\onnxruntime\\capi\\onnxruntime.dll";
  const char* mdl_a  = argc > 2 ? argv[2] : "E:\\Projects\\Desktop2Stereo\\runtime\\depth-224.onnx";
  int device_id      = argc > 3 ? atoi(argv[3]) : 0;
  wchar_t* dll_w = to_w(dll_a);
  wchar_t* mdl_w = to_w(mdl_a);
  printf("p0_dml_probe  dll=%ls\n              model=%ls  device_id=%d\n", dll_w, mdl_w, device_id);

  // ---- A: API surface -----------------------------------------------------
  HMODULE h = LoadLibraryExW(dll_w, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!h) { fprintf(stderr, "LoadLibraryExW failed err=%lu\n", GetLastError()); return 1; }
  const OrtApiBase* (*pGetBase)(void) =
      (const OrtApiBase * (*)(void))(uintptr_t) GetProcAddress(h, "OrtGetApiBase");
  if (!pGetBase) { fprintf(stderr, "no OrtGetApiBase export\n"); return 1; }
  const OrtApiBase* base = pGetBase();
  printf("[A] version_string: %s\n", base->GetVersionString());
  g_ort = (const OrtApi*)base->GetApi(ORT_API_VERSION);
  if (!g_ort) { fprintf(stderr, "GetApi(%d) returned NULL\n", ORT_API_VERSION); return 1; }
  printf("[A] GetApi(%d) OK (header ORT_API_VERSION=%d)\n", ORT_API_VERSION, ORT_API_VERSION);

  const OrtDmlApi* dml = NULL;
  for (uint32_t v = 1; v <= 3 && !dml; v++) {
    void* p = NULL;
    OrtStatus* st = g_ort->GetExecutionProviderApi("DML", v, &p);
    if (st) {
      printf("[A] GetExecutionProviderApi(\"DML\",%u): %s\n", v, g_ort->GetErrorMessage(st));
      g_ort->ReleaseStatus(st);
    } else if (p) {
      dml = (const OrtDmlApi*)p;
      printf("[A] GetExecutionProviderApi(\"DML\",%u) OK -> %p\n", v, p);
    }
  }
  if (!dml) { fprintf(stderr, "FATAL: cannot obtain OrtDmlApi\n"); return 1; }
  dump_dml_api(dml);
  if (!dml->CreateGPUAllocationFromD3DResource || !dml->FreeGPUAllocation ||
      !dml->SessionOptionsAppendExecutionProvider_DML1) {
    fprintf(stderr, "FATAL: zero-copy entry points missing from OrtDmlApi\n"); return 1;
  }

  // ---- B: DML session + timed inference -----------------------------------
  OrtEnv* env = NULL;
  CHK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "p0probe", &env));
  OrtSessionOptions* so = NULL;
  CHK(g_ort->CreateSessionOptions(&so));
  CHK(dml->SessionOptionsAppendExecutionProvider_DML(so, device_id));
  OrtSession* session = NULL;
  CHK(g_ort->CreateSession(env, mdl_w, so, &session));

  size_t nin = 0, nout = 0;
  CHK(g_ort->SessionGetInputCount(session, &nin));
  CHK(g_ort->SessionGetOutputCount(session, &nout));
  OrtAllocator* alloc = NULL;
  CHK(g_ort->GetAllocatorWithDefaultOptions(&alloc));
  char *in_name = NULL, *out_name = NULL;
  CHK(g_ort->SessionGetInputName(session, 0, alloc, &in_name));
  CHK(g_ort->SessionGetOutputName(session, 0, alloc, &out_name));
  printf("[B] session OK: inputs=%zu outputs=%zu  in[0]=\"%s\" out[0]=\"%s\"\n",
         nin, nout, in_name, out_name);

  OrtTypeInfo* ti = NULL;
  CHK(g_ort->SessionGetInputTypeInfo(session, 0, &ti));
  const OrtTensorTypeAndShapeInfo* tsi = NULL;
  CHK(g_ort->CastTypeInfoToTensorInfo(ti, &tsi));
  size_t ndim = 0;
  CHK(g_ort->GetDimensionsCount(tsi, &ndim));
  int64_t dims[8] = {0};
  CHK(g_ort->GetDimensions(tsi, dims, ndim));
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  CHK(g_ort->GetTensorElementType(tsi, &et));
  size_t count = 1;
  printf("[B] input type=%d dims=[", (int)et);
  for (size_t i = 0; i < ndim; i++) {
    if (dims[i] <= 0) dims[i] = (i == 0) ? 1 : 224;  // dynamic axes -> NCHW 224
    printf("%lld%s", (long long)dims[i], i + 1 < ndim ? "," : "");
    count *= (size_t)dims[i];
  }
  printf("] elems=%zu\n", count);
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { fprintf(stderr, "FATAL: input not float\n"); return 2; }
  if (count == 0 || count > (16u << 20)) { fprintf(stderr, "FATAL: bad element count\n"); return 2; }

  float* data = (float*)malloc(count * sizeof(float));
  for (size_t i = 0; i < count; i++) data[i] = ((float)(i % 251)) / 250.0f * 2.0f - 1.0f;
  OrtMemoryInfo* mi = NULL;
  CHK(g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi));
  OrtValue* in = NULL;
  CHK(g_ort->CreateTensorWithDataAsOrtValue(mi, data, count * sizeof(float), dims, ndim,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
  const char* in_names[1] = {in_name};
  const char* out_names[1] = {out_name};
  OrtValue* out = NULL;

  for (int w = 0; w < 3; w++) {  // warmup
    CHK(g_ort->Run(session, NULL, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
  }
  double t0 = now_ms(), t_best = 1e9;
  for (int r = 0; r < 10; r++) {
    double a = now_ms();
    CHK(g_ort->Run(session, NULL, in_names, (const OrtValue* const*)&in, 1, out_names, 1, &out));
    double b = now_ms();
    if (b - a < t_best) t_best = b - a;
  }
  double total = now_ms() - t0;

  OrtTensorTypeAndShapeInfo* osi = NULL;
  CHK(g_ort->GetTensorTypeAndShape(out, &osi));
  size_t ondim = 0;
  CHK(g_ort->GetDimensionsCount(osi, &ondim));
  int64_t odims[8] = {0};
  CHK(g_ort->GetDimensions(osi, odims, ondim));
  float* od = NULL;
  CHK(g_ort->GetTensorMutableData(out, (void**)&od));
  size_t ocount = 1;
  printf("[B] output dims=[");
  for (size_t i = 0; i < ondim; i++) { ocount *= (size_t)odims[i]; printf("%lld%s", (long long)odims[i], i + 1 < ondim ? "," : ""); }
  printf("] elems=%zu  run: best=%.2fms avg=%.2fms\n", ocount, t_best, total / 10.0);
  double mn = 1e30, mx = -1e30; int nonfinite = 0;
  for (size_t i = 0; i < ocount; i++) {
    if (!isfinite(od[i])) { nonfinite++; continue; }
    if (od[i] < mn) mn = od[i];
    if (od[i] > mx) mx = od[i];
  }
  printf("[B] out[0..3]=%.4f %.4f %.4f %.4f  min=%.4f max=%.4f nonfinite=%d\n",
         od[0], od[1], od[2], od[3], mn, mx, nonfinite);
  printf(nonfinite ? "P0 RESULT: FAIL (nonfinite outputs)\n" : "P0 RESULT: PASS\n");
  return nonfinite ? 3 : 0;
}
