# -*- coding: utf-8 -*-
"""零拷贝 DML shim 的 ctypes 封装（P3，链路经 P0/P1/P2 逐位验证）。

native DLL（d2s_dml_shim.dll）持有 D3D11/D3D12 设备、共享纹理、ORT 会话，
每帧从渲染线程（GL 上下文常驻）调 run()：
    _prep_tex(GL) -> 注册纹理 -> NTH|KM 共享 -> UAV buffer -> DML 推理
约定：create/run 必须在持有 GL 上下文的线程；任何失败由上层
DmlDepthRunner 捕获并永久回退 glReadPixels 路径。
"""
import ctypes
import os
import time

_TRUE = ("1", "true", "yes", "on", "enable", "enabled")


def zc_enabled():
    """D2S_ZEROCOPY 是否开启。

    必须容忍 "0"/"false"/"off"：之前用 `os.environ.get(...)` 的真值判断，
    `D2S_ZEROCOPY=0` 反而会打开零拷贝 —— 显式关闭变成了显式开启。
    """
    # 默认开启：零拷贝是净收益（read 1.2→~1.0ms，阶段 6.3~8.8→5.5~6.7ms），
    # DLL 缺失/任何失败都会自动回退回读路径（DmlDepthRunner 内部处理）。
    # 显式关闭：D2S_ZEROCOPY=0/false/off。
    v = os.environ.get("D2S_ZEROCOPY", "1").strip().lower()
    return v not in ("", "0", "false", "off", "no")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # D2S/gpu -> D2S -> 仓库根

# DLL 搜索顺序：runtime（部署目录）→ 仓库根（开发就地跑）→ _toolchain/src（构建产物）
_DLL_CANDIDATES = (
    os.path.join(_REPO, "runtime", "d2s_dml_shim.dll"),
    os.path.join(_REPO, "d2s_dml_shim.dll"),
    os.path.join(_REPO, "_toolchain", "src", "d2s_dml_shim.dll"),
)


def find_dll():
    for p in _DLL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def find_ort_dir():
    """ORT+DML 运行目录（必须同时有 onnxruntime.dll + DirectML.dll）：

    顺序：runtime/（可选部署位）→ _toolchain/dl/**（开发树，ort-dml）→
    site-packages/onnxruntime/capi（发布包形态）。
    注意：dev 树 site-packages 里常是 CPU-only ORT（无 DirectML.dll），
    若被优先选中则 shim2 的 DML EP 取不到（GetExecutionProviderApi failed）。"""
    ok = lambda p: (os.path.isfile(os.path.join(p, "onnxruntime.dll"))
                    and os.path.isfile(os.path.join(p, "DirectML.dll")))
    cands = [os.path.join(_REPO, "runtime")]
    base = os.path.join(_REPO, "_toolchain", "dl")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if "onnxruntime.dll" in files and "DirectML.dll" in files:
                cands.append(root)
                break
    try:
        import onnxruntime as _ort
        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        cands.append(capi)
    except Exception:
        pass
    for p in cands:
        if ok(p):
            return p
    return None


def open_shim2(ren, prep_tex, res, model_path=None):
    """shim v2 工厂（产品 D3D 主路径 + T3 测试壳共用）。

    prep_tex: d3drender.Tex（D3DDepthPrep.prep）；ren: D3DRenderer。
    """
    ort_dir = find_ort_dir()
    if not ort_dir:
        raise RuntimeError(
            "[zc2] 缺 onnxruntime.dll+DirectML.dll（runtime/ 或 "
            "_toolchain/dl/onnxruntime-directml）")
    if model_path is None:
        import depth as _dm
        model_path = _dm._resolve_onnx_path()
    return DmlZeroCopyShim2(
        ort_dir, model_path, int(res), ren.get_dev_ptr(),
        ren.get_queue_ptr(), prep_tex.res, prep_tex.w, prep_tex.h)


class DmlZeroCopyShim:
    """create/run/destroy 的直通封装；异常一律抛 RuntimeError 由上层回退。"""

    def __init__(self, res, tex_w, tex_h, model_path, ort_dir, luid=0):
        dll_path = find_dll()
        if not dll_path:
            raise RuntimeError("[zc] d2s_dml_shim.dll 不存在（runtime/ 或 _toolchain/src/）")
        self._dll = ctypes.WinDLL(dll_path)
        self._dll.d2s_shim_create.restype = ctypes.c_void_p
        self._dll.d2s_shim_create.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int64, ctypes.c_wchar_p, ctypes.c_size_t]
        self._dll.d2s_shim_run.restype = ctypes.c_int
        self._dll.d2s_shim_run.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._dll.d2s_shim_times.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self._dll.d2s_shim_times6.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_double * 6)]
        self._dll.d2s_shim_destroy.argtypes = [ctypes.c_void_p]

        ort_dll = os.path.join(ort_dir, "onnxruntime.dll")
        dml_dll = os.path.join(ort_dir, "DirectML.dll")
        if not os.path.isfile(dml_dll):
            dml_dll = "DirectML.dll"  # capi 无随包 DirectML 时用系统自带（system32）
        err = ctypes.create_unicode_buffer(512)
        self._h = self._dll.d2s_shim_create(
            ort_dll, dml_dll, model_path, res, tex_w, tex_h, luid, err, 512)
        if not self._h:
            raise RuntimeError(f"[zc] shim 初始化失败: {err.value or '未知'}")

    def run(self, src_glo, out_ptr, out_bytes):
        rc = self._dll.d2s_shim_run(self._h, src_glo, out_ptr, out_bytes)
        if rc != 0:
            raise RuntimeError(f"[zc] shim run 失败 rc={rc}")

    def times(self):
        ti, tr = ctypes.c_double(), ctypes.c_double()
        self._dll.d2s_shim_times(self._h, ctypes.byref(ti), ctypes.byref(tr))
        return ti.value, tr.value

    def times6(self):
        """最近一帧 6 段分解（毫秒）：lock / glCopy+finish / unlock / km+fence+flush / d3d12 / run。"""
        a = (ctypes.c_double * 6)()
        self._dll.d2s_shim_times6(self._h, a)
        return list(a)

    def close(self):
        if getattr(self, "_h", None):
            self._dll.d2s_shim_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DmlZeroCopyShim2:
    """shim v2（迁移 T2）：D3D12 直入——输入 = d3drender 管线的 prep 纹理
    （同 device/queue 注入，无 GL/KM/fence，同队列 FIFO）。

    create(ort_dir, model_path, res, dev_ptr, queue_ptr, prep_res_ptr,
           prep_w, prep_h)
    run(out_ptr, out_bytes)  -> 0=ok
    times6() -> [copy_exec_ms, run_ms]
    """

    def __init__(self, ort_dir, model_path, res, dev_ptr, queue_ptr,
                 prep_res_ptr, prep_w, prep_h):
        dll_path = find_dll()
        if not dll_path:
            raise RuntimeError("[zc2] d2s_dml_shim.dll 不存在")
        self._dll = ctypes.WinDLL(dll_path)
        self._dll.d2s_shim2_create.restype = ctypes.c_void_p
        self._dll.d2s_shim2_create.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_wchar_p, ctypes.c_size_t]
        self._dll.d2s_shim2_run.restype = ctypes.c_int
        self._dll.d2s_shim2_run.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._dll.d2s_shim2_times6.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double * 2)]
        self._dll.d2s_shim2_run_gpu.restype = ctypes.c_int
        self._dll.d2s_shim2_run_gpu.argtypes = [ctypes.c_void_p]
        self._dll.d2s_shim2_out_buf.restype = ctypes.c_void_p
        self._dll.d2s_shim2_out_buf.argtypes = [ctypes.c_void_p]
        self._dll.d2s_shim2_gpu_error.restype = ctypes.c_char_p
        self._dll.d2s_shim2_gpu_error.argtypes = [ctypes.c_void_p]
        self._dll.d2s_shim2_destroy.argtypes = [ctypes.c_void_p]

        ort_dll = os.path.join(ort_dir, "onnxruntime.dll")
        dml_dll = os.path.join(ort_dir, "DirectML.dll")
        if not os.path.isfile(dml_dll):
            dml_dll = "DirectML.dll"
        err = ctypes.create_unicode_buffer(512)
        self._h = self._dll.d2s_shim2_create(
            ort_dll, dml_dll, model_path, res,
            dev_ptr, queue_ptr, prep_res_ptr, prep_w, prep_h, err, 512)
        if not self._h:
            raise RuntimeError(f"[zc2] shim2 初始化失败: {err.value or '未知'}")
        # ---- 异步执行（流水线用）：单工作线程 + 完成表 ----
        # shim2 的 Run 对 CPU 输出必然同步（ORT DML EP 的 Sync 会等到 GPU
        # 完成再 memcpy），所以"不阻塞"只能靠把它搬离主线程；session 由本
        # 线程独占，主线程只投递/取结果。
        self._q = None
        self._th = None
        self._res = {}
        self.last_times = [0.0, 0.0]
        self.last_job_ms = 0.0

    # ---- 异步接口 -----------------------------------------------------
    def _ensure_worker(self):
        if self._th is not None:
            return
        import queue as _queue
        import threading as _thr
        self._q = _queue.Queue(maxsize=1)
        self._th = _thr.Thread(target=self._worker, name="d2s-dml",
                               daemon=True)
        self._th.start()

    def _worker(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            tag, ptr, nb = job
            _t0 = time.perf_counter()
            try:
                self.run(ptr, nb)
                self._res[tag] = None        # None = 成功
            except BaseException as e:       # noqa: BLE001 - 交回主线程
                self._res[tag] = e
            self.last_job_ms = (time.perf_counter() - _t0) * 1000.0
            try:
                self.last_times = self.times6()
            except Exception:
                pass

    def submit(self, tag, out_ptr, out_bytes):
        """把一次 run 投给工作线程（不阻塞）。同一时刻只允许一个在飞。"""
        self._ensure_worker()
        if self._q.full():
            raise RuntimeError("[zc2] 推理队列已满（上一帧还没跑完）")
        self._q.put_nowait((tag, out_ptr, out_bytes))

    def poll(self, tag):
        """已完成返回 True 并清表；未完成返回 False。异常在主线程抛出。"""
        if tag not in self._res:
            return False
        err = self._res.pop(tag)
        if err is not None:
            raise err
        return True

    def run(self, out_ptr, out_bytes):
        if isinstance(out_ptr, int):
            out_ptr = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float))
        rc = self._dll.d2s_shim2_run(self._h, out_ptr, out_bytes)
        if rc != 0:
            raise RuntimeError(f"[zc2] shim2 run 失败 rc={rc}")

    def times6(self):
        """最近一帧 2 段分解（毫秒）：[copy+exec, run]。"""
        a = (ctypes.c_double * 2)()
        self._dll.d2s_shim2_times6(self._h, a)
        return list(a)

    # ---- GPU 输出（推理结果留在显存） -----------------------------------
    def out_buf_ptr(self):
        """输出 buffer 的 ID3D12Resource*（未启用 GPU 输出时返回 0）。"""
        if not getattr(self, "_h", None):
            return 0
        return int(self._dll.d2s_shim2_out_buf(self._h) or 0)

    def gpu_error(self):
        """GPU 输出初始化失败原因（空串 = 没失败过）。"""
        try:
            return (self._dll.d2s_shim2_gpu_error(self._h) or b"").decode(
                "ascii", "replace")
        except Exception:
            return ""

    def run_gpu(self):
        """只入队、不等待。返回 True=已入队；False=上一轮还在跑（这一拍跳过）。"""
        rc = self._dll.d2s_shim2_run_gpu(self._h)
        if rc == 3:
            return False
        if rc != 0:
            _extra = ""
            if rc in (9, 13):
                try:
                    _t = self.times6()
                    _extra = " hr=%#x" % (int(_t[0] if rc == 9 else _t[1])
                                          & 0xFFFFFFFF)
                except Exception:
                    pass
            raise RuntimeError(f"[zc2] run_gpu 失败 rc={rc}{_extra}")
        return True

    def close(self):
        if getattr(self, "_h", None):
            self._dll.d2s_shim2_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

