# -*- coding: utf-8 -*-
"""CUDA-GL 互操作 + TRT IOBinding 推理器（M0 闸 3 验证路径的产品化封装）。

帧路径（GL 线程独占使用）：
  viewer.gpu_preprocess() 把色纹理降采样+归一化进 tex_prep（R32F 3R×R，CHW 平铺）
  → infer(): map 两纹理 → cudaMemcpy2DFromArray(tex_prep→dev_in)
    → session.run_with_iobinding → cudaMemcpy2DToArray(dev_out→tex_depth R32F R×R)
    → unmap → GL 读回 tex_depth 得模型原始输出（numpy R×R float32）

数值与 CPU 路径逐位一致已由 M0 闸 3 证明（max=0.00e+00）。线程规则：
cudaGraphicsGLRegisterImage/Map/Unmap 全部在持有 GL context 的线程调用。
"""
import ctypes

import numpy as np

P = ctypes.c_void_p
cudaSuccess = 0
cudaMemcpyDeviceToDevice = 3
GL_TEXTURE_2D = 0x0DE1


def _load_cudart():
    dll = ctypes.CDLL("cudart64_12.dll")
    dll.cudaMalloc.argtypes = [ctypes.POINTER(P), ctypes.c_size_t]
    dll.cudaFree.argtypes = [P]
    dll.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(P), ctypes.c_uint,
                                                ctypes.c_uint, ctypes.c_uint]
    dll.cudaGraphicsMapResources.argtypes = [ctypes.c_int, ctypes.POINTER(P), P]
    dll.cudaGraphicsUnmapResources.argtypes = [ctypes.c_int, ctypes.POINTER(P), P]
    dll.cudaGraphicsSubResourceGetMappedArray.argtypes = [
        ctypes.POINTER(P), P, ctypes.c_uint, ctypes.c_uint]
    dll.cudaMemcpy2DFromArray.argtypes = [P, ctypes.c_size_t, P, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_int]
    dll.cudaMemcpy2DToArray.argtypes = [P, ctypes.c_size_t, ctypes.c_size_t, P,
                                        ctypes.c_size_t, ctypes.c_size_t,
                                        ctypes.c_size_t, ctypes.c_int]
    return dll


class GpuDepthRunner:
    """绑定一对 GL 纹理（prep 入 / depth 出）与 ORT 会话的常驻推理器。

    init 与 infer 都必须在 GL 线程（context current）调用。
    失败（无 CUDA/无 GPU 纹理/EP 不支持）抛异常，由调用方降级 CPU 路径。
    """

    def __init__(self, session, tex_prep_glo, tex_depth_glo, res: int):
        prov = session.get_providers()
        if prov and prov[0] not in ("TensorrtExecutionProvider",
                                    "CUDAExecutionProvider"):
            raise RuntimeError(
                f"[gpu] 会话无可用的 GPU EP（{prov[0]}），GPU 直通不可用")
        self.session = session
        self.res = res
        self.fn = _load_cudart()
        self._res_in = P()
        self._res_out = P()
        self._ck(self.fn.cudaGraphicsGLRegisterImage(
            ctypes.byref(self._res_in), tex_prep_glo, GL_TEXTURE_2D, 0),
            "register tex_prep")
        self._ck(self.fn.cudaGraphicsGLRegisterImage(
            ctypes.byref(self._res_out), tex_depth_glo, GL_TEXTURE_2D, 0),
            "register tex_depth")
        in_bytes = res * res * 3 * 4          # [1,3,R,R] float32
        out_bytes = res * res * 4             # [1,1,R,R] float32
        self._dev_in = P()
        self._dev_out = P()
        self._ck(self.fn.cudaMalloc(ctypes.byref(self._dev_in), in_bytes), "malloc in")
        self._ck(self.fn.cudaMalloc(ctypes.byref(self._dev_out), out_bytes), "malloc out")

        io = session.io_binding()
        io.bind_input(session.get_inputs()[0].name, "cuda", 0, np.float32,
                      [1, 3, res, res], self._dev_in.value)
        io.bind_output(session.get_outputs()[0].name, "cuda", 0, np.float32,
                       [1, 1, res, res], self._dev_out.value)
        self.io = io
        self._pair = (P * 2)(self._res_in, self._res_out)

    @staticmethod
    def _ck(hr, what):
        if hr != 0:
            raise OSError(hr, f"[gpu] {what} hr={hr}")

    def infer(self, tex_depth) -> np.ndarray:
        """跑一帧：GL 纹理 → GPU 推理 → 写回 depth 纹理 → 读回 numpy (R,R)。"""
        fn = self.fn
        res = self.res
        import time as _t
        _t0 = _t.perf_counter()
        self._ck(fn.cudaGraphicsMapResources(2, self._pair, None), "map")
        _t1 = _t.perf_counter()
        try:
            arr_in, arr_out = P(), P()
            self._ck(fn.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(arr_in), self._res_in, 0, 0), "array in")
            self._ck(fn.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(arr_out), self._res_out, 0, 0), "array out")
            # prep 纹理是「每行 R|G|B 三块横排」，模型输入要的是 CHW
            # （每个通道一张完整平面）。整行拷贝会把行内 RGB 交错进 CHW，
            # 模型等于在读一张错乱的图（实测 prep 与画面相关 0.985，
            # 但模型吃到的是错排数据 → 深度与画面无关）。
            # 逐通道拷：源列偏移 c*res，目标偏移 c*res*res。
            for _c in range(3):
                self._ck(fn.cudaMemcpy2DFromArray(
                    ctypes.c_void_p(self._dev_in.value + _c * res * res * 4),
                    res * 4, arr_in, _c * res * 4, 0,
                    res * 4, res, cudaMemcpyDeviceToDevice), f"copy in c{_c}")
            _t2 = _t.perf_counter()
            self.session.run_with_iobinding(self.io)
            _t3 = _t.perf_counter()
            self._ck(fn.cudaMemcpy2DToArray(
                arr_out, 0, 0, self._dev_out, res * 4,
                res * 4, res, cudaMemcpyDeviceToDevice), "copy out")
            _t4 = _t.perf_counter()
        finally:
            fn.cudaGraphicsUnmapResources(2, self._pair, None)
        # R32F 纹理 read 返回原始字节即 float32（alignment=1 免行距填充）
        _arr = np.frombuffer(tex_depth.read(alignment=1),
                             np.float32).reshape(res, res)
        _t5 = _t.perf_counter()
        self.times = {"map": _t1 - _t0, "copy_in": _t2 - _t1,
                      "run": _t3 - _t2, "copy_out": _t4 - _t3,
                      "read": _t5 - _t4}
        return _arr

    def keepalive(self, tex_depth):
        """闲置保温：重跑上一输入（结果丢弃）。"""
        self.infer(tex_depth)
