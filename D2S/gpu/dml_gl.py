"""DirectML 推理器：默认 GL 回读，D2S_ZEROCOPY=1 时走零拷贝 shim。

回读路径（默认）：
    预处理纹理（r32f）→ glReadPixels → numpy [1,3,R,R] → ORT(DML) → _raw_tex
    开销实测（224）：回读 ~1.0ms + 推理 ~4.2ms + 上传 ~0.3ms ≈ 5.5ms。

零拷贝路径（P0/P1/P2 验证链，gpu/dml_shim.py + runtime/d2s_dml_shim.dll）：
    预处理纹理（flat 几何，W%64==0）→ interop 拷贝 → 共享 D3D12 buffer
    → OrtDmlApi 零拷贝输入 → 推理，省掉回读+pack（~1.3ms）。
    首帧做零输入 sanity（与 python 会话对账，P2 实测位级一致）；
    任何失败打印一次原因并永久回退回读路径，主循环无感。
"""
import ctypes
import os

import numpy as np

try:
    from gpu.dml_shim import zc_enabled      # 仅读环境变量，不加载 DLL
except Exception:                            # 没装/没构建时也能走回读路径
    def zc_enabled():
        return False


class DmlDepthRunner:
    """与 GpuDepthRunner 接口一致：infer(tex_raw) -> numpy 深度（未归一化）。"""

    def __init__(self, session, tex_prep_glo, tex_raw_glo, res: int,
                 tex_prep=None, tex_raw=None):
        prov = session.get_providers()
        if not prov or prov[0] != "DmlExecutionProvider":
            raise RuntimeError(f"[dml] 会话 provider={prov}，不是 DirectML")
        self.session = session
        self.res = int(res)
        self.tex_prep = tex_prep
        self.tex_raw = tex_raw
        if self.tex_prep is None or self.tex_raw is None:
            raise RuntimeError("[dml] 需要传 tex_prep / tex_raw 纹理对象")
        self.in_name = session.get_inputs()[0].name
        self._buf = np.empty((self.res, self.res * 3), dtype=np.float32)
        self._chw = np.empty((1, 3, self.res, self.res), dtype=np.float32)
        # flat 几何标记（viewer.gpu_depth_init 在 D2S_ZEROCOPY=1 时创建）：
        # W*H==3R² 且 W%64==0 且不是旧布局的 3R 宽
        _w, _h = tex_prep.size
        self._flat = (_w * _h == self.res * self.res * 3
                      and _w != self.res * 3 and _w % 64 == 0)
        # 零拷贝状态：None=未试，shim=在用，False=已回退
        self._zc = None
        self._zc_tried = False

    # ------------------------------------------------------------------ 零拷贝
    def _zc_try_init(self):
        """首次 infer 时懒初始化（必须在 GL 线程：interop 绑定当前上下文）。"""
        self._zc_tried = True
        if not self._flat:
            print("[zc] 预处理纹理不是 flat 几何（D2S_ZEROCOPY 需在 gpu_depth_init "
                  "前设置），零拷贝关闭", flush=True)
            return
        try:
            import onnxruntime as _ort
            _ort_dir = os.path.dirname(_ort.capi.__file__)
            _model = None
            try:
                import depth as _depth
                _model = _depth.depth_model.onnx_path
            except Exception:
                pass
            if not _model or not os.path.isfile(_model):
                raise RuntimeError("找不到模型路径（depth.depth_model.onnx_path）")
            _luid = 0
            try:
                from gpu import placement as _pl
                _luid = _pl.interop_luid()
            except Exception:
                pass
            from gpu.dml_shim import DmlZeroCopyShim, zc_enabled
            shim = DmlZeroCopyShim(self.res, self.tex_prep.size[0],
                                   self.tex_prep.size[1], _model, _ort_dir, _luid)
            # 零输入 sanity：shim 链 vs python 会话（P2 位级一致；对不上立刻回退）。
            # 注意：sanity 要把纹理临时清零，必须先把当帧真实内容存下来再还原，
            # 否则这一帧的深度就是全零（首帧真实深度丢失）。
            _saved = None
            try:
                _saved = self.tex_prep.read()
            except Exception:
                pass
            _mae = 1.0
            try:
                self.tex_prep.write(
                    np.zeros(self.res * self.res * 3, np.float32).tobytes())
                out0 = np.empty((self.res, self.res), np.float32)
                shim.run(self.tex_prep.glo,
                         out0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         out0.nbytes)
                ref = self.session.run(
                    None, {self.in_name: np.zeros((1, 3, self.res, self.res),
                                                  np.float32)})[0]
                _mae = float(np.abs(out0 - ref[0, 0]).max())
            finally:
                if _saved is not None:
                    try:
                        self.tex_prep.write(_saved)     # 还原当帧真实输入
                    except Exception:
                        pass
            if not (_mae < 1e-5):
                raise RuntimeError(f"sanity MAE={_mae:.3g} 超阈值")
            self._zc = shim
            self._zc_out = out0
            self._zc_ptr = out0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._zc_live_checked = False   # 首个真帧实况对账（覆盖 glFlush 化后的同步竞态）
            self._zc_next_check = 0.0       # 之后每 60s 复检一次
            print(f"[zc] 零拷贝链路激活（sanity MAE={_mae:.3g}，"
                  f"luid={hex(_luid) if _luid else '默认卡'}）", flush=True)
        except Exception as e:
            print(f"[zc] 零拷贝不可用，回退回读路径：{e}", flush=True)
            try:
                shim.close()
            except Exception:
                pass
            self._zc = False

    def _infer_zc(self):
        import time as _t
        t0 = _t.perf_counter()
        self._zc.run(self.tex_prep.glo, self._zc_ptr, self._zc_out.nbytes)
        _ti_ms, _tr_ms = self._zc.times()
        d = self._zc_out
        _now = _t.perf_counter()
        _chk_ms = 0.0
        if (not getattr(self, "_zc_live_checked", True)
                or _now >= getattr(self, "_zc_next_check", 0.0)):
            _c0 = _t.perf_counter()
            # 实况对账：同一帧的 shim 输出 vs python 回读+会话输出。
            # 首帧必做；之后每 60s 抽一次 —— glFlush 化的竞态往往要到负载高
            # 才出现，只查首帧覆盖不到。
            try:
                _raw = self.tex_prep.read()
                _arr = np.frombuffer(_raw, np.float32, self.res * self.res * 3)
                _arr = (_arr.reshape(3, self.res, self.res) if self._flat
                        else _arr.reshape(self.res, 3, self.res).transpose(1, 0, 2))
                np.copyto(self._chw[0], _arr)
                _ref = self.session.run(None, {self.in_name: self._chw})[0]
                _mae = float(np.abs(d - _ref[0, 0]).max())
                if _mae < 1e-5:
                    self._zc_live_checked = True
                    self._zc_next_check = _now + 60.0
                    print(f"[zc] 实况对账通过（MAE={_mae:.3g}）", flush=True)
                else:
                    raise RuntimeError(f"实况 MAE={_mae:.3g}：同步竞态或数据错位")
            except Exception as e:
                print(f"[zc] 实况对账失败，永久回退回读路径：{e}", flush=True)
                try:
                    self._zc.close()
                except Exception:
                    pass
                self._zc = False
                self._zc_live_checked = True
                return self._infer_readback()
            finally:
                _chk_ms = (_t.perf_counter() - _c0) * 1000.0
        _tw0 = _t.perf_counter()
        self.tex_raw.write(d.tobytes())
        _tw1 = _t.perf_counter()
        t4 = _tw1
        # 分段（秒，主循环 ×1000 显示）：read=GL 拷贝+KM+D3D12 拷贝（输入链）
        # pack=0 run=ORT 推理 write=GL 上传；other=不属于以上三段的 Python 侧
        # 开销（诊断用：曾在零拷贝下看到 write 残余 2ms，需要定位它到底在哪）
        _w_ms = _tw1 - _tw0
        self.times = {"read": _ti_ms / 1000.0, "pack": 0.0,
                      "run": _tr_ms / 1000.0, "write": _w_ms}
        self._zc_py_other = max(0.0, t4 - t0 - _ti_ms / 1000.0
                                - _tr_ms / 1000.0 - _w_ms - _chk_ms / 1000.0)
        # 每 5s 打一条 6 段分解（lock/copy/finish/unlock/km+d3d12/run），定位输入链耗时
        try:
            self._zc_dbg_n = getattr(self, "_zc_dbg_n", 0) + 1
            self._zc_dbg_acc = [a + b for a, b in
                                zip(getattr(self, "_zc_dbg_acc", [0.0] * 6), self._zc.times6())]
            if _t.perf_counter() - getattr(self, "_zc_dbg_t", 0.0) > 5.0:
                self._zc_dbg_t = _t.perf_counter()
                n = max(1, self._zc_dbg_n)
                if os.environ.get("D2S_ZC_DEBUG"):
                    print("[zc] lock %.2f copy %.2f unlock %.2f km %.2f d3d12 %.2f "
                          "run %.2f py %.2f (n=%d)"
                          % tuple([x / n for x in self._zc_dbg_acc]
                                  + [getattr(self, "_zc_py_other", 0.0) * 1000.0, n]),
                          flush=True)
                self._zc_dbg_n = 0
                self._zc_dbg_acc = [0.0] * 6
        except Exception:
            pass
        return d

    # ------------------------------------------------------------------ 回读
    def _infer_readback(self):
        import time as _t
        res = self.res
        _t0 = _t.perf_counter()
        raw = self.tex_prep.read()
        _t1 = _t.perf_counter()
        arr = np.frombuffer(raw, dtype=np.float32, count=res * res * 3)
        if self._flat:
            arr = arr.reshape(3, res, res)          # flat 几何：天然 CHW，免 transpose
        else:
            arr = arr.reshape(res, 3, res).transpose(1, 0, 2)   # (3, res, res)
        np.copyto(self._chw[0], arr)
        _t2 = _t.perf_counter()
        out = self.session.run(None, {self.in_name: self._chw})[0]
        _t3 = _t.perf_counter()
        d = np.ascontiguousarray(out[0, 0], dtype=np.float32)
        self.tex_raw.write(d.tobytes())
        _t4 = _t.perf_counter()
        # 分段耗时（秒）：read=glReadPixels（同步等 GPU） pack=重排/拷成 CHW
        # run=ORT/DML 推理 write=glTexSubImage
        self.times = {"read": _t1 - _t0, "pack": _t2 - _t1,
                      "run": _t3 - _t2, "write": _t4 - _t3}
        return d

    def infer(self, tex_raw=None):
        if not self._zc_tried and zc_enabled():
            self._zc_try_init()
        if self._zc is not None and self._zc is not False:
            return self._infer_zc()
        return self._infer_readback()
