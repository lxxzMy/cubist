# -*- coding: utf-8 -*-
"""D3D12 实时流水线（迁移 T3）：DDA 采集 → D3D12 prep → DML shim v2 →
深度链 → 立体显示，事件驱动（跟随 DDA 新帧交付率，无新帧不空转推理）。

帧编排（同队列 FIFO，帧内 CPU 同步点仅 2 个）：
  1. DdaBridge queue_wait(v)（GPU 端门控，CPU 不阻塞）
  2. begin_frame(A)：录 prep pass（桥纹理采样 → prep 纹理）→ submit
  3. dml_shim2.run（录 prep→rBuf copy + DML Run 阻塞至 CPU float）
  4. percentile_stats / cut_hit（CPU，现状生产同款）
  5. begin_frame(B)：录深度链 + 显示 pass → present（屏障+提交一条龙）
per-frame allocator 轮换（C 侧 2 组 + 组级 fence 对账）消除帧间硬等。
"""
import ctypes
import os
import time

import numpy as np

from . import d3drender as dr
from . import adapter as _ad
from . import d3d_view as dv
from .d3d_post import D3DDepthPost
from .d3d_prep import D3DDepthPrep
from .dda_bridge import DdaBridge
from .dml_shim import DmlZeroCopyShim2


class D3DPipeline:
    """D3D12 实时深度立体管线（测试壳：独立 Win32 窗口 + 热键切档）。"""

    def __init__(self, res=224, mode="interleaved", width=1280, height=720,
                 show=True, dml=True):
        self.res = res
        self.mode = mode          # interleaved | half_sbs | depth_map | passthrough
        self.width, self.height = width, height
        adv = _ad.pick()
        self.ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
        self.win = dv.Win32Window(width, height, show=show)
        self.disp = dv.D3DDisplay(self.ren, self.win.hwnd, width, height)
        self.prep = D3DDepthPrep(self.ren, res)
        self.post = D3DDepthPost(self.ren, res)

        # DDA 采集（d3d 桥模式）+ 桥
        os.environ.setdefault("D2S_RENDER", "d3d")   # 建环前必须已设
        from gpu import dxgi_gpu_grab as _zcc
        from gpu.rtcom import ComPtr

        class _Q:
            def publish(self, item):
                pass
        self.zcc = _zcc.DxgiGpuColor(_Q(), 0)
        self.zcc.build_inline(gl_ctx=None)   # 环已 NTHANDLE|KM（env 门控）
        self.bridge = DdaBridge(self.ren)
        f11, ctx4 = self.bridge.producer_handles()
        self.zcc.attach_bridge_fence(f11, ctx4)
        self.ring_texes = self.bridge.open_ring(self.zcc._ring,
                                                self.zcc._w, self.zcc._h)

        # shim v2（DML 直入；失败回落 CPU ONNX 由调用方处理——本壳 fail-loud）
        if dml:
            import depth as depth_mod
            model = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
                "runtime", "depth-224.onnx")
            capi = None
            for base, _dirs, files in os.walk(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__)))), "_toolchain", "dl")):
                if "onnxruntime.dll" in files and "DirectML.dll" in files:
                    capi = base
                    break
            if capi is None:
                raise RuntimeError("[pipeline] 无 DML capi（跑 "
                                   "pip install --target _toolchain/dl/ort-dml "
                                   "onnxruntime-directml）")
            self.shim = DmlZeroCopyShim2(
                capi, model, res, self.ren.get_dev_ptr(),
                self.ren.get_queue_ptr(), self.prep.prep.res,
                self.prep.prep_w, self.prep.prep_h)
        self.dml = dml

        self.out_buf = np.empty(res * res, np.float32)
        self._ema_first = True
        self._prev_z = None
        self._cut_t = 0.0
        # 遥测
        self.dda_fps = 0.0    # DDA 新帧交付率
        self.pipe_fps = 0.0   # 管线消费率
        self.proc_ms = 0.0    # 每新帧处理耗时（p50 由调用方采样）

    # ---- 单帧处理（新帧到达时调用） -------------------------------------
    def process_frame(self, idx):
        """idx=桥环 slot。链+显示全跑；返回 (raw, lo, hi, alpha)。"""
        from depth_norm import percentile_stats

        t0 = time.perf_counter()
        v = self.zcc.bridge_seq
        self.bridge.queue_wait(v)   # GPU 端门控

        # 1) prep pass（桥纹理直接采样，无 CPU 上传）
        self.ren.begin_frame()
        src = self.ring_texes[idx]
        src.state = dr.ST_COMMON
        src.transition(dr.ST_PIXEL_SRV)
        self.prep.prep.state = dr.ST_COMMON
        self.prep.prep.transition(dr.ST_RENDER_TARGET)
        self.ren.draw_pass(self.prep._pso, [src],
                           [float(self.res), float(self.prep.flat_w), 0.0,
                            0.0],
                           self.prep.prep, self.prep.prep_w, self.prep.prep_h)
        self.prep.prep.state = dr.ST_COMMON
        self.ren.submit()

        # 2) DML 推理（内部 copy prep→rBuf + Run 阻塞至 CPU）
        self.shim.run(ctypes.cast(self.out_buf.ctypes.data,
                                  ctypes.POINTER(ctypes.c_float)),
                      self.out_buf.nbytes)
        raw = self.out_buf.reshape(self.res, self.res)
        sub = raw[::4, ::4]
        lo = float(np.percentile(sub, 0.5))
        hi = float(np.percentile(sub, 99.5))
        alpha = 1.0 if self._ema_first else 0.19

        # 3) 深度链 + 显示（同一列表，present 一条龙提交）
        self.post.upload_raw(raw)
        self.post.spatial_chain(lo, hi, aa_texels=3.0, temporal="ema",
                                alpha=alpha)
        self._ema_first = False

        color_tex = self.ring_texes[idx]
        depth_tex = self.post.depth
        if self.mode == "interleaved":
            self.disp.render_interleaved(
                color_tex, depth_tex,
                dict(ipd_uv=0.064, parallax_px=12.0, convergence=0.5,
                     depth_strength=1.0, fill=0.0, ic_search=1.0,
                     phase=0.0, edge_snap=1.0, relief=0.22, warp=1.0))
        elif self.mode == "half_sbs":
            self.disp.render_stereo(
                "Half-SBS", color_tex, depth_tex,
                dict(ipd_uv=0.064, parallax_px=12.0, convergence=0.5,
                     depth_strength=1.0, fill=1.0, ic_search=1.0))
        elif self.mode == "depth_map":
            self.disp.render("depth_map", depth_tex=depth_tex)
        else:
            self.disp.render("passthrough", color_tex=color_tex)

        self.proc_ms = (time.perf_counter() - t0) * 1000
        return raw, lo, hi, alpha

    # ---- 主循环（事件驱动） ---------------------------------------------
    def run(self, max_frames=0, on_frame=None):
        """事件驱动主循环。max_frames=0 无限；on_frame(proc_ms) 回调。
        无新帧：管线 idle（sleep，不空转推理；窗口内容由 DWM 保持）。"""
        n = 0
        new_cnt = 0
        t_stat = time.perf_counter()
        proc_samples = []
        while True:
            idx = self.zcc.poll()
            if idx >= 0:
                new_cnt += 1
                self.process_frame(idx)
                proc_samples.append(self.proc_ms)
                if len(proc_samples) > 300:
                    proc_samples.pop(0)
                self.proc_p50 = sorted(proc_samples)[len(proc_samples) // 2] \
                    if proc_samples else 0.0
                n += 1
                if on_frame:
                    on_frame(self.proc_ms)
                if max_frames and n >= max_frames:
                    break
            else:
                time.sleep(0.002)   # 无新帧：idle（不空转推理）
            now = time.perf_counter()
            if now - t_stat >= 1.0:
                self.dda_fps = new_cnt
                self.pipe_fps = n if not max_frames else min(n, new_cnt)
                new_cnt = 0
                t_stat = now

    def close(self):
        try:
            self.win.close()
        except Exception:
            pass
        try:
            self.ren.destroy()
        except Exception:
            pass
