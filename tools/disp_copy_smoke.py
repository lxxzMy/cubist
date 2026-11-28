# -*- coding: utf-8 -*-
"""显示拷贝门禁：新一对上屏前把环槽拷进一份私有纹理，之后的重呈现只读这份拷贝。

为什么要它：以前重呈现直接读环槽，每次都往渲染队列尾部再追加一个这个槽的
消费信号，采集要等它——而它排在推理、深度链后面。现在环槽只被上屏那一次
拷贝读，重呈现不碰它（main._tick_pipeline 第 2 步）。

断言（真实 DDA 采集 + 隐藏窗口离屏渲染）：
  D1 ring → disp 拷贝后，用 disp 渲染的结果与直接用环槽渲染的逐字节一致，
     且不是纯色（拷贝的确实是画面）
  D2 环槽随后被别的画面覆盖，用 disp 重呈现的结果不变
     （对照：这时直接读环槽渲染出来的是覆盖后的画面）
用法：D2S\\python3\\python.exe -I tools\\disp_copy_smoke.py
"""
import ctypes
import os
import sys
import time

os.environ["D2S_RENDER"] = "d3d"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

import numpy as np

from gpu import d3drender as dr
from gpu import dxgi_gpu_grab as zg
from gpu.dda_bridge import DdaBridge
from gpu.d3d_window import D3DStereoWindow
from gpu.rtcom import TexDesc, create_tex, ctx_copy_resource


class _Q:
    def publish(self, item):
        pass


def grab_frame(zcc, secs=4.0):
    """等到一帧真实采集（动一下鼠标让 DDA 产帧）。"""
    import ctypes.wintypes as wt
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < secs:
        if k % 5 == 0:
            ctypes.windll.user32.SetCursorPos(300 + (k % 8) * 9, 260 + (k % 5) * 7)
            k += 1
        idx = zcc.poll()
        if idx >= 0:
            return idx
        time.sleep(0.004)
    return -1


def main():
    fails = []
    win = D3DStereoWindow(display_mode="Interleaved", fill_16_9=True,
                          frame_size=(640, 360), same_screen=True,
                          cursor_overlay=False, convergence=0.5,
                          parallax_px=12.0, monitor_index=0, depth_res=(224, 224))
    ren = win.ren
    zcc = zg.DxgiGpuColor(_Q(), int(getattr(win, "output_index", 0)),
                          adapter=getattr(win, "adapter", None))
    try:
        zcc.build_inline(gl_ctx=None)
        bridge = DdaBridge(ren, dev11=zcc._dev, ctx11=zcc._ctx,
                           adv=getattr(zcc, "adapter", None))
        zcc.attach_bridge_fence(*bridge.producer_handles(),
                                bridge.consumer_handles()[0])
        ring = bridge.open_ring(zcc._ring, zcc._w, zcc._h)
        w, h = zcc._w, zcc._h
        idx = grab_frame(zcc)
        if idx < 0:
            fails.append("D0 没采到帧（DDA 未交付）")
            raise RuntimeError(fails)
        bridge.queue_wait(zcc.bridge_seq)      # 等采集拷贝落地
        # 与 main._tick_pipeline 第 2 步相同的拷贝
        disp = ren.create_tex(w, h, dr.FMT_BGRA8, renderable=False)
        bridge.queue_wait(zcc.slot_seq[idx])
        ren.begin_frame()
        ren.copy_tex_to_tex(disp, ring[idx])
        ren.submit()
        win.post.upload_raw(np.full((224, 224), 0.5, np.float32))
        win.post.spatial_chain(0.0, 1.0, aa_texels=2.0, temporal="ema", alpha=1.0)
        depth = win.post.last_disp_depth
        dw, dh = win.disp.w, win.disp.h

        def show(color):
            out = ren.create_tex(dw, dh, dr.FMT_BGRA8, renderable=True)
            win.disp.draw("interleaved", color_tex=color, depth_tex=depth,
                          src_w=w, src_h=h, out=out)
            a = ren.readback(out, np.uint8)
            out.release()
            return a

        from_ring = show(ring[idx])
        from_disp = show(disp)
        if float(from_ring[:, :, :3].std()) < 1.0:
            fails.append("D0 采到的画面是纯色，没法验证")
        if not np.array_equal(from_ring, from_disp):
            diffs = int(np.count_nonzero(from_ring != from_disp))
            fails.append(f"D1 用 disp 渲染与直接用环槽不一致（{diffs} 个字节不同）")
        # 环槽随后被覆盖成黑（模拟采集把新帧写进这个槽）
        black = create_tex(zcc._dev, TexDesc(w, h, 1, 1, 87, 1, 0, 0, 8, 0, 0))
        zeros = np.zeros((h, w, 4), np.uint8)       # 保持引用：只取 data 会悬空
        zcc._ctx.call(48, None, (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_uint),
                      black.p, 0, None, zeros.ctypes.data, w * 4)
        v = zcc.bridge_seq
        ctx_copy_resource(zcc._ctx, zcc._ring[idx], black)
        zcc.bridge_signal_after_copy(idx)
        bridge.queue_wait(zcc.bridge_seq)
        rep = show(disp)
        now_ring = show(ring[idx])
        if not np.array_equal(rep, from_disp):
            fails.append("D2 环槽被覆盖后，用 disp 重呈现的结果变了")
        if np.array_equal(now_ring, from_disp) or float(now_ring[:, :, :3].mean()) > 2.0:
            fails.append("D2 对照失败：环槽没有被覆盖成黑（测试本身没生效）")
        print(f"[D1] 拷贝后渲染一致；画面均值 {float(from_ring[:, :, :3].mean()):.1f} "
              f"标准差 {float(from_ring[:, :, :3].std()):.1f}", flush=True)
        print(f"[D2] 环槽覆盖成黑后：用 disp 重呈现不变；直接读环槽 均值 "
              f"{float(now_ring[:, :, :3].mean()):.1f}", flush=True)
        disp.release()
        black.release()
        for t in ring:
            t._srv = None
        bridge.close()
    finally:
        try:
            zcc.close()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        try:
            ren.destroy()
        except Exception:
            pass
    if fails:
        print(f"[disp-copy] FAIL: {fails}")
        return 1
    print("[disp-copy] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
