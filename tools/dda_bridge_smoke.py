# -*- coding: utf-8 -*-
"""T1 门禁：DDA 桥生产化冒烟（真实屏幕流，D2S_RENDER=d3d）。

链路：DxgiGpuColor(d3d 环 NTHANDLE|KM) → poll 拷贝+KM+Signal →
DdaBridge(D3D12 OpenSharedHandle 缓存 + queue.Wait) → readback 验证内容。

断言：
  B1 环纹理带 NTHANDLE|KM 创建成功（无 B8G8R8A8 兼容问题）
  B2 D3D12 open 全槽成功 + LUID 校验通过
  B3 1000 次 poll 循环零异常；有新帧时 signal/seq 对账一致
  B4 readback 内容非恒黑（桥数据真实流动）
  B5 描述符槽计数稳定（不随 poll 增长）
用法：D2S\\python3\\python.exe -I tools\\dda_bridge_smoke.py
"""
import ctypes
import os
import sys
import time

os.environ["D2S_RENDER"] = "d3d"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

import numpy as np

from gpu import d3drender as dr
from gpu import adapter as _ad
from gpu import dxgi_gpu_grab as _zcc
from gpu.dda_bridge import DdaBridge
from gpu.rtcom import ComPtr

POLL_LOOPS = 1000


def nudge_mouse():
    """合成小幅鼠标移动（让 DDA 产帧）。"""
    class _INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            class _MI(ctypes.Structure):
                _fields_ = [("type", ctypes.c_uint), ("dx", ctypes.c_long),
                            ("dy", ctypes.c_long), ("mouseData", ctypes.c_uint),
                            ("dwFlags", ctypes.c_uint), ("time", ctypes.c_uint),
                            ("exInfo", ctypes.c_size_t)]
            _fields_ = [("mi", _MI)]
        _anonymous_ = ("u",)
        _fields_ = [("u", _U)]
    inp = _INPUT()
    inp.mi = (_INPUT._U._MI(type=0, dx=6, dy=3, mouseData=0, dwFlags=1,
                            time=0, exInfo=0))
    u32 = ctypes.windll.user32
    u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def main():
    fails = []

    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    bridge = DdaBridge(ren)

    class _Q:
        def publish(self, item):
            self.last = item
    q = _Q()
    zcc = _zcc.DxgiGpuColor(q, 0)
    zcc.build_inline(gl_ctx=None)   # d3d 模式：无 GL 注册；环已带 NTHANDLE|KM
    if not zcc.bridge_kms:
        fails.append("B1 环纹理未进桥模式（bridge_kms 空）")
        print(f"[bridge-smoke] FAIL: {fails}")
        return 1
    print(f"[B1] 环 NTHANDLE|KM 创建 OK（{len(zcc.bridge_kms)} 槽）", flush=True)

    f11, ctx4 = bridge.producer_handles()
    zcc.attach_bridge_fence(f11, ctx4)
    ring_texes = bridge.open_ring(zcc._ring, zcc._w, zcc._h)
    print(f"[B2] D3D12 open 全槽 OK（{len(ring_texes)} 纹理）", flush=True)

    # 1000 次 poll：静止/运动混合（每 20 次循环动一次鼠标）
    n_frames = 0
    err = 0
    d = dr.dll()
    d.d3dr_get_desc_counts.restype = ctypes.c_uint64
    d.d3dr_get_desc_counts.argtypes = [ctypes.c_void_p]
    cc0 = d.d3dr_get_desc_counts(ren.ctx)
    v_last = 0
    mismatches = 0
    t0 = time.perf_counter()
    for i in range(POLL_LOOPS):
        if i % 20 == 0:
            nudge_mouse()
        idx = zcc.poll()
        if idx >= 0:
            n_frames += 1
            v = zcc.bridge_seq
            if v != v_last + 1:
                mismatches += 1
            v_last = v
            # GPU 门控 + readback 验证（每 100 帧；前 5 帧是 DDA duplication
            # 初始化期已知的空帧——不计黑帧断言）
            if n_frames % 100 == 1:
                bridge.queue_wait(v)
                arr = ren.readback(ring_texes[idx], np.uint8)
                if arr.std() < 1.0 and arr.mean() < 1.0:
                    if n_frames <= 5:
                        print(f"  [预热黑帧] 帧{n_frames}（DDA 初始化空帧，跳过）",
                              flush=True)
                    else:
                        err += 1
                        print(f"  [黑帧] 帧{n_frames} slot{idx} v={v} "
                              f"min={arr.min()} max={arr.max()} "
                              f"mean={arr.mean():.2f} "
                              f"t={time.perf_counter()-t0:.1f}s", flush=True)
        if i % 200 == 0:
            cc = d.d3dr_get_desc_counts(ren.ctx)
            print(f"  poll {i}: 帧数={n_frames} seq={zcc.bridge_seq} "
                  f"rtv={cc & 0xFFFFFFFF} arena={cc >> 32}", flush=True)
    dt = time.perf_counter() - t0
    cc1 = d.d3dr_get_desc_counts(ren.ctx)
    print(f"[B3] {POLL_LOOPS} 次 poll 完成 {dt:.1f}s：帧数={n_frames} "
          f"seq 对账失配={mismatches} 黑帧={err}", flush=True)
    print(f"[B5] 描述符: 始 {cc0 & 0xFFFFFFFF}/{cc0 >> 32} → "
          f"末 {cc1 & 0xFFFFFFFF}/{cc1 >> 32}", flush=True)

    if n_frames == 0:
        fails.append("B3 零帧（DDA 未交付）")
    if mismatches:
        fails.append(f"B3 seq 失配 {mismatches}")
    if err:
        fails.append(f"B4 黑帧 {err}")
    if (cc1 & 0xFFFFFFFF) != (cc0 & 0xFFFFFFFF) or (cc1 >> 32) != (cc0 >> 32):
        fails.append("B5 描述符泄漏")

    for t in ring_texes:
        t._srv = None   # 外部资源：res 不归 Tex 管（桥生命周期）
    bridge.close()
    ren.destroy()
    if fails:
        print(f"[bridge-smoke] FAIL: {fails}")
        return 1
    print("[bridge-smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
