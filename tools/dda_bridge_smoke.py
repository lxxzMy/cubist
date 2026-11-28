# -*- coding: utf-8 -*-
"""T1 门禁：DDA 桥生产化冒烟（真实屏幕流，D2S_RENDER=d3d）。

链路：DxgiGpuColor(d3d 环 NTHANDLE|KM，锁建环时已取) → poll 拷贝+Signal →
DdaBridge(D3D12 OpenSharedHandle 缓存 + queue.Wait) → readback 验证内容。

断言：
  B0 桥必须建在采集设备上：建在别的设备上时，接线（attach_bridge_fence）直接拒绝
  B1 环纹理带 NTHANDLE|KM 创建成功（无 B8G8R8A8 兼容问题）
  B2 D3D12 open 全槽成功 + LUID 校验通过
  B3 1000 次 poll 循环零异常；有新帧时 signal/seq 对账一致
  B4 readback 内容非恒黑（桥数据真实流动）。最前面 6 帧逐帧查：带锁共享的
     环槽没取锁时，每个槽的第一次拷贝会被丢（2026-11-28 实测进 3D 的前 3 帧
     全黑、poll 却报有新帧）；之后每 100 帧查一次
  B5 描述符槽计数稳定（不随 poll 增长）
  B6 拷贝 → Signal → queue_wait → 立刻读回：渲染队列被重活压着时，读到的
     必须是刚拷进去的颜色（0 次旧内容）。桥建在别的设备上时，Signal 与拷贝
     不在同一设备的队列里、没有先后保证，这一步会读到环槽里上一轮的内容
     （2026-11-28 "帧号在涨、画面往回跳"的根因）。
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
FIRST_FRAMES = 6                 # 逐帧读回的最前面几帧（三个槽各用两次）
RACE_STEPS, RACE_HEAVY = 150, 20
RACE_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))     # BGRA


def nudge_mouse(dx=6):
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
    inp.mi = (_INPUT._U._MI(type=0, dx=dx, dy=3, mouseData=0, dwFlags=1,
                            time=0, exInfo=0))
    u32 = ctypes.windll.user32
    u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def race_check(ren, zcc, bridge, ring_texes):
    """B6：照产品的协议走——采集设备上 拷贝 → Signal → Flush；渲染队列
    queue_wait(信号值) → 立刻把环槽读回。每步先给渲染队列压 RACE_HEAVY 次
    4K 清屏（模拟推理占着 GPU 时拷贝被排队）。返回读到旧内容的次数。

    颜色按"轮"换（一轮把同一个颜色写遍所有槽，下一轮换色）：每次写进某槽
    的颜色都与这个槽里原有的不同，读到旧内容才会暴露。
    """
    from gpu.rtcom import TexDesc, create_tex, ctx_copy_resource
    w, h = zcc._w, zcc._h
    n = len(ring_texes)
    srcs = []
    for c in RACE_COLORS:
        t = create_tex(zcc._dev, TexDesc(w, h, 1, 1, 87, 1, 0, 0, 8, 0, 0))
        img = np.empty((h, w, 4), np.uint8)
        img[:, :, 0], img[:, :, 1] = c[0], c[1]
        img[:, :, 2], img[:, :, 3] = c[2], 255
        # ID3D11DeviceContext::UpdateSubresource（槽 48）：整张填成纯色
        zcc._ctx.call(48, None, (ctypes.c_void_p, ctypes.c_uint,
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_uint),
                      t.p, 0, None, img.ctypes.data, w * 4)
        zcc._ctx.call(111, None, ())            # Flush
        srcs.append(t)
    heavy = ren.create_tex(3840, 2160, dr.FMT_BGRA8, renderable=True)
    stale = 0
    for k in range(RACE_STEPS):
        idx = k % n
        ci = (k // n) % len(RACE_COLORS)
        for _ in range(RACE_HEAVY):
            ren.begin_frame()
            heavy.state = dr.ST_COMMON
            heavy.transition(dr.ST_RENDER_TARGET)
            ren.clear_rtv(heavy, (0.0, 0.0, 0.0, 1.0))
            ren.submit()
        ctx_copy_resource(zcc._ctx, zcc._ring[idx], srcs[ci])
        v = zcc.bridge_signal_after_copy(idx)   # seq+1 → Signal → Flush
        bridge.queue_wait(v)
        got = tuple(int(x) for x in ren.readback(ring_texes[idx], np.uint8)[5, 5, :3])
        if got != RACE_COLORS[ci]:
            stale += 1
            print(f"  [旧内容] 步{k} slot{idx} v={v} 期望{RACE_COLORS[ci]} "
                  f"读到{got}", flush=True)
    heavy.release()
    return stale


def main():
    fails = []

    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)

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

    # B0：桥自建设备（旧接法）时，接线必须被拒绝——Signal/Wait 必须与环纹理
    # 的拷贝同在采集设备的上下文上，否则渲染队列会读到拷贝未落地的旧内容
    foreign = DdaBridge(ren)
    try:
        zcc.attach_bridge_fence(*foreign.producer_handles())
        fails.append("B0 异设备的 ctx4 没被拒绝")
    except RuntimeError:
        print("[B0] 异设备的 ctx4 被拒绝 OK", flush=True)
    finally:
        foreign.close()

    bridge = DdaBridge(ren, dev11=zcc._dev, ctx11=zcc._ctx,
                       adv=getattr(zcc, "adapter", None))
    f11, ctx4 = bridge.producer_handles()
    fcons11, _ = bridge.consumer_handles()
    zcc.attach_bridge_fence(f11, ctx4, fcons11)
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
        if i % 10 == 0:
            nudge_mouse(6 if (i // 10) % 2 == 0 else -6)   # 原地来回，不漂移
        idx = zcc.poll()
        if idx >= 0:
            n_frames += 1
            v = zcc.bridge_seq
            if v != v_last + 1:
                mismatches += 1
            v_last = v
            # GPU 门控 + readback 验证
            if n_frames <= FIRST_FRAMES or n_frames % 100 == 0:
                bridge.queue_wait(v)
                arr = ren.readback(ring_texes[idx], np.uint8)
                if arr.std() < 1.0 and arr.mean() < 1.0:
                    err += 1
                    print(f"  [黑帧] 帧{n_frames} slot{idx} v={v} "
                          f"min={arr.min()} max={arr.max()} "
                          f"mean={arr.mean():.2f} "
                          f"t={time.perf_counter()-t0:.1f}s", flush=True)
        else:
            time.sleep(0.002)   # 空转 1000 次只要 0.3s，DDA 来不及产帧
        if i % 200 == 0:
            cc = d.d3dr_get_desc_counts(ren.ctx)
            print(f"  poll {i}: 帧数={n_frames} seq={zcc.bridge_seq} "
                  f"rtv={cc & 0xFFFFFFFF} arena={cc >> 32}", flush=True)
    dt = time.perf_counter() - t0
    cc1 = d.d3dr_get_desc_counts(ren.ctx)
    print(f"[B3] {POLL_LOOPS} 次 poll 完成 {dt:.1f}s：帧数={n_frames} "
          f"seq 对账失配={mismatches} 黑帧={err}", flush=True)
    print(f"[B4] 前 {min(n_frames, FIRST_FRAMES)} 帧逐帧读回"
          f"{'全部非黑' if not err else f'黑帧 {err}'}"
          f"（之后每 100 帧查一次）", flush=True)
    print(f"[B5] 描述符: 始 {cc0 & 0xFFFFFFFF}/{cc0 >> 32} → "
          f"末 {cc1 & 0xFFFFFFFF}/{cc1 >> 32}", flush=True)

    stale = race_check(ren, zcc, bridge, ring_texes)
    print(f"[B6] {RACE_STEPS} 步（每步先压 {RACE_HEAVY} 次 4K 清屏）："
          f"读到旧内容 {stale} 次", flush=True)

    if n_frames == 0:
        fails.append("B3 零帧（DDA 未交付）")
    if mismatches:
        fails.append(f"B3 seq 失配 {mismatches}")
    if n_frames < FIRST_FRAMES:
        fails.append(f"B4 帧数不足 {n_frames}（< {FIRST_FRAMES}，前几帧没法逐帧查）")
    if err:
        fails.append(f"B4 黑帧 {err}")
    if (cc1 & 0xFFFFFFFF) != (cc0 & 0xFFFFFFFF) or (cc1 >> 32) != (cc0 >> 32):
        fails.append("B5 描述符泄漏")
    if stale:
        fails.append(f"B6 读到旧内容 {stale} 次（Signal 与拷贝不同设备，"
                     f"或环槽没取到锁）")

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
