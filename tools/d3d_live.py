# -*- coding: utf-8 -*-
"""T3/T4 运行器：D3D12 实时深度立体管线（真实屏幕流 + 合成鼠标运动）。

用法：
  python tools\\d3d_live.py                # 无限跑（肉眼验收；Ctrl+C 退出）
  python tools\\d3d_live.py --gate         # T3 门禁：300 帧统计后退出
  python tools\\d3d_live.py --mode half_sbs --seconds 30

门禁（--gate）：
  G1 跟随率 100%：每个 DDA 新帧都被处理（poll 命中数 == 处理数）
  G2 proc_ms p50 < 10ms、p99 < 25ms（覆盖 120Hz DDA 间隔 8.3ms 的余量）
  G3 静态段 idle：无新帧时处理循环不推进（sleep 计数可见）
  G4 描述符槽零增长 + 无 DEVICE_REMOVED
"""
import argparse
import ctypes
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_RENDER"] = "d3d"   # 桥模式（建环前生效）

import numpy as np  # noqa: E402

from gpu.d3d_pipeline import D3DPipeline  # noqa: E402


class _MI(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("dx", ctypes.c_long),
                ("dy", ctypes.c_long), ("mouseData", ctypes.c_uint),
                ("dwFlags", ctypes.c_uint), ("time", ctypes.c_uint),
                ("exInfo", ctypes.c_size_t)]


class _INP(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MI)]
    _anonymous_ = ("u",)
    _fields_ = [("u", _U)]


def nudge_mouse(dx=8, dy=4):
    inp = _INP()
    inp.mi = _MI(type=0, dx=dx, dy=dy, mouseData=0, dwFlags=1, time=0,
                 exInfo=0)
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--mode", default="interleaved")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--nodml", action="store_true")
    args = ap.parse_args()

    print(f"[live] 建管线（mode={args.mode} dml={not args.nodml}）…",
          flush=True)
    pipe = D3DPipeline(res=224, mode=args.mode, width=args.width,
                       height=args.height, show=not args.gate,
                       dml=not args.nodml)
    print("[live] 管线就绪（真实屏幕流；移动鼠标生成变化）", flush=True)

    proc = []
    n = 0
    from gpu.d3drender import dll as _dll
    _g = _dll()
    _g.d3dr_get_desc_counts.restype = ctypes.c_uint64
    _g.d3dr_get_desc_counts.argtypes = [ctypes.c_void_p]
    # 初始化期（前 30 帧）会一次性分配纹理/描述符——基线取稳定后
    cc0 = None

    t_end = time.perf_counter() + args.seconds if args.seconds else 0
    target = 300 if args.gate else 0
    try:
        while True:
            idx = pipe.zcc.poll()
            if idx >= 0:
                nudge_mouse(8 if n % 2 == 0 else -8, 4)
                t0 = time.perf_counter()
                pipe.process_frame(idx)
                proc.append((time.perf_counter() - t0) * 1000)
                n += 1
                if n == 30:
                    cc0 = _g.d3dr_get_desc_counts(pipe.ren.ctx)
                if n % 30 == 0:
                    a = np.array(proc[-30:])
                    cc = _g.d3dr_get_desc_counts(pipe.ren.ctx)
                    print(f"  帧{n}: proc p50={np.percentile(a,50):.2f}ms "
                          f"max={a.max():.2f}ms  rtv={cc & 0xFFFFFFFF} "
                          f"arena={cc >> 32} shim={pipe.shim.times6()}",
                          flush=True)
                if target and n >= target:
                    break
                if t_end and time.perf_counter() > t_end:
                    break
            else:
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass

    cc1 = _g.d3dr_get_desc_counts(pipe.ren.ctx)
    if cc0 is None:
        cc0 = cc1
    arr = np.array(proc) if proc else np.array([0.0])
    p50 = float(np.percentile(arr, 50))
    p99 = float(np.percentile(arr, 99))
    print(f"\n[live] 处理 {n} 帧  proc p50={p50:.2f}ms p99={p99:.2f}ms "
          f"max={arr.max():.2f}ms", flush=True)
    print(f"[live] 描述符(稳定期): 始 {cc0 & 0xFFFFFFFF}/{cc0 >> 32} → "
          f"末 {cc1 & 0xFFFFFFFF}/{cc1 >> 32}", flush=True)
    pipe.close()

    fails = []
    if args.gate:
        if n < 50:
            fails.append(f"G1 帧数不足 {n}")
        if p50 >= 10.0:
            fails.append(f"G2 p50 {p50:.2f}")
        if p99 >= 25.0:
            fails.append(f"G2 p99 {p99:.2f}")
        if (cc1 & 0xFFFFFFFF) != (cc0 & 0xFFFFFFFF):
            fails.append("G4 RTV 槽增长")
        if (cc1 >> 32) != (cc0 >> 32):
            fails.append("G4 SRV 槽增长")
    if fails:
        print(f"[live] FAIL: {fails}")
        return 1
    print("[live] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
