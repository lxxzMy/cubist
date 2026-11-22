# -*- coding: utf-8 -*-
"""DDA 原始呈现速率探针（独立进程，可与产品同时运行）。

测的是【桌面合成器】实际交付的帧：AcquireNextFrame 的 LastPresentTime
间隔 + AccumulatedFrames 分布。用来区分"源只有 30fps"与"产品掉帧"。

用法：D2S\\python3\\python.exe -I tools\\dda_rate_probe.py [秒数] [输出序号]
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time

ROOT = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
sys.path.insert(0, ROOT + "/D2S")

import numpy as np  # noqa: E402

from gpu.dxgi_dup import (ComPtr, DXGI_OUTDUPL_FRAME_INFO,  # noqa: E402
                          DxgiPointerSource)


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    mon = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    src = DxgiPointerSource(mon)
    info = DXGI_OUTDUPL_FRAME_INFO()
    res = ctypes.c_void_p()
    freq = ctypes.c_longlong()
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
    pts, accs, acc0 = [], [], 0
    t0 = time.perf_counter()
    last_pt = 0
    while time.perf_counter() - t0 < secs:
        hr = src._dup.call(
            8, ctypes.c_long,
            (wt.UINT, ctypes.POINTER(DXGI_OUTDUPL_FRAME_INFO),
             ctypes.POINTER(ctypes.c_void_p)),
            16, ctypes.byref(info), ctypes.byref(res))
        if hr == 0:
            try:
                src._dup.call(14, ctypes.c_long, ())
                if res.value:
                    try:
                        ComPtr(res.value).release()
                    except Exception:
                        pass
                if info.AccumulatedFrames:
                    accs.append(int(info.AccumulatedFrames))
                    pt = int(info.LastPresentTime)
                    if pt and last_pt and pt != last_pt:
                        pts.append((pt - last_pt) * 1000.0 / freq.value)
                    if pt:
                        last_pt = pt
                else:
                    acc0 += 1
            except Exception as e:
                print("err", e)
                break
        res = ctypes.c_void_p()
    src.close()
    el = time.perf_counter() - t0
    print(f"new={len(accs)} in {el:.1f}s -> {len(accs) / el:.1f} fps  "
          f"acc0={acc0}  AF={ {k: accs.count(k) for k in sorted(set(accs))} }")
    if pts:
        a = np.array(pts)
        print(f"present interval ms: mean={a.mean():.2f} "
              f"median={np.median(a):.2f} p10={np.percentile(a, 10):.2f} "
              f"p90={np.percentile(a, 90):.2f} "
              f"<=20ms={int((a <= 20).sum())} >20ms={int((a > 20).sum())}")


if __name__ == "__main__":
    main()
