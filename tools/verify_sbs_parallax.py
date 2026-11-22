"""量化验证 SBS 是否真的有视差：估计左右半屏之间的位移量。

原理
----
SBS 左半屏 = 左眼图，右半屏 = 右眼图；两者相差一个水平位移（视差）。
把右半屏整体左/右平移 s 像素，与左半屏做差，**差异最小时的 s 就是视差**。

* 有 3D：最佳偏移明显偏离 0（Half-SBS 下约 = parallax_px/2）
* 没 3D：最佳偏移 ≈ 0，且差异曲线在 0 附近是谷底

这个判据不受"压缩误差/光标/抗锯齿"影响：那些只会抬高整体差异，
不会把谷底从非零位置挪回 0。

用法：D2S_DUMP_FRAME=1 启动 → 进 3D → 跑本脚本 [图片路径]
"""
import os
import sys

import cv2
import numpy as np


def shift_diff(left, right, s):
    """把 right 平移 s 像素后与 left 的差异（只取有效重叠区）。"""
    if s > 0:
        a = left[:, s:]
        b = right[:, :-s]
    elif s < 0:
        a = left[:, :s]
        b = right[:, -s:]
    else:
        a, b = left, right
    if a.shape[1] < 32 or a.shape[0] < 32:
        return None
    d = np.abs(a - b).mean(axis=2)
    # 去掉边缘（插值/压缩噪声）
    return float(d[6:-6, 6:-6].mean())


def estimate_parallax(path, max_shift=48):
    img = cv2.imread(path)
    if img is None:
        print(f"读不到 {path}（先用 D2S_DUMP_FRAME=1 导一帧）")
        return None
    h, w = img.shape[:2]
    half = w // 2
    left = img[:, :half]
    right = img[:, half:half * 2]
    print(f"帧 {w}x{h} → 每半屏 {left.shape[1]}x{left.shape[0]}")

    # 相位相关：对整幅图的位移最敏感，不受"大片平坦区域"影响（均值差法在
    # 文字/桌面这种低梯度画面上会误判，实测 40px 时 +23、24px 时只有 +4）。
    lg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(lg, rg)
    print(f"\n相位相关：位移 dx={dx:+.2f}px dy={dy:+.2f}px  置信度={resp:.3f}")
    print(f"（Half-SBS 下视差 px 的一半应 ≈ |dx|）")

    # 再用均值差法扫一遍，作为交叉验证
    curve = {}
    for s in range(-max_shift, max_shift + 1):
        v = shift_diff(left.astype(np.float32), right.astype(np.float32), s)
        if v is not None:
            curve[s] = v
    best_s = min(curve, key=curve.get) if curve else 0
    print(f"均值差法最佳位移 = {best_s:+d}px")

    est = abs(dx) if resp > 0.05 else abs(best_s)
    if est >= 3:
        print(f"结论：有视差，约 {est:.1f} px")
    else:
        print("结论：疑似【没有视差】")
    return dx, best_s



if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["TEMP"], "d2s_frame.png")
    estimate_parallax(p)
