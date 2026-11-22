"""在真实快速运动画面上，比较各种时域/空域滤波对 MAE 的影响。

参照（伪真值）：把首帧深度按已知变换（平移20px + 旋转1°/帧）warp 过去。
注意：伪真值本身来自 raw[0]，含噪声；绝对值只用于横向比较。

同时也给出旋转合成场景（有真实真值）的 MAE，避免"平滑深度场"单一场景误导。

用法: python tools/filter_mae_bench.py
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from filter_bench import motion_reference           # noqa: E402
from rotation_lag_test import make_sequence, add_noise, ema as ema_rot  # noqa: E402


# ---------------- 滤波器 ----------------
def ema(seq, n):
    a = 1.0 - 0.37 ** (1.0 / n)
    out = seq.copy()
    for i in range(1, len(out)):
        out[i] = a * seq[i] + (1 - a) * out[i - 1]
    return out


def holt(seq, n, br=0.3):
    a = 1.0 - 0.37 ** (1.0 / n)
    b = max(0.02, a * br)
    lvl = seq[0].copy(); trd = np.zeros_like(lvl); out = np.empty_like(seq); out[0] = lvl
    for i in range(1, len(seq)):
        pred = lvl + trd
        new_lvl = a * seq[i] + (1 - a) * pred
        trd = b * (new_lvl - lvl) + (1 - b) * trd
        lvl = new_lvl; out[i] = lvl
    return out


def med(seq, k=3):
    out = seq.copy()
    half = k // 2
    for i in range(1, len(seq)):
        out[i] = np.median(seq[max(0, i - half):i + 1], axis=0)
    return out


def spatial_med(seq, k=3):
    return np.stack([cv2.medianBlur(s.astype(np.float32), k) for s in seq])


def adaptive_ema(seq, n_static=20, n_move=1, thr=0.05):
    """逐像素自适应：变化小→强平滑；变化大→直接跟。"""
    a_s = 1.0 - 0.37 ** (1.0 / n_static)
    a_m = 1.0 - 0.37 ** (1.0 / n_move)
    out = seq.copy()
    for i in range(1, len(seq)):
        diff = np.abs(seq[i] - out[i - 1])
        t = np.clip(diff / max(1e-6, thr), 0, 1)
        a = a_s * (1 - t) + a_m * t
        out[i] = a * seq[i] + (1 - a) * out[i - 1]
    return out


def edge_aware(seq, n=10, edge_boost=4.0):
    """深度边缘处少平滑（保边），平坦处强平滑。"""
    a = 1.0 - 0.37 ** (1.0 / n)
    out = seq.copy()
    for i in range(1, len(seq)):
        gy, gx = np.gradient(seq[i])
        g = np.hypot(gx, gy)
        w = 1.0 / (1.0 + edge_boost * g * 50.0)      # 边缘→0
        ai = a + (0.95 - a) * (1 - w)
        out[i] = ai * seq[i] + (1 - ai) * out[i - 1]
    return out


def kalman1d(seq, q=1e-4, r=4e-2):
    """逐像素 1D 卡尔曼（常速模型）：噪声大时增益小，趋势持续时增益自动变大。"""
    out = np.empty_like(seq)
    x = seq[0].copy(); v = np.zeros_like(x)
    P = np.eye(2, dtype=np.float32) * 0.01
    out[0] = x
    for i in range(1, len(seq)):
        xp = x + v
        P00 = P[0, 0]; P01 = P[0, 1]; P10 = P[1, 0]; P11 = P[1, 1]
        # 预测协方差（F=[[1,1],[0,1]]）
        p00 = P00 + P01 + P10 + P11 + q
        p01 = P01 + P11
        p10 = P10 + P11
        p11 = P11 + q
        # 更新
        S = p00 + r
        k0 = p00 / S; k1 = p10 / S
        innov = seq[i] - xp
        x = xp + k0 * innov
        v = v + k1 * innov
        out[i] = x
        P[0, 0] = (1 - k0) * p00
        P[0, 1] = (1 - k0) * p01
        P[1, 0] = p10 - k1 * p00
        P[1, 1] = p11 - k1 * p01
    return out


def trimmed_mean(seq, k=4):
    out = seq.copy()
    for i in range(1, len(seq)):
        win = seq[max(0, i - k + 1):i + 1]
        srt = np.sort(win, axis=0)
        lo = 1 if len(win) >= 3 else 0
        hi = len(win) - 1 if len(win) >= 3 else len(win)
        out[i] = srt[lo:hi].mean(axis=0)
    return out


def blend_raw_med(seq, w=0.5):
    m = med(seq, 3)
    return (1 - w) * m + w * seq


CANDIDATES = [
    ("raw", lambda s: s.copy()),
    ("EMA N=3", lambda s: ema(s, 3)),
    ("EMA N=5", lambda s: ema(s, 5)),
    ("EMA N=8", lambda s: ema(s, 8)),
    ("EMA N=10", lambda s: ema(s, 10)),
    ("EMA N=16", lambda s: ema(s, 16)),
    ("EMA N=20", lambda s: ema(s, 20)),
    ("Holt N=5", lambda s: holt(s, 5)),
    ("Holt N=10", lambda s: holt(s, 10)),
    ("Holt N=20", lambda s: holt(s, 20)),
    ("时域中值3", lambda s: med(s, 3)),
    ("时域中值5", lambda s: med(s, 5)),
    ("截尾均值 K=4", lambda s: trimmed_mean(s, 4)),
    ("中值3+EMA5", lambda s: ema(med(s, 3), 5)),
    ("中值3+EMA10", lambda s: ema(med(s, 3), 10)),
    ("raw/中值3 混合", lambda s: blend_raw_med(s, 0.5)),
    ("自适应 EMA(20/1)", lambda s: adaptive_ema(s)),
    ("边缘感知 EMA10", lambda s: edge_aware(s, 10)),
    ("卡尔曼 1D", lambda s: kalman1d(s)),
    ("空间中值3", lambda s: spatial_med(s, 3)),
]


def mae_vs(out, ref):
    return float(np.mean([np.mean(np.abs(out[t] - ref[t])) for t in range(len(out))]))


def load_motion():
    z = np.load(os.path.join(ROOT, "dataset", "depth_seq_motion.npz"))
    raw = z["depths"].astype(np.float32)
    seq = np.empty_like(raw)
    for i, d in enumerate(raw):
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        seq[i] = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    ref = motion_reference(seq, step=20, deg=1.0, size=seq.shape[2])
    return seq, ref


def main():
    seq, ref = load_motion()
    base = mae_vs(seq, ref)
    print("== 真实快速运动画面（伪真值参照）==")
    print(f"{'方案':18s} {'MAE':>8s} {'相对raw':>9s}")
    rows = []
    for name, fn in CANDIDATES:
        out = fn(seq)
        m = mae_vs(out, ref)
        rows.append((name, m))
        print(f"{name:18s} {m:8.4f} {100*(1-m/base):8.1f}%")
    best = min(rows, key=lambda r: r[1])
    print(f"\n最佳: {best[0]}  MAE={best[1]:.4f}  相对raw {100*(1-best[1]/base):.1f}%")

    # 交叉验证：合成旋转场景（有真值）
    truth = make_sequence(40, 3.0)
    inp = add_noise(truth, 0.06)
    b2 = mae_vs(inp, truth)
    print(f"\n== 合成旋转场景（真实真值）==")
    print(f"{'方案':18s} {'MAE':>8s} {'相对raw':>9s}")
    for name, fn in CANDIDATES:
        out = fn(inp)
        m = mae_vs(out, truth)
        print(f"{name:18s} {m:8.4f} {100*(1-m/b2):8.1f}%")


if __name__ == "__main__":
    main()
