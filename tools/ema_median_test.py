# -*- coding: utf-8 -*-
"""时域滤波器对比：纯 EMA vs 时域中值（MC 停用后的选型依据，2026-11-20）。

用户原始疑问的定量回答：单帧离群 0.5→0.6 后 EMA 输出 0.52 且拖 ~5 帧
（幅值换时长的数学必然）——中值滤波对 ≤⌊W/2⌋ 连续帧离群【完全免疫】，
代价是固定 W/2 帧群延迟与 W 张纹理的 GPU 环。

场景（全部同种子可复现；深度场=两电平结构块，边在场）：
  S1 静止+噪声       稳态抑噪（MAE）与帧间抖动
  S2 单帧尖峰        用户原案：+0.1 全场一帧；峰值残余/恢复帧数
  S3 双帧尖峰        中值 W=3 的免疫边界
  S4 场景切换        阶跃：达 90% 帧数/过渡期 MAE
  S5 斜坡跟踪        每帧 +0.01 的稳态滞后
  S6 重尾噪声        10% 像素 ±0.3 离群（鲁棒性）

滤波器：EMA τ=80ms(α=0.188)/τ=50ms(α=0.307)；中值 W=3/W=5。
用法：D2S\\python3\\python.exe -I tools\\ema_median_test.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

SIZE = 224
NOISE = 0.05
N = 60                     # 每场景帧数
ALPHA = {"EMA τ80 (N4.8)": 1 - 0.37 ** (1 / 4.8),
         "EMA τ50 (N3.0)": 1 - 0.37 ** (1 / 3.0)}
MED = {"中值 W3": 3, "中值 W5": 5}


def base_field():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    d = np.full((SIZE, SIZE), 0.35, np.float32)
    d[((yy - 112) ** 2 + (xx - 112) ** 2) <= 60 ** 2] = 0.7     # 前景圆
    d[((yy - 112) ** 2 + (xx - 180) ** 2) <= 24 ** 2] = 0.55    # 次物体
    return d


def run_filter(name, seq):
    """按滤波器定义逐帧输出（中值=滑窗逐像素，边界帧用可得样本）。"""
    if name in ALPHA:
        a, out = ALPHA[name], []
        prev = None
        for z in seq:
            prev = z.copy() if prev is None else a * z + (1 - a) * prev
            out.append(prev.copy())
        return out
    w = MED[name]
    out = []
    buf = []
    for i, z in enumerate(seq):
        buf.append(z)
        if len(buf) > w:
            buf.pop(0)
        out.append(np.median(np.stack(buf), axis=0).astype(np.float32))
    return out


def measure(clean, outs, steady=20):
    e = np.stack([np.abs(o - clean) for o in outs])
    jit = np.mean([np.abs(outs[i] - outs[i - 1]) for i in range(1, len(outs))])
    return float(e[steady:].mean()), float(jit)


def scenario(name, build):
    print(f"\n== {name} ==")
    print(f"{'滤波器':14s} {'稳态MAE':>9s} {'帧间抖动':>9s}  {'场景特定指标'}")
    seq, clean, extra = build()
    for fname in list(ALPHA) + list(MED):
        outs = run_filter(fname, seq)
        mae, jit = measure(clean[-1] if isinstance(clean, list) else clean,
                           outs, steady=20 if len(outs) >= 40 else 5)
        line = f"{fname:14s} {mae:9.4f} {jit:9.4f}"
        if extra:
            line += "  " + extra(fname, outs)
        print(line)


def main():
    rng = np.random.default_rng(3)
    B = base_field()

    def noised(field, r):
        return np.clip(field + r.normal(0, NOISE, field.shape).astype(np.float32),
                       0, 1)

    # S1 静止+噪声
    scenario("S1 静止+噪声（σ=0.05）",
             lambda: ([noised(B, rng) for _ in range(N)], B, None))

    # S2/S3 尖峰：第 30 帧（S3：30-31 帧）全场 +0.1 估计误差
    def spike(tail):
        def build():
            seq = [noised(B, rng) for _ in range(N)]
            clean = B
            for k in range(tail):
                seq[30 + k] = np.clip(seq[30 + k] + 0.1, 0, 1)

            def extra(fname, outs):
                e = np.array([float(np.mean(o - clean)) for o in outs])
                rec = next((i for i in range(32, N)
                            if abs(e[i]) < 2 * NOISE / np.sqrt(SIZE * SIZE) + 0.002
                            and all(abs(e[j]) < 0.01 for j in range(i, min(i + 3, N)))),
                           N)
                return f"尖峰后瞬时偏差 {e[31]:+.3f} → 恢复(<0.01) 第 {rec-30} 帧"
            return seq, clean, extra
        return build

    scenario("S2 单帧尖峰（+0.1，用户原案）", spike(1))
    scenario("S3 双帧尖峰（+0.1×2，W3 免疫边界）", spike(2))

    # S4 切换：第 30 帧整场换结构（圆挪位）
    B2 = base_field()[::-1, ::-1].copy()

    def cut_build():
        seq = [noised(B, rng) if i < 30 else noised(B2, rng) for i in range(N)]

        def extra(fname, outs):
            t90 = next((i for i in range(30, N)
                        if float(np.mean(np.abs(outs[i] - B2))) <
                        0.1 * float(np.mean(np.abs(B - B2)))), N)
            tr = float(np.mean([np.mean(np.abs(outs[i] - (B2 if i >= 30 else B)))
                                for i in range(30, 36)]))
            return f"达 90% 需 {t90-30} 帧；切换后 6 帧过渡 MAE {tr:.3f}"
        return seq, [B] * 30 + [B2] * 30, extra

    scenario("S4 场景切换（第 30 帧整场换）", cut_build)

    # S5 斜坡：全场每帧 +0.01（深度缓变，如镜头缓慢推进）
    def ramp_build():
        seq = [noised(np.clip(B + 0.01 * i, 0, 1), rng) for i in range(N)]
        clean = [np.clip(B + 0.01 * i, 0, 1) for i in range(N)]

        def extra(fname, outs):
            lag = float(np.mean([np.mean(outs[i] - clean[i])
                                 for i in range(20, N)]))
            return f"稳态滞后 {lag:+.4f}（=每帧增量×滞后帧数）"
        return seq, clean, extra

    scenario("S5 斜坡跟踪（+0.01/帧）", ramp_build)

    # S6 重尾噪声：10% 像素 ±0.3
    def heavy(field, r):
        z = field + r.normal(0, NOISE * 0.5, field.shape).astype(np.float32)
        m = r.random(field.shape) < 0.10
        z[m] += r.choice([-0.3, 0.3], m.sum())
        return np.clip(z, 0, 1)

    scenario("S6 重尾噪声（10% 像素 ±0.3）",
             lambda: ([heavy(B, rng) for _ in range(N)], B, None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
