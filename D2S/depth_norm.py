# -*- coding: utf-8 -*-
"""深度归一化块：逐帧分位统计 → [0,1] 阈值域。

产品语义：归一化域 = P2/P98 分位（2026-11-21 用户定案），逐帧现算不做时间平滑——
慢跟随试过，场景切换时区间滞后，观感更差。分位施药分两处：GPU 侧由
depth_post.DepthPost.spatial_chain 第一步（normalize pass 的 lo/hi uniforms）
执行；CPU 侧 z_norm() 供诊断/离线副本。两侧用同一对 (lo, hi)。
中位数喂自动收敛面（P50 直取，main.auto_convergence_update）。
"""
import numpy as np

# 百分位下/上沿（%）。::4 子采样：224²→3136 样本，分位稳定且 ~16× 省时。
# 2026-11-21 用户定案：P2/P98（双端各裁 2%，映射带 [0.02, 0.98]）
DEPTH_PCT_LO = 2.0
DEPTH_PCT_HI = 98.0


def percentile_stats(raw):
    """raw（推理回读 numpy 副本）→ (lo, hi, med)。"""
    sub = raw[::4, ::4]
    return (float(np.percentile(sub, DEPTH_PCT_LO)),
            float(np.percentile(sub, DEPTH_PCT_HI)),
            float(np.median(sub)))


# 切阀（2026-11-21 改 CPU 判据）：z_t vs z_{t-1}（归一化 pre-AA 域，::4）——
# 首版用 GPU 1x1 残差 pass（cur vs 状态），4 字节读回本身 ~0，但把整条
# GL+DML 流水线从异步拽成同步（跨 API fence 等待 ~7ms/帧，fps 60→47 实锤）
# ——改在推理回读 numpy 上算，零新增读回。GPU(main) 与 CPU(depth) 同函数
# 同域，parity 逐位。语义：输入对输入（非输入对状态）——真切换两者等价，
# 仅"切换后第二帧"这类陈旧态不补枪（视觉无差）。
CUT_RES = 0.35
CUT_AREA = 0.20


def z_sub(raw, lo, hi):
    """::4 归一化子样本（切阀/诊断用，与 percentile_stats 同采样域）。"""
    return np.clip((raw[::4, ::4] - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def cut_hit(z, prev_z):
    """|z_t − z_{t−1}| > CUT_RES 的面积占比 > CUT_AREA。prev None=首帧不切。"""
    if prev_z is None or z.shape != prev_z.shape:
        return False
    return bool(np.mean(np.abs(z - prev_z) > CUT_RES) > CUT_AREA)


def near_peak_median(d, bins=64, min_mass=0.05):
    """双峰深度分布的近景峰中位——收敛面锚点（2026-11-20 用户提案）。

    病灶：全局 P50 在双峰内容上落在两峰谷底，前景占比一过半就翻侧，
    收敛面大幅摆动。Otsu 分裂出近景类（近=高值），取其中位 = 主体贴
    屏幕平面、背景后撤，峰存在期间稳定。单峰（近类质量 <5% 或 >95%）
    自动回退全局 P50——桌面/全近/全远内容行为与从前一致。
    D2S_CONV_PEAK=0 关闭（回纯 P50）。
    """
    import os as _os
    v = np.asarray(d, np.float32).ravel()
    sub = v[::4]
    if _os.environ.get("D2S_CONV_PEAK", "1") == "0":
        return float(np.median(sub))
    h, edges = np.histogram(sub, bins=bins, range=(0.0, 1.0))
    p = h / max(1, h.sum())
    centers = (edges[1:] + edges[:-1]) * 0.5
    c = np.cumsum(p)
    m = np.cumsum(p * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (m[-1] * c - m) ** 2 / np.maximum(c * (1 - c), 1e-9)
    k = int(np.nanargmax(sb[:-1]))
    near = sub[sub >= edges[k + 1]]
    if not (min_mass <= len(near) / max(1, len(sub)) <= 1 - min_mass):
        return float(np.median(sub))
    return float(np.median(near))


def z_norm(raw, lo, hi):
    """按 (lo, hi) 归一化到 [0,1]（CPU 侧；GPU 侧由 normalize pass 施行）。"""
    return np.clip((raw - lo) / max(1e-6, hi - lo), 0.0, 1.0)
