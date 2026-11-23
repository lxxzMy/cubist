# -*- coding: utf-8 -*-
"""V2 门禁：GPU 判据链 vs CPU 对账（分位/Otsu 锚/RGB 切变/z 乒乓）。

  S1 GPU 分位 lo/hi vs depth_norm.percentile_stats（::4 同域）
  S2 GPU Otsu 近峰中位 vs near_peak_median（归一化域）
  S6/S7/S8 归一化范围（平滑/棘轮/口径对比）
  S10 RGB 场景切换（同色不切 / 换色切；深度判据已删除）
  S12 切镜冷却按实际时间；S13 自适应阈值（只收 ≤ 下限的帧、封顶、切镜后清窗）
  S16 防锁死：同一对画面来回切，阈值不被自己抬高、过冷却就判切
用法：D2S\\python3\\python.exe -I tools\\v2_stats_smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402

import depth_norm  # noqa: E402
from gpu import adapter as _ad  # noqa: E402
from gpu import d3drender as dr  # noqa: E402
from gpu.d3d_stats import (CUT_RGB, DEPTH_RANGE_LO,  # noqa: E402
                           DEPTH_RANGE_SPAN, D3DStats)

RES = 224
SUB = (RES + 3) // 4
LO, SPAN = DEPTH_RANGE_LO, DEPTH_RANGE_SPAN


def upload_tex(ren, arr):
    """(h,w) float32 → R32F 纹理（一次性上传，返回 Tex）。尺寸取数组自身。"""
    h, w = arr.shape
    t = ren.upload_tex(w, h, dr.FMT_R32F, renderable=False)
    ren.upload(t, np.ascontiguousarray(arr, np.float32))
    ren.begin_frame()
    ren.copy_upload(t)
    ren.submit()
    ren.wait()
    return t


def main():
    fails = []
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    st = D3DStats(ren, RES)     # 直方图量程 = 每帧 raw 的 min/max（自动）
    # 切镜冷却用的假时钟（秒）：整个门禁里时间静止，只有 S12/S16 手动推进
    T = [0.0]
    st.clock = lambda: T[0]
    # 逻辑测试关掉冷却（生产默认 0.25s），冷却单独在 S12/S16 验证
    st.cut_min_scene_s = 0.0
    rng = np.random.default_rng(3)

    # 合成双峰 raw（前景 3.8~4.8 / 背景 1.2~2.0）
    base = np.zeros((RES, RES), np.float32)
    base[: RES // 2] = 4.3 + rng.normal(0, 0.15, (RES // 2, RES))
    base[RES // 2:] = 1.6 + rng.normal(0, 0.15, (RES - RES // 2, RES))
    raw = np.clip(base, 0.2, 5.4)

    raw_tex = upload_tex(ren, raw)

    def bgra_tex(rgb):
        arr = np.empty((RES, RES, 4), np.uint8)
        arr[..., :3] = rgb
        arr[..., 3] = 255
        buf = ren.make_upload_buffer(dr.row_pitch(RES, 4) * RES)
        ren.upload_to_buf(buf, np.ascontiguousarray(arr), bytes_per_px=4)
        t = ren.create_tex(RES, RES, dr.FMT_BGRA8, renderable=False)
        ren.begin_frame()
        t.state = dr.ST_COMMON
        ren.copy_buf_to_tex(buf, t)
        ren.submit(); ren.wait()
        return t

    cA = np.zeros((RES, RES, 3), np.uint8)
    cA[:, : RES // 2] = (200, 40, 40)
    cA[:, RES // 2:] = (40, 40, 200)
    cB = np.zeros((RES, RES, 3), np.uint8)
    cB[: RES // 2] = (40, 200, 40)
    cB[RES // 2:] = (200, 200, 40)
    # cC 与 cA 共享上半（≈50% 概率质量）→ TV≈0.5，用于自适应阈值测试
    cC = cA.copy()
    cC[RES // 2:] = (40, 200, 40)
    tA, tB, tC = bgra_tex(cA), bgra_tex(cB), bgra_tex(cC)
    # cD = cA 的顶部 36% 换成 cB 的颜色 → 与 cA 的色差≈0.36（下限以内的"阵发"波动）
    cD = cA.copy()
    _rows = int(RES * 0.36)
    cD[:_rows] = cB[:_rows]
    tD = bgra_tex(cD)

    ren.begin_frame()
    st.record(raw_tex, cut=False, alpha_range=1.0, range_mode="lerp")
    ren.submit(); ren.wait()
    lo_g, hi_g, anchor_g, _, p50_g, _ad_g = st.readback()

    lo_c, hi_c, _med_c = depth_norm.percentile_stats(raw)
    print(f"[S1] lo GPU={lo_g:.4f} CPU={lo_c:.4f}  hi GPU={hi_g:.4f} "
          f"CPU={hi_c:.4f}", flush=True)
    if abs(lo_g - lo_c) > 0.03:
        fails.append(f"S1 lo {lo_g:.4f} vs {lo_c:.4f}")
    if abs(hi_g - hi_c) > 0.03:
        fails.append(f"S1 hi {hi_g:.4f} vs {hi_c:.4f}")

    # Otsu 近峰中位（归一化域对账）：归一化域 = 本帧量程 [min, max]
    # （::4 域，自动取；GPU 的 minmax pass 与之同域）
    _sub = raw[::4, ::4]
    r_mn, r_span = float(_sub.min()), float(_sub.max() - _sub.min())
    anchor_raw = r_mn + anchor_g * r_span
    anchor_cpu = depth_norm.near_peak_median((raw - r_mn) / r_span)
    anchor_cpu_raw = r_mn + anchor_cpu * r_span
    print(f"[S2] 锚 GPU(raw)={anchor_raw:.4f} CPU(raw)={anchor_cpu_raw:.4f}",
          flush=True)
    if abs(anchor_raw - anchor_cpu_raw) > 0.10:
        fails.append(f"S2 锚 {anchor_raw:.3f} vs {anchor_cpu_raw:.3f}")

    # S2b 收敛面 P50（归一化域）vs CPU 中位
    z_cpu = depth_norm.z_sub(raw, lo_c, hi_c)
    p50_cpu = float(np.median(z_cpu))
    p50_cpu_scaled = DEPTH_RANGE_LO + DEPTH_RANGE_SPAN * p50_cpu
    print(f"[S2b] P50 GPU={p50_g:.4f} CPU(映射 0.02-0.98)="
          f"{p50_cpu_scaled:.4f}", flush=True)
    if abs(p50_g - p50_cpu_scaled) > 0.02:
        fails.append(f"S2b P50 {p50_g:.4f} vs {p50_cpu_scaled:.4f}")

    # S6 归一化范围的时域平滑（物体移出场景场景）：
    #   A 帧 snap；B 帧把前景整块换成背景（P99.5 突降）→ 平滑只走一半；
    #   C 帧前帧 cut>阈 → 直接快照
    raw_b = raw.copy()
    raw_b[: RES // 2] = np.clip(1.6 + rng.normal(0, 0.15, (RES // 2, RES)),
                                0.2, 5.4)
    rawb_tex = upload_tex(ren, raw_b)
    ren.begin_frame()
    st.record(raw_tex, cut=False, alpha_range=1.0, range_mode="lerp")
    ren.submit(); ren.wait()
    r_a = st.readback()
    lo_a, hi_a = r_a[0], r_a[1]
    ren.begin_frame()
    st.record(rawb_tex, cut=False, alpha_range=0.5, range_mode="lerp")
    ren.submit(); ren.wait()
    r_b = st.readback()
    exp_hi = hi_a + 0.5 * (st.hi_raw - hi_a)   # 收缩走 50%
    exp_lo = lo_a + 0.5 * (st.lo_raw - lo_a)
    print(f"[S6] 平滑 hi: prev={hi_a:.3f} raw={st.hi_raw:.3f} "
          f"got={r_b[1]:.3f} exp={exp_hi:.3f} | lo: {r_b[0]:.3f}/"
          f"{exp_lo:.3f}", flush=True)
    if abs(r_b[1] - exp_hi) > 0.02 or abs(r_b[0] - exp_lo) > 0.02:
        fails.append(f"S6 平滑 {r_b[1]:.3f}/{exp_hi:.3f}")
    # C 帧：颜色切换触发 cut（纯 RGB 判据）→ 同帧 QUANT 快照；
    # 下一帧同色（不切）+ alpha=0 应保持 cut 帧的范围（不再有隔帧锁存）
    ren.begin_frame()
    st.record(raw_tex, cut=True, alpha_range=1.0, range_mode="lerp",
              color_tex=tA, color_size=(RES, RES))
    ren.submit(); ren.wait()
    st.readback()
    ren.begin_frame()
    st.record(rawb_tex, cut=True, alpha_range=0.5, range_mode="lerp",
              color_tex=tB, color_size=(RES, RES))
    ren.submit(); ren.wait()
    _rc = st.readback()
    print(f"[S6c-pre] 换色 cut RGB TV={st.rgb_tv:.3f} flag={st.cut_flag:.0f} "
          f"hi={_rc[1]:.3f} raw={st.hi_raw:.3f}", flush=True)
    if st.cut_flag < 0.5:
        fails.append(f"S6c 换色未触发 cut tv={st.rgb_tv:.3f}")
    if abs(_rc[1] - st.hi_raw) > 0.02:
        fails.append(f"S6c cut 帧未快照 {_rc[1]:.3f} vs {st.hi_raw:.3f}")
    ren.begin_frame()
    st.record(raw_tex, cut=False, alpha_range=0.0, range_mode="lerp",
              color_tex=tB, color_size=(RES, RES))
    ren.submit(); ren.wait()
    r_c = st.readback()
    print(f"[S6c] cut 后 alpha=0 保持 hi={r_c[1]:.3f}"
          f"（cut 帧 {_rc[1]:.3f}，同色 flag={st.cut_flag:.0f}）", flush=True)
    if st.cut_flag != 0.0:
        fails.append(f"S6c 同色仍切 flag={st.cut_flag}")
    if abs(r_c[1] - _rc[1]) > 0.02:
        fails.append(f"S6c 保持 {r_c[1]:.3f} vs {_rc[1]:.3f}")

    # S7 棘轮范围：拿走最远物体，剩余物体 z 必须不变（范围只扩不缩）
    raw6 = np.zeros((RES, RES), np.float32)
    band = RES // 6
    vals = [0.3 + i * 4.0 / 5.0 for i in range(6)]
    for i in range(6):
        y0 = i * band
        y1 = RES if i == 5 else (i + 1) * band
        raw6[y0:y1] = vals[i]
    raw5 = raw6.copy()
    raw5[5 * band:] = vals[4]          # 拿走最远物体（值 4.3 那一条）
    t6 = upload_tex(ren, raw6)
    t5 = upload_tex(ren, raw5)
    ren.begin_frame()
    st.record(t6, cut=False, range_mode="ratchet")
    ren.submit(); ren.wait()
    r6 = st.readback()
    z4_a = LO + SPAN * ((vals[4] - r6[0]) / max(1e-6, r6[1] - r6[0]))
    ren.begin_frame()
    st.record(t5, cut=False, range_mode="ratchet")
    ren.submit(); ren.wait()
    r5 = st.readback()
    z4_b = LO + SPAN * ((vals[4] - r5[0]) / max(1e-6, r5[1] - r5[0]))
    print(f"[S7] 拿走最远物体：hi_s {r6[1]:.3f}→{r5[1]:.3f} "
          f"物体4 z {z4_a:.4f}→{z4_b:.4f}", flush=True)
    if abs(r5[1] - r6[1]) > 0.02 or abs(z4_b - z4_a) > 0.01:
        fails.append(f"S7 棘轮 {r5[1]:.3f}/{z4_b:.4f}")

    # S8 归一化口径对比（同三场景：尺度抖动 / 拿掉物体 / 加入物体）
    #   percentile = P2/P98→[0.02,0.98] 每帧快照（产品默认，无时域）
    #   decile     = P10/P90→[0.1,0.9] 快照，越界才截断
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32)
    ramp = (xx / RES) * 2.0 + 1.0
    bump = 1.6 * np.exp(-(((xx - 60) ** 2 + (yy - 60) ** 2) /
                          (2 * 28.0 ** 2)))
    base_c = (ramp + bump).astype(np.float32)
    bump2 = 1.4 * np.exp(-(((xx - 170) ** 2 + (yy - 150) ** 2) /
                           (2 * 26.0 ** 2)))
    add_c = (ramp + bump + bump2).astype(np.float32)
    rm_c = ramp.astype(np.float32)
    probe_vals = [float(ramp[10, 10]), float(ramp[100, 10]),
                  float(ramp[200, 10]), float(ramp[100, 200])]

    def suite(mode, band_lo, band_span):
        def step(raw_arr):
            t = upload_tex(ren, raw_arr)
            ren.begin_frame()
            st.record(t, cut=False, range_mode=mode)
            ren.submit(); ren.wait()
            return st.readback()

        def zs(rv, vals):
            return [band_lo + band_span * ((v - rv[0]) /
                                           max(1e-6, rv[1] - rv[0]))
                    for v in vals]

        step(base_c)         # 首帧重定标
        r = step(base_c)     # 建立基准
        z_ref = zs(r, probe_vals)
        rng3 = np.random.default_rng(31)
        wob = 0.0
        for k in range(10):
            a = float(rng3.uniform(0.7, 1.3))
            b = float(rng3.uniform(-0.3, 0.3))
            rv = step(base_c * a + b)
            zz = zs(rv, [v * a + b for v in probe_vals])
            wob = max(wob, max(abs(zz[j] - z_ref[j]) for j in range(4)))
        step(base_c)         # 抖动循环后先回基准（让仿射变化单独收敛）
        rv = step(rm_c)
        zz = zs(rv, probe_vals)
        d_rm = max(abs(zz[j] - z_ref[j]) for j in range(4))
        step(base_c)         # 回基准后再测"加入"
        rv = step(add_c)
        zz = zs(rv, probe_vals)
        d_add = max(abs(zz[j] - z_ref[j]) for j in range(4))
        print(f"[S8] {mode:10s}: 抖动 {wob:.4f} | 拿掉 {d_rm:.4f} | "
              f"加入 {d_add:.4f}", flush=True)
        return wob, d_rm, d_add

    t1 = suite("percentile", LO, SPAN)
    t2 = suite("decile", 0.1, 0.8)

    # near_q_anchor：GPU ANCHOR_CS 同算法（近侧质量分位，bin 内线性插值）
    def near_q_anchor(d, q):
        v = np.clip(np.asarray(d, np.float32).reshape(-1), 0.0, 1.0)
        bins = np.minimum((v * 255.0 + 0.5).astype(np.int32), 255)
        h = np.bincount(bins, minlength=256).astype(np.float64)
        total = h.sum()
        target = q * total
        acc = 0.0
        for b in range(255, -1, -1):
            c = h[b]
            if acc + c >= target:
                need = max(0.0, target - acc)
                frac = need / c if c > 0 else 0.0
                return float(np.clip((b + 0.5 - frac) / 255.0, 0.0, 1.0))
            acc += c
        return 1.0

    # S5 链接线：spatial_chain(stats) vs spatial_chain(CPU lo/hi) 深度输出对账
    # （稳态同输入、双方 cut=False：EMA 收敛后只剩 lo/hi 来源差异）
    from gpu.d3d_post import D3DDepthPost
    p1 = D3DDepthPost(ren, RES)
    p2 = D3DDepthPost(ren, RES)
    lo_c, hi_c, _ = depth_norm.percentile_stats(raw)
    for i in range(6):
        p1.upload_raw(raw)
        p1.spatial_chain(lo_c, hi_c, aa_texels=3.0, temporal="ema",
                         alpha=0.19, cut=False, stats=st)
        p2.upload_raw(raw)
        p2.spatial_chain(lo_c, hi_c, aa_texels=3.0, temporal="ema",
                         alpha=0.19, cut=False, stats=None)
    da = p1.readback_depth()
    db = DEPTH_RANGE_LO + DEPTH_RANGE_SPAN * p2.readback_depth()
    mae = float(np.mean(np.abs(da - db)))
    print(f"[S5] 链 stats vs CPU-lo/hi MAE={mae:.5f}", flush=True)
    # S2d 显示域收敛锚：最终深度的近侧主体峰（ANCHOR_CS）
    _ad_gpu = st.readback()[5]
    _ad_cpu = near_q_anchor(da, st.anchor_q)
    print(f"[S2d] 显示域锚 GPU={_ad_gpu:.4f} CPU={_ad_cpu:.4f}", flush=True)
    if abs(_ad_gpu - _ad_cpu) > 0.05:
        fails.append(f"S2d 显示域锚 {_ad_gpu:.4f} vs {_ad_cpu:.4f}")
    if mae > 0.02:
        fails.append(f"S5 链 MAE {mae:.4f}")

    # S9 纯平移稳定性：位置平移、内容刚性移动 → 同一内容的 z 必须不变
    # （复现"图像平移时深度剧烈波动"：同像素配对拟合在平移下稀释，
    #   合法区间内的稀释斜率逐帧复合 → 映射范围被压缩 → 泵动）
    rng9 = np.random.default_rng(91)
    noise9 = rng9.normal(0, 1, (RES, RES)).astype(np.float32)
    fy = np.fft.fftfreq(RES)[:, None]
    fx = np.fft.fftfreq(RES)[None, :]
    sig9 = 9.0
    h9 = np.exp(-2 * np.pi ** 2 * sig9 ** 2 * (fx ** 2 + fy ** 2))
    f9 = np.real(np.fft.ifft2(np.fft.fft2(noise9) * h9))
    f9 = (f9 - f9.mean()) / f9.std()
    tex0 = np.clip(2.6 + 0.9 * f9, 0.25, 5.3).astype(np.float32)
    PAN = 4                      # 每帧平移 4 texels（≈1.8% 宽度）
    probe_x0, probe_y = 30, 60
    rho = float(np.mean((f9 - f9.mean()) *
                        (np.roll(f9, -PAN, axis=1) - f9.mean())) /
                f9.var())
    print(f"[S9] 纹理自相关 ρ(Δ={PAN}px)={rho:.3f}", flush=True)

    def z_at(rv, v):
        return LO + SPAN * ((v - rv[0]) / max(1e-6, rv[1] - rv[0]))

    z_ref = None
    worst = (0, 0.0, 0.0, 0.0)
    for t in range(22):
        arr = np.roll(tex0, -t * PAN, axis=1)
        tt = upload_tex(ren, arr)
        ren.begin_frame()
        st.record(tt, cut=False, range_mode="percentile")
        ren.submit(); ren.wait()
        rv = st.readback()
        x_probe = (probe_x0 - t * PAN) % RES
        zz = z_at(rv, float(arr[probe_y, x_probe]))
        if t == 1:
            z_ref = zz
            w_ref = rv[1] - rv[0]
        if t >= 2:
            dev = abs(zz - z_ref)
            if dev > worst[1]:
                worst = (t, dev, rv[0], rv[1])
        if t in (1, 5, 10, 21):
            print(f"[S9] t={t:2d} lo={rv[0]:.3f} hi={rv[1]:.3f} "
                  f"z(内容)={zz:.4f}（基准 {z_ref:.4f}）", flush=True)
    print(f"[S9] 平移 {PAN}px/帧：最大内容 z 偏移 {worst[1]:.4f}"
          f" @t={worst[0]}（lo={worst[2]:.3f} hi={worst[3]:.3f}，"
          f"基准宽 {w_ref:.3f}）", flush=True)
    if worst[1] > 0.03:
        fails.append(f"S9 平移 z 偏移 {worst[1]:.3f}")

    # S9b 滑动窗口平移（内容从窗口一侧滑出、另一侧滑入，分布本身在变）
    # 每帧快照口径会跟随窗口内分布缓慢漂移（设计使然，非泵动）；
    # 这里只防失控：内容 z 偏移上限 0.10
    wide_w = RES + PAN * 40
    noiseb = rng9.normal(0, 1, (RES, wide_w)).astype(np.float32)
    fyb = np.fft.fftfreq(RES)[:, None]
    fxb = np.fft.fftfreq(wide_w)[None, :]
    hb = np.exp(-2 * np.pi ** 2 * sig9 ** 2 * (fxb ** 2 + fyb ** 2))
    fb = np.real(np.fft.ifft2(np.fft.fft2(noiseb) * hb))
    fb = (fb - fb.mean()) / fb.std()
    sceneb = np.clip(2.6 + 0.9 * fb, 0.25, 5.3).astype(np.float32)
    v_probe = float(sceneb[60, 30])
    zb_ref = None
    worst_b = (0, 0.0)
    for t in range(42):
        off = t * PAN
        arr = sceneb[:, off:off + RES]
        tt = upload_tex(ren, arr)
        ren.begin_frame()
        st.record(tt, cut=False, range_mode="percentile")
        ren.submit(); ren.wait()
        rv = st.readback()
        zz = z_at(rv, v_probe)
        if t == 1:
            zb_ref = zz
        if t >= 2 and abs(zz - zb_ref) > worst_b[1]:
            worst_b = (t, abs(zz - zb_ref))
        if t in (1, 10, 25, 41):
            print(f"[S9b] t={t:2d} lo={rv[0]:.3f} hi={rv[1]:.3f} "
                  f"z(内容)={zz:.4f}（基准 {zb_ref:.4f}）", flush=True)
    print(f"[S9b] 滑动窗口 {PAN}px/帧：最大内容 z 偏移 {worst_b[1]:.4f}"
          f" @t={worst_b[0]}", flush=True)
    if worst_b[1] > 0.10:
        fails.append(f"S9b 滑动窗口 z 偏移 {worst_b[1]:.3f}")

    # S10 RGB 场景切换（唯一判据）：同色 → 不切；换色 → 切
    ren.begin_frame()
    st.record(raw_tex, cut=True, color_tex=tA, color_size=(RES, RES))
    ren.submit(); ren.wait()
    st.readback()
    tv0, f0 = st.rgb_tv, st.cut_flag
    ren.begin_frame()
    st.record(raw_tex, cut=True, color_tex=tA, color_size=(RES, RES))
    ren.submit(); ren.wait()
    st.readback()
    tv_same, f_same = st.rgb_tv, st.cut_flag
    ren.begin_frame()
    st.record(raw_tex, cut=True, color_tex=tB, color_size=(RES, RES))
    ren.submit(); ren.wait()
    st.readback()
    tv_diff, f_diff = st.rgb_tv, st.cut_flag
    print(f"[S10] RGB cut TV: first={tv0:.3f} same={tv_same:.4f} "
          f"diff={tv_diff:.4f}（阈 {CUT_RGB}） 决策 first={f0:.0f} "
          f"same={f_same:.0f} diff={f_diff:.0f}", flush=True)
    if tv_same > 0.03 or f_same != 0.0:
        fails.append(f"S10 同色误切 tv={tv_same:.3f} f={f_same}")
    if tv_diff <= CUT_RGB or f_diff != 1.0:
        fails.append(f"S10 换色未切 tv={tv_diff:.3f} f={f_diff}")

    # S12 冷却按【实际时间】（2026-11-24；原按采集帧数——静止桌面几乎不出新帧，
    # 冷却在实际时间里会被拉长到几秒）：切后 0.20s 内换色不得切；满 0.25s 才允许。
    # 这里只出 3 帧就过了冷却（按帧数的旧规则需要 15 帧）。
    st.cut_k = 0.0            # 关自适应，隔离冷却逻辑（阈=floor）
    st.cut_min_scene_s = 0.25

    def _rec_color(tex, cut=True, dt=1.0 / 60.0):
        T[0] += dt
        ren.begin_frame()
        st.record(raw_tex, cut=cut, color_tex=tex, color_size=(RES, RES))
        ren.submit(); ren.wait()
        st.readback()

    _rec_color(tA)                         # 距上次切 0.017s → 不得切
    f_cool = st.cut_flag
    _rec_color(tB, dt=0.20 - 1.0 / 60.0)   # 累计 0.20s → 仍在冷却期内
    f_cool2 = st.cut_flag
    _rec_color(tA, dt=0.06)                # 累计 0.26s → 允许
    f_ok = st.cut_flag
    print(f"[S12] 冷却（按时间）：切后 0.017s flag={f_cool:.0f}（期望 0）｜"
          f"0.20s flag={f_cool2:.0f}（期望 0）｜0.26s flag={f_ok:.0f}"
          f"（期望 1）", flush=True)
    if f_cool != 0.0 or f_cool2 != 0.0:
        fails.append(f"S12 冷却期内仍切 {f_cool:.0f}/{f_cool2:.0f}")
    if f_ok != 1.0:
        fails.append(f"S12 冷却后未切 flag={f_ok}")

    # S13 自适应阈值（2026-11-24 修"锁死"之后的语义）：
    #   ① 安静 40 帧 → thr = 下限
    #   ② 超过下限的跳变不入窗：0.5 级来回跳，每帧都判切，thr 仍是下限
    #      （旧实现把这些跳变收进窗口 → thr 被顶到 0.775、反过来压住切）
    #   ③ 下限以内的阵发波动（0 / 0.36）把阈值抬高，但不超过封顶
    #   ④ 切镜后清窗：切镜的下一帧阈值立即回到下限
    st.cut_k = 3.0
    st.cut_min_scene_s = 0.0
    for _ in range(40):
        _rec_color(tC)
    thr_quiet = st.cut_thr
    f_jump = []
    for i in range(20):
        _rec_color(tC if i % 2 == 0 else tA)
        f_jump.append(int(st.cut_flag))
    thr_busy, tv_busy, f_busy = st.cut_thr, st.rgb_tv, st.cut_flag
    print(f"[S13] 安静 thr={thr_quiet:.3f}（floor {CUT_RGB}）｜0.5 级来回跳 "
          f"TV={tv_busy:.3f} thr={thr_busy:.3f} flag={f_busy:.0f}｜"
          f"后 19 帧判切 {sum(f_jump[1:])}/19", flush=True)
    if abs(thr_quiet - CUT_RGB) > 0.01:
        fails.append(f"S13 安静期阈值 {thr_quiet:.3f} != floor")
    if abs(thr_busy - CUT_RGB) > 0.01:
        fails.append(f"S13 阈值被切镜级跳变抬高 {thr_busy:.3f}（应留在 floor）")
    if f_busy != 1.0 or sum(f_jump[1:]) != 19:
        fails.append(f"S13 超下限的跳变未逐帧判切 {f_jump}")
    # ③ 阵发的下限内波动：A→D、D→A（0.36）、A→A（0）循环
    for i in range(24):
        _rec_color(tD if i % 3 != 2 else tA)
    thr_burst, tv_burst, f_burst = st.cut_thr, st.rgb_tv, st.cut_flag
    print(f"[S13] 阵发 0/0.36：TV={tv_burst:.3f} thr={thr_burst:.3f}"
          f"（封顶 {st.cut_adapt_max:.2f}）flag={f_burst:.0f}", flush=True)
    if f_burst != 0.0 or tv_burst > CUT_RGB:
        fails.append(f"S13 下限内波动被判切 tv={tv_burst:.3f}")
    if not (CUT_RGB < thr_burst <= st.cut_adapt_max + 1e-3):
        fails.append(f"S13 阵发后阈值 {thr_burst:.3f} 不在 "
                     f"({CUT_RGB}, {st.cut_adapt_max}]")
    # ④ 切镜后清窗
    _rec_color(tB)                         # 色差 1.0 → 切
    f_clear = st.cut_flag
    _rec_color(tB)                         # 同画面再来一帧：窗已清 → 阈值回下限
    thr_after = st.cut_thr
    print(f"[S13] 切镜清窗：切帧 flag={f_clear:.0f}，下一帧 thr={thr_after:.3f}"
          f"（期望 floor）", flush=True)
    if f_clear != 1.0 or abs(thr_after - CUT_RGB) > 1e-3:
        fails.append(f"S13 切镜后未清窗 flag={f_clear:.0f} thr={thr_after:.3f}")

    # S16 防锁死（2026-11-24 用户实测"同一对画面有时切有效、有时切无效"）：
    # 同一对画面来回切，间隔 0.06s（≈静止桌面：一次切换只出两三个采集帧）。
    # 旧实现每帧色差都入窗 → 阈值 0.50→0.71→…→1.76，第 2 次起永远不判切。
    # 现在：阈值不超过封顶；每逢过了冷却（0.25s）的那一次切换都判切 →
    # 每 5 次一切（0.06×5=0.30s ≥ 0.25s）。
    st.cut_min_scene_s = 0.25
    for _ in range(40):
        _rec_color(tA)                     # 回到 A 并离开上一个切镜的冷却期
    flags, thrs = [], []
    for i in range(12):
        _tex = tB if i % 2 == 0 else tA
        _rec_color(_tex, dt=0.06)          # 切换帧
        flags.append(int(st.cut_flag))
        thrs.append(st.cut_thr)
        _rec_color(_tex, dt=0.0)           # 同画面再出一个采集帧
    expect = [1, 0, 0, 0, 0] * 2 + [1, 0]
    print(f"[S16] 同一对画面来回切 12 次：判切 {flags}｜阈值最大 "
          f"{max(thrs):.3f}（封顶 {st.cut_adapt_max:.2f}）", flush=True)
    if flags != expect:
        fails.append(f"S16 判切序列 {flags} != {expect}")
    if max(thrs) > st.cut_adapt_max + 1e-3:
        fails.append(f"S16 阈值越顶 {max(thrs):.3f}")

    # S11 近侧主体峰锚：两个近景主体 + 远景 → 锚到更近的主体。
    # 锚 = 近侧 q=10% 质量分位（2026-11-22 定案，原 5%）：主体须占画面
    # >10%（80×80 ≈ 12.7%）锚才落在它身上。旧几何 60×60 只占 7.2%，凑不够
    # 10% 质量，锚会继续往远端伸到第二主体——那是 q=10% 的设计行为。
    def _anchor_of(arr):
        t = upload_tex(ren, arr)
        ren.begin_frame()
        st.anchor_from(t)
        ren.submit(); ren.wait()
        return st.readback()[5]

    d11 = np.full((RES, RES), 0.25, np.float32)
    d11[20:100, 20:100] = 0.80       # 主体 A（更近，12.7% 画面）
    d11[120:200, 140:220] = 0.60     # 主体 B（12.7% 画面）
    d11 = np.clip(d11 + rng.normal(0, 0.01, d11.shape), 0.0, 1.0)
    a11 = _anchor_of(d11.astype(np.float32))
    e11 = near_q_anchor(d11, st.anchor_q)
    d12 = np.full((RES, RES), 0.25, np.float32)
    d12[120:200, 140:220] = 0.60
    d12 = np.clip(d12 + rng.normal(0, 0.01, d12.shape), 0.0, 1.0)
    a12 = _anchor_of(d12.astype(np.float32))
    e12 = near_q_anchor(d12, st.anchor_q)
    print(f"[S11] 双主体锚 GPU={a11:.4f} CPU={e11:.4f} | "
          f"仅主体B GPU={a12:.4f} CPU={e12:.4f}", flush=True)
    if abs(a11 - e11) > 0.03 or abs(a12 - e12) > 0.03:
        fails.append(f"S11 近侧峰锚 {a11:.3f}/{e11:.3f} {a12:.3f}/{e12:.3f}")
    if abs(a11 - 0.80) > 0.04:
        fails.append(f"S11 锚未落在最近主体 {a11:.3f}")
    if abs(a12 - 0.60) > 0.04:
        fails.append(f"S11 锚未跟随第二主体 {a12:.3f}")

    # S14 近端饱和主体：主体顶到 z≈1.0（bin 254/255），远景大峰 0.60。
    # 旧扫描上界 i3=253 会漏掉近端峰 → 锚掉到远景（现场 0.99→0.57 的跳变）
    d14 = np.full((RES, RES), 0.60, np.float32)
    d14[40:140, 40:140] = 1.0      # 100×100 = 20% 像素
    d14 = np.clip(d14 + rng.normal(0, 0.004, d14.shape), 0.0, 1.0)
    a14 = _anchor_of(d14.astype(np.float32))
    e14 = near_q_anchor(d14, st.anchor_q)
    print(f"[S14] 近端饱和主体锚 GPU={a14:.4f} CPU={e14:.4f}"
          f"（期望≈1.0，不得掉到远景 0.60）", flush=True)
    if abs(a14 - e14) > 0.03:
        fails.append(f"S14 锚 GPU/CPU {a14:.3f}/{e14:.3f}")
    if a14 < 0.95:
        fails.append(f"S14 近端主体丢失 {a14:.3f}")

    # S15 锚稳定性：正常连续内容 + 轻微噪声/尺度抖动 → 帧间锚变化必须很小
    # （单一分位估计器：分布连续变化 ⇒ 锚只随内容平滑移动，不跳变）
    yy15, xx15 = np.mgrid[0:RES, 0:RES].astype(np.float32)
    base15 = np.full((RES, RES), 0.30, np.float32)
    base15[50:170, 60:180] = 0.86          # 主体 25.7% 像素
    base15 = np.clip(base15 + 0.06 * np.sin(xx15 / 25.0), 0.0, 1.0)
    prev15 = None
    dmax15 = 0.0
    for _ in range(20):
        arr = base15 * (1.0 + rng.normal(0, 0.004)) + rng.normal(0, 0.002)
        arr = np.clip(arr + rng.normal(0, 0.003, arr.shape), 0.0, 1.0)
        a15 = _anchor_of(arr.astype(np.float32))
        if prev15 is not None:
            dmax15 = max(dmax15, abs(a15 - prev15))
        prev15 = a15
    print(f"[S15] 连续内容：锚帧间最大变化 {dmax15:.4f}"
          f"（锚≈{prev15:.4f}）", flush=True)
    if dmax15 > 0.015:
        fails.append(f"S15 锚帧间抖动 {dmax15:.4f}")

    ren.destroy()
    if fails:
        print(f"[v2-smoke] FAIL: {fails}")
        return 1
    print("[v2-smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
