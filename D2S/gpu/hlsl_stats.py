# -*- coding: utf-8 -*-
"""GPU 判据链 HLSL（V2）：raw 深度 → 直方图 → 分位数/Otsu 锚；RGB 切变。

数据流（全部 compute，链 shader 直读 stats buffer）：
  hist_cs : raw(R32F Texture2D) → hist[256] uint（atomic）
  quant_cs: hist → stats[0]=lo(P2 值域) stats[1]=hi(P98)
            stats[2]=近峰中位（Otsu 分裂近类中位，收敛锚）
  rgb_hist_cs: 颜色 4096-bin(16³) 直方图 TV → stats[8]（遥测）/[9]（cut 决策）
               /[10]（自适应阈值）；thr = max(0.40, mean+3σ) 30 帧窗 + 最小场景长 15

stats buffer：uint[8]（StructuredBuffer/RWStructuredBuffer，stride 4）。
  [4] = P50（归一化域，本帧原始值；收敛面直取，无 EMA）
范围更新模式 P[2].w：1=percentile（默认，每帧快照 P2/P98→[0.02,0.98]）
                    2=ratchet 0=lerp（α 平滑，调试用）
**注意**：D3D12 runtime 拒收 raw（ByteAddressBuffer）视图——strided 0 的
UAV/SRV 会把设备置 invalid-call，故一律用 structured 视图。
值域：raw ∈ [raw_lo, raw_hi]（P[0].x/.y，d224 固定区间 0.20~5.40）。
cbuffer：P[0]=(raw_lo, raw_hi, full_res, cut_res_eff)，
         P[1]=(sub_res, sample_step, 0, range_alpha)，
         P[2]=(0, range_lo, range_span, range_mode: 2=棘轮 1=percentile 0=lerp)
         （与 depth_norm ::4 域对齐；扩张/收缩共用 range_alpha；
          归一化映射到 [range_lo, range_lo+range_span] = [0.02, 0.98]）
"""

# 清零：hist[256] + stats（每帧判据链开头）
ZERO_CS = """
RWStructuredBuffer<uint> h : register(u0);
RWStructuredBuffer<uint> s : register(u1);

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    for (uint k = tid.x; k < 256; k += 64) h[k] = 0;
    // 跨帧状态只保留：[0]/[1] 显示仿射、[27] cut 锁存
    if (tid.x < 32 &&
        !(tid.x == 0 || tid.x == 1 || tid.x == 27)) {
        s[tid.x] = 0;
    }
}
"""

# 直方图：::4 子采样（对齐 depth_norm.percentile_stats 的采样域），atomic 累加
HIST_CS = """
Texture2D<float> tex_raw : register(t0);
RWStructuredBuffer<uint> uav_out : register(u0);
cbuffer CB : register(b0) { float4 P[16]; };

[numthreads(16, 16, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint sub = (uint)P[1].x;
    // 纵向子采样数（P[1].z，非方形用；0 → 方形回退）
    uint sub_h = (P[1].z > 0.5) ? (uint)P[1].z : sub;
    uint step = (uint)P[1].y;
    if (tid.x >= sub || tid.y >= sub_h) return;
    float lo = P[0].x, hi = P[0].y;
    float v = tex_raw.Load(int3((int)(tid.x * step), (int)(tid.y * step), 0));
    float t = saturate((v - lo) / max(1e-6, hi - lo));
    uint bin = min((uint)(t * 255.0 + 0.5), 255u);
    uint prev = 0;
    InterlockedAdd(uav_out[bin], 1u, prev);
}
"""

# RGB 场景切换（唯一判据）：RGB(16³)=4096 bins 直方图 TV 距离。
# 依据（2026-11-22 用户调研的成熟经验）：
#   - 4 bit/通道 = 4096 bins 是论文常用分箱（TRECVID 系）；
#   - TV 经验域：同镜头运动 0.02~0.12、摇镜 0.12~0.30、硬切 0.35~0.75；
#   - 固定阈值必然在某类片子上失效 → thr = max(floor, mean + K·std)
#     （30 帧滑窗，K=3；样本 <8 时只用 floor）；
#   - 最小场景长防抖（PySceneDetect min_scene_len=15）。
# 采样：1/2 网格（112²，P[1].z=step）× 2×2 box 平均（抗混叠）；
# 统计上 12.5k 面积平均样本对 0.4 阈值的 TV 噪声 ~0.005，远低于判别余量。
# uav_tv 环形状态：[0]帧号 [1]上次切+1 [2]写指针 [3]样本数 [4..35]TV 环。
RGB_HIST_CS = """
Texture2D<float4> tex_color : register(t0);   // BGRA8（D3D 自动 .x=R）
RWStructuredBuffer<uint> uav_prev : register(u0);   // 上帧 4096-bin
RWStructuredBuffer<uint> uav_stats : register(u1);
RWStructuredBuffer<uint> uav_tv : register(u2);     // 自适应 TV 历史
cbuffer CB : register(b0) { float4 P[16]; };

groupshared uint g_hist[4096];
groupshared float g_abs[1024];

[numthreads(1024, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint t = gtid.x;
    for (uint i = t; i < 4096; i += 1024) g_hist[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint res = (uint)P[0].x;
    uint step = max(1u, (uint)P[1].z);       // 采样网格步长（2 = 112² 网格）
    uint sres = max(1u, res / step);
    uint cw = max(1u, (uint)P[0].y), ch = max(1u, (uint)P[0].z);
    uint total = sres * sres;
    float dx = max(0.5, (float)cw * (float)step / (float)res);
    float dy = max(0.5, (float)ch * (float)step / (float)res);
    for (uint idx = t; idx < total; idx += 1024) {
        uint x = idx % sres, y = idx / sres;
        // 2x2 box 平均（源像素足迹四个 1/4 点），抗混叠
        float fx = (float)(x * step) * (float)cw / (float)res;
        float fy = (float)(y * step) * (float)ch / (float)res;
        float4 c = 0.0;
        c += tex_color.Load(int3(min(cw - 1u, (uint)(fx + dx * 0.25)),
                                 min(ch - 1u, (uint)(fy + dy * 0.25)), 0));
        c += tex_color.Load(int3(min(cw - 1u, (uint)(fx + dx * 0.75)),
                                 min(ch - 1u, (uint)(fy + dy * 0.25)), 0));
        c += tex_color.Load(int3(min(cw - 1u, (uint)(fx + dx * 0.25)),
                                 min(ch - 1u, (uint)(fy + dy * 0.75)), 0));
        c += tex_color.Load(int3(min(cw - 1u, (uint)(fx + dx * 0.75)),
                                 min(ch - 1u, (uint)(fy + dy * 0.75)), 0));
        c *= 0.25;
        uint r = min((uint)(saturate(c.x) * 15.999), 15u);
        uint g = min((uint)(saturate(c.y) * 15.999), 15u);
        uint b = min((uint)(saturate(c.z) * 15.999), 15u);
        InterlockedAdd(g_hist[(r * 16u + g) * 16u + b], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    float sum_abs = 0.0;
    for (uint j = t; j < 4096u; j += 1024) {
        uint c2 = g_hist[j];
        uint p2 = uav_prev[j];
        sum_abs += (float)(c2 > p2 ? c2 - p2 : p2 - c2);
        uav_prev[j] = c2;                 // 留给下帧
    }
    g_abs[t] = sum_abs;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 512; s > 0; s >>= 1) {
        if (t < s) g_abs[t] += g_abs[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) {
        float tv = 0.5 * g_abs[0] / (float)max(1u, total);
        uint frame = uav_tv[0];
        uint lastp = uav_tv[1];               // 上次切帧号 +1（0=从未切）
        uint wi = uav_tv[2] % 32u;
        uint cnt = min(uav_tv[3], 32u);
        // 自适应阈值：只看【本帧之前】的窗口（本帧大跳变不得抬高自己的门槛）
        float thr = P[0].w;                   // 固定下限（硬切起点 0.40）
        if (cnt >= 8u) {
            float m = 0.0, sq = 0.0;
            for (uint k = 0; k < 32u; ++k) {
                if (k >= cnt) break;
                float v = asfloat(uav_tv[4u + k]);
                m += v; sq += v * v;
            }
            m /= (float)cnt;
            float var = max(0.0, sq / (float)cnt - m * m);
            thr = max(thr, m + P[1].y * sqrt(var));
        }
        bool cooldown = (lastp == 0u) ||
            (frame + 1u - lastp >= (uint)P[2].x);
        bool cut = (P[1].x > 0.5) && (tv > thr) && cooldown;
        uav_stats[8] = asuint(tv);            // RGB TV（遥测）
        uav_stats[9] = asuint(cut ? 1.0 : 0.0);
        uav_stats[10] = asuint(thr);          // 本帧实际阈值（遥测）
        if (cut) uav_stats[27] = 1u;
        // 入窗（决策之后，供后续帧使用）
        uav_tv[4u + wi] = asuint(tv);
        uav_tv[2] = (wi + 1u) % 32u;
        uav_tv[3] = min(cnt + 1u, 32u);
        if (cut) uav_tv[1] = frame + 1u;
        uav_tv[0] = frame + 1u;
    }
}
"""

# 分位 + Otsu 近峰中位：单 group 256 线程（shared 前缀和只求 total；
# 分位/Otsu/中位一律用 bin 权重 g_bin —— 前缀和是累计值不能当计数累加）
QUANT_CS = """
StructuredBuffer<uint> hist : register(t0);
RWStructuredBuffer<uint> uav_stats : register(u0);
cbuffer CB : register(b0) { float4 P[16]; };

groupshared uint g_bin[256];
groupshared uint g_scan[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    uint v = hist[t];
    g_bin[t] = v;
    g_scan[t] = v;
    GroupMemoryBarrierWithGroupSync();
    // Hillis-Steele 前缀和（256 元素 8 步）
    [unroll]
    for (uint off = 1; off < 256; off <<= 1) {
        uint val = (t >= off) ? g_scan[t - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_scan[t] += val;
        GroupMemoryBarrierWithGroupSync();
    }
    uint total = g_scan[255];

    // ---- 分位：P2 / P98 / P50 的 bin + bin 内线性插值（thread 0 串行） ----
    if (t == 0) {
        uint t_lo = (uint)((float)total * 0.02 + 0.5);
        uint t_hi = (uint)((float)total * 0.98 + 0.5);
        // 中位按 numpy.median 语义：两个中间秩的平均（偶数为两中间值，
        // 奇数为同一秩）——"累计≥N/2 的首个 bin"在双峰并列时会落到低峰
        // 边缘，与 GL 时代的 LAST_MEDIAN（np.median）不一致
        // 1-based 秩：t1=(N+1)/2（奇=正中，偶=左中），t2=N/2+1（偶=右中）
        uint t_mid1 = (total + 1) / 2;
        uint t_mid2 = total / 2 + 1;
        uint t_q10 = (uint)((float)total * 0.10);
        uint t_q90 = (uint)((float)total * 0.90);
        uint acc = 0;
        float lo_v = 0.0, hi_v = 1.0, mid1_v = 0.5, mid2_v = 0.5;
        float q10_v = 0.0, q90_v = 1.0;
        bool got_lo = false, got_hi = false, got_m1 = false, got_m2 = false;
        bool got_q10 = false, got_q90 = false;
        for (uint b = 0; b < 256; ++b) {
            uint c = g_bin[b];
            uint base = acc;
            acc += c;
            if (!got_m1 && c > 0 && t_mid1 <= acc) {
                float frac = saturate((float)(t_mid1 - base) / (float)c);
                mid1_v = ((float)b + frac) / 255.0;
                got_m1 = true;
            }
            if (!got_m2 && c > 0 && t_mid2 <= acc) {
                float frac = saturate((float)(t_mid2 - base) / (float)c);
                mid2_v = ((float)b + frac) / 255.0;
                got_m2 = true;
            }
            if (!got_lo && c > 0 && t_lo <= acc) {
                float frac = saturate((float)(t_lo - base) / (float)c);
                lo_v = ((float)b + frac) / 255.0;
                got_lo = true;
            }
            if (!got_hi && c > 0 && t_hi <= acc) {
                float frac = saturate((float)(t_hi - base) / (float)c);
                hi_v = ((float)b + frac) / 255.0;
                got_hi = true;
            }
            if (!got_q10 && c > 0 && t_q10 <= acc) {
                float frac = saturate((float)(t_q10 - base) / (float)c);
                q10_v = ((float)b + frac) / 255.0;
                got_q10 = true;
            }
            if (!got_q90 && c > 0 && t_q90 <= acc) {
                float frac = saturate((float)(t_q90 - base) / (float)c);
                q90_v = ((float)b + frac) / 255.0;
                got_q90 = true;
            }
            if (got_lo && got_hi && got_m1 && got_m2 && got_q10 && got_q90)
                break;
        }
        // ---- 显示仿射（归一化范围）的更新 ----
        // 无时域跟踪：默认 percentile = 每帧快照（P2/P98 → [0.02,0.98]）。
        // 内容/平移/模型标定变化一律当帧生效，不做帧间拟合或门控。
        float span = P[0].y - P[0].x;
        float lo_raw = P[0].x + saturate(lo_v) * span;
        float hi_raw = P[0].x + saturate(hi_v) * span;
        float mid_n = 0.5 * (saturate(mid1_v) + saturate(mid2_v));
        float mid_raw = P[0].x + mid_n * span;
        // decile（mode 5）：P10/P90 → [0.1, 0.9] 快照，越界才截断
        if ((int)P[2].w == 5) {
            float lo_d = P[0].x + saturate(q10_v) * span;
            float hi_d = P[0].x + saturate(q90_v) * span;
            uav_stats[0] = asuint(lo_d);
            uav_stats[1] = asuint(hi_d);
            uav_stats[6] = asuint(lo_d);
            uav_stats[7] = asuint(hi_d);
            float p50n = saturate(P[2].y + P[2].z *
                                  ((mid_raw - lo_d) / max(1e-6, hi_d - lo_d)));
            uav_stats[4] = asuint(p50n);
            return;
        }
        // ---- 显示仿射更新（无时域跟踪）----
        // 1=percentile（默认）：每帧快照（P2/P98 → [0.02,0.98]）
        // 2=ratchet：只扩不缩  0=lerp：α 平滑（调试用）
        float prev_lo = asfloat(uav_stats[0]);
        float prev_hi = asfloat(uav_stats[1]);
        // cut 用一次性锁存：[27]（RGB_HIST 判切时置 1，本处消费清零）。
        // 注意必须按 **uint** 读（asfloat(1u)=1.4e-45≈0 永远不触发）
        bool cut_latch = (uav_stats[27] != 0u);
        float a_r = saturate(P[1].w);
        bool seeded = (prev_lo > 0.0 && prev_hi > prev_lo + 1e-6);
        bool snap = (!seeded) || cut_latch;
        int mode_c = (int)P[2].w;      // 2=ratchet 1=percentile 0=lerp
        float lo_s, hi_s;
        if (snap) {
            lo_s = lo_raw;
            hi_s = hi_raw;
        } else if (mode_c >= 2) {
            lo_s = min(prev_lo, lo_raw);
            hi_s = max(prev_hi, hi_raw);
        } else {
            lo_s = lerp(prev_lo, lo_raw, a_r);
            hi_s = lerp(prev_hi, hi_raw, a_r);
        }
        uav_stats[0] = asuint(lo_s);
        uav_stats[1] = asuint(hi_s);
        uav_stats[27] = 0;               // 消费锁存
        uav_stats[6] = asuint(lo_raw);   // 诊断：本帧原始分位
        uav_stats[7] = asuint(hi_raw);
        // 收敛面：归一化域 P50（mid_n/mid_raw 已在上方算好）
        // 收敛面与链内 z 同域：用【平滑后】的 lo/hi 归一化
        float p50n = saturate(P[2].y + P[2].z *
                              ((mid_raw - lo_s) / max(1e-6, hi_s - lo_s)));
        uav_stats[4] = asuint(p50n);
    }

    GroupMemoryBarrierWithGroupSync();
    // ---- Otsu 分裂 + 近峰中位（thread 1；对齐 depth_norm.near_peak_median：
    // 近类 = 大值侧，质量不在 [5%,95%] 回退全局 P50） ----
    if (t == 1) {
        if (total == 0) { uav_stats[2] = asuint(0.0); return; }
        float sum_all = (float)total, m_all = 0.0;
        for (uint b = 0; b < 256; ++b) m_all += (float)b * (float)g_bin[b];
        float wB = 0.0, mB = 0.0, best = -1.0;
        uint best_t = 0;
        for (uint k = 0; k < 256; ++k) {
            float c = (float)g_bin[k];
            if (c <= 0.0) continue;
            wB += c;
            mB += (float)k * c;
            float wF = sum_all - wB;
            if (wF <= 0.0) break;
            float d = mB / wB - (m_all - mB) / wF;
            float var = wB * wF * d * d;
            if (var > best) { best = var; best_t = k; }
        }
        uint base = (best_t < 255) ? g_scan[best_t] : total;
        float near_frac = (float)(total - base) / sum_all;
        bool fallback = (near_frac < 0.05) || (near_frac > 0.95);
        uint start = fallback ? 0u : (best_t + 1u);
        // 目标累计数：近类中位 = base + 近类总量/2；回退 = 全局 P50
        uint half = fallback ? (total / 2) : (base + (total - base) / 2);
        uint acc = fallback ? 0u : base;
        float anchor = (float)best_t / 255.0;
        for (uint b = start; b < 256; ++b) {
            uint c = g_bin[b];
            uint baseb = acc;
            acc += c;
            if (c > 0 && acc >= half) {
                float frac = saturate((float)(half - baseb) / (float)c);
                anchor = ((float)b + frac) / 255.0;
                break;
            }
        }
        uav_stats[2] = asuint(anchor);
    }
}
"""

# 显示域收敛锚：对【最终显示深度】（标定→高斯→EMA 之后）做【近侧质量分位】。
# 单一估计器、无方法切换：从近端 bin255 向远累加，累计质量达到 q·N 的深度
# 就是锚（bin 内线性插值）。q = P[1].x（默认 0.05）。
# 没有峰检测 / 显著性门 / Otsu 回退——那套"峰找到了就用峰、否则换 Otsu"
# 正是抖动源；分位对分布连续，帧间只随内容平滑移动 ≤1-2 bin。
ANCHOR_CS = """
Texture2D<float> tex_d : register(t0);
RWStructuredBuffer<uint> uav_stats : register(u0);
cbuffer CB : register(b0) { float4 P[16]; };

groupshared uint g_bin[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    g_bin[t] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint res = (uint)P[0].x;
    uint res_h = (P[0].z > 0.5) ? (uint)P[0].z : res;   // 非方形（0=方形）
    uint step = max(1u, (uint)P[0].y);
    uint sub = (res + step - 1) / step;
    uint sub_h = (res_h + step - 1) / step;
    uint total = sub * sub_h;
    for (uint idx = t; idx < total; idx += 256) {
        uint sx = idx % sub;
        uint sy = idx / sub;
        float v = tex_d.Load(int3((int)(sx * step), (int)(sy * step), 0));
        uint b = (uint)clamp(v * 255.0 + 0.5, 0.0, 255.0);
        InterlockedAdd(g_bin[b], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 1) {
        if (total == 0) { uav_stats[31] = asuint(0.0); return; }
        float target = saturate(P[1].x) * (float)total;
        float acc = 0.0;
        float anchor = 1.0;
        for (int b = 255; b >= 0; --b) {
            float c = (float)g_bin[b];
            if (acc + c >= target) {
                // bin b 覆盖 (b±0.5)/255；从近沿向远沿插值
                float need = max(0.0, target - acc);
                float frac = (c > 0.0) ? need / c : 0.0;
                anchor = saturate(((float)b + 0.5 - frac) / 255.0);
                break;
            }
            acc += c;
        }
        uav_stats[31] = asuint(anchor);
    }
}
"""

