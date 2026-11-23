# -*- coding: utf-8 -*-
"""深度链 HLSL 移植（P1）：GLSL（depth_post.py）→ HLSL 逐式翻译，公式零改动。

对照表（GLSL → HLSL）：texture() → tex.Sample(samp, uv)；vecN → floatN；
mix → lerp；gl_FragCoord → SV_Position 对应像素中心（+0.5 修正）。
每段注明 GLSL 源（depth_post.py 行内着色器）。

cbuffer b0 布局（64 floats，与 d3d12.D3D12Context.CB_SLOT_STRIDE 对齐）：
  c0-c7   公共：res, lo, hi, grow/aa/k, alpha/sigma, target, pad, pad
  c8-c15  预留（dir、视差等）
"""
import struct

# （原 DEPTH_VS 死代码已删——生产全用 hlsl.FULLSCREEN_VS，DXC/SM6 兼容）

# ---------------------------------------------------------------------------
# prep（生产版，迁移 R3）：BGRA 帧 → R32F CHW 平铺 + ImageNet 归一化
#   （viewer.gpu_preprocess 双模式逐式移植）：
#   flat 模式（flat_w>0，D2S_ZEROCOPY 生产路径）：输出 192×(3R²/192)，
#     线性序 CHW、行距 768B=256 对齐且无 padding；
#   非 flat：3R×R 行内 [R|G|B] 平铺。
#   BGRA8 采样 .xyzw=(B,G,R,A)：c=0(R) -> col.z；c=1(G) -> col.y；c=2(B) -> col.x。
#   pack: P[0]=(res, flat_w, 0, 0)。shader 内注释一律 ASCII（FXC ANSI 解析坑）。
# ---------------------------------------------------------------------------
PREP_FS = """
Texture2D tex_c : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float res = P[0].x;
    int flat_w = (int)P[0].y;
    int c;
    float xr, yr;
    if (flat_w > 0) {
        int fl = int(i.pos.y) * flat_w + int(i.pos.x);
        int rr = int(res) * int(res);
        c = fl / rr;
        int rem = fl - c * rr;
        int row = rem / int(res);
        xr = float(rem - row * int(res)) + 0.5;
        yr = float(row) + 0.5;
    } else {
        int ri = int(i.pos.x);
        c = (int)(ri / int(res));
        xr = float(ri - c * int(res)) + 0.5;
        yr = i.pos.y;
    }
    float2 suv = float2(xr / res, yr / res);
    // BGRA8 samples as RGBA semantics on D3D10+ (auto swizzle, .x=R) --
    // same as GL texture().rgb; no manual channel remap needed.
    // NOTE: no const float3 here -- FXC (d3dcompiler_47, this env) emits
    // E_FAIL with a garbage err blob on const-float3 + dynamic component
    // select; literal ternaries compile fine.
    float3 col = tex_c.SampleLevel(samp, suv, 0).xyz;
    float v = (c == 0) ? col.x : ((c == 1) ? col.y : col.z);
    float m = (c == 0) ? 0.485 : ((c == 1) ? 0.456 : 0.406);
    float s = (c == 0) ? 0.229 : ((c == 1) ? 0.224 : 0.225);
    return float4((v - m) / s, 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 可选：13×13 max 膨胀（独立 pass，默认关；settings "Depth Grow (px)" 打开）
# 在 raw 域做 max 滤波，再做归一化（max 与单调仿射映射可交换，与原
# 融合实现 NORM_DIL_FS 等价）。实测膨胀会把轮廓推出背景，故默认 0。
# ---------------------------------------------------------------------------
DIL_FS = """
Texture2D tex_raw : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float res = P[0].x;
    float grow = P[0].w;
    float best = -1.0;
    for (int dy = -6; dy <= 6; ++dy) {
        for (int dx = -6; dx <= 6; ++dx) {
            if (abs((float)dx) > grow || abs((float)dy) > grow) continue;
            best = max(best, tex_raw.SampleLevel(samp,
                uv + float2((float)dx, (float)dy) / res, 0).r);
        }
    }
    return float4(best, 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 区间归一化（lo/hi 由 cbuffer 传入；生产用 NORM_STATS_FS 从 stats 读）。
# 只做 lo/hi → 显示带的线性映射；闭运算与坡度限幅已删除，膨胀拆成上面
# 的独立可选 pass（2026-11-23）
# ---------------------------------------------------------------------------
NORM_FS = """
Texture2D tex_raw : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float lo = P[0].y, hi = P[0].z;
    float v = tex_raw.SampleLevel(samp, uv, 0).r;
    return float4(saturate((v - lo) / max(1e-6, hi - lo)), 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 分离高斯（depth_post._gauss_prog：19-tap，dir 由 cbuffer 传入）
# ---------------------------------------------------------------------------
GAUSS_FS = """
Texture2D tex_d : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float res = P[0].x;
    float sigma = P[1].z;        // pack layout: sigma @ P[1].z
    float2 dir = P[2].yz;        // dirx/diry @ P[2].y/.z
    float acc = 0.0, wsum = 0.0;
    float s2 = 2.0 * sigma * sigma;
    for (int q = -9; q <= 9; ++q) {
        float fi = (float)q;
        float w = exp(-(fi * fi) / max(1e-6, s2));
        acc += w * tex_d.Sample(samp, uv + dir * fi / res).r;
        wsum += w;
    }
    return float4(acc / max(1e-6, wsum), 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# legacy EMA（depth_post._pp_prog：normalize + 9-tap AA + EMA）
# ---------------------------------------------------------------------------
PP_FS = """
Texture2D tex_raw : register(t0);
Texture2D tex_prev : register(t1);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float res = P[0].x;
    // pack layout: P[0]=(res,lo,hi,w) P[1]=(alpha,aa,sigma,m)
    float lo = P[0].y, hi = P[0].z;
    float alpha = P[1].x, aa = P[1].y;
    float sc = 1.0 / max(1e-6, hi - lo);
    float d = clamp((tex_raw.Sample(samp, uv).r - lo) * sc, 0.0, 1.0);
    if (aa > 0.001) {
        float w1 = 0.6065, w2 = 0.1353;
        float acc = d, wsum = 1.0;
        float2 e1 = float2(aa / res, 0.0), e2 = float2(0.0, aa / res);
        float2 e3 = float2(2.0 * aa / res, 0.0), e4 = float2(0.0, 2.0 * aa / res);
        acc += w1 * clamp((tex_raw.Sample(samp, uv + e1).r - lo) * sc, 0.0, 1.0);
        acc += w1 * clamp((tex_raw.Sample(samp, uv - e1).r - lo) * sc, 0.0, 1.0);
        acc += w1 * clamp((tex_raw.Sample(samp, uv + e2).r - lo) * sc, 0.0, 1.0);
        acc += w1 * clamp((tex_raw.Sample(samp, uv - e2).r - lo) * sc, 0.0, 1.0);
        acc += w2 * clamp((tex_raw.Sample(samp, uv + e3).r - lo) * sc, 0.0, 1.0);
        acc += w2 * clamp((tex_raw.Sample(samp, uv - e3).r - lo) * sc, 0.0, 1.0);
        acc += w2 * clamp((tex_raw.Sample(samp, uv + e4).r - lo) * sc, 0.0, 1.0);
        acc += w2 * clamp((tex_raw.Sample(samp, uv - e4).r - lo) * sc, 0.0, 1.0);
        d = acc / wsum;
    }
    float p = tex_prev.Sample(samp, uv).r;
    return float4(alpha * d + (1.0 - alpha) * p, 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 时域中值 W5（depth_post.MEDIAN_FS：9-swap 排序网络，纹理版 5 输入）
# ---------------------------------------------------------------------------
MEDIAN_FS = """
Texture2D tex_c  : register(t0);
Texture2D tex_h0 : register(t1);
Texture2D tex_h1 : register(t2);
Texture2D tex_h2 : register(t3);
Texture2D tex_h3 : register(t4);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float a = tex_c.Sample(samp, uv).r;
    float b = tex_h0.Sample(samp, uv).r;
    float c = tex_h1.Sample(samp, uv).r;
    float d = tex_h2.Sample(samp, uv).r;
    float e = tex_h3.Sample(samp, uv).r;
    float t;
    t = min(a, b); b = max(a, b); a = t;
    t = min(d, e); e = max(d, e); d = t;
    t = min(c, e); e = max(c, e); c = t;
    t = min(c, d); d = max(c, d); c = t;
    t = min(a, d); d = max(a, d); a = t;
    t = min(a, c); c = max(a, c); a = t;
    t = min(b, e); e = max(b, e); b = t;
    t = min(b, d); d = max(b, d); b = t;
    t = min(b, c); c = max(b, c); b = t;
    return float4(c, 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 生产版区间归一化：lo/hi 直读 GPU 判据 stats（t7），不走 cbuffer
# （V2 判据链；只做映射，无膨胀）
# ---------------------------------------------------------------------------
NORM_STATS_FS = """
Texture2D tex_raw : register(t0);
StructuredBuffer<uint> g_stats : register(t7);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float lo = asfloat(g_stats[0]), hi = asfloat(g_stats[1]);
    float v = tex_raw.SampleLevel(samp, uv, 0).r;
    float n = saturate((v - lo) / max(1e-6, hi - lo));
    return float4(saturate(P[3].y + P[3].z * n), 0.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# 生产版 EMA：同 PP_FS（输入已归一化，等价 lo=0/hi=1 直通），另读 stats[9]
# （RGB 场景切换决策）——=1 时 alpha=1.0 直通（切变帧重置时域状态）
# ---------------------------------------------------------------------------
PP_STATS_FS = """
Texture2D tex_raw : register(t0);
Texture2D tex_prev : register(t1);
StructuredBuffer<uint> g_stats : register(t7);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    float res = P[0].x;
    float alpha = P[1].x, aa = P[1].y;
    float d = saturate(tex_raw.Sample(samp, uv).r);
    if (aa > 0.001) {
        float w1 = 0.6065, w2 = 0.1353;
        float acc = d, wsum = 1.0;
        float2 e1 = float2(aa / res, 0.0), e2 = float2(0.0, aa / res);
        float2 e3 = float2(2.0 * aa / res, 0.0), e4 = float2(0.0, 2.0 * aa / res);
        acc += w1 * saturate(tex_raw.Sample(samp, uv + e1).r);
        acc += w1 * saturate(tex_raw.Sample(samp, uv - e1).r);
        acc += w1 * saturate(tex_raw.Sample(samp, uv + e2).r);
        acc += w1 * saturate(tex_raw.Sample(samp, uv - e2).r);
        acc += w2 * saturate(tex_raw.Sample(samp, uv + e3).r);
        acc += w2 * saturate(tex_raw.Sample(samp, uv - e3).r);
        acc += w2 * saturate(tex_raw.Sample(samp, uv + e4).r);
        acc += w2 * saturate(tex_raw.Sample(samp, uv - e4).r);
        d = acc / wsum;
    }
    if (asfloat(g_stats[9]) > 0.5) alpha = 1.0;   // 切变帧重置（纯 RGB 决策）
    float p = tex_prev.Sample(samp, uv).r;
    return float4(alpha * d + (1.0 - alpha) * p, 0.0, 0.0, 1.0);
}
"""

# cbuffer 打包：res, lo, hi, grow | alpha, aa, sigma, m | target, dir.x, dir.y, k
#              | cut_area(c12)
def pack_consts(res, lo=0.0, hi=1.0, w=0.0, alpha=0.0, aa=0.0, sigma=0.0,
                m=0.0, target=0.0, dirx=0.0, diry=0.0, k=0.0, cut_area=0.0):
    return struct.pack("<13f", res, lo, hi, w, alpha, aa, sigma, m,
                       target, dirx, diry, k, cut_area) + b"\x00" * 51 * 4

# ---------------------------------------------------------------------------
# buf→tex（GPU 输出推理结果的纹理化）：StructuredBuffer<float> → R32F 纹理。
# 为什么不叫 CopyTextureRegion：res*4（224→896B）不满足 placed footprint
# 的 256 字节 RowPitch 对齐，Close 会 E_INVALIDARG；用全屏 PS 一读一写即可，
# 224² 的代价约 0.01ms。P[0].x = res。
# ---------------------------------------------------------------------------
BUF_TO_TEX_FS = """
StructuredBuffer<float> srcB : register(t0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    uint w = (uint)P[0].x;
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    return srcB[y * w + x];
}
"""
