# -*- coding: utf-8 -*-
"""显示链 HLSL（迁移 R4）：shaders.py GLSL → HLSL 逐式翻译。

R4a：PASSTHROUGH（plain_prog 纯 blit）、DEPTH_MAP（depth_prog spectral）。
R4b：DIBR_FS（主 FRAGMENT_SHADER 完整移植——视差/disocclusion 置信/
push-pull 填充/逆一致性搜索/row-split TAB）。

DIBR cbuffer 布局（pack_dibr_consts，64 floats = P[16]）：
  P[0]  = (edge_falloff, eye_offset, depth_strength, convergence)
  P[1]  = (parallax, parallax_curve, roll, fill)
  P[2]  = (ic_search, depth_step_uv, ic_debug, row_split)
  P[3]  = (row_parity, content_feather, full_alpha, corner_radius)
  P[4]  = (search_radius, depth_tolerance, blur_radius, feather_enabled)
  P[5]  = (feather_width, win_w, win_h, res)
  P[6]  = (content.x, content.y, content.z, content.w)
  P[7]  = (viewport.x, viewport.y, viewport.z, viewport.w)
shader 内注释一律 ASCII（FXC ANSI 解析坑）。
"""
import struct

# overlay/光标（alpha blend quad：输出直通 alpha，PSO 走 SrcAlpha/InvSrcAlpha）
OVERLAY_FS = """
Texture2D tex_ov : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    return tex_ov.Sample(samp, i.uv);
}
"""


# 纯 blit（viewer.plain_prog：2D mirror 模式，无立体 warp）
PASSTHROUGH_FS = """
Texture2D tex_color : register(t0);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float4 PSMain(VOut i) : SV_Target {
    return float4(tex_color.Sample(samp, i.uv).rgb, 1.0);
}
"""

# 深度图模式：直接显示**模型原始深度**（post.raw，224²），线性上采样到
# 当前分辨率；按本帧 P2/P98（stats t7 的 g_stats[0]/[1]）归一化到 [0,1] 再
# spectral 上色（近红远蓝）。stats 未绑定时（离屏工具/测试）回退为直接
# saturate(raw)（测试喂的就是 0..1）。
DEPTH_MAP_FS = """
Texture2D tex_depth : register(t0);
StructuredBuffer<uint> g_stats : register(t7);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float3 spectral_r_ultrafast(float t) {
    float3 color1 = float3(0.0, 0.298, 0.651);
    float3 color2 = float3(0.0, 0.5, 0.0);
    float3 color3 = float3(1.0, 0.851, 0.0);
    float3 color4 = float3(0.988, 0.0, 0.0);
    float w1 = max(0.0, 1.0 - abs(t - 0.125) * 4.0);
    float w2 = max(0.0, 1.0 - abs(t - 0.375) * 4.0);
    float w3 = max(0.0, 1.0 - abs(t - 0.625) * 4.0);
    float w4 = max(0.0, 1.0 - abs(t - 0.875) * 4.0);
    float total = w1 + w2 + w3 + w4;
    if (total > 0.0) {
        w1 /= total; w2 /= total; w3 /= total; w4 /= total;
    }
    return color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4;
}

float4 PSMain(VOut i) : SV_Target {
    // D3D i.uv.y already matches texture space (screen top = v 0) --
    // GL depth_prog needed a flip because GL screen y is bottom-up;
    // D3D must NOT flip (vertical-flip bug found by real-frame parity).
    float raw = tex_depth.Sample(samp, i.uv).r;
    float lo = asfloat(g_stats[0]), hi = asfloat(g_stats[1]);
    float depth = (hi > lo + 1e-6)
        ? saturate((raw - lo) / (hi - lo)) : saturate(raw);
    float3 color = spectral_r_ultrafast(depth);
    return float4(color, 1.0);
}
"""


def pack_dibr_consts(edge_falloff, eye_offset, depth_strength, convergence,
                     parallax, parallax_curve, roll, fill,
                     ic_search, depth_step_uv, ic_debug, row_split,
                     row_parity, content_feather, full_alpha, corner_radius,
                     search_radius, depth_tolerance, blur_radius,
                     feather_enabled, feather_width,
                     win_w, win_h, res,
                     content, viewport,
                     interleave_phase=0.0, edge_snap=0.0, edge_snap_gain=1.0,
                     relief_strength=0.0, warp_iterations=0.0):
    """DIBR cbuffer 打包（64 floats；布局见模块头注释）。
    P[8]=(interleave_phase, edge_snap, edge_snap_gain, relief_strength)
    P[9]=(warp_iterations, 0, 0, 0) —— Interleaved 档用。"""
    vals = [edge_falloff, eye_offset, depth_strength, convergence,
            parallax, parallax_curve, roll, fill,
            ic_search, depth_step_uv, ic_debug, row_split,
            row_parity, content_feather, full_alpha, corner_radius,
            search_radius, depth_tolerance, blur_radius, float(feather_enabled),
            feather_width, float(win_w), float(win_h), float(res),
            content[0], content[1], content[2], content[3],
            viewport[0], viewport[1], viewport[2], viewport[3],
            interleave_phase, edge_snap, edge_snap_gain, relief_strength,
            warp_iterations, 0.0, 0.0, 0.0]
    b = struct.pack("<40f", *vals) + b"\x00" * (24 * 4)
    return list(struct.unpack("<64f", b))


# 主 DIBR（viewer FRAGMENT_SHADER 逐式移植；sv_position 用 i.pos）。
# 全局 static g_par_dir/g_sweep_sign 与 GLSL 同语义（PS 内可写）。
DIBR_FS = """
Texture2D tex_color : register(t0);
Texture2D tex_depth : register(t1);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float parallax_units(float depth) {
    float c = clamp(P[0].w, 0.0, 1.0);
    float lin = -2.0 * (depth - c);
    float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
    return lerp(lin, leg, P[1].y) * P[1].x;
}

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    // screen-space uv, y downward = texture space (GL used flipped_uv
    // because GL screen y is bottom-up; D3D i.uv is already top-down)
    float2 flipped_uv = uv;
    float2 win = float2(P[5].y, P[5].z);
    float res = P[5].w;
    float2 pixel_size = 1.0 / win;

    // Row-split (TAB row-weave): snap to alternating source rows
    if (P[2].w > 0.5) {
        uint tw, th, nl;
        tex_color.GetDimensions(0, tw, th, nl);
        float src_h = (float)th;
        float half_row = floor(clamp(flipped_uv.y, 0.0, 0.9999) * src_h * 0.5);
        flipped_uv.y = (half_row * 2.0 + P[3].x + 0.5) / src_h;
    }

    // parallax direction precompute
    float cr = cos(P[1].z);
    float sr = sin(P[1].z);
    float2 par_dir = float2(cr, sr) * sign(P[0].y);
    float sweep_sign = (P[0].y > 0.0) ? -1.0 : 1.0;

    float d0 = tex_depth.Sample(samp, flipped_uv).r;
    float depth = d0;
    float depth_bg = 1.0 - depth;

    float shift = parallax_units(depth);
    float edge_falloff = 1.0;
    if (P[0].x > 0.0001) {
        edge_falloff = smoothstep(0.0, P[0].x, flipped_uv.x)
                     * smoothstep(1.0, 1.0 - P[0].x, flipped_uv.x);
    }
    float content_mask = 1.0;
    if (P[6].z > P[6].x && P[6].w > P[6].y) {
        float f = max(P[3].y, 0.0005);
        // 只向内淡出：区域外严格 0（外溢的 feather 会蹭到任务栏/窗口外）
        float mx = smoothstep(P[6].x, P[6].x + 2.0 * f, flipped_uv.x)
                 * (1.0 - smoothstep(P[6].z - 2.0 * f, P[6].z, flipped_uv.x));
        float my = smoothstep(P[6].y, P[6].y + 2.0 * f, flipped_uv.y)
                 * (1.0 - smoothstep(P[6].w - 2.0 * f, P[6].w, flipped_uv.y));
        content_mask = mx * my;
    }
    float px = P[0].y * shift * P[0].z * edge_falloff * content_mask;
    float2 shifted_uv = flipped_uv - float2(px * cr, px * sr);

    float4 color = tex_color.Sample(samp, shifted_uv);

    float2 border = smoothstep(-0.001, 0.001, shifted_uv)
                  * smoothstep(1.001, 0.999, shifted_uv);
    color.a = min(border.x, border.y);
    if (P[3].z > 0.5 || P[6].z > P[6].x) {
        color.a = 1.0;
    }

    // Natural edge feathering (D3D SV_Position.y is top-down; GL
    // gl_FragCoord.y was bottom-up -- invert y distances to keep the
    // same screen-side semantics: top distance shrinks toward screen top)
    if ((int)P[4].w == 1) {
        float4 vvp = P[7];
        float2 fuv = (i.pos.xy - vvp.xy) / vvp.zw;
        float left = fuv.x;
        float right = 1.0 - fuv.x;
        float top = 1.0 - fuv.y;
        float bottom = fuv.y;
        float feather = P[5].x;
        float fadeL = smoothstep(0.0, feather, left);
        float fadeR = smoothstep(0.0, feather, right);
        float fadeT = smoothstep(0.0, feather, top);
        float fadeB = smoothstep(0.0, feather, bottom);
        float falloff = pow(fadeL * fadeR * fadeT * fadeB, 0.7);
        color.rgb *= falloff;
    }

    // Rounded corners (SDF)
    float corner_r = P[3].w;
    float2 dd = abs(uv - 0.5) - 0.5 + corner_r;
    float corner_sdf = length(max(dd, float2(0.0, 0.0)))
                     + min(max(dd.x, dd.y), 0.0) - corner_r;
    float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
    color.a = min(color.a, corner_alpha);

    return color;
}
"""


# 行交错立体（viewer.interleaved_prog 完整移植，产品主力档——LG 偏光屏）。
# 相对 DIBR_FS 的增量：逐行 eye_dir（row parity from top + phase）、
# edge_snap（ParallaxRT 轮廓处理）、RGB relief（对比度拉深度）、
# warp 不动点修正。cbuffer 复用 pack 布局（P[8]/P[9] 扩展段）。
INTERLEAVED_FS = """
Texture2D tex_color : register(t0);
Texture2D tex_depth : register(t1);
SamplerState samp : register(s0);
cbuffer CB : register(b0) { float4 P[16]; };

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

float parallax_units_il(float depth) {
    float c = clamp(P[0].w, 0.0, 1.0);
    float lin = -2.0 * (depth - c);
    float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
    return lerp(lin, leg, P[1].y) * P[1].x;
}

float4 PSMain(VOut i) : SV_Target {
    float2 uv = i.uv;
    // screen-space uv, y downward = texture space (GL used flipped_uv;
    // see DIBR_FS note -- D3D must NOT flip)
    float2 flipped_uv = uv;
    float2 win = float2(P[5].y, P[5].z);
    float res = P[5].w;
    float2 pixel_size = 1.0 / win;
    float2 depth_texel = 1.0 / float2(res, res);

    // Row parity from the TOP of the panel (GL gl_FragCoord.y was
    // bottom-up; D3D i.pos.y is already top-down -- no flip needed)
    int row_idx = (int)floor(clamp(flipped_uv.y, 0.0, 0.999999) * P[7].w);
    float eye_dir = (fmod((float)row_idx + P[8].x, 2.0) < 1.0) ? -1.0 : 1.0;
    float my_offset = eye_dir * P[0].y;

    float cr = cos(P[1].z);
    float sr = sin(P[1].z);

    float d0 = tex_depth.Sample(samp, flipped_uv).r;
    float depth = d0;
    float depth_bg = 1.0 - depth;

    // ParallaxRT-style silhouette: snap depth onto background at edges
    if (P[8].y > 0.0) {
        float2 e_dir = float2(cr, sr) * depth_texel * 2.0;
        float dL = tex_depth.Sample(samp, flipped_uv - e_dir).r;
        float dR = tex_depth.Sample(samp, flipped_uv + e_dir).r;
        float gradL = depth - dL;
        float gradR = depth - dR;
        if (eye_dir < 0.0) {
            float thr = 0.12 / max(0.05, P[8].z);
            if (gradL > thr) depth = dL;
            else if (gradR > thr * 0.85) depth -= 0.25 * gradR;
        } else {
            if (gradR > 0.12) depth = dR;
            else if (gradL > 0.10) depth -= 0.25 * gradL;
        }
        depth = clamp(depth, 0.0, 1.0);
        depth_bg = 1.0 - depth;
    }

    // RGB relief: local contrast pulls depth onto real edge
    if (P[8].w > 0.0) {
        float3 LW = float3(0.299, 0.587, 0.114);
        float2 r_dir = pixel_size * 2.0;
        float lc = dot(tex_color.Sample(samp, flipped_uv).rgb, LW);
        float ll = dot(tex_color.Sample(samp, flipped_uv - float2(r_dir.x, 0.0)).rgb, LW);
        float lr = dot(tex_color.Sample(samp, flipped_uv + float2(r_dir.x, 0.0)).rgb, LW);
        float lt = dot(tex_color.Sample(samp, flipped_uv - float2(0.0, r_dir.y)).rgb, LW);
        float lb = dot(tex_color.Sample(samp, flipped_uv + float2(0.0, r_dir.y)).rgb, LW);
        depth = clamp(depth + (lc - 0.25 * (ll + lr + lt + lb)) * P[8].w,
                      0.0, 1.0);
        depth_bg = 1.0 - depth;
    }

    float shift = parallax_units_il(depth);
    float edge_falloff = 1.0;
    if (P[0].x > 0.0001) {
        edge_falloff = smoothstep(0.0, P[0].x, flipped_uv.x)
                     * smoothstep(1.0, 1.0 - P[0].x, flipped_uv.x);
    }
    float content_mask = 1.0;
    if (P[6].z > P[6].x && P[6].w > P[6].y) {
        float f = max(P[3].y, 0.0005);
        float mx = smoothstep(P[6].x, P[6].x + 2.0 * f, flipped_uv.x)
                 * (1.0 - smoothstep(P[6].z - 2.0 * f, P[6].z, flipped_uv.x));
        float my = smoothstep(P[6].y, P[6].y + 2.0 * f, flipped_uv.y)
                 * (1.0 - smoothstep(P[6].w - 2.0 * f, P[6].w, flipped_uv.y));
        content_mask = mx * my;
    }
    float px = my_offset * shift * P[0].z * edge_falloff * content_mask;
    float2 shifted_uv = flipped_uv - float2(px * cr, px * sr);

    // one fixed-point warp refinement (damped 50%)
    if (P[9].x > 0.0) {
        float d_src = tex_depth.Sample(samp, clamp(shifted_uv, 0.0, 1.0)).r;
        float px_src = my_offset * parallax_units_il(d_src) * P[0].z
                     * edge_falloff * content_mask;
        float2 fixed_uv = flipped_uv - float2(px_src * cr, px_src * sr);
        shifted_uv = lerp(shifted_uv, fixed_uv, 0.5);
    }

    float4 color = tex_color.Sample(samp, shifted_uv);

    float2 border = smoothstep(-0.001, 0.001, shifted_uv)
                  * smoothstep(1.001, 0.999, shifted_uv);
    color.a = min(border.x, border.y);

    // feather (D3D i.pos.y top-down -- inverted y distances, see DIBR_FS)
    if ((int)P[4].w == 1) {
        float4 vvp = P[7];
        float2 fuv = (i.pos.xy - vvp.xy) / vvp.zw;
        float feather = P[5].x;
        float fadeL = smoothstep(0.0, feather, fuv.x);
        float fadeR = smoothstep(0.0, feather, 1.0 - fuv.x);
        float fadeT = smoothstep(0.0, feather, 1.0 - fuv.y);
        float fadeB = smoothstep(0.0, feather, fuv.y);
        color.rgb *= pow(fadeL * fadeR * fadeT * fadeB, 0.7);
    }

    // rounded corners SDF
    float corner_r = P[3].w;
    float2 fuv2 = (i.pos.xy - P[7].xy) / P[7].zw;
    float2 dd = abs(fuv2 - 0.5) - 0.5 + corner_r;
    float corner_sdf = length(max(dd, float2(0.0, 0.0)))
                     + min(max(dd.x, dd.y), 0.0) - corner_r;
    float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
    color.a = min(color.a, corner_alpha);

    return color;
}
"""
