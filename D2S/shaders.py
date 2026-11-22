# -*- coding: utf-8 -*-
"""GLSL 着色器源码单源（原内联于 viewer.py，2026-11-16 抽出）。

tools/live_offline.py 靠解析本文件的字符串常量做离线对拍——改这里的
常量名/引号风格前先同步它。
"""

VERTEX_SHADER = """
    #version 330
    in vec2 in_position;
    in vec2 in_uv;
    out vec2 uv;
    void main() {
        uv = in_uv;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

FRAGMENT_SHADER = """
    #version 330
    // Optimized stereoscopic inpainting with push-pull method
    
    in vec2 uv;
    out vec4 frag_color;

    uniform sampler2D tex_color;
    vec4 sample_color(vec2 uv){ return texture(tex_color, uv); }  // interop 视图通道本就正确
   // RGB image
    uniform sampler2D tex_depth;
    uniform float u_edge_falloff;   // 边缘渐隐宽度（0=关，0.01=旧默认）   // Single-channel depth (0 = near, 1 = far)
    uniform vec2 u_resolution;
    uniform vec2 u_depth_resolution;  // 深度纹理分辨率（S3 起为模型分辨率 224）     // viewport resolution
    uniform float u_eye_offset;    // e.g. +/-0.03 (positive = right eye)
    uniform float u_depth_strength;// parallax intensity
    uniform float u_convergence;   // depth value at screen plane (0..1)
    uniform float u_parallax;
    uniform float u_parallax_curve; // 0 = linear in depth, 1 = legacy non-linear shaping
    uniform float u_roll;          // screen roll (radians), rotates parallax direction
    uniform float u_fill;          // 1 = spatial disocclusion fill
    uniform float u_ic_search;     // 1 = 逆一致性搜索
    uniform float u_depth_step_uv; // 1/depth_res
    uniform float u_ic_debug;      // 1 = 输出路径染色（调试）
    uniform float u_row_split;     // 1.0 = row-split sampling (TAB eye-separation mode)
    uniform float u_row_parity;    // 0.0 = even source rows, 1.0 = odd source rows
    // 内容区域（UV 空间，含翻转）：只有这块区域做视差，其余原样投射
    // z <= x 表示"未启用"（整屏都做）
    uniform vec4 u_content;
    uniform float u_content_feather;
    // 1 = 全屏输出（整屏都由我们画）→ 强制不透明，否则视差越界会把 alpha
    // 压到 0，DWM 会把 layered 窗口合成成"透明"（屏幕上看不到任何东西）
    uniform float u_full_alpha;

    // Optimized inpainting controls
    uniform float u_search_radius = 12.0;  // horizontal search distance
    uniform float u_depth_tolerance = 0.012; // background detection threshold
    uniform float u_blur_radius = 2.5;     // final smoothing
    // Edge feathering controls
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;

    // Rounded corners
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    vec2 depth_texel = 1.0 / u_depth_resolution;  // 深度域 tap 的步长

    // Signed shift multiplier for u_eye_offset (half the IPD in UV units).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    float parallax_units(float depth) {
        float c = clamp(u_convergence, 0.0, 1.0);
        float lin = -2.0 * (depth - c);
        // Legacy (pre-calibration) shaping: non-linear in depth, zero plane at
        // ~0.415 when Convergence = 0.5, which boosts the near-mid ground.
        float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
        return mix(lin, leg, u_parallax_curve) * u_parallax;
    }

    // Precomputed parallax direction (set once in main, reused by helpers)
    vec2  g_par_dir;     // normalized roll direction (cos,sin) * sign(eye_offset)
    float g_sweep_sign;  // +1 / -1: side the background is revealed from

    // SOFT DISOCCLUSION CONFIDENCE (0 = none, 1 = full)
    // The test is depth CONSISTENCY: if the sample we fetched sits on a
    // different surface than the pixel we are rendering, the source was
    // occluded there and a hole has to be filled. (A depth-gradient test only
    // flags the 2 px seam and leaves the rest of the hole showing stretched
    // foreground - that is what looked like "holes" along object edges.)
    float disocclusion_confidence(vec2 base_uv, vec2 shifted_uv) {
        // Out-of-bounds shifted UV -> fully disoccluded
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return 1.0;

        float d_here  = texture(tex_depth, base_uv).r;
        float d_there = texture(tex_depth, shifted_uv).r;
        // Smooth ramp instead of hard threshold (soft edges -> fewer seams)
        return smoothstep(0.03, 0.10, abs(d_here - d_there));
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        // Hole fill radius tracks the parallax budget: a depth edge opens a gap
        // as wide as (2 x budget), so a fixed 12px search left anything
        // bigger unfilled (the "torn" streaks between near and far surfaces).
        float _budget_px = 2.0 * u_parallax * abs(u_eye_offset) * u_viewport.z;
        int search_range = int(clamp(_budget_px * 1.2, 12.0, 48.0));
        // Sweep toward the side the background is revealed from (trig precomputed)
        vec2 sweep = g_par_dir * pixel_size.x * g_sweep_sign;
        
        // Phase 1: Directional sweep (most samples here)
        for (int i = 1; i <= search_range; i++) {
            vec2 sample_uv = uv_coord + sweep * float(i);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            
            // Only accept background pixels (farther than current)
            if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                vec4 sample_color = sample_color(sample_uv);
                
                // Weight by: distance + depth similarity to other background
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                
                best_color += sample_color * w;
                best_weight += w;
                
                // Early exit if we found strong background
                if (best_weight > 5.0) break;
            }
        }

        // Phase 2: opposite sweep fallback if not enough background found
        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                vec2 sample_uv = uv_coord - sweep * float(i);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                    vec4 sample_color = sample_color(sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }
        
        // Phase 3: Small vertical blur for smoothness (3 taps)
        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (abs(vert_depth_inv - center_depth_inv) < u_depth_tolerance * 2.0) {
                        float w = 0.25;
                        vert_accum += sample_color(vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            
            return vert_accum / vert_weight;
        }
        
        // Fallback: return original pixel if no background found
        return sample_color(uv_coord);
    }

    // ALTERNATIVE: FAST SEPARABLE BLUR (Uncomment to use instead)
    vec4 separable_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 accum = vec4(0.0);
        float total_w = 0.0;
        int R = 4;
        
        // Horizontal pass only (vertical would be a second shader pass)
        for (int x = -R; x <= R; ++x) {
            vec2 sample_uv = uv_coord + vec2(x * pixel_size.x, 0.0);
            
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0) continue;
            
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            
            // Background filter
            float depth_ok = step(center_depth_inv + 0.015, sample_depth_inv);
            float w = exp(-float(x*x) * 0.25) * (1.0 + depth_ok * 10.0);
            
            accum += sample_color(sample_uv) * w;
            total_w += w;
        }
        
        return total_w > 0.01 ? accum / total_w : sample_color(uv_coord);
    }

    // MAIN
    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // Row-split mode: snap sampling to alternating source rows so the two
        // TAB halves carry opposite row phases - a row-zebra source reaches
        // the two eyes as solid black / solid white. Row index is derived
        // from flipped_uv (image space, top-down), NOT gl_FragCoord (GL
        // window space is bottom-up - using it flipped the picture).
        // Parity is set per viewport (top half = even rows, bottom = odd).
        if (u_row_split > 0.5) {
            float src_h = float(textureSize(tex_color, 0).y);
            float half_row = floor(clamp(flipped_uv.y, 0.0, 0.9999) * src_h * 0.5);
            flipped_uv.y = (half_row * 2.0 + u_row_parity + 0.5) / src_h;
        }

        // Precompute parallax direction once (perf win: was recomputed per-loop)
        float c = cos(u_roll);
        float s = sin(u_roll);
        g_par_dir = vec2(c, s) * sign(u_eye_offset);
        g_sweep_sign = (u_eye_offset > 0.0) ? -1.0 : 1.0;

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir.
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width.
        // 3-tap 深度平滑已移除：它沿视差方向糊 ±1.5 texel（224 纹理上
        // ≈ 屏幕 ±13px），把深度边界摊平，是'脸被拉裂/糊开'的主要原因。
        float d0 = texture(tex_depth, flipped_uv).r;
        float depth = d0;
        float depth_inv = -depth;
        // Background metric for push_pull_inpaint(): the function compares a
        // sample against (1 - depth) to decide "farther than the centre".
        // Passing depth_inv (-depth) there compared two different spaces and
        // made the test almost always true (every sample accepted -> smear).
        float depth_bg = 1.0 - depth;

        // Calibrated parallax: the screen plane sits at u_convergence, with
        // independent near/far budgets (screen-plane-relative mapping).
        float shift = parallax_units(depth);
        // u_edge_falloff=0（关）时 smoothstep(0,0,x) 是 GLSL 未定义行为
        //（NVIDIA 返回 0 → 全像素视差乘零 = 3D 灭）——显式短路为 1.0
        float edge_falloff = 1.0;
        if (u_edge_falloff > 0.0001) {
            edge_falloff = smoothstep(0.0, u_edge_falloff, flipped_uv.x)
                         * smoothstep(1.0, 1.0 - u_edge_falloff, flipped_uv.x);
        }
        // 内容区域遮罩：框内做视差，框外为 0（两个半屏完全相同 = 无立体）。
        // u_content.z <= u_content.x 表示未启用，整屏都做。
        float content_mask = 1.0;
        if (u_content.z > u_content.x && u_content.w > u_content.y) {
            float f = max(u_content_feather, 0.0005);
            float mx = smoothstep(u_content.x - f, u_content.x + f, flipped_uv.x)
                     * (1.0 - smoothstep(u_content.z - f, u_content.z + f, flipped_uv.x));
            float my = smoothstep(u_content.y - f, u_content.y + f, flipped_uv.y)
                     * (1.0 - smoothstep(u_content.w - f, u_content.w + f, flipped_uv.y));
            content_mask = mx * my;
        }
        float px = u_eye_offset * shift * u_depth_strength * edge_falloff * content_mask;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // ---- 逆一致性搜索（方案 C）----
        float d_self = texture(tex_depth, clamp(shifted_uv, 0.0, 1.0)).r;
        if (u_ic_search > 0.5 && u_depth_strength > 0.0
                && abs(d_self - depth) > 0.08) {
            float best_d = -1.0;
            vec2 best_uv = shifted_uv;
            float tol_uv = u_depth_step_uv * 1.5;
            for (int i = -16; i <= 16; i++) {
                vec2 uv_s = flipped_uv + vec2(float(i) * u_depth_step_uv * c,
                                              float(i) * u_depth_step_uv * s);
                float ds = texture(tex_depth, clamp(uv_s, 0.0, 1.0)).r;
                float px_s = u_eye_offset * parallax_units(ds) * u_depth_strength
                             * edge_falloff * content_mask;
                vec2 dest = uv_s - vec2(px_s * c, px_s * s);
                if (abs(dest.x - flipped_uv.x) < tol_uv
                        && abs(dest.y - flipped_uv.y) < tol_uv
                        && ds > best_d) {
                    best_d = ds;
                    best_uv = uv_s;
                }
            }
            if (best_d >= 0.0) {
                shifted_uv = best_uv;
            } else {
                float dir_s = (px > 0.0) ? -1.0 : 1.0;
                float back = abs(2.0 * u_parallax * u_eye_offset) + 0.002;
                shifted_uv = flipped_uv + vec2(dir_s * back * c, dir_s * back * s);
            }
        }

        // Soft disocclusion: blend inpaint in by confidence to hide hard seams
        float conf = disocclusion_confidence(flipped_uv, shifted_uv);

        // Normal sampling
        vec4 color = sample_color(shifted_uv);
        // Gate on real parallax: in the 2D mirror depth_strength == 0, so the
        // (stale) depth texture must not trigger inpainting - it would replace
        // 2D pixels at depth edges with background colour.
        if (u_fill > 0.0 && conf > 0.001 && u_depth_strength > 0.0) {
            // Disoccluded region: optimized inpainting, blended by confidence
            vec4 filled = push_pull_inpaint(flipped_uv, depth_bg);
            // Alternative: vec4 filled = separable_inpaint(flipped_uv, depth_bg);
            color = mix(color, filled, conf);
        }

        // Screen-edge alpha clip: keep the out-of-bounds safety net (alpha -> 0
        // if parallax somehow over-shoots into negative UV) but use a
        // sub-pixel fade band so the user does not see a visible soft border
        // between the desktop image and the screen edge. 
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        // 全屏输出必须不透明：视差会把采样坐标推出 [0,1]，alpha→0 时 DWM
        // 会把 layered 窗口合成成"透明"，屏幕上等于什么都看不到
        //（导出帧却正常，因为读的是 RGB 不看 alpha）。
        if (u_full_alpha > 0.5 || u_content.z > u_content.x) {
            color.a = 1.0;
        }
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {

            vec2 uv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;

            // Distance to each edge
            float left   = uv.x;
            float right  = 1.0 - uv.x;
            float top    = uv.y;
            float bottom = 1.0 - uv.y;

            float feather = u_feather_width;

            // Smooth edge fades
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);

            // Combine smoothly
            float falloff = fadeL * fadeR * fadeT * fadeB;

            // Perceptual shaping
            // <1 softer
            // >1 harder
            falloff = pow(falloff, 0.7);

            color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF (screen-space uv, not shifted_uv)
        // Inigo Quilez rounded-box SDF
        float corner_r = u_corner_radius;
        vec2 d = abs(uv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;

        // Mask out rounded corners
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        color.a = min(color.a, corner_alpha);

        // Glow outside the screen edge only (sdf > 0 = outside the rounded rect).
        // Moved to dedicated glow quad in XR view.

        frag_color = color;
    }
"""

DEPTH_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_depth;
    uniform float u_edge_falloff;   // 边缘渐隐宽度（0=关，0.01=旧默认）
    
    // Optimized spectral-like colormap without normalization
    vec3 spectral_r_ultrafast(float t) {
        // Precomputed key colors for spectral_r approximation
        // Blue (far) -> Green -> Yellow -> Red (near)
        
        // Single branch - optimized for GPU parallelism
        vec3 color1 = vec3(0.0, 0.298, 0.651);      // Blue
        vec3 color2 = vec3(0.0, 0.5, 0.0);          // Green
        vec3 color3 = vec3(1.0, 0.851, 0.0);        // Yellow
        vec3 color4 = vec3(0.988, 0.0, 0.0);        // Red
        
        // Piecewise linear interpolation without conditional branches
        float w1 = max(0.0, 1.0 - abs(t - 0.125) * 4.0);
        float w2 = max(0.0, 1.0 - abs(t - 0.375) * 4.0);
        float w3 = max(0.0, 1.0 - abs(t - 0.625) * 4.0);
        float w4 = max(0.0, 1.0 - abs(t - 0.875) * 4.0);
        
        // Normalize weights
        float total = w1 + w2 + w3 + w4;
        if (total > 0.0) {
            w1 /= total; w2 /= total; w3 /= total; w4 /= total;
        }
        
        return color1 * w1 + color2 * w2 + color3 * w3 + color4 * w4;
    }

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float depth = texture(tex_depth, flipped_uv).r;
        
        // Depth is already in [0,1] range, no normalization needed
        // Apply optimized spectral colormap directly
        vec3 color = spectral_r_ultrafast(depth);
        
        frag_color = vec4(color, 1.0);
    }
"""

# Anaglyph red-cyan composite shader: samples both eyes and blends.
ANAGLYPH_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    vec4 sample_color(vec2 uv){ return texture(tex_color, uv); }  // interop 视图通道本就正确

    uniform sampler2D tex_depth;
    uniform float u_edge_falloff;   // 边缘渐隐宽度（0=关，0.01=旧默认）
    uniform vec2 u_resolution;
    uniform vec2 u_depth_resolution;  // 深度纹理分辨率（S3 起为模型分辨率 224）
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_parallax;
    uniform float u_parallax_curve; // 0 = linear in depth, 1 = legacy non-linear shaping
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    vec2 depth_texel = 1.0 / u_depth_resolution;  // 深度域 tap 的步长

    // Signed shift multiplier for u_eye_offset (half the IPD in UV units).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    float parallax_units(float depth) {
        float c = clamp(u_convergence, 0.0, 1.0);
        float lin = -2.0 * (depth - c);
        // Legacy (pre-calibration) shaping: non-linear in depth, zero plane at
        // ~0.415 when Convergence = 0.5, which boosts the near-mid ground.
        float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
        return mix(lin, leg, u_parallax_curve) * u_parallax;
    }

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth, float eye_dir) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        // depth consistency: the sample landed on another surface
        return abs(center_depth - texture(tex_depth, shifted_uv).r) > 0.08;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv, float eye_dir) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        // Hole fill radius tracks the parallax budget: a depth edge opens a gap
        // as wide as (2 x budget), so a fixed 12px search left anything
        // bigger unfilled (the "torn" streaks between near and far surfaces).
        float _budget_px = 2.0 * u_parallax * abs(u_eye_offset) * u_viewport.z;
        int search_range = int(clamp(_budget_px * 1.2, 12.0, 48.0));
        int search_dir = eye_dir > 0.0 ? -1 : 1;

        for (int i = 1; i <= search_range; i++) {
            float c = cos(u_roll);
            float s = sin(u_roll);
            vec2 sample_uv = uv_coord + vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                vec4 sample_color = sample_color(sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                float c = cos(u_roll);
                float s = sin(u_roll);
                vec2 sample_uv = uv_coord - vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                    vec4 sample_color = sample_color(sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (abs(vert_depth_inv - center_depth_inv) < u_depth_tolerance * 2.0) {
                        float w = 0.25;
                        vert_accum += sample_color(vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            return vert_accum / vert_weight;
        }
        return sample_color(uv_coord);
    }

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float c = cos(u_roll);
        float s = sin(u_roll);
        // DIBR asymmetric depth smoothing along parallax direction
        // 3-tap 深度平滑已移除：它沿视差方向糊 ±1.5 texel（224 纹理上
        // ≈ 屏幕 ±13px），把深度边界摊平，是'脸被拉裂/糊开'的主要原因。
        float d0 = texture(tex_depth, flipped_uv).r;
        float depth = d0;
        float depth_inv = -depth;
        // Background metric for push_pull_inpaint(): the function compares a
        // sample against (1 - depth) to decide "farther than the centre".
        // Passing depth_inv (-depth) there compared two different spaces and
        // made the test almost always true (every sample accepted -> smear).
        float depth_bg = 1.0 - depth;
        float shift_amount = parallax_units(depth) * u_depth_strength;
        // Edge-aware border constraint: both eyes rendered in one pass
        // u_edge_falloff=0（关）时 smoothstep(0,0,x) 是 GLSL 未定义行为
        //（NVIDIA 返回 0 → 全像素视差乘零 = 3D 灭）——显式短路为 1.0
        float edge_falloff = 1.0;
        if (u_edge_falloff > 0.0001) {
            edge_falloff = smoothstep(0.0, u_edge_falloff, flipped_uv.x)
                         * smoothstep(1.0, 1.0 - u_edge_falloff, flipped_uv.x);
        }
        shift_amount *= edge_falloff;

        vec2 left_uv  = flipped_uv + vec2(u_eye_offset * shift_amount * c, u_eye_offset * shift_amount * s);
        vec2 right_uv = flipped_uv - vec2(u_eye_offset * shift_amount * c, u_eye_offset * shift_amount * s);

        vec4 left_color, right_color;
        if (u_depth_strength > 0.0 && is_disoccluded(flipped_uv, left_uv, depth, -1.0))
            left_color = push_pull_inpaint(flipped_uv, depth_bg, -1.0);
        else
            left_color = sample_color(left_uv);

        if (u_depth_strength > 0.0 && is_disoccluded(flipped_uv, right_uv, depth, 1.0))
            right_color = push_pull_inpaint(flipped_uv, depth_bg, 1.0);
        else
            right_color = sample_color(right_uv);

        frag_color = vec4(left_color.r, right_color.g, right_color.b, 1.0);

        vec2 border = smoothstep(0.0, 0.015, left_uv) * smoothstep(1.0, 0.985, left_uv);
        vec2 border_r = smoothstep(0.0, 0.015, right_uv) * smoothstep(1.0, 0.985, right_uv);
        frag_color.a = min(min(border.x, border.y), min(border_r.x, border_r.y));

        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;

        if (u_feather_enabled == 1) {
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        float corner_r = u_corner_radius;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""

# Row-interleaved stereo shader: eye determined by row parity.
INTERLEAVED_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    vec4 sample_color(vec2 uv){ return texture(tex_color, uv); }  // interop 视图通道本就正确

    uniform sampler2D tex_depth;
    uniform float u_edge_falloff;   // 边缘渐隐宽度（0=关，0.01=旧默认）
    uniform vec2 u_resolution;
    uniform vec2 u_depth_resolution;  // 深度纹理分辨率（S3 起为模型分辨率 224）
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_parallax;
    uniform float u_parallax_curve; // 0 = linear in depth, 1 = legacy non-linear shaping
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    vec2 depth_texel = 1.0 / u_depth_resolution;  // 深度域 tap 的步长
    float eye_dir;

    // Signed shift multiplier for u_eye_offset (half the IPD in UV units).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    float parallax_units(float depth) {
        float c = clamp(u_convergence, 0.0, 1.0);
        float lin = -2.0 * (depth - c);
        // Legacy (pre-calibration) shaping: non-linear in depth, zero plane at
        // ~0.415 when Convergence = 0.5, which boosts the near-mid ground.
        float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
        return mix(lin, leg, u_parallax_curve) * u_parallax;
    }

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        vec2 grad_dir = vec2(cos(u_roll), sin(u_roll)) * eye_dir;
        float d_left  = texture(tex_depth, base_uv - grad_dir * depth_texel * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * depth_texel * 2.0).r;
        if (abs(d_left - d_right) > 0.08)
            return true;
        return false;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        // Hole fill radius tracks the parallax budget: a depth edge opens a gap
        // as wide as (2 x budget), so a fixed 12px search left anything
        // bigger unfilled (the "torn" streaks between near and far surfaces).
        float _budget_px = 2.0 * u_parallax * abs(u_eye_offset) * u_viewport.z;
        int search_range = int(clamp(_budget_px * 1.2, 12.0, 48.0));
        vec2 sweep = vec2(cos(u_roll), sin(u_roll)) * eye_dir;

        // Phase 1: Directional sweep (most samples here)
        for (int i = 1; i <= search_range; i++) {
            vec2 sample_uv = uv_coord + sweep * pixel_size.x * float(i);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                vec4 sample_color = sample_color(sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        // Phase 2: opposite sweep fallback if not enough background found
        if (best_weight < 2.0) {
            for (int i = 1; i <= search_range; i++) {
                vec2 sample_uv = uv_coord - sweep * pixel_size.x * float(i);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                    vec4 sample_color = sample_color(sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        // Phase 3: Small vertical blur for smoothness (3 taps)
        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;

            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (abs(vert_depth_inv - center_depth_inv) < u_depth_tolerance * 2.0) {
                        float w = 0.25;
                        vert_accum += sample_color(vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }

            return vert_accum / vert_weight;
        }

        // Fallback: return original pixel if no background found
        return sample_color(uv_coord);
    }

    // Panel row phase: 0 = even rows (counted from the TOP) are the left eye,
    // 1 = swapped. Which parity the polarizer film maps to which eye is a
    // hardware property, so it stays configurable via settings.yaml
    // "Interleave Phase" (no runtime toggle: the viewer window does not
    // reliably hold keyboard focus, so a hotkey here proved useless).
    uniform float u_interleave_phase;

    uniform float u_edge_snap;        // 1 = ParallaxRT-style edge snapping
    uniform float u_edge_snap_gain;   // >1 = 降低阈值，让锯齿轮廓也触发
    uniform float u_relief_strength;  // RGB relief injected into the depth
    uniform float u_warp_iterations;  // 1 = one fixed-point warp refinement
    uniform float u_ic_search;        // 1 = 逆一致性搜索（消跨表面拖拽）
    uniform float u_depth_step_uv;    // 1/depth_res：搜索步长（UV）
    uniform float u_ic_debug;         // 1 = 输出路径染色（调试）
    uniform float u_fill;             // 1 = spatial disocclusion fill

    void main() {
        // Row parity MUST be counted from the top of the panel. gl_FragCoord.y
        // is bottom-up (and would flip parity again for odd window heights),
        // which is exactly the trap documented in the TAB row-split path.
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        int row_idx = int(floor(clamp(flipped_uv.y, 0.0, 0.999999) * u_viewport.w));
        eye_dir = (mod(float(row_idx) + u_interleave_phase, 2.0) < 1.0) ? -1.0 : 1.0;
        float my_offset = eye_dir * u_eye_offset;

        // Precompute parallax direction once (perf win: was recomputed per-loop)
        float c = cos(u_roll);
        float s = sin(u_roll);
        vec2 parallax_dir = vec2(c, s) * eye_dir;

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width
        // 3-tap 深度平滑已移除：它沿视差方向糊 ±1.5 texel（224 纹理上
        // ≈ 屏幕 ±13px），把深度边界摊平，是'脸被拉裂/糊开'的主要原因。
        float d0 = texture(tex_depth, flipped_uv).r;
        float depth = d0;
        float depth_inv = -depth;
        // Background metric for push_pull_inpaint(): the function compares a
        // sample against (1 - depth) to decide "farther than the centre".
        // Passing depth_inv (-depth) there compared two different spaces and
        // made the test almost always true (every sample accepted -> smear).
        float depth_bg = 1.0 - depth;

        // ---- ParallaxRT-style silhouette handling -------------------------
        // 1) Snap the depth onto the background side at a strong depth edge.
        //    The model erodes a person's silhouette, so the rim pixels carry
        //    background-ish depth and sample ACROSS the layer; snapping them
        //    (per eye, hole side first) stops background blocks from being
        //    dragged inside the outline. (ParallaxRT: dibr_renderer_cuda.cu)
        if (u_edge_snap > 0.0) {
            vec2 e_dir = vec2(c, s) * depth_texel * 2.0;
            float dL = texture(tex_depth, flipped_uv - e_dir).r;
            float dR = texture(tex_depth, flipped_uv + e_dir).r;
            float gradL = depth - dL;
            float gradR = depth - dR;
            if (eye_dir < 0.0) {                 // left eye: hole on the -dir side
                float _thr = 0.12 / max(0.05, u_edge_snap_gain);
                if (gradL > _thr)      depth = dL;
                else if (gradR > _thr * 0.85) depth -= 0.25 * gradR;
            } else {                             // right eye: mirrored
                if (gradR > 0.12)      depth = dR;
                else if (gradL > 0.10) depth -= 0.25 * gradL;
            }
            depth = clamp(depth, 0.0, 1.0);
            depth_bg = 1.0 - depth;
        }

        // 2) RGB relief: local image contrast pulls the depth onto the real
        //    edge (ParallaxRT enableRgbDetail / detailStrength).
        if (u_relief_strength > 0.0) {
            const vec3 RELIEF_LW = vec3(0.299, 0.587, 0.114);
            vec2 r_dir = pixel_size * 2.0;
            float lc = dot(sample_color(flipped_uv).rgb, RELIEF_LW);
            float ll = dot(sample_color(flipped_uv - vec2(r_dir.x, 0.0)).rgb, RELIEF_LW);
            float lr = dot(sample_color(flipped_uv + vec2(r_dir.x, 0.0)).rgb, RELIEF_LW);
            float lt = dot(sample_color(flipped_uv - vec2(0.0, r_dir.y)).rgb, RELIEF_LW);
            float lb = dot(sample_color(flipped_uv + vec2(0.0, r_dir.y)).rgb, RELIEF_LW);
            depth = clamp(depth + (lc - 0.25 * (ll + lr + lt + lb)) * u_relief_strength,
                          0.0, 1.0);
            depth_bg = 1.0 - depth;
        }

        // Calibrated parallax: the screen plane sits at u_convergence, with
        // independent near/far budgets (screen-plane-relative mapping).
        float shift = parallax_units(depth);
        // u_edge_falloff=0（关）时 smoothstep(0,0,x) 是 GLSL 未定义行为
        //（NVIDIA 返回 0 → 全像素视差乘零 = 3D 灭）——显式短路为 1.0
        float edge_falloff = 1.0;
        if (u_edge_falloff > 0.0001) {
            edge_falloff = smoothstep(0.0, u_edge_falloff, flipped_uv.x)
                         * smoothstep(1.0, 1.0 - u_edge_falloff, flipped_uv.x);
        }
        float px = my_offset * shift * u_depth_strength * edge_falloff;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // 3) One fixed-point refinement of the sample position: re-read the
        //    depth where we landed and correct the warp (damped 50 % so it
        //    cannot oscillate across an edge).
        if (u_warp_iterations > 0.0) {
            float d_src = texture(tex_depth, clamp(shifted_uv, 0.0, 1.0)).r;
            float px_src = my_offset * parallax_units(d_src) * u_depth_strength * edge_falloff;
            vec2 fixed_uv = flipped_uv - vec2(px_src * c, px_src * s);
            shifted_uv = mix(shifted_uv, fixed_uv, 0.5);
        }

        // ---- 逆一致性搜索（方案 C）--------------------------------------
        // 只接受"它自己的位移正好落到当前像素"的源样本；这样跨表面的错误
        // 样本（背景被拖进近景 / 近景被拖进背景）会被一致性检验拒绝。
        // 仅当前反向样本跨了表面时才进入循环，普通像素不付代价。
        float d_self = texture(tex_depth, clamp(shifted_uv, 0.0, 1.0)).r;
        if (u_ic_search > 0.5 && u_depth_strength > 0.0
                && abs(d_self - depth) > 0.08) {
            float best_d = -1.0;
            vec2 best_uv = shifted_uv;
            float tol_uv = u_depth_step_uv * 1.5;
            for (int i = -16; i <= 16; i++) {
                vec2 uv_s = flipped_uv + vec2(float(i) * u_depth_step_uv * c,
                                              float(i) * u_depth_step_uv * s);
                float ds = texture(tex_depth, clamp(uv_s, 0.0, 1.0)).r;
                float px_s = my_offset * parallax_units(ds) * u_depth_strength * edge_falloff;
                vec2 dest = uv_s - vec2(px_s * c, px_s * s);
                if (abs(dest.x - flipped_uv.x) < tol_uv
                        && abs(dest.y - flipped_uv.y) < tol_uv
                        && ds > best_d) {
                    best_d = ds;
                    best_uv = uv_s;
                }
            }
            if (u_ic_debug > 0.5) {
                // 调试：红=命中一致性样本，蓝=仍有遮挡(回退)，绿=未触发搜索
                frag_color = (best_d >= 0.0) ? vec4(1.0, 0.0, 0.0, 1.0)
                                             : vec4(0.0, 0.0, 1.0, 1.0);
                return;
            }
            if (u_ic_debug > 0.5) {
                frag_color = (best_d >= 0.0) ? vec4(1.0, 0.0, 0.0, 1.0)
                                             : vec4(0.0, 0.0, 1.0, 1.0);
                return;
            }
            if (best_d >= 0.0) {
                shifted_uv = best_uv;      // 找到正确的源
            } else {
                // 真空洞（没有任何源落在这里）：朝背景侧外推，
                // 避免把前景颜色拉伸进来（黑色回退已按用户要求撤销）
                float dir_s = (px > 0.0) ? -1.0 : 1.0;
                float back = abs(2.0 * u_parallax * u_eye_offset) + 0.002;
                shifted_uv = flipped_uv + vec2(dir_s * back * c, dir_s * back * s);
            }
        }

        // Soft disocclusion: depth CONSISTENCY of the fetched sample (see the
        // FRAGMENT shader note) - flags the whole revealed band, not just the
        // 2 px seam where the depth gradient is steep.
        float conf = 0.0;
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0) {
            conf = 1.0;
        } else {
            conf = smoothstep(0.03, 0.10,
                              abs(depth - texture(tex_depth, shifted_uv).r));
        }

        vec4 color;
        if (u_fill > 0.0 && conf > 0.001 && u_depth_strength > 0.0) {
            color = push_pull_inpaint(flipped_uv, depth_bg);
        } else {
            color = sample_color(shifted_uv);
        }

        // Screen-edge alpha clip: keep the out-of-bounds safety net (alpha -> 0
        // if parallax somehow over-shoots into negative UV) but use a
        // sub-pixel fade band so the user does not see a visible soft border
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {
            vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF
        float corner_r = u_corner_radius;
        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""

# Interleaved-V: column-interleaved stereo (alternating columns per eye).
VERTICAL_INTERLEAVED_FRAGMENT = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    vec4 sample_color(vec2 uv){ return texture(tex_color, uv); }  // interop 视图通道本就正确

    uniform sampler2D tex_depth;
    uniform float u_edge_falloff;   // 边缘渐隐宽度（0=关，0.01=旧默认）
    uniform vec2 u_resolution;
    uniform vec2 u_depth_resolution;  // 深度纹理分辨率（S3 起为模型分辨率 224）
    uniform float u_eye_offset;
    uniform float u_depth_strength;
    uniform float u_convergence;
    uniform float u_parallax;
    uniform float u_parallax_curve; // 0 = linear in depth, 1 = legacy non-linear shaping
    uniform float u_roll;
    uniform float u_search_radius = 12.0;
    uniform float u_depth_tolerance = 0.012;
    uniform float u_blur_radius = 2.5;
    uniform int u_feather_enabled;
    uniform float u_feather_width;
    uniform vec4 u_viewport;
    uniform float u_corner_radius = 0.0;

    vec2 pixel_size = 1.0 / u_resolution;
    vec2 depth_texel = 1.0 / u_depth_resolution;  // 深度域 tap 的步长
    float eye_dir;

    // Signed shift multiplier for u_eye_offset (half the IPD in UV units).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    // Linear in the depth map: shift = -2 * u_parallax * (depth - plane).
    // Negative above the plane (near -> crossed disparity, in front of the
    // screen), positive below it (far -> behind the screen).
    float parallax_units(float depth) {
        float c = clamp(u_convergence, 0.0, 1.0);
        float lin = -2.0 * (depth - c);
        // Legacy (pre-calibration) shaping: non-linear in depth, zero plane at
        // ~0.415 when Convergence = 0.5, which boosts the near-mid ground.
        float leg = 2.0 * (-depth * (1.0 + 0.35 * (1.0 - depth)) + c);
        return mix(lin, leg, u_parallax_curve) * u_parallax;
    }

    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;
        vec2 grad_dir = vec2(cos(u_roll), sin(u_roll)) * eye_dir;
        float d_left  = texture(tex_depth, base_uv - grad_dir * depth_texel * 2.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * depth_texel * 2.0).r;
        if (abs(d_left - d_right) > 0.08)
            return true;
        return false;
    }

    vec4 push_pull_inpaint(vec2 uv_coord, float center_depth_inv) {
        vec4 best_color = vec4(0.0);
        float best_weight = 0.0;
        // Hole fill radius tracks the parallax budget: a depth edge opens a gap
        // as wide as (2 x budget), so a fixed 12px search left anything
        // bigger unfilled (the "torn" streaks between near and far surfaces).
        float _budget_px = 2.0 * u_parallax * abs(u_eye_offset) * u_viewport.z;
        int search_range = int(clamp(_budget_px * 1.2, 12.0, 48.0));
        int search_dir = eye_dir > 0.0 ? -1 : 1;

        for (int i = 1; i <= search_range; i++) {
            float c = cos(u_roll);
            float s = sin(u_roll);
            vec2 sample_uv = uv_coord + vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
            if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
            float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
            if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                vec4 sample_color = sample_color(sample_uv);
                float dist_weight = exp(-float(i) * 0.15);
                float depth_weight = 1.0 + (sample_depth_inv - center_depth_inv) * 10.0;
                float w = dist_weight * depth_weight;
                best_color += sample_color * w;
                best_weight += w;
                if (best_weight > 5.0) break;
            }
        }

        if (best_weight < 2.0) {
            int opposite_range = search_range;
            for (int i = 1; i <= opposite_range; i++) {
                float c = cos(u_roll);
                float s = sin(u_roll);
                vec2 sample_uv = uv_coord - vec2(search_dir * i * pixel_size.x * c, search_dir * i * pixel_size.x * s);
                if (sample_uv.x < 0.0 || sample_uv.y < 0.0 || sample_uv.x > 1.0 || sample_uv.y > 1.0) continue;
                float sample_depth_inv = 1.0 - texture(tex_depth, sample_uv).r;
                if (abs(sample_depth_inv - center_depth_inv) < u_depth_tolerance * 3.0) {
                    vec4 sample_color = sample_color(sample_uv);
                    float w = exp(-float(i) * 0.2);
                    best_color += sample_color * w;
                    best_weight += w;
                }
            }
        }

        if (best_weight > 0.01) {
            vec4 blurred = best_color / best_weight;
            vec4 vert_accum = blurred * 0.5;
            float vert_weight = 0.5;
            for (int dy = -1; dy <= 1; dy += 2) {
                vec2 vert_uv = uv_coord + vec2(0.0, dy * pixel_size.y * u_blur_radius);
                if (vert_uv.y >= 0.0 && vert_uv.y <= 1.0) {
                    float vert_depth_inv = 1.0 - texture(tex_depth, vert_uv).r;
                    if (abs(vert_depth_inv - center_depth_inv) < u_depth_tolerance * 2.0) {
                        float w = 0.25;
                        vert_accum += sample_color(vert_uv) * w;
                        vert_weight += w;
                    }
                }
            }
            return vert_accum / vert_weight;
        }
        return sample_color(uv_coord);
    }

    uniform float u_edge_snap;        // 1 = ParallaxRT-style edge snapping
    uniform float u_edge_snap_gain;   // >1 = 降低阈值，让锯齿轮廓也触发
    uniform float u_relief_strength;  // RGB relief injected into the depth
    uniform float u_warp_iterations;  // 1 = one fixed-point warp refinement
    uniform float u_fill;             // 1 = spatial disocclusion fill
    uniform float u_ic_search;        // 1 = 逆一致性搜索
    uniform float u_depth_step_uv;    // 1/depth_res

    void main() {
        // Interleaved-V: alternate columns by X coordinate (odd cols = left eye, even cols = right eye)
        eye_dir = (int(mod(gl_FragCoord.x, 2.0)) == 0) ? -1.0 : 1.0;
        float my_offset = eye_dir * u_eye_offset;

        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // DIBR asymmetric depth smoothing: 3-tap Gaussian along parallax dir
        // Per Fehn 2004, smooths sharp depth edges to narrow disocclusion width
        float c = cos(u_roll);
        float s = sin(u_roll);
        vec2 parallax_dir = vec2(c, s) * eye_dir;
        // 3-tap 深度平滑已移除：它沿视差方向糊 ±1.5 texel（224 纹理上
        // ≈ 屏幕 ±13px），把深度边界摊平，是'脸被拉裂/糊开'的主要原因。
        float d0 = texture(tex_depth, flipped_uv).r;
        float depth = d0;
        float depth_inv = -depth;
        // Background metric for push_pull_inpaint(): the function compares a
        // sample against (1 - depth) to decide "farther than the centre".
        // Passing depth_inv (-depth) there compared two different spaces and
        // made the test almost always true (every sample accepted -> smear).
        float depth_bg = 1.0 - depth;

        // ---- ParallaxRT-style silhouette handling -------------------------
        // 1) Snap the depth onto the background side at a strong depth edge.
        //    The model erodes a person's silhouette, so the rim pixels carry
        //    background-ish depth and sample ACROSS the layer; snapping them
        //    (per eye, hole side first) stops background blocks from being
        //    dragged inside the outline. (ParallaxRT: dibr_renderer_cuda.cu)
        if (u_edge_snap > 0.0) {
            vec2 e_dir = vec2(c, s) * depth_texel * 2.0;
            float dL = texture(tex_depth, flipped_uv - e_dir).r;
            float dR = texture(tex_depth, flipped_uv + e_dir).r;
            float gradL = depth - dL;
            float gradR = depth - dR;
            if (eye_dir < 0.0) {                 // left eye: hole on the -dir side
                float _thr = 0.12 / max(0.05, u_edge_snap_gain);
                if (gradL > _thr)      depth = dL;
                else if (gradR > _thr * 0.85) depth -= 0.25 * gradR;
            } else {                             // right eye: mirrored
                if (gradR > 0.12)      depth = dR;
                else if (gradL > 0.10) depth -= 0.25 * gradL;
            }
            depth = clamp(depth, 0.0, 1.0);
            depth_bg = 1.0 - depth;
        }

        // 2) RGB relief: local image contrast pulls the depth onto the real
        //    edge (ParallaxRT enableRgbDetail / detailStrength).
        if (u_relief_strength > 0.0) {
            const vec3 RELIEF_LW = vec3(0.299, 0.587, 0.114);
            vec2 r_dir = pixel_size * 2.0;
            float lc = dot(sample_color(flipped_uv).rgb, RELIEF_LW);
            float ll = dot(sample_color(flipped_uv - vec2(r_dir.x, 0.0)).rgb, RELIEF_LW);
            float lr = dot(sample_color(flipped_uv + vec2(r_dir.x, 0.0)).rgb, RELIEF_LW);
            float lt = dot(sample_color(flipped_uv - vec2(0.0, r_dir.y)).rgb, RELIEF_LW);
            float lb = dot(sample_color(flipped_uv + vec2(0.0, r_dir.y)).rgb, RELIEF_LW);
            depth = clamp(depth + (lc - 0.25 * (ll + lr + lt + lb)) * u_relief_strength,
                          0.0, 1.0);
            depth_bg = 1.0 - depth;
        }

        // Calibrated parallax: the screen plane sits at u_convergence, with
        // independent near/far budgets (screen-plane-relative mapping).
        float shift = parallax_units(depth);
        // u_edge_falloff=0（关）时 smoothstep(0,0,x) 是 GLSL 未定义行为
        //（NVIDIA 返回 0 → 全像素视差乘零 = 3D 灭）——显式短路为 1.0
        float edge_falloff = 1.0;
        if (u_edge_falloff > 0.0001) {
            edge_falloff = smoothstep(0.0, u_edge_falloff, flipped_uv.x)
                         * smoothstep(1.0, 1.0 - u_edge_falloff, flipped_uv.x);
        }
        float px = my_offset * shift * u_depth_strength * edge_falloff;
        vec2 shifted_uv = flipped_uv - vec2(px * c, px * s);

        // 3) One fixed-point refinement of the sample position: re-read the
        //    depth where we landed and correct the warp (damped 50 % so it
        //    cannot oscillate across an edge).
        if (u_warp_iterations > 0.0) {
            float d_src = texture(tex_depth, clamp(shifted_uv, 0.0, 1.0)).r;
            float px_src = my_offset * parallax_units(d_src) * u_depth_strength * edge_falloff;
            vec2 fixed_uv = flipped_uv - vec2(px_src * c, px_src * s);
            shifted_uv = mix(shifted_uv, fixed_uv, 0.5);
        }

        // Soft disocclusion confidence (smooth ramp, hides hard seams)
        float conf = 0.0;
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0) {
            conf = 1.0;
        } else {
            vec2 step2 = parallax_dir * depth_texel * 2.0;
            float d_left  = texture(tex_depth, flipped_uv - step2).r;
            float d_right = texture(tex_depth, flipped_uv + step2).r;
            float jump = abs(d_left - d_right);
            conf = smoothstep(0.06, 0.12, jump);
        }

        vec4 color;
        if (u_fill > 0.0 && conf > 0.001 && u_depth_strength > 0.0) {
            color = push_pull_inpaint(flipped_uv, depth_bg);
        } else {
            color = sample_color(shifted_uv);
        }

        // Screen-edge alpha clip
        vec2 border = smoothstep(-0.001, 0.001, shifted_uv) * smoothstep(1.001, 0.999, shifted_uv);
        color.a = min(border.x, border.y);
        frag_color = color;

        // Natural edge feathering
        if (u_feather_enabled == 1) {
            vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
            float left   = fuv.x;
            float right  = 1.0 - fuv.x;
            float top    = fuv.y;
            float bottom = 1.0 - fuv.y;
            float feather = u_feather_width;
            float fadeL = smoothstep(0.0, feather, left);
            float fadeR = smoothstep(0.0, feather, right);
            float fadeT = smoothstep(0.0, feather, top);
            float fadeB = smoothstep(0.0, feather, bottom);
            float falloff = fadeL * fadeR * fadeT * fadeB;
            falloff = pow(falloff, 0.7);
            frag_color.rgb *= falloff;
        }

        // Rounded corners via 2D SDF
        float corner_r = u_corner_radius;
        vec2 fuv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
        vec2 d = abs(fuv - 0.5) - 0.5 + corner_r;
        float corner_sdf = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - corner_r;
        float corner_alpha = 1.0 - smoothstep(0.0, 0.01, corner_sdf);
        frag_color.a = min(frag_color.a, corner_alpha);
    }
"""
