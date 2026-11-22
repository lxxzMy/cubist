# -*- coding: utf-8 -*-
"""MC-EMA 时域块（运动补偿时域平滑，产品化 2026-11-18；2026-11-20 自
viewer.py 解耦为独立时域模块）。

块内流水：亮度半采样 → 块匹配(SAD±4) → uniq 门种子 → 深度屏障泛洪 ×12 →
判切（trust=连续性<0.60）→ 运动补偿 EMA pass（pred = state(u−du, v−dv)，
局部暴露阀 |z−old|>0.3，切帧整幅直通）。

着色器常量与 mc_flood_grid 放模块级：tools/mc_gl_test.py 等离线对拍 import
同一份（同码同源，防两端漂移）。离线证据链：tools/mc_seam_test.py /
mc_object_test.py / mc_valve_test.py / mc_halfcut_test.py / mc_trust_noise_test.py。

class MC 宿主属性面（外部契约，tools/mc_repro.py 的 FakeWin 依赖）：
只读 ctx / color_tex / _ema_tex / _pp_res / _gpu_res；
宿主旗标 _mc_on（main.py 总开关）与 _mc_cut_now（main.py 每帧写，ema_pass 读）。
内部 _mc_* 资源全部惰性建在本对象上，生命周期与宿主 GL 上下文一致。
"""
import moderngl
import numpy as np

MC_QUAD_VS = """
    #version 330
    in vec2 in_pos;
    void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# 色彩 → 224 亮度（4-tap box 降采样；块匹配的输入域）
MC_LUM_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_color;
    uniform float res;
    void main(){
        vec2 uv = gl_FragCoord.xy / res;
        vec2 hp = 0.5 / vec2(textureSize(tex_color, 0));
        float l = 0.0;
        for (int dy = -1; dy <= 1; dy += 2)
            for (int dx = -1; dx <= 1; dx += 2) {
                vec3 c = texture(tex_color, uv + vec2(float(dx), float(dy)) * hp).rgb;
                l += dot(c, vec3(0.299, 0.587, 0.114));
            }
        out_c = vec4(l * 0.25, 0.0, 0.0, 1.0);
    }
"""

# 块匹配（28×28 网格，每片段一块）：SAD ±8 搜索 + 次优 SAD（平谷检测）。
# 输出 (du, dv, best_sad, sad2)，du=−dx（位移约定，离线实验教训：符号反了
# 全场补偿反向）。注意 ±8 是 224 尺度的像素；448 档下等效覆盖减半（待实测）。
MC_MATCH_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_prev, tex_cur;
    uniform float res;      // 采样域分辨率（半分辨率匹配时=112）
    uniform float block;    // 块边长（半分辨率=4）
    uniform float search;   // 搜索半径（半分辨率=4，等效全幅±8）
    void main(){
        float g = res / block;
        float bx = floor(gl_FragCoord.x - 0.5) * block + block * 0.5;
        float by = floor(gl_FragCoord.y - 0.5) * block + block * 0.5;
        int S = int(search);
        float best = 1e9, second = 1e9;
        int bdx = 0, bdy = 0;
        int hb = int(block * 0.5);
        float area = block * block;
        for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) {
            float sad = 0.0;
            for (int wy = -hb; wy < hb; ++wy)
            for (int wx = -hb; wx < hb; ++wx) {
                vec2 cc = vec2(bx + float(wx), by + float(wy));
                vec2 pc = clamp(cc + vec2(float(dx), float(dy)),
                                vec2(0.5), vec2(res - 0.5));
                sad += abs(texture(tex_cur, cc / res).r
                           - texture(tex_prev, pc / res).r);
            }
            sad /= area;
            if (sad < best) { second = best; best = sad; bdx = dx; bdy = dy; }
            else if (sad < second) { second = sad; }
        }
        out_c = vec4(float(-bdx), float(-bdy), best, second);
    }
"""

# 切镜头统计（1×1 目标）：|z_norm − 状态| > 0.35 的像素占比（面积占比主判据）
MC_CUT_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_lum_prev, tex_lum_cur;   // 半分辨率亮度乒乓
    uniform float res;                             // = half
    // 双信号单 pass：x = 32-bin 直方图 L1/n（运动不变）；y = |Δlum|>0.25
    // 面积占比（快运动与内容互换都会大——由 CPU 侧结合信任坍缩区分）。
    void main(){
        float h0[32]; float h1[32];
        for (int i = 0; i < 32; ++i) { h0[i] = 0.0; h1[i] = 0.0; }
        float n = res * res;
        float area = 0.0;
        for (int y = 0; y < 10000; ++y) {
            if (float(y) >= res) break;
            for (int x = 0; x < 10000; ++x) {
                if (float(x) >= res) break;
                vec2 uv = (vec2(float(x), float(y)) + 0.5) / res;
                float p = texture(tex_lum_prev, uv).r;
                float c = texture(tex_lum_cur, uv).r;
                int b0 = int(clamp(p, 0.0, 0.9999) * 32.0);
                int b1 = int(clamp(c, 0.0, 0.9999) * 32.0);
                h0[b0] += 1.0;
                h1[b1] += 1.0;
                area += step(0.25, abs(c - p));
            }
        }
        float l1 = 0.0;
        for (int i = 0; i < 32; ++i) l1 += abs(h0[i] - h1[i]);
        out_c = vec4(l1 / n, area / n, 0.0, 1.0);
    }
"""

# 原始信任坍缩检测（1×1）：match 网格上 ok 块占比——快运动匹配成功（高），
# 内容互换匹配崩塌（低）。泛洪前的原始值（泛洪会"治好"它，那正是要检测的）。
MC_TRUST_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_match;
    void main(){
        ivec2 sz = textureSize(tex_match, 0);
        float acc = 0.0;
        for (int y = 0; y < 10000; ++y) {
            if (y >= sz.y) break;
            for (int x = 0; x < 10000; ++x) {
                if (x >= sz.x) break;
                vec4 m = texelFetch(tex_match, ivec2(x, y), 0);
                // 判切信任 = 内容连续性（存在 SAD≤0.05 匹配的块占比），与 uniq 解耦。
                // 旧门 (≤0.005 直通 | ≤0.05 且 uniq≥1.3) 有裂缝：平坦+DDA噪声内容
                // conf≈0.007 落进 0.005<conf≤0.05 且 uniq≈1 的缝里被整片误拒 →
                // 近静止帧 trust 0.10-0.51 → 假切 α=1 整幅直通 → 画面突变
                // （app.log 实锤 pixArea=0.003 而 trust=0.369；
                //   tools/mc_trust_noise_test.py 复现 old=0.126 new=1.000）。
                // uniq<1.3 只说明【运动】不确定（块匹配位置歧义，交给泛洪
                // 深度屏障继承），不是内容互换的证据——切与运动场两问分开答。
                acc += (m.z <= 0.05) ? 1.0 : 0.0;
            }
        }
        out_c = vec4(acc / float(sz.x * sz.y), 0.0, 0.0, 1.0);
    }
"""

# 补偿 EMA（替代 _pp_prog 的 EMA 步，AA 已在前置 gauss pass 完成）：
#   pred = 状态(u−du, v−dv)；trust=0 退同像素；局部暴露阀；切镜头整幅直通。
MC_EMA_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_raw, tex_prev, tex_motion;
    uniform float res, alpha;
    uniform int mc_cut;
    void main(){
        vec2 uv = gl_FragCoord.xy / res;
        float z = texture(tex_raw, uv).r;
        if (mc_cut == 1) { out_c = vec4(z, 0.0, 0.0, 1.0); return; }
        vec4 m = texture(tex_motion, uv);       // 双线性上采样的 (du,dv,trust)
        float pred = texture(tex_prev, uv - m.xy / res).r;
        float plain = texture(tex_prev, uv).r;
        float trusted = step(0.5, m.z);
        float old = mix(plain, pred, trusted);
        float a = alpha;
        // 局部暴露阀 v2：对【实际混合源】比（可信比 pred / 不可信比 plain）。
        // v1 只在可信时比 pred——半切换实验实锤：恰是最需重置的不可信区被
        // 豁免，退回普通 EMA 混旧场景状态。
        if (abs(z - old) > 0.3) a = 1.0;
        out_c = vec4(a * z + (1.0 - a) * old, 0.0, 0.0, 1.0);
    }
"""


# 种子 pass：匹配网格 + 当前深度 → (du, dv, ok, 块中位深度)。
# ok = uniq 平谷门；块中位用 8x8 插入排序（中位不能换均值——均值会把物体
# 边界块抹成中间深度，恰在最需屏障处失效）。
MC_SEED_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_match, tex_raw;
    uniform float res, lo, hi;
    void main(){
        ivec2 gc = ivec2(gl_FragCoord.xy - 0.5);
        vec4 m = texelFetch(tex_match, gc, 0);
        float uniq = m.w / max(m.z, 1e-6);
        bool ok = (m.z <= 0.05) && (uniq >= 1.3);
        float arr[64];
        int n = 0;
        for (int wy = 0; wy < 8; ++wy)
        for (int wx = 0; wx < 8; ++wx) {
            vec2 uv = (vec2(float(gc.x * 8 + wx), float(gc.y * 8 + wy)) + 0.5) / res;
            float z = clamp((texture(tex_raw, uv).r - lo) / max(1e-6, hi - lo),
                            0.0, 1.0);
            arr[n++] = z;
        }
        for (int i = 1; i < 64; ++i) {
            float v = arr[i]; int j = i - 1;
            while (j >= 0 && arr[j] > v) { arr[j + 1] = arr[j]; --j; }
            arr[j + 1] = v;
        }
        out_c = vec4(m.x * 2.0, m.y * 2.0, ok ? 1.0 : 0.0, arr[31]);  // 半幅px→全幅px
    }
"""

# 泛洪 pass（28x28 ping-pong）：失败块 <- 同深度侧 ok 邻居的下中位。
# 12 遍覆盖 12 块半径传播（环场景内部 7 块；更大平坦物体时可调 passes）。
MC_FLOOD_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_flood;
    uniform float dthr;
    void main(){
        ivec2 gc = ivec2(gl_FragCoord.xy - 0.5);
        vec4 st = texelFetch(tex_flood, gc, 0);
        if (st.z > 0.5) { out_c = st; return; }
        float us[8]; float vs[8]; int n = 0;
        ivec2 sz = textureSize(tex_flood, 0);
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            ivec2 nc = gc + ivec2(dx, dy);
            if (nc.x < 0 || nc.y < 0 || nc.x >= sz.x || nc.y >= sz.y) continue;
            vec4 nb = texelFetch(tex_flood, nc, 0);
            if (nb.z > 0.5 && abs(nb.w - st.w) <= dthr) {
                us[n] = nb.x; vs[n] = nb.y; ++n;
            }
        }
        if (n == 0) { out_c = st; return; }
        for (int i = 1; i < n; ++i) {
            float v = us[i]; int j = i - 1;
            while (j >= 0 && us[j] > v) { us[j + 1] = us[j]; --j; }
            us[j + 1] = v;
        }
        for (int i = 1; i < n; ++i) {
            float v = vs[i]; int j = i - 1;
            while (j >= 0 && vs[j] > v) { vs[j + 1] = vs[j]; --j; }
            vs[j + 1] = v;
        }
        int k = (n - 1) / 2;
        out_c = vec4(us[k], vs[k], 1.0, st.w);
    }
"""


def mc_flood_grid(du, dv, conf, sad2, dgrid, passes=28, sad_thr=0.05,
                  uniq_thr=1.3, dthr=0.12):
    """uniq 平谷门 + 深度屏障迭代泛洪（离线定案的 votek，向量化版）。

    输入输出均为 28×28 网格（GL 内存序）。dgrid = 当前归一化深度的块中位
    （深度边屏障：块中位深度差 > dthr 不传染）。返回 (du, dv, ok0)；ok0 =
    原始置信集（继承块须与天然置信块区分信任来源——离线实验教训）。
    每 pass 并行填充（8 邻移位视图 + nanmedian），~30ms 纯 Python 版的
    向量化替代，语义为"同时填充"而非栅格序（更对称，对拍已验证同档）。
    """
    uniq = sad2 / np.maximum(conf, 1e-6)
    # 种子门与 GL MC_SEED_FS 同步：只看 uniq，无 conf≤0.005 直通——直通会给
    # 【运动中的平坦区】钉 (0,0) 错运动；泛洪优先让它继承同深度边缘块的
    # 真运动，泛洪够不着时 EMA 的 trusted=0 分支退回 plain（等效 (0,0)）。
    ok = (conf <= sad_thr) & (uniq >= uniq_thr)
    ok0 = ok.copy()
    du2 = du.astype(np.float32).copy()
    dv2 = dv.astype(np.float32).copy()
    h, w = du.shape
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy, dx) != (0, 0)]

    def shift(a, dy, dx):
        out = np.full_like(a, np.nan)
        ys = slice(max(0, dy), h + min(0, dy))
        xs = slice(max(0, dx), w + min(0, dx))
        yd = slice(max(0, -dy), h + min(0, -dy))
        xd = slice(max(0, -dx), w + min(0, -dx))
        out[ys, xs] = a[yd, xd]
        return out

    dgrid_f = dgrid.astype(np.float32)
    for _ in range(passes):
        if ok.all():
            break
        cu, cv = [], []
        for dy, dx in shifts:
            nb_ok = shift(ok.astype(np.float32), dy, dx)
            nb_d = shift(dgrid_f, dy, dx)
            nb_u = shift(du2, dy, dx)
            nb_v = shift(dv2, dy, dx)
            valid = (nb_ok > 0.5) & (np.abs(nb_d - dgrid_f) <= dthr)
            cu.append(np.where(valid, nb_u, np.nan))
            cv.append(np.where(valid, nb_v, np.nan))
        # 掩码中位：np.nanmedian 在 (8,28,28) 尺度上单次 ~0.4ms 固定开销，
        # 28 pass 全开 ~20ms（对拍窗口细分实锤）——sort 把 NaN 排尾后取
        # 第 k 个即中位（偶数个取下中位，与 nanmedian 的均值中位差可忽略）。
        su, sv = np.sort(np.stack(cu), axis=0), np.sort(np.stack(cv), axis=0)
        cnt = np.stack(cu)
        cnt = np.sum(~np.isnan(cnt), axis=0).astype(np.int32)
        k = np.clip((cnt - 1) >> 1, 0, 7)
        idx = k[None, :, :]
        mu = np.take_along_axis(su, idx, axis=0)[0]
        mv = np.take_along_axis(sv, idx, axis=0)[0]
        mu[cnt == 0] = np.nan
        mv[cnt == 0] = np.nan
        filled = (~ok) & ~np.isnan(mu)
        if not filled.any():
            break
        du2[filled] = mu[filled]
        dv2[filled] = mv[filled]
        ok |= filled
    return du2, dv2, ok0

_tel_n = 0


class MC:
    """运动补偿时域管线（host = StereoWindow 或诊断替身 FakeWin）。"""

    def __init__(self, host):
        self.host = host
        self._mc_res = None

    @property
    def on(self):
        return getattr(self.host, "_mc_on", True)

    def ready(self, pp_res):
        """postprocess EMA 分支门（原 getattr(self,'_mc_res')==_pp_res 语义）。"""
        return self._mc_res is not None and self._mc_res == pp_res

    def init(self, res: int):
        if getattr(self, "_mc_res", None) == res:
            return
        self._mc_res = res
        half = res // 2
        self._mc_half = half
        g = half // 4                     # 半分辨率 4×4 块 = 全幅 8×8 同尺度
        self._mc_g = g
        # 亮度乒乓（R8，NEAREST：块匹配要精确 texel）
        self._mc_lum = []
        for _ in range(2):
            t = self.host.ctx.texture((half, half), 1, dtype='f1')
            t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._mc_lum.append((t, self.host.ctx.framebuffer([t])))
        self._mc_lum_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_LUM_FS)
        self._mc_match_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_MATCH_FS)
        self._mc_cut_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_CUT_FS)
        self._mc_ema_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_EMA_FS)
        self._mc_seed_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_SEED_FS)
        self._mc_flood_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_FLOOD_FS)
        self._mc_trust_prog = self.host.ctx.program(
            vertex_shader=MC_QUAD_VS, fragment_shader=MC_TRUST_FS)
        _quad = self.host.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32))
        self._mc_match_prog['block'].value = 4.0
        self._mc_match_prog['search'].value = 4.0
        self._mc_match_prog['res'].value = float(half)   # 匹配在半分辨率域
        self._mc_lum_prog['res'].value = float(half)     # 亮度也渲到 half
        self._mc_cut_prog['res'].value = float(half)   # 色度判切在半幅亮度域
        self._mc_ema_prog['res'].value = float(res)
        self._mc_seed_prog['res'].value = float(res)
        self._mc_flood_prog['dthr'].value = 0.12
        self._mc_vao = {p: self.host.ctx.vertex_array(p, [(_quad, "2f", "in_pos")])
                        for p in (self._mc_lum_prog, self._mc_match_prog,
                                  self._mc_cut_prog, self._mc_ema_prog,
                                  self._mc_seed_prog, self._mc_flood_prog,
                                  self._mc_trust_prog)}
        # 网格目标（RGBA32F: du,dv,best,sad2）+ 切镜头统计（1×1）+ 运动场（上传）
        self._mc_grid_tex = self.host.ctx.texture((g, g), 4, dtype='f4')
        self._mc_grid_fbo = self.host.ctx.framebuffer([self._mc_grid_tex])
        self._mc_cut_tex = self.host.ctx.texture((1, 1), 4, dtype='f4')
        self._mc_cut_fbo = self.host.ctx.framebuffer([self._mc_cut_tex])
        self._mc_trust_tex = self.host.ctx.texture((1, 1), 4, dtype='f4')
        self._mc_trust_fbo = self.host.ctx.framebuffer([self._mc_trust_tex])
        # 泛洪乒乓（RGBA32F: du,dv,ok,块中位深度）——末态直接当运动场纹理
        self._mc_flood = []
        for _ in range(2):
            t = self.host.ctx.texture((g, g), 4, dtype='f4')
            t.repeat_x = t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._mc_flood.append((t, self.host.ctx.framebuffer([t])))
        # 泛洪纹理必须零初始化（trust 通道=0 → plain EMA）：真机上 MC 未跑
        # 时 EMA 仍按此场采样，垃圾 float 会持续扭曲静止画面（用户实报根因）
        for t, _f in self._mc_flood:
            t.write(np.zeros((g, g, 4), np.float32).tobytes())
        self._mc_field_tex = self._mc_flood[0][0]
        self._mc_lum_idx = 0
        self._mc_prev_glo = -1
        self._mc_stats = {"gpu": 0.0, "n": 0}
        self._mc_flood_idx = 0

    def step(self, raw_tex, lo: float, hi: float, z_norm):
        """运动补偿一步（渲染线程调用，postprocess 之前）。返回 (cut, stats)。

        raw_tex/z_norm(lo,hi) = 当前帧原始深度（GPU 纹理 / numpy 归一化副本）；
        z_norm 用于 CPU 侧深度屏障（块中位），GL 内存序。
        """
        import time as _t
        # 首帧守卫：_ema_tex 由 gpu_post_init 首跑创建；未就绪时整步跳过。
        # 注意不得触碰 self._mc_stats（mc_init 未跑时不存在——曾致首帧
        # AttributeError 且把 MC 整链打入死路）。
        if not hasattr(self.host, "_ema_tex"):
            return False, {}
        self.init(self.host._pp_res if hasattr(self.host, "_pp_res") else
                     int(getattr(self.host, "_gpu_res", 224)))
        res, g = self._mc_res, self._mc_g
        t0 = _t.perf_counter()
        color = getattr(self.host, "color_tex", None)
        if color is None:
            return False, {}
        # same_frame 判据已删除：真实零拷贝环的 color_tex.glo 恒定（GL 纹理
        # 对象永生复用），按 glo 判新帧→永远"同帧"→匹配/切阀从未运行（真机
        # 12s 3D 零遥测实锤）。新帧语义由调用方保证（_gpu_depth_step 只在
        # 新帧跑）；重复调用同一帧的后果只是匹配出 (0,0)，无害。
        same_frame = False
        # 1) 亮度乒乓：本帧 → 空位
        cur_idx = 1 - self._mc_lum_idx
        cur_tex, cur_fbo = self._mc_lum[cur_idx]
        if not same_frame:
            color.use(0)
            self._mc_lum_prog['tex_color'].value = 0
            cur_fbo.use()
            self.host.ctx.viewport = (0, 0, self._mc_half, self._mc_half)
            self._mc_vao[self._mc_lum_prog].render(moderngl.TRIANGLE_STRIP)
        prev_tex = self._mc_lum[self._mc_lum_idx][0]
        first = (self._mc_lum_idx == 0 and self._mc_prev_glo == -1)
        cut = False
        if not first and not same_frame:
            # 2) 块匹配网格
            prev_tex.use(0)
            cur_tex.use(1)
            self._mc_match_prog['tex_prev'].value = 0
            self._mc_match_prog['tex_cur'].value = 1
            self._mc_grid_fbo.use()
            self.host.ctx.viewport = (0, 0, g, g)
            self._mc_vao[self._mc_match_prog].render(moderngl.TRIANGLE_STRIP)
            # 3) 种子 pass：匹配网格 + 当前深度 → (du,dv,ok,块中位深度)
            self._mc_grid_tex.use(0)
            raw_tex.use(1)
            self._mc_seed_prog['tex_match'].value = 0
            self._mc_seed_prog['tex_raw'].value = 1
            self._mc_seed_prog['lo'].value = float(lo)
            self._mc_seed_prog['hi'].value = float(hi)
            dst = self._mc_flood[1 - self._mc_flood_idx]
            dst[1].use()
            self.host.ctx.viewport = (0, 0, g, g)
            self._mc_vao[self._mc_seed_prog].render(moderngl.TRIANGLE_STRIP)
            # 4) 泛洪 ping-pong x12（28x28 绘制白送级；覆盖 12 块传播半径）
            #    ——末态纹理即运动场（du,dv 全幅像素×2 已在种子处换算）
            for _k in range(12):
                src = self._mc_flood[1 - self._mc_flood_idx]
                self._mc_flood_idx = 1 - self._mc_flood_idx
                dst = self._mc_flood[1 - self._mc_flood_idx]
                src[0].use(0)
                self._mc_flood_prog['tex_flood'].value = 0
                dst[1].use()
                self.host.ctx.viewport = (0, 0, g, g)
                self._mc_vao[self._mc_flood_prog].render(moderngl.TRIANGLE_STRIP)
            self._mc_field_tex = self._mc_flood[1 - self._mc_flood_idx][0]
            # 5) 切镜头统计（1x1，色度判据）——全 GPU 链唯一 CPU 读回=4字节
            #    色度>深度：同深度分布的切换（对话镜头互切）深度阀失明
            #    （面积0.071<0.15），色度命中（0.271>0.20）
            prev_tex.use(0)
            cur_tex.use(1)
            self._mc_cut_prog['tex_lum_prev'].value = 0
            self._mc_cut_prog['tex_lum_cur'].value = 1
            self._mc_cut_fbo.use()
            self.host.ctx.viewport = (0, 0, 1, 1)
            self._mc_vao[self._mc_cut_prog].render(moderngl.TRIANGLE_STRIP)
            _cs = np.frombuffer(
                self._mc_cut_fbo.read(components=4, dtype='f4'), np.float32)
            self._mc_grid_tex.use(0)
            self._mc_trust_prog['tex_match'].value = 0
            self._mc_trust_fbo.use()
            self.host.ctx.viewport = (0, 0, 1, 1)
            self._mc_vao[self._mc_trust_prog].render(moderngl.TRIANGLE_STRIP)
            _tr = float(np.frombuffer(
                self._mc_trust_fbo.read(components=4, dtype='f4'), np.float32)[0])
            self._mc_stats["trust"] = _tr
            hist_l1, pix_area = float(_cs[0]), float(_cs[1])
            # 判切（2026-11-19 用户定案：只留信任坍缩，一个信号）。
            # trust = 连续性 = SAD≤0.05 的块占比（MC_TRUST_FS，与 uniq 解耦——
            # uniq 裂缝曾把近静止平坦+噪声内容整片误拒成假切/突变，当日修）。
            # trust < 0.60 = 超过 40% 的块在上一帧【任何偏移都找不到低残差
            # 匹配】= 内容互换。静止 trust≈1.0；歧义纹理（周期条纹）≈1.0
            # （有匹配只是不确定匹配谁——泛洪继承，不判切）；弹窗/拖窗≈0.75-0.9
            # （局部崩）；只有真切换才 <0.60。直方图/pix_area 仅遥测留档。
            cut = _tr < 0.60
            # 切阀遥测（标定用）：报三信号实值——真实内容的阈值依据
            global _tel_n
            _tel_n += 1
            _n_mc = _tel_n
            if _n_mc % 120 == 1:
                print("[mc-tel] histL1=%.3f pixArea=%.3f trust=%.3f cut=%s"
                      % (hist_l1, pix_area, _tr, cut), flush=True)
            self._mc_stats['n'] += 1
            self._mc_stats['hist_l1'] = hist_l1
            self._mc_stats['pix_area'] = pix_area
            self._mc_stats['trust'] = _tr
        if not same_frame:
            self._mc_lum_idx = cur_idx
            self._mc_prev_glo = color.glo
        self.host.ctx.screen.use()
        self._mc_stats['gpu'] = (_t.perf_counter() - t0) * 1000
        self._mc_stats['cut'] = cut
        return cut, self._mc_stats

    def ema_pass(self, cur, alpha: float, dst_fbo, res: int):
        """运动补偿 EMA pass（postprocess 链末步，替代 legacy _pp_prog 的 EMA）。

        tex_prev 必须显式绑定到 1 号单元：GLSL sampler 默认值 0，不绑就会
        和 tex_raw 撞在同一个单元上，EMA 退化成"输出=输入"（实测 EMA 无效）。
        """
        host = self.host
        host._ema_tex.use(location=1)
        self._mc_field_tex.use(location=2)
        p = self._mc_ema_prog
        p['tex_raw'].value = 0
        p['tex_prev'].value = 1
        p['tex_motion'].value = 2
        p['alpha'].value = float(alpha)
        p['mc_cut'].value = 1 if getattr(host, "_mc_cut_now", False) else 0
        dst_fbo.use()
        host.ctx.viewport = (0, 0, res, res)
        cur.use(location=0)
        self._mc_vao[p].render(moderngl.TRIANGLE_STRIP)
