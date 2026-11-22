# -*- coding: utf-8 -*-
# 时域中值 W5 着色器（2026-11-20 用户定案，替代纯 EMA 为产品时域级）：
# median(空间链输出_t..t-4)。9-swap 最优 5 元排序网络取中位（120 排列
# 离线全验）。特性：≤2 连续帧离群免疫、重尾噪声最优、斜坡滞后 ~2 帧、
# 切换=保持 2 帧后整幅跳净（无跨场景混合）。D2S_TEMPORAL=ema 回退 legacy。
MEDIAN_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_c, tex_h0, tex_h1, tex_h2, tex_h3;
    uniform float res;
    void main(){
        vec2 uv = gl_FragCoord.xy / res;
        float a = texture(tex_c, uv).r;
        float b = texture(tex_h0, uv).r;
        float c = texture(tex_h1, uv).r;
        float d = texture(tex_h2, uv).r;
        float e = texture(tex_h3, uv).r;
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
        out_c = vec4(c, 0.0, 0.0, 1.0);
    }
"""

"""空间后处理块：归一化(+前景膨胀) → 闭运算 → 坡度限幅 → 分离高斯 AA。

自 viewer.py 解耦（2026-11-20）。pass 图语义：每步只声明"读谁、写谁、参数
多少"，run_pass 统一执行并断言 src != dst（GL 同纹理同时采样+渲染 =
未定义行为，P1 feedback loop 教训）。着色器为块私有内联常量（随行搬运）。
时域 EMA 不在本块：MC 版在 depth_mc.MC.ema_pass，legacy 版在本类
ema_pass_legacy——由 viewer.postprocess_gpu 胶水在空间链之后编排。
"""
import os


import moderngl
import numpy as np


# 切阀残差统计（2026-11-20 复位，MC 停用后的切判据）：量"时域状态失效没"
CUTVALVE_FS = """
    #version 330
    out vec4 out_c;
    uniform sampler2D tex_cur, tex_prev;   // 空间链输出 vs 时域状态
    uniform float res, thr;
    void main(){
        // 残差面积：|cur−状态|>thr 的像素占比 → 1x1（4 字节读回）
        float acc = 0.0;
        for (int y = 0; y < 10000; ++y) {
            if (float(y) >= res) break;
            for (int x = 0; x < 10000; ++x) {
                if (float(x) >= res) break;
                vec2 uv = (vec2(float(x), float(y)) + 0.5) / res;
                acc += step(thr, abs(texture(tex_cur, uv).r
                                     - texture(tex_prev, uv).r));
            }
        }
        out_c = vec4(acc / (res * res), 0.0, 0.0, 1.0);
    }
"""




class DepthPost:
    """宿主属性面（只读）：ctx / depth_tex（最终渲染目标）/ _texture_size。"""

    # 切阀阈值（与 depth.py _ema_apply 同步；ec63e9f 时代离线标定值复位）：
    # 残差 >0.35 的面积占比 >0.20 = 时域状态失效（切换）。快平移面积 ~5-15%
    # 留有分离带；半切换（弹窗 <20% 面积）不触发=交给逐像素暴露行为。
    CUT_RES = 0.35
    CUT_AREA = 0.20

    def __init__(self, host):
        self.host = host
        self._pp_res = None
        # 时域模式（boot 时定死，2026-11-20 用户回摆）：ema = 产品默认；
        # median W5 = opt-in（离群免疫/重尾最优，但静态边缘有沙沙）
        self.temporal_median = (os.environ.get("D2S_TEMPORAL", "ema") == "median")
        self._med_ptr = 0

    @property
    def res(self):
        return self._pp_res

    def take_ema_first(self):
        """取走首帧旗标（True 恰好一次；ensure 重建时重立）。"""
        v = getattr(self, "_ema_first", False)
        self._ema_first = False
        return v

    @property
    def ema_tex(self):
        return self._ema_tex

    @property
    def depth_fbo(self):
        return self._depth_fbo

    @property
    def ema_fbo(self):
        return self._ema_fbo

    def ensure(self, res: int):
        if getattr(self, "_pp_res", None) == res:
            return
        self._pp_res = res
        for name in ("_ema_tex",):
            t = getattr(self, name, None)
            if t is not None:
                t.release()
        # dilate（Depth Grow）：先把被"啃掉"的轮廓向外补回几 texel，
        # 再做 AA —— 否则边缘侵蚀会被高斯直接抹平（耳廓消失那种）。
        self._dil_tex = self.host.ctx.texture((res, res), 1, dtype='f4')
        self._dil_tex.repeat_x = False
        self._dil_tex.repeat_y = False
        self._dil_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._dil_fbo = self.host.ctx.framebuffer([self._dil_tex])
        self._dil_prog = self.host.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex_raw;
                uniform float res, lo, hi, grow;
                void main(){
                    vec2 uv = gl_FragCoord.xy / res;
                    float best = -1.0;
                    for (int dy = -6; dy <= 6; ++dy) {
                        for (int dx = -6; dx <= 6; ++dx) {
                            if (abs(float(dx)) > grow || abs(float(dy)) > grow) continue;
                            best = max(best, texture(tex_raw,
                                uv + vec2(float(dx), float(dy)) / res).r);
                        }
                    }
                    float v = clamp((best - lo) / max(1e-6, hi - lo), 0.0, 1.0);
                    out_c = vec4(v, 0.0, 0.0, 1.0);
                }
            """)
        self._dil_prog['res'].value = float(res)
        # 坡度限幅（slope limiter）：只在 k·|d'| 超阈值处做一次小高斯，
        # 平坦区完全不动 —— 对比"全局 AA"，它能保住前景占比不掉。
        self._sl_tex = [self.host.ctx.texture((res, res), 1, dtype='f4') for _ in range(2)]
        for t in self._sl_tex:
            t.repeat_x = False
            t.repeat_y = False
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._sl_fbo = [self.host.ctx.framebuffer([t]) for t in self._sl_tex]
        self._sl_prog = self.host.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex_d;
                uniform float res, m, target;
                void main(){
                    vec2 uv = gl_FragCoord.xy / res;
                    vec2 e = vec2(1.0 / res, 0.0);
                    float d0 = texture(tex_d, uv).r;
                    float gx = (texture(tex_d, uv + e.xy).r - texture(tex_d, uv - e.xy).r) * 0.5;
                    float gy = (texture(tex_d, uv + e.yx).r - texture(tex_d, uv - e.yx).r) * 0.5;
                    float g = m * length(vec2(gx, gy));   // k·|d'| 在屏幕像素口径下
                    // 半径 2 texel 的 5x5 高斯（只在需要的像素上取）
                    float sm = 0.0, ws = 0.0;
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            float w = exp(-float(dx*dx + dy*dy) * 0.5);
                            sm += w * texture(tex_d, uv + vec2(float(dx), float(dy)) / res).r;
                            ws += w;
                        }
                    }
                    sm /= ws;
                    out_c = vec4(g > target ? sm : d0, 0.0, 0.0, 1.0);
                }
            """)
        self._sl_prog['res'].value = float(res)
        # 腐蚀（min 滤波）：与下面的 max 滤波组成闭运算，专门填"孤立凹陷"
        self._ero_prog = self.host.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex_raw;
                uniform float res, k;
                void main(){
                    vec2 uv = gl_FragCoord.xy / res;
                    float best = 2.0;
                    for (int dy = -4; dy <= 4; ++dy) {
                        for (int dx = -4; dx <= 4; ++dx) {
                            if (abs(float(dx)) > k || abs(float(dy)) > k) continue;
                            best = min(best, texture(tex_raw,
                                uv + vec2(float(dx), float(dy)) / res).r);
                        }
                    }
                    out_c = vec4(best, 0.0, 0.0, 1.0);
                }
            """)
        self._ero_prog['res'].value = float(res)
        # 真·分离式高斯（水平/垂直各一遍）：9 点近似在 σ=4 时平滑量只有
        # 真高斯的约 1/3，导致深度轮廓锯齿明显、3D 里出现碎缝。
        self._gauss_prog = self.host.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex_d;
                uniform float res, sigma;
                uniform vec2 dir;
                void main(){
                    vec2 uv = gl_FragCoord.xy / res;
                    float acc = 0.0, wsum = 0.0;
                    float s2 = 2.0 * sigma * sigma;
                    // 19-tap（±9）：覆盖 σ=3 至 1.1% 权重——与 CPU 侧
                    // d2s_ops.anti_alias 核宽对齐（2026-11-20 parity 定案），
                    // 边缘行为 = 纹理钳制 ≡ CPU BORDER_REPLICATE
                    for (int i = -9; i <= 9; ++i) {
                        float fi = float(i);
                        float w = exp(-(fi * fi) / max(1e-6, s2));
                        acc += w * texture(tex_d, uv + dir * fi / res).r;
                        wsum += w;
                    }
                    out_c = vec4(acc / max(1e-6, wsum), 0.0, 0.0, 1.0);
                }
            """)
        self._gauss_prog['res'].value = float(res)
        self._ema_tex = self.host.ctx.texture((res, res), 1, dtype='f4')
        self._ema_tex.repeat_x = False
        self._ema_tex.repeat_y = False
        self._ema_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._depth_fbo = self.host.ctx.framebuffer([self.host.depth_tex])
        self._ema_fbo = self.host.ctx.framebuffer([self._ema_tex])
        self._pp_prog = self.host.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex_raw;
                uniform sampler2D tex_prev;
                uniform float res, lo, hi, alpha, aa;
                void main(){
                    // 不做上下翻：CUDA 写进 raw 的行序与 gl_FragCoord.y 一致，
                    // CPU 版 numpy 写纹理也是 row0→row0，翻反了会让深度上下颠倒
                    // （实测：不翻 corr 0.46，翻一次 0.9999）。
                    vec2 uv = gl_FragCoord.xy / res;
                    float sc = 1.0 / max(1e-6, hi - lo);
                    float d = clamp((texture(tex_raw, uv).r - lo) * sc, 0.0, 1.0);
                    if (aa > 0.001) {
                        // 9-tap 高斯近似：半径 sigma=aa、2*aa 两圈，
                        // 权重 exp(-0.5), exp(-2.0)（与 CPU 版 sigma 语义对齐）
                        vec2 e1 = vec2(aa / res, 0.0);
                        vec2 e2 = vec2(0.0, aa / res);
                        vec2 e3 = vec2(2.0 * aa / res, 0.0);
                        vec2 e4 = vec2(0.0, 2.0 * aa / res);
                        float w1 = 0.6065, w2 = 0.1353;
                        float acc = d;
                        float wsum = 1.0;
                        acc += w1 * clamp((texture(tex_raw, uv + e1).r - lo) * sc, 0.0, 1.0);
                        acc += w1 * clamp((texture(tex_raw, uv - e1).r - lo) * sc, 0.0, 1.0);
                        acc += w1 * clamp((texture(tex_raw, uv + e2).r - lo) * sc, 0.0, 1.0);
                        acc += w1 * clamp((texture(tex_raw, uv - e2).r - lo) * sc, 0.0, 1.0);
                        acc += w2 * clamp((texture(tex_raw, uv + e3).r - lo) * sc, 0.0, 1.0);
                        acc += w2 * clamp((texture(tex_raw, uv - e3).r - lo) * sc, 0.0, 1.0);
                        acc += w2 * clamp((texture(tex_raw, uv + e4).r - lo) * sc, 0.0, 1.0);
                        acc += w2 * clamp((texture(tex_raw, uv - e4).r - lo) * sc, 0.0, 1.0);
                        wsum += 4.0 * w1 + 4.0 * w2;
                        d = acc / wsum;
                    }
                    float p = texture(tex_prev, uv).r;
                    out_c = vec4(alpha * d + (1.0 - alpha) * p, 0.0, 0.0, 1.0);
                }
            """)
        self._pp_prog['res'].value = float(res)
        _quad = self.host.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1],
                                         np.float32))
        self._pp_vao = self.host.ctx.vertex_array(self._pp_prog, [(_quad, "2f", "in_pos")])
        self._dil_vao = self.host.ctx.vertex_array(self._dil_prog, [(_quad, "2f", "in_pos")])
        self._sl_vao = self.host.ctx.vertex_array(self._sl_prog, [(_quad, "2f", "in_pos")])
        self._gauss_vao = self.host.ctx.vertex_array(self._gauss_prog, [(_quad, "2f", "in_pos")])
        self._ero_vao = self.host.ctx.vertex_array(self._ero_prog, [(_quad, "2f", "in_pos")])
        self._ema_tex.write(np.full((res, res), 0.5, np.float32).tobytes())
        # 首帧直通旗标：首帧 EMA α=1（状态=首帧 raw）——对齐 CPU 链
        # _ema_apply 的 prev=None 直通语义（parity 定案 2026-11-20），
        # 顺带消除 0.5 灰种子的开场淡入（~2N 帧）。
        self._ema_first = True
        # 切阀残差统计（1x1）：空间链输出 vs _ema_tex（时域状态）
        self._cut_tex = self.host.ctx.texture((1, 1), 4, dtype='f4')
        self._cut_fbo = self.host.ctx.framebuffer([self._cut_tex])
        self._cut_prog = self.host.ctx.program(
            vertex_shader='''#version 330
            in vec2 in_pos;
            void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }''',
            fragment_shader=CUTVALVE_FS)
        self._cut_prog['res'].value = float(res)
        self._cut_prog['thr'].value = self.CUT_RES
        self._cut_vao = self.host.ctx.vertex_array(
            self._cut_prog, [(_quad, "2f", "in_pos")])
        # 时域中值 W5 的历史环（4 张；窗口第 5 元=当帧 cur）与着色器
        if self.temporal_median:
            self._med_tex = []
            self._med_fbo = []
            for _ in range(4):
                t = self.host.ctx.texture((res, res), 1, dtype='f4')
                t.repeat_x = t.repeat_y = False
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._med_tex.append(t)
                self._med_fbo.append(self.host.ctx.framebuffer([t]))
            self._med_prog = self.host.ctx.program(
                vertex_shader='''#version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }''',
                fragment_shader=MEDIAN_FS)
            self._med_prog['res'].value = float(res)
            self._med_vao = self.host.ctx.vertex_array(
                self._med_prog, [(_quad, "2f", "in_pos")])
            self._med_ptr = 0

    # ------------------------------------------------------------------
    # 后处理 pass 图：每个 pass = (shader, 输入纹理, 输出 fbo, uniforms)
    # run_pass 在执行前强制 src != dst —— GL 里同一纹理同时当采样器和
    # 渲染目标属于未定义行为（P1 feedback loop 就是这么来的）。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 后处理 pass 图：每个 pass = (shader, 输入纹理, 输出 fbo, uniforms)
    # run_pass 在执行前强制 src != dst —— GL 里同一纹理同时当采样器和
    # 渲染目标属于未定义行为（P1 feedback loop 就是这么来的）。
    # ------------------------------------------------------------------
    def run_pass(self, label, prog, vao, dst_fbo, src_tex, uniforms=None):
        dst_tex = dst_fbo.color_attachments[0]
        if src_tex is dst_tex:
            raise RuntimeError(f"[pass:{label}] 读写同一纹理 -> feedback loop")
        src_tex.use(location=0)
        if uniforms:
            for k, v in uniforms.items():
                prog[k].value = v
        dst_fbo.use()
        self.host.ctx.viewport = (0, 0, self._pp_res, self._pp_res)
        vao.render(moderngl.TRIANGLE_STRIP)
        return dst_tex

    def spatial_chain(self, raw_tex, lo: float, hi: float, aa_texels: float = 0.0,
                      grow: int = 0, slope_target: float = 0.0,
                      parallax_px: float = 0.0, iters: int = 6,
                      close_px: int = 0):
        """归一化(+grow) → [闭运算] → [坡度限幅×iters] → [分离高斯 AA]。

        原 viewer.postprocess_gpu 的 pass 1-4a 逐行搬运（块边界：时域 EMA
        归 depth_mc.MC / 本类 ema_pass_legacy，由 viewer 胶水编排）。
        """
        cur = self.run_pass(
            "normalize", self._dil_prog, self._dil_vao, self._dil_fbo, raw_tex,
            {"lo": float(lo), "hi": float(hi), "grow": float(min(6, int(grow))),
             "tex_raw": 0})

        if close_px and close_px > 0:
            k = float(min(6, int(close_px)))
            tmp = self.run_pass(
                "close-max", self._dil_prog, self._dil_vao, self._sl_fbo[0], cur,
                {"lo": 0.0, "hi": 1.0, "grow": k, "tex_raw": 0})
            cur = self.run_pass(
                "close-min", self._ero_prog, self._ero_vao, self._sl_fbo[1], tmp,
                {"k": float(min(4, int(close_px))), "tex_raw": 0})

        if slope_target > 0.0 and parallax_px > 0.0:
            try:
                screen_w = float(self.host._texture_size[0])
            except Exception:
                screen_w = 1920.0
            m = 2.0 * float(parallax_px) * float(self._pp_res) / max(1.0, screen_w)
            cands = [self._dil_fbo, self._sl_fbo[0], self._sl_fbo[1]]
            for i in range(max(1, int(iters))):
                dst = next(f for f in cands if f.color_attachments[0] is not cur)
                cur = self.run_pass(f"slope{i}", self._sl_prog, self._sl_vao, dst, cur,
                                    {"m": m, "target": float(slope_target), "tex_d": 0})

        if aa_texels and aa_texels > 0.0:
            g = self._gauss_prog
            # 各向异性（2026-11-20 用户定案）：水平 σ=基准（AA=3→σ3），垂直
            # =×2/3（σ2）——224²→16:9 回映下水平 texel 占 8.6 屏幕px、垂直
            # 4.8px，方向配比按屏幕域观感校准
            for label, d, s in (("gauss-h", (1.0, 0.0), aa_texels),
                                ("gauss-v", (0.0, 1.0), aa_texels * (2.0 / 3.0))):
                g['sigma'].value = float(s)
                dst = next(f for f in (self._dil_fbo, self._sl_fbo[0], self._sl_fbo[1])
                           if f.color_attachments[0] is not cur)
                g['dir'].value = d
                cur = self.run_pass(label, g, self._gauss_vao, dst, cur, {"tex_d": 0})
        return cur

    def residual_area(self, cur):
        """切阀残差统计：|cur−时域状态|>CUT_RES 的面积占比（1x1 读回）。

        时域状态 = _ema_tex（每帧末尾 copy 为最终输出，任何时域模式下都
        是"上一帧输出"）。与 depth.py _ema_apply 的 numpy 判据同式同域。
        """
        cur.use(location=0)
        self._ema_tex.use(location=1)
        p = self._cut_prog
        p['tex_cur'].value = 0
        p['tex_prev'].value = 1
        self._cut_fbo.use()
        self.host.ctx.viewport = (0, 0, 1, 1)
        self._cut_vao.render(moderngl.TRIANGLE_STRIP)
        import numpy as _np
        return float(_np.frombuffer(
            self._cut_fbo.read(components=4, dtype='f4'), _np.float32)[0])

    def median_pass(self, cur, reset=False):
        """时域中值 W5：median(cur, 环0..3) → depth_fbo；环滚动存 cur。

        首帧（take_ema_first 一次性旗标）：环 4 槽全部播种为 cur、输出=cur
        ——与 CPU _median_apply 的播种语义逐帧对齐（parity A2 在岗）。
        环滚动 = 最老槽 ← cur（copy_framebuffer，~0.05ms）。
        """
        host = self.host
        cur_fbo = next(f for f in (self._dil_fbo, self._sl_fbo[0], self._sl_fbo[1])
                       if f.color_attachments[0] is cur)
        if reset:   # 首帧播种 或 切阀命中（清环=整幅跳净，无跨场景混合）
            for f in self._med_fbo:
                host.ctx.copy_framebuffer(f, cur_fbo)
            host.ctx.copy_framebuffer(self._depth_fbo, cur_fbo)
            return
        cur.use(location=0)
        for i, t in enumerate(self._med_tex):
            t.use(location=1 + i)
        p = self._med_prog
        p['tex_c'].value = 0
        p['tex_h0'].value = 1
        p['tex_h1'].value = 2
        p['tex_h2'].value = 3
        p['tex_h3'].value = 4
        self._depth_fbo.use()
        host.ctx.viewport = (0, 0, self._pp_res, self._pp_res)
        self._med_vao.render(moderngl.TRIANGLE_STRIP)
        host.ctx.copy_framebuffer(self._med_fbo[self._med_ptr], cur_fbo)
        self._med_ptr = (self._med_ptr + 1) % 4

    def ema_pass_legacy(self, cur, alpha: float):
        """legacy 时域 EMA（无 MC 时的 _pp_prog pass，渲到 depth_tex）。"""
        self._ema_tex.use(location=1)
        return self.run_pass(
            "ema", self._pp_prog, self._pp_vao, self._depth_fbo, cur,
            {"lo": 0.0, "hi": 1.0, "alpha": float(alpha), "aa": 0.0,
             "tex_raw": 0, "tex_prev": 1})
