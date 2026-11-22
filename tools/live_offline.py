"""离线跑"实时链路"：用无头 GL + 程序自己的着色器源码，不启动窗口。

用途：把程序渲染路径与 numpy 模拟放在**同一输入**下对比，排除"源图不同"的干扰。

链：
  image → ONNX(DepthART-spf-448) → raw 448
        → [GL] normalize → gauss-h → gauss-v            （= 程序 postprocess，EMA 关）
        → [GL] 程序 FRAGMENT_SHADER（单眼，u_eye_offset=±ipd/2）→ PNG

用法：
  python tools/live_offline.py IMAGE [-o OUTDIR] [--parallax 20] [--plane 0.35]
                                    [--aa 2] [--eye both|left|right]
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

import cv2
import moderngl
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))

MODEL = os.path.join(ROOT, "runtime", "DepthART-spf-448.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IPD_UV = 0.064          # 程序里的 ipd_uv（half = 0.032）


def shader_sources():
    """从 shaders.py 里抽出着色器源码，保证和程序完全一致。"""
    src = io.open(os.path.join(ROOT, "D2S", "shaders.py"), encoding="utf-8").read()
    def grab(name):
        i = src.index(f"{name} = ")
        a = src.index('"""', i) + 3
        b = src.index('"""', a)
        return src[a:b]
    return grab("VERTEX_SHADER"), grab("FRAGMENT_SHADER")


def depth_raw(img, sess):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.squeeze(sess.run(None, {sess.get_inputs()[0].name: x})[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--outdir", default=os.path.join(ROOT, "_debug_frames"))
    ap.add_argument("--parallax", type=float, default=20.0)
    ap.add_argument("--plane", type=float, default=0.35)
    ap.add_argument("--aa", type=float, default=2.0, help="面板抗锯齿值 → σ = 2×该值 texel")
    ap.add_argument("--eye", default="both", choices=["both", "left", "right"])
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不到 {args.image}")
    H, W = img.shape[:2]
    raw = depth_raw(img, sess)
    sub = raw[::4, ::4]
    lo, hi = float(np.percentile(sub, 2)), float(np.percentile(sub, 98))
    sigma = float(args.aa) * 2.0        # = AA_STRENGTH，单位 texel（448 网格）

    ctx = moderngl.create_standalone_context(require=330)

    # ---------- 深度后处理（等价于程序的 normalize + gauss-h + gauss-v） ----------
    depth_tex = ctx.texture((448, 448), 1, dtype='f4')
    depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    depth_tex.repeat_x = depth_tex.repeat_y = False
    depth_tex.write(np.ascontiguousarray(raw, np.float32).tobytes())
    tmp_tex = [ctx.texture((448, 448), 1, dtype='f4') for _ in range(2)]
    for t in tmp_tex:
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
    tmp_fbo = [ctx.framebuffer([t]) for t in tmp_tex]
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32))

    pp_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_pos; out vec2 v_uv;
            void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
        """,
        fragment_shader="""
            #version 330
            in vec2 v_uv; out vec4 out_c;
            uniform sampler2D tex_in;
            uniform float lo, hi;
            uniform vec2 dir;          // (1,0) 水平 / (0,1) 垂直
            uniform float sigma;
            uniform int pass_id;       // 0=归一化 1=高斯
            void main(){
                if (pass_id == 0) {
                    float v = texture(tex_in, v_uv).r;
                    out_c = vec4(clamp((v - lo) / max(1e-6, hi - lo), 0.0, 1.0), 0.0, 0.0, 1.0);
                } else {
                    float acc = 0.0, wsum = 0.0;
                    float s2 = 2.0 * sigma * sigma;
                    for (int i = -9; i <= 9; ++i) {   // 19-tap 对齐产品（2026-11-20）
                        float fi = float(i);
                        float w = exp(-(fi * fi) / max(1e-6, s2));
                        acc += w * texture(tex_in, v_uv + dir * fi / 448.0).r;
                        wsum += w;
                    }
                    out_c = vec4(acc / max(1e-6, wsum), 0.0, 0.0, 1.0);
                }
            }
        """)
    pp_vao = ctx.vertex_array(pp_prog, [(quad, "2f", "in_pos")])
    pp_prog['tex_in'].value = 0
    pp_prog['lo'].value = lo
    pp_prog['hi'].value = hi
    pp_prog['sigma'].value = sigma

    cur = depth_tex
    # 各向异性（2026-11-20）：水平 σ=基准，垂直 ×2/3 —— 与产品同口径
    for idx, (label, pid, d, s) in enumerate((
            ("normalize", 0, (1, 0), 0.0),
            ("gauss-h", 1, (1, 0), sigma),
            ("gauss-v", 1, (0, 1), sigma * (2.0 / 3.0)))):
        dst = tmp_fbo[idx % 2]
        cur.use(location=0)
        pp_prog['pass_id'].value = pid
        pp_prog['dir'].value = d
        pp_prog['sigma'].value = float(s)
        dst.use()
        ctx.viewport = (0, 0, 448, 448)
        pp_vao.render(moderngl.TRIANGLE_STRIP)
        cur = dst.color_attachments[0]
    depth_out = np.frombuffer(cur.read(), np.float32).reshape(448, 448)

    # ---------- 渲染（程序 FRAGMENT_SHADER） ----------
    vs, fs = shader_sources()
    color_tex = ctx.texture((W, H), 3, data=np.ascontiguousarray(img[:, :, ::-1]))
    color_tex.build_mipmaps() if False else None
    color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    color_tex.repeat_x = color_tex.repeat_y = False
    dtex = ctx.texture((448, 448), 1, dtype='f4')
    dtex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    dtex.repeat_x = dtex.repeat_y = False
    dtex.write(np.ascontiguousarray(depth_out, np.float32).tobytes())

    out_tex = ctx.texture((W, H), 3, dtype='f1')
    out_fbo = ctx.framebuffer([out_tex])
    prog = ctx.program(vertex_shader=vs, fragment_shader=fs)
    # 程序的 VBO 是 '2f 2f'（位置 + uv），这里照做
    quad2 = ctx.buffer(np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    ], np.float32))
    vao = ctx.vertex_array(prog, [(quad2, "2f 2f", "in_position", "in_uv")])

    def setu(name, val):
        if name in prog:
            prog[name].value = val

    half_ipd = IPD_UV / 2.0
    setu("tex_color", 0)
    setu("tex_depth", 1)
    setu("u_convergence", float(args.plane))
    setu("u_depth_resolution", (448.0, 448.0))
    setu("u_depth_step_uv", 1.0 / 448.0)
    setu("u_depth_strength", 1.0)
    setu("u_edge_falloff", 0.0)
    setu("u_edge_snap", 0.0)
    setu("u_edge_snap_gain", 1.0)
    setu("u_feather_enabled", 0)
    setu("u_feather_width", 0.02)
    setu("u_fill", 0.0)
    setu("u_ic_debug", 0.0)
    setu("u_ic_search", 0.0)
    setu("u_interleave_phase", 0.0)
    setu("u_parallax", float(args.parallax) * (1.0 / max(1e-6, half_ipd * W)))
    setu("u_parallax_curve", 0.0)
    setu("u_relief_strength", 0.0)
    setu("u_resolution", (float(W), float(H)))
    setu("u_roll", 0.0)
    setu("u_viewport", (0.0, 0.0, float(W), float(H)))
    setu("u_warp_iterations", 0.0)
    setu("u_content", (0.0, 0.0, 1.0, 1.0))
    setu("u_content_feather", 0.001)

    os.makedirs(args.outdir, exist_ok=True)
    eyes = {"left": -half_ipd, "right": half_ipd}
    if args.eye != "both":
        eyes = {args.eye: eyes[args.eye]}
    outs = {}
    for name, off in eyes.items():
        setu("u_eye_offset", float(off))
        color_tex.use(location=0)
        dtex.use(location=1)
        out_fbo.use()
        ctx.viewport = (0, 0, W, H)
        vao.render(moderngl.TRIANGLE_STRIP)
        arr = np.frombuffer(out_tex.read(), np.uint8).reshape(H, W, 3)
        arr = cv2.cvtColor(arr[::-1], cv2.COLOR_RGB2BGR)
        p = os.path.join(args.outdir, f"live_{name}.png")
        cv2.imwrite(p, arr)
        outs[name] = arr
        print(f"写出 {p}")

    np.save(os.path.join(args.outdir, "live_depth.npy"), depth_out)
    print(f"深度已存 live_depth.npy  lo={lo:.3f} hi={hi:.3f} sigma={sigma:g} "
          f"parallax={args.parallax:g} plane={args.plane:g}")


if __name__ == "__main__":
    main()
