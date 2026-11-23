# -*- coding: utf-8 -*-
"""R4b 门禁（Interleaved 档）：GL interleaved_prog vs D3D INTERLEAVED_FS。

同输入（真帧 color + depth 链输出）、同参数（viewer Interleaved 默认：
edge_snap=1 gain=1 relief=0.22 warp=1 phase=0/1）逐帧对比。
门禁：逐帧 MAE < 10（u8；采样器数值差量级）
用法：D2S\\python3\\python.exe -I tools\\d3d_interleaved_parity.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_TEMPORAL"] = "ema"
os.environ["D2S_DEPTH_MODEL"] = "d224"   # 门禁固定 224（产品默认模型已不是它）
os.environ["D2S_ONNX"] = os.path.join(ROOT, "runtime", "depth-224.onnx")

WIN_W, WIN_H = 640, 240
TEX_W, TEX_H = 320, 240
RES = 224
FRAMES = 8
IPD_UV = 0.064
PARALLAX_PX = 12.0
CONV = 0.5
PHASE = float(os.environ.get("IL_PHASE", "0"))


def load_inputs():
    import cv2
    import depth as depth_mod
    frames = []
    for f in sorted(os.listdir(os.path.join(ROOT, "_refactor_baseline", "inputs"))):
        if f.endswith(".png"):
            frames.append(cv2.cvtColor(cv2.imread(
                os.path.join(ROOT, "_refactor_baseline", "inputs", f)),
                cv2.COLOR_BGR2RGB))
    frames = [cv2.resize(f, (TEX_W, TEX_H), interpolation=cv2.INTER_AREA)
              for f in frames]
    seq = [frames[i % len(frames)] for i in range(FRAMES)]

    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_post import D3DDepthPost
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    post = D3DDepthPost(ren, RES)
    depths = []
    for f in seq:
        resized = cv2.resize(f, (RES, RES), interpolation=cv2.INTER_LINEAR)
        chw = np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1))
        raw = depth_mod.depth_model(chw)
        sub = raw[::4, ::4]
        lo = float(np.percentile(sub, 0.5))
        hi = float(np.percentile(sub, 99.5))
        post.upload_raw(raw)
        post.spatial_chain(lo, hi, aa_texels=3.0, temporal="ema", alpha=0.19)
        depths.append(post.readback_depth())
    ren.destroy()
    return seq, depths


def gl_il(seq, depths):
    import moderngl
    from shaders import INTERLEAVED_FRAGMENT, VERTEX_SHADER
    from dibr import parallax_uv, half_ipd
    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VERTEX_SHADER,
                       fragment_shader=INTERLEAVED_FRAGMENT)
    vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1],
                        dtype="f4")
    vao = ctx.vertex_array(prog, [(ctx.buffer(vertices), "2f 2f",
                                   "in_position", "in_uv")])
    out_tex = ctx.texture((WIN_W, WIN_H), 4, dtype="f1")
    fbo = ctx.framebuffer(out_tex)
    vp = (0, 0, WIN_W, WIN_H)
    outs = []
    for rgb, dep in zip(seq, depths):
        ctex = ctx.texture((TEX_W, TEX_H), 3, rgb.tobytes())
        ctex.repeat_x = ctex.repeat_y = False
        dtex = ctx.texture((RES, RES), 1, np.ascontiguousarray(dep).tobytes(),
                           dtype="f4")
        dtex.repeat_x = dtex.repeat_y = False
        ctex.use(0)
        dtex.use(1)
        prog["tex_color"].value = 0
        prog["tex_depth"].value = 1
        for name, val in (
                ("u_edge_falloff", 0.0), ("u_resolution", (WIN_W, WIN_H)),
                ("u_depth_resolution", (RES, RES)),
                ("u_eye_offset", half_ipd(IPD_UV)),
                ("u_depth_strength", 1.0), ("u_convergence", CONV),
                ("u_parallax", parallax_uv(PARALLAX_PX, IPD_UV, WIN_W)),
                ("u_parallax_curve", 0.0), ("u_roll", 0.0),
                ("u_search_radius", 12.0), ("u_depth_tolerance", 0.012),
                ("u_blur_radius", 2.5), ("u_feather_enabled", 0),
                ("u_feather_width", 0.05), ("u_corner_radius", 0.0),
                ("u_interleave_phase", PHASE), ("u_edge_snap", 1.0),
                ("u_edge_snap_gain", 1.0), ("u_relief_strength", 0.22),
                ("u_warp_iterations", 1.0), ("u_ic_search", 0.0),
                ("u_depth_step_uv", 1.0 / RES), ("u_ic_debug", 0.0),
                ("u_fill", 0.0), ("u_viewport", vp)):
            if name in prog:
                prog[name].value = val
        fbo.use()
        ctx.viewport = vp
        ctx.scissor = vp
        vao.render(moderngl.TRIANGLE_STRIP)
        raw = np.frombuffer(fbo.read(components=4, dtype="f1"), np.uint8)
        outs.append(raw.reshape(WIN_H, WIN_W, 4)[::-1].copy())
    return outs


def d3d_il(seq, depths):
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu import d3d_view as dv
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    disp = dv.D3DDisplay(ren, None, WIN_W, WIN_H)
    params = dict(ipd_uv=IPD_UV, parallax_px=PARALLAX_PX, convergence=CONV,
                  depth_strength=1.0, fill=0.0, ic_search=0.0,
                  phase=PHASE, edge_snap=1.0, edge_snap_gain=1.0,
                  relief=0.22, warp=1.0)
    outs = []
    for rgb, dep in zip(seq, depths):
        bgra = np.empty((TEX_H, TEX_W, 4), np.uint8)
        bgra[:, :, 0] = rgb[:, :, 2]
        bgra[:, :, 1] = rgb[:, :, 1]
        bgra[:, :, 2] = rgb[:, :, 0]
        bgra[:, :, 3] = 255
        ctex = ren.upload_tex(TEX_W, TEX_H, dr.FMT_BGRA8, renderable=False)
        ren.upload(ctex, bgra)
        dtex = ren.upload_tex(RES, RES, dr.FMT_R32F, renderable=False)
        ren.upload(dtex, np.ascontiguousarray(dep))
        ren.begin_frame()
        ren.copy_upload(ctex)
        ren.copy_upload(dtex)
        ren.submit(); ren.wait()
        out = ren.create_tex(WIN_W, WIN_H, dr.FMT_BGRA8, renderable=True)
        disp.draw_interleaved_offscreen(out, ctex, dtex, params)
        outs.append(ren.readback(out, np.uint8))
        out.release(); ctex.release(); dtex.release()
    ren.destroy()
    return outs


def main():
    print("[il-parity] 载入输入…", flush=True)
    seq, depths = load_inputs()
    print(f"[il-parity] {len(seq)} 帧 phase={PHASE}", flush=True)
    print("[il-parity] GL Interleaved…", flush=True)
    gl = gl_il(seq, depths)
    print("[il-parity] D3D INTERLEAVED_FS…", flush=True)
    d3 = d3d_il(seq, depths)
    maes = []
    for g, dd in zip(gl, d3):
        d_rgb = dd[:, :, [2, 1, 0]]
        maes.append(float(np.mean(np.abs(
            g[:, :, :3].astype(np.float32) - d_rgb[:, :, :3].astype(np.float32)))))
    print(f"[il-parity] 逐帧 MAE: 中位 {sorted(maes)[len(maes)//2]:.2f} "
          f"最大 {max(maes):.2f} (u8)", flush=True)
    if max(maes) < 10.0:
        print("[il-parity] PASS")
        return 0
    print("[il-parity] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
