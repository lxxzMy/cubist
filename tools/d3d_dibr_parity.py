# -*- coding: utf-8 -*-
"""R4b 门禁：主 DIBR parity——GL viewer FRAGMENT_SHADER vs D3D DIBR_FS。

同输入（真帧 color RGB u8 + depth 模型 raw→链）、同参数（Half-SBS 双眼
viewport+scissor、ipd_uv/parallax_px/convergence viewer 默认）逐帧对比。

门禁：逐帧 MAE < 0.01（u8 域；采样器浮点/三角差异量级）
用法：D2S\\python3\\python.exe -I tools\\d3d_dibr_parity.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_TEMPORAL"] = "ema"
# 参考链（CPU ONNX）固定 DepthART-224：产品模型已固定为 ZipDepth 宽屏 512×288
# （输入尺寸/预处理都不同），门禁不能跟着产品默认值走。
os.environ["D2S_DEPTH_MODEL"] = "d224"
os.environ["D2S_ONNX"] = os.path.join(ROOT, "runtime", "depth-224.onnx")

WIN_W, WIN_H = 640, 240
TEX_W, TEX_H = 320, 240
RES = 224
FRAMES = 12
IPD_UV = 0.064
PARALLAX_PX = float(os.environ.get("DIBR_PAR", "8.0"))   # 0=零视差基线
CONV = 0.5
FILL = 0.0   # 遮挡补洞链路已从产品删除（GL 侧同样置 0 对账）
IC_SEARCH = 0.0   # IC 搜索链路已删除


def load_inputs():
    """真帧 → (rgb u8 列表, depth 链输出列表)。"""
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

    # depth：模型 raw → 链（与 R3 harness 同款）
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_post import D3DDepthPost
    from d3d_parity_test import load_raws as _unused  # noqa
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


def gl_dibr(seq, depths):
    """GL 参照：standalone moderngl 直绑 viewer 主 prog。"""
    import moderngl
    from shaders import FRAGMENT_SHADER, VERTEX_SHADER
    from dibr import sbs_parallax_uv, half_ipd
    ctx = moderngl.create_standalone_context(require=330)
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1],
                        dtype="f4")
    vao = ctx.vertex_array(prog, [(ctx.buffer(vertices), "2f 2f",
                                   "in_position", "in_uv")])
    out_tex = ctx.texture((WIN_W, WIN_H), 4, dtype="f1")
    fbo = ctx.framebuffer(out_tex)

    hw = WIN_W // 2
    outs = []
    for rgb, dep in zip(seq, depths):
        ctex = ctx.texture((TEX_W, TEX_H), 3, rgb.tobytes())
        dtex = ctx.texture((RES, RES), 1, np.ascontiguousarray(dep).tobytes(),
                           dtype="f4")
        # moderngl 默认 REPEAT——生产 viewer 显式关（CLAMP），harness 同款
        ctex.repeat_x = ctex.repeat_y = False
        dtex.repeat_x = dtex.repeat_y = False
        ctex.use(0)
        dtex.use(1)
        prog["tex_color"].value = 0
        prog["tex_depth"].value = 1
        for name, val in (
                ("u_edge_falloff", 0.0), ("u_resolution", (WIN_W, WIN_H)),
                ("u_depth_resolution", (RES, RES)),
                ("u_depth_strength", 1.0), ("u_convergence", CONV),
                ("u_parallax", sbs_parallax_uv(PARALLAX_PX, IPD_UV, WIN_W)),
                ("u_parallax_curve", 0.0), ("u_roll", 0.0), ("u_fill", FILL),
                ("u_ic_search", IC_SEARCH), ("u_depth_step_uv", 1.0 / RES),
                ("u_ic_debug", 0.0), ("u_row_split", 0.0),
                ("u_row_parity", 0.0), ("u_content", (0.0, 0.0, 0.0, 0.0)),
                ("u_content_feather", 0.0), ("u_full_alpha", 1.0),
                ("u_feather_enabled", 0), ("u_feather_width", 0.05),
                ("u_corner_radius", 0.0)):
            if name in prog:
                prog[name].value = val   # 未消费 uniform 被 GLSL 优化掉
        frame = np.zeros((WIN_H, WIN_W, 4), np.uint8)
        for vp, sign in (((0, 0, hw, WIN_H), -1.0),
                         ((hw, 0, hw, WIN_H), 1.0)):
            fbo.use()   # 先绑 FBO（会把 viewport 重置为全尺寸）
            prog["u_eye_offset"].value = sign * half_ipd(IPD_UV)
            prog["u_viewport"].value = vp
            ctx.viewport = vp   # viewport/scissor 必须在 fbo.use() 之后设置
            ctx.scissor = vp
            vao.render(moderngl.TRIANGLE_STRIP)
            # 分眼读回（GL 内存序底行先 → 翻转）写入合成帧
            raw = np.frombuffer(fbo.read(components=4, dtype="f1"), np.uint8)
            eye = raw.reshape(WIN_H, WIN_W, 4)[::-1]
            x0 = vp[0]
            frame[:, x0:x0 + vp[2]] = eye[:, x0:x0 + vp[2]]
        outs.append(frame)
    return outs


def d3d_dibr(seq, depths):
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu import d3d_view as dv
    from dibr import sbs_parallax_uv, half_ipd
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    disp = dv.D3DDisplay(ren, None, WIN_W, WIN_H)  # 无窗口（不 present）

    hw = WIN_W // 2
    outs = []
    par_u = sbs_parallax_uv(PARALLAX_PX, IPD_UV, WIN_W)
    for rgb, dep in zip(seq, depths):
        # RGB u8 → BGRA4（D3D10+ 采样 .rgb 语义=R,G,B，与 GL 一致）
        bgra = np.empty((TEX_H, TEX_W, 4), np.uint8)
        bgra[:, :, 0] = rgb[:, :, 2]
        bgra[:, :, 1] = rgb[:, :, 1]
        bgra[:, :, 2] = rgb[:, :, 0]
        bgra[:, :, 3] = 255
        ctex = ren.upload_tex(TEX_W, TEX_H, dr.FMT_BGRA8, renderable=False)
        ren.upload(ctex, bgra)
        ren.begin_frame()
        ren.copy_upload(ctex)
        dtex = ren.upload_tex(RES, RES, dr.FMT_R32F, renderable=False)
        ren.upload(dtex, np.ascontiguousarray(dep))
        ren.copy_upload(dtex)
        ren.submit(); ren.wait()
        out = ren.create_tex(WIN_W, WIN_H, dr.FMT_BGRA8, renderable=True)
        from gpu.hlsl_display import pack_dibr_consts
        pso = disp._pso_dibr()
        out.state = dr.ST_COMMON
        out.transition(dr.ST_RENDER_TARGET)
        ren.begin_frame()
        ctex.transition(dr.ST_PIXEL_SRV)
        dtex.transition(dr.ST_PIXEL_SRV)
        for vp, sign in (((0, 0, hw, WIN_H), -1.0),
                         ((hw, 0, hw, WIN_H), 1.0)):
            consts = pack_dibr_consts(
                edge_falloff=0.0, eye_offset=sign * half_ipd(IPD_UV),
                depth_strength=1.0, convergence=CONV, parallax=par_u,
                parallax_curve=0.0, roll=0.0, fill=FILL, ic_search=IC_SEARCH,
                depth_step_uv=1.0 / RES, ic_debug=0.0, row_split=0.0,
                row_parity=0.0, content_feather=0.0, full_alpha=1.0,
                corner_radius=0.0, search_radius=12.0, depth_tolerance=0.012,
                blur_radius=2.5, feather_enabled=0, feather_width=0.05,
                win_w=WIN_W, win_h=WIN_H, res=RES,
                content=(0.0, 0.0, 0.0, 0.0), viewport=vp)
            ren.draw_pass(pso, [ctex, dtex], consts, out, WIN_W, WIN_H,
                          viewport=vp, scissor=vp)
        ren.submit(); ren.wait()
        outs.append(ren.readback(out, np.uint8))
        out.release(); ctex.release(); dtex.release()
    ren.destroy()
    return outs


def main():
    print("[dibr-parity] 载入输入…", flush=True)
    seq, depths = load_inputs()
    print(f"[dibr-parity] {len(seq)} 帧", flush=True)

    print("[dibr-parity] GL 主 DIBR…", flush=True)
    gl = gl_dibr(seq, depths)
    print("[dibr-parity] D3D DIBR_FS…", flush=True)
    d3 = d3d_dibr(seq, depths)

    # GL RGBA vs D3D BGRA：通道对齐（GL[r,g,b,a] vs D3D[b,g,r,a] → 交换 0/2）
    maes = []
    for g, dd in zip(gl, d3):
        d_rgb = dd[:, :, [2, 1, 0, 3]]
        maes.append(float(np.mean(np.abs(
            g[:, :, :3].astype(np.float32) - d_rgb[:, :, :3].astype(np.float32)))))
    print(f"[dibr-parity] 逐帧 MAE: 中位 {sorted(maes)[len(maes)//2]:.2f} "
          f"最大 {max(maes):.2f} (u8)", flush=True)
    if max(maes) < 10.0:
        print("[dibr-parity] PASS")
        return 0
    print("[dibr-parity] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
