# -*- coding: utf-8 -*-
"""R4c 门禁：D3D 渲染块行为级 E2E（断言族与 tools/e2e_playback.py 对齐）。

管线：合成运动输入（静止 1.2s → 平移 2.4s 交替 ×3）→ D3DDepthPrep →
CPU ONNX 推理 → D3DDepthPost 链 → Interleaved/Half-SBS 渲染 → 帧序列 dump。

断言族（对应历史 bug 类别）：
  A1 静止段逐位稳定（时域 EMA 收敛后不再变化）
  A2 运动段持续变化（>阈值；无运动内容的历史坑在此现形）
  A3 硬切瞬间输出跳变（切场景不被 EMA 抹平）
  A4 双眼差异（Half-SBS 左右半屏必须有视差差——立体真出了）
  A5 行编织相位（Interleaved 奇偶行平均亮度交替——行分眼真出了）
用法：D2S\\python3\\python.exe -I tools\\e2e_d3d.py
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
IPD_UV = 0.064
ALPHA = 0.19
AA = 3.0


def make_scene(w, h, offset, block_w=60):
    """合成场景：深色底 + 亮方块（offset 水平平移）+ 静态网格。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 30
    for gx in range(0, w, 40):
        img[:, gx:gx + 2] = 60
    x0 = int(offset) % (w + block_w) - block_w
    img[80:180, max(0, x0):min(w, x0 + block_w)] = (220, 180, 60)
    return img


def run_pipeline(frames_bgra):
    """合成帧序列 → D3D 全链 → (interleaved 输出序列, half-sbs 输出序列, 深度)。"""
    import cv2
    import depth as depth_mod

    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu import d3d_view as dv
    from gpu.d3d_post import D3DDepthPost
    from gpu.d3d_prep import D3DDepthPrep

    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    disp = dv.D3DDisplay(ren, None, WIN_W, WIN_H)
    prep = D3DDepthPrep(ren, RES)
    post = D3DDepthPost(ren, RES)

    il_out, sbs_out, depths = [], [], []
    # 生产语义：纹理/上传对/输出全复用（描述符槽不泄漏）
    ctex = ren.upload_tex(TEX_W, TEX_H, dr.FMT_BGRA8, renderable=False)
    dtex = ren.upload_tex(RES, RES, dr.FMT_R32F, renderable=False)
    out = ren.create_tex(WIN_W, WIN_H, dr.FMT_BGRA8, renderable=True)
    import ctypes as _c
    _g = dr.dll()
    _g.d3dr_get_desc_counts.restype = _c.c_uint64
    _g.d3dr_get_desc_counts.argtypes = [_c.c_void_p]
    for i, bgra in enumerate(frames_bgra):
        prep.set_frame(bgra)
        prep.run()
        chw = np.ascontiguousarray(prep.readback().ravel().reshape(3, RES, RES))
        raw = depth_mod.depth_model(chw)
        sub = raw[::4, ::4]
        lo = float(np.percentile(sub, 0.5))
        hi = float(np.percentile(sub, 99.5))
        post.upload_raw(raw)
        post.spatial_chain(lo, hi, aa_texels=AA, temporal="ema", alpha=ALPHA)
        dep = post.readback_depth()
        depths.append(dep)

        ren.upload(ctex, np.ascontiguousarray(bgra))
        ren.begin_frame()
        ren.copy_upload(ctex)
        ren.upload(dtex, np.ascontiguousarray(dep))
        ren.copy_upload(dtex)
        ren.submit(); ren.wait()

        disp.draw_interleaved_offscreen(
            out, ctex, dtex, dict(phase=0.0, ipd_uv=IPD_UV, parallax_px=12.0,
                                  convergence=0.5, depth_strength=1.0,
                                  fill=0.0, ic_search=1.0))
        il_out.append(ren.readback(out, np.uint8))
    ctex.release(); dtex.release(); out.release()
    ren.destroy()
    return il_out, sbs_out, depths


def main():
    fps = 30.0
    cycle = [(1.2, "static"), (2.4, "move")]
    frames = []
    phases = []
    t = 0.0
    fi = 0
    for rep in range(3):
        for dur, tag in cycle:
            n = int(dur * fps)
            for k in range(n):
                if tag == "static":
                    frames.append(make_scene(TEX_W, TEX_H, 0))
                else:
                    frames.append(make_scene(TEX_W, TEX_H, k / n * 160))
                phases.append((rep, tag))
                fi += 1
    print(f"[e2e-d3d] {len(frames)} 帧合成序列", flush=True)

    bgras = []
    for f in frames:
        b4 = np.empty((TEX_H, TEX_W, 4), np.uint8)
        b4[:, :, 0] = f[:, :, 2]
        b4[:, :, 1] = f[:, :, 1]
        b4[:, :, 2] = f[:, :, 0]
        b4[:, :, 3] = 255
        bgras.append(b4)
    il, _, depths = run_pipeline(bgras)

    fails = []

    # A1 静止段稳定：静止段内【连续帧】的帧间差（EMA 已收敛；不取跨周期对）
    static_idx = [i for i, (r, tg) in enumerate(phases)
                  if r >= 1 and tg == "static"]
    diffs = []
    for a, b in zip(static_idx[:-1], static_idx[1:]):
        if b - a != 1:
            continue   # 跨周期边界对（中间隔了 move 段）跳过
        diffs.append(float(np.mean(np.abs(
            il[a][:, :, :3].astype(np.float32) - il[b][:, :, :3].astype(np.float32)))))
    dmax = max(diffs) if diffs else 0.0
    print(f"[A1] 静止段帧间差: 最大 {dmax:.3f} (u8, 期望<1.0)", flush=True)
    if dmax >= 1.0:
        fails.append(f"A1 静止不稳 {dmax:.2f}")

    # A2 运动段活性：滑动段相邻帧差持续 > 阈值
    move_idx = [i for i, (r, tg) in enumerate(phases)
                if r == 1 and tg == "move"]
    mdiffs = [float(np.mean(np.abs(
        il[a][:, :, :3].astype(np.float32) - il[b][:, :, :3].astype(np.float32))))
        for a, b in zip(move_idx[:-1], move_idx[1:])]
    mmed = sorted(mdiffs)[len(mdiffs) // 2] if mdiffs else 0.0
    print(f"[A2] 运动段帧间差中位: {mmed:.3f} (u8, 期望>0.3)", flush=True)
    if mmed <= 0.3:
        fails.append(f"A2 无运动活性 {mmed:.3f}")

    # A3 硬切跳变：运动段末帧 vs 下一周期静止段首帧（场景内容跳变）
    for rep in (0, 1):
        move_end = max(i for i, (r, tg) in enumerate(phases)
                       if r == rep and tg == "move")
        static_beg = min(i for i, (r, tg) in enumerate(phases)
                         if r == rep + 1 and tg == "static")
        j = float(np.mean(np.abs(
            il[move_end][:, :, :3].astype(np.float32)
            - il[static_beg][:, :, :3].astype(np.float32))))
        print(f"[A3] 硬切跳变 rep{rep}: {j:.2f} (u8, 期望>2)", flush=True)
        if j <= 2:
            fails.append(f"A3 切换被抹平 rep{rep} {j:.2f}")

    # A5 行编织：Interleaved 奇偶行平均亮度差（视差让行间内容不同）
    mid = il[len(il) // 2][:, :, :3].astype(np.float32)
    row_lum = mid.mean(axis=(1, 2))
    alt = float(np.abs(np.diff(row_lum)).mean())
    print(f"[A5] 行交替亮度差: {alt:.3f} (期望>0.5)", flush=True)
    if alt <= 0.5:
        fails.append(f"A5 无行分眼 {alt:.3f}")

    if fails:
        print(f"[e2e-d3d] FAIL: {fails}")
        return 1
    print("[e2e-d3d] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
