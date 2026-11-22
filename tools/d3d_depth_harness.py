# -*- coding: utf-8 -*-
"""R3 门禁：D3D 深度管线真帧 harness（prep → 推理 → spatial_chain → 读回）。

两档对账：
  A. prep parity：GL gpu_preprocess vs D3D PREP_FS，同 224 RGB 帧输入
     → 各自 3R×R R32F 读回，MAE < 1e-5（同款行内平铺+归一化公式）
  B. 端到端：各自 prep 输出 → reshape CHW → 同一 CPU ONNX → raw →
     各自 spatial_chain(EMA α19 AA3) → depth 读回，MAE < 0.001
     （prep 微差经推理放大的量级上限）+ 连续 120 帧 + 设备态。

用法：D2S\python3\python.exe -I tools\d3d_depth_harness.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_TEMPORAL"] = "ema"
os.environ["D2S_ONNX"] = ""

RES = 224
FRAMES = 120
ALPHA = 0.19
AA = 3.0


def load_frames():
    """真实帧 → RGB u8 224×224 序列。"""
    import cv2
    frames = []
    for f in sorted(os.listdir(os.path.join(ROOT, "_refactor_baseline", "inputs"))):
        if f.endswith(".png"):
            frames.append(cv2.cvtColor(cv2.imread(
                os.path.join(ROOT, "_refactor_baseline", "inputs", f)),
                cv2.COLOR_BGR2RGB))
    frames = [cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA)
              for f in frames]
    return [frames[i % len(frames)] for i in range(FRAMES)]


def gl_prep_outputs(frames):
    """GL 参照 prep：standalone moderngl + FakeWin（d3d_parity_test 同款）。"""
    import moderngl
    import viewer
    from cpu_gpu_parity_test import _DummyProg, _CtxProxy
    ctx = moderngl.create_standalone_context(require=330)

    class FakeWin:
        pass

    w = FakeWin()
    w.ctx = _CtxProxy(ctx)
    w._gpu_res = RES
    w._depth_tex_size = None
    w.color_tex = ctx.texture((RES, RES), 3, dtype="f1")
    w.depth_tex = None
    w._mc_on = False
    for k in ("prog", "anaglyph_prog", "interleaved_prog",
              "vertical_interleaved_prog"):
        setattr(w, k, _DummyProg())
    for m in ("gpu_depth_init", "gpu_post_init", "gpu_preprocess",
              "postprocess_gpu", "read_depth_median", "_channel_selfcheck",
              "_post_get"):
        setattr(w, m, getattr(viewer.StereoWindow, m).__get__(w))
    w.gpu_depth_init(RES)
    _pw, _ph = w._prep_tex.size
    outs = []
    for f in frames:
        w.color_tex.write(np.ascontiguousarray(f, np.uint8).tobytes())
        w.gpu_preprocess()
        outs.append(np.frombuffer(w._prep_fbo.read(components=1, dtype="f4"),
                                  np.float32).reshape(_ph, _pw).copy())
    return outs, w


def d3d_prep_outputs(frames):
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_prep import D3DDepthPrep
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    prep = D3DDepthPrep(ren, RES)
    outs = []
    for f in frames:
        # GL color_tex 是 RGB u8；D3D BGRA8 采 .z=R —— 喂 BGRA 保证 R 语义
        bgra = np.ascontiguousarray(f[:, :, ::-1], np.uint8)
        prep.set_frame(bgra)
        prep.run()
        outs.append(prep.readback())
    return outs, ren, prep


def main():
    print("[harness] 载入帧…", flush=True)
    frames = load_frames()
    print(f"[harness] {len(frames)} 帧({RES}x{RES})", flush=True)

    print("[harness] GL prep 参照…", flush=True)
    gl_preps, glw = gl_prep_outputs(frames)

    print("[harness] D3D prep…", flush=True)
    d3d_preps, ren, dpre = d3d_prep_outputs(frames)

    fails = []
    # ---- A. prep parity ----
    maes = [float(np.mean(np.abs(a - b)))
            for a, b in zip(gl_preps, d3d_preps)]
    print(f"[prep-parity] 逐帧 MAE: 中位 {sorted(maes)[len(maes)//2]:.3e} "
          f"最大 {max(maes):.3e}", flush=True)
    if max(maes) >= 1e-5:
        fails.append(f"prep MAE {max(maes):.3e}")

    # ---- B. 端到端：prep → CPU ONNX → raw → spatial_chain ----
    import depth as depth_mod

    from gpu.d3d_post import D3DDepthPost
    post = D3DDepthPost(ren, RES)
    d3d_depths = []
    for i in range(len(frames)):
        chw = np.ascontiguousarray(d3d_preps[i].ravel().reshape(3, RES, RES),
                                   np.float32)   # flat 线性序即 CHW
        raw = depth_mod.depth_model(chw)
        sub = raw[::4, ::4]
        lo = float(np.percentile(sub, 0.5))
        hi = float(np.percentile(sub, 99.5))
        post.upload_raw(raw)
        post.spatial_chain(lo, hi, aa_texels=AA, temporal="ema", alpha=ALPHA)
        d3d_depths.append(post.readback_depth())

    # GL 端到端参照：GL prep 输出同喂 ONNX → GL spatial_chain + EMA
    gl_depths = []
    w = glw
    for i in range(len(frames)):
        chw = np.ascontiguousarray(gl_preps[i].ravel().reshape(3, RES, RES),
                                   np.float32)
        raw = depth_mod.depth_model(chw)
        sub = raw[::4, ::4]
        lo = float(np.percentile(sub, 0.5))
        hi = float(np.percentile(sub, 99.5))
        w._raw_tex.write(np.ascontiguousarray(raw, np.float32).tobytes())
        w.postprocess_gpu(w._raw_tex, lo, hi, ALPHA, AA, 0, 0.0, 0.0,
                          iters=6, close_px=0)
        gl_depths.append(np.frombuffer(
            w._post.depth_fbo.read(components=1, dtype="f4"),
            np.float32).reshape(RES, RES).copy())

    emaes = [float(np.mean(np.abs(a - b)))
             for a, b in zip(gl_depths, d3d_depths)]
    print(f"[e2e] 逐帧 MAE: 中位 {sorted(emaes)[len(emaes)//2]:.5f} "
          f"最大 {max(emaes):.5f}", flush=True)
    if max(emaes) >= 0.001:
        fails.append(f"e2e MAE {max(emaes):.5f}")

    n_bad = sum(1 for o in d3d_depths
                if np.isnan(o).any() or o.min() < -0.01 or o.max() > 1.01)
    if n_bad:
        fails.append(f"帧异常 {n_bad}")

    ren.destroy()
    if fails:
        print(f"[harness] FAIL: {fails}")
        return 1
    print("[harness] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
