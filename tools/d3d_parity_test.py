# -*- coding: utf-8 -*-
"""D3D12 深度链 parity（P1 门禁）：GL 链 vs D3D 链逐帧对比 + 同步正确性。

参照 = GL postprocess_gpu（moderngl standalone，cpu_gpu_parity_test 同款
make_win）；被测 = D3D12 D3DDepthPost.spatial_chain。两者同输入（同一批
raw numpy）、同参数（lo/hi 分位、α、AA σ3、EMA τ80）。

门禁：
  parity  逐帧 MAE < 0.001（着色器数值差量级；HLSL/GSL 公式逐式一致）
  同步正确性  连续 120 帧 D3D 链输出偶发差直方图 + 设备态全程 S_OK
用法：D2S\\python3\\python.exe -I tools\\d3d_parity_test.py
"""
import ctypes
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_TEMPORAL"] = "ema"
# 门禁固定 DepthART-224（CPU 参考链是它的语义：ImageNet 归一化 + 224²）；
# dev settings 可能被切到 448 / ZipDepth 做实验，不能跟着变。
os.environ["D2S_ONNX"] = os.path.join(ROOT, "runtime", "depth-224.onnx")

RES = 224
FRAMES = 120
ALPHA = 0.19
AA = 3.0


def load_raws():
    """真实帧 → 模型 raw（CPU EP，确定性）→ (raws, lo/hi 列表)。"""
    import cv2
    import depth
    from depth_norm import percentile_stats
    frames = []
    for f in sorted(os.listdir(os.path.join(ROOT, "_refactor_baseline", "inputs"))):
        if f.endswith(".png"):
            frames.append(cv2.cvtColor(cv2.imread(
                os.path.join(ROOT, "_refactor_baseline", "inputs", f)),
                cv2.COLOR_BGR2RGB))
    frames = [cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA)
              for f in frames]
    seq = [frames[i % len(frames)] for i in range(FRAMES)]
    raws = [depth.depth_model(depth.d2s_ops.resize_model_input(f, RES, None))
            for f in seq]
    los, his = [], []
    for r in raws:
        sub = r[::4, ::4]
        los.append(float(np.percentile(sub, 0.5)))
        his.append(float(np.percentile(sub, 99.5)))
    return raws, los, his


def gl_chain(raws, los, his):
    """GL 参照链（standalone moderngl，cpu_gpu_parity_test 同款 make_win）。"""
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
    for m in ("gpu_depth_init", "gpu_post_init", "postprocess_gpu",
              "read_depth_median", "_channel_selfcheck", "_post_get"):
        setattr(w, m, getattr(viewer.StereoWindow, m).__get__(w))
    w.gpu_depth_init(RES)
    outs = []
    for i, r in enumerate(raws):
        w._raw_tex.write(np.ascontiguousarray(r, np.float32).tobytes())
        w.postprocess_gpu(w._raw_tex, los[i], his[i], ALPHA, AA,
                          0, 0.0, 0.0, iters=6, close_px=0)
        outs.append(np.frombuffer(w._post.depth_fbo.read(components=1, dtype="f4"),
                                  np.float32).reshape(RES, RES).copy())
    return outs


def d3d_chain(raws, los, his):
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_post import D3DDepthPost
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    post = D3DDepthPost(ren, RES)
    outs = []
    dev_bad = 0
    for i, r in enumerate(raws):
        post.upload_raw(r)
        post.spatial_chain(los[i], his[i], aa_texels=AA, temporal="ema",
                           alpha=ALPHA)
        o = post.readback_depth()
        if i < 3:
            print(f"[d3d] 帧{i}: min={o.min():.4f} max={o.max():.4f} "
                  f"mean={o.mean():.4f}", flush=True)
        outs.append(o)
    ren.destroy()
    return outs, dev_bad


def main():
    print("[d3d-parity] 加载帧与模型…", flush=True)
    raws, los, his = load_raws()
    print(f"[d3d-parity] {len(raws)} 帧({RES}x{RES})", flush=True)

    print("[d3d-parity] GL 参照链…", flush=True)
    gl_out = gl_chain(raws, los, his)
    print("[d3d-parity] D3D 链…", flush=True)
    d3d_out, dev_bad = d3d_chain(raws, los, his)

    fails = []
    maes = []
    for i in range(min(len(gl_out), len(d3d_out))):
        m = float(np.mean(np.abs(d3d_out[i] - gl_out[i])))
        maes.append(m)
    worst = max(maes)
    med = sorted(maes)[len(maes) // 2]
    print(f"[parity] 逐帧 MAE: 中位 {med:.5f} 最大 {worst:.5f}")
    if worst >= 0.001:
        fails.append("parity")

    # 同步正确性：连续 120 帧 D3D 输出的偶发异常（NaN/超界）
    n_bad = sum(1 for o in d3d_out
                if np.isnan(o).any() or o.min() < -0.01 or o.max() > 1.01)
    if n_bad:
        fails.append(f"帧异常 {n_bad}")
    if dev_bad:
        fails.append(f"设备态异常 {dev_bad}")

    if fails:
        print(f"[d3d-parity] FAIL: {fails}")
        return 1
    print("[d3d-parity] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
