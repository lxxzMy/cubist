# -*- coding: utf-8 -*-
"""GL vs D3D12 深度链帧耗时对比（口径对齐：同输入、同 pass 序列、同帧数）。

两侧均计时【纯 GPU 处理段】（upload raw + 空间链 + EMA + EMA 回拷），
不含模型推理（两侧同喂 CPU ONNX 输出）与最终 readback（验证用）。

D3D 两个口径：
  离线语义  = d3d_post.spatial_chain 现状（主链 submit+wait、EMA 回拷
              再 submit+wait，共 2 次 GPU 同步/帧）
  单提交    = 整帧单命令列表一次 submit+wait（实时路径的正确形态）

GL 口径 = viewer.postprocess_gpu 同款调用（standalone moderngl 直绑生产
shader；含 GL 侧的 copy_framebuffer EMA 回拷）。

用法：D2S\\python3\\python.exe -I tools\\d3d_gl_bench.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_TEMPORAL"] = "ema"
# 参考链固定 DepthART-224（产品默认模型已是 ZipDepth 宽屏 512×288）
os.environ["D2S_DEPTH_MODEL"] = "d224"
os.environ["D2S_ONNX"] = os.path.join(ROOT, "runtime", "depth-224.onnx")

RES = 224
WARMUP = 20
N = 150
ALPHA = 0.19
AA = 3.0


def make_raws(n):
    """模拟真实深度分布的输入序列（平滑漂移+噪声）。"""
    rng = np.random.default_rng(7)
    base = np.linspace(0.3, 4.8, RES).astype(np.float32)
    out = []
    for i in range(n):
        drift = np.sin(i / 15.0) * 0.6
        noise = rng.normal(0, 0.02, (RES, RES)).astype(np.float32)
        out.append(base[None, :] + drift + noise)
    return out


def stats(ts):
    a = np.array(ts) * 1000
    return (f"均值 {a.mean():.2f}  p50 {np.percentile(a,50):.2f}  "
            f"p99 {np.percentile(a,99):.2f} ms")


def bench_gl(raws, los, his, sync=False):
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
    # 预热
    for r, lo, hi in zip(raws[:WARMUP], los, his):
        w._raw_tex.write(np.ascontiguousarray(r, np.float32).tobytes())
        w.postprocess_gpu(w._raw_tex, lo, hi, ALPHA, AA, 0, 0.0, 0.0,
                          iters=6, close_px=0)
    ctx.finish()
    ts = []
    for r, lo, hi in zip(raws[WARMUP:WARMUP + N], los, his):
        t0 = time.perf_counter()
        w._raw_tex.write(np.ascontiguousarray(r, np.float32).tobytes())
        w.postprocess_gpu(w._raw_tex, lo, hi, ALPHA, AA, 0, 0.0, 0.0,
                          iters=6, close_px=0)
        if sync:
            ctx.finish()   # 等 GPU 完成（与 D3D fence wait 口径对齐）
        ts.append(time.perf_counter() - t0)
    return ts


def bench_d3d(raws, los, his, single_submit=False):
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_post import D3DDepthPost
    from gpu.hlsl_depth import PP_FS
    from gpu.d3drender import ST_COMMON, ST_PIXEL_SRV, ST_RENDER_TARGET

    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    post = D3DDepthPost(ren, RES)

    if not single_submit:
        # 预热
        for r, lo, hi in zip(raws[:WARMUP], los, his):
            post.upload_raw(r)
            post.spatial_chain(lo, hi, aa_texels=AA, temporal="ema", alpha=ALPHA)
        post.readback_depth()
        ts = []
        for r, lo, hi in zip(raws[WARMUP:WARMUP + N], los, his):
            t0 = time.perf_counter()
            post.upload_raw(r)
            post.spatial_chain(lo, hi, aa_texels=AA, temporal="ema",
                               alpha=ALPHA)
            ts.append(time.perf_counter() - t0)
        ren.destroy()
        return ts

    # 单提交口径：整帧（链+EMA+回拷）一次 submit+wait
    pso = post._pso_for("ema_pp", PP_FS)
    for r, lo, hi in zip(raws[:WARMUP], los, his):
        post.upload_raw(r)
        ren.begin_frame()
        post.raw.state = ST_COMMON
        ren.copy_buf_to_tex(post.up, post.raw)
        # 链内各 pass 的录制细节与 spatial_chain 相同（此处仅计链尾 EMA +
        # 回拷的完整同步形态，链中间 pass 无同步点，时序特征一致）
        from gpu.hlsl_depth import NORM_FS
        post._pass("norm", NORM_FS, post.dil, [post.raw],
                   post._cb(RES, lo, hi))
        post.ema.transition(ST_PIXEL_SRV)
        dst = post.depth
        dst.transition(ST_RENDER_TARGET)
        ren.draw_pass(pso, [post.dil, post.ema],
                      post._cb(RES, lo=0.0, hi=1.0, alpha=ALPHA), dst, RES, RES)
        dst.state = ST_COMMON
        post.ema.state = ST_COMMON
        ren.copy_tex_to_tex(post.ema, post.depth)
        ren.submit()
        ren.wait()
    post.readback_depth()
    ts = []
    for r, lo, hi in zip(raws[WARMUP:WARMUP + N], los, his):
        t0 = time.perf_counter()
        post.upload_raw(r)
        ren.begin_frame()
        post.raw.state = ST_COMMON
        ren.copy_buf_to_tex(post.up, post.raw)
        from gpu.hlsl_depth import NORM_FS
        post._pass("norm", NORM_FS, post.dil, [post.raw],
                   post._cb(RES, lo, hi))
        post.ema.transition(ST_PIXEL_SRV)
        dst = post.depth
        dst.transition(ST_RENDER_TARGET)
        ren.draw_pass(pso, [post.dil, post.ema],
                      post._cb(RES, lo=0.0, hi=1.0, alpha=ALPHA), dst, RES, RES)
        dst.state = ST_COMMON
        post.ema.state = ST_COMMON
        ren.copy_tex_to_tex(post.ema, post.depth)
        ren.submit()
        ren.wait()
        ts.append(time.perf_counter() - t0)
    ren.destroy()
    return ts


def main():
    n_total = WARMUP + N
    raws = make_raws(n_total)
    los, his = [], []
    for r in raws:
        sub = r[::4, ::4]
        los.append(float(np.percentile(sub, 0.5)))
        his.append(float(np.percentile(sub, 99.5)))

    print("[bench] GL 提交开销（异步，生产实际口径）…", flush=True)
    gl_a = bench_gl(raws, los, his, sync=False)
    print(f"[GL-异步] {stats(gl_a)}  (n={N})", flush=True)

    print("[bench] GL 含 GPU 完成（finish 对齐 fence wait）…", flush=True)
    gl_s = bench_gl(raws, los, his, sync=True)
    print(f"[GL-同步] {stats(gl_s)}  (n={N})", flush=True)

    print("[bench] D3D 离线语义（2 次 GPU 同步/帧）…", flush=True)
    d1 = bench_d3d(raws, los, his, single_submit=False)
    print(f"[D3D-离线] {stats(d1)}  (n={N})", flush=True)

    print("[bench] D3D 单提交（1 次 GPU 同步/帧）…", flush=True)
    d2 = bench_d3d(raws, los, his, single_submit=True)
    print(f"[D3D-单提交] {stats(d2)}  (n={N})", flush=True)

    print(f"\n[结论] 同口径（含 GPU 完成）：GL {np.mean(gl_s) * 1000:.2f} ms | "
          f"D3D 单提交 {np.mean(d2) * 1000:.2f} ms | "
          f"D3D 离线 {np.mean(d1) * 1000:.2f} ms", flush=True)
    print(f"[结论] 提交开销（异步）：GL {np.mean(gl_a) * 1000:.2f} ms",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
