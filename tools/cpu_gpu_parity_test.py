# -*- coding: utf-8 -*-
"""CPU vs GPU 深度管线端到端对拍：两条**产品**路径同输入 → 逐级数值对比。

背景：CPU 路径（depth.predict_depth：d2s_ops 链）与 GPU 路径（main.
_gpu_depth_step：GL 着色器链）是同一产品语义的两套实现——黄金基线门禁
只盖 CPU 链（depth 探针），GPU 链只有着色器对拍（mc_gl_test）与行为门禁
（e2e）；两条管线从没数值对过。本工具在发布包解释器下（DML 只在彼处）
按检查点对拍，防"改了一边忘另一边"：

  A1 模型 EP 对拍   ：同一预处理输入 → DML vs CPU-EP 的 raw 输出
  A2 同口径后处理   ：同一 raw 序列喂两侧，GPU 归一化强制用 CPU 的
                      (lo,hi) → 剩余差 = AA 核/EMA α/着色器浮点（实现级）
  A3 产品语义后处理 ：两侧各按自己的归一化（GPU=逐帧分位 / CPU=固定标定
                      区间）→ 差 = **策略分叉**（已知项，量化用）
  A4 全链端到端     ：彩色帧 → CPU predict_depth vs GPU 预处理+DML+分位
                      +后处理 → 读回 depth_tex（含预处理着色器/编排差）
  A4a 附带          ：prep 纹理链（预处理着色器→DML→读回）vs 直喂 x 的 raw

对齐后基线（2026-11-20 当日三修后：AA 两侧统一 19-tap σ=3 + 边缘复制；
GPU EMA 首帧直通对齐 CPU）：
  A1 raw EP 1.5e-06 | A4a prep 纹理链 1.1e-06（EP/传输完全等价）
  A2 同口径实现残差 0.0001 / 最大 0.0003 —— 浮点精度级等价
  A3 策略分叉 0.0012（GPU ::4 子采样 vs CPU 全量分位——唯一残余差）
  A4 端到端 0.0013 / 最大 0.0017
历史（修前）：A2=0.0227（CPU 19-tap BORDER_CONSTANT 零填充 vs GPU
17-tap 钳制的核宽+边界分叉）；A4 曾 0.0734（EMA 种子 0.5 vs 首帧直通的
暂态差，预热段隔离后消）。阈值=当前基线+余量；改任一侧 AA/EMA/归一化须复跑。

用法（发布包解释器，仓库根）：
  release\\Cubist-1.0.224\\python\\python.exe -I tools\\cpu_gpu_parity_test.py
  [--rel Cubist-1.0.224] [--frames 24] [--steady 8]
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "_refactor_baseline", "inputs")


class _DummyU:
    value = 0.0


class _DummyProg:
    """FakeWin 的显示着色器占位（gpu_depth_init 只写 u_depth_resolution）。

    必须 __contains__=False：`'u_x' in prog` 对无 contains/iter 的对象
    回退到整数下标迭代协议（逐个 getitem 直到 IndexError）——占位若
    永不抛 IndexError 就是无限循环（单核打满，py-spy 实锤过）。
    """

    def __getitem__(self, k):
        if not isinstance(k, str):
            raise IndexError(k)
        return _DummyU()

    def __contains__(self, _k):
        return False


class _CtxProxy:
    """standalone（无窗口）上下文没有 ctx.screen——产品代码假定有窗。

    代理透传一切、screen 替成 noop（产品端不动：真实运行恒有窗口）。
    """

    def __init__(self, ctx):
        object.__setattr__(self, "_real", ctx)

    def __getattr__(self, k):
        return getattr(self._real, k)

    def __setattr__(self, k, v):
        setattr(self._real, k, v)

    @property
    def screen(self):
        class _NoFB:
            @staticmethod
            def use():
                pass
        return _NoFB()


def make_win(ctx, res):
    """mc_repro 式 FakeWin：承载产品 GL 后处理链（P1-P3 解耦后的委托面）。"""
    import viewer

    class FakeWin:
        pass

    w = FakeWin()
    w.ctx = _CtxProxy(ctx)
    w._gpu_res = res
    w._texture_size = (res, res)
    w._mc_on = False                      # 对拍不含 MC 层（GPU-only 特性）
    w.depth_tex = None
    w._depth_tex_size = None
    w.color_tex = ctx.texture((res, res), 3, dtype="f1")
    w.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    for k in ("prog", "anaglyph_prog", "interleaved_prog",
              "vertical_interleaved_prog"):
        setattr(w, k, _DummyProg())
    for m in ("gpu_depth_init", "gpu_preprocess", "gpu_post_init",
              "postprocess_gpu", "read_depth_median", "_channel_selfcheck",
              "_post_get"):
        setattr(w, m, getattr(viewer.StereoWindow, m).__get__(w))
    w.gpu_depth_init(res)
    return w


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float32)
                                - np.asarray(b, np.float32))))


def seq_mae_report(outs_a, outs_b, steady):
    per = [mae(a, b) for a, b in zip(outs_a, outs_b)]
    st = per[steady:]
    ctr = [mae(a[16:-16, 16:-16], b[16:-16, 16:-16])
           for a, b in zip(outs_a, outs_b)][steady:]
    return (per, float(np.mean(st)) if st else float("nan"), max(per),
            float(np.mean(ctr)) if ctr else float("nan"))


def main():
    global moderngl
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default="Cubist-1.0.224")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--steady", type=int, default=8)
    args = ap.parse_args()

    app = os.path.join(ROOT, "release", args.rel, "app")
    sys.path.insert(0, app)
    os.environ["PATH"] = (os.path.join(ROOT, "release", args.rel, "runtime")
                          + os.pathsep + os.environ.get("PATH", ""))
    os.environ["D2S_ZEROCOPY"] = "0"       # 走读回路径：对拍管数学，不管传输
    os.environ["D2S_ONNX"] = ""

    import moderngl  # noqa: F401
    import onnxruntime as _ort
    import depth
    import viewer  # noqa: F401
    from depth_norm import percentile_stats

    res = int(depth.DEPTH_RESOLUTION)
    frames_u8 = []
    for f in sorted(os.listdir(INPUTS)):
        if f.endswith(".png"):
            frames_u8.append(cv2.cvtColor(cv2.imread(os.path.join(INPUTS, f)),
                                          cv2.COLOR_BGR2RGB))
    assert frames_u8, "无输入帧（_refactor_baseline/inputs）"
    frames_u8 = [cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA)
                 for f in frames_u8]
    seq = [frames_u8[i % len(frames_u8)] for i in range(args.frames)]

    prov = depth.depth_model.session.get_providers()[0]
    aa = float(getattr(depth, "AA_STRENGTH", 0))
    nfrm = float(getattr(depth, "DEPTH_SMOOTHING_FRAMES", 4.8))
    print(f"[parity] 解释器={sys.executable}", flush=True)
    print(f"[parity] EP={prov}  res={res}  AA_STRENGTH={aa}  "
          f"SMOOTH_FRAMES={nfrm:.2f}")
    print(f"[parity] CPU 归一化分支: FIXED_ON={depth._DEPTH_FIXED_ON} "
          f"AUTO={depth._AUTO_RANGE_ON} NORM_MODE={depth._NORM_MODE} "
          f"PLAIN={depth._PLAIN_NORM[0]} RAW={depth._RAW_MODE[0]} "
          f"METRIC={depth._IS_METRIC}")
    print(f"[parity] 序列: {len(frames_u8)} 真帧轮播 × {args.frames} 帧，"
          f"稳态=丢前 {args.steady}\n")

    fails = []

    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails.append(name)

    # ---------- A1 模型 EP 对拍（同预处理输入） ----------
    x = depth.d2s_ops.resize_model_input(seq[0], res, depth.get_patch_size())
    raw_dml = depth.depth_model(x)
    if prov == "DmlExecutionProvider":
        sess_cpu = _ort.InferenceSession(depth.depth_model.onnx_path,
                                         providers=["CPUExecutionProvider"])
        _in = sess_cpu.get_inputs()[0].name
        raw_cpu = np.asarray(
            sess_cpu.run(None, {_in: np.asarray(x, np.float32)[None]})[0],
            np.float32).squeeze()
        d1 = mae(raw_dml, raw_cpu)
        check("A1 模型 EP 对拍", d1 < 5e-3,
              f"raw MAE={d1:.2e}（DML vs CPU-EP）")
    else:
        d1 = 0.0
        print("  [skip] A1（当前 EP 非 DML，dev 解释器）")

    # ---------- 公共：raw 序列（DML 输出，确定性） ----------
    raws = [depth.depth_model(
        depth.d2s_ops.resize_model_input(f, res, depth.get_patch_size()))
        for f in seq]

    def cpu_chain():
        depth._EMA_STATE["prev"] = None
        outs = []
        for r in raws:
            outs.append(depth.postprocess_depth(np.array(r, np.float32),
                                                use_temporal_smooth=True))
        return outs

    def gpu_chain(lo_hi_fn):
        ctx = moderngl.create_standalone_context(require=330)
        w = make_win(ctx, res)
        alpha = 1.0 - math.exp(-1.0 / 60.0 / (nfrm / 60.0))
        aa_t = aa * 0.5 * (res / 224.0)
        from depth_norm import z_sub, cut_hit
        prevz = None
        outs = []
        for r in raws:
            w._raw_tex.write(np.ascontiguousarray(r, np.float32).tobytes())
            lo, hi = lo_hi_fn(r)
            # 镜像 main._gpu_depth_step 的 CPU 切判据（同函数同域）
            z = z_sub(r, lo, hi)
            cut = cut_hit(z, prevz)
            prevz = z
            w.postprocess_gpu(w._raw_tex, lo, hi, alpha, aa_t,
                              cut=cut)
            buf = w._post.depth_fbo.read(components=1, dtype="f4")
            outs.append(np.frombuffer(buf, np.float32).reshape(res, res).copy())
        ctx.release()
        return outs

    # ---------- A2 同口径后处理（GPU 归一化用 CPU 的 lo/hi） ----------
    # 镜像 postprocess_depth 实际分支（da3：全量数组 P{LOW}/{HIGH}）
    def cpu_lo_hi(r):
        sub = r[::4, ::4]      # 镜像 _da3_normalize 的 ::4 子采样（5/95 后必须同采样）
        return (float(np.percentile(sub, depth.DEPTH_PCT_LOW)),
                float(np.percentile(sub, depth.DEPTH_PCT_HIGH)))
    per2, st2, mx2, c2 = seq_mae_report(gpu_chain(cpu_lo_hi), cpu_chain(),
                                        args.steady)
    check("A2 同口径后处理", st2 < 5e-3,
          f"稳态MAE={st2:.4f} 中心区(去16px边带)={c2:.4f} 最大={mx2:.4f}"
          f"（AA核/边界语义/EMAα；中心区小而全幅大=边界效应）")

    # ---------- A3 产品语义后处理（各用各的归一化） ----------
    per3, st3, mx3, c3 = seq_mae_report(
        gpu_chain(lambda r: percentile_stats(r)[:2]), cpu_chain(), args.steady)
    strat = st3 - st2
    check("A3 产品语义后处理", st3 < 5e-3,
          f"稳态MAE={st3:.4f} 其中策略分叉≈{strat:.4f}"
          f"（GPU ::4子采样 vs CPU 全量分位 P{depth.DEPTH_PCT_LOW}/"
          f"P{depth.DEPTH_PCT_HIGH}）实现残差≈{st2:.4f}")

    # ---------- A4 全链端到端 ----------
    from gpu.dml_gl import DmlDepthRunner
    ctx = moderngl.create_standalone_context(require=330)
    w = make_win(ctx, res)
    runner = DmlDepthRunner(depth.depth_model.session, None, None, res,
                            tex_prep=w._prep_tex, tex_raw=w._raw_tex)
    alpha = 1.0 - math.exp(-1.0 / 60.0 / (nfrm / 60.0))
    aa_t = aa * 0.5 * (res / 224.0)
    depth._EMA_STATE["prev"] = None
    outs_c, outs_g, raws_g, raws_c = [], [], [], []
    # CPU-EP 逐帧推理是全场最慢一环（DINOv2 级骨干 × 笔记本 CPU），
    # A4 只取 8 帧（稳态=丢前 4）——端到端语义 8 帧已覆盖 EP/预处理/编排
    import time as _tm
    # 预热 6 帧：GPU EMA 种子 0.5 vs CPU 首帧直通是【已知暂态语义差】
    # （非回归项），先耗掉种子（0.81^13≈5%）再计量——量的是稳态一致性
    from depth_norm import z_sub as _zs, cut_hit as _ch
    _prevz = None

    def _pp_with_cut(w, raw, lo, hi, alpha, aa_t):
        """镜像 main：CPU 切判据 → postprocess_gpu(cut=...)。"""
        nonlocal _prevz
        z = _zs(raw, lo, hi)
        cut = _ch(z, _prevz)
        _prevz = z
        w.postprocess_gpu(w._raw_tex, lo, hi, alpha, aa_t,
                          cut=cut)

    for f in [seq[0]] * 6:
        depth.predict_depth(f, use_temporal_smooth=True)
        w.color_tex.write(np.ascontiguousarray(f).tobytes())
        w.gpu_preprocess()
        _rw = runner.infer(w._raw_tex)
        _lo, _hi, _m = percentile_stats(_rw)
        _pp_with_cut(w, _rw, _lo, _hi, alpha, aa_t)
        # 预热必须走与计量帧完全相同的完整链（此前简化链 lo=0/aa=0 恰好
        # 把要消除的 EMA 状态失配又注入了一遍——A4 残留 0.02 的根因）
    for f in seq[:16]:
        _t0 = _tm.perf_counter()
        _rc = depth.predict_depth(f, use_temporal_smooth=True)
        outs_c.append(_rc)
        _x = depth.d2s_ops.resize_model_input(f, res, depth.get_patch_size())
        raws_c.append(depth.depth_model(_x))
        w.color_tex.write(np.ascontiguousarray(f).tobytes())
        w.gpu_preprocess()
        raw = runner.infer(w._raw_tex)
        raws_g.append(raw)
        lo, hi, _m = percentile_stats(raw)
        _pp_with_cut(w, raw, lo, hi, alpha, aa_t)
        buf = w._post.depth_fbo.read(components=1, dtype="f4")
        outs_g.append(np.frombuffer(buf, np.float32).reshape(res, res).copy())
        print(f"[parity]   A4 帧{len(outs_c)-1} CPU推理 "
              f"{(_tm.perf_counter()-_t0)*1000:.0f}ms", flush=True)
    ctx.release()
    _r = [mae(a, b) for a, b in zip(raws_g, raws_c)]
    print(f"[parity]   A4a raw 通路（prep纹理链 vs 直喂x）MAE="
          f"{np.mean(_r):.2e} 最大={max(_r):.2e}", flush=True)
    per4, st4, mx4, c4 = seq_mae_report(outs_g, outs_c, 8)
    check("A4 全链端到端", st4 < max(8e-3, st3 + 5e-3),
          f"稳态MAE={st4:.4f} 中心区={c4:.4f} 最大={mx4:.4f}"
          f"（含预处理着色器/EP读回/编排）")

    print()
    if fails:
        print(f"[parity] FAIL: {fails}", flush=True)
        return 1
    print("[parity] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
