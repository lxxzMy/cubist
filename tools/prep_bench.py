# -*- coding: utf-8 -*-
"""PREP 降采样 pass 的 GPU 时间成本：单点双线性 vs N×N 多抽头。

产品 PREP_FS 现状 = 单点双线性（1920×1080 → flat 192×784 = 150,528 px）。
本工具在同一 PREP 几何上测 1/4×4/8×8 抽头的单 pass GPU 时间（200 次连录）。

用法：D2S\\python3\\python.exe -I tools\\prep_bench.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))

import numpy as np  # noqa: E402

from gpu import adapter as _ad  # noqa: E402
from gpu import d3drender as dr  # noqa: E402
from gpu.hlsl import FULLSCREEN_VS, compile_shader  # noqa: E402
from gpu.hlsl_depth import PREP_FS  # noqa: E402

SRC_W, SRC_H = 1920, 1080
RES, FLAT_W = 224, 192
PREP_W, PREP_H = FLAT_W, (RES * RES * 3) // FLAT_W
N = 200


def variant_src(k):
    if k == 1:
        return PREP_FS
    old = "float3 col = tex_c.SampleLevel(samp, suv, 0).xyz;"
    new = f"""
    float3 col = float3(0, 0, 0);
    [unroll]
    for (int jj = 0; jj < {k}; ++jj) {{
        [unroll]
        for (int ii = 0; ii < {k}; ++ii) {{
            float2 o = float2(((float)ii + 0.5) / {k} - 0.5,
                              ((float)jj + 0.5) / {k} - 0.5) / res;
            col += tex_c.SampleLevel(samp, suv + o, 0).xyz;
        }}
    }}
    col /= {k} * {k}.0;"""
    assert old in PREP_FS
    return PREP_FS.replace(old, new)


def bench(ren, pso, ctex, prep, label):
    cb = [float(RES), float(FLAT_W), 0.0, 0.0]
    # warmup
    for _ in range(5):
        ren.begin_frame()
        ctex.transition(dr.ST_PIXEL_SRV)
        prep.state = dr.ST_COMMON
        prep.transition(dr.ST_RENDER_TARGET)
        ren.draw_pass(pso, [ctex], cb, prep, PREP_W, PREP_H)
        prep.state = dr.ST_COMMON
        ren.submit(); ren.wait()
    t0 = time.perf_counter()
    ren.begin_frame()
    for _ in range(N):
        ctex.transition(dr.ST_PIXEL_SRV)
        prep.state = dr.ST_COMMON
        prep.transition(dr.ST_RENDER_TARGET)
        ren.draw_pass(pso, [ctex], cb, prep, PREP_W, PREP_H)
        prep.state = dr.ST_COMMON
    ren.submit()
    ren.wait()
    dt = (time.perf_counter() - t0) * 1000.0 / N
    print(f"{label:<12}{dt:>8.4f} ms/pass")
    return dt


def main():
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    ctex = ren.upload_tex(SRC_W, SRC_H, dr.FMT_BGRA8, renderable=False)
    arr = np.random.randint(0, 255, (SRC_H, SRC_W, 4), np.uint8)
    ren.upload(ctex, np.ascontiguousarray(arr))
    ren.begin_frame()
    ren.copy_upload(ctex)
    ren.submit(); ren.wait()
    prep = ren.create_tex(PREP_W, PREP_H, dr.FMT_R32F, renderable=True)
    print(f"PREP {SRC_W}x{SRC_H} → {PREP_W}x{PREP_H} "
          f"({PREP_W * PREP_H} px)，{N} 次/pass")
    vs = compile_shader("prep_vs", FULLSCREEN_VS, "vs")
    for k in (1, 4, 8):
        pso = ren.create_pso(
            f"prep_k{k}", vs,
            compile_shader(f"prep_k{k}", variant_src(k), "ps"),
            dr.FMT_R32F)
        bench(ren, pso, ctex, prep, f"{k}x{k} taps")
    ren.destroy()


if __name__ == "__main__":
    main()
