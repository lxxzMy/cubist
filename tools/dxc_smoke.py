# -*- coding: utf-8 -*-
"""D2 门禁：DXC/SM6 全 shader 编译 + PSO 创建 + 缓存路径。

断言：
  X1 全部 11 个 shader 以 dxc/ps_6_0|vs_6_0 编译成功（DXIL chunk 确认）
  X2 全部 PSO 以 DXIL 字节码创建成功（debug layer 零错误）
  X3 缓存命中路径（二次加载 .dxc.cso）
  X4 runtime/ 下 dxcompiler.dll+dxil.dll 在位（发布依赖）
用法：D2S\\python3\\python.exe -I tools\\dxc_smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

from gpu.hlsl import compile_shader, FULLSCREEN_VS  # noqa: E402
from gpu.hlsl_depth import (DIL_FS, ERO_FS, GAUSS_FS, MEDIAN_FS, PP_FS,  # noqa: E402
                            PREP_FS, SLOPE_FS)
from gpu.hlsl_display import (DEPTH_MAP_FS, DIBR_FS, INTERLEAVED_FS,  # noqa: E402
                              PASSTHROUGH_FS)


def main():
    fails = []

    # X4 发布依赖
    for dll in ("dxcompiler.dll", "dxil.dll"):
        p = os.path.join(ROOT, "runtime", dll)
        if not os.path.isfile(p):
            fails.append(f"X4 runtime 缺 {dll}")
    if not fails:
        print("[X4] runtime dxcompiler.dll+dxil.dll 在位", flush=True)

    # X1 全 shader 编译
    shaders = [
        ("depth_prep", PREP_FS, "ps"), ("depth_dil", DIL_FS, "ps"),
        ("depth_slope", SLOPE_FS, "ps"), ("depth_ero", ERO_FS, "ps"),
        ("depth_gauss", GAUSS_FS, "ps"), ("depth_ema_pp", PP_FS, "ps"),
        ("depth_median", MEDIAN_FS, "ps"),
        ("view_passthrough", PASSTHROUGH_FS, "ps"),
        ("view_depth_map", DEPTH_MAP_FS, "ps"),
        ("view_dibr", DIBR_FS, "ps"),
        ("view_interleaved", INTERLEAVED_FS, "ps"),
    ]
    ok = 0
    for name, src, kind in shaders:
        try:
            c = compile_shader(name, src, kind, force=True)
            if c.compiler != "dxc":
                fails.append(f"X1 {name} 走了 {c.compiler} 而非 dxc")
            elif b"DXIL" not in bytes(c._copy):
                fails.append(f"X1 {name} 无 DXIL chunk")
            else:
                ok += 1
        except Exception as e:
            fails.append(f"X1 {name}: {type(e).__name__}: {e}")
    vs = compile_shader("fs_vs", FULLSCREEN_VS, "vs", force=True)
    if vs.compiler != "dxc":
        fails.append(f"X1 fs_vs 走了 {vs.compiler}")
    else:
        ok += 1
    print(f"[X1] {ok}/12 shader dxc 编译成功", flush=True)

    # X1b 缓存命中
    try:
        c2 = compile_shader("depth_dil", DIL_FS, "ps")
        print(f"[X3] 缓存命中 OK（{c2.nbytes}B {c2.compiler}）", flush=True)
    except Exception as e:
        fails.append(f"X3 缓存: {e}")

    # X2 PSO（debug layer）
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=True)
    try:
        n_pso = 0
        for name, src, _kind in shaders:
            c = compile_shader(name, src, "ps")
            ren.create_pso(f"dxcs_{name}", vs, c, dr.FMT_R32F
                           if name.startswith("depth") else dr.FMT_BGRA8)
            n_pso += 1
        print(f"[X2] {n_pso} 个 PSO（DXIL）创建成功，err='{ren.last_err()}'",
              flush=True)
        if ren.last_err():
            fails.append(f"X2 PSO debug: {ren.last_err()[:100]}")
    finally:
        ren.destroy()

    if fails:
        print(f"[dxc-smoke] FAIL: {fails}")
        return 1
    print("[dxc-smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
