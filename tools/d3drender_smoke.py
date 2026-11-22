# -*- coding: utf-8 -*-
"""R1 冒烟：d3drender.dll 全链（设备→上传→PSO pass→读回）数值正确性。

也是迁移各阶段的常驻回归门禁（每阶段提交前必跑）。
用法：python tools/d3drender_smoke.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D2S"))

import numpy as np

from gpu import d3drender as dr
from gpu.hlsl import compile_shader, FULLSCREEN_VS

PS_SOLID = """
float4 PSMain() : SV_Target { return float4(0.25, 0.5, 0.75, 1.0); }
"""
PS_UV = """
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 PSMain(VOut i) : SV_Target { return float4(i.uv.x, i.uv.y, 0.0, 1.0); }
"""


def main():
    r = dr.D3DRenderer(debug=True)
    print("[1] device OK, ctx=%s" % bool(r.ctx))

    vs = compile_shader("fs_vs", FULLSCREEN_VS, kind="vs")

    # 纯色 PS：上传纹理 + pass 到 64x64 BGRA + 读回验证
    pso_solid = r.create_pso("solid", vs, compile_shader("solid", PS_SOLID))
    src = r.create_tex(64, 64, dr.FMT_BGRA8, renderable=True)
    dst = r.create_tex(64, 64, dr.FMT_BGRA8, renderable=True)

    r.begin_frame()
    dst.transition(dr.ST_RENDER_TARGET)
    r.draw_pass(pso_solid, [], [], dst, 64, 64)
    r.submit(); r.wait()

    arr = r.readback(dst, np.uint8)
    px = arr[32, 32]   # BGRA
    assert abs(int(px[0]) - 191) <= 2, "B=0.75→191 got %s" % px
    assert abs(int(px[1]) - 128) <= 2, "G=0.5→128 got %s" % px
    assert abs(int(px[2]) - 64) <= 2, "R=0.25→64 got %s" % px
    print("[2] solid pass readback OK  px(BGR)=%s" % (px[:3].tolist(),))

    # UV PS + SRV 输入：dst 作 SRV 二次 pass 验证描述符链
    pso_uv = r.create_pso("uv", vs, compile_shader("uv", PS_UV))
    r.begin_frame()
    src.transition(dr.ST_RENDER_TARGET)
    r.draw_pass(pso_uv, [], [], src, 64, 64)
    r.submit(); r.wait()

    r.begin_frame()
    dst.transition(dr.ST_RENDER_TARGET)
    r.draw_pass(pso_uv, [], [], dst, 64, 64)   # 复用同一 pso（缓存命中路径）
    r.submit(); r.wait()

    arr2 = r.readback(src, np.uint8)
    tl = arr2[0, 0]      # uv(0,0) → R=0 B=0
    br = arr2[63, 63]    # uv(1,1) → R≈255 G≈255
    assert tl[2] <= 8 and tl[1] <= 8, "uv TL got %s" % tl
    assert br[2] >= 240 and br[1] >= 240, "uv BR got %s" % br
    print("[3] uv pass + pso cache OK  tl=%s br=%s" % (tl[:3].tolist(),
                                                        br[:3].tolist()))

    # R32F 上传→拷贝→SRV 采样链（深度管线入口语义）
    pso_sample = r.create_pso("sample", vs,
                              compile_shader("sample", PS_SAMPLE_R32F),
                              rtv_fmt=dr.FMT_R32F)
    up = r.upload_tex(128, 64, dr.FMT_R32F)
    out = r.create_tex(128, 64, dr.FMT_R32F, renderable=True)
    data = np.arange(128 * 64, dtype=np.float32).reshape(64, 128) / 8192.0
    r.upload(up, data)
    r.begin_frame()
    r.copy_upload(up)                 # COMMON→COPY_DEST→ALL_SRV 屏障内含
    out.transition(dr.ST_RENDER_TARGET)
    r.draw_pass(pso_sample, [up], [], out, 128, 64)
    r.submit(); r.wait()

    arr3 = r.readback(out, np.float32)
    d = np.abs(arr3 - data).max()
    assert d < 1e-5, "R32F copy+sample drift %g" % d
    print("[4] R32F upload→copy→srv→pass→readback OK  max|d|=%.2e" % d)

    # 生命周期：tex 释放 + renderer 销毁
    src.release(); dst.release(); up.release(); out.release()
    r.destroy()
    print("[5] lifecycle release/destroy OK")
    print("SMOKE PASS")


PS_SAMPLE_R32F = """
Texture2D t0 : register(t0);
SamplerState s0 : register(s0);
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 PSMain(VOut i) : SV_Target {
    return t0.Sample(s0, i.uv).x;
}
"""


if __name__ == "__main__":
    main()
