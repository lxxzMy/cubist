# -*- coding: utf-8 -*-
"""R4a 冒烟：显示链两段验证。

1) 着色器正确性（离屏）：Passthrough/DepthMap 渲到离屏 BGRA 纹理 → 读回
   验像素（与 R2 parity 同款读回语义，无 swapchain 状态耦合）。
2) swapchain 生命周期：隐藏窗口 FLIP(waitable) 40 帧 present（屏障+submit+
   Present 一条龙）→ 重建式 resize → 再出图。
用法：python tools/d3d_view_smoke.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D2S"))

import numpy as np

from gpu import d3drender as dr
from gpu import d3d_view as dv

W, H = 320, 240


def main():
    ren = dr.D3DRenderer(debug=False)
    win = dv.Win32Window(W, H, show=False)
    disp = dv.D3DDisplay(ren, win.hwnd, W, H)

    # 输入：左红右蓝 BGRA + 渐变 R32F
    up = ren.upload_tex(W, H, dr.FMT_BGRA8, renderable=False)
    src = np.zeros((H, W, 4), np.uint8)
    src[:, :W // 2] = (60, 60, 220, 255)    # BGRA 左=红
    src[:, W // 2:] = (220, 60, 60, 255)    # BGRA 右=蓝
    ren.upload(up, src)
    dup = ren.upload_tex(W, H, dr.FMT_R32F, renderable=False)
    ren.upload(dup, np.linspace(0.0, 1.0, W * H, dtype=np.float32)
               .reshape(H, W))
    ren.begin_frame()
    ren.copy_upload(up)
    ren.copy_upload(dup)
    ren.submit(); ren.wait()

    # ---- 1) 离屏着色器验证 ----
    out = ren.create_tex(W, H, dr.FMT_BGRA8, renderable=True)
    pso_pt = disp._pso_for("passthrough", dv.PASSTHROUGH_FS)
    pso_dm = disp._pso_for("depth_map", dv.DEPTH_MAP_FS)

    ren.begin_frame()
    out.state = dr.ST_COMMON
    out.transition(dr.ST_RENDER_TARGET)
    up.transition(dr.ST_PIXEL_SRV)
    ren.draw_pass(pso_pt, [up], [float(W), 0, 0, 0], out, W, H)
    ren.submit(); ren.wait()
    arr = ren.readback(out, np.uint8)
    px_l = arr[H // 2, 10]      # BGRA
    px_r = arr[H // 2, W - 10]
    assert px_l[2] > 180 and px_l[0] < 100, f"passthrough 左半应红={px_l}"
    assert px_r[0] > 180 and px_r[2] < 100, f"passthrough 右半应蓝={px_r}"
    print("[1] passthrough 离屏 OK  L(BGR)=%s R(BGR)=%s"
          % (px_l[:3].tolist(), px_r[:3].tolist()))

    ren.begin_frame()
    out.transition(dr.ST_RENDER_TARGET)
    dup.transition(dr.ST_PIXEL_SRV)
    ren.draw_pass(pso_dm, [dup], [float(W), 0, 0, 0], out, W, H)
    ren.submit(); ren.wait()
    arr2 = ren.readback(out, np.uint8)
    # spectral colormap（D3D uv 与纹理空间同向——真实帧 parity 实锤的
    # 垂直翻转修复后）：渐变 top=0(远→蓝) bottom=1(近→红)
    px_top = arr2[5, W // 2]     # BGRA: [0]=B [2]=R
    px_bot = arr2[H - 5, W // 2]
    assert px_top[0] > 160 and px_top[2] < 120, f"depth_map 顶部应蓝={px_top}"
    assert px_bot[2] > 150 and px_bot[0] < 120, f"depth_map 底部应红={px_bot}"
    print("[2] depth_map 离屏 OK  top(BGR)=%s bottom(BGR)=%s"
          % (px_top[:3].tolist(), px_bot[:3].tolist()))
    # 色表不得有"平台"：旧版 4 段色表在 t<=0.125 / t>=0.875 各是一整段同色，
    # 最远/最近的画面被压成单色（2026-11-24）。渐变沿 y（见上），取同一段里
    # 的两个深度值，颜色必须不一样。
    def _at(t):
        return arr2[int(H * t), W // 2].astype(int)
    for ta, tb in ((0.02, 0.10), (0.90, 0.98)):
        d = int(np.abs(_at(ta) - _at(tb)).sum())
        assert d > 30, (
            f"depth_map 色表有平台：t={ta} 与 t={tb} 颜色几乎相同"
            f"（{_at(ta).tolist()} vs {_at(tb).tolist()}）")
    print("[2b] 色表两端无平台 OK  t=0.02/0.10 → %s/%s，t=0.90/0.98 → %s/%s"
          % (_at(0.02)[:3].tolist(), _at(0.10)[:3].tolist(),
             _at(0.90)[:3].tolist(), _at(0.98)[:3].tolist()))

    # ---- 2) swapchain 生命周期 ----
    for i in range(20):
        win.pump()
        disp.render("passthrough", color_tex=up)
    for i in range(20):
        win.pump()
        disp.render("depth_map", depth_tex=dup)
    print("[3] 40 帧 present OK")

    disp.resize(200, 150)
    for i in range(10):
        win.pump()
        disp.render("passthrough", color_tex=up)
    print("[4] 重建式 resize 200x150 + 10 帧 present OK")

    win.pump()
    win.close()
    up.release(); dup.release(); out.release()
    ren.destroy()
    print("[5] 生命周期 release/destroy OK")
    print("VIEW SMOKE PASS")


if __name__ == "__main__":
    main()
