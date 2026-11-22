# -*- coding: utf-8 -*-
"""DIBR 偏移块：深度→2D 图像视差位移的参数换算（纯函数，无 GL）。

视差预算以【屏幕像素】记（parallax_px = 深度满偏端的全偏转；收敛面=0.5），
换算成各显示着色器的视差 uniform 域（u_parallax，单位 = IPD 半宽的 UV 比）。
两条口径（与原 viewer.render 内联式逐字对应，黄金基线在岗）：
- 单 pass 模式（Anaglyph/Interleaved/Interleaved-V）：全幅窗口宽直接换算，
  带 1e-6 下限保护；
- SBS/TAB 族：先按 1920 基准折算有效视差再换算，带 1e-6 下限保护（与单 pass 口径一致）。
"""


def half_ipd(ipd_uv):
    """IPD 半宽（UV）。眼渲染时 u_eye_offset = ∓half_ipd。"""
    return ipd_uv / 2.0


def px_to_uv(half_ipd_, win_w):
    """屏幕像素 → u_parallax 域系数（单 pass 口径，带下限保护）。"""
    return 1.0 / max(1e-6, half_ipd_ * win_w)


def parallax_uv(parallax_px, ipd_uv, win_w):
    """单 pass 模式（Anaglyph/Interleaved/Interleaved-V）的 u_parallax。"""
    return parallax_px * px_to_uv(half_ipd(ipd_uv), win_w)


def sbs_parallax_uv(parallax_px, ipd_uv, win_w):
    """SBS/TAB 族的 u_parallax（/1920 基准折算，窗口宽在两步中消去）。"""
    _par_eff = parallax_px * (win_w / 1920)
    _p_scale = 1.0 / max(1e-6, half_ipd(ipd_uv) * win_w)
    return _par_eff * _p_scale
