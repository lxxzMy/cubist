# -*- coding: utf-8 -*-
"""D3DStereoWindow（1.1-d3d V1）：D3D12 版产品窗口壳。

与 viewer.StereoWindow 的鸭子接口兼容（main.py 渲染循环的调用集），实现：
Win32 窗口（含 SAME_SCREEN 隐藏形态）→ D3D12 swapchain → 深度管线
（D3DDepthPrep/D3DDepthPost）→ 立体显示（Half-SBS/Interleaved/
Passthrough/DepthMap）→ overlay/光标 → present。

零 GL 依赖（cursor_sprite/dibr 为纯逻辑模块）。GL 栈保留在 viewer.py
（D2S_RENDER=gl 回退）。
"""
import ctypes
import os
import time
from ctypes import wintypes as _wt

import numpy as np

import dibr
from cursor_sprite import _CursorSprite, _cursor_base_size, \
    _real_cursor_glyph_height, _CursorInfoT
from gpu import d3drender as dr
from gpu import d3d_view as dv
from gpu.d3d_post import D3DDepthPost
from gpu.d3d_prep import D3DDepthPrep

_user32 = ctypes.WinDLL("user32")
_k32 = ctypes.WinDLL("kernel32")
_gdi32 = ctypes.WinDLL("gdi32")

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
WS_EX_NOACTIVATE = 0x8000000
WS_EX_TOOLWINDOW = 0x80
WS_EX_APPWINDOW = 0x40000
WS_EX_NOREDIRECTIONBITMAP = 0x00200000
WDA_EXCLUDEFROMCAPTURE = 0x11
# D2S_DCOMP=1：DirectComposition 交换链（逐像素 alpha 的"正规版 254"）
_DCOMP = os.environ.get("D2S_DCOMP") == "1"
# 分层窗口颜色键（COLORREF 0x00BBGGRR = RGB(1,2,3)）：只有等于它的像素透明，
# 其余全不透明；分层 + 颜色键必须走 DWM 合成 → 不会被提升为 MPO 独立翻转。
_COLORKEY = 0x00030201
# D2S_SYNC1=1：完全不透明（alpha=255）+ Present(sync=1) 对齐 VBlank
_SYNC1 = os.environ.get("D2S_SYNC1") == "1"
# D2S_COLORKEY=1：颜色键分层（实测在本机仍会被提升成独立翻转 → 撕裂回来了，
# 所以默认不用；留作对照/其它机器）。
_COLORKEY_ON = os.environ.get("D2S_COLORKEY") == "1"

_MODE_MAP = {
    "Half-SBS": "half_sbs", "Full-SBS": "full_sbs",
    "Half-TAB": "half_tab", "Full-TAB": "full_tab",
    "Interleaved": "interleaved",
    "Passthrough": "passthrough", "Depth Map": "depth_map",
}


def _screen_size():
    return (int(_user32.GetSystemMetrics(0)), int(_user32.GetSystemMetrics(1)))


def _monitor_rect(monitor_index=None):
    """目标显示器矩形 (x, y, w, h)。

    口径 = placement.monitor_rects() 的 (left, top) 排序（与设置面板列表
    一致）；mss/win32api/DXGI 三套原生顺序互不一致，不能直接用。
    失败回退主屏。
    """
    try:
        from gpu import placement as _plc
        r = _plc.monitor_rect(monitor_index)
        if r:
            return (int(r[0]), int(r[1]), int(r[2] - r[0]), int(r[3] - r[1]))
    except Exception:
        pass
    sw, sh = _screen_size()
    return (0, 0, sw, sh)


def _set_exstyle(hwnd, add=0, clear=0):
    get = _user32.GetWindowLongPtrW
    setw = _user32.SetWindowLongPtrW
    get.restype = ctypes.c_ssize_t
    get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    setw.restype = ctypes.c_ssize_t
    setw.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    st = get(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
    setw(ctypes.c_void_p(hwnd), GWL_EXSTYLE, (st | add) & ~clear)


class D3DStereoWindow:
    """D3D12 产品窗口（主路径）。"""

    def __init__(self, capture_mode="Monitor", monitor_index=0,
                 display_mode="Half-SBS", fill_16_9=True, convergence=0.5,
                 parallax_px=12.0, ipd_uv=0.064, interleave_phase=0,
                 feather_enabled=False, cursor_overlay=False, show_fps=False,
                 frame_size=(1280, 720), vsync=False, fix_aspect=False,
                 specify_display=False, same_screen=True, **_kw):
        from utils import VERSION  # noqa: F401  （导入期健康检查）
        self.display_mode = display_mode
        self._mode = _MODE_MAP.get(display_mode, "half_sbs")
        self.fill_16_9 = bool(fill_16_9)
        self.fix_aspect = fix_aspect
        self.specify_display = specify_display
        self.same_screen = bool(same_screen)
        self.convergence = float(convergence)
        self.parallax_px = float(parallax_px)
        self.ipd_uv = float(ipd_uv)
        self.interleave_phase = float(interleave_phase)
        self.feather_enabled = int(bool(feather_enabled))
        self.feather_width = 0.05
        self.vsync = bool(vsync)
        self.use_3d = False
        self._frame_has_depth = False
        self._texture_size = tuple(frame_size)
        self.actual_fps = 0.0
        self.total_latency = 0.0

        # 窗口尺寸/样式：SAME_SCREEN → 无边框全屏覆盖层（WS_POPUP，铺满
        # 被采集那块屏的原点）；否则普通窗口（带边框，客户区=frame_size）
        # 选卡/输出**按 MONITOR_INDEX 指向的那块屏解析**：双屏分属两块 GPU
        # 时（主屏 NVIDIA + 副屏 AMD）不能用全局输出序号。
        from gpu import placement as _plc
        self.adapter, self.output_index, _mrect = _plc.pick_for_monitor(
            monitor_index, refresh=True)
        if self.same_screen:
            _mr = _mrect or _monitor_rect(monitor_index)
            w, h = int(_mr[2] - _mr[0]), int(_mr[3] - _mr[1])
            win_style, win_pos = WS_POPUP, (_mr[0], _mr[1])
            show = False
        else:
            w, h = int(frame_size[0]), int(frame_size[1])
            win_style, win_pos = WS_OVERLAPPEDWINDOW, None
            show = True
        self.monitor_index = monitor_index
        self._pending_geom = None      # WM_DISPLAYCHANGE 后的待应用几何
        self._client_w, self._client_h = w, h
        self._mon = (_mr[0], _mr[1], w, h) if self.same_screen else None

        # D3D12 后端（与上面的解析同卡）
        self.ren = dr.D3DRenderer(
            luid=(self.adapter.luid if self.adapter else 0), debug=False)
        self.win = dv.Win32Window(w, h, show=False, style=win_style,
                                  pos=win_pos)
        self.hwnd = self.win.hwnd
        # 【必须 LAYERED】实测：去掉分层后切 3D 直接黑屏——非分层的不透明
        # 窗口会被 DWM 当成真正的遮挡者，窗口下面的内容不再合成；而我们的
        # 架构是"采集自己窗口下面的桌面"（WDA 把自己的窗口排除出采集），
        # 底下没合成 = 采到黑。分层窗口 DWM 不能假设覆盖（有 alpha），
        # 下方内容照常合成，采集才拿得到画面。
        # D2S_DCOMP=1：改走 DirectComposition（交换链 PREMULTIPLIED alpha
        # + 全 1 alpha）——语义上是"半透明平面"，DWM 同样必须合成，但视觉
        # 完全不透明，彻底去掉 alpha=254。此时窗口不能是 LAYERED（DComp
        # 与分层窗口互斥），改 NOREDIRECTIONBITMAP + WM_NCHITTEST 穿透。
        if _DCOMP:
            _set_exstyle(self.hwnd, WS_EX_NOREDIRECTIONBITMAP |
                         WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                         WS_EX_APPWINDOW | WS_EX_LAYERED)
        else:
            _set_exstyle(self.hwnd, WS_EX_LAYERED | WS_EX_TRANSPARENT |
                         WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, WS_EX_APPWINDOW)
            # 【颜色键，不是 alpha】用 LWA_COLORKEY 而不是 LWA_ALPHA：
            # alpha<255 会让整窗 0.4% 透明（能透过去看到底下的东西）；
            # 颜色键则"只有等于 key 的像素透明、其余 100% 不透明"，而分层
            # 窗口 + 颜色键在 Windows 上**必须走 DWM 合成**（无法提升为 MPO
            # 独立翻转），滚动撕裂的根因同样被去掉。
            # key = RGB(1,2,3)（COLORREF 0x00BBGGRR）：视频里几乎不出现，
            # 万一出现也只丢一个近黑像素，肉眼不可见。
            _user32.SetLayeredWindowAttributes(
                ctypes.c_void_p(self.hwnd),
                _COLORKEY if _COLORKEY_ON else 0,
                0 if _COLORKEY_ON else 254,
                0x1 if _COLORKEY_ON else 0x2)   # 默认仍是 alpha=254
        if show:
            self.show()

        self.disp = dv.D3DDisplay(self.ren, self.hwnd, w, h, use_flip=True)
        self.disp.fill_16_9 = self.fill_16_9
        self.prep = D3DDepthPrep(self.ren, 224)
        self.post = D3DDepthPost(self.ren, 224)
        self._res = 224
        self._post_inited = True
        # FPS HUD（GL 同款：Pillow 文字 → 独立 OverlayDraw，每眼底左、横向压缩）
        self.show_fps = bool(show_fps)
        self._hud_ov = dv.OverlayDraw(self.ren, self.disp) \
            if self.show_fps else None
        self._hud_arr = None
        self._hud_pad = 10
        self._hud_t = 0.0
        self._hud_txt = ""
        self._hud_cache = {}
        self.win.on_resize = self._on_resize
        self.win.on_display_change = self._on_display_change

        # 光标
        self._color_tex = None
        self._cursor_sprite = _CursorSprite() if cursor_overlay else None
        self._cursor_ov = dv.OverlayDraw(self.ren, self.disp) \
            if cursor_overlay else None
        self._curs_cache = {}
        self._cursor_drawn_seq = -1
        self._render_seq = 0
        self._last_pos = None
        self._cur_move_t = 0.0      # 最近一次鼠标移动时刻（探测门控）
        self._curs_alpha = 1.0      # 应用自带光标的位图透明度（0.25s 采样）
        self._curs_alpha_t = 0.0

        # uniform 语义参数（IC 搜索默认与 GL 同：D2S_IC_SEARCH 未设 = 关）
        self.disp.set_params(
            convergence=self.convergence, parallax_px=self.parallax_px,
            ipd_uv=self.ipd_uv, phase=self.interleave_phase,
            feather_enabled=self.feather_enabled,
            feather_width=self.feather_width,
            fill=0.0)   # 遮挡补洞/IC 搜索链路已从 shader 删除（用户定案）

    # ---- 窗口 -----------------------------------------------------------
    def show(self):
        _user32.ShowWindow(ctypes.c_void_p(self.hwnd), 5)

    def hide(self):
        _user32.ShowWindow(ctypes.c_void_p(self.hwnd), 0)

    def pump(self):
        self.win.pump()

    def is_visible(self):
        return bool(_user32.IsWindowVisible(ctypes.c_void_p(self.hwnd)))

    def should_close(self):
        """WM_CLOSE 收到 = 用户要求退出进程（产品语义与 GL 路径一致）。"""
        return bool(self.win.closed)

    def set_title(self, t):
        _user32.SetWindowTextW(ctypes.c_void_p(self.hwnd), str(t))

    def _on_resize(self, w, h):
        if w > 0 and h > 0 and (int(self.disp.w), int(self.disp.h)) != (w, h):
            try:
                self.disp.resize(w, h)
                self._client_w, self._client_h = w, h
            except Exception as e:
                print(f"[view] resize 失败: {e}", flush=True)

    def _on_display_change(self):
        """WM_DISPLAYCHANGE：分辨率/显示器布局变了（切分辨率、换屏、拔插）。

        覆盖层要重新铺满"被采集的那块屏"；真正的 swapchain 重建放到
        apply_pending_geometry（渲染线程里做，失败可重试——模式切换期间
        DXGI 会短时拒绝）。
        """
        if self._mon is not None:
            self._pending_geom = _monitor_rect(self.monitor_index)
            print(f"[view] 显示模式变化 → 覆盖层待重建 "
                  f"{self._pending_geom}", flush=True)

    def apply_pending_geometry(self):
        """应用显示变化后的覆盖层几何（失败保留，节流重试）。"""
        g = self._pending_geom
        if g is None:
            return True
        mx, my, mw, mh = g
        if mw <= 0 or mh <= 0:
            self._pending_geom = None
            return True
        # 模式切换期间 DXGI 会短时拒建 swapchain：0.5s 一次，别每帧撞
        _now = time.perf_counter()
        if _now - getattr(self, "_geom_try_t", 0.0) < 0.5:
            return False
        self._geom_try_t = _now
        try:
            _user32.SetWindowPos(ctypes.c_void_p(self.hwnd), None,
                                 int(mx), int(my), int(mw), int(mh),
                                 0x0004 | 0x0010)   # NOZORDER|NOACTIVATE
            if (int(self.disp.w), int(self.disp.h)) != (int(mw), int(mh)):
                self.disp.resize(int(mw), int(mh))
            self._mon = (int(mx), int(my), int(mw), int(mh))
            self._client_w, self._client_h = int(mw), int(mh)
            self._content_key = "unset"    # 内容区域按新屏重新计算并打日志
            self._pending_geom = None
            print(f"[view] 覆盖层已重建 {mw}x{mh} @({mx},{my})", flush=True)
            return True
        except Exception as e:
            print(f"[view] 覆盖层重建失败（稍后重试）: {e}", flush=True)
            return False

    @property
    def depth_tex(self):
        return self.post.depth

    # main 调用集的兼容占位（GL 专属诊断接口；D3D 路径不提供逐帧读回）
    def gpu_color_mean(self):
        return None

    def gpu_bind_color(self, *a, **kw):
        return None

    def gpu_unlock(self, *a, **kw):
        return None

    def _reassert_overlay_style(self):
        # LAYERED 是采集架构的硬需求（见 __init__ 注释：非分层会被 DWM 当
        # 真遮挡者，下方桌面不合成 → 采集黑屏）。每次显隐后必须重申。
        if _DCOMP:
            # DComp 版：非分层 + 无重定向位图（DWM 仍合成，语义见 __init__）
            _set_exstyle(self.hwnd, WS_EX_NOREDIRECTIONBITMAP |
                         WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
                         WS_EX_APPWINDOW | WS_EX_LAYERED)
            return
        # 【颜色键】见 __init__：alpha=255 + LWA_COLORKEY → 视觉全不透明，
        # 但分层窗口仍必须由 DWM 合成（无法走 MPO 独立翻转）→ 无撕裂。
        # logs/alpha255.on：A/B 用（回到纯 255 不透明平面，验证是否复现撕裂）。
        if _SYNC1:
            _user32.SetLayeredWindowAttributes(
                ctypes.c_void_p(self.hwnd), 0, 255, 0x2)       # 不透明
            return
        if _COLORKEY_ON:
            _user32.SetLayeredWindowAttributes(
                ctypes.c_void_p(self.hwnd), _COLORKEY, 0, 0x1)  # 颜色键
            return
        _user32.SetLayeredWindowAttributes(
            ctypes.c_void_p(self.hwnd), _COLORKEY, 0, 0x1)     # LWA_COLORKEY

    def _hide_from_taskbar(self):
        _set_exstyle(self.hwnd, WS_EX_TOOLWINDOW, WS_EX_APPWINDOW)

    def set_wda(self, on=True):
        ok = _user32.SetWindowDisplayAffinity(
            ctypes.c_void_p(self.hwnd),
            WDA_EXCLUDEFROMCAPTURE if on else 0)
        return bool(ok)

    def keep_topmost(self):
        _user32.SetWindowPos(ctypes.c_void_p(self.hwnd), ctypes.c_void_p(-1),
                             0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # NOSIZE|NOMOVE|NOACTIVATE

    def position_on_monitor(self, monitor_index=0):
        if self._mon is not None:          # 同屏：严格铺满采集屏
            mx, my, mw, mh = self._mon
            _user32.SetWindowPos(ctypes.c_void_p(self.hwnd), None,
                                 mx, my, mw, mh, 0)
            return
        sw, sh = _screen_size()
        ww, wh = self._client_w, self._client_h
        _user32.SetWindowPos(ctypes.c_void_p(self.hwnd), None,
                             (sw - ww) // 2, (sh - wh) // 2, ww, wh, 0)

    def toggle_fullscreen(self):
        if self._mon is not None:
            mx, my, mw, mh = self._mon
            _user32.SetWindowPos(ctypes.c_void_p(self.hwnd),
                                 ctypes.c_void_p(-1), mx, my, mw, mh,
                                 0x0040)   # SWP_SHOWWINDOW
            return
        # 窗口化查看器：最大化到主屏
        sw, sh = _screen_size()
        _user32.SetWindowPos(ctypes.c_void_p(self.hwnd), ctypes.c_void_p(-1),
                             0, 0, sw, sh, 0x0040)   # SWP_SHOWWINDOW

    # ---- 深度管线（main 调用集） ----------------------------------------
    def gpu_depth_init(self, res):
        self._res = int(res)

    def gpu_preprocess(self, src_tex=None):
        """色源 → prep 纹理。src_tex=None → CPU 上传路径（2D 回退）。"""
        if src_tex is not None:
            self.ren.begin_frame()
            src_tex.transition(dr.ST_PIXEL_SRV)
            self.prep.prep.state = dr.ST_COMMON
            self.prep.prep.transition(dr.ST_RENDER_TARGET)
            self.ren.draw_pass(self.prep._pso, [src_tex],
                               [float(self._res), float(self.prep.flat_w),
                                0.0, 0.0],
                               self.prep.prep, self.prep.prep_w,
                               self.prep.prep_h)
            self.prep.prep.state = dr.ST_COMMON
            self.ren.submit()
        else:
            self.prep.run()

    def postprocess_gpu(self, raw, lo, hi, alpha, aa, grow=0, slope=0.0,
                        parallax=0.0, iters=6, close_px=0, cut=False):
        if isinstance(raw, np.ndarray):
            self.post.upload_raw(raw)
        self.post.spatial_chain(lo, hi, aa_texels=aa, grow=grow,
                                slope_target=slope, slope_m=parallax,
                                iters=iters, temporal="ema", alpha=alpha,
                                cut=cut)
        self._frame_has_depth = True

    def read_depth_median(self):
        """收敛锚（延迟一帧语义由调用方保证；GPU 判据在 V2 接入）。"""
        from depth_norm import near_peak_median
        arr = self.ren.readback(self.post.depth, np.float32)
        return near_peak_median(arr)

    # ---- 参数 -----------------------------------------------------------
    def apply_live_params(self, params):
        for k, v in params.items():
            if k == "Parallax (px)":
                self.parallax_px = float(v)
            elif k in ("Convergence Bias", "Convergence"):
                self.convergence = float(v)
            elif k == "VSync":
                self.vsync = bool(v)
            elif k == "Interleave Phase":
                self.interleave_phase = 1.0 if v else 0.0
            elif k == "Show FPS":
                self.set_show_fps(v)
        self.disp.set_params(convergence=self.convergence,
                             parallax_px=self.parallax_px,
                             phase=self.interleave_phase)

    def set_show_fps(self, on):
        """运行时开关 FPS HUD（GL apply_live_params 同语义）。

        之前 D3D 这里是 `pass`：HUD 在构造时按启动值创建，面板开关保存后
        只改了 yaml，运行中的画面不变 —— 表现成"开关和显示反着来"。
        """
        self.show_fps = bool(on)
        if self.show_fps and self._hud_ov is None:
            self._hud_ov = dv.OverlayDraw(self.ren, self.disp)
            self._hud_arr = None
            self._hud_cache = {}

    def set_edge_snap_gain(self, gain):
        self.disp.set_params(edge_snap_gain=float(gain))

    def set_content_rect(self, l=None, t=None, r=None, b=None, feather=0.012):
        """设置"内容区域"（屏幕像素，**所在显示器坐标**）。

        只对这块区域做视差，其余区域两眼一致（桌面/任务栏保持清晰）——
        GL viewer.set_content_rect 同语义（flipped_uv.y = 1 - y/H）。
        None/无效 → 整屏都做（content=(0,0,0,0) 关闭遮罩）。
        """
        if l is None or r is None or t is None or b is None or \
                r <= l or b <= t:
            self.disp.set_params(content=(0.0, 0.0, 0.0, 0.0),
                                 content_feather=0.0)
            self._log_content(None)
            return
        mx, my, mw, mh = self._mon if self._mon is not None else \
            (0, 0) + _screen_size()
        # 屏幕像素 → 内容 UV。注意：D3D 的 i.uv 是 **top-down**（hlsl_display
        # 里 flipped_uv=uv 就是这个原因），所以 y 不做 GL 那套 1-y/H 翻转，
        # 否则 3D 区域在纵向上下镜像（实测 150..850 → 230..930）。
        x0 = max(0.0, min(1.0, (l - mx) / max(1.0, float(mw))))
        x1 = max(0.0, min(1.0, (r - mx) / max(1.0, float(mw))))
        y0 = max(0.0, min(1.0, (t - my) / max(1.0, float(mh))))
        y1 = max(0.0, min(1.0, (b - my) / max(1.0, float(mh))))
        if x1 <= x0 or y1 <= y0:
            self.disp.set_params(content=(0.0, 0.0, 0.0, 0.0),
                                 content_feather=0.0)
            self._log_content(None)
            return
        # 覆盖 ≥99% 屏（最大化/全屏窗口）→ 关遮罩：否则边缘 feather 会在
        # 屏幕最外圈压掉一条视差，且整屏都是内容时遮罩没有意义
        if (x1 - x0) >= 0.99 and (y1 - y0) >= 0.99:
            self.disp.set_params(content=(0.0, 0.0, 0.0, 0.0),
                                 content_feather=0.0)
            self._log_content(None)
            return
        self.disp.set_params(content=(x0, y0, x1, y1),
                             content_feather=float(feather))
        self._log_content((l, t, r, b))

    def _log_content(self, rect):
        """内容区域变更日志（拖窗口时应看到逐次跟随）。"""
        key = None if rect is None else tuple(int(v) for v in rect)
        if key == getattr(self, "_content_key", "unset"):
            return
        self._content_key = key
        if rect is None:
            print("[content] 3D 区域 = 整屏（无前台窗口约束）", flush=True)
        else:
            print(f"[content] 3D 区域 = ({rect[0]},{rect[1]})-"
                  f"({rect[2]},{rect[3]})", flush=True)

    # ---- 呈现 -----------------------------------------------------------
    def _draw_overlay(self):
        """帧内 overlay 总入口：HUD 在下、光标在上（GL 同序）。"""
        self._draw_hud()
        self._draw_cursor()

    def render(self, src_w=None, src_h=None, depth_tex=None):
        """录一帧（显示 pass + overlay/光标），不 present。

        depth_tex：流水线模式下传入与颜色同帧的那张显示深度（post 的
        disp_depth 乒乓）；None = 当前链输出（串行模式）。
        """
        self._render_seq += 1

        def _overlay():
            self._draw_hud()
            self._draw_cursor()

        self.disp.draw(self._mode, self._color_tex,
                       self.post.depth if depth_tex is None else depth_tex,
                       src_w=src_w or self._texture_size[0],
                       src_h=src_h or self._texture_size[1],
                       overlay=_overlay)

    def present(self, nowait=False):
        # D2S_SYNC1=1：不透明平面（alpha=255）时必须让翻转对齐 VBlank，
        # 否则 MPO 独立翻转会直扫 flip 缓冲 → 运动横缝。代价是 Present 等
        # 一次合成；nowait 时若没槽就跳过（不阻塞）。
        _sync = 1 if (_SYNC1 or self.vsync) else 0
        return self.disp.present(_sync, nowait=nowait)

    def attach_color(self, tex, w=None, h=None):
        """绑定本帧色源（DDA 桥环纹理或 2D 上传纹理）。"""
        self._color_tex = tex
        if w and h:
            self._texture_size = (w, h)

    def prepare_cursor_sprite(self, force_arrow=False):
        """建系统光标句柄缓存表（viewer 同款；必须在光标被隐藏【之前】调）。

        force_arrow：全屏输出时系统光标已被替换成空白，GetCursorInfo 的
        形状不可靠——固定用箭头方案位图（GL 路径同语义）。"""
        sp = self._cursor_sprite
        if sp is None:
            return
        try:
            sp._handle_map = sp._build_handle_map()
            if len(sp._handle_map) < 8:
                try:
                    import cursor_hide as _chr
                    _chr.restore()
                except Exception:
                    pass
                sp._handle_map = sp._build_handle_map()
                print(f"[Main] cursor handle map 重建: {len(sp._handle_map)} 个",
                      flush=True)
            if force_arrow:
                sp._force_arrow = True
                print("[Main] 光标固定箭头模式（系统光标已隐藏）", flush=True)
            print(f"[Main] cursor handle map: {len(sp._handle_map)} 个句柄已缓存",
                  flush=True)
        except Exception as _e:
            print(f"[Main] cursor handle map 建立失败: {_e}", flush=True)

    def _sprite_scaled(self, rgba, sx, sy):
        """按内容缩放比预缩光标位图（cv2 INTER_AREA，精确盒式平均）。

        直接线性采样缩小会糊/锯齿（GL 版同样结论：mipmap 也只是近似）——
        位图在 CPU 侧精确缩小，GPU 侧 1:1 绘制。缓存最近一份。"""
        key = (id(rgba), round(sx, 3), round(sy, 3))
        hit = self._curs_cache.get(key)
        if hit is not None:
            return hit
        arr = rgba
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            try:
                import cv2 as _cv
                h, w = rgba.shape[:2]
                arr = _cv.resize(
                    rgba, (max(1, int(round(w * sx))),
                           max(1, int(round(h * sy)))),
                    interpolation=_cv.INTER_AREA)
            except Exception:
                arr = rgba
        out = (arr, arr.shape[1], arr.shape[0])
        self._curs_cache.clear()
        self._curs_cache[key] = out
        return out

    def _hud_rgba(self):
        """FPS HUD 位图（GL viewer._generate_overlay_image 同款参数：
        字号 = 18 × min(win_w/1080, 2) × 0.8，padding=0.4×字号，
        绿色 "FPS: x.x"，透明底）。0.5s 节流且文本不变不重画。"""
        import time as _t
        now = _t.perf_counter()
        txt = f"FPS: {float(getattr(self, 'actual_fps', 0.0)):.1f}"
        if (self._hud_arr is not None and now - self._hud_t < 0.5):
            return self._hud_arr
        self._hud_t = now
        if txt == self._hud_txt and self._hud_arr is not None:
            return self._hud_arr
        self._hud_txt = txt
        try:
            from PIL import Image, ImageDraw, ImageFont
            from platform_probe import get_font_type
            size = int(18 * min(self.disp.w / 1080.0, 2.0) * 0.8)
            font = ImageFont.truetype(get_font_type(), size)
            pad = max(5, int(size * 0.4))
            bbox = font.getbbox(txt)
            w = bbox[2] - bbox[0] + pad * 2
            h = bbox[3] - bbox[1] + pad * 2
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((pad - bbox[0], pad - bbox[1]), txt, font=font,
                      fill=(0, 255, 0, 255))
            self._hud_arr = np.array(img, dtype=np.uint8)
            self._hud_pad = pad
        except Exception:
            self._hud_arr = None
        return self._hud_arr

    def _hud_scaled(self, arr, sx, sy):
        key = (id(arr), round(sx, 3), round(sy, 3))
        hit = self._hud_cache.get(key)
        if hit is not None:
            return hit
        out_arr = arr
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            try:
                import cv2 as _cv
                h, w = arr.shape[:2]
                out_arr = _cv.resize(arr, (max(1, int(round(w * sx))),
                                           max(1, int(round(h * sy)))),
                                     interpolation=_cv.INTER_AREA)
            except Exception:
                out_arr = arr
        out = (out_arr, out_arr.shape[1], out_arr.shape[0])
        self._hud_cache.clear()
        self._hud_cache[key] = out
        return out

    def _draw_hud(self):
        """FPS HUD：每眼底左（GL 同位置/同压缩），独立 OverlayDraw 纹理。"""
        if not self.show_fps or self._hud_ov is None:
            return
        arr = self._hud_rgba()
        if arr is None:
            return
        src_w, src_h = self._texture_size
        if self._mode in ("half_sbs", "full_sbs", "half_tab", "full_tab"):
            vps = list(self.disp.sbs_viewports(self._mode, src_w, src_h))
        else:
            vps = [self.disp.single_viewport(src_w, src_h)]
        win_w = max(1.0, float(self.disp.w))
        win_h = max(1.0, float(self.disp.h))
        for vp in vps:
            sx = vp[2] / win_w
            sy = vp[3] / win_h
            a2, gw, gh = self._hud_scaled(arr, sx, sy)
            self._hud_ov.record(a2)
            x = vp[0] + self._hud_pad
            y = vp[1] + self._hud_pad        # 左上（GL 的 position.y 距顶部）
            self._hud_ov.draw((x, y, gw, gh), scissor=vp)

    def _cursor_system_visible(self):
        """系统此刻是否"想显示"光标（viewer._render_cursor_overlay 同判据）。

        全屏播放器自动隐藏光标的三种做法都要跟随：
          ① SetCursor(NULL)     → GetCursorInfo 的 hCursor 为空
          ② ShowCursor(FALSE)   → CURSOR_SHOWING 位为 0
          ③ 换成全透明位图       → 句柄有效但位图 alpha≈0（0.25s 采样一次）
        读不到状态按"不显示"处理（fail-closed；以前异常后继续画，等于永远
        不知道系统已经藏了光标）。
        """
        try:
            import ctypes as _ct
            ci = _CursorInfoT()
            ci.cbSize = _ct.sizeof(_CursorInfoT)
            if not _user32.GetCursorInfo(_ct.byref(ci)):
                return False
            try:
                if os.environ.get("D2S_FAKE_CURSOR_NULL") == "1":
                    return False
            except Exception:
                pass
            if not ci.hCursor or not (ci.flags & 0x1):
                return False
            # ③ 应用自带光标且位图全透明 → 应用在隐藏，我们也不画
            try:
                import cursor_hide as _ch
                if _ch.identify(ci.hCursor) == -1 and self._cursor_sprite:
                    _now = time.perf_counter()
                    if _now - self._curs_alpha_t > 0.25:
                        self._curs_alpha_t = _now
                        _bm = self._cursor_sprite._extract(ci.hCursor)
                        if _bm is not None:
                            self._curs_alpha = \
                                float(_bm[0][..., 3].mean()) / 255.0
                    if self._curs_alpha < 0.02:
                        return False
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _draw_cursor(self):
        if self._cursor_ov is None or self._cursor_sprite is None:
            return
        # 跟随系统：全屏播放器把系统光标藏了（NULL / SHOWING=0 / 全透明
        # 位图），我们自绘的光标也必须消失——这是"全屏播放光标该自动隐藏"
        # 的判据（GL 管线同款）。
        if not self._cursor_system_visible():
            if os.environ.get("D2S_CURSOR_DBG") and self._render_seq % 30 == 0:
                print(f"[cursor-dbg] system-hidden seq={self._render_seq}",
                      flush=True)
            return
        # 探测（WM_SETCURSOR 让应用 SetCursor）只在鼠标移动后 0.3s 内做：
        # 每帧探测会把播放器的自动隐藏"打醒"（实测 GL 那边的老坑）。
        try:
            _pt = _wt.POINT()
            if _user32.GetCursorPos(ctypes.byref(_pt)):
                _pos = (_pt.x, _pt.y)
                if self._last_pos is None:
                    self._last_pos = _pos
                    self._cur_move_t = time.perf_counter()
                elif self._last_pos != _pos:
                    self._last_pos = _pos
                    self._cur_move_t = time.perf_counter()
        except Exception:
            pass
        _moved = (time.perf_counter() - self._cur_move_t) < 0.3
        got = self._cursor_sprite.get(allow_probe=_moved)
        if os.environ.get("D2S_CURSOR_DBG") and self._render_seq % 30 == 0:
            print(f"[cursor-dbg] got={'Y' if got else 'N'} seq={self._render_seq}",
                  flush=True)
        if got is None:
            return
        rgba, hx, hy, (cx, cy) = got
        if self._mon is not None:
            # 同屏：按采集屏原点归一（多屏负坐标/副屏在右侧；GL capture_origin 同款）
            mx, my, mw, mh = self._mon
            px = (cx - mx) / max(1.0, float(mw))
            py = (cy - my) / max(1.0, float(mh))
        else:
            sw, sh = _screen_size()
            px = cx / max(1.0, sw)
            py = cy / max(1.0, sh)
        src_w, src_h = self._texture_size
        # 每眼视口与内容缩放（与显示 pass 同公式）：Half-SBS 每眼 960×1080
        # → 横向压 0.5、纵向 1.0；Half-TAB 是纵向压 0.5；Fill16:9 双轴缩放。
        if self._mode in ("half_sbs", "full_sbs", "half_tab", "full_tab"):
            vps = list(self.disp.sbs_viewports(self._mode, src_w, src_h))
        else:
            vps = [self.disp.single_viewport(src_w, src_h)]
        win_w = max(1.0, float(self.disp.w))
        win_h = max(1.0, float(self.disp.h))
        for vp in vps:
            sx = vp[2] / win_w
            sy = vp[3] / win_h
            arr, gw, gh = self._sprite_scaled(rgba, sx, sy)
            self._cursor_ov.record(arr)
            x = vp[0] + int(round(px * vp[2] - hx * sx))
            y = vp[1] + int(round(py * vp[3] - hy * sy))
            # scissor 裁在本眼视口：光标越过左右半屏分割线被裁掉，
            # 不会糊到另一只眼里（GL 版同语义）
            self._cursor_ov.draw((x, y, gw, gh), scissor=vp)

    def frame_dump(self):
        """present 前抓帧（诊断）。返回 BGRA (h,w,4)。"""
        return self.ren.readback(self.ren.backbuffer, np.uint8)

    def close(self):
        try:
            self.win.close()
        except Exception:
            pass
        try:
            self.ren.destroy()
        except Exception:
            pass
