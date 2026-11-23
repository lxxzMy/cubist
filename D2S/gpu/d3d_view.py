# -*- coding: utf-8 -*-
"""D3D12 显示链（R4a→1.1-d3d V1）：swapchain + 全屏显示 pass + overlay + Present。

呈现语义（C 侧一条龙）：draw_* 后【不得】手动 submit——
D3DDisplay.present() 内部做 backbuffer 屏障(RT→COMMON) + 命令列表提交 +
Present。1.1-d3d 起拆成 draw（录制）→ [frame_dump 读回] → present 两步。

视口数学逐式移植 viewer._render_sbs_tab / _render_singlepass /
_render_passthrough / _render_depth_map（Fill16:9 与非 Fill 两分支不并）。
"""
import ctypes
import os

import numpy as np

import dibr

from . import d3drender as dr
from .d3drender import FMT_BGRA8, FMT_R32F, ST_RENDER_TARGET
from .hlsl import compile_shader, FULLSCREEN_VS
from .hlsl_display import DEPTH_MAP_FS, DIBR_FS, INTERLEAVED_FS, \
    OVERLAY_FS, PASSTHROUGH_FS, pack_dibr_consts

_user32 = ctypes.WinDLL("user32")
_k32 = ctypes.WinDLL("kernel32")
_user32.DefWindowProcW.restype = ctypes.c_ssize_t
_user32.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.c_size_t, ctypes.c_ssize_t]
_user32.DestroyWindow.argtypes = [ctypes.c_void_p]


# 消息泵结构体必须是模块级：在 pump() 里 `class POINT(ctypes.Structure)`
# 每次调用都会生成一个**新类型**，类型 ↔ StgDict ↔ 字段描述符成环 →
# 每拍约 40 个循环垃圾。pump 每帧跑 5~8 次 → 每秒上万，gen2 每十几秒
# 要处理 2000+ 对象、停顿 20~40ms（实测 22.9/29.0/30.3/38.4ms）。
class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MSG(ctypes.Structure):
    _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                ("wParam", ctypes.c_size_t), ("lParam", ctypes.c_ssize_t),
                ("time", ctypes.c_uint), ("pt", _POINT),
                ("lPrivate", ctypes.c_uint)]


class Win32Window:
    """最小 Win32 窗口（swapchain 宿主）。show=False 建隐藏窗口（测试）。"""

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, ctypes.c_void_p,
                                 ctypes.c_uint, ctypes.c_size_t,
                                 ctypes.c_ssize_t)

    def __init__(self, w, h, title=b"D2S", show=True, style=None,
                 pos=None):
        self._proc = self.WNDPROC(self._on_msg)   # 防 GC
        self.closed = False
        self.on_resize = None        # (w, h) 客户区回调（WM_SIZE）
        self.on_display_change = None   # 显示模式/布局变化（WM_DISPLAYCHANGE）
        self._pending_size = None
        self._display_changed = False
        _style = 0x00CF0000 if style is None else int(style)   # WS_OVERLAPPEDWINDOW
        # WNDCLASSW 手工布局（x64：LPVOID 对齐 8）
        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", ctypes.c_void_p),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                        ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
                        ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                        ("lpszMenuName", ctypes.c_wchar_p),
                        ("lpszClassName", ctypes.c_wchar_p)]
        self.hinst = _k32.GetModuleHandleW(None)
        wcx = WNDCLASSW(0, ctypes.cast(self._proc, ctypes.c_void_p).value,
                        0, 0, self.hinst, None, None, None, None, "D2S_view")
        _user32.RegisterClassW(ctypes.byref(wcx))
        # 客户区尺寸 → 窗口尺寸：OVERLAPPEDWINDOW 需算边框；POPUP/无边框
        # 客户区==窗口区，且允许外部指定落点（全屏覆盖层用显示器原点）
        rect = (ctypes.c_int * 4)(0, 0, w, h)
        if _style & 0x00CF0000:
            _user32.AdjustWindowRect(ctypes.byref(rect), _style, 0)
        ww = rect[2] - rect[0]
        wh = rect[3] - rect[1]
        px, py = (0, 0) if pos is None else (int(pos[0]), int(pos[1]))
        # WS_EX_NOREDIRECTIONBITMAP 只能在创窗时给（事后 SetWindowLong 无效）：
        # DirectComposition 覆盖层靠它省掉重定向位图。D2S_DCOMP=1 才加。
        _ex = 0x00200000 if os.environ.get("D2S_DCOMP") == "1" else 0
        self.hwnd = _user32.CreateWindowExW(
            _ex, "D2S_view", title, _style, px, py, ww, wh, None, None,
            self.hinst, None)
        if not self.hwnd:
            raise RuntimeError("[view] CreateWindowExW 失败")
        if show:
            _user32.ShowWindow(self.hwnd, 5)   # SW_SHOW

    def _on_msg(self, hwnd, msg, wp, lp):
        if msg == 0x0084:            # WM_NCHITTEST
            # DComp 覆盖层没有 LAYERED 的"点穿"语义 → 显式 HTTRANSPARENT，
            # 否则全屏覆盖层会把鼠标事件全吃掉。
            if os.environ.get("D2S_DCOMP") == "1":
                return -1
            return 1                 # HTCLIENT（layered 版行为）
        if msg == 0x0010:            # WM_CLOSE：不销毁（产品语义=退出信号）
            self.closed = True
            _user32.ShowWindow(hwnd, 0)
            return 0
        if msg == 0x0002:            # WM_DESTROY
            self.closed = True
            return 0
        if msg == 0x0005:            # WM_SIZE：记录客户区尺寸（消息泵后应用）
            self._pending_size = (lp & 0xFFFF, (lp >> 16) & 0xFFFF)
            return 0
        if msg == 0x007E:            # WM_DISPLAYCHANGE：分辨率/显示器布局变了
            self._display_changed = True
            return 0
        return _user32.DefWindowProcW(hwnd, msg, wp, lp)

    def pump(self):
        """消息泵一拍（非阻塞）。"""
        msg = _MSG()
        while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))
        if self._pending_size and self.on_resize:
            w, h = self._pending_size
            self._pending_size = None
            if w > 0 and h > 0:
                self.on_resize(w, h)
        if self._display_changed:
            self._display_changed = False
            if self.on_display_change:
                self.on_display_change()

    def close(self):
        if self.hwnd:
            _user32.DestroyWindow(self.hwnd)
            self.hwnd = None


def _compute_render_size(max_w, max_h, src_w, src_h):
    """viewer._compute_render_size 逐式（等比 fit）。"""
    if src_w == 0 or src_h == 0:
        return 0, 0
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _fit_viewport(win_w, win_h, src_w, src_h):
    """居中 aspect-fit 视口（viewer 非 Fill / 单 pass 分支公式）。"""
    target_aspect = src_h / src_w
    try:
        wa = win_h / win_w
    except ZeroDivisionError:
        wa = 9.0 / 16.0
    if wa <= target_aspect:
        vh = win_h
        vw = int(vh / target_aspect)
    else:
        vw = win_w
        vh = int(vw * target_aspect)
    return ((win_w - vw) // 2, (win_h - vh) // 2, vw, vh)


def _default_params(win_w, win_h, res=224):
    return dict(
        ipd_uv=0.064, parallax_px=12.0, convergence=0.5,
        parallax_curve=0.0, roll=0.0,
        fill=1.0, ic_search=1.0, depth_strength=1.0,
        phase=0.0,
        # 三项轮廓后处理已弃用（与 GL 生产同步：utils.DIBR_EDGE_SNAP=False/
        # DEPTH_RELIEF=0.0/DIBR_WARP_ITERATIONS=False）。edge_snap 把深度
        # 换成 ±2 texel 邻居=硬台阶锯齿；relief 把颜色拉普拉斯灌进深度；
        # warp 二次重采样出双边缘。三者在产品里一律关。
        edge_snap=0.0, edge_snap_gain=1.0,
        relief=0.0, warp=0.0,
        feather_enabled=0, feather_width=0.05,
        content=(0.0, 0.0, 0.0, 0.0), content_feather=0.0,
        edge_falloff=0.0, corner_radius=0.0, full_alpha=1.0,
        res=res,
    )


class D3DDisplay:
    """swapchain 显示编排（宿主 = D3DRenderer）。

    draw_*(mode)：录制到 backbuffer（不 present）；present()：屏障+提交+
    Present。旧 render_*（=draw+present）保留供工具兼容。
    """

    def __init__(self, ren, hwnd, w, h, use_flip=True):
        self.ren = ren
        if hwnd is not None:   # hwnd=None → 离屏模式（parity/测试，无 present）
            ren.create_swapchain(hwnd, w, h, use_flip)
        self.w, self.h = w, h
        self._pso = {}
        self._vs = compile_shader("fs_vs", FULLSCREEN_VS, "vs")
        self.p = _default_params(w, h)
        self.fill_16_9 = True   # 产品默认；非 Fill 由 set_params 覆盖

    # ---- 旧方法名兼容 ---------------------------------------------------
    def _pso_dibr(self):
        return self._pso_for("dibr", DIBR_FS)

    def _pso_interleaved(self):
        return self._pso_for("interleaved", INTERLEAVED_FS)

    # ---- PSO ------------------------------------------------------------
    def _pso_for(self, name, ps_src, blend=False):
        key = f"{name}{'_blend' if blend else ''}"
        if key in self._pso:
            return self._pso[key]
        cso = compile_shader(f"view_{name}", ps_src)
        if blend:
            pso = self.ren.create_pso_blend(f"view_{key}", self._vs, cso,
                                            FMT_BGRA8)
        else:
            pso = self.ren.create_pso(f"view_{key}", self._vs, cso, FMT_BGRA8)
        self._pso[key] = pso
        return pso

    def set_params(self, **kw):
        self.p.update({k: v for k, v in kw.items() if k in self.p})
        if "fill_16_9" in kw:
            self.fill_16_9 = bool(kw["fill_16_9"])

    # ---- 通用 consts ----------------------------------------------------
    def _consts(self, vp, eye_offset, par_u, res, row_split=0.0,
                row_parity=0.0, alpha_override=None):
        p = self.p
        full_alpha = p.get("full_alpha", 1.0)
        return pack_dibr_consts(
            edge_falloff=p.get("edge_falloff", 0.0),
            eye_offset=eye_offset,
            depth_strength=p.get("depth_strength", 1.0),
            convergence=p.get("convergence", 0.5),
            parallax=par_u,
            parallax_curve=p.get("parallax_curve", 0.0),
            roll=p.get("roll", 0.0),
            fill=p.get("fill", 0.0) if alpha_override is None else alpha_override,
            ic_search=p.get("ic_search", 1.0),
            depth_step_uv=1.0 / max(1.0, float(res)),
            ic_debug=0.0,
            row_split=row_split,
            row_parity=row_parity,
            content_feather=p.get("content_feather", 0.0),
            full_alpha=full_alpha,
            corner_radius=p.get("corner_radius", 0.0),
            search_radius=12.0,
            depth_tolerance=0.012,
            blur_radius=2.5,
            feather_enabled=p.get("feather_enabled", 0),
            feather_width=p.get("feather_width", 0.05),
            win_w=self.w, win_h=self.h, res=res,
            content=p.get("content", (0.0, 0.0, 0.0, 0.0)),
            viewport=vp,
            interleave_phase=p.get("phase", 0.0),
            edge_snap=p.get("edge_snap", 0.0),
            edge_snap_gain=p.get("edge_snap_gain", 1.0),
            relief_strength=p.get("relief", 0.0),
            warp_iterations=p.get("warp", 0.0),
        )

    # ---- 视口（主力档） --------------------------------------------------
    def sbs_viewports(self, mode, src_w=None, src_h=None):
        """Half/Full-SBS（左右）、Half/Full-TAB（上下）两眼视口。

        Fill16:9 与非 Fill 两分支，公式与 viewer._render_sbs_tab 逐式一致
        （Fill = 每眼半区等比 fit；非 Fill = 整幅 fit 到窗口再对半切）。
        src_w/src_h=None → 全窗（parity/工具语义）；TAB 返回 (上, 下)。
        """
        win_w, win_h = self.w, self.h
        _tab = mode in ("half_tab", "full_tab")
        if src_w is None:
            if _tab:
                hh = win_h // 2
                return (0, 0, win_w, hh), (0, hh, win_w, hh)
            hw = win_w // 2
            return (0, 0, hw, win_h), (hw, 0, hw, win_h)
        if self.fill_16_9:
            if _tab:
                # viewer fill 分支：Half-TAB 源高取半、Full-TAB 取全高，
                # 各自 fit 进半个窗口，居中于上/下四分之一处
                _sh = (src_h / 2.0) if mode == "half_tab" else src_h
                rw, rh = _compute_render_size(win_w, win_h / 2.0,
                                              src_w, _sh)
                cx = win_w / 2.0
                top = (int(cx - rw / 2), int(win_h / 4.0 - rh / 2), rw, rh)
                bot = (int(cx - rw / 2), int(3 * win_h / 4.0 - rh / 2),
                       rw, rh)
                return top, bot
            rw, rh = _compute_render_size(
                win_w / 2.0, win_h, src_w, src_h)
            cy = win_h / 2.0
            left = (int(win_w / 4.0 - rw / 2), int(cy - rh / 2), rw, rh)
            right = (int(3 * win_w / 4.0 - rw / 2), int(cy - rh / 2), rw, rh)
            return left, right
        if mode == "full_sbs":
            disp_w, disp_h = 2 * src_w, src_h
        elif mode == "full_tab":
            disp_w, disp_h = src_w, 2 * src_h
        else:                       # half_sbs / half_tab
            disp_w, disp_h = src_w, src_h
        ox, oy, vw, vh = _fit_viewport(win_w, win_h, disp_w, disp_h)
        if _tab:
            _vp2 = ((ox, oy + vh // 2, vw, vh // 2),
                    (ox, oy, vw, vh // 2))
        else:
            _vp2 = ((ox, oy, vw // 2, vh), (ox + vw // 2, oy, vw // 2, vh))
        # 每眼视口变化时打一行：AR 眼镜/3840×1080 输出时这里应显示
        # 每眼 1920×1080（swapchain = 显示器原生尺寸，full_sbs 不缩放）。
        _key = (mode, _vp2)
        if getattr(self, "_vp_log", None) != _key:
            self._vp_log = _key
            print(f"[disp] {mode} 输出 {win_w}x{win_h}，源 {src_w}x{src_h}"
                  f" → 每眼 {_vp2[0][2]}x{_vp2[0][3]}", flush=True)
        return _vp2

    def single_viewport(self, src_w=None, src_h=None):
        """Interleaved / Passthrough / DepthMap 单 pass 视口（None=全窗）。"""
        win_w, win_h = self.w, self.h
        if src_w is None:
            return (0, 0, win_w, win_h)
        if self.fill_16_9:
            vw, vh = _compute_render_size(win_w, win_h, src_w, src_h)
            return ((win_w - vw) // 2, (win_h - vh) // 2, vw, vh)
        return _fit_viewport(win_w, win_h, src_w, src_h)

    # ---- draw（录制，不 present） ---------------------------------------
    def draw(self, mode, color_tex=None, depth_tex=None, src_w=None,
             src_h=None, out=None, overlay=None, stats=None):
        """录一帧显示。mode: half_sbs|full_sbs|interleaved|passthrough|
        depth_map。out=None → backbuffer（随后 present）；out=Tex → 离屏
        （不 present，调用方 readback）。

        src_w/src_h = 采集源尺寸（视口宽高比语义）；tex 内容全用归一化 uv。
        overlay: 回调（无参），在显示 pass 之后、同帧内录制 overlay/cursor。
        stats: D3DStats——depth_map 模式用它把原始深度按本帧 P2/P98
        归一化（shader 读 t7）；其它模式忽略。
        """
        win_w, win_h = self.w, self.h
        res = depth_tex.w if depth_tex is not None else self.p.get("res", 224)
        if mode == "passthrough":
            pso = self._pso_for("passthrough", PASSTHROUGH_FS)
            srvs = [color_tex] if color_tex is not None else []
        elif mode == "depth_map":
            pso = self._pso_for("depth_map", DEPTH_MAP_FS)
            srvs = [depth_tex] if depth_tex is not None else []
        elif mode == "interleaved":
            pso = self._pso_for("interleaved", INTERLEAVED_FS)
            srvs = [color_tex] + ([depth_tex] if depth_tex is not None else [])
        elif mode in ("half_sbs", "full_sbs", "half_tab", "full_tab"):
            pso = self._pso_for("dibr", DIBR_FS)
            srvs = [color_tex] + ([depth_tex] if depth_tex is not None else [])
        else:
            raise ValueError(mode)

        for t in srvs:
            t.transition(dr.ST_PIXEL_SRV)

        target = out if out is not None else self.ren.backbuffer
        self._last_target = out          # overlay 回调取同一目标
        self.ren.begin_frame()
        target.state = dr.ST_COMMON
        target.transition(ST_RENDER_TARGET)
        # GL viewer 每帧 ctx.clear(0,0,0)：TAB/深度图这类带黑边的模式如果
        # 不清，视口外会残留轮换缓冲里的上上帧内容（3 缓冲尤其明显）。
        self.ren.clear_rtv(target)

        if mode in ("passthrough", "depth_map"):
            vp = self.single_viewport(src_w, src_h)
            self.ren.draw_pass(pso, srvs, self._consts(vp, 0.0, 0.0, res),
                               target, win_w, win_h, viewport=vp,
                               stats=(stats.stats if (mode == "depth_map"
                                                      and stats is not None)
                                      else None))
        elif mode == "interleaved":
            vp = self.single_viewport(src_w, src_h)
            ipd = self.p.get("ipd_uv", 0.064)
            par_u = dibr.parallax_uv(self.p.get("parallax_px", 12.0), ipd,
                                     win_w)
            self.ren.draw_pass(pso, srvs,
                               self._consts(vp, ipd / 2.0, par_u, res),
                               target, win_w, win_h, viewport=vp)
        else:
            left, right = self.sbs_viewports(mode, src_w, src_h)
            ipd = self.p.get("ipd_uv", 0.064)
            par_u = dibr.sbs_parallax_uv(self.p.get("parallax_px", 12.0), ipd,
                                         win_w)
            for vp, sign in ((left, -1.0), (right, 1.0)):
                self.ren.draw_pass(
                    pso, srvs, self._consts(vp, sign * ipd / 2.0, par_u, res),
                    target, win_w, win_h, viewport=vp, scissor=vp)
        if overlay is not None:
            overlay()   # 同帧 overlay/cursor（目标跟随 _last_target：屏幕或离屏）
        if out is None:
            return None    # present() 由调用方发起（draw→dump→present）
        self.ren.submit()
        self.ren.wait()
        return out

    def present(self, sync_interval=1, nowait=False):
        return self.ren.present(sync_interval, nowait=nowait)

    # ---- 旧接口兼容（draw+present 一体化） -------------------------------
    def render(self, mode, color_tex=None, depth_tex=None):
        m = {"passthrough": "passthrough", "depth_map": "depth_map"}.get(mode)
        if m is None:
            raise ValueError(mode)
        self.draw(m, color_tex=color_tex, depth_tex=depth_tex)
        self.present(0)

    def render_interleaved(self, color_tex, depth_tex, params=None):
        if params:
            self.set_params(**params)
        self.draw("interleaved", color_tex, depth_tex)
        self.present(0)

    def render_stereo(self, mode, color_tex, depth_tex, params=None):
        m = "full_sbs" if mode == "Full-SBS" else "half_sbs"
        if params:
            self.set_params(**params)
        self.draw(m, color_tex, depth_tex)
        self.present(0)

    def draw_interleaved_offscreen(self, out, color_tex, depth_tex, params=None):
        if params:
            self.set_params(**params)
        self.draw("interleaved", color_tex, depth_tex, out=out)

    def resize(self, w, h):
        self.ren.resize_swapchain(w, h)
        self.w, self.h = w, h

    def read_backbuffer(self):
        """present 前抓 backbuffer（诊断/验证用；present 后内容不可靠）。"""
        return self.ren.readback(self.ren.backbuffer, np.uint8)


class OverlayDraw:
    """overlay/光标绘制（RGBA 纹理 + alpha blend quad）。

    复用 viewer 的位图生成/位置数学（纯逻辑段），这里只做 D3D 呈现。
    【帧内语义】record() 只录进调用方的当前命令列表（绝不 begin_frame/
    submit——渲染帧内自起一帧会重置主列表）；draw() 随后绑 SRV 绘制。
    """

    def __init__(self, ren, disp):
        self.ren = ren
        self.disp = disp
        self.pso = disp._pso_for("overlay", OVERLAY_FS, blend=True)
        self._tex = None
        self._size = (0, 0)
        self._last_arr = None    # 同一位图对象不重复上传（SBS 两眼一次共享）

    def record(self, rgba):
        """rgba: (h,w,4) uint8（RGBA 序）→ CPU 写 + 帧内录制拷贝（不提交）。

        幂等：同一数组对象 + 尺寸不变时直接返回（每帧两只眼只付一次上传）。"""
        d = dr.dll()
        h, w = rgba.shape[:2]
        if (rgba is self._last_arr and self._tex is not None
                and self._size == (w, h)):
            # 仍需保证本帧纹理处于可采样态（上帧 ECL 后已 decay 回 COMMON，
            # 记录里是 PIXEL_SRV——直接按记录对齐即可，不再发拷贝）
            self._tex.state = dr.ST_COMMON
            self._tex.transition(dr.ST_PIXEL_SRV)
            return
        if self._tex is None or self._size != (w, h):
            if self._tex is not None:
                self._tex.release()
            self._tex = self.ren.upload_tex(w, h, _FMT_RGBA8,
                                            renderable=False)
            self._size = (w, h)
            self._tex.state = dr.ST_COMMON
        self.ren.upload(self._tex, np.ascontiguousarray(rgba))
        self._last_arr = rgba
        # 帧内：COMMON(上帧 decay)→COPY_DEST→copy→PIXEL_SRV
        self._tex.transition(dr.ST_COPY_DEST)
        d.d3dr_copy_buf_to_tex(self.ren.ctx, self._tex._upbuf,
                               self._tex.res, w, h, _FMT_RGBA8,
                               dr.row_pitch(w, 4))
        self._tex.transition(dr.ST_PIXEL_SRV)

    def draw(self, viewport, scissor=None):
        """录制到当前帧目标（display pass 的同一目标；缺省 backbuffer）。"""
        if self._tex is None:
            return
        vp = viewport
        tgt = getattr(self.disp, "_last_target", None) or self.ren.backbuffer
        self.ren.draw_pass(self.pso, [self._tex],
                           [float(self._size[0]), float(self._size[1]), 0, 0],
                           tgt, self.disp.w, self.disp.h,
                           viewport=vp, scissor=scissor or vp)


_FMT_RGBA8 = 28   # DXGI_FORMAT_R8G8B8A8_UNORM
