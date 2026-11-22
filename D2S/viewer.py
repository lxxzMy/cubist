# viewer.py
import ctypes, os, sys
import glfw
import moderngl
import numpy as np
import time
import cv2          # 自绘光标（_self_shape 系列）直接用裸 cv2；部分函数用 _cv2 别名
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
# Get OS name and settings
from utils import OS_NAME, crop_icon, get_font_type, SAME_SCREEN, EDGE_FALLOFF
# 3D monitor mode to hide viewer
if OS_NAME == "Windows":
    from utils import hide_window_from_capture, show_window_in_capture
elif OS_NAME == "Darwin":
    from utils import send_ctrl_cmd_f
from shaders import (VERTEX_SHADER, FRAGMENT_SHADER, DEPTH_FRAGMENT,
                      ANAGLYPH_FRAGMENT, INTERLEAVED_FRAGMENT,
                      VERTICAL_INTERLEAVED_FRAGMENT)


# ============================================================================
# MC-EMA（运动补偿时域平滑，产品化 2026-11-18）——着色器/泛洪放模块级，
# tools/mc_gl_test.py 离线对拍 import 同一份（同码同源，防两端漂移）。
# 离线证据链：tools/mc_seam_test.py / mc_object_test.py / mc_valve_test.py
# ============================================================================
# MC-EMA 时域块已解耦至 depth_mc.py（2026-11-20）：MC_* 着色器常量、
# mc_flood_grid、MC 类均在彼处（tools 离线对拍同源 import depth_mc）。
# ============================================================================
from depth_mc import MC
from depth_post import DepthPost
import dibr


def add_logo(window):
    """Optimized logo loading with lazy imports"""
    from PIL import Image
    # 产品图标（icon2.ico 是历史遗留的非产品图标，已删除）
    _icon = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "icon.ico")
    glfw_img = Image.open(_icon)  # Path to your icon file
    if OS_NAME != "Darwin":
        glfw_img = crop_icon(glfw_img)
        glfw.set_window_icon(window, 1, [glfw_img])


from cursor_sprite import (
    _CursorSprite, _cursor_base_size, _real_cursor_glyph_height,
    _CursorInfoT)

from overlay import OverlayTextureRenderer
class StereoWindow:
    """Optimized stereo viewer with performance improvements"""

    def __init__(self, capture_mode="Monitor", monitor_index=0,  depth_ratio=1.0, convergence=0.0, display_mode="Half-SBS", fill_16_9=True, show_fps=True, use_3d=False, fix_aspect=False, stream_mode=None, lossless_scaling=False, specify_display=False, stereo_display_index=0, feather_enabled=False, frame_size=(1280, 720), vsync=False, interleave_phase=0, parallax_px=12.0, parallax_curve="linear", dibr_edge_snap=True, depth_relief=0.22, warp_iterations=True, disocclusion_fill=False, cursor_overlay=False, fullscreen_output=False, **kwargs):
        try:
            print(f"[boot] StereoWindow.__init__ 进入 +"
                  f"{time.perf_counter() - _D2S_T0:.1f}s", flush=True)
        except Exception as _e:
            print(f"[boot] init戳失败: {_e}", flush=True)
        # 全屏输出模式：整屏都由我们绘制，必须强制不透明（见着色器 u_full_alpha）
        self._fullscreen_output = bool(fullscreen_output)
        # Initialize with default values
        self._has_real_frame = False
        self.use_3d = use_3d
        self.title = "Stereo Viewer"
        self.capture_mode = capture_mode
        self.input_monitor_index = monitor_index
        self.ipd_uv = 0.064   # 内部常量：位移公式中与 u_parallax 分母相消，不参与立体强度
        self.depth_strength = 0.1
        # 2D-mirror frames (depth=None) zero the parallax in render(); assume
        # 3D until a frame says otherwise.
        self._frame_has_depth = True
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio
        self._modes = ["Full-SBS", "Half-SBS", "Half-TAB", "Depth Map", "Full-TAB", "Anaglyph", "Interleaved", "Interleaved-V"]
        # Edge feathering toggle
        self.feather_enabled = feather_enabled
        self.feather_width = 0.02       # 2% of view width
        self.display_mode = display_mode
        # Which row parity the panel's polarizer film maps to the left eye.
        # 0 = even rows (from the top) are left, 1 = swapped. Set once from
        # settings.yaml ("Interleave Phase"), calibrated by eye on this panel.
        self.interleave_phase = 1.0 if interleave_phase else 0.0
        # Parallax budget in screen pixels per eye: the offset at the ends of
        # the depth range (when Convergence = 0.5). One number, linear in depth.
        self.parallax_px = float(parallax_px)
        # 1.0 = legacy non-linear depth->parallax curve, 0.0 = linear.
        self._parallax_curve = 1.0 if str(parallax_curve).strip().lower() == "legacy" else 0.0
        self._edge_falloff = EDGE_FALLOFF   # 边缘渐隐宽度（0=关），settings Edge Falloff
        self._disocclusion_fill = bool(disocclusion_fill)
        # Render-time cursor overlay: see _CursorSprite for why it is not baked
        # into the captured frame.
        self._cursor_overlay = None
        self._cursor_sprite = None
        self._capture_origin = (0, 0)
        self._texture_size = None
        self.fill_16_9 = fill_16_9
        self.frame_size = frame_size
        self.aspect = self.frame_size[0] / self.frame_size[1]
        self.fix_aspect = fix_aspect
        self.show_fps = show_fps
        self.vsync = vsync
        self.stream_mode = stream_mode
        self.window_size = self.frame_size
        self.convergence = convergence
        self.lossless_scaling = lossless_scaling

        # FPS and latency tracking variables, will be set externally
        self.actual_fps = 0.0
        self.total_latency = 0.0
        
        # Add PBO for streamer
        self._pbo_ids = None
        self._pbo_index = 0
        self._pbo_initialized = False
        
        # Add toggle state for depth map mode
        self.show_original_in_depth_mode = False
        
        # Cache for uniforms to avoid redundant updates
        self._last_eye_offset_set = 0.0
        self._last_depth_strength_set = 0.0
        
        # Depth ratio display variables
        self.last_depth_change_time = 0
        self.show_depth_ratio = False
        self.depth_display_duration = 2.0  # Show depth for 2 seconds after change
        
        # Font and text sizing
        self.font = None
        self.font_type = get_font_type()
        self.base_font_size = 18  # 1280 窗口下的基准字号（原先 60 太大）
        self.current_font_size = self.base_font_size
        self.text_padding = 10
        self.text_spacing = 5

        # Mouse Passthrough
        self.last_mouse_toggle_time = 0
        self.show_mouse_state = False
        self.mouse_display_duration = 2.0

        # Cache for viewport calculations
        self._viewport_cache = {
            'win_size': (0, 0),
            'tex_size': (0, 0),
            'display_mode': None,
            'fill_16_9': None,
            'viewport': None,
            'show_original': None
        }

        # Overlay cache & throttle
        self.overlay_update_interval = 0.5  # seconds, throttle overlay regeneration
        self._overlay_cache = {
            'image': None,         # numpy RGBA image
            'overlay_rgb_f': None, # cached float32 RGB of overlay (avoids per-frame convert)
            'alpha_f': None,       # cached float32 alpha (0..1) of overlay
            'fps_text': None,
            'latency_text': None,
            'depth_text': None,
            'mouse_text': None,
            'last_update': 0.0,
            'values_update': 0.0,  # last time the displayed FPS/latency numbers refreshed
            'disp_fps': None,      # throttled FPS value actually shown
            'disp_latency': None,  # throttled latency value actually shown
            'pos': (self.text_padding, self.text_padding)
        }
        
        # Stereo Display Settings
        self.specify_display = specify_display
        self.stereo_display_index = stereo_display_index

        # Fullscreen mode
        self._fullscreen = False

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        self.monitor_index = self.get_glfw_mon_index_mss(self.input_monitor_index)
        if self.specify_display:
            self.monitor_index = self.get_glfw_mon_index_mss(self.stereo_display_index)
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE) # window decoration
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # keep the stereo viewer on top

        if self.use_3d or (self.capture_mode == "Window" and self.specify_display and self.input_monitor_index == self.stereo_display_index):
            glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)  # clicks pass through
            glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)     # 显示永不抢焦点
            glfw.window_hint(glfw.DECORATED, glfw.FALSE) # remove window decoration
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)  # enable fullscreen
        # Create window
        if SAME_SCREEN:
            # 同屏自投射：创建即最终形态（显示器全尺寸 + 目标屏原点 + 无装饰），
            # 之后只 show/hide —— 绝不 toggle_fullscreen/重定位/改装饰。
            # 全屏切换的样式改写会反复重置分层属性与 GL 表面尺寸（实测
            # 时序竞争后窗口矩形 1920x1080 而 fb 卡在创建尺寸且全黑），
            # 唯一可靠的做法是从根上不给它 churn 的机会。
            import ctypes as _ct_for_hint
            _mons = glfw.get_monitors()
            _mi = min(self.monitor_index, len(_mons) - 1)
            _mx, _my = glfw.get_monitor_pos(_mons[_mi])
            _vm = glfw.get_video_mode(_mons[_mi])
            self.window_size = (_vm.size.width, _vm.size.height)
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        try:
            print(f"[boot] __init__ 前段完成，创建 GLFW 窗口 +"
                  f"{time.perf_counter() - _D2S_T0:.1f}s", flush=True)
        except Exception as _e:
            print(f"[boot] 前段戳失败: {_e}", flush=True)
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        if SAME_SCREEN:
            glfw.set_window_pos(self.window, _mx, _my)
        # Hide from taskbar and Alt+Tab from creation; re-asserted after the
        # window is shown (see _hide_from_taskbar).
        add_logo(self.window)
        self._reassert_overlay_style()
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        # Hide the OS cursor inside the viewer: with passthrough on, the
        # in-frame captured cursor is the correct pointer (it lives inside
        # the 3D-rendered desktop); the OS cursor on top would duplicate it.
        # 趁系统光标还正常时量一次"实际可见高度"（替换成空白后就量不到了）
        try:
            self._real_glyph_h = _real_cursor_glyph_height(_cursor_base_size())
            globals()["_REAL_GLYPH_H"] = self._real_glyph_h
        except Exception:
            self._real_glyph_h = 0
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        # 基准测试：强制窗口/framebuffer 尺寸。显示器比目标小时，"贴合显示器"
        # 逻辑会把窗口缩回去，测不出 4K 渲染成本。D2S_FORCE_WIN=3840x2160
        self._force_win = False
        try:
            import os as _osf
            _fw_env = _osf.environ.get("D2S_FORCE_WIN", "").strip().lower()
            if _fw_env:
                _fw, _fh = (int(v) for v in _fw_env.split("x"))
                if _fw > 0 and _fh > 0:
                    self._force_win = True
                    self.window_size = (_fw, _fh)
                    glfw.set_window_size(self.window, _fw, _fh)
                    glfw.poll_events()
                    _fb = glfw.get_framebuffer_size(self.window)
                    print(f"[viewer] D2S_FORCE_WIN → 窗口 {_fw}x{_fh}，"
                          f"framebuffer={_fb[0]}x{_fb[1]}（基准测试）", flush=True)
        except Exception as _fe:
            print(f"[viewer] D2S_FORCE_WIN 解析失败: {_fe}", flush=True)

        # Set up OpenGL context
        try:
            print(f"[boot] 窗口创建完成 +{time.perf_counter() - _D2S_T0:.1f}s",
                  flush=True)
        except Exception as _e:
            print(f"[boot] 窗口戳失败: {_e}", flush=True)
        glfw.make_context_current(self.window)
        try:
            print(f"[boot] GL 上下文就绪 +{time.perf_counter() - _D2S_T0:.1f}s",
                  flush=True)
        except Exception as _e:
            print(f"[boot] GL 戳失败: {_e}", flush=True)
        self.ctx = moderngl.create_context()
        try:
            print(f"[boot] moderngl ctx 就绪 +{time.perf_counter() - _D2S_T0:.1f}s",
                  flush=True)
        except Exception as _e:
            print(f"[boot] ctx 戳失败: {_e}", flush=True)
        # 记录 GL 落在哪块显卡上；D3D11 侧（采集/互操作）会按这块选同一 adapter。
        # 跨卡 = wglDXOpenDeviceNV 必失败 = 有 SBS 但视差为 0（退化成 2D）。
        try:
            from OpenGL.GL import glGetString as _gls, GL_RENDERER as _GLR
            from gpu import placement as _pl
            _pl.gl_device(_gls(_GLR).decode())
            if not globals().get("_ADAPTER_SUMMARY_LOGGED"):
                globals()["_ADAPTER_SUMMARY_LOGGED"] = True
                for _ln in _pl.report(detail=True):
                    print(_ln, flush=True)
        except Exception as _ae:
            print(f"[3D] adapter 自检失败: {_ae}", flush=True)
        try:
            print(f"[boot] adapter 摘要完成 +{time.perf_counter() - _D2S_T0:.1f}s",
                  flush=True)
        except Exception as _e:
            print(f"[boot] adapter 戳失败: {_e}", flush=True)
        except Exception as _ae:
            print(f"[3D] adapter 自检失败: {_ae}", flush=True)
        glfw.swap_interval(1 if self.vsync else 0)
        
        # Precompile shaders and create VAO
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.prog['u_convergence'].value = self.convergence  # e.g. self.convergence = 0.5
        vertices = np.array([
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1,  1, 0, 1,
            1,  1, 1, 1,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.depth_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=DEPTH_FRAGMENT
        )
        # Plain blit program for the 2D mirror mode: draws the RGB texture
        # fullscreen with no stereo warp at all.
        _plain_fs = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;
    uniform sampler2D tex_color;
    vec4 sample_color(vec2 uv){ return texture(tex_color, uv); }  // interop 视图通道本就正确

    void main() {
        frag_color = vec4(sample_color(uv).rgb, 1.0);
    }
"""
        self.plain_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=_plain_fs
        )
        self.plain_prog['tex_color'].value = 0
        self.plain_vao = self.ctx.vertex_array(
            self.plain_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.depth_vao = self.ctx.vertex_array(
            self.depth_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Anaglyph shader program
        self.anaglyph_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=ANAGLYPH_FRAGMENT
        )
        self.anaglyph_prog['u_convergence'].value = self.convergence
        self.anaglyph_vao = self.ctx.vertex_array(
            self.anaglyph_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Interleaved (row-interleaved) shader program
        self.interleaved_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=INTERLEAVED_FRAGMENT
        )
        self.interleaved_prog['u_convergence'].value = self.convergence
        self.interleaved_vao = self.ctx.vertex_array(
            self.interleaved_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Interleaved-V shader program: columns alternate per eye.
        self.vertical_interleaved_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=VERTICAL_INTERLEAVED_FRAGMENT
        )
        self.vertical_interleaved_prog['u_convergence'].value = self.convergence
        self.vertical_interleaved_vao = self.ctx.vertex_array(
            self.vertical_interleaved_prog, [(self.vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Sampler uniforms default to texture unit 0. Without these bindings
        # the three composite programs sample tex_depth from unit 0 as well,
        # i.e. they read the COLOUR texture as depth (depth := red channel) and
        # the depth map uploaded to unit 1 is never used.
        for _prog in (self.anaglyph_prog, self.interleaved_prog,
                      self.vertical_interleaved_prog):
            _prog['tex_color'].value = 0
            _prog['tex_depth'].value = 1
        try:
            print(f"[boot] shader 编译完成 +{time.perf_counter() - _D2S_T0:.1f}s",
                  flush=True)
        except Exception as _e:
            print(f"[boot] shader 戳失败: {_e}", flush=True)

        # ParallaxRT-style silhouette handling (interleaved shaders only for
        # now): edge snapping + RGB relief + one fixed-point warp refinement.
        self._edge_gain_progs = [self.interleaved_prog, self.vertical_interleaved_prog]
        for _prog in self._edge_gain_progs:
            _prog['u_edge_snap'].value = 1.0 if dibr_edge_snap else 0.0
            try:
                _prog['u_edge_snap_gain'].value = float(globals().get("EDGE_SNAP_GAIN", 1.0))
            except Exception:
                pass
            _prog['u_relief_strength'].value = float(depth_relief)
            _prog['u_warp_iterations'].value = 1.0 if warp_iterations else 0.0

        self.overlay_renderer = OverlayTextureRenderer(self.ctx)

        if cursor_overlay:
            try:
                self._cursor_sprite = _CursorSprite()
                # 必须在隐藏光标【之前】建立句柄缓存表（隐藏后 LoadCursor
                # 拿到的就是我们自己的空白光标，表就废了）
                try:
                    _sp = self._cursor_sprite
                    _sp._handle_map = _sp._build_handle_map()
                    if len(_sp._handle_map) < 8:
                        # 切 2D→3D / 重启时可能先 hide() 了：这时建表拿到的全
                        # 是空白形状。先 restore() 再建一次，别把空白缓存下来。
                        try:
                            import cursor_hide as _chr
                            _chr.restore()
                        except Exception:
                            pass
                        _sp._handle_map = _sp._build_handle_map()
                        print(f"[Main] cursor handle map 重建: "
                              f"{len(_sp._handle_map)} 个", flush=True)
                except Exception as _hm:
                    print(f"[Main] cursor handle map 建立失败: {_hm}", flush=True)
                try:
                    import utils as _u2
                    if getattr(_u2, "FULLSCREEN_OUTPUT", False):
                        self._cursor_sprite._force_arrow = True
                        print("[Main] 光标固定箭头模式（系统光标已隐藏）", flush=True)
                except Exception:
                    pass
                self._cursor_overlay = OverlayTextureRenderer(self.ctx)
                import mss as _mss
                _mon = _mss.mss().monitors[monitor_index] if monitor_index else None
                if _mon:
                    self._capture_origin = (_mon["left"], _mon["top"])
                print(f"[Main] cursor overlay enabled "
                      f"(capture origin {self._capture_origin})")
            except Exception as _e:
                print(f"[Main] cursor overlay disabled ({_e})")
                self._cursor_overlay = None
                self._cursor_sprite = None

        # Cache uniform objects to avoid per-frame dict lookups.
        # prog (SBS/stereo)
        self._u_eye_offset = self.prog['u_eye_offset']
        self._u_depth_strength = self.prog['u_depth_strength']
        self._u_feather_enabled = self.prog['u_feather_enabled']
        self._u_feather_width = self.prog['u_feather_width']
        self._u_viewport = self.prog['u_viewport']
        self._u_row_split = self.prog['u_row_split']
        self._u_row_parity = self.prog['u_row_parity']
        self._u_p_parallax = self.prog['u_parallax']
        try:
            self._u_ic = self.prog['u_ic_search']
            self._u_dstep = self.prog['u_depth_step_uv']
            self._u_icdbg = self.prog['u_ic_debug']
        except Exception:
            self._u_ic = None
            self._u_dstep = None
            self._u_icdbg = None
        # anaglyph
        self._u_ana_eye_offset = self.anaglyph_prog['u_eye_offset']
        self._u_ana_depth_strength = self.anaglyph_prog['u_depth_strength']
        self._u_ana_parallax = self.anaglyph_prog['u_parallax']
        self._u_ana_feather_enabled = self.anaglyph_prog['u_feather_enabled']
        self._u_ana_feather_width = self.anaglyph_prog['u_feather_width']
        self._u_ana_viewport = self.anaglyph_prog['u_viewport']
        # interleaved
        self._u_int_eye_offset = self.interleaved_prog['u_eye_offset']
        self._u_int_depth_strength = self.interleaved_prog['u_depth_strength']
        self._u_int_feather_enabled = self.interleaved_prog['u_feather_enabled']
        self._u_int_feather_width = self.interleaved_prog['u_feather_width']
        self._u_int_viewport = self.interleaved_prog['u_viewport']
        self._u_int_phase = self.interleaved_prog['u_interleave_phase']
        self._u_int_parallax = self.interleaved_prog['u_parallax']
        try:
            self._u_int_ic = self.interleaved_prog['u_ic_search']
            self._u_int_dstep = self.interleaved_prog['u_depth_step_uv']
        except Exception:
            self._u_int_ic = None
            self._u_int_dstep = None
        # 全屏输出：给所有程序设置 u_full_alpha（没有该 uniform 的程序跳过）
        try:
            _fa = 1.0 if getattr(self, "_fullscreen_output", False) else 0.0
            for _attr in ("prog", "anaglyph_prog", "interleaved_prog",
                          "vertical_interleaved_prog"):
                _p = getattr(self, _attr, None)
                if _p is not None and "u_full_alpha" in _p:
                    _p["u_full_alpha"].value = _fa
        except Exception as _fae:
            print(f"[viewer] u_full_alpha 设置失败: {_fae}", flush=True)
        # vertical interleaved
        self._u_vint_eye_offset = self.vertical_interleaved_prog['u_eye_offset']
        self._u_vint_depth_strength = self.vertical_interleaved_prog['u_depth_strength']
        self._u_vint_parallax = self.vertical_interleaved_prog['u_parallax']
        self._u_vint_feather_enabled = self.vertical_interleaved_prog['u_feather_enabled']
        self._u_vint_feather_width = self.vertical_interleaved_prog['u_feather_width']
        self._u_vint_viewport = self.vertical_interleaved_prog['u_viewport']

        # Initialize textures as None
        self.color_tex = None
        # 1×1 zero depth placeholder: render() 的贴图守卫要求 depth_tex 存在，
        # 2D 镜像（无深度帧）也要能画。第一帧真实深度到来时按模型分辨率重建。
        self.depth_tex = self.ctx.texture((1, 1), 1, dtype='f4',
                                          data=b'\x00\x00\x00\x00')
        self.depth_tex.repeat_x = False
        self.depth_tex.repeat_y = False
        self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._depth_tex_size = (1, 1)

        # 深度域 tap 的纹素步长（S3 起深度纹理 = 模型分辨率 224）。占位纹理
        # 阶段是 1×1；真实深度帧到来时 update_frame 会重设。anaglyph 的该
        # uniform 被编译器裁掉，用 in 守卫跳过。
        for _prog in (self.prog, self.anaglyph_prog,
                      self.interleaved_prog, self.vertical_interleaved_prog):
            if 'u_depth_resolution' in _prog:
                _prog['u_depth_resolution'].value = self._depth_tex_size

        # Cache for display mode and original toggle state
        self._last_display_mode = None
        self._last_show_original = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        
        # Load initial font
        self._update_font()
        vidmode = glfw.get_video_mode(glfw.get_monitors()[self.monitor_index])
        self.mon_w, self.mon_h = vidmode.size.width, vidmode.size.height

    def __del__(self):
        if hasattr(self, "overlay_renderer") and self.overlay_renderer is not None:
            self.overlay_renderer.release()

    def _reassert_overlay_style(self):
        """重申 overlay 必需的 exstyle 位。

        LAYERED|TRANSPARENT（点击穿透）与 NOACTIVATE（禁程序性激活）可能在
        GLFW 的 DECORATED/FLOATING 属性改写（全屏切换路径）中被清掉 —— 本
        家族的坑 _hide_from_taskbar 注释里早有记录。创建后、每次
        toggle_fullscreen 内部切换后、首帧上屏后各重申一次。
        """
        import sys as _sys
        if _sys.platform != "win32" or not self.window:
            return
        try:
            _hwnd = glfw.get_win32_window(self.window)
            _GWL_EXSTYLE = -20
            _WS_EX_LAYERED = 0x00080000
            _WS_EX_TRANSPARENT = 0x00000020
            _WS_EX_NOACTIVATE = 0x08000000
            _WS_EX_TOOLWINDOW = 0x00000080
            _WS_EX_APPWINDOW = 0x00040000
            _ex = ctypes.windll.user32.GetWindowLongPtrW(_hwnd, _GWL_EXSTYLE)
            _want = _WS_EX_LAYERED | _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE
            # 任务栏/Alt-Tab 隐藏也并入本函数（TOOLWINDOW 置位 + APPWINDOW
            # 清除）：显隐循环重显后没有任何路径再调 _hide_from_taskbar，
            # 样式纪律必须跟穿透位一样走统一重申。
            _target = (_ex | _want | _WS_EX_TOOLWINDOW) & ~_WS_EX_APPWINDOW
            if _target != _ex:
                ctypes.windll.user32.SetWindowLongPtrW(_hwnd, _GWL_EXSTYLE, _target)
            if _target & _WS_EX_LAYERED:
                # 分层窗口必须显式定不透明度：GLFW 的穿透路径只设样式位、
                # 从不调 SetLayeredWindowAttributes —— 未初始化的分层窗
                # alpha=0 全透明，像素根本不上屏（实测踩过：管线全绿、
                # LG 上却一直是原生桌面）。无条件钉死 255，幂等。
                ctypes.windll.user32.SetLayeredWindowAttributes(
                    ctypes.c_void_p(_hwnd), 0, 255, 2)  # LWA_ALPHA
        except Exception:
            pass

    def set_row_parity(self, offset):
        """行奇偶自动对齐：皮肤顶端相对显示器原点偏了奇数行时，
        翻转 u_interleave_phase —— 面板偏光膜按物理行分眼，窗口化
        皮肤随目标拖动，y 每变 1px 左右眼就对调一次（用户实测）。
        翻相位与把窗挪 1px 数学等价，但内容零偏移。
        offset = (客户区顶Y - 所在显示器顶Y) & 1
        """
        # 只存偏移：每帧渲染处统一应用（此前直写 uniform 会被渲染分支的
        # 每帧赋值覆盖）
        self._row_parity = int(offset) & 1

    def apply_live_params(self, params):
        """实时调参（logs/params.request 通道，设置对话框写入）。

        只接受每帧重算 uniform 的参数（parallax_px；旧 depth_ratio 分支保留兼容，
        每帧读取，IPD 是标定值不在实时组）；Convergence 的 uniform 是
        初始化时一次性赋值，必须对 4 个 program 各自重设，漏一个就是
        该显示模式下"调了没反应"。
        """
        try:
            v = params.get("Parallax (px)")
            if v is not None:
                self.parallax_px = float(v)
            v = params.get("Show FPS")
            if v is not None:
                self.show_fps = bool(v)
            v = params.get("VSync")
            if v is not None:
                self.vsync = bool(v)
                try:
                    import glfw as _g
                    _g.swap_interval(1 if self.vsync else 0)
                except Exception:
                    pass
            v = params.get("Interleave Phase")
            if v is not None:
                self.interleave_phase = bool(v)
            v = params.get("Convergence")
            if v is not None:
                self.convergence = c = float(v)
                for attr in ("prog", "anaglyph_prog", "interleaved_prog",
                             "vertical_interleaved_prog"):
                    prog = getattr(self, attr, None)
                    if prog is not None and 'u_convergence' in prog:
                        prog['u_convergence'].value = c
        except Exception as e:
            print(f"[viewer] apply_live_params: {type(e).__name__}: {e}", flush=True)

    def set_content_rect(self, l=None, t=None, r=None, b=None, feather=0.012):
        """设置"内容区域"（屏幕像素）。None 表示不启用（整屏都做视差）。

        只对这块区域做深度/视差，其余区域两个半屏完全相同 → 无立体感，
        用于"只把前台窗口/播放器做成 3D，桌面和 UI 保持清晰"。
        """
        try:
            if l is None or r is None or r <= l or b <= t:
                self._content_uv = (0.0, 0.0, 0.0, 0.0)
            else:
                w = float(self.mon_w or 1)
                h = float(self.mon_h or 1)
                x0 = max(0.0, min(1.0, l / w))
                x1 = max(0.0, min(1.0, r / w))
                # flipped_uv.y = 1 - y/H（屏幕上方 → 1）
                y0 = max(0.0, min(1.0, 1.0 - b / h))
                y1 = max(0.0, min(1.0, 1.0 - t / h))
                self._content_uv = (x0, y0, x1, y1)
                self._content_feather = float(feather)
            for attr in ("prog", "anaglyph_prog", "interleaved_prog",
                         "vertical_interleaved_prog"):
                prog = getattr(self, attr, None)
                if prog is None:
                    continue
                if "u_content" in prog:
                    prog["u_content"].value = self._content_uv
                if "u_content_feather" in prog:
                    prog["u_content_feather"].value = getattr(self, "_content_feather", 0.012)
        except Exception as e:
            print(f"[viewer] set_content_rect: {type(e).__name__}: {e}", flush=True)

    def _calculate_depth_map_viewport(self, win_w, win_h, tex_w, tex_h):
        """Cached viewport calculation for depth map mode"""
        cache_key = (win_w, win_h, tex_w, tex_h, self.display_mode, 
                     self.fill_16_9, self.show_original_in_depth_mode)
        
        # Check if we can reuse cached viewport
        if (self._viewport_cache['win_size'] == (win_w, win_h) and
            self._viewport_cache['tex_size'] == (tex_w, tex_h) and
            self._viewport_cache['display_mode'] == self.display_mode and
            self._viewport_cache['fill_16_9'] == self.fill_16_9 and
            self._viewport_cache['show_original'] == self.show_original_in_depth_mode):
            return self._viewport_cache['viewport']
        
        # Calculate viewport
        if self.fill_16_9:
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            center_x = win_w / 2.0
            center_y = win_h / 2.0
            viewport = (
                int(center_x - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
        else:
            disp_w, disp_h = tex_w, tex_h
            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9/16

            # Scale to fit window, preserving aspect ratio
            if window_aspect <= target_aspect:
                # Window is wider than content
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                # Window is taller than content
                view_w = win_w
                view_h = int(view_w * target_aspect)

            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2
            viewport = (offset_x, offset_y, view_w, view_h)
        
        # Update cache
        self._viewport_cache = {
            'win_size': (win_w, win_h),
            'tex_size': (tex_w, tex_h),
            'display_mode': self.display_mode,
            'fill_16_9': self.fill_16_9,
            'show_original': self.show_original_in_depth_mode,
            'viewport': viewport
        }
        
        return viewport
    
    def update_monitor_size(self):
        vidmode = glfw.get_video_mode(glfw.get_monitors()[self.monitor_index])
        self.mon_w, self.mon_h = vidmode.size.width, vidmode.size.height
    
    def set_edge_snap_gain(self, gain: float):
        """调整轮廓吸附阈值（越大越容易触发）。实时生效，无需重启。"""
        g = max(0.05, float(gain))
        for prog in getattr(self, "_edge_gain_progs", []):
            try:
                prog["u_edge_snap_gain"].value = g
            except Exception:
                pass
        print(f"[dibr] edge snap gain = {g:g}（阈值 {0.12/g:.3f}）", flush=True)

    def apply_3d_settings(self):
        if self.monitor_index == self.get_glfw_mon_index_mss(self.input_monitor_index):
            # WDA_EXCLUDEFROMCAPTURE 在本机（Win10 19045 + GTX1650 驱动）杀死
            # 一切 OpenGL 窗口的屏幕呈现（最小复现实测：饱和度 250→1.1，
            # 与置顶/全屏/哪块屏无关，摘除即恢复）—— D3D 窗口不受影响。
            # （历史注：同屏防递归曾依赖 WGC 窗口采集。）
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)    # Always on top
        else:
            if OS_NAME == "Windows":
                show_window_in_capture(self.window)
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)    # Disable

    def get_glfw_mon_index_mss(self, mss_monitor_index=1):
        """
        Map an MSS monitor index (1-based) to a GLFW monitor handle.

        MSS provides monitors like:
            monitors[0] -> all monitors bounding box
            monitors[1] -> first display
            monitors[2] -> second display, etc.

        GLFW gives a list of monitor handles that we can position windows on.
        This function matches them by position and size.
        """
        try:
            import mss
        except ImportError:
            print("[StereoWindow] mss not installed; using default monitor.")
            return 0

        with mss.mss() as sct:
            mss_monitors = sct.monitors

        # MSS uses index 1-based (0 = virtual bounding box)
        if mss_monitor_index < 1 or mss_monitor_index >= len(mss_monitors):
            print(f"[StereoWindow] Invalid MSS monitor index {mss_monitor_index}, defaulting to 1.")
            mss_monitor_index = 1

        mss_mon = mss_monitors[mss_monitor_index]
        mss_x, mss_y = mss_mon["left"], mss_mon["top"]
        mss_w, mss_h = mss_mon["width"], mss_mon["height"]

        glfw_monitors = glfw.get_monitors()
        if not glfw_monitors:
            print("[StereoWindow] No GLFW monitors detected.")
            return 0

        for i, gmon in enumerate(glfw_monitors):
            gx, gy = glfw.get_monitor_pos(gmon)
            gvm = glfw.get_video_mode(gmon)
            gw, gh = gvm.size.width, gvm.size.height

            if abs(gx - mss_x) <= 5 and abs(gy - mss_y) <= 5 and abs(gw - mss_w) <= 5 and abs(gh - mss_h) <= 5:
                # Found matching GLFW monitor
                return i

        print("[StereoWindow] No matching GLFW monitor found for MSS index, defaulting to primary.")
        return 0
    
    def _on_window_resize(self, window, width, height):
        """Handle window resize events"""
        self.window_size = (width, height)
        self._update_font()
        self._overlay_cache['image'] = None
        self._overlay_cache['fps_text'] = None

    def _update_font(self):
        """Update font size based on window dimensions"""
        # Calculate dynamic font size based on window height
        base_height = 1080  # Reference height
        scale_factor = min(self.frame_size[0] / base_height, 2.0)  # Cap scaling at 2x
        self.current_font_size = int(self.base_font_size * scale_factor * 0.8)  # Slightly smaller than linear scale
        
        try:
            self.font = ImageFont.truetype(self.font_type, self.current_font_size)
        except Exception:
            try:
                # Try default font (Pillow default)
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        
        # Update padding and spacing based on font size
        self.text_padding = max(5, int(self.current_font_size * 0.4))
        self.text_spacing = max(2, int(self.current_font_size * 0.2))
        
        # Update overlay cache position in case padding changed
        self._overlay_cache['pos'] = (self.text_padding, self.text_padding)

    def _generate_overlay_image(self, fps_text, latency_text, depth_text, mouse_text):
        """Rasterize the small overlay to RGBA numpy array (transparent background)."""
        if self.font is None:
            return None

        # Compose lines
        lines = []
        if self.show_fps and fps_text:
            lines.append(fps_text)
        if self.show_fps and self.total_latency > 0:
            latency_text = ""
            lines.append(latency_text)
        if self.show_depth_ratio and depth_text:
            lines.append(depth_text)
        if self.show_mouse_state and mouse_text:  # Add mouse state line
            lines.append(mouse_text)
        
        if not lines:
            return None

        # Estimate text size using PIL
        # small margin for padding/spacing
        padding = self.text_padding
        spacing = self.text_spacing

        # measure bounding boxes
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        widths = []
        heights = []
        bboxes = []
        for line in lines:
            try:
                # textbbox if available provides better metrics
                bbox = draw.textbbox((0, 0), line, font=self.font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = draw.textsize(line, font=self.font)
            widths.append(w)
            heights.append(h)
            bboxes.append((w, h))
        overlay_w = max(widths) + padding * 2
        overlay_h = sum(heights) + spacing * (len(lines) - 1) + padding * 2

        # Create overlay image and draw text
        overlay_img = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        x = padding
        y = padding
        for i, line in enumerate(lines):
            # Assign specific colors based on line content
            if "FPS:" in line:
                color = (0, 255, 0, 255)        # Green for FPS
            elif "Delay:" in line:
                color = (0, 255, 255, 255)        # Blue for Latency
            elif "Depth:" in line:
                color = (255, 255, 255, 255)      # Cyan for Depth
            elif "Mouse:" in line:
                color = (255, 255, 0, 255)      # Yellow for Mouse
            else:
                color = (255, 255, 255, 255)    # White fallback for other text
            
            draw.text((x, y), line, font=self.font, fill=color)
            y += heights[i] + spacing

        overlay_arr = np.array(overlay_img, dtype=np.uint8)  # H x W x 4
        return overlay_arr

    def _add_overlay(self, rgb_frame):
        """Add FPS and depth ratio overlay to the frame with minimal allocations."""
        # Skip overlay for depth map mode
        if self.display_mode == "Depth Map":
            return rgb_frame
        
        # If nothing to show or no font available, do nothing fast
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state) or self.font is None:
            return rgb_frame

        h, w, _ = rgb_frame.shape
        
        # freeze window size for rtmp streaming
        if self.display_mode == "Full-SBS":
            w = 2 * w
        self.frame_size = (w, h)
                
        # Depth ratio visibility check
        current_time = time.perf_counter()
        if current_time - self.last_depth_change_time < self.depth_display_duration:
            self.show_depth_ratio = True
        else:
            self.show_depth_ratio = False
        
        # Mouse state visibility check
        if current_time - self.last_mouse_toggle_time < self.mouse_display_duration:
            self.show_mouse_state = True
        else:
            self.show_mouse_state = False

        cache = self._overlay_cache

        # Throttle the *displayed* FPS/latency numbers. These change every frame
        # (e.g. "FPS: 59.8" -> "FPS: 60.1"), and the live values were previously
        # forcing a full PIL text re-rasterization on every single frame, which is
        # the main overlay-induced FPS drop. We only refresh the shown numbers a
        # couple of times per second; the frame's real FPS is unaffected.
        if (cache.get('disp_fps') is None or
                (current_time - cache.get('values_update', 0.0)) >= self.overlay_update_interval):
            cache['disp_fps'] = self.actual_fps
            cache['disp_latency'] = self.total_latency
            cache['values_update'] = current_time

        # Compose the strings to display (from the throttled snapshot values)
        fps_text = f"FPS: {cache['disp_fps']:.1f}" if self.show_fps else ""
        # 延迟不显示：多线程管线拿不到可靠端到端时间戳，只保留 FPS。
        latency_text = ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if self.mouse_pass_through else 'Normal'}" if self.show_mouse_state else ""

        # Decide whether to regenerate the rasterized overlay. Because the numbers
        # above are throttled, the text strings only change a few times per second,
        # so regeneration (the expensive part) is naturally rate-limited.
        needs_regen = False
        if cache['image'] is None:
            needs_regen = True
        elif (fps_text != cache.get('fps_text') or
              latency_text != cache.get('latency_text') or
              depth_text != cache.get('depth_text') or
              mouse_text != cache.get('mouse_text')):
            needs_regen = True

        if needs_regen:
            overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
            cache['image'] = overlay_arr
            cache['fps_text'] = fps_text
            cache['latency_text'] = latency_text
            cache['depth_text'] = depth_text
            cache['mouse_text'] = mouse_text
            cache['last_update'] = current_time
            # Pre-convert the overlay to float32 once so the per-frame blend below
            # doesn't repeat the RGBA->float conversion every frame.
            if overlay_arr is not None:
                cache['alpha_f'] = overlay_arr[..., 3:4].astype(np.float32) / 255.0
                cache['overlay_rgb_f'] = overlay_arr[..., :3].astype(np.float32)
            else:
                cache['alpha_f'] = None
                cache['overlay_rgb_f'] = None

        overlay_arr = cache['image']
        if overlay_arr is None:
            return rgb_frame

        ov_h, ov_w = overlay_arr.shape[:2]
        pos_x, pos_y = cache.get('pos', (self.text_padding, self.text_padding))

        # Clip overlay to frame boundaries
        if pos_x >= w or pos_y >= h:
            return rgb_frame
        end_x = min(w, pos_x + ov_w)
        end_y = min(h, pos_y + ov_h)
        ov_w_clipped = end_x - pos_x
        ov_h_clipped = end_y - pos_y
        if ov_w_clipped <= 0 or ov_h_clipped <= 0:
            return rgb_frame

        # Use the cached float32 overlay (already converted at regen time)
        alpha = cache['alpha_f'][0:ov_h_clipped, 0:ov_w_clipped]
        overlay_rgb = cache['overlay_rgb_f'][0:ov_h_clipped, 0:ov_w_clipped]
        frame_region = rgb_frame[pos_y:end_y, pos_x:end_x]

        # Alpha blending: result = overlay.rgb * alpha + frame * (1 - alpha)
        blended = overlay_rgb * alpha + frame_region.astype(np.float32) * (1.0 - alpha)
        # write back blended region into original frame (as uint8)
        rgb_frame[pos_y:end_y, pos_x:end_x] = np.clip(blended, 0, 255).astype(np.uint8)

        return rgb_frame

    def _update_overlay_texture(self):
        """Refresh the small text overlay texture without touching the RGB frame."""
        if self.font is None:
            if not getattr(self, "_font_warned", False):
                self._font_warned = True
                print(f"[overlay] 字体加载失败（font_type={getattr(self,'font_type',None)}）"
                      f"→ FPS/延迟不显示", flush=True)
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return
        if not getattr(self, "_overlay_state_logged", False):
            self._overlay_state_logged = True
            print(f"[overlay] show_fps={self.show_fps} font={getattr(self,'current_font_size',None)}px "
                  f"renderer={self.overlay_renderer is not None}", flush=True)
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return

        current_time = time.perf_counter()
        self.show_depth_ratio = current_time - self.last_depth_change_time < self.depth_display_duration
        self.show_mouse_state = current_time - self.last_mouse_toggle_time < self.mouse_display_duration
        if not (self.show_fps or self.show_depth_ratio or self.show_mouse_state):
            if self.overlay_renderer is not None:
                self.overlay_renderer.clear()
            return

        cache = self._overlay_cache
        if (cache.get('disp_fps') is None or
                (current_time - cache.get('values_update', 0.0)) >= self.overlay_update_interval):
            cache['disp_fps'] = self.actual_fps
            cache['disp_latency'] = self.total_latency
            cache['values_update'] = current_time

        fps_text = f"FPS: {cache['disp_fps']:.1f}" if self.show_fps else ""
        # 延迟不显示（测不准，已从 UI 移除）
        latency_text = ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""
        mouse_text = f"Mouse: {'Pass' if getattr(self, 'mouse_pass_through', False) else 'Normal'}" if self.show_mouse_state else ""

        if (fps_text == cache.get('fps_text') and
                latency_text == cache.get('latency_text') and
                depth_text == cache.get('depth_text') and
                mouse_text == cache.get('mouse_text') and
                cache.get('image') is not None):
            return

        overlay_arr = self._generate_overlay_image(fps_text, latency_text, depth_text, mouse_text)
        cache['image'] = overlay_arr
        cache['fps_text'] = fps_text
        cache['latency_text'] = latency_text
        cache['depth_text'] = depth_text
        cache['mouse_text'] = mouse_text
        cache['last_update'] = current_time
        if overlay_arr is not None and self.overlay_renderer is not None:
            self.overlay_renderer.update_texture(overlay_arr)
        elif self.overlay_renderer is not None:
            self.overlay_renderer.clear()

    def _render_overlay(self):
        if self.stream_mode is not None or self.overlay_renderer is None:
            return
        window_size = glfw.get_framebuffer_size(self.window)
        win_w, win_h = window_size
        base_x, base_y = self._overlay_cache.get(
            'pos', (self.text_padding, self.text_padding)
        )
        positions = [(base_x, base_y)]

        if self.display_mode == "Full-SBS" and self._texture_size:
            tex_w, tex_h = self._texture_size
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            center_y = win_h / 2.0
            left_vp = (
                int(win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w,
                render_h,
            )
            right_vp = (
                int(3 * win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w,
                render_h,
            )
            positions = [
                (left_vp[0] + base_x, win_h - (left_vp[1] + left_vp[3]) + base_y),
                (right_vp[0] + base_x, win_h - (right_vp[1] + right_vp[3]) + base_y),
            ]
        elif self.display_mode == "Half-SBS":
            positions = [
                (base_x, base_y),
                (max(base_x, win_w // 2 + base_x), base_y),
            ]
        elif self.display_mode in ("Full-TAB", "Half-TAB"):
            positions = [
                (base_x, base_y),
                (base_x, max(base_y, win_h // 2 + base_y)),
            ]

        # SBS/TAB 的画面在半屏里被压过（左右各 0.5 或上下各 0.5），
        # 叠加文字必须同样压缩，否则相对画面显得又宽又扁、位置也对不上。
        _ov_scale = (1.0, 1.0)
        if self.display_mode in ("Full-SBS", "Half-SBS"):
            _ov_scale = (0.5, 1.0)
        elif self.display_mode in ("Full-TAB", "Half-TAB"):
            _ov_scale = (1.0, 0.5)
        for position in positions:
            self.overlay_renderer.render(window_size, position, _ov_scale)

        try:
            globals()["_OV_CALLS"] = globals().get("_OV_CALLS", 0) + 1
        except Exception:
            pass
        self._render_seq = getattr(self, "_render_seq", 0) + 1
        _before = globals().get("_CUR_DRAWS", 0)
        self._render_cursor_overlay(window_size)
        try:
            import os as _osd2, time as _t2
            if _osd2.environ.get("D2S_CURSOR_DEBUG") == "1":
                _now = _t2.perf_counter()
                if _now - globals().get("_CUR_T", 0.0) > 2.0:
                    globals()["_CUR_T"] = _now
                    _n = globals().get("_CUR_DRAWS", 0) - _before
                    _calls = globals().get("_OV_CALLS", 0)
                    globals()["_OV_CALLS"] = 0
                    _tot = globals().get("_CUR_DRAWS", 0) - globals().get("_CUR_TOT0", 0)
                    globals()["_CUR_TOT0"] = globals().get("_CUR_DRAWS", 0)
                    print(f"[cursor-dbg] 2秒内 overlay 调用 {_calls} 次, "
                          f"光标绘制合计 {_tot} 次 (每次调用 "
                          f"{(_tot / _calls if _calls else 0):.1f} 个), "
                          f"display={self.display_mode}", flush=True)
        except Exception:
            pass

        # 调试：D2S_DUMP_FRAME=1 时把当前 GL 帧导出一张 PNG（形状问题定位用）
        import os as _os
        if _os.environ.get("D2S_DUMP_FRAME") == "1" and not getattr(self, "_frame_dumped", False):
            self._frame_dumped = True
            try:
                import cv2 as _cv2
                import numpy as _np
                w, h = window_size
                data = self.ctx.screen.read(components=3, alignment=1)
                img = _np.frombuffer(data, dtype=_np.uint8).reshape(h, w, 3)[::-1]
                path = _os.path.join(_os.environ.get("TEMP", "."), "d2s_frame.png")
                _cv2.imwrite(path, img)
                print(f"[dump] 帧已导出 {path}", flush=True)
            except Exception as _de:
                print(f"[dump] 失败: {_de}", flush=True)

    def _render_cursor_overlay(self, window_size):
        """Draw the live pointer at the screen plane, straight after the stereo
        pass. Its latency is then one render frame, not the whole pipeline."""
        if self._cursor_overlay is None or self._cursor_sprite is None:
            return
        # 是否绘制 = 系统此刻是否还有"光标句柄"。播放器空闲隐藏时调用
        # ShowCursor(FALSE)/SetCursor(NULL)，GetCursorInfo 的 hCursor 会变 NULL；
        # 我们只做透明替换（不动句柄），所以这个判据能如实反映"该不该显示"。
        # 不用计时器：计时器会导致正常静置时光标凭空消失。
        try:
            import ctypes as _ct
            _CI = _CursorInfoT
            _ci = _CI()
            _ci.cbSize = _ct.sizeof(_CI)
            if not _ct.windll.user32.GetCursorInfo(_ct.byref(_ci)):
                return
            # 两条判据任一成立就认为"系统不想显示光标"：
            #   ① hCursor 为空 —— 播放器用 SetCursor(NULL) 隐藏
            #   ② CURSOR_SHOWING 位为 0 —— 播放器用 ShowCursor(FALSE) 隐藏
            # 空白替换不会动句柄，也不会清这个位，所以两条都只在真正隐藏时成立。
            # 调试钩子：模拟"浏览器 SetCursor(NULL)"，用来验证判据真的生效
            try:
                import os as _osg
                if _osg.environ.get("D2S_FAKE_CURSOR_NULL") == "1":
                    return
            except Exception:
                pass
            if not _ci.hCursor or not (_ci.flags & 0x00000001):
                return
            # 有些播放器不是 SetCursor(NULL)/ShowCursor(FALSE)，而是把自己
            # 的光标换成一张**全透明位图**——此时句柄有效、SHOWING=1，上面
            # 两条判据都不成立。所以再看一眼位图透明度：
            #   句柄是我们的空白槽位（identify 成功）→ 应用要形状，照画；
            #   句柄是应用自己的且位图全透明 → 应用在隐藏，我们也不画。
            try:
                import cursor_hide as _chv
                if _chv.identify(_ci.hCursor) == -1:
                    _nowv = time.perf_counter()
                    if _nowv - getattr(self, "_alpha_t", 0.0) > 0.25:
                        self._alpha_t = _nowv
                        _bm = self._cursor_sprite._extract(_ci.hCursor)
                        if _bm is not None:
                            _a = float(_bm[0][..., 3].mean()) / 255.0
                            self._app_alpha = _a
                    if getattr(self, "_app_alpha", 1.0) < 0.02:
                        return
            except Exception:
                pass
        except Exception as _e:
            # 判据读不到就不画：以前这里是 pass，异常后继续往下画，
            # 等于永远不知道系统已经藏了光标（这是"浏览器全屏藏光标无效"的根因）。
            if not getattr(self, "_cursor_gate_err", False):
                self._cursor_gate_err = True
                print(f"[cursor] 状态判据异常（按不画处理）: {_e}", flush=True)
            return
        # SBS 每帧要渲染两遍（左右眼各一遍），光标只能在其中一遍画那一对。
        # 用【时间戳】节流而不是帧号：GPU 直通路径不走 update_frame，
        # 帧号永远是 None（实测：闸永远闭合 → 光标完全不画）。
        # §6 每渲染帧只画一批：自增渲染序号去重（时间戳节流是反模式）
        _seq = getattr(self, "_render_seq", 0)
        if getattr(self, "_cursor_drawn_seq", -1) == _seq:
            return
        self._cursor_drawn_seq = _seq
        # §3 探测只在鼠标移动时（探测会打败播放器的自动隐藏）
        # 鼠标是否在动：探测只在这段时间内进行（探测会打败播放器自动隐藏）
        try:
            import ctypes as _ctm
            class _PT2(_ctm.Structure):
                _fields_ = [("x", _ctm.c_long), ("y", _ctm.c_long)]
            _pt2 = _PT2()
            _ctm.windll.user32.GetCursorPos(_ctm.byref(_pt2))
            _pos2 = (_pt2.x, _pt2.y)
            if getattr(self, "_last_pos", None) is None:
                self._last_pos = _pos2
                self._cur_move_t = time.perf_counter()
            elif self._last_pos != _pos2:
                self._last_pos = _pos2
                self._cur_move_t = time.perf_counter()
        except Exception:
            pass
        _now = time.perf_counter()
        _moved = (_now - getattr(self, "_cur_move_t", 0.0)) < 0.3
        # 注意：这里曾经有一条"鼠标静止 3 秒就主动收起"的兜底，已删除。
        # 它的代价是任何"隐藏是否生效"的验证都会被它干扰（停手 3 秒就消失，
        # 分不清是我们藏住了还是定时器收起了）。隐藏只应由真实状态决定：
        # 系统光标句柄为空 / CURSOR_SHOWING=0 / 应用位图透明。
        got = self._cursor_sprite.get(allow_probe=_moved)
        if got is None:
            return
        rgba, hx, hy, (cx, cy) = got
        _size_norm = False
        # 尺寸归一：我们内置的光标位图是 112px 级别（远大于系统 32x32），
        # 直接画会明显偏大。这里按【alpha 包围盒】对齐系统光标尺寸：
        #   目标高度 = SM_CYCURSOR（内容空间像素）× 半屏压缩比
        # 再按包围盒裁掉透明边，顺带修正热点。
        try:
            import numpy as _npz
            _m0 = getattr(self, "display_mode", "")
            _half = 0.5 if _m0 in ("Half-SBS", "Half-TAB") else 1.0
            _alpha = _npz.asarray(rgba)[..., 3]
            _ys, _xs = _npz.nonzero(_alpha > 8)
            if len(_xs):
                _x0, _x1 = int(_xs.min()), int(_xs.max()) + 1
                _y0, _y1 = int(_ys.min()), int(_ys.max()) + 1
                _sysh = float(getattr(self, "_real_glyph_h", 0) or
                              globals().get("_REAL_GLYPH_H") or
                              _real_cursor_glyph_height(_cursor_base_size()))
                globals()["_REAL_GLYPH_H"] = _sysh
                _crop = _npz.asarray(rgba)[_y0:_y1, _x0:_x1]
                # 只按高度对齐系统光标：半屏的水平压缩由后面的
                # _prescaled 分支统一做（这里再乘 0.5 会压两次）
                try:
                    from utils import CURSOR_SCALE as _csc
                except Exception:
                    _csc = 1.0
                _factor = (_sysh * float(_csc)) / max(1.0, float(_crop.shape[0]))
                import cv2 as _cvz
                _nw = max(1, int(round(_crop.shape[1] * _factor)))
                _nh = max(1, int(round(_crop.shape[0] * _factor)))
                rgba = _cvz.resize(_crop, (_nw, _nh), interpolation=_cvz.INTER_AREA)
                hx = (hx - _x0) * _factor
                hy = (hy - _y0) * _factor
                _size_norm = True
                globals()["_CURSOR_SIZELOG"] = (
                    f"bbox={_x1 - _x0}x{_y1 - _y0} sysh={_sysh:g} half={_half:g} "
                    f"factor={_factor:.3f} -> {_nw}x{_nh}")
        except Exception:
            pass
        # 空白光标（我们替换的透明 DIB）提取后会变成"全黑不透明"，
        # 直接画就是黑方块 —— 这种坚决不画
        try:
            import numpy as _np
            _a = rgba[..., 3].astype(_np.float32).mean() / 255.0
            _rgb = rgba[..., :3].astype(_np.float32).mean()
            if _a > 0.98 and _rgb < 4.0:
                return
        except Exception:
            pass
        # SBS/TAB 会把画面横向（或纵向）压到 0.5，光标必须同样压缩才对得上。
        # 交给 GL 缩小采样会糊（mipmap 也只是三线性近似）；这里用 cv2 的
        # INTER_AREA 做精确 2:1 盒式平均，再以 1:1 尺寸上传绘制。
        _mode0 = getattr(self, "display_mode", "")
        _mx, _my = ((0.5, 1.0) if _mode0 == "Half-SBS" else
                    (1.0, 0.5) if _mode0 == "Half-TAB" else (1.0, 1.0))
        _prescaled = False
        if _mx != 1.0 or _my != 1.0:
            try:
                import cv2 as _cv2
                _nw = max(1, int(round(rgba.shape[1] * _mx)))
                _nh = max(1, int(round(rgba.shape[0] * _my)))
                rgba = _cv2.resize(rgba, (_nw, _nh), interpolation=_cv2.INTER_AREA)
                # 注意：热点不能在这里缩放。绘制位置后面还会按半屏做 //2，
                # 热点若也乘 0.5 就会被缩两次 → 文本光标（热点居中）偏移半个
                # 热点宽度。位图缩小、位置减半，热点保持原值即等价于半屏内
                # 的 (cx/2 - 热点/2)。
                _prescaled = True
            except Exception:
                _prescaled = False
        if not self._cursor_overlay.update_texture(rgba):
            return
        # 诊断：把实际画的光标位图尺寸/透明像素占比打一次（形状问题定位用）
        try:
            _sig = (rgba.shape[0], rgba.shape[1],
                    float(rgba[..., 3].astype("float32").mean()))
            if getattr(self, "_cursor_sig", None) != _sig:
                self._cursor_sig = _sig
                _nz = int((rgba[..., 3] > 8).sum())
                if os.environ.get("D2S_CURSOR_DEBUG") == "1":
                    print(f"[cursor] sprite {rgba.shape[1]}x{rgba.shape[0]} "
                          f"alpha_mean={_sig[2]:.3f} 非透明像素={_nz}", flush=True)
        except Exception:
            pass
        # The sprite is already at its final pixel size; sampling it with
        # LINEAR adds a second blur pass (half-texel offset). Nearest keeps it
        # as sharp as the upscaled bitmap.
        if self._cursor_overlay.texture is not None:
            # SBS 里光标要横向压到 0.5（112px → 56px）。GL 缩小采样若没有
            # mipmap 就会混叠成硬锯齿 —— 这正是"我们画的丑、系统的漂亮"的原因：
            # 系统是合成器做的带过滤缩放。建 mipmap 后用三线性缩小即可接近。
            try:
                self._cursor_overlay.texture.build_mipmaps()
            except Exception:
                pass
            try:
                self._cursor_overlay.texture.filter = (
                    moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            except Exception:
                self._cursor_overlay.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        win_w, win_h = window_size
        tex_w, tex_h = self._texture_size or (win_w, win_h)
        sx = win_w / float(tex_w)
        sy = win_h / float(tex_h)
        ox, oy = self._capture_origin
        x = int(round((cx - ox) * sx)) - hx
        y = int(round((cy - oy) * sy)) - hy
        if x < -rgba.shape[1] or y < -rgba.shape[0] or x > win_w or y > win_h:
            return
        # SBS/TAB：画面被左右（或上下）分成两只眼，光标要在两个半屏各画一次
        # 每只眼的画面在半屏里的缩放比不同：
        #   Full-*  ：1:1（整幅画占半屏）
        #   Half-*  ：0.5（整幅画压进半屏）→ 光标坐标也要乘 0.5
        mode = getattr(self, "display_mode", "")
        hw, hh = win_w // 2, win_h // 2
        # SBS 只压水平、TAB 只压垂直
        if mode == "Full-SBS":
            pts, scale, axis = [(x, y), (x + hw, y)], (1.0, 1.0), "h"
        elif mode == "Half-SBS":
            pts = [(x // 2, y), (hw + x // 2, y)]
            scale = (1.0, 1.0) if _prescaled else (0.5, 1.0)
            axis = "h"
        elif mode == "Full-TAB":
            pts, scale, axis = [(x, y), (x, y + hh)], (1.0, 1.0), "v"
        elif mode == "Half-TAB":
            pts = [(x, y // 2), (x, hh + y // 2)]
            scale = (1.0, 1.0) if _prescaled else (1.0, 0.5)
            axis = "v"
        else:
            pts, scale, axis = [(x, y)], (1.0, 1.0), ""

        # 一次性诊断：系统光标尺寸 / 位图尺寸 / 是否预缩放 / 最终绘制尺寸
        _szkey = (getattr(self, "_cursor_scale_seen", None), mode)
        try:
            from utils import CURSOR_SCALE as _cs_now
        except Exception:
            _cs_now = 1.0
        _szkey = (round(float(_cs_now), 2), mode)
        if _szkey != getattr(self, "_cursor_size_key", None):
            self._cursor_size_key = _szkey
            try:
                import ctypes as _cts2
                _smw = _smh = int(_cursor_base_size())   # 注册表实际指针尺寸
                print(f"[cursor] size: system={_smw}x{_smh} "
                      f"final={rgba.shape[1]}x{rgba.shape[0]} "
                      f"prescaled={_prescaled} scale={scale} mode={mode} "
                      f"{globals().get('_CURSOR_SIZELOG', '')}",
                      flush=True)
            except Exception:
                pass

        # 每只眼裁在自己那半屏：越过分割线就被裁掉（而不是整只不画）
        import moderngl as _mgl
        for _i, _p in enumerate(pts):
            scissor = None
            if axis == "h" and mode != "Full-SBS":
                scissor = (0 if _i == 0 else hw, 0, hw, win_h)
            elif axis == "h":
                scissor = (0 if _i == 0 else hw, 0, hw, win_h)
            elif axis == "v":
                scissor = (0, 0 if _i == 0 else hh, win_w, hh)
            _sc_ok = False
            if scissor:
                # moderngl 本机没有 SCISSOR_TEST 常量：直接调 opengl32
                try:
                    import ctypes as _ct2
                    _gl = ctypes.windll.opengl32
                    _gl.glEnable(0x0C11)                 # GL_SCISSOR_TEST
                    _gl.glScissor(int(scissor[0]), int(scissor[1]),
                                  int(scissor[2]), int(scissor[3]))
                    _sc_ok = True
                except Exception:
                    _sc_ok = False
            # 光标贴图是直通 alpha：必须开混合，否则透明区画成黑色方块
            try:
                import ctypes as _ctb
                _glb = _ctb.windll.opengl32
                _glb.glEnable(0x0BE2)                    # GL_BLEND
                _glb.glBlendFunc(0x0302, 0x0303)         # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
            except Exception:
                pass
            try:
                import os as _osdbg
                if _osdbg.environ.get("D2S_CURSOR_DEBUG") == "1":
                    globals().setdefault("_CUR_POSLOG", [])
                    _lg = globals()["_CUR_POSLOG"]
                    _lg.append((int(_p[0]), int(_p[1]), float(scale[0])))
                    if len(_lg) >= 2:
                        print(f"[cursor-pos] {_lg[-2]} {_lg[-1]} display={self.display_mode}",
                              flush=True)
                        globals()["_CUR_POSLOG"] = []
            except Exception:
                pass
            self._cursor_overlay.render(window_size, _p, scale)
            try:
                import os as _osd
                if _osd.environ.get("D2S_CURSOR_DEBUG") == "1":
                    globals()["_CUR_DRAWS"] = globals().get("_CUR_DRAWS", 0) + 1
            except Exception:
                pass
            try:
                import ctypes as _ctc
                _ctc.windll.opengl32.glDisable(0x0BE2)
            except Exception:
                pass
            if _sc_ok:
                try:
                    import ctypes as _ct3
                    _ct3.windll.opengl32.glDisable(0x0C11)
                except Exception:
                    pass

    def position_on_monitor(self, monitor_index=0):
        """Optimized monitor positioning"""
        if getattr(self, "_force_win", False):
            return                      # 基准测试强制尺寸时不做贴合缩放
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height
            if self.window_size[0] >= mon_w or self.window_size[1] >= mon_h:
                if self.window_size[0] >= mon_w:
                    target_w = mon_w
                    target_h = int(self.window_size[1] * mon_w / self.window_size[0]) 
                    self.window_size = (target_w, target_h)
                if self.window_size[1] >= mon_h:
                    target_h = mon_h
                    target_w = int(self.window_size[0] * mon_h / self.window_size[1])
                    self.window_size = (target_w, target_h)
                glfw.set_window_size(self.window, self.window_size[0], self.window_size[1])
            x = mon_x + (mon_w - self.window_size[0]) // 2
            y = mon_y + (mon_h - self.window_size[1]) // 2
            glfw.set_window_pos(self.window, x, y)
                

    def get_current_monitor(self):
        """Optimized monitor detection with early returns"""
        monitors = glfw.get_monitors()
        if not monitors:
            return None
            
        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        window_center_x = win_x + win_w // 2
        window_center_y = win_y + win_h // 2
        
        for monitor in monitors:
            monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            if (monitor_x <= window_center_x < monitor_x + vidmode.size.width and
                monitor_y <= window_center_y < monitor_y + vidmode.size.height):
                return monitor
        return monitors[0]

    def move_to_adjacent_monitor(self, direction):
        """Optimized monitor switching"""
        monitors = glfw.get_monitors()
        if len(monitors) > 1:
            new_index = (self.monitor_index + direction) % len(monitors)
            self.position_on_monitor(new_index)
            self.monitor_index = new_index
        else:
            self.position_on_monitor(self.monitor_index)
        if self.use_3d:
            self.apply_3d_settings()
        self.update_monitor_size()

    def get_glfw_monitor_index(self, monitor):
        """Get the GLFW monitor index for a given monitor."""
        monitors = glfw.get_monitors()
        for idx, mon in enumerate(monitors):
            if mon == monitor:
                return idx
        return -1

    def toggle_fullscreen(self):
        """Optimized fullscreen toggle with reduced GLFW calls"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            self._reassert_overlay_style()
            return

        if not self._fullscreen:
            if not self.use_3d and self.capture_mode == "Window" and self.get_glfw_mon_index_mss(self.input_monitor_index) == self.get_glfw_monitor_index(current_monitor):
                glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, True)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
            else:
                # Enter fullscreen
                self._last_window_position = glfw.get_window_pos(self.window)
                self._last_window_size = glfw.get_window_size(self.window)

                # Get monitor info
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                vidmode = glfw.get_video_mode(current_monitor)
                full_w, full_h = vidmode.size.width, vidmode.size.height

                # Make the window undecorated and floating
                glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                if self.fix_aspect:
                    monitor_aspect = full_w / full_h
                    if monitor_aspect > self.aspect:
                        # Monitor is wider than target aspect
                        new_h = full_h
                        new_w = int(new_h * self.aspect)

                    else:
                        # Screen is taller: fit by width.
                        new_w = full_w
                        new_h = int(full_w / self.aspect)

                    glfw.set_window_size(self.window, new_w, new_h)

                    # Center window on screen
                    center_x = mon_x + (full_w - new_w) // 2
                    center_y = mon_y + (full_h - new_h) // 2
                    glfw.set_window_pos(self.window, center_x, center_y)
                else:
                    glfw.set_window_size(self.window, full_w, full_h)
                    glfw.set_window_pos(self.window, mon_x, mon_y)
            self._fullscreen = True
        else:
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
            else:
                # Exit fullscreen
                glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)

                restore_w, restore_h = self._last_window_size or self.window_size
                restore_x, restore_y = self._last_window_position or (0, 0)
                
                if not self._last_window_position:
                    vidmode = glfw.get_video_mode(current_monitor)
                    mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                    restore_x = mon_x + (vidmode.size.width - restore_w) // 2
                    restore_y = mon_y + (vidmode.size.height - restore_h) // 2

                glfw.set_window_size(self.window, restore_w, restore_h)
                glfw.set_window_pos(self.window, restore_x, restore_y)
            self._fullscreen = False
            glfw.set_window_attrib(self.window, glfw.MOUSE_PASSTHROUGH, False)
        self.window_size = glfw.get_window_size(self.window)
    
    def on_key_event(self, window, key, scancode, action, mods):
        """皮肤模式不响应任何键盘：Pause 的 2D/3D 切换由主程序
        RegisterHotKey 全局处理，这里留空避免误触（ESC 曾直接把窗口关掉、
        看起来就像黑屏；ENTER/TAB/方向键等会打乱皮肤几何与视差）。"""
        return

    def _hide_from_taskbar(self):
        """Definitively hide from taskbar and Alt+Tab.

        Sets WS_EX_TOOLWINDOW *and clears WS_EX_APPWINDOW*. The old patch
        only set TOOLWINDOW; later exstyle rewrites (passthrough/fullscreen
        handling) have been observed re-adding APPWINDOW, which forces a
        taskbar button even on tool windows.
        """
        import sys as _sys
        if _sys.platform != "win32" or not self.window:
            return
        try:
            import ctypes as _ct
            _hwnd = glfw.get_win32_window(self.window)
            _GWL_EXSTYLE = -20
            _WS_EX_TOOLWINDOW = 0x00000080
            _WS_EX_APPWINDOW = 0x00040000
            _ex = _ct.windll.user32.GetWindowLongPtrW(_hwnd, _GWL_EXSTYLE)
            _ct.windll.user32.SetWindowLongPtrW(
                _hwnd, _GWL_EXSTYLE,
                (_ex | _WS_EX_TOOLWINDOW) & ~_WS_EX_APPWINDOW)
        except Exception:
            pass

    # -------------------------------------------------- GPU 直通（M1）
    def gpu_depth_init(self, res: int, prep_dtype: str = 'f4'):
        """GPU 直通的一次性资源：224 预处理 FBO（CHW 平铺）+ 深度纹理。

        prep_dtype：预处理纹理的像素格式。DML 路径用 'f2'（R16F，回读量
        600KB→300KB，省约 0.15ms）；CUDA 互操作路径必须 'f4'（内核按 float
        读显存）。必须在 GL 线程调用（与渲染同 context）。
        """
        if (getattr(self, "_gpu_res", None) == res
                and getattr(self, "_prep_dtype", None) == prep_dtype):
            return
        try:
            import os as _osp
            if _osp.environ.get("D2S_PREP_F32", ""):   # A/B 对照：强制 R32F
                prep_dtype = 'f4'
            # 零拷贝实验（P3）：flat 几何 —— W%64==0、W*H==3R²，行距天然
            # 256 对齐无 padding，D3D12 单拷贝稠密（P2 结论）。需要 R32F。
            from gpu.dml_shim import zc_enabled as _zc_enabled
            _zc = _zc_enabled()
            if _zc and prep_dtype != 'f4':
                prep_dtype = 'f4'
            self._prep_flat = _zc and (res * res * 3) % 192 == 0
        except Exception:
            self._prep_flat = False
        self._gpu_res = res
        self._prep_dtype = prep_dtype
        # 深度纹理（GPU 模式下 depth=None 走 update_frame，不会自建）
        if self._depth_tex_size != (res, res):
            if self.depth_tex:
                self.depth_tex.release()
            self.depth_tex = self.ctx.texture((res, res), 1, dtype='f4')
            self.depth_tex.repeat_x = False
            self.depth_tex.repeat_y = False
            self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._depth_tex_size = (res, res)
        # 模型原始输出的落地纹理（与 depth_tex 分开）：CUDA 推理会把原始
        # 量纲（0.2~5.4）直接写进绑定纹理，若与着色器采样的 depth_tex 是
        # 同一张，渲染就会拿到未归一化的深度 → 视差放大数倍。
        if getattr(self, "_raw_tex", None) is None or self._raw_tex.size != (res, res):
            if getattr(self, "_raw_tex", None):
                self._raw_tex.release()
            self._raw_tex = self.ctx.texture((res, res), 1, dtype='f4')
            self._raw_tex.repeat_x = False
            self._raw_tex.repeat_y = False
            self._raw_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            for _prog in (self.prog, self.anaglyph_prog,
                          self.interleaved_prog, self.vertical_interleaved_prog):
                if 'u_depth_resolution' in _prog:
                    _prog['u_depth_resolution'].value = (res, res)
        # 预处理目标：默认 3R x R 单通道 float（行内 [R|G|B] 平铺）；
        # D2S_ZEROCOPY 时改 flat 几何（W=192、H=3R²/W）—— 线性序同样是 CHW，
        # 但行距 768B 满足 D3D12 拷贝的 256 对齐且无 padding（P2）。
        if getattr(self, "_prep_flat", False):
            _pw, _ph = 192, (res * res * 3) // 192
        else:
            _pw, _ph = res * 3, res
        self._prep_tex = self.ctx.texture((_pw, _ph), 1, dtype=prep_dtype)
        self._prep_tex.repeat_x = False
        self._prep_tex.repeat_y = False
        self._prep_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._prep_fbo = self.ctx.framebuffer([self._prep_tex])
        # 归一化与 d2s_ops.resize_model_input 完全一致（ImageNet）；
        # color_tex 是 RGB8（GL 采样已归一化到 [0,1]，等价 numpy 的 /255）。
        # 模型是 DepthART（MobileNetV4-M-slim-SPF, 224），官方输入约定是
        # RGB 图像 + ImageNet 归一化；ONNX 文件本身不记录通道顺序，所以
        # 靠"上游约定 + 本类 _channel_selfcheck 的已知色自检"共同保证。
        self._prep_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                out vec4 out_c;
                uniform sampler2D tex;
                uniform float res;
                uniform int flat_w;   // 0=行内平铺(3R 宽)；>0=flat 几何宽度
                const vec3 MEAN = vec3(0.485, 0.456, 0.406);
                const vec3 STD  = vec3(0.229, 0.224, 0.225);
                void main(){
                    int c; float xr; float yr;
                    if (flat_w > 0) {
                        int fl = int(gl_FragCoord.y) * flat_w + int(gl_FragCoord.x);
                        int rr = int(res) * int(res);
                        c = fl / rr;
                        int rem = fl - c * rr;
                        int row = rem / int(res);
                        xr = float(rem - row * int(res)) + 0.5;
                        yr = float(row) + 0.5;
                    } else {
                        int i = int(gl_FragCoord.x);
                        c = i / int(res);
                        xr = float(i - c * int(res)) + 0.5;
                        yr = gl_FragCoord.y;
                    }
                    vec2 suv = vec2(xr / res, yr / res);
                    vec3 col = texture(tex, suv).rgb;
                    float v = c == 0 ? col.r : (c == 1 ? col.g : col.b);
                    out_c = vec4((v - MEAN[c]) / STD[c], 0.0, 0.0, 1.0);
                }
            """)
        self._prep_prog['res'].value = float(res)
        self._prep_prog['flat_w'].value = _pw if getattr(self, "_prep_flat", False) else 0
        self._prep_vao = self.ctx.vertex_array(
            self._prep_prog,
            [(self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1],
                                       np.float32)), "2f", "in_pos")])
        self._channel_selfcheck(res)

    def _channel_selfcheck(self, res: int):
        """已知纯色走一遍真实预处理链，验证 color_tex 的通道顺序是 RGB。

        输入 R=200/G=100/B=50 的纹理 → prep shader（ImageNet 归一化 + CHW
        平铺）→ 回读三个通道平面。期望：
            R (200/255-0.485)/0.229 ≈ +1.31
            G (100/255-0.456)/0.224 ≈ -0.28
            B ( 50/255-0.406)/0.225 ≈ -0.93
        顺序错了（如 BGR）三个值会明显对不上，据此可直接判定。
        """
        try:
            import numpy as _np
            val = (200, 100, 50)
            img = _np.empty((res, res, 3), _np.uint8)
            img[:, :, 0] = val[0]
            img[:, :, 1] = val[1]
            img[:, :, 2] = val[2]
            t = self.ctx.texture((res, res), 3, img.tobytes())
            t.repeat_x = False
            t.repeat_y = False
            t.use(0)
            self._prep_prog['tex'].value = 0
            self._prep_fbo.use()
            self.ctx.viewport = (0, 0, self._prep_tex.size[0], self._prep_tex.size[1])
            self._prep_vao.render(moderngl.TRIANGLE_STRIP)
            arr = _np.frombuffer(self._prep_tex.read(),
                                 _np.float16 if res and getattr(self, "_prep_dtype", "f4") == "f2"
                                 else _np.float32,
                                 count=res * res * 3).astype(_np.float32)
            if getattr(self, "_prep_flat", False):
                got = [float(arr.reshape(3, res, res)[c].mean()) for c in range(3)]
            else:
                arr = arr.reshape(res, 3, res)
                got = [float(arr[:, c, :].mean()) for c in range(3)]
            exp = [(val[c] / 255.0 - m) / s for c, (m, s) in
                   enumerate(zip((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)))]
            err = max(abs(g - e) for g, e in zip(got, exp))
            if err < 0.02:
                print(f"[color] 通道顺序自检通过 RGB（R={got[0]:+.2f} "
                      f"G={got[1]:+.2f} B={got[2]:+.2f}，误差 {err:.3f}）",
                      flush=True)
            else:
                order = sorted(range(3), key=lambda c: -got[c])
                print(f"[color] 通道顺序异常！期望 RGB，实测最亮通道={order[0]}"
                      f"（R={got[0]:+.2f} G={got[1]:+.2f} B={got[2]:+.2f}，"
                      f"误差 {err:.3f}）", flush=True)
            t.release()
            self.ctx.screen.use()
        except Exception as e:
            print(f"[color] 通道自检失败: {type(e).__name__}: {e}", flush=True)

    # ------------------------------------------------------------------
    # GPU 后处理：归一化 + AA + EMA（替代 CPU 上的 numpy 链路）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 空间后处理块（depth_post.DepthPost）：归一化→闭运算→坡度限幅→AA。
    # gpu_post_init 保留方法名并镜像 _pp_res/_ema_tex 两个契约属性
    # （depth_mc.MC 经 host._pp_res/_ema_tex 读取；FakeWin 诊断同面）。
    # ------------------------------------------------------------------
    def _post_get(self):
        p = getattr(self, "_post", None)
        if p is None:
            p = self._post = DepthPost(self)
        return p

    def gpu_post_init(self, res: int):
        post = self._post_get()
        post.ensure(res)
        self._pp_res = post.res        # 契约镜像（MC.init 引导读）
        self._ema_tex = post.ema_tex   # 契约镜像（MC.ema_pass 读）


    # ------------------------------------------------------------------
    # MC-EMA 时域块（depth_mc.MC）薄委托：保留 mc_init/mc_step 方法签名是
    # 外部契约——main.py 调用点与 tools/mc_repro.py 的 FakeWin 未绑定调用
    # 都依赖这对名字。每帧在 postprocess 前跑一次 mc_step。
    # ------------------------------------------------------------------
    def _mc_pipe_get(self):
        p = getattr(self, "_mc_pipe", None)
        if p is None:
            p = self._mc_pipe = MC(self)
        return p

    def mc_init(self, res: int):
        self._mc_pipe_get().init(res)

    def mc_step(self, raw_tex, lo: float, hi: float, z_norm):
        """运动补偿一步（postprocess 前）。返回 (cut, stats)。详见 depth_mc.MC.step。"""
        return self._mc_pipe_get().step(raw_tex, lo, hi, z_norm)

    def postprocess_gpu(self, raw_tex, lo: float, hi: float,
                        alpha: float, aa_texels: float = 0.0, grow: int = 0,
                        slope_target: float = 0.0, parallax_px: float = 0.0,
                        iters: int = 6, close_px: int = 0, cut: bool = False):
        """raw → [归一化(+grow)] → [闭运算] → [坡度限幅] → AA → EMA → depth_tex。

        编排胶水（块边界）：空间链 = depth_post.DepthPost.spatial_chain；
        时域 EMA = depth_mc.MC.ema_pass（D2S_MC 开）或 DepthPost.ema_pass_legacy。
        cut：残差切阀命中（调用方 main 在推理回读 numpy 上算，depth_norm.
        cut_hit——GPU 1x1 读回版因同步代价 7ms/帧已退役）→ 本帧直通/清环。
        """
        self.gpu_post_init(self._gpu_res)
        post = self._post_get()
        cur = post.spatial_chain(raw_tex, lo, hi, aa_texels, grow, slope_target,
                                 parallax_px, iters, close_px)
        # 时域级（D2S_TEMPORAL）：MC 实验路径优先（自带 trust 切）
        _pipe = getattr(self, "_mc_pipe", None)
        _mc_active = (getattr(self, "_mc_on", False) and _pipe is not None
                      and _pipe.ready(post.res))
        _first = post.take_ema_first()
        if _mc_active:
            _pipe.ema_pass(cur, 1.0 if _first else alpha,
                           post.depth_fbo, post.res)
        elif post.temporal_median:
            post.median_pass(cur, reset=(_first or cut))
        else:
            post.ema_pass_legacy(cur, 1.0 if (_first or cut) else alpha)
        self.ctx.copy_framebuffer(post.ema_fbo, post.depth_fbo)
        self.ctx.screen.use()

    def read_depth_median(self):
        """时域后深度的收敛面锚点（2026-11-20 改源两次）。

        现为 depth_norm.near_peak_median：双峰内容取近景峰中位（主体贴
        屏幕平面），单峰自动回退全局 P50；D2S_CONV_PEAK=0 回纯 P50。
        失败返回 None（调用方保持旧值，收敛面永不因读回故障跳变）。
        """
        try:
            from depth_norm import near_peak_median
            buf = self._post.depth_fbo.read(components=1, dtype='f4')
            a = np.frombuffer(buf, np.float32)
            r = int(round(len(a) ** 0.5))
            return near_peak_median(a.reshape(r, r))
        except Exception:
            return None

    def gpu_preprocess(self):
        """色纹理 → 224 预处理纹理（降采样 + ImageNet 归一化，CHW 平铺）。"""
        if getattr(self, "color_tex", None) is None:
            return   # 首个颜色环帧尚未绑定（启动竞态）：本帧跳过推理，
                     # 渲染走 2D；此前在此处 NoneTex.use 崩掉整个深度链
        self.color_tex.use(0)
        self._prep_prog['tex'].value = 0
        self._prep_fbo.use()
        self.ctx.viewport = (0, 0, self._prep_tex.size[0], self._prep_tex.size[1])
        self._prep_vao.render(moderngl.TRIANGLE_STRIP)
        # 关键：切回屏幕帧缓冲——渲染循环假定后续 draw 落在默认帧缓冲，
        # 不切的话 DIBR 全画进这个 672x224 的 FBO，swap 出去的是黑屏
        # （用户实测"按快捷键变黑窗"的根因：静态帧亮 ~50ms 后活帧接管即黑）
        self.ctx.screen.use()

    # -------------------------------------------------- 色路零拷贝（M2）
    def gpu_bind_color(self, dx, gen, ring_idx, ring_tex_ptrs, w, h):
        """把采集环纹理换装为 color_tex（锁在 swap 后由 gpu_unlock 解）。

        代数变化（换靶/重开会话）时重建 GL 占位纹理并重新注册；本方法
        承担 update_frame 的簿记语义（u_resolution/首帧块）。
        """
        if getattr(self, "_dx_gen", None) != gen:
            # 换代重挂：GL 纹理对象永生复用（只重注册，绝不删除——
            # 删除已注册纹理会让 interop 记账坏死，下次 glBind 报 1282
            # 并炸掉渲染线程）。GL 纹理自身尺寸无所谓：采样全用归一化
            # uv，真实尺寸由 u_resolution uniform 承担。
            ptrs = list(ring_tex_ptrs)
            if getattr(self, "_dx_ring_gl", None) is None:
                self._dx_ring_gl = []
                for _ in ptrs:
                    t = self.ctx.texture((8, 8), 4, dtype='f1')
                    t.repeat_x = False
                    t.repeat_y = False
                    t._glo_interop = True   # 互操作纹理：release 免疫
                    self._dx_ring_gl.append(t)
            while len(self._dx_ring_gl) < len(ptrs):
                t = self.ctx.texture((8, 8), 4, dtype='f1')
                t.repeat_x = False
                t.repeat_y = False
                t._glo_interop = True
                self._dx_ring_gl.append(t)
            for t, ptr in zip(self._dx_ring_gl, ptrs):
                dx.reregister(t, ptr)
            self._dx_gen = gen
        tex = self._dx_ring_gl[ring_idx]
        dx.lock(tex)
        # 镜传修复（闪烁定案 2026-11-15）：重呈现/重显路径在锁外采样环纹理
        # （上一拍 swap 前已解锁；NV 规约锁外采样未定义，驱动回黑）——
        # [zc2] 实测 from=represent 100% drawread=0 / from=new 100% 亮。
        # 改为：锁内把环纹理 GPU 拷进普通镜像纹理后立即解锁，color_tex
        # 指向镜像——所有 draw（新帧/重呈现/黑洗重显）与预处理永远采样
        # 普通纹理，锁只存在 ~0.1ms、不跨 SwapBuffers。字节原样拷贝
        # （RGBA8→RGBA8），shader 侧 .bgra 通道交换不受影响。
        # 代价：每新帧一次 w×h GPU 拷贝 ≈0.1~0.2ms。D2S_ZC_MIRROR=0 对照。
        try:
            import os as _os_m
            if _os_m.environ.get("D2S_ZC_MIRROR", "1").strip().lower() not in ("0", "false", "off", "no"):
                if getattr(self, "_zc_mirror", None) is None or self._zc_mirror.size != (w, h):
                    if getattr(self, "_zc_mirror", None) is not None:
                        self._zc_mirror.release()
                    self._zc_mirror = self.ctx.texture((w, h), 4, dtype='f1')
                    self._zc_mirror.repeat_x = False
                    self._zc_mirror.repeat_y = False
                    self._zc_mirror.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    self._zc_mirror_fbo = self.ctx.framebuffer([self._zc_mirror])
                    if getattr(self, "_zc_mirror_prog", None) is None:
                        self._zc_mirror_prog = self.ctx.program(
                            vertex_shader="#version 330\nin vec2 p; out vec2 uv;\n"
                                          "void main(){uv = p*0.5+0.5; gl_Position = vec4(p,0,1);}",
                            fragment_shader="#version 330\nin vec2 uv; out vec4 c;\n"
                                            "uniform sampler2D t;\n"
                                            "void main(){c = texture(t, uv);}")
                        self._zc_mirror_vao = self.ctx.vertex_array(
                            self._zc_mirror_prog,
                            [(self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1],
                                                      np.float32)), "2f", "p")])
                # 绘制式拷贝（glCopyImageSubData 对 interop 挂名纹理报 GLError，
                # 驱动按占位元数据校验尺寸；全屏 quad 渲染按真实 D3D 尺寸走，
                # 与 _ctex_fbo 同款做法）。RGBA 字节原样透传（含 alpha）。
                tex.use(0)
                self._zc_mirror_prog['t'].value = 0
                self._zc_mirror_fbo.use()
                self._zc_mirror_fbo.viewport = (0, 0, w, h)
                self._zc_mirror_vao.render(moderngl.TRIANGLE_STRIP)
                self.ctx.screen.use()
                dx.unlock(tex)            # 锁只保住这一次绘制拷贝，立即归还采集
                tex = self._zc_mirror     # 后续所有采样走普通镜像
        except Exception as _e:
            print(f"[zc-mirror] 镜传失败，本帧回退直采: {type(_e).__name__}: {_e}", flush=True)
        if self._texture_size != (w, h) or self.color_tex is not tex:
            self.color_tex = tex
            for _prog in (self.prog, self.anaglyph_prog,
                          self.interleaved_prog, self.vertical_interleaved_prog):
                _prog['tex_color'].value = 0
                _prog['tex_depth'].value = 1
                _prog['u_resolution'].value = (w, h)
                _prog['u_parallax_curve'].value = self._parallax_curve
                _prog['u_edge_falloff'].value = self._edge_falloff
            for _prog in (self.prog, self.interleaved_prog,
                          self.vertical_interleaved_prog):
                _prog['u_fill'].value = 1.0 if self._disocclusion_fill else 0.0
            self._texture_size = (w, h)
        # 帧语义旗标：**不清零**。颜色零拷贝下 bind 每帧都会发生，而深度块
        # 每帧不一定跑；清零会让画面在 3D/2D 之间来回切（静态画面也闪）。
        # 保持上一次深度，让新帧沿用上一帧的深度图，直到新深度到达。
        # 首帧仍为 False（_frame_has_depth 初始值），由首帧深度块置 True。
        if not self._has_real_frame:
            self._has_real_frame = True
            if not SAME_SCREEN:
                self.position_on_monitor(self.monitor_index)
            self._hide_from_taskbar()
            if self.use_3d:
                self.apply_3d_settings()
                self._reassert_overlay_style()

    def gpu_color_mean(self):
        """互操作色纹理内容探针：32x32 下采样回读均值 RGB。

        区分"纹理内容黑"（采集/互操作问题）与"渲染黑"（shader/管线问题）。
        首帧前 / CPU 采集早期 color_tex 尚未就绪时返回零值（此前在
        NoneTex 上调 .use 抛 AttributeError，被 [fps] 吞掉致探针失明）。
        """
        if getattr(self, "color_tex", None) is None:
            return (0, 0, 0, 255), 0.0, None
        if getattr(self, "_probe_fbo", None) is None:
            self._probe_tex = self.ctx.texture((32, 32), 4, dtype='f1')
            self._probe_prog = self.ctx.program(
                vertex_shader="#version 330\nin vec2 p; out vec2 uv;\n"
                              "void main(){uv=p*0.5+0.5; gl_Position=vec4(p,0,1);}",
                fragment_shader="#version 330\nin vec2 uv; out vec4 c;\n"
                                "uniform sampler2D t;\nvoid main(){c=texture(t,uv);}")
            self._probe_fbo = self.ctx.framebuffer([self._probe_tex])
            self._probe_vao = self.ctx.vertex_array(
                self._probe_prog,
                [(self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1],
                                           np.float32)), "2f", "p")])
        import moderngl as _mg
        self.color_tex.use(0)
        self._probe_prog['t'].value = 0
        prev_vp = self.ctx.viewport
        self._probe_fbo.use()
        self.ctx.viewport = (0, 0, 32, 32)
        self._probe_vao.render(_mg.TRIANGLE_STRIP)
        d = np.frombuffer(self._probe_fbo.read(components=4), np.uint8)
        d = d.reshape(32, 32, 4).mean(axis=(0, 1))
        # swizzle 活体状态回读（诊断 R<->B 交换：设了还交换=驱动层问题，
        # 没设=注册路径没走到）
        swz = None
        try:
            from OpenGL.GL import (glGetTexParameteriv, glBindTexture,
                                   GL_TEXTURE_SWIZZLE_RGBA)
            glBindTexture(0x0DE1, self.color_tex.glo)
            buf = (ctypes.c_int * 4)()
            glGetTexParameteriv(0x0DE1, GL_TEXTURE_SWIZZLE_RGBA, buf)
            swz = list(buf)
        except Exception:
            pass
        # 顺带读皮肤后缓冲（上帧 swap 内容）：区分 DIBR 输出黑 vs 合成黑。
        # 只读左下角 64x64：整屏 read(1920x1080) 是同步 glReadPixels，每 5 秒
        # 诊断打点都会把 GPU 等一遍、掉一帧（用户看到"屏幕隔几秒抖一下"）。
        # 均值/是否黑屏的判据用一小块完全够用，拷贝量 2MB → 4KB。
        try:
            self.ctx.screen.use()
            self.ctx.viewport = prev_vp
            import ctypes as _ct3
            _gl3 = _ct3.windll.opengl32
            _pw = _ph = 64
            _pbuf = (ctypes.c_ubyte * (_pw * _ph))()
            _gl3.glReadPixels(0, 0, _pw, _ph, 0x1903, 0x1401, _pbuf)  # RED, UNSIGNED_BYTE
            fb = float(np.frombuffer(bytes(_pbuf), np.uint8).mean())
        except Exception:
            fb = -1.0
        return d, float(fb), swz

    def gpu_set_static(self, rgb):
        """静态帧（PrintWindow）专用：普通 3 通道纹理上传 + 换装 color_tex。

        互操作环纹理是 4 通道（BGRA8），RGB numpy 直写会尺寸不匹配；
        且新靶会话未建、环还属于旧代 —— 静态帧与环完全解耦。
        """
        h, w = rgb.shape[:2]
        if getattr(self, "_static_tex", None) is None \
                or self._static_tex_size != (w, h):
            if getattr(self, "_static_tex", None) is not None:
                self._static_tex.release()
            t = self.ctx.texture((w, h), 3, dtype='f1')
            t.repeat_x = False
            t.repeat_y = False
            self._static_tex = t
            self._static_tex_size = (w, h)
        self._static_tex.write(rgb)
        if self._texture_size != (w, h) or self.color_tex is not self._static_tex:
            self.color_tex = self._static_tex
            for _prog in (self.prog, self.anaglyph_prog,
                          self.interleaved_prog, self.vertical_interleaved_prog):
                _prog['tex_color'].value = 0
                _prog['tex_depth'].value = 1
                _prog['u_resolution'].value = (w, h)
                _prog['u_parallax_curve'].value = self._parallax_curve
                _prog['u_edge_falloff'].value = self._edge_falloff
            for _prog in (self.prog, self.interleaved_prog,
                          self.vertical_interleaved_prog):
                _prog['u_fill'].value = 1.0 if self._disocclusion_fill else 0.0
            self._texture_size = (w, h)
        # _frame_has_depth **不清零**：颜色零拷贝下 bind 每帧发生、深度块不一定
        # 每帧跑；清零会让该帧退回 2D（用户反馈"3D 不是逐帧都有深度"）。
        # 保持上一次深度，新帧沿用上一帧深度图，直到新深度到达。
        if not self._has_real_frame:
            self._has_real_frame = True
            if not SAME_SCREEN:
                self.position_on_monitor(self.monitor_index)
            self._hide_from_taskbar()
            if self.use_3d:
                self.apply_3d_settings()
                self._reassert_overlay_style()

    def gpu_unlock(self, ring_idx):
        """swap 后解环纹理锁（采集线程才能再写该槽）。"""
        dx = getattr(self, "_dx", None)
        ring = getattr(self, "_dx_ring_gl", None)
        if dx is not None and ring:
            try:
                dx.unlock(ring[ring_idx])
            except Exception:
                pass

    def update_frame(self, rgb, depth, current_fps=None, current_latency=None):
        """Update frame textures（S3：numpy 上传，glTexSubImage2D 走驱动 DMA）。

        rgb: HWC uint8 numpy（工作分辨率）；depth: HW float32 numpy，模型分辨率
        （224）或 None（2D 镜像）。深度纹理保持模型分辨率，shader 采样时双线性
        放大 —— CPU 不再做 224->1080 的上采样。
        """
        if current_fps is not None:
            self.actual_fps = current_fps
        # 每来一帧新画面就递增：SBS 每帧渲染两遍，光标靠它判断"这一帧画过了"
        self._frame_id = getattr(self, "_frame_id", 0) + 1
        if current_latency is not None:
            self.total_latency = current_latency * 1000

        # 2D mirror frames carry no depth: render() must zero the parallax
        # for them, otherwise the stale depth texture keeps warping both
        # eyes and the "2D" mirror stays pseudo-3D.
        self._frame_has_depth = depth is not None

        rgb_np = np.asarray(rgb)
        h, w = rgb_np.shape[0], rgb_np.shape[1]
        depth_np = None if depth is None else np.asarray(depth, dtype=np.float32)

        if self.stream_mode is None:
            self._update_overlay_texture()

        # 信纹理对象不信缓存：GPU 直通把 color_tex 换装成互操作环纹理
        #（如 8x8 占位），GPU 禁用回落 CPU 时缓存仍说尺寸一致 -> 写爆。
        # 用 color_tex.size 实测兜底（环纹理/静态纹理/常规纹理通吃）。
        if self._texture_size != (w, h) or \
                (self.color_tex is not None and self.color_tex.size != (w, h)):
            if self.color_tex and getattr(self.color_tex, "_glo_interop", None) is None:
                self.color_tex.release()   # 互操作纹理不可 release（记账坏死）
            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            # The parallax can push sampling outside the image near the left and
            # right edges. moderngl textures default to REPEAT, which would wrap
            # the opposite edge into view; clamp so the edge pixel is extended.
            self.color_tex.repeat_x = False
            self.color_tex.repeat_y = False
            # u_resolution is the COLOR texture size: pixel_size is a texel step
            # for color-domain work (inpaint sweeps, blur). Depth-domain taps
            # (3-tap smoothing, gradient tests) step by u_depth_resolution —
            # the depth texture keeps the model resolution since S3.
            for _prog in (self.prog, self.anaglyph_prog,
                          self.interleaved_prog, self.vertical_interleaved_prog):
                _prog['tex_color'].value = 0
                _prog['tex_depth'].value = 1
                _prog['u_resolution'].value = (w, h)
                _prog['u_parallax_curve'].value = self._parallax_curve
                _prog['u_edge_falloff'].value = self._edge_falloff
            # Spatial disocclusion fill is optional: on a soft depth map it
            # paints flat background blocks INSIDE silhouettes (measured), so
            # it defaults off. The anaglyph shader has no u_fill.
            for _prog in (self.prog, self.interleaved_prog,
                          self.vertical_interleaved_prog):
                _prog['u_fill'].value = 1.0 if self._disocclusion_fill else 0.0
            self._texture_size = (w, h)

        # Depth texture at the model resolution (rebuilt only if it changes).
        if depth_np is not None:
            dh, dw = depth_np.shape
            if self._depth_tex_size != (dw, dh):
                if self.depth_tex:
                    self.depth_tex.release()
                self.depth_tex = self.ctx.texture((dw, dh), 1, dtype='f4')
                # CLAMP_TO_EDGE + LINEAR filtering: bilinear upscale for free,
                # edge texels extended when parallax pulls samples outside.
                self.depth_tex.repeat_x = False
                self.depth_tex.repeat_y = False
                self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._depth_tex_size = (dw, dh)
                for _prog in (self.prog, self.anaglyph_prog,
                              self.interleaved_prog, self.vertical_interleaved_prog):
                    if 'u_depth_resolution' in _prog:
                        _prog['u_depth_resolution'].value = (dw, dh)

        if self.stream_mode is not None and self.display_mode != "Depth Map":
            rgb_np = self._add_overlay(rgb_np)

        # contiguous ndarray 经 buffer 协议直达 glTexSubImage2D，免 tobytes() 拷贝
        rgb_buf = rgb_np.data if rgb_np.flags.c_contiguous else rgb_np.tobytes()

        if self.display_mode == "Depth Map" and not self.show_original_in_depth_mode:
            if depth_np is not None:
                self.depth_tex.write(depth_np.tobytes())
            if self._last_display_mode != "Depth Map" or self._last_show_original != self.show_original_in_depth_mode:
                self.color_tex.write(rgb_buf)
        elif depth_np is None:
            # 2D mirror: colour texture only.
            self.color_tex.write(rgb_buf)
        else:
            self.color_tex.write(rgb_buf)
            self.depth_tex.write(depth_np.tobytes())

        # Cache current state
        self._last_display_mode = self.display_mode
        self._last_show_original = self.show_original_in_depth_mode

        # Show window after first frame.
        if not self._has_real_frame:
            self._has_real_frame = True
            # 同屏自投射：显示与否由 main 渲染循环按 2D/3D 模式管理
            # （2D 原生隐藏），这里不再无条件闪窗；定位等准备照常执行。
            if not SAME_SCREEN:
                glfw.show_window(self.window)
            if not SAME_SCREEN:
                self.position_on_monitor(self.monitor_index)
            # Window just became visible and may get repositioned/fullscreened:
            # re-assert the no-taskbar style after those transitions.
            self._hide_from_taskbar()
            if self.use_3d:
                if not self.specify_display and not SAME_SCREEN:
                    self.toggle_fullscreen()
                self.apply_3d_settings()
                self._reassert_overlay_style()
            if self.specify_display:
                self.toggle_fullscreen()

    def _compute_render_size(self, max_w, max_h, src_w, src_h):
        """Calculate render size maintaining aspect ratio"""
        if src_w == 0 or src_h == 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale)))
        )
        
    def _init_pbos(self, width, height):
        """Initialize two PBOs for asynchronous pixel transfers."""
        if self._pbo_initialized:
            return
        self._pbo_ids = glGenBuffers(2)
        buffer_size = width * height * 3  # RGB8
        for pbo in self._pbo_ids:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_index = 0
        self._pbo_initialized = True
        # print(f"[StereoWindow] Initialized {len(self._pbo_ids)} PBOs ({buffer_size/1e6:.2f} MB each)")

    def capture_glfw_image(self):
        """Asynchronous GPU readback using double-buffered PBOs."""
        width, height = glfw.get_framebuffer_size(self.window)
        if not self._pbo_initialized:
            self._init_pbos(width, height)
        next_index = (self._pbo_index + 1) % 2
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        # Bind current PBO and start async read
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[self._pbo_index])
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        # Bind previous PBO to map and read CPU data (async from previous frame)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[next_index])
        ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        image = None
        if ptr:
            try:
                # Copy from GPU memory into numpy array
                buf = (GLubyte * (width * height * 3)).from_address(int(ptr))
                image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                image = np.flipud(image.copy())  # Flip vertically; .copy() detaches from mapped memory
            finally:
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)

        # Unbind PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Swap PBO indices for next frame
        self._pbo_index = next_index

        # Note: the very first call will return None (no previous frame ready)
        return image

    def _frame_dump(self):
        """把当前帧缓冲读回并保存（GL 内部读取，不受 WDA 截图限制影响）。
        由环境变量 D2S_FRAME_DUMP=<dir> 启用；D2S_FRAME_DUMP_HZ 控制频率。"""
        import os as _os
        d = _os.environ.get("D2S_FRAME_DUMP", "")
        if not d:
            return
        if not getattr(self, "_fdump_seen", False):
            self._fdump_seen = True
            print(f"[fdump] 已启用 -> {d}", flush=True)
        import time as _t
        hz = float(_os.environ.get("D2S_FRAME_DUMP_HZ", "1") or 1)
        now = _t.time()
        if now - getattr(self, "_fdump_t", 0.0) < (1.0 / max(0.1, hz)):
            return
        self._fdump_t = now
        try:
            w, h = glfw.get_framebuffer_size(self.window)
            raw = self.ctx.screen.read(components=3)
            import numpy as _np
            img = _np.frombuffer(raw, _np.uint8).reshape(h, w, 3)[::-1]
            _os.makedirs(d, exist_ok=True)
            import cv2 as _cv
            _cv.imwrite(_os.path.join(d, f"frame_{int(now*1000)}.png"), img)
        except Exception as e:
            print(f"[fdump] {e}", flush=True)

    def _zc_draw_probe(self, tag):
        """D2S_ZC_PROBE2=1：draw 前一瞬间的真实 GL 状态三件套（限 24 帧）。

        与外层 [blk] 探针（render() 之前采样）的区别：这里在 color_tex.use()
        之后、vao.render 之前，看到的就是 draw 时刻：
          1. unit0 的 GL_TEXTURE_BINDING_2D（真实 GL，不信 python 簿记）
          2. 对"当前绑定纹理"glGetTexImage 直读均值（纹理内容）
          3. 用当前绑定渲染 32×32 FBO 的均值（draw 能画出什么；viewport 修正版）
        判定：direct 亮+drawread 黑 → 采样/单元问题；direct 黑 → 纹理在
        bind 后失效（驱动）→ 双缓冲环；两者都亮而屏幕黑 → present/更晚环节。
        """
        import os as _osp2
        if not _osp2.environ.get("D2S_ZC_PROBE2"):
            return
        try:
            n = getattr(self, "_zc2_n", 0)
            if n >= 24:
                return
            self._zc2_n = n + 1
            import ctypes as _ct2
            _gi = _ct2.c_int(0)
            _act = 0
            glGetIntegerv(GL_ACTIVE_TEXTURE, _gi); _act = _gi.value
            glActiveTexture(GL_TEXTURE0)
            glGetIntegerv(GL_TEXTURE_BINDING_2D, _gi); _bnd = _gi.value or None
            glActiveTexture(_act)
            # 1) 直读当前绑定的纹理（不管它是不是 python 以为的那张）。
            # 尺寸用 app 自己的 _texture_size（glGetTexLevelParameteriv 对
            # interop 挂名纹理返回占位元数据，此前按它分配缓冲会溢出）。
            _direct = None
            _tsz = None
            if _bnd:
                _w2, _h2 = getattr(self, "_texture_size", (None, None)) or (None, None)
                if _w2 and _h2 and _w2 * _h2 <= 8_000_000:
                    _tsz = (_w2, _h2)
                    while glGetError() != GL_NO_ERROR:
                        pass
                    _buf = _ct2.create_string_buffer(_w2 * _h2 * 3)
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, _buf)
                    _direct = ("glerr_read" if glGetError() != GL_NO_ERROR
                               else float(np.frombuffer(_buf.raw, np.uint8)[::499].mean()))
            # 2) 用当前 unit0 绑定渲染 32×32 FBO（修正 viewport 的 FBO 探针）
            _drawmean = None
            if _bnd:
                if getattr(self, "_zc2_fbo", None) is None:
                    _t = self.ctx.texture((32, 32), 3)
                    self._zc2_fbo = self.ctx.framebuffer([_t])
                _old_vp = self.ctx.viewport
                self._zc2_fbo.use()
                self._zc2_fbo.viewport = (0, 0, 32, 32)
                self.plain_vao.render(moderngl.TRIANGLE_STRIP)
                _raw = self._zc2_fbo.read(components=3, alignment=1)
                _drawmean = float(np.frombuffer(_raw, np.uint8).mean())
                self.ctx.screen.use()
                self.ctx.viewport = _old_vp
            _samp = "?"
            try:
                if self.plain_prog is not None and 'tex_color' in self.plain_prog:
                    _samp = int(self.plain_prog['tex_color'].value)
            except Exception:
                pass
            print(f"[zc2] #{n} {tag} from={getattr(self, '_zc_from', '?')} "
                  f"active_unit={_act - GL_TEXTURE0} "
                  f"unit0_bound={_bnd} py_glo={self.color_tex.glo if self.color_tex else '?'} "
                  f"samp_unit={_samp} tex_sz={_tsz} direct={_direct} drawread={_drawmean} "
                  f"gen={getattr(self, '_dx_gen', '?')} mode={getattr(self, 'display_mode', '?')}",
                  flush=True)
        except Exception as _e:
            print(f"[zc2] err {type(_e).__name__}: {_e}", flush=True)

    def render(self):
        # 诊断（D2S_DEPTH_TRACE=1）：记录 has_depth 的每次翻转，用来定位
        # "暂停/播放闪一帧"是否为该帧被渲染成 2D。
        try:
            import os as _osd3
            if _osd3.environ.get("D2S_DEPTH_TRACE") == "1":
                _cur = bool(getattr(self, "_frame_has_depth", True))
                if getattr(self, "_dtrace_last", None) != _cur:
                    self._dtrace_last = _cur
                    print(f"[trace] has_depth={_cur} t={time.perf_counter():.3f} "
                          f"gen={getattr(self, '_dx_gen', None)}", flush=True)
        except Exception:
            pass
        """Ultra-optimized rendering with minimal GL calls"""
        if not self.color_tex or not self.depth_tex:
            return

        # Get window dimensions once
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        tex_w, tex_h = self._texture_size
        # 逆一致性搜索的 uniform 必须在所有分支前无条件设置：
        # 之前它被缩进进了 if self.fill_16_9: 里，该开关为 false 时
        # 永远不赋值 → u_ic_search 保持 0 → 搜索根本不生效。
        try:
            import os as _os_ic0
            if getattr(self, "_u_ic", None) is not None:
                self._u_ic.value = (0.0 if _os_ic0.environ.get("D2S_IC_SEARCH", "0") == "0" else 1.0)
                if getattr(self, "_u_dstep", None) is not None:
                    self._u_dstep.value = 1.0 / max(1.0, float(getattr(self, "_gpu_res", 0) or 224))
                if getattr(self, "_u_icdbg", None) is not None:
                    self._u_icdbg.value = (1.0 if _os_ic0.environ.get("D2S_IC_DEBUG", "0") == "1" else 0.0)
        except Exception:
            pass

        # Inverse row-weave pre-compensation: always on for 2D-mirror frames
        # in TAB layouts (the monitor is a fixed row weaver). Reset the
        # uniform each render so 3D / SBS / Depth-Map paths never inherit it.
        self._u_row_split.value = 0.0
        self._row_split_active = (
            not self._frame_has_depth
            and self.display_mode in ("Half-TAB", "Full-TAB")
        )
        
        # 2D mirror fast path: one texture, one fullscreen quad, no warp.
        if self.display_mode == "Passthrough":
            return self._render_passthrough()
        if self.display_mode == "Depth Map":
            return self._render_depth_map(win_w, win_h, tex_w, tex_h)
        if self.display_mode in ["Anaglyph", "Interleaved", "Interleaved-V"]:
            return self._render_singlepass(win_w, win_h, tex_w, tex_h)
        return self._render_sbs_tab(win_w, win_h, tex_w, tex_h)


    def _draw_eye(self, vp, sign, parity=None):
        """SBS/TAB 眼绘制（render 各模式 ~20 处重复 uniform 块的收敛）。

        sign=-1 左眼 / +1 右眼；parity 给 TAB 行编织（_row_split_active 时
        写 u_row_split/u_row_parity）。SBS 模式 parity=None 不碰行编织。
        """
        self.ctx.viewport = vp
        self._u_eye_offset.value = sign * self.ipd_uv / 2.0
        if parity is not None and self._row_split_active:
            self._u_row_split.value = 1.0
            self._u_row_parity.value = float(parity)
        self._u_feather_enabled.value = self.feather_enabled
        self._u_feather_width.value = self.feather_width
        self._u_viewport.value = vp
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    # ------------------------------------------------------------------
    # 显示格式块（P3 解耦）：每模式一个方法，body 自原 render() 逐行搬运
    # （仅眼绘制 uniform 块收敛进 _draw_eye、视差换算接线 dibr）。
    # ------------------------------------------------------------------

    def _render_passthrough(self):
        if not self.color_tex:
            return
        win_w2, win_h2 = glfw.get_framebuffer_size(self.window)
        tex_w2, tex_h2 = self._texture_size
        self.ctx.clear(0.0, 0.0, 0.0)
        sc = min(win_w2 / tex_w2, win_h2 / tex_h2)
        vw, vh = int(tex_w2 * sc), int(tex_h2 * sc)
        self.ctx.viewport = ((win_w2 - vw) // 2, (win_h2 - vh) // 2, vw, vh)
        self.color_tex.use(0)
        self._zc_draw_probe("passthrough")
        self.plain_vao.render(moderngl.TRIANGLE_STRIP)
        self._render_overlay()
        glfw.poll_events()
        return

    # Early exit for depth map mode - fully optimized path

    def _render_depth_map(self, win_w, win_h, tex_w, tex_h):
        # Fast path - minimal operations for depth map mode
        
        # Skip aspect ratio updates for depth map mode if not needed
        if self.fix_aspect:
            glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        
        # Clear screen with optimized clear color
        self.ctx.clear(0.0, 0.0, 0.0)  # Black background for better contrast
        
        # Use cached viewport calculation
        viewport = self._calculate_depth_map_viewport(win_w, win_h, tex_w, tex_h)
        self.ctx.viewport = viewport
        
        # Toggle between depth map and original RGB with minimal state changes
        if self.show_original_in_depth_mode:
            # Fast RGB rendering path
            self.color_tex.use(location=0)
            
            # Update uniforms only if changed
            if self._last_eye_offset_set != 0.0:
                self._u_eye_offset.value = 0.0
                self._last_eye_offset_set = 0.0

            if self._last_depth_strength_set != 0.0:
                self._u_depth_strength.value = 0.0
                self._last_depth_strength_set = 0.0
            
            # Direct render call without extra state changes
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
        else:
            # Ultra-fast depth map rendering
            self.depth_tex.use(location=0)
            
            # Direct render call - no uniform updates needed for depth shader
            self.depth_vao.render(moderngl.TRIANGLE_STRIP)
        
        # Skip swap buffers if frame hasn't changed (for headless/streaming modes)
        self._render_overlay()
        if self.stream_mode is None:
            glfw.poll_events()
        return

    # Composite display modes (Mono, Anaglyph, Interleaved, Interleaved-V)

    def _render_singlepass(self, win_w, win_h, tex_w, tex_h):
        if self.fix_aspect:
            glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)

        self.ctx.clear(0.0, 0.0, 0.0)

        if self.fill_16_9:
            render_w, render_h = self._compute_render_size(win_w, win_h, tex_w, tex_h)
            center_x, center_y = win_w / 2.0, win_h / 2.0
            viewport = (int(center_x - render_w / 2), int(center_y - render_h / 2), render_w, render_h)
        else:
            target_aspect = tex_h / tex_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9.0 / 16.0
            if window_aspect <= target_aspect:
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                view_w = win_w
                view_h = int(view_w * target_aspect)
            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2
            viewport = (offset_x, offset_y, view_w, view_h)

        self.ctx.viewport = viewport
        half_ipd = self.ipd_uv / 2.0

        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)
        self._zc_draw_probe("main")
        # Parallax budget is authored in SCREEN PIXELS per eye (full
        # deflection at the ends of the depth range when plane = 0.5);
        # convert to the shader's u_eye_offset units (half the IPD in UV).
        # 视差 = 视差px × 逐像素深度（shader 内）；px 即立体强度，
        # 不再有 /2.5 的 gain 归一化层（2026-11-09 用户定名）。
        depth_val = 1.0 if self._frame_has_depth else 0.0
        par_u = dibr.parallax_uv(self.parallax_px, self.ipd_uv, win_w)

        if self.display_mode == "Anaglyph":
            self._u_ana_eye_offset.value = half_ipd
            self._u_ana_depth_strength.value = depth_val
            self._u_ana_parallax.value = par_u
            self._u_ana_feather_enabled.value = self.feather_enabled
            self._u_ana_feather_width.value = self.feather_width
            self._u_ana_viewport.value = viewport
            self.anaglyph_vao.render(moderngl.TRIANGLE_STRIP)
        elif self.display_mode == "Interleaved":
            try:
                import os as _os_u
                if (_os_u.environ.get("D2S_UNIFORM_DEBUG") == "1"
                        and not getattr(self, "_uni2", False)):
                    self._uni2 = True
                    print(f"[uniforms] win={glfw.get_window_size(self.window)} "
                          f"viewport={viewport} tex=({tex_w},{tex_h}) "
                          f"half_ipd={half_ipd} par_u={par_u:.6f} "
                          f"parallax_px={self.parallax_px} "
                          f"c={self.convergence} "
                          f"shift_px_at_d0.26={(-2*(0.26-self.convergence)*self.parallax_px):.2f}",
                          flush=True)
            except Exception as _e:
                print(f"[uniforms] err {_e}", flush=True)
            self._u_int_eye_offset.value = half_ipd
            self._u_int_depth_strength.value = depth_val
            self._u_int_parallax.value = par_u
            try:
                if self._u_int_ic is not None:
                    import os as _os_ic
                    self._u_int_ic.value = (0.0 if _os_ic.environ.get("D2S_IC_SEARCH", "0") == "0" else 1.0)
                    _dres = float(getattr(self, "_gpu_res", None) or self._depth_tex_size[0] or 224)
                    self._u_int_dstep.value = 1.0 / max(1.0, _dres)
            except Exception:
                pass
            self._u_int_feather_enabled.value = self.feather_enabled
            self._u_int_feather_width.value = self.feather_width
            self._u_int_viewport.value = viewport
            # 每帧带行奇偶校正：渲染窗口 y 奇偶决定的相位偏移由
            # set_row_parity 维护（此前此处每帧用原始值覆盖，校正
            # 从未活到渲染 —— 用户实测眼仍翻转的根因）
            self._u_int_phase.value = float(
                (int(self.interleave_phase) + getattr(self, '_row_parity', 0)) & 1)
            self.interleaved_vao.render(moderngl.TRIANGLE_STRIP)
            # ---- 调试：同步 glReadPixels 抓渲染帧 + 深度/彩色纹理 ----
            try:
                import os as _os_rd, time as _tm_rd
                import cv2 as cv2
                _rd = _os_rd.environ.get("D2S_RENDER_DUMP", "")
                if _rd and _tm_rd.perf_counter() - getattr(self, "_rd_t", 0.0) > 1.0:
                    self._rd_t = _tm_rd.perf_counter()
                    _os_rd.makedirs(_rd, exist_ok=True)
                    _ts = int(self._rd_t * 1000)
                    _fw, _fh = glfw.get_framebuffer_size(self.window)
                    _buf = ctypes.create_string_buffer(_fw * _fh * 3)
                    glPixelStorei(GL_PACK_ALIGNMENT, 1)
                    glReadPixels(0, 0, _fw, _fh, GL_RGB, GL_UNSIGNED_BYTE, _buf)
                    _arr = np.frombuffer(_buf.raw, np.uint8).reshape(_fh, _fw, 3)[::-1]
                    cv2.imwrite(_os_rd.path.join(_rd, f"fb_{_ts}.png"),
                                np.ascontiguousarray(_arr[:, :, ::-1]))
                    try:
                        _psz = self._prep_tex.size
                        _parr = np.frombuffer(self._prep_tex.read(), np.float32)
                        # 原样保存扁平数据：布局是每行 [R块|G块|B块]，
                        # 解码时须按行切三段，不能直接 reshape(3,side,side)
                        np.save(_os_rd.path.join(_rd, f"prep_{_ts}.npy"),
                                _parr.reshape(_psz[1], _psz[0]))
                    except Exception as _e0:
                        print(f"[rdump] prep 失败 {_e0}", flush=True)
                    try:
                        _dsz = self.depth_tex.size
                        _darr = np.frombuffer(self.depth_tex.read(), np.float32).reshape(_dsz[1], _dsz[0])
                        np.save(_os_rd.path.join(_rd, f"dtex_{_ts}.npy"), _darr)
                    except Exception as _e1:
                        print(f"[rdump] depth tex 失败 {_e1}", flush=True)
                    try:
                        # GPU 直通模式下彩色来自 CUDA 互操作环纹理，直接 read()
                        # 拿不到（实测 8x8 空图）。这里先把它渲染到普通 FBO 再读，
                        # 得到的就是着色器真正采样的那张全分辨率彩图。
                        if getattr(self, "_ctex_fbo", None) is None:
                            self._ctex_tex = self.ctx.texture((960, 540), 3, dtype='f1')
                            self._ctex_fbo = self.ctx.framebuffer([self._ctex_tex])
                            self._ctex_prog = self.ctx.program(
                                vertex_shader="""
                                    #version 330
                                    in vec2 in_pos;
                                    out vec2 v_uv;
                                    void main(){ v_uv = in_pos * 0.5 + 0.5;
                                                 gl_Position = vec4(in_pos, 0.0, 1.0); }
                                """,
                                fragment_shader="""
                                    #version 330
                                    in vec2 v_uv;
                                    out vec4 out_c;
                                    uniform sampler2D tex_c;
                                    void main(){ out_c = vec4(texture(tex_c, v_uv).rgb, 1.0); }
                                """)
                            self._ctex_vao = self.ctx.vertex_array(
                                self._ctex_prog,
                                [(self.ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],np.float32)),
                                  "2f", "in_pos")])
                        self.color_tex.use(location=0)
                        self._ctex_prog['tex_c'].value = 0
                        self._ctex_fbo.use()
                        self.ctx.viewport = (0, 0, 960, 540)
                        self._ctex_vao.render(moderngl.TRIANGLE_STRIP)
                        _carr = np.frombuffer(self._ctex_tex.read(), np.uint8).reshape(540, 960, 3)
                        self.ctx.screen.use()
                        if _carr is not None:
                            cv2.imwrite(_os_rd.path.join(_rd, f"ctex_{_ts}.png"),
                                        np.ascontiguousarray(_carr[:, :, ::-1]))
                    except Exception as _e2:
                        print(f"[rdump] color tex 失败 {_e2}", flush=True)
                    print(f"[rdump] {_ts} fb={_fw}x{_fh}", flush=True)
            except Exception as _e:
                print(f"[rdump] 失败 {_e}", flush=True)
        else:
            self._u_vint_eye_offset.value = half_ipd
            self._u_vint_depth_strength.value = depth_val
            self._u_vint_parallax.value = par_u
            self._u_vint_feather_enabled.value = self.feather_enabled
            self._u_vint_feather_width.value = self.feather_width
            self._u_vint_viewport.value = viewport
            self.vertical_interleaved_vao.render(moderngl.TRIANGLE_STRIP)

        self._render_overlay()
        if self.stream_mode is None:
            glfw.poll_events()
        return

    def _render_sbs_tab(self, win_w, win_h, tex_w, tex_h):
        # 原 fill_16_9 / 非填充两分支逐行保留（公式不并）
        if self.fix_aspect:
            if self.display_mode == "Full-SBS":
                glfw.set_window_aspect_ratio(self.window, 2*tex_w, tex_h)
            elif self.display_mode == "Full-TAB":
                glfw.set_window_aspect_ratio(self.window, tex_w, 2*tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)
        
        # Clear screen once
        self.ctx.clear(0.0, 0.0, 0.0)
        
        # Handle other display modes (non-depth map)
        if self.fill_16_9:
            if self.display_mode in ["Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB"]:
                self.color_tex.use(location=0)
                self.depth_tex.use(location=1)
                self._zc_draw_probe("sbs")
                self._u_depth_strength.value = 1.0 if self._frame_has_depth else 0.0
                # 视差按宽度归一化到 1920 基准：P 的语义是"1080p 下的像素数"，
                # 4K(3840) 时等效位移自动 ×2，观感与 1080p 一致。
                self._u_p_parallax.value = dibr.sbs_parallax_uv(
                    self.parallax_px, self.ipd_uv, win_w)
            try:
                if self._u_ic is not None:
                    import os as _os_ic
                    self._u_ic.value = (0.0 if _os_ic.environ.get("D2S_IC_SEARCH", "0") == "0" else 1.0)
                    self._u_dstep.value = 1.0 / max(1.0, float(getattr(self, "_gpu_res", 0) or 224))
                    if self._u_icdbg is not None:
                        self._u_icdbg.value = (1.0 if _os_ic.environ.get("D2S_IC_DEBUG", "0") == "1" else 0.0)
            except Exception:
                pass

                if self.display_mode == "Full-SBS":
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0

                    # Left view
                    left_vp = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(left_vp, -1)

                    # Right view
                    right_vp = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(right_vp, 1)

                elif self.display_mode == "Half-SBS":
                    # 我们是 2D→3D：每只眼都用【整张图】再各自做视差位移，
                    # 不能按"源图已是左右拼接"去取一半（那是立体直通模式，
                    # 取一半会让左右眼完全相同 → 实测视差 ≈ 0，3D 失效）。
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w / 2.0, win_h
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
                    center_y = win_h / 2.0

                    # Left view
                    left_vp = (
                        int(win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(left_vp, -1)

                    # Right view
                    right_vp = (
                        int(3 * win_w / 4.0 - render_w / 2),
                        int(center_y - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(right_vp, 1)
                    self._last_eye_offset_set = self.ipd_uv / 2.0

                elif self.display_mode in ["Half-TAB"]:
                    src_w, src_h = tex_w, tex_h / 2.0
                    max_w, max_h = win_w, win_h / 2.0
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)

                    # Top view (left eye)
                    top_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(top_vp, -1, 0)

                    # Bottom view (right eye)
                    bottom_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(3 * win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(bottom_vp, 1, 1)

                elif self.display_mode == "Full-TAB":
                    src_w, src_h = tex_w, tex_h
                    max_w, max_h = win_w, win_h / 2.0
                    render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)

                    # Top view (left eye)
                    top_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(top_vp, -1, 0)

                    # Bottom view (right eye)
                    bottom_vp = (
                        int(win_w / 2.0 - render_w / 2),
                        int(3 * win_h / 4.0 - render_h / 2),
                        render_w, render_h
                    )
                    self._draw_eye(bottom_vp, 1, 1)

        else:
            # Determine effective stereo frame size by display mode
            if self.display_mode == "Full-SBS":
                disp_w, disp_h = 2 * tex_w, tex_h
            elif self.display_mode == "Half-SBS":
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode in ["Half-TAB"]:
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode == "Full-TAB":
                disp_w, disp_h = tex_w, 2 * tex_h
            elif self.display_mode == "Depth Map":
                disp_w, disp_h = tex_w, tex_h
            else:
                disp_w, disp_h = 2 * tex_w, tex_h  # default full SBS

            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9/16

            # Scale to fit window, preserving aspect ratio
            if window_aspect <= target_aspect:
                # Window is wider than content
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                # Window is taller than content
                view_w = win_w
                view_h = int(view_w * target_aspect)

            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2

            if self.display_mode in ["Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB"]:
                self.color_tex.use(0)
                self.depth_tex.use(1)
                self._zc_draw_probe("sbs-nofill")
                self._u_depth_strength.value = 1.0 if self._frame_has_depth else 0.0
                # 视差按宽度归一化到 1920 基准：P 的语义是"1080p 下的像素数"，
                # 4K(3840) 时等效位移自动 ×2，观感与 1080p 一致。
                self._u_p_parallax.value = dibr.sbs_parallax_uv(
                    self.parallax_px, self.ipd_uv, win_w)
                # 诊断（D2S_SBS_DEBUG=1）：打印实际生效的 uniform
                import os as _os_dbg
                if _os_dbg.environ.get("D2S_SBS_DEBUG") == "1":
                    _nowdbg = time.perf_counter()
                    if _nowdbg - getattr(self, "_dbg_ts", 0.0) > 1.0:
                        self._dbg_ts = _nowdbg
                        print(f"[sbs-dbg] eye_off={self.ipd_uv/2:.4f} "
                              f"u_parallax={self._u_p_parallax.value:.4f} "
                              f"depth_strength={self._u_depth_strength.value} "
                              f"conv={self.convergence:.2f} curve={getattr(self, 'parallax_curve', '?')} "
                              f"has_depth={self._frame_has_depth} "
                              f"ipd_uv={self.ipd_uv:.4f} win_w={win_w}", flush=True)

                if self.display_mode == "Full-SBS":
                    # Left eye
                    left_vp = (offset_x, offset_y, view_w // 2, view_h)
                    self._draw_eye(left_vp, -1)

                    # Right eye
                    right_vp = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self._draw_eye(right_vp, 1)

                elif self.display_mode == "Half-SBS":
                    # Left eye
                    left_vp = (offset_x, offset_y, view_w // 2, view_h)
                    self._draw_eye(left_vp, -1)

                    # Right eye
                    right_vp = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                    self._draw_eye(right_vp, 1)

                elif self.display_mode == "Half-TAB":
                    # Top eye (left)
                    top_vp = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                    self._draw_eye(top_vp, -1, 0)

                    # Bottom eye (right)
                    bottom_vp = (offset_x, offset_y, view_w, view_h // 2)
                    self._draw_eye(bottom_vp, 1, 1)

                elif self.display_mode == "Full-TAB":
                    # Top eye (left)
                    top_vp = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                    self._draw_eye(top_vp, -1, 0)

                    # Bottom eye (right)
                    bottom_vp = (offset_x, offset_y, view_w, view_h // 2)
                    self._draw_eye(bottom_vp, 1, 1)

        # Skip swap buffers if window is not visible (for headless/streaming)
        self._render_overlay()
        if self.stream_mode is None:
            glfw.poll_events()


# ============================================================================
# DWM 合成节拍钟（Present 对齐实验）


# ============================================================================
class DwmComposeClock:
    """后台 DwmFlush() 循环：记录每次 DWM 合成完成的 QPC 时刻。

    渲染循环节奏由新帧驱动（相位随机），swap 落点相对 DWM 合成边界
    随机（0~16.7ms 均匀）——"赶上截止"全凭运气。此钟预测下一次合成
    边界，供 swap 前对齐（settings `Present Guard (ms)`，0=关闭）。
    2D 静止桌面 DWM 可能不合成 → 时钟陈旧 → next_boundary 返回 None，
    对齐自动失效（零行为变化）。

    附带收益：timeBeginPeriod(1) 让本进程的 sleep 达到 1ms 精度
    （Win10 2004+ 进程作用域），guard 的 1~16ms 等待在默认 15.6ms
    分辨率下会睡过头。
    """
    def __init__(self):
        import threading as _th
        from collections import deque as _deq
        self._dwm = ctypes.windll.dwmapi
        self._dwm.DwmFlush.restype = ctypes.c_long
        self._winmm = None
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            self._winmm = ctypes.windll.winmm
        except Exception:
            pass
        self._t = _deq(maxlen=8)
        self._stop = _th.Event()
        _th.Thread(target=self._run, daemon=True,
                   name="DwmComposeClock").start()

    def _run(self):
        perf = time.perf_counter
        while not self._stop.is_set():
            try:
                hr = self._dwm.DwmFlush()
            except Exception:
                return
            if hr != 0:
                time.sleep(0.5)     # DWM 未就绪（会话切换等）：慢速重试
                continue
            self._t.append(perf())

    def stop(self):
        self._stop.set()
        if self._winmm is not None:
            try:
                self._winmm.timeEndPeriod(1)
            except Exception:
                pass

    def next_boundary(self, now=None):
        """下一次合成完成时刻的预测（QPC）；样本不足/时钟陈旧返回 None。"""
        if len(self._t) < 2:
            return None
        now = time.perf_counter() if now is None else now
        last = self._t[-1]
        if now - last > 0.2:                       # 2D 静止：时钟停摆
            return None
        interval = (self._t[-1] - self._t[0]) / (len(self._t) - 1)
        if not (0.004 < interval < 0.050):         # 20~250Hz 之外视为异常
            return None
        k = int((now - last) / interval) + 1
        nb = last + k * interval
        while nb <= now:                           # 预测不许落在过去
            nb += interval
        return nb
