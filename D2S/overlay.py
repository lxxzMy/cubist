# -*- coding: utf-8 -*-
"""FPS/Delay 文字覆盖层的 GL 纹理渲染器（原 viewer.py 内联类，
2026-11-16 抽出；位图生成仍在 viewer.StereoWindow._generate_overlay_image）。"""

import glfw
import moderngl
import numpy as np

class OverlayTextureRenderer:
    """Small RGBA overlay rendered as a separate GL texture."""

    VERTEX_SHADER = """
        #version 330
        in vec2 in_position;
        in vec2 in_uv;
        out vec2 uv;
        void main() {
            uv = in_uv;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
    """

    FRAGMENT_SHADER = """
        #version 330
        in vec2 uv;
        out vec4 frag_color;
        uniform sampler2D u_overlay;
        void main() {
            frag_color = texture(u_overlay, uv);
        }
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )
        self.prog["u_overlay"].value = 0
        self.texture = None
        self.size = None
        self.vbo = None
        self.vao = None
        self._geometry_key = None

    def update_texture(self, rgba):
        if rgba is None:
            return False
        h, w = rgba.shape[:2]
        if w <= 0 or h <= 0:
            return False
        if self.texture is None or self.size != (w, h):
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture((w, h), 4, dtype="f1")
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.size = (w, h)
        self.texture.write(np.ascontiguousarray(rgba).tobytes())
        return True

    def clear(self):
        if self.texture is not None:
            try:
                self.texture.release()
            except Exception:
                pass
        self.texture = None
        self.size = None
        self._geometry_key = None

        try:
            import os as _os_dbg
            if (_os_dbg.environ.get("D2S_UNIFORM_DEBUG") == "1"
                    and getattr(self, "window", None) is not None
                    and not getattr(self, "_uni_dbg", False)):
                self._uni_dbg = True
                def _val(name):
                    u = getattr(self, name, None)
                    return None if u is None else getattr(u, "value", None)
                print(f"[uniforms] parallax={_val('_u_p_parallax')} "
                      f"eye_offset={_val('_u_eye_offset')} "
                      f"convergence={_val('_u_convergence')} "
                      f"depth_strength={_val('_u_depth_strength')} "
                      f"parallax_px={getattr(self, 'parallax_px', None)} "
                      f"ipd_uv={getattr(self, 'ipd_uv', None)} "
                      f"win={glfw.get_window_size(self.window)} "
                      f"fb={glfw.get_framebuffer_size(self.window)}", flush=True)
        except Exception as _e:
            print(f"[uniforms] dump failed: {_e}", flush=True)
    def render(self, window_size, position, scale=(1.0, 1.0)):
        if self.texture is None or self.size is None:
            return
        win_w, win_h = window_size
        tex_w, tex_h = self.size
        sx, sy = scale
        if sx != 1.0 or sy != 1.0:
            tex_w, tex_h = tex_w * sx, tex_h * sy
        if win_w <= 0 or win_h <= 0:
            return
        x, y = position
        left = (x / win_w) * 2.0 - 1.0
        right = ((x + tex_w) / win_w) * 2.0 - 1.0
        top = 1.0 - (y / win_h) * 2.0
        bottom = 1.0 - ((y + tex_h) / win_h) * 2.0
        geometry_key = (win_w, win_h, tex_w, tex_h, x, y, sx, sy)
        if self.vao is None or self._geometry_key != geometry_key:
            verts = np.array([
                left, bottom, 0.0, 1.0,
                right, bottom, 1.0, 1.0,
                left, top, 0.0, 0.0,
                right, top, 1.0, 0.0,
            ], dtype="f4")
            if self.vbo is None:
                self.vbo = self.ctx.buffer(verts.tobytes())
            else:
                self.vbo.orphan(verts.nbytes)
                self.vbo.write(verts.tobytes())
            if self.vao is None:
                self.vao = self.ctx.vertex_array(
                    self.prog, [(self.vbo, "2f 2f", "in_position", "in_uv")]
                )
            self._geometry_key = geometry_key

        previous_viewport = self.ctx.viewport
        self.ctx.viewport = (0, 0, win_w, win_h)
        self.texture.use(location=0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.viewport = previous_viewport

    def release(self):
        for obj in (self.vao, self.vbo, self.texture, self.prog):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        self.vao = None
        self.vbo = None
        self.texture = None
        self.prog = None
        self.size = None


