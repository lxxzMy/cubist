# -*- coding: utf-8 -*-
"""柱状透镜预览（DIBR）：网格验证 + 深度场景，slope=0.5 已确认。

按键：Pause=切预设  空格=网格/3D场景  A=翻转视点顺序  P=翻转视差符号
      左/右方向键 = 手动眼位偏移（BaseOffset，模拟眼动）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glfw
import moderngl
from OpenGL.GL import *

# (pitch, slope, views) —— slope=0.5/pitch=8 已由用户确认对位
PRESETS = []
_slope_c = (0.40, 0.45, 0.50, 0.55, 0.60)
for _sl in _slope_c:
    for _p in (8, 7, 6, 9, 12, 10):
        PRESETS.append((_p, _sl, 8))
PRESETS.append((16, 0.5, 8)); PRESETS.append((4, 0.5, 8))

VERT = """
    #version 330
    in vec2 in_pos;
    in vec2 in_uv;
    out vec2 uv;
    void main() { uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# DIBR 五步：
# 1 Depth = 场景亮度（白=近黑=远）  2 BaseOffset = 眼位*深度
# 3 view_id 由 pitch/slope          4 FinalOffset = Base + (id-center)*Depth*Step
# 5 采样 uv + FinalOffset
FRAG_WEAVE = """
    #version 330
    in vec2 uv;
    out vec4 frag;
    uniform sampler2D u_scene;
    uniform vec2  u_res;
    uniform float u_pitch;       // 透镜节距（物理像素）
    uniform float u_slope;       // 斜率 dY/dX
    uniform int   u_views;       // 视角数
    uniform float u_eye;         // 手动眼位 -0.5..+0.5
    uniform float u_par_px;      // 每视点步长（像素）
    uniform int   u_flip;
    uniform float u_par_sign;
    uniform int   u_scene_mode;  // 0=对位点阵 1=DIBR
    uniform float u_shift;

    void main() {
        // 亚像素坐标：物理横向分辨率 = 像素数 x 3（RGB 条纹）
        float sub_x = gl_FragCoord.x * 3.0 - 0.5;      // 亚像素中心列号（0=R,1=G,2=B 循环）
        int   my_comp = int(mod(sub_x, 3.0));          // 我的物理颜色类型 0=R 1=G 2=B
        float axis = sub_x + gl_FragCoord.y * u_slope * 3.0
                   + u_shift * 3.0 + u_eye * u_res.x * 3.0;
        float cell = axis / (u_pitch * 3.0);           // 透镜单元坐标（一个周期 = views*3 亚像素）
        float frac = fract(cell);
        int slot = int(frac * float(u_views * 3));     // 槽位：view*3 + comp
        int view = slot / 3;
        int comp = slot - view * 3;                    // 该槽位期望的颜色分量
        if (u_flip == 1) view = u_views - 1 - view;

        if (u_scene_mode == 0) {
            // 红线条：只有"槽位期望 R 分量 && 我物理是 R 亚像素"才亮红
            if (comp == 0 && my_comp == 0) {
                frag = vec4(1.0, 0.0, 0.0, 1.0);
            } else {
                frag = vec4(0.0);
            }
            return;
        }

        // ===== DIBR：本亚像素属于视点 view，且我物理上是 comp 分量 =====
        // 只有 comp == my_comp 的亚像素承载可见光（透镜把 R 亚像素的光送进
        // 眼睛时它只能发红——发别的色会串到相邻视点，这就是彩边/串扰根源）
        if (my_comp != comp) { frag = vec4(0.0); return; }

        float t = (float(view) + 0.5) / float(u_views) - 0.5;   // -0.5..+0.5
        float depth = texture(u_scene, uv).r;                    // 亮=近
        float base = u_eye * depth * u_par_px * 2.0;
        float off = base + (t) * depth * u_par_px * u_par_sign;
        vec2 src = uv - vec2(off / u_res.x, 0.0);
        float d2 = texture(u_scene, src).r;
        float off2 = base + t * d2 * u_par_px * u_par_sign;
        src = uv - vec2(off2 / u_res.x, 0.0);
        if (src.x < 0.0 || src.x > 1.0) { frag = vec4(0.0); return; }

        vec4 c = texture(u_scene, src);
        frag = vec4(c[my_comp], c[my_comp], c[my_comp], 1.0);
    }
"""

FRAG_OSD = """
    #version 330
    in vec2 uv;
    out vec4 frag;
    uniform sampler2D u_osd;
    void main() { frag = texture(u_osd, uv); }
"""


def make_osd(lines, W, H, ctx):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("msyh.ttc", 28)
    except Exception:
        f = None
    y = 12
    for line in lines:
        dr.rectangle([8, y - 4, 8 + 620, y + 34], fill=(0, 0, 0, 170))
        dr.text((16, y), line, font=f, fill=(255, 255, 255, 255))
        y += 40
    tex = ctx.texture((W, H), 4, img.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return tex


def make_scene(W, H, ctx):
    """深度场景：白=近（文字/球），灰=远。返回 rgba_tex。"""
    img = Image.new("L", (W, H), 30)
    dr = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("msyh.ttc", 300)
        dr.text((W // 2 - 380, H // 2 - 220), "3D", font=f, fill=230)
        f2 = ImageFont.truetype("msyh.ttc", 120)
        dr.text((W // 2 - 300, H // 2 + 120), "8 view", font=f2, fill=200)
    except Exception:
        pass
    ball = Image.new("L", (400, 400), 0)
    bd = ImageDraw.Draw(ball)
    bd.ellipse([60, 60, 340, 340], fill=240)
    img.paste(ball, (W - 640, H - 640))
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgba = np.dstack([arr * 255] * 3 + [np.full_like(arr, 255)]).astype(np.uint8)
    tex = ctx.texture((W, H), 4, rgba.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def main():
    glfw.init()
    mons = glfw.get_monitors()
    mon = mons[-1]
    vm = glfw.get_video_mode(mon)
    W, H = vm.size.width, vm.size.height
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    win = glfw.create_window(W, H, "lens-preview", mon, None)
    glfw.make_context_current(win)
    glfw.set_window_pos(win, *glfw.get_monitor_pos(mon))
    glfw.swap_interval(1)
    ctx = moderngl.get_context()

    prog_w = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_WEAVE)
    prog_o = ctx.program(vertex_shader=VERT, fragment_shader=FRAG_OSD)
    vbo = ctx.buffer(np.array(
        [-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0], np.float32).tobytes())
    vao_w = ctx.vertex_array(prog_w, vbo, "in_pos", "in_uv")
    vao_o = ctx.vertex_array(prog_o, vbo, "in_pos", "in_uv")

    scene_tex = make_scene(W, H, ctx)
    osd_tex = make_osd([""], W, H, ctx)

    u_res = prog_w["u_res"]; u_pitch = prog_w["u_pitch"]; u_slope = prog_w["u_slope"]
    u_views = prog_w["u_views"]; u_eye = prog_w["u_eye"]; u_par = prog_w["u_par_px"]
    u_flip = prog_w["u_flip"]; u_sign = prog_w["u_par_sign"]
    u_mode = prog_w["u_scene_mode"]
    u_res.value = (float(W), float(H))
    prog_w["u_scene"].value = 0
    prog_o["u_osd"].value = 0

    pi = 0
    pitch, slope, views = PRESETS[pi]
    eye = 0.0
    flip = 0
    sign = 1.0
    scene_mode = 0   # 固定红线条

    def apply():
        u_pitch.value = float(pitch)
        u_slope.value = float(slope)
        u_views.value = views
        u_eye.value = eye
        u_par.value = 2.5
        u_flip.value = flip
        u_sign.value = sign
        u_mode.value = scene_mode

    def refresh_osd():
        lines = [
            "[%d] pitch=%g slope=%g views=%d" % (pi, pitch, slope, views),
            "scene=%s eye=%+.2f flip=%d sign=%+.0f" % (
                ("VIEWS", "DIBR", "LINES")[scene_mode], eye, flip, sign),
            "Pause=preset  Space=scene(VIEWS/DIBR/LINES)  A=flip  P=sign  Left/Right=eye",
        ]
        return make_osd(lines, W, H, ctx)

    apply()
    osd_tex = refresh_osd()
    print("keys: Pause=preset Space=scene A=flip P=sign Left/Right=eye", flush=True)

    import ctypes
    u32 = ctypes.windll.user32
    prev = {}

    def key(vk):
        return bool(u32.GetAsyncKeyState(vk) & 0x8000)

    while not glfw.window_should_close(win):
        st = key(0x13)
        if st and not prev.get("p", False):
            pi = (pi + 1) % len(PRESETS)
            pitch, slope, views = PRESETS[pi]
            apply(); osd_tex = refresh_osd()
            print("-> [%d] pitch=%g slope=%g views=%d" % (pi, pitch, slope, views),
                  flush=True)
        prev["p"] = st

        st = key(0x41)
        if st and not prev.get("a", False):
            flip = 1 - flip; apply(); osd_tex = refresh_osd()
        prev["a"] = st

        st = key(0x50)
        if st and not prev.get("g", False):
            sign = -sign; apply(); osd_tex = refresh_osd()
        prev["g"] = st

        if key(0x25):
            eye = max(-0.5, eye - 0.01); apply()
        if key(0x27):
            eye = min(0.5, eye + 0.01); apply()
        u_eye.value = eye

        ctx.screen.use()
        ctx.viewport = (0, 0, W, H)
        ctx.clear(0.03, 0.03, 0.03, 1.0)
        scene_tex.use(0)
        prog_w["u_scene"].value = 0
        apply()
        vao_w.render(moderngl.TRIANGLE_STRIP)
        osd_tex.use(0)
        prog_o["u_osd"].value = 0
        ctx.enable(moderngl.BLEND)
        vao_o.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
