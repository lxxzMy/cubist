# -*- coding: utf-8 -*-
"""裸眼3D 红线对位工具：黑底 + 红色斜线阵。

左上角 OSD 显示当前参数。Pause 键轮预设（含细步进）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glfw
import moderngl

# 预设：粗扫（大步）在前面，细扫（slope 微调）在后面
PRESETS = []
for sl in (0.40, 0.45, 0.50, 0.55, 0.60):
    for p in (8, 6, 4, 10, 12, 5, 7, 9):
        PRESETS.append((p, sl))
for sl in (0.48, 0.52, 0.46, 0.54):
    for p in (8, 6):
        PRESETS.append((p, sl))

VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

VERT_OSD = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() { uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG_OSD = """
#version 330
in vec2 uv;
out vec4 frag;
uniform sampler2D u_t;
void main() { frag = texture(u_t, uv); }
"""

FRAG = """
#version 330
out vec4 frag;
uniform vec2  u_res;      // 窗口像素
uniform float u_pitch;    // 透镜节距（像素）
uniform float u_slope;    // 斜率 dY/dX
uniform float u_shift;    // 相位偏移（像素）

void main() {
    // 亚像素列号（RGB 条纹：横向分辨率 = 像素 x 3）
    float sub = gl_FragCoord.x * 3.0 - 0.5 + gl_FragCoord.y * u_slope * 3.0 + u_shift * 3.0;
    float ph = fract(sub / (u_pitch * 3.0));
    // 每 pitch 周期一根红线，线宽 = 1 个亚像素
    float w = 1.0 / (u_pitch * 3.0);
    frag = vec4(vec3(1.0, 0.0, 0.0) * step(ph, w), 1.0);
}
"""


def main():
    glfw.init()
    # 按 Windows 坐标精确匹配目标屏：柱状透镜屏 = 非主屏 = 位于 (1920,0)
    # （GLFW 枚举顺序与 Windows 编号不对应，mon[-1] 曾开到主屏）
    target = None
    for m in glfw.get_monitors():
        mx, my = glfw.get_monitor_pos(m)
        if (mx, my) == (1920, 0):
            target = m
            break
    if target is None:
        target = glfw.get_primary_monitor()
    vm = glfw.get_video_mode(target)
    W, H = vm.size.width, vm.size.height
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    win = glfw.create_window(W, H, "red-align", target, None)
    glfw.make_context_current(win)
    glfw.set_window_pos(win, *glfw.get_monitor_pos(target))
    glfw.swap_interval(1)
    ctx = moderngl.create_context()

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    prog_osd = ctx.program(vertex_shader=VERT_OSD, fragment_shader=FRAG_OSD)
    vbo_osd = ctx.buffer(np.array(
        [-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0], np.float32).tobytes())
    vao_osd = ctx.vertex_array(prog_osd, vbo_osd, "in_pos", "in_uv")
    vbo = ctx.buffer(np.array(
        [-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0], np.float32).tobytes())
    vao = ctx.vertex_array(prog, vbo, "in_pos")

    u_pitch = prog["u_pitch"]
    u_slope = prog["u_slope"]; u_shift = prog["u_shift"]

    # OSD 纹理（左上角）
    def refresh_osd(idx):
        p, sl = PRESETS[idx]
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("msyh.ttc", 32)
        except Exception:
            f = None
        txt = "[%d/%d]  pitch=%g  slope=%g" % (idx + 1, len(PRESETS), p, sl)
        dr.rectangle([10, 10, 640, 60], fill=(0, 0, 0, 200))
        dr.text((20, 14), txt, font=f, fill=(255, 80, 80))
        t = ctx.texture((W, H), 4, img.tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return t

    osd_tex = refresh_osd(0)
    pitch, sl = PRESETS[0]
    u_pitch.value, u_slope.value = float(pitch), float(sl)

    import ctypes
    u32 = ctypes.windll.user32
    prev = {}
    idx = 0
    shift = 0.0

    def key(vk):
        return bool(u32.GetAsyncKeyState(vk) & 0x8000)

    hwnd = glfw.get_win32_window(win)

    def reclaim_topmost():
        # 每帧抢回 topmost：其他窗口（浏览器弹窗等）置顶会压住本窗，
        # "后到者胜" —— 不抢回用户就看不到红线（主 viewer 同款机制）
        u32.SetWindowPos(hwnd, -1, 0, 0, 0, 0,
                         0x0001 | 0x0002 | 0x0010 | 0x0040)  # TOPMOST|NOMOVE|NOSIZE|NOACTIVATE|SHOWWINDOW

    while not glfw.window_should_close(win):
        reclaim_topmost()
        ctx.screen.use()
        ctx.viewport = (0, 0, W, H)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        osd_tex.use(0)
        prog_osd["u_t"].value = 0
        ctx.enable(moderngl.BLEND)
        vao_osd.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        glfw.swap_buffers(win)
        glfw.poll_events()

        st = key(0x13)   # Pause
        if st and not prev.get("p", False):
            idx = (idx + 1) % len(PRESETS)
            pitch, sl = PRESETS[idx]
            u_pitch.value, u_slope.value = float(pitch), float(sl)
            old = osd_tex
            osd_tex = refresh_osd(idx)
            old.release()
        prev["p"] = st
        # 左/右 = 相位微调
        if key(0x25):
            shift -= 0.25; u_shift.value = shift
        if key(0x27):
            shift += 0.25; u_shift.value = shift

    glfw.terminate()


if __name__ == "__main__":
    main()
