"""全屏 3D 输出时隐藏系统光标（形状可反查），带三重恢复保障。

关键设计（都是实测结论）：
  1. 空白光标必须是【合法尺寸】32x32 全透明；1xN 怪尺寸会让 GetCursorInfo
     返回 None，形状就再也查不到了。
  2. 用【热点坐标】编码原光标 ID 的序号：全透明的光标热点取哪都看不见，
     但 GetIconInfo 能读回来 —— 这是"隐藏后仍知道当前是什么形状"的唯一办法。
  3. SetSystemCursor 会接管并销毁传入的句柄，所以每个 ID 都要新建一个。
  4. 恢复走三重：模式切换 / atexit / 独立看护进程（主进程崩溃也能救）。
"""
import atexit
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time

u32 = ctypes.windll.user32
g32 = ctypes.windll.gdi32
SPI_SETCURSORS = 0x0057

# 系统光标 ID 表（顺序即编码序号）
OCR_IDS = (32512, 32513, 32514, 32515, 32516, 32642, 32643, 32644,
           32645, 32646, 32648, 32649, 32650, 32651)

ID_NAME = {32512: "Arrow", 32513: "IBeam", 32514: "Wait", 32515: "Crosshair",
           32516: "UpArrow", 32642: "SizeNWSE", 32643: "SizeNESW",
           32644: "SizeWE", 32645: "SizeNS", 32646: "SizeAll", 32648: "No",
           32649: "Hand", 32650: "AppStarting", 32651: "Help"}

_SIZE = 64          # 必须够大：热点编码基址 32 + 序号(0~13) 要在 0..63 内
_ENC_BASE = 32      # 真实光标热点都在 0~16，不冲突
_state = {"hidden": False}
_last_ok = [None]      # 上次打印的替换成功数（日志降噪：仅在变化时打）
_lock = threading.Lock()


class _ICONINFO_T(ctypes.Structure):
    _fields_ = [("fIcon", wt.BOOL), ("xHotspot", wt.DWORD), ("yHotspot", wt.DWORD),
                ("hbmMask", wt.HANDLE), ("hbmColor", wt.HANDLE)]


# identify() 每帧都调（d3d_window._cursor_system_visible），结构体必须模块级：
# 函数内 `class _BM(Structure)` 每次都造新类型 → 类型↔StgDict↔字段描述符
# 成环 = 每秒上千个循环垃圾（GC 探针实测 cursor_hide.py:186 是大头之一）。
class _BM(ctypes.Structure):
    _fields_ = [("bmType", ctypes.c_long), ("bmWidth", ctypes.c_long),
                ("bmHeight", ctypes.c_long), ("bmWidthBytes", ctypes.c_long),
                ("bmPlanes", ctypes.c_ushort), ("bmBitsPixel", ctypes.c_ushort),
                ("bmBits", ctypes.c_void_p)]


class _BMIH(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPels", wt.LONG), ("biYPels", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


class _BMIH_V4(ctypes.Structure):
    """BITMAPV4HEADER：带显式 RGBA mask，才能逐像素 alpha。"""
    _fields_ = [("bV4Size", wt.DWORD), ("bV4Width", wt.LONG), ("bV4Height", wt.LONG),
                ("bV4Planes", wt.WORD), ("bV4BitCount", wt.WORD),
                ("bV4V4Compression", wt.DWORD), ("bV4SizeImage", wt.DWORD),
                ("bV4XPelsPerMeter", wt.LONG), ("bV4YPelsPerMeter", wt.LONG),
                ("bV4ClrUsed", wt.DWORD), ("bV4ClrImportant", wt.DWORD),
                ("bV4RedMask", wt.DWORD), ("bV4GreenMask", wt.DWORD),
                ("bV4BlueMask", wt.DWORD), ("bV4AlphaMask", wt.DWORD),
                ("bV4CSType", wt.DWORD), ("bV4Endpoints", ctypes.c_byte * 36),
                ("bV4GammaRed", wt.DWORD), ("bV4GammaGreen", wt.DWORD),
                ("bV4GammaBlue", wt.DWORD)]


BI_BITFIELDS = 3


def _blank_cursor(index):
    """64x64 光标：完全透明（颜色 alpha 全 0、掩码全 1）。

    四个实测坑：
    1. 完全透明【不会】让 GetCursorInfo 返回 NULL —— BI_BITFIELDS + alpha 0、
       掩码全 1 实测 hCursor 仍然有效。反过来，早期为了"图像非空"在掩码里
       留 1 个"要画"的像素，会 XOR 出 1~2 个黑点（112px 下放大成 2x2），
       所以现在不留了。
    2. 32bpp【BI_RGB】的 DIB 没有 alpha 通道，Windows 按 mask 位图判透明度
       （mask 全 0 = 黑方块，实测踩过）。要逐像素 alpha 必须用 BI_BITFIELDS
       并显式给出 alpha mask。
    3. 序号用【宽度】编码：宽 = 64 + 序号（64..77），GetIconInfo 读 mask
       宽度反查（与掩码内容无关）。不用热点编码——实测 Windows 会把热点
       规范化，读回来对不上。
    4. 1bpp 掩码的扫描线必须 WORD 对齐（见下方 row_bytes）。
    """
    _w = _SIZE + int(index)
    v4 = _BMIH_V4()
    v4.bV4Size = ctypes.sizeof(_BMIH_V4)
    v4.bV4Width = _w
    v4.bV4Height = -_SIZE
    v4.bV4Planes = 1
    v4.bV4BitCount = 32
    v4.bV4V4Compression = BI_BITFIELDS
    v4.bV4RedMask = 0x00FF0000
    v4.bV4GreenMask = 0x0000FF00
    v4.bV4BlueMask = 0x000000FF
    v4.bV4AlphaMask = 0xFF000000
    bits = ctypes.c_void_p()
    hbm = g32.CreateDIBSection(None, ctypes.byref(v4), 0,
                               ctypes.byref(bits), None, 0)
    if not hbm:
        return 0
    try:
        # 颜色位图整块 alpha=0（掩码全 1，任何像素都不该被画出来）。
        # 注意：连"顺手写 1 个像素"都不要 —— 实测 alpha=0 但 RGB 非零的像素
        # 仍会被 DrawIconEx 画出来（在 112px 下变成 2x2 的黑点）。调试时才写。
        buf = (ctypes.c_ubyte * (_SIZE * _w * 4)).from_address(bits.value)
        _a = int(os.environ.get("D2S_CURSOR_ALPHA", "0")) & 0xFF
        if _a:
            _rgb = os.environ.get("D2S_CURSOR_RGB", "FFFFFF")
            try:
                _r, _g, _b = (int(_rgb[i:i + 2], 16) for i in (0, 2, 4))
            except Exception:
                _r = _g = _b = 255
            buf[0], buf[1], buf[2], buf[3] = _b, _g, _r, _a
    except Exception:
        pass
    # 1bpp 掩码的扫描线必须【WORD 对齐】（GDI 规定）。(w+7)//8 在宽度
    # 65~72 时得到 9 字节，实际行距是 10 字节 → 之后每一行都错位，
    # 空白光标变成几百个黑像素（实测 924~1136 个不透明像素，用户看到
    # "有的形状是一堆黑块"）。宽度 64 / 73~80 恰好两者相等，才没暴露。
    row_bytes = ((_w + 15) // 16) * 2
    mask_buf = bytearray(b"\xff" * (row_bytes * _SIZE))   # 全 1 = 全透明
    # 不再清某一位留"要画"像素：那一位会 XOR 出黑点（见 docstring 坑 1）
    mask = g32.CreateBitmap(_w, _SIZE, 1, 1, bytes(mask_buf))
    ii = _ICONINFO_T(0, 0, 0, mask, hbm)
    cur = u32.CreateIconIndirect(ctypes.byref(ii))
    g32.DeleteObject(ctypes.c_void_p(hbm))
    g32.DeleteObject(ctypes.c_void_p(mask))
    return cur


def hide(force=False):
    """空白替换系统光标表（SBS 模式必须：真实光标只有 1 个，而两半各要一个）。
    每个 ID 用宽度编码序号，届时 identify() 能反查出"这个位置原本该是哪种
    光标"，再从自带方案表里取对应位图绘制。"""
    with _lock:
        if _state["hidden"] and not force:
            return True
        ok = 0
        for idx, cid in enumerate(OCR_IDS):
            h = _blank_cursor(idx)
            if not h:
                continue
            if u32.SetSystemCursor(h, cid):
                ok += 1
        _state["hidden"] = ok > 0
        # 关键：SetSystemCursor 只改系统光标表，屏幕上【当前显示】的那一支
        # 常常还是旧句柄（要等下一次 SetCursor 才换）。所以主动把箭头 ID
        # 再 Set 一次 —— 此时表里已是空白位图，屏幕上立刻变透明。
        try:
            u32.SetCursor(u32.LoadCursorW(None, 32512))
        except Exception:
            pass
        # 日志降噪：3D 期间每 5s 的无条件重贴是常态，只在"结果变化"或
        # 主动隐藏（进 3D 那次）时打一行，发布版日志不再每 5s 刷一条。
        if (not force) or ok != _last_ok[0]:
            _last_ok[0] = ok
            print(f"[cursor] 空白光标替换 {ok}/{len(OCR_IDS)} 成功", flush=True)
        return _state["hidden"]


def restore():
    with _lock:
        try:
            u32.SystemParametersInfoW(SPI_SETCURSORS, 0, None, 0)
        except Exception:
            pass
        _state["hidden"] = False


def identify(hcur):
    """由空白光标句柄反查原光标 ID（-1 = 不是我们的空白光标）。

    序号编码在 mask 位图的【宽度】上（64 + 序号）：Windows 会规范化热点，
    但位图宽度是我们自己定的，读回来是准的（实测）。
    """
    if not hcur:
        return -1
    try:
        ii = _ICONINFO_T()
        if not u32.GetIconInfo(ctypes.c_void_p(int(hcur)), ctypes.byref(ii)):
            return -1
        idx = -1
        try:
            bm = _BM()
            if ii.hbmMask and g32.GetObjectW(ctypes.c_void_p(ii.hbmMask),
                                             ctypes.sizeof(bm), ctypes.byref(bm)):
                idx = int(bm.bmWidth) - _SIZE      # 宽 = _SIZE + 序号
        except Exception:
            idx = -1
        for hb in (ii.hbmMask, ii.hbmColor):
            if hb:
                try:
                    g32.DeleteObject(ctypes.c_void_p(hb))
                except Exception:
                    pass
        if 0 <= idx < len(OCR_IDS):
            return OCR_IDS[idx]
        return -1        # 真实光标（宽度不匹配）→ 不是我们的
    except Exception:
        return -1


atexit.register(restore)


# ------------------------------------------------------------------ 看护进程
_GUARD_SRC = r'''
import ctypes, sys, time
u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32
h = k32.OpenProcess(0x00100000, False, int(sys.argv[1]))
if h:
    k32.WaitForSingleObject(ctypes.c_void_p(h), 0xFFFFFFFF)
    time.sleep(0.2)
    u32.SystemParametersInfoW(0x0057, 0, None, 0)
'''


def _spawn_guard():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        exe = os.path.join(base, "python3", "python.exe")
        if not os.path.exists(exe):
            exe = sys.executable
        subprocess.Popen([exe, "-c", _GUARD_SRC, str(os.getpid())],
                         creationflags=0x00000008 | 0x08000000,
                         close_fds=True,
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


from winstructs import CURSORINFO as _CURSORINFO


def _current_cursor():
    """当前系统光标句柄（无光标返回 None）。"""
    try:
        ci = _CURSORINFO()
        ci.cbSize = ctypes.sizeof(ci)
        if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
            return None
        return ci.hCursor or None
    except Exception:
        return None

def start_watcher(mode_file, stop_evt=None):
    """轮询 2d/3d 模式文件：进 3D 隐藏、回 2D 恢复。"""
    def _loop():
        last = None
        guard_done = False
        while stop_evt is None or not stop_evt.is_set():
            try:
                with open(mode_file, encoding="utf-8") as f:
                    want = f.read().strip() == "3d"
            except OSError:
                want = False
            if want != last:
                last = want
                if want:
                    hide()
                    # 注意：这里【不能】再用 force_hide_cursor —— ShowCursor(FALSE)
                    # 会让 GetCursorInfo 的 hCursor 变 NULL，identify() 就查不到
                    # "这个位置原本是哪种光标"，SBS 自绘会永远画成箭头。
                    if not guard_done:
                        guard_done = _spawn_guard()
                else:
                    restore()
            elif want:
                # 3D 期间若被外部重置（独占全屏游戏、SPI_SETCURSORS 调用、
                # 某些安装器/UAC 提示），系统光标会变回正常形状。定期复查：
                # identify() 认不出"空白 #k"就说明被换掉了，重新贴一遍。
                _now = time.monotonic()
                if _now - globals().get("_last_verify", 0.0) > 1.0:
                    globals()["_last_verify"] = _now
                    try:
                        # 看"光标表里的箭头"，不要看"当前显示的那支"：
                        # 浏览器/编辑器常持私有光标句柄，identify() 对它必然
                        # 返回 -1，会每秒误报一次"被外部重置"。
                        _h = u32.LoadCursorW(None, 32512)
                        # identify(): 我们的空白 → OCR id；真实光标 → -1
                        _lost = bool(_h) and identify(_h) in (None, -1)
                        if _lost:
                            hide(force=True)
                            print("[cursor] 检测到被外部重置，已重新隐藏", flush=True)
                        elif os.environ.get("D2S_CURSOR_DEBUG") == "1":
                            print(f"[cursor] verify h={_h} id={identify(_h) if _h else None}",
                                  flush=True)
                    except Exception:
                        pass
                # 兜底：ShowCursor 路径下 GetCursorInfo 可能返回 NULL 句柄，
                # 上面识别不出"被重置"。每 5 秒无条件重贴一次（14 次
                # SetSystemCursor，成本 ~1ms），独占全屏/UAC 抢走也不会漏。
                if _now - globals().get("_last_force", 0.0) > 5.0:
                    globals()["_last_force"] = _now
                    hide(force=True)
            time.sleep(0.2)
        restore()

    t = threading.Thread(target=_loop, name="cursor-hide", daemon=True)
    t.start()
    return t


# ---- 真隐藏：挂到前台线程后 ShowCursor(FALSE) 才生效 -------------------
# 光标显示计数是【每线程】的，直接在本进程线程调用 ShowCursor 对前台窗口无效
# （实测：返回 -1 但 GetCursorInfo 仍 SHOWING=True）。挂接输入队列后可见性
# 由中国共享的输入状态决定，才真正消失。
import ctypes as _ct2
_u32b = _ct2.windll.user32
_k32b = _ct2.windll.kernel32
_HIDE_N = [0]          # 我们压低的计数次数（恢复时要等量加回）
_ATTACHED = [0]


def _attach_fg():
    try:
        fg = _u32b.GetForegroundWindow()
        if not fg:
            return False
        t_fg = _u32b.GetWindowThreadProcessId(fg, None)
        t_me = _k32b.GetCurrentThreadId()
        if not t_fg or t_fg == t_me:
            return False
        if _ATTACHED[0] == t_fg:
            return True
        if _u32b.AttachThreadInput(t_me, t_fg, True):
            _ATTACHED[0] = t_fg
            return True
    except Exception:
        pass
    return False


def _detach():
    try:
        if _ATTACHED[0]:
            t_me = _k32b.GetCurrentThreadId()
            _u32b.AttachThreadInput(t_me, _ATTACHED[0], False)
            _ATTACHED[0] = 0
    except Exception:
        pass


def cursor_showing():
    """系统光标当前是否显示（给 viewer 判断用）。"""
    try:
        class _CI(_ct2.Structure):
            _fields_ = [("cbSize", _ct2.c_ulong), ("flags", _ct2.c_ulong),
                        ("hCursor", _ct2.c_void_p), ("pt", _ct2.c_long * 2)]
        ci = _CI()
        ci.cbSize = _ct2.sizeof(_CI)
        if _u32b.GetCursorInfo(_ct2.byref(ci)):
            return bool(ci.flags & 1)
    except Exception:
        pass
    return True


_DBG = [0]


def force_hide_cursor():
    """真隐藏系统光标；重复调用是幂等的（只压低一次计数）。"""
    ok_attach = _attach_fg()
    try:
        # 显示计数不一定从 0 起（别的组件可能加过），必须一直压到真的隐藏
        hidden = not cursor_showing()
        # 计数上限 8：够把任何常见正计数压负；超过就不再压（否则恢复时失控）
        while not hidden and _HIDE_N[0] < 8:
            _u32b.ShowCursor(False)
            _HIDE_N[0] += 1
            hidden = not cursor_showing()
        if os.environ.get("D2S_CURSOR_DEBUG") == "1" and _DBG[0] < 5:
            _DBG[0] += 1
            fg = _u32b.GetForegroundWindow()
            print(f"[cursor-hide-dbg] attach={ok_attach} attached_to={_ATTACHED[0]} "
                  f"show_calls={_HIDE_N[0]} hidden={hidden} fg={fg}", flush=True)
        return hidden
    except Exception as e:
        if os.environ.get("D2S_CURSOR_DEBUG") == "1" and _DBG[0] < 5:
            _DBG[0] += 1
            print(f"[cursor-hide-dbg] 异常 {e}", flush=True)
        return False


def force_show_cursor():
    try:
        while _HIDE_N[0] > 0:
            _u32b.ShowCursor(True)
            _HIDE_N[0] -= 1
        _detach()
    except Exception:
        pass
