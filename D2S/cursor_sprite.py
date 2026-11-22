# -*- coding: utf-8 -*-
"""自绘光标 sprite（纯 Win32/GDI + PIL/cv2，零 GL 依赖）。

从 viewer.py 抽出（1.1-d3d V1，2026-11-21）：GL 与 D3D12 双渲染后端共用。
_CursorSprite 负责系统光标位图提取/自绘/缩放；_cursor_base_size 与
_real_cursor_glyph_height 提供尺寸归一化基准（viewer 与 D3D 窗口都用）。
"""
import ctypes
import os
import time

import cv2
import numpy as np


class _CursorInfoT(ctypes.Structure):
    """模块级 CURSORINFO：判据和 sprite 共用同一份，避免 ctypes 类型不匹配。"""
    _fields_ = [("cbSize", ctypes.c_ulong), ("flags", ctypes.c_ulong),
                ("hCursor", ctypes.c_void_p), ("pt", ctypes.c_long * 2)]


def _cursor_base_size():
    """用户在系统里设定的光标基准尺寸（像素）。

    GetSystemMetrics(SM_CYCURSOR) 永远返回 32/48，不反映"指针大小"设置；
    真正的值在 HKCU\\Control Panel\\Cursors\\CursorBaseSize（本机实测 112）。
    读不到才退回 SM_CYCURSOR。
    """
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Cursors")
        try:
            val, _ = winreg.QueryValueEx(key, "CursorBaseSize")
        finally:
            winreg.CloseKey(key)
        val = int(val)
        if 16 <= val <= 256:
            return val
    except Exception:
        pass
    try:
        import ctypes as _c
        return int(_c.windll.user32.GetSystemMetrics(14)) or 32
    except Exception:
        return 32


def _real_cursor_glyph_height(fallback):
    """量出系统光标"实际画出来的高度"。

    CursorBaseSize 是 .cur 位图的画布尺寸（本机 112），但箭头只占画布的
    一部分（实测约 60%）。我们自绘的图形是填满画布的，所以按 112 画会
    明显偏大 —— 用户实测"调到 0.6 才差不多"就是这个比例。

    做法：把当前系统光标用 DrawIconEx 画进内存 DC，量 alpha 包围盒高度。
    """
    try:
        import ctypes
        import ctypes.wintypes as wt
        import numpy as np

        u32 = ctypes.windll.user32
        g32 = ctypes.windll.gdi32

        class CI(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("flags", wt.DWORD),
                        ("hCursor", wt.HANDLE), ("pt", wt.POINT)]

        ci = CI()
        ci.cbSize = ctypes.sizeof(ci)
        if not u32.GetCursorInfo(ctypes.byref(ci)) or not ci.hCursor:
            return fallback
        hcur = ci.hCursor

        n = int(fallback)
        hdc_screen = u32.GetDC(0)
        hdc_mem = g32.CreateCompatibleDC(hdc_screen)
        hbmp = g32.CreateCompatibleBitmap(hdc_screen, n, n)
        old = g32.SelectObject(hdc_mem, hbmp)
        try:
            # 32bpp DIB，全 0（透明）
            class BMIH(ctypes.Structure):
                _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG),
                            ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                            ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                            ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                            ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                            ("biClrImportant", wt.DWORD)]
            bi = BMIH()
            bi.biSize = ctypes.sizeof(BMIH)
            bi.biWidth = n
            bi.biHeight = -n          # top-down
            bi.biPlanes = 1
            bi.biBitCount = 32
            bi.biCompression = 0      # BI_RGB
            bits = ctypes.c_void_p()
            hdib = g32.CreateDIBSection(hdc_mem, ctypes.byref(bi), 0,
                                        ctypes.byref(bits), None, 0)
            old2 = g32.SelectObject(hdc_mem, hdib)
            try:
                u32.DrawIconEx(hdc_mem, 0, 0, hcur, n, n, 0, None, 3)
                buf = (ctypes.c_ubyte * (n * n * 4)).from_address(bits.value)
                arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(n, n, 4)
                alpha = arr[..., 3]
                ys, xs = np.nonzero(alpha > 8)
                if len(ys):
                    return float(ys.max() - ys.min() + 1)
            finally:
                g32.SelectObject(hdc_mem, old2)
                g32.DeleteObject(ctypes.c_void_p(hdib))
        finally:
            g32.SelectObject(hdc_mem, old)
            g32.DeleteObject(ctypes.c_void_p(hbmp))
            g32.DeleteDC(hdc_mem)
            u32.ReleaseDC(0, hdc_screen)
    except Exception:
        pass
    return fallback


class _CursorSprite:
    """Mouse-cursor bitmap for the render-time overlay, at native resolution.

    DXGI capture never contains the pointer, and compositing it into the
    captured image makes it inherit the whole pipeline latency (~3 frames),
    which feels laggy. Drawing it after the stereo pass costs one render frame.

    Resolution: Windows cursor schemes (.cur) ship several images, and the
    pointer Windows draws at 112 px is the *native* 112 px bitmap from the
    file - not an upscaled 32 px one (upscaling it is what made the overlay
    look blurry). So the scheme files are loaded with LoadImage at the system
    cursor size and the live cursor is matched against them by comparing the
    small (32 px) rendering. Unmatched shapes fall back to an upscaled sprite.

    Colour: cursors are BGR in memory; the channels are swapped into RGBA.
    Alpha: drawn twice, on black and on white,
        black = C*a,  white = C*a + 255*(1-a)  ->  a = 1-(white-black)/255,
    which also handles the mask-based monochrome cursors correctly.
    """

    _SHOWING = 0x0001
    _DI_NORMAL = 0x0003
    _REFRESH = 0.25       # re-extract even when the handle is unchanged:
                          # Windows hands out the same HCURSOR while the shape
                          # underneath changes (app-drawn cursors), which left
                          # the overlay showing the previous shape
    _IMAGE_CURSOR = 2
    _LR_LOADFROMFILE = 0x0010
    _NAMES = ("Arrow", "Hand", "IBeam", "Wait", "AppStarting", "Crosshair",
              "Help", "No", "SizeAll", "SizeNS", "SizeWE", "SizeNWSE",
              "SizeNESW", "UpArrow", "NWPen")

    @staticmethod
    def _base_size():
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Control Panel\Cursors") as k:
                return max(16, min(256, int(winreg.QueryValueEx(k, "CursorBaseSize")[0])))
        except Exception:
            return 32

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._u32 = ctypes.windll.user32
        self._gdi = ctypes.windll.gdi32

        from winstructs import CURSORINFO, ICONINFO

        class BITMAP(ctypes.Structure):
            _fields_ = [("bmType", wintypes.LONG), ("bmWidth", wintypes.LONG),
                        ("bmHeight", wintypes.LONG), ("bmWidthBytes", wintypes.LONG),
                        ("bmPlanes", wintypes.WORD), ("bmBitsPixel", wintypes.WORD),
                        ("bmBits", ctypes.c_void_p)]

        class BIH(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class BI(ctypes.Structure):
            _fields_ = [("bmiHeader", BIH), ("bmiColors", wintypes.DWORD * 3)]

        self._CURSORINFO = CURSORINFO
        self._ICONINFO = ICONINFO
        self._BITMAP = BITMAP
        self._BI = BI

        # 不要设 GetCursorInfo.argtypes：windll 的函数对象全进程共享，
        # 设了以后别的模块用自己的 CURSORINFO 调用会抛 ArgumentError
        # （实测：判据每帧抛异常被 except 吞掉 → 永远不生效、每帧照画）
        # 不要设 GetIconInfo.argtypes：windll 的函数对象全进程共享，
        # cursor_hide.identify() 用自己那份 ICONINFO 调用会抛 ArgumentError，
        # 被它的 except 吞成 -1 → 看护线程每秒误判"被外部重置"（实测）。
        self._u32.DrawIconEx.argtypes = [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HANDLE,
            ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HANDLE, wintypes.UINT]
        self._u32.LoadImageW.restype = wintypes.HANDLE
        self._u32.LoadImageW.argtypes = [
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self._u32.DestroyCursor.argtypes = [wintypes.HANDLE]
        self._u32.GetDC.restype = wintypes.HDC
        self._u32.GetDC.argtypes = [wintypes.HWND]
        self._u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self._gdi.GetObjectW.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p]
        self._gdi.CreateCompatibleDC.restype = wintypes.HDC
        self._gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self._gdi.CreateDIBSection.restype = wintypes.HBITMAP
        self._gdi.CreateDIBSection.argtypes = [
            wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
        self._gdi.SelectObject.restype = wintypes.HGDIOBJ
        self._gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        self._gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        self._gdi.DeleteDC.argtypes = [wintypes.HDC]

        self._cache_handle = None
        self._cache_rgba = None
        self._cache_hotspot = (0, 0)
        self._last_build_ts = 0.0
        self._table = None
        self._base_px = self._base_size()

    # ---- bitmap plumbing ---------------------------------------------------
    def _dib(self, hdc, w, h):
        bi = self._BI()
        bi.bmiHeader.biSize = self._ct.sizeof(self._BI().bmiHeader)
        bi.bmiHeader.biWidth = w
        bi.bmiHeader.biHeight = -h
        bi.bmiHeader.biPlanes = 1
        bi.bmiHeader.biBitCount = 32
        bi.bmiHeader.biCompression = 0
        bits = self._ct.c_void_p()
        hbmp = self._gdi.CreateDIBSection(hdc, self._ct.byref(bi), 0,
                                          self._ct.byref(bits), None, 0)
        arr = np.ctypeslib.as_array(
            self._ct.cast(bits, self._ct.POINTER(self._ct.c_uint8)),
            shape=(w * h * 4,)).reshape(h, w, 4)
        return hbmp, arr

    def _extract(self, hcursor, size=None):
        """Render an HCURSOR into RGBA。

        size=None：按光标自身位图尺寸渲染（原始分辨率）
        size=N   ：让 DrawIconEx 按 N 像素渲染 —— 这一步由 Windows 自己做缩放，
                   和我们自己 cv2.resize 的结果不同（系统用的是它自己的滤波），
                   与屏幕上系统光标的观感一致。
        """
        ii = self._ICONINFO()
        # 句柄显式转 c_void_p：没有 argtypes 时 Python int 会按 32 位传，
        # 64 位句柄（0x10003 这类虽小，但约定上）不保险
        if not self._u32.GetIconInfo(self._ct.c_void_p(int(hcursor)), self._ct.byref(ii)):
            return None
        hdc = self._u32.GetDC(0)
        bmp = mem = None
        try:
            bm = self._BITMAP()
            src = ii.hbmColor if ii.hbmColor else ii.hbmMask
            if not self._gdi.GetObjectW(src, self._ct.sizeof(self._BITMAP),
                                        self._ct.byref(bm)):
                return None
            w, h = int(bm.bmWidth), int(bm.bmHeight)
            if not ii.hbmColor:
                h //= 2
            if w <= 0 or h <= 0 or w > 256 or h > 256:
                return None
            # 目标绘制尺寸：交给 DrawIconEx 缩放（Windows 自己的滤波）
            k = 1
            if size:
                k = max(1, int(round(float(size) / max(w, h))))
            dw, dh = w * k, h * k

            bmp, arr = self._dib(hdc, dw, dh)
            mem = self._gdi.CreateCompatibleDC(hdc)
            old = self._gdi.SelectObject(mem, bmp)
            shots = []
            for fill in (0, 255):
                arr[...] = fill
                self._u32.DrawIconEx(mem, 0, 0, hcursor, dw, dh, 0, None, self._DI_NORMAL)
                self._gdi.GdiFlush()
                shots.append(arr[..., :3].astype(np.float32))
            self._gdi.SelectObject(mem, old)

            black, white = shots
            alpha = np.clip(1.0 - (white - black).mean(axis=2) / 255.0, 0.0, 1.0)
            rgb = np.zeros_like(black)
            nz = alpha > 1e-3
            rgb[nz] = np.clip(black[nz] / alpha[nz][:, None], 0.0, 255.0)
            rgba = np.empty((dh, dw, 4), np.uint8)
            rgba[..., 0] = rgb[..., 2]
            rgba[..., 1] = rgb[..., 1]
            rgba[..., 2] = rgb[..., 0]
            rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
            return rgba, int(ii.xHotspot) * k, int(ii.yHotspot) * k
        finally:
            if bmp:
                self._gdi.DeleteObject(bmp)
            if mem:
                self._gdi.DeleteDC(mem)
            self._u32.ReleaseDC(0, hdc)
            for hb in (ii.hbmMask, ii.hbmColor):
                if hb:
                    self._gdi.DeleteObject(self._ct.c_void_p(hb))

    def _load_file(self, path, size):
        h = self._u32.LoadImageW(None, path, self._IMAGE_CURSOR, size, size,
                                 self._LR_LOADFROMFILE)
        if not h:
            return None
        try:
            return self._extract(h)
        finally:
            self._u32.DestroyCursor(h)

    def _build_table(self):
        """name -> (big_rgba, hotspot, small_rgba) for the active scheme."""
        import os
        import winreg
        table = []
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Cursors")
        except OSError:
            return table
        with key:
            for name in self._NAMES:
                try:
                    path = os.path.expandvars(winreg.QueryValueEx(key, name)[0])
                except OSError:
                    continue
                if not path or not os.path.exists(path):
                    continue
                small = self._load_file(path, 32)
                big = self._load_file(path, self._base_px)
                if small and big:
                    table.append((small[0], big, name))
        if table:
            print(f"[Main] cursor sprites: {len(table)} scheme shapes at "
                  f"{self._base_px}px")
        return table

    # 系统光标 ID → 名字（用于句柄缓存表）
    _IDC_IDS = ((32512, "Arrow"), (32513, "IBeam"), (32514, "Wait"),
                (32515, "Crosshair"), (32516, "UpArrow"), (32642, "SizeNWSE"),
                (32643, "SizeNESW"), (32644, "SizeWE"), (32645, "SizeNS"),
                (32646, "SizeAll"), (32648, "No"), (32649, "Hand"),
                (32650, "AppStarting"), (32651, "Help"))

    def _build_handle_map(self):
        """启动时（隐藏光标之前）把每个系统光标【句柄】和位图缓存起来。

        关键：应用（浏览器/编辑器）通常在启动时 LoadCursor 拿到句柄并缓存，
        之后 SetCursor(那个句柄)。即使我们把"系统光标表"换成了透明，
        这些应用手里的旧句柄仍指向原来的光标对象
        → GetCursorInfo 返回它 → 我们直接查这张表就知道形状（不用再猜）。
        """
        m = {}
        try:
            for cid, name in self._IDC_IDS:
                h = self._u32.LoadCursorW(None, cid)
                if not h:
                    continue
                try:
                    got = self._extract(h, self._base_px)
                except Exception:
                    got = None
                if got is None:
                    continue
                if self._is_blank(got[0]):                  # 空白光标，跳过
                    continue
                big, hx, hy = got
                try:
                    big = self._upscale(big, hx, hy)
                except Exception:
                    pass
                m[int(h)] = big
            if m:
                print(f"[Main] cursor handle map: {len(m)} 个句柄已缓存", flush=True)
            else:
                print("[Main] cursor handle map 为空（建表时系统光标已是空白？）",
                      flush=True)
        except Exception as e:
            print(f"[Main] cursor handle map 失败: {e}", flush=True)
        return m

    def _match(self, rgba):
        """Pick the scheme shape whose 32 px rendering matches this cursor."""
        if not self._table:
            return None
        best, best_err = None, None
        a = rgba[..., 3].astype(np.float32)
        for small, big, name in self._table:
            b = small[..., 3].astype(np.float32)
            if a.shape != b.shape:
                continue
            err = float(np.abs(a - b).mean())
            if best_err is None or err < best_err:
                best, best_err = (big, name), err
        if best is not None and best_err is not None and best_err < 12.0:
            return best[0]
        return None

    _ARROW_POLY = [(0, 0), (0, 16), (4, 12), (7, 19), (9, 18), (6, 11),
                   (12, 11)]   # 经典箭头轮廓（20 单位坐标系，尖在 0,0）

    def _self_arrow(self, size=32):
        """自绘抗锯齿箭头：4x 超采样多边形 → 盒式降采样。

        不依赖系统光标集 —— 用户把系统光标换成全透明后，任何
        LoadCursor/提取路径拿到的都是空位图，只有自绘可用。
        白底黑边经典配色，热点在尖端 (0,0)。
        """
        try:
            import cv2 as _cv2
            ss = 4                                   # 超采样倍数
            S = max(16, int(size)) * ss
            poly = np.array([(x * S / 20.0, y * S / 20.0) for x, y in self._ARROW_POLY],
                            np.float32)
            img = np.zeros((S, S, 4), np.uint8)
            _cv2.fillPoly(img, [np.round(poly).astype(np.int32)], (255, 255, 255, 255))
            _cv2.polylines(img, [np.round(poly).astype(np.int32)],
                           True, (0, 0, 0, 255), thickness=max(2, ss // 2))
            out = _cv2.resize(img, (S // ss, S // ss),
                              interpolation=_cv2.INTER_AREA)
            return out, 0, 0
        except Exception:
            return None

    _SELF_CACHE = {}

    def _self_shape(self, name, size=32):
        """按形状名自绘光标（4x 超采样抗锯齿，白芯黑边）。

        系统光标集全透明后形状位图无法提取，但形状【名】仍可知
        （cursor_hide 热点编码 / NCHITTEST 推断）——缺的只是画法。
        支持：Arrow/IBeam/SizeWE/SizeNS/SizeNWSE/SizeNESW/Crosshair，
        其余（Hand/Wait 等）回退 Arrow。
        """
        key = (name, size)
        if key in self._SELF_CACHE:
            return self._SELF_CACHE[key]
        import cv2 as _cv2
        ss = 4
        S = size * ss
        img = np.zeros((S, S, 4), np.uint8)
        C = float(size) / 2.0
        white, black = (255, 255, 255, 255), (0, 0, 0, 255)

        def L(p0, p1, w):
            cv2.line(img, (int(p0[0]*ss), int(p0[1]*ss)),
                     (int(p1[0]*ss), int(p1[1]*ss)), black, w + max(2, ss))
            cv2.line(img, (int(p0[0]*ss), int(p0[1]*ss)),
                     (int(p1[0]*ss), int(p1[1]*ss)), white, w, cv2.LINE_AA)

        def HEAD(tip, d, a):
            L2 = max(1e-3, (d[0]**2 + d[1]**2) ** 0.5)
            ux, uy = d[0]/L2, d[1]/L2
            px, py = -uy, ux
            t = (tip[0]*ss, tip[1]*ss)
            b1 = ((tip[0]-ux*a+px*a*0.55)*ss, (tip[1]-uy*a+py*a*0.55)*ss)
            b2 = ((tip[0]-ux*a-px*a*0.55)*ss, (tip[1]-uy*a-py*a*0.55)*ss)
            tri = np.array([t, b1, b2], np.float32)
            cv2.fillPoly(img, [np.round(tri).astype(np.int32)], black)
            inset = np.array([ux*ss*0.8, uy*ss*0.8])
            cv2.fillPoly(img, [np.round(tri - inset).astype(np.int32)], white)

        if name == "IBeam":
            L((C, 7), (C, 25), 2)
            L((C-3, 7), (C+3, 7), 2)
            L((C-3, 25), (C+3, 25), 2)
            hot = (C, 9)
        elif name == "SizeNS":
            L((C, 10), (C, 22), 2)
            HEAD((C, 7), (0, 1), 5); HEAD((C, 25), (0, -1), 5)
            hot = (C, C)
        elif name == "SizeWE":
            L((10, C), (22, C), 2)
            HEAD((7, C), (1, 0), 5); HEAD((25, C), (-1, 0), 5)
            hot = (C, C)
        elif name == "SizeNWSE":
            L((9, 9), (23, 23), 2)
            HEAD((7, 7), (1, 1), 5); HEAD((25, 25), (-1, -1), 5)
            hot = (C, C)
        elif name == "SizeNESW":
            L((23, 9), (9, 23), 2)
            HEAD((25, 7), (-1, 1), 5); HEAD((7, 25), (1, -1), 5)
            hot = (C, C)
        elif name == "Crosshair":
            L((C, 8), (C, 24), 1); L((8, C), (24, C), 1)
            hot = (C, C)
        else:
            poly = np.array([(x * S / 20.0, y * S / 20.0)
                             for x, y in self._ARROW_POLY], np.float32)
            cv2.fillPoly(img, [np.round(poly).astype(np.int32)], black)
            poly_in = (poly - C) * 0.78 + C
            cv2.fillPoly(img, [np.round(poly_in).astype(np.int32)], white)
            hot = (0, 0)

        out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        self._SELF_CACHE[key] = (out, int(hot[0]), int(hot[1]))
        return self._SELF_CACHE[key]

    @staticmethod
    def _is_blank(rgba):
        """判"这就是我们换上去的空白光标"。

        不能只判 alpha.sum()==0：空白光标在 mask 里留了 1 个"要画"的像素，
        DrawIconEx 两背景差分后 128x128 会得到 alpha_sum≈1806、>8 的像素 8 个；
        真形状是 alpha_sum>1e5、非透明像素>1000（实测 sprite 日志 1078~3634）。
        两个数量级的差距，用像素个数判最稳。
        """
        try:
            a = rgba[..., 3]
            return int((a > 8).sum()) < 32
        except Exception:
            return False

    def _arrow_fallback(self):
        """Standard arrow, used when the live cursor cannot be read back."""
        try:
            if self._table:
                for _small, big, name in self._table:
                    if name == "Arrow":
                        return big
            h = self._u32.LoadCursorW(None, 32512)      # IDC_ARROW
            got = self._extract(h) if h else None
            # 系统光标集是全透明（cursor_hide 或用户自设）时提取为空 →
            # 回退到自绘箭头（唯一不依赖系统光标集的来源）
            if got is not None and not self._is_blank(got[0]):
                return self._upscale(got[0], got[1], got[2])
            return self._self_arrow()
        except Exception:
            return self._self_arrow()

    def _upscale(self, rgba, hx, hy):
        # Scale to the system cursor size RELATIVE TO THIS BITMAP's width, not
        # to 32: app cursors (the browser magnifier among them) come at 48/64/112
        # px already, and dividing by a fixed 32 blew them up 2-3.5x on top of
        # their native size - which is what made their strokes look huge.
        w = int(rgba.shape[1])
        if w <= 0:
            return rgba, hx, hy
        scale = self._base_px / float(w)
        if scale <= 1.01 or scale > 8.0:
            return rgba, hx, hy
        try:
            import cv2
            size = (max(1, int(round(rgba.shape[1] * scale))),
                    max(1, int(round(rgba.shape[0] * scale))))
            return (cv2.resize(rgba, size, interpolation=cv2.INTER_LANCZOS4),
                    int(round(hx * scale)), int(round(hy * scale)))
        except Exception:
            return rgba, hx, hy

    # WM_NCHITTEST 返回值 → 光标形状
    _HT_SHAPE = {10: "SizeWE", 11: "SizeWE",          # HTLEFT / HTRIGHT
                 12: "SizeNS", 15: "SizeNS",          # HTTOP / HTBOTTOM
                 13: "SizeNWSE", 17: "SizeNWSE",      # HTTOPLEFT / HTBOTTOMRIGHT
                 14: "SizeNESW", 16: "SizeNESW"}      # HTTOPRIGHT / HTBOTTOMLEFT

    _TEXT_DLG_CLASSES = ("Edit", "RichEdit", "RichEdit20", "RichEdit50",
                         "Scintilla", "RICHEDIT60W")

    def _guess_shape(self, x, y):
        """隐藏系统光标后推断"这里本该是什么光标"。

        系统层面已读不到形状（热点/宽度都被规范化，实测），所以靠命中测试：
          WM_NCHITTEST 命中边框 → 对应双箭头；命中编辑控件 → I 型；其余箭头。
        """
        try:
            from ctypes import wintypes as _wt
            u = self._u32
            pt = _wt.POINT(int(x), int(y))
            hwnd = u.WindowFromPoint(pt)
            # 我们的全屏 overlay 是 topmost：WindowFromPoint 会命中它自己，
            # 顺着 Z 序往下找，直到拿到别的进程的窗口
            mypid = os.getpid()
            for _ in range(16):
                if not hwnd:
                    break
                pid = _wt.DWORD(0)
                u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != mypid:
                    break
                hwnd = u.GetWindow(hwnd, 2)      # GW_HWNDNEXT
            if not hwnd:
                return "Arrow"
            code = int(u.SendMessageW(hwnd, 0x0084, 0,      # WM_NCHITTEST
                                      (int(y) << 16) | (int(x) & 0xFFFF)))
            if code in self._HT_SHAPE:
                return self._HT_SHAPE[code]
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            cname = (cls.value or "")
            for t in self._TEXT_DLG_CLASSES:
                if cname.startswith(t):
                    return "IBeam"
        except Exception:
            pass
        return "Arrow"

    def _probe_app_cursor(self, x, y):
        """带节流的包装：100ms 内复用上次结果（跨进程 WM_SETCURSOR 很贵）。"""
        now = time.perf_counter()
        if now - getattr(self, "_probe_ts", 0.0) < 0.10:
            return getattr(self, "_probe_last", None)
        self._probe_ts = now
        got = self._probe_app_cursor_impl(x, y)
        self._probe_last = got
        return got

    def _probe_app_cursor_impl(self, x, y):
        """形状来源（按优先级）：

        ① 句柄缓存表：启动时（隐藏前）缓存了 14 个系统光标句柄→位图。
           应用自己缓存的句柄会一路传下来，GetCursorInfo 返回它 → 直接查表，
           零成本、且是真实形状。
        ② WM_SETCURSOR 探测：让上层窗口现设一次光标再读回（带内容判空）。
           可行原因：Chrome/编辑器这类程序带自己的光标资源，收到
           WM_SETCURSOR 会 SetCursor(自己的 HCURSOR)，GetCursorInfo 就能
           读到真实光标（不受我们替换系统光标表的影响）。
        """
        try:
            from ctypes import wintypes as _wt
            u = self._u32
            # 节流：WM_SETCURSOR 是跨进程同步消息，每帧发会卡主线程
            # （实测帧率从 60 掉到 30）。上面包装已做 100ms 节流。
            pt = _wt.POINT(int(x), int(y))
            hwnd = u.WindowFromPoint(pt)
            mypid = os.getpid()
            for _ in range(16):
                if not hwnd:
                    return None
                pid = _wt.DWORD(0)
                u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != mypid:
                    break
                hwnd = u.GetWindow(hwnd, 2)
            if not hwnd:
                return None
            # 转客户端坐标填 lParam（HIWORD = WM_MOUSEMOVE）
            cpt = _wt.POINT(int(x), int(y))
            u.ScreenToClient(hwnd, ctypes.byref(cpt))
            ci0 = self._CURSORINFO()
            ci0.cbSize = self._ct.sizeof(self._CURSORINFO)
            if not u.GetCursorInfo(self._ct.byref(ci0)) or not ci0.hCursor:
                return None
            # ① 句柄缓存表命中：应用沿用了隐藏前的旧句柄 → 直接用
            hmap = getattr(self, "_handle_map", None)
            if hmap is None:
                hmap = self._handle_map = self._build_handle_map()
            _hit = hmap.get(int(ci0.hCursor))
            if _hit is not None:
                return _hit
            lparam = ((cpt.y & 0xFFFF) << 16) | (cpt.x & 0xFFFF)
            u.SendMessageW(hwnd, 0x0020, hwnd, 1 | (0x0200 << 16))   # WM_SETCURSOR, HTCLIENT
            ci = self._CURSORINFO()
            ci.cbSize = self._ct.sizeof(self._CURSORINFO)
            if not u.GetCursorInfo(self._ct.byref(ci)) or not ci.hCursor:
                return None
            h = int(ci.hCursor)
            try:
                import cursor_hide as _ch
                if _ch.identify(h) != -1:      # 还是我们的空白光标 → 没拿到
                    return None
            except Exception:
                pass
            cache = getattr(self, "_app_cursor_cache", None)
            if cache is None:
                cache = self._app_cursor_cache = {}
            if h in cache:
                return cache[h]
            got = self._extract(ci.hCursor, self._base_px)
            if got is None:
                return None
            # 内容判空：我们自己的空白光标（或窗口给了 NULL 光标）提取出来
            # 是几乎全透明的；这种绝不能画（画出来就是"光标消失"）。
            if self._is_blank(got[0]):
                return None
            # 由 DrawIconEx 按系统尺寸渲染（Windows 自己的缩放，边缘清晰），
            # 不再用 cv2 插值二次放大。
            val = got
            if len(cache) > 32:
                cache.clear()
            cache[h] = val
            return val
        except Exception:
            return None

    def get(self, allow_probe=True):
        """-> (rgba, hotspot_x, hotspot_y, (x, y)) or None when hidden."""
        ci = self._CURSORINFO()
        ci.cbSize = self._ct.sizeof(self._CURSORINFO)
        if not self._u32.GetCursorInfo(self._ct.byref(ci)):
            return None
        # 系统光标已被替换成空白：此时 SHOWING 位不可靠，位置用 GetCursorPos
        if getattr(self, "_force_arrow", False):
            pt = self._ct.wintypes.POINT() if hasattr(self._ct, "wintypes") else None
            if pt is None:
                from ctypes import wintypes as _wt
                pt = _wt.POINT()
            if not self._u32.GetCursorPos(self._ct.byref(pt)):
                return None
            if self._table is None:
                self._table = self._build_table()
            # ⓪ 系统光标表未被替换时，句柄能直接 identify 出是哪个标准光标
            #    （箭头/I型/手型…）→ 用方案表里的高清位图，最准且零探测成本。
            try:
                import cursor_hide as _ch0
                ci_now = self._CURSORINFO()
                ci_now.cbSize = self._ct.sizeof(self._CURSORINFO)
                if self._u32.GetCursorInfo(self._ct.byref(ci_now)) and ci_now.hCursor:
                    cid = _ch0.identify(ci_now.hCursor)
                    nm = _ch0.ID_NAME.get(cid) if cid != -1 else None
                    if nm:
                        if self._table is None:
                            self._table = self._build_table()
                        picked = next((b for _s, b, n in self._table if n == nm), None)
                        if picked:
                            return (picked[0], picked[1], picked[2], (pt.x, pt.y))
            except Exception:
                pass
            # ① 先让上层窗口"自己把光标设出来"：发 WM_SETCURSOR，
            #    应用（浏览器/编辑器等自带光标资源的程序）会 SetCursor，
            #    这时 GetCursorInfo 拿到的是【它自己的光标】，可以提取位图。
            got_app = (self._probe_app_cursor(pt.x, pt.y)
                       if allow_probe else None)
            if got_app is not None:
                self._last_good = (got_app[0], got_app[1], got_app[2])
                return (got_app[0], got_app[1], got_app[2], (pt.x, pt.y))
            # 静止期探测被门控：沿用上一次的真实形状，别退回猜箭头
            _lg = getattr(self, "_last_good", None)
            if _lg is not None and not allow_probe:
                return (_lg[0], _lg[1], _lg[2], (pt.x, pt.y))
            # ② 拿不到（经典控件用系统光标表，已被我们替换成空白）→
            #    命中测试推断：NCHITTEST 命中边框 → 双箭头；编辑控件 → I 型
            name = self._guess_shape(pt.x, pt.y)
            picked = next((b for _s, b, n in (self._table or []) if n == name), None)
            if picked is None:
                picked = next((b for _s, b, n in (self._table or []) if n == "Arrow"), None)
            if picked:
                return (picked[0], picked[1], picked[2], (pt.x, pt.y))
            return None
        if not (ci.flags & self._SHOWING) or not ci.hCursor:
            return None
        if False:
            if self._table is None:
                self._table = self._build_table()
            name = None
            try:
                import cursor_hide as _ch
                cid = _ch.identify(ci.hCursor)
                name = _ch.ID_NAME.get(cid)
            except Exception:
                pass
            want = name or "Arrow"
            picked = next((b for _s, b, n in (self._table or []) if n == want), None)
            if picked is None:
                picked = next((b for _s, b, n in (self._table or []) if n == "Arrow"), None)
            if picked:
                return (picked[0], picked[1], picked[2],
                        (ci.ptScreenPos.x, ci.ptScreenPos.y))
        now = time.time()
        stale = (now - self._last_build_ts) > self._REFRESH
        if ci.hCursor != self._cache_handle or self._cache_rgba is None or stale:
            if self._table is None:
                self._table = self._build_table()
            try:
                small = self._extract(ci.hCursor)
                if small is not None:
                    big = self._match(small[0])
                    if big is not None:
                        self._cache_rgba = big[0]
                        self._cache_hotspot = (big[1], big[2])
                    else:
                        # Shape not in the scheme: it is an application-provided
                        # cursor (the browser magnifier, for instance). Windows
                        # draws those at their own pixel size instead of blowing
                        # them up to the system cursor size, so do the same -
                        # upscaling them is what made their thin outlines look
                        # enormous.
                        self._cache_rgba = small[0]
                        self._cache_hotspot = (small[1], small[2])
                else:
                    # Some cursors (the magnifier among them) cannot be read
                    # back through GetIconInfo. Drawing nothing made the pointer
                    # vanish, so fall back to the scheme's arrow.
                    fb = self._arrow_fallback()
                    if fb is not None:
                        self._cache_rgba = fb[0]
                        self._cache_hotspot = (fb[1], fb[2])
            except Exception as _e:
                if not getattr(self, "_err_logged", False):
                    self._err_logged = True
                    print(f"[cursor] sprite build failed: {_e}")
            self._cache_handle = ci.hCursor
            self._last_build_ts = now
            if self._cache_rgba is None:
                return None
        return (self._cache_rgba, self._cache_hotspot[0], self._cache_hotspot[1],
                (ci.ptScreenPos.x, ci.ptScreenPos.y))


