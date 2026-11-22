import time

_diag_cap = []
import os
_GRAB_LOG = os.environ.get('D2S_GRAB_LOG', '') not in ('', '0')
import numpy as np
import mss
from utils import OS_NAME, CAPTURE_TOOL, DRAW_CURSOR, FULLSCREEN_OUTPUT
from gpu.rtcom import (DXGI_ERROR_INVALID_CALL, DXGI_ERROR_DEVICE_REMOVED,
                        DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET,
                        DXGI_ERROR_ACCESS_LOST)
# 致命 HRESULT 的字符串形态（异常文本匹配；数值单源在 gpu/rtcom.py）
_DXGI_FATAL_STRS = (
    "access loss", "access lost", "access denied", "device lost",
) + tuple(f"{v:#010x}" for v in (
    DXGI_ERROR_INVALID_CALL, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET))




import win32gui
from ctypes import windll
# Enable DPI awareness to improve capture quality on high-resolution displays
try:
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    windll.user32.SetProcessDPIAware()  # Silently ignore failure to set DPI awareness


def get_window_client_bounds(hwnd):
    """窗口客户区的屏幕坐标 (left, top, w, h)。
    （原 DXCamera 分支内的同名 helper，2a 误随分支删除——DXGI 抓取器的
    窗口跟随路径仍在用，ruff F821 抓回。）"""
    rc = win32gui.GetClientRect(hwnd)
    if rc is None:
        raise Exception(f"Window not found {hwnd}")
    left, top, right, bottom = rc
    w = right - left
    h = bottom - top
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    return left, top, w, h


class _CursorComposite:
    """DXGI desktop duplication never contains the mouse cursor.

    The pointer lives on a separate hardware overlay plane; IDXGIOutputDuplication
    only reports it through GetFramePointerShape/GetMouse (which the Python
    wrapper does not expose). WGC used to fold it into the frame for us, so
    draw it back by hand: copy the destination pixels under the cursor into
    a DIB, DrawIconEx the live cursor over them (Windows applies the mask and
    the alpha), then copy the result back. Everything is cached, so the
    steady-state cost is two small memcpys per frame.
    """

    _SZ = 128
    _SHOWING = 0x0001
    _DI_NORMAL = 0x0003

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._u32 = ctypes.windll.user32
        self._gdi = ctypes.windll.gdi32

        from winstructs import CURSORINFO, ICONINFO

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER),
                        ("bmiColors", wintypes.DWORD * 3)]

        self._CURSORINFO = CURSORINFO
        self._ICONINFO = ICONINFO

        # Declare signatures explicitly: handles are pointer-sized, and the
        # ctypes defaults (int) truncate them on 64-bit Python.
        self._u32.GetDC.restype = wintypes.HDC
        self._u32.GetDC.argtypes = [wintypes.HWND]
        self._gdi.CreateCompatibleDC.restype = wintypes.HDC
        self._gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self._gdi.CreateDIBSection.restype = wintypes.HBITMAP
        self._gdi.CreateDIBSection.argtypes = [
            wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
        self._gdi.SelectObject.restype = wintypes.HGDIOBJ
        self._gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        self._gdi.DeleteObject.restype = wintypes.BOOL
        self._gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        self._gdi.GdiFlush.restype = wintypes.BOOL
        # 不设 argtypes：全进程共享的函数对象，会和别的模块的结构体类型冲突
        # 不要设 GetIconInfo.argtypes：与 cursor_hide.identify() 共用同一
        # 个 windll 函数对象，类型不匹配会抛 ArgumentError（见 viewer 同处）。
        self._u32.DrawIconEx.restype = wintypes.BOOL
        self._u32.DrawIconEx.argtypes = [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HANDLE,
            ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HANDLE, wintypes.UINT]

        hdc_screen = self._u32.GetDC(0)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self._SZ
        bmi.bmiHeader.biHeight = -self._SZ          # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0             # BI_RGB
        bits = ctypes.c_void_p()
        self._hbmp = self._gdi.CreateDIBSection(
            hdc_screen, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
        self._memdc = self._gdi.CreateCompatibleDC(hdc_screen)
        self._gdi.SelectObject(self._memdc, self._hbmp)
        self._dib = np.ctypeslib.as_array(
            ctypes.cast(bits, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self._SZ * self._SZ * 4,)).reshape(self._SZ, self._SZ, 4)
        self._ok = bool(self._hbmp and self._memdc)

    def info(self):
        """(visible, x, y, hotspot_x, hotspot_y) in screen coordinates."""
        ci = self._CURSORINFO()
        ci.cbSize = self._ct.sizeof(self._CURSORINFO)
        if not self._u32.GetCursorInfo(self._ct.byref(ci)):
            return None
        if not (ci.flags & self._SHOWING) or not ci.hCursor:
            return None
        ii = self._ICONINFO()
        hx = hy = 0
        if self._u32.GetIconInfo(self._ct.c_void_p(int(ci.hCursor or 0)),
                                 self._ct.byref(ii)):
            hx, hy = int(ii.xHotspot), int(ii.yHotspot)
            for h in (ii.hbmMask, ii.hbmColor):
                if h:
                    self._gdi.DeleteObject(self._ct.c_void_p(h))
        return ci.hCursor, int(ci.ptScreenPos.x), int(ci.ptScreenPos.y), hx, hy

    def paint(self, frame, origin_x, origin_y):
        """Draw the live cursor into `frame` (BGRA HxWx4, writable)."""
        if not self._ok:
            return frame
        info = self.info()
        if info is None:
            return frame
        hcursor, px, py, hx, hy = info
        h, w = frame.shape[:2]
        x, y = px - origin_x - hx, py - origin_y - hy   # DrawIconEx does not apply the hotspot
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + self._SZ), min(h, y + self._SZ)
        if x0 >= x1 or y0 >= y1:
            return frame
        sub = frame[y0:y1, x0:x1]
        self._dib[:y1 - y0, :x1 - x0] = sub
        self._u32.DrawIconEx(self._memdc, x - x0, y - y0, hcursor, 0, 0, 0, None,
                             self._DI_NORMAL)
        self._gdi.GdiFlush()
        sub[...] = self._dib[:y1 - y0, :x1 - x0]
        return frame

if CAPTURE_TOOL == "DesktopDuplication":
    class DesktopGrabber:
        """
        Desktop grabber using Windows DXGI Desktop Duplication API.
        Provides direct access to the desktop frame buffer with hardware acceleration.
        """
        def __init__(self, output_resolution=1080, fps=60, window_title=None,
                     capture_mode="Monitor", monitor_index=0):
            """
            Initialize the DXGI desktop grabber for either a window or monitor.

            Args:
                output_resolution (int): Output image height (used for scaling).
                fps (int): Frames per second (informational).
                window_title (str): Title of the window to capture (for Window mode).
                capture_mode (str): 'Window' to capture an app window, 'Monitor' to capture a screen.
                monitor_index (int): Monitor index to capture (0 = primary monitor).
            """
            try:
                from windows_capture import DxgiDuplicationSession
            except ImportError:
                raise RuntimeError(
                    "windows_capture module not found. "
                    "Install it with: pip install windows-capture"
                )

            self.scaled_height = output_resolution
            self.fps = fps
            self.monitor_index = monitor_index
            self.window_title = window_title
            self.capture_mode = capture_mode
            self.session = None
            self.last_frame = None
            self._frame_count = 0
            self.prev_rect = None
            self._mss = mss.mss()
            self.hwnd = None

            # Initialize based on capture mode
            # DXGI frames never contain the mouse cursor (it is a separate
            # overlay plane), so composite it ourselves. Only meaningful for
            # monitor capture, where the frame maps 1:1 to a screen region.
            self._cursor = None
            self._origin = (0, 0)

            if self.capture_mode == "Monitor":
                # For monitor capture, use the specified monitor index
                try:
                    self.session = DxgiDuplicationSession(monitor_index=monitor_index)
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to create Desktop Duplication session for monitor {monitor_index}: {e}")
                # 全屏输出时绝不在采集端合成光标：SBS 会把画面复制到
                # 左右两半，合成一次就变成两个（叠加的根源）；行交错下
                # 光标由系统负责，也不需要我们画。
                if DRAW_CURSOR and not FULLSCREEN_OUTPUT:
                    try:
                        self._cursor = _CursorComposite()
                        mon = self._mss.monitors[monitor_index]
                        self._origin = (mon["left"], mon["top"])
                    except Exception as e:
                        print(f"[capture] cursor overlay disabled ({e})")
                        self._cursor = None
            else:
                # For window capture, we need to get the window bounds
                self.hwnd = win32gui.FindWindow(None, self.window_title)
                if not self.hwnd:
                    raise RuntimeError(f"Window '{self.window_title}' not found")
                # Initialize with primary monitor for window capture
                try:
                    self.session = DxgiDuplicationSession(monitor_index=1)
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to create Desktop Duplication session: {e}")

        def _monitor_contains(self, mon, rect):
            """
            Check whether a rectangle is completely inside a monitor's bounds.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h
            mon_left, mon_top = mon['left'], mon['top']
            mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
            return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

        def _monitor_intersection_area(self, mon, rect):
            """
            Compute the area of overlap between a rectangle and a monitor.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h
            mon_left, mon_top = mon['left'], mon['top']
            mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
            inter_w = max(0, min(mon_right, right) - max(mon_left, left))
            inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
            return inter_w * inter_h

        def _choose_monitor_and_rect(self, rect):
            """
            Select the best monitor for the window and clamp the rectangle to fit.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h

            # Check if the window is fully inside any secondary monitor (index >= 1)
            for mon in self._mss.monitors[1:]:
                if self._monitor_contains(mon, rect):
                    return mon, rect

            # Find monitor with largest overlapping area
            best_mon, best_area = None, -1
            for mon in self._mss.monitors[1:]:
                area = self._monitor_intersection_area(mon, rect)
                if area > best_area:
                    best_area = area
                    best_mon = mon

            # Fallback to first non-primary monitor if no overlap
            if best_mon is None or best_area <= 0:
                best_mon = self._mss.monitors[1]

            # Clamp rectangle to monitor bounds
            mon_left, mon_top = best_mon['left'], best_mon['top']
            mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
            new_left = max(left, mon_left)
            new_top = max(top, mon_top)
            new_right = min(right, mon_right)
            new_bottom = min(bottom, mon_bottom)
            new_w = max(0, new_right - new_left)
            new_h = max(0, new_bottom - new_top)

            # Default to full monitor if clamping results in empty area
            if new_w == 0 or new_h == 0:
                return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

            return best_mon, (new_left, new_top, new_w, new_h)

        def _get_monitor_for_window(self):
            """
            Get the monitor index that the window is primarily on.
            """
            try:
                bounds = get_window_client_bounds(self.hwnd)
                if bounds is None:
                    return 0

                _, rect = self._choose_monitor_and_rect(bounds)
                left, top, w, h = rect

                # Find which monitor this rectangle is on
                for idx, mon in enumerate(self._mss.monitors[1:], 1):
                    if self._monitor_contains(mon, rect):
                        return idx - 1  # Convert back to DXGI monitor index

                return 0  # Default to primary monitor
            except Exception:
                return 0

        def _ensure_session_matches_window(self):
            """
            Ensure the DXGI session is on the correct monitor for window capture.
            """
            if self.capture_mode == "Monitor":
                return

            try:
                bounds = get_window_client_bounds(self.hwnd)
                if bounds is None:
                    self.prev_rect = None
                    return

                if bounds == self.prev_rect:
                    return

                self.prev_rect = bounds

                # Determine which monitor the window is on
                monitor_idx = self._get_monitor_for_window()

                # Switch to that monitor if needed
                if self.session and monitor_idx != self.monitor_index:
                    try:
                        self.session.switch_monitor(monitor_idx)
                        self.monitor_index = monitor_idx
                    except Exception as e:
                        # If switch fails, try to recreate the session
                        try:
                            self.session.recreate()
                        except Exception:
                            pass
            except Exception:
                pass

        def _deliver(self, frame, repeat=False):
            """Return a frame the consumer may keep: with the cursor drawn in
            when DXGI gave us a cursor-free image.

            On a repeat of a frame that was already painted, with an unmoved
            pointer, the previously painted array is returned as-is: that
            skips an 8 MB copy per frame, which on a static desktop is the
            entire cost of the capture stage.
            """
            if self._cursor is None:
                return frame
            if repeat and getattr(self, "_last_painted_frame", None) is not None:
                try:
                    info = self._cursor.info()
                except Exception:
                    info = None
                if info == getattr(self, "_last_cursor_info", "unset"):
                    return self._last_painted_frame
            out = frame.copy()
            out = self._cursor.paint(out, self._origin[0], self._origin[1])
            try:
                self._last_cursor_info = self._cursor.info()
            except Exception:
                self._last_cursor_info = None
            self._last_painted_frame = out
            return out

        def grab(self):
            """
            Capture a single frame from the DXGI desktop duplication session.

            Returns:
                tuple: (image_array, scaled_height, is_new). is_new 为 False
                表示这是 repeat 路径原样返回的上一次交付对象（静态桌面 +
                光标未动）—— 下游据此跳过重复处理。真新帧与"光标动了需要
                重绘"两种情况都是新数组对象。
            """
            if self.session is None:
                raise RuntimeError("DXGI session is not initialized")

            # For window capture, ensure we're capturing from the right monitor
            if self.capture_mode != "Monitor":
                self._ensure_session_matches_window()

            try:
                # Attempt to acquire a frame with timeout based on FPS
                # Short timeout: the old 16 ms floor meant every grab
                    # on a static desktop cost 16 ms + copy, capping the
                    # pipeline at ~38 fps and letting stale frames pile up.
                _t0 = time.perf_counter()
                # Never block inside acquire: the native call adds ~15 ms on
                # top of whatever timeout it is given (measured: 0 -> 0.01,
                # 2 -> 15.5, 6 -> 15.5, 16 -> 30.8, 33 -> 46.2 ms). That
                # overhead alone capped the pipeline at ~38 fps and let two
                # stale frames pile up in the queues. Poll with 0 instead and
                # pace the "nothing new" case ourselves.
                frame = self.session.acquire_frame(timeout_ms=0)
                # 会话健康戳：acquire 正常应答（无论有没有新帧）都算活着；
                # 死会话走异常路径不会更新 -> main 的采集看门狗据此重建。
                self._acq_ok_ts = time.perf_counter()

                _t1 = time.perf_counter()
                if frame is not None:
                    # to_numpy() has no `copy` kwarg and returns a zero-copy
                    # view backed by the native frame (BGRA HxWx4); take an
                    # owned copy before the frame object is recycled, and
                    # cache it for the no-update fallback below.
                    image_rgb = np.array(frame.to_numpy())
                    _t2 = time.perf_counter()
                    self.last_frame = image_rgb
                    self._frame_count += 1
                    _out = self._deliver(image_rgb)
                    self._last_delivered = _out
                    _t3 = time.perf_counter()
                    if _GRAB_LOG:
                        _diag_cap.append((_t1 - _t0, _t2 - _t1, _t3 - _t2))
                    if _GRAB_LOG and len(_diag_cap) >= 60:
                        import statistics as _st
                        print(f"[grab] acquire {_st.median([d[0] for d in _diag_cap])*1000:.1f}ms | "
                            f"to_numpy {_st.median([d[1] for d in _diag_cap])*1000:.1f}ms | "
                            f"deliver {_st.median([d[2] for d in _diag_cap])*1000:.1f}ms", flush=True)
                        _diag_cap.clear()
                    return _out, self.scaled_height, True
                else:
                    # No new frame available, return last cached frame if available
                    if self.last_frame is not None:
                        time.sleep(0.004)      # ~200 Hz poll, negligible cost
                        _out = self._deliver(self.last_frame, repeat=True)
                        # repeat+光标未动 → _deliver 返回上次的同一对象：
                        # 内容没变，标 not-new 让下游休眠（省掉静态桌面
                        # 上 60~250 次/秒的重复推理）。
                        is_new = _out is not getattr(self, "_last_delivered", None)
                        self._last_delivered = _out
                        return _out, self.scaled_height, is_new
                    else:
                        # Return a black frame as fallback
                        return np.zeros((self.scaled_height, int(self.scaled_height * 16/9), 3), dtype=np.uint8), self.scaled_height, True

            except RuntimeError as e:
                # A DXGI duplication never recovers on its own once it has
                # been invalidated (display mode change, device reset, lock
                # screen): the only fix is to recreate it. Match the HRESULTs
                # as well as the English keywords - this Windows is localized,
                # so the message text is Chinese and the old keyword check
                # never matched. That failure mode kept retrying forever at
                # full rate: the picture froze on the last good frame (and
                # the log grew by hundreds of thousands of lines).
                _msg = str(e).lower()
                _fatal = any(x in _msg for x in _DXGI_FATAL_STRS)
                if _fatal:
                    now = time.time()
                    # recreate at most once a second: a burst of failures
                    # otherwise rebuilds the session dozens of times a second
                    if now - getattr(self, "_last_recreate_ts", 0.0) > 1.0:
                        self._last_recreate_ts = now
                        try:
                            self.session.recreate()
                            frame = self.session.acquire_frame(timeout_ms=50)
                            if frame is not None:
                                image_rgb = np.array(frame.to_numpy())
                                self.last_frame = image_rgb
                                self._frame_count += 1
                                return self._deliver(image_rgb), self.scaled_heigh, True
                        except Exception as recreate_error:
                            print(f"[capture] DXGI recreate failed: {recreate_error}")
                    # keep the pipeline alive on the cached frame meanwhile
                    if self.last_frame is not None:
                        return self._deliver(self.last_frame), self.scaled_heigh, False
                    return (np.zeros((self.scaled_height,
                                      int(self.scaled_height * 16 / 9), 3),
                                     dtype=np.uint8), self.scaled_height, True)
                raise

        def stop(self):
            """
            Clean up and release the Desktop Duplication session.
            """
            if self.session is not None:
                try:
                    # Try to close the session if it has a close method
                    if hasattr(self.session, 'close'):
                        self.session.close()
                except Exception:
                    pass
                finally:
                    self.session = None

            self.last_frame = None
            if self._mss:
                try:
                    self._mss.close()
                except Exception:
                    pass

