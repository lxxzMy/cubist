# -*- coding: utf-8 -*-
"""平台探测（显示器/刷新率/设备）与 WDA 窗口采集排除。

从 utils.py 拆出（2026-11-16）。导入副作用与原 utils 相同：
SetProcessDpiAwareness(2) 仍在导入期执行（utils 顶部导入本模块）。
"""
import os
import platform

OS_NAME = platform.system()

def get_font_type(os_name=None):
    os_name = OS_NAME if os_name is None else os_name
    if os_name == "Darwin":
        return "Verdana.ttf"
    elif os_name == "Windows": 
        return "verdana.ttf"
    elif os_name == "Linux":
        try:
            return "/usr/share/fonts/truetype/freefont/FreeSans.ttf" # fix for Ubuntu
        except Exception:
            return "Verdana.ttf"
    else:
        return "Verdana.ttf"


def get_fps(window_title="", monitor_index=None):
    """Return monitor refresh rate for the target monitor.
    If window_title is set, finds the monitor containing that window.
    If monitor_index is set (mss 1-based), uses that monitor directly.
    Falls back to primary monitor. Returns 60 if detection fails."""
    try:
        return _get_fps_windows(window_title, monitor_index)
    except Exception:
        return 60


def get_monitor_size(monitor_index=None):
    """Return (width, height) for an mss monitor index, falling back to primary."""
    try:
        import mss
        with mss.mss() as sct:
            if monitor_index is None or monitor_index <= 0 or monitor_index >= len(sct.monitors):
                monitor_index = 1
            mon = sct.monitors[monitor_index]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 3840, 2160


def compute_output_resolution(setting_value, display_mode, input_monitor_index,
                              stereo_monitor_index=None, run_mode=None):
    """Compute the source processing height used before depth inference."""
    try:
        if isinstance(setting_value, str):
            value = setting_value.strip()
            if value and value.lower() != "auto":
                parsed = int(value)
                if parsed > 0:
                    return parsed
        elif setting_value:
            parsed = int(setting_value)
            if parsed > 0:
                return parsed
    except (TypeError, ValueError):
        pass

    # Only output modes that own an actual desktop/stereo viewer window should
    # derive Auto from monitor geometry. OpenXR uses the native captured frame.
    auto_compute_modes = {"Local Viewer", "3D Monitor", "RTMP Streamer"}
    if run_mode not in auto_compute_modes:
        return 8640  # high no-resize sentinel for process()

    monitor_index = stereo_monitor_index or input_monitor_index
    _, out_h = get_monitor_size(monitor_index)
    if display_mode == "Full-TAB":
        out_h = max(1, out_h // 2)
    return max(2, (int(out_h) // 2) * 2)


def _get_device_name_from_mss_monitor(monitor_index):
    """Map mss monitor index (1-based) to win32api device name by matching rects."""
    import win32api
    import mss
    with mss.mss() as sct:
        if monitor_index is None or monitor_index >= len(sct.monitors):
            monitor_index = 1
        target = sct.monitors[monitor_index]
        tl, tt, tr, tb = target['left'], target['top'], target['left'] + target['width'], target['top'] + target['height']

    monitors = win32api.EnumDisplayMonitors()
    for hmon, hdc, rect in monitors:
        if rect[0] == tl and rect[1] == tt:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    # Fallback: match by overlap
    for hmon, hdc, rect in monitors:
        if rect[0] <= tl < rect[2] and rect[1] <= tt < rect[3]:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    return win32api.EnumDisplayDevices(None, 0).DeviceName


def _get_fps_windows(window_title, monitor_index):
    import win32api
    import win32gui
    import mss

    device_name = None

    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            r = win32gui.GetWindowRect(hwnd)
            wx, wy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
            with mss.mss() as sct:
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue
                    if mon['left'] <= wx < mon['left'] + mon['width'] and mon['top'] <= wy < mon['top'] + mon['height']:
                        device_name = _get_device_name_from_mss_monitor(i)
                        break

    if device_name is None and monitor_index is not None:
        try:
            device_name = _get_device_name_from_mss_monitor(monitor_index)
        except Exception:
            pass

    if device_name is None:
        try:
            device_name = win32api.EnumDisplayDevices(None, 0).DeviceName
        except Exception:
            return 60

    try:
        settings = win32api.EnumDisplaySettings(device_name, -1)
        return settings.DisplayFrequency
    except Exception:
        return 60


def _get_display_id_for_window_macos(window_title):
    """Find the CGDirectDisplayID that contains the given window."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            CGGetOnlineDisplayList,
            CGDisplayBounds,
        )
    except ImportError:
        return None

    info = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    if not info:
        return None

    max_displays = 16
    display_ids, count = CGGetOnlineDisplayList(max_displays, None, None)
    display_bounds = {}
    for d_id in display_ids[:count]:
        bounds = CGDisplayBounds(d_id)
        display_bounds[d_id] = bounds

    for win_dict in info:
        name = win_dict.get('kCGWindowName', '')
        if window_title in name:
            bounds = win_dict.get('kCGWindowBounds', {})
            wx = bounds.get('X', 0) + bounds.get('Width', 0) / 2
            wy = bounds.get('Y', 0) + bounds.get('Height', 0) / 2
            for d_id, db in display_bounds.items():
                if (db.origin.x <= wx < db.origin.x + db.size.width and
                    db.origin.y <= wy < db.origin.y + db.size.height):
                    return d_id
            return display_ids[0] if count > 0 else None
    return None


def crop_icon(icon_img):
    """Crop to make icon larger by cropping for Windows"""
    if OS_NAME == "Windows":
        icon_img = icon_img.convert("RGBA")
        bbox = icon_img.getbbox()
        icon_img = icon_img.crop(bbox)
    return icon_img


# WDA 窗口采集排除（原 utils 平台块，Windows-only 产品化）
if OS_NAME == "Windows":
    import ctypes, glfw

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    WDA_EXCLUDEFROMCAPTURE = 0x00000011   # Windows 10 2004+


    def hide_window_from_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        print("StereoWindow is now hidden from screen capture.")

    def show_window_in_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, 0)
        print("StereoWindow is now visible to screen capture.")

def _gpu_name():
    try:
        import subprocess as _sp
        out = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        import wmi
        cards = wmi.WMI().Win32_VideoController()
        if cards:
            return cards[0].Name
    except Exception:
        pass
    return "GPU"


def get_device(index=0):
    try:
        import onnxruntime as _ort
        providers = _ort.get_available_providers()
        if "MIGraphXExecutionProvider" in providers:
            return "rocm", f"Using ROCm device: {_gpu_name()}"
        if "CUDAExecutionProvider" in providers:
            return "cuda", f"Using CUDA device: {_gpu_name()}"
    except Exception:
        pass
    return "cpu", "Using CPU device"
