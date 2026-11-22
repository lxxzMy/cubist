"""系统托盘：让"双击启动"有可见反馈与入口。

为什么需要它：pythonw 启动时没有控制台，程序默认又是 2D 原生显示，
用户双击后看不到任何东西 = "没反应"。托盘图标至少能证明程序活着，
并提供「打开设置 / 切换 2D·3D / 退出」三个入口。

实现用 pywin32（包里已带）：隐藏窗口 + Shell_NotifyIcon + 消息泵，
运行在独立线程，双击图标打开设置面板。
"""
import os
import subprocess
import sys
import threading

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ICON = os.path.join(_APP_DIR, "icon.ico")
_TITLE = "立体派 Cubist"
_WM_TRAY = 0x0400 + 20          # WM_APP + 20
_CMD_SETTINGS, _CMD_TOGGLE, _CMD_QUIT, _CMD_FIXCUR = 1001, 1002, 1003, 1004


def _pythonw():
    exe = sys.executable or ""
    if exe.lower().endswith("python.exe"):
        cand = exe[:-len("python.exe")] + "pythonw.exe"
        if os.path.exists(cand):
            return cand
    return exe


class _Tray:
    def __init__(self, mode_file=None, on_quit=None):
        self.mode_file = mode_file
        self.on_quit = on_quit
        self._hwnd = None
        self._thread = None

    # ---------- Win32 ----------
    def _open_settings(self, *_a):
        try:
            subprocess.Popen([_pythonw(), "-m", "settings_dialog"],
                             cwd=_APP_DIR, close_fds=True)
        except Exception as e:
            print(f"[tray] 打开设置失败: {e}", flush=True)

    def _toggle(self, *_a):
        try:
            if not self.mode_file:
                return
            cur = "2d"
            try:
                with open(self.mode_file, encoding="utf-8") as f:
                    cur = f.read().strip() or "2d"
            except OSError:
                pass
            new = "2d" if cur == "3d" else "3d"
            with open(self.mode_file, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"[tray] 切换 -> {new}", flush=True)
        except Exception as e:
            print(f"[tray] 切换失败: {e}", flush=True)

    def _quit(self, *_a):
        try:
            if self.on_quit:
                self.on_quit()
        except Exception:
            pass
        try:
            import win32gui
            win32gui.PostMessage(self._hwnd, 0x0010, 0, 0)   # WM_CLOSE
        except Exception:
            pass

    def _menu(self):
        import win32gui, win32con
        m = win32gui.CreatePopupMenu()
        win32gui.AppendMenu(m, win32con.MF_STRING, _CMD_SETTINGS, "打开设置")
        win32gui.AppendMenu(m, win32con.MF_STRING, _CMD_TOGGLE, "切换 2D / 3D")
        win32gui.AppendMenu(m, win32con.MF_STRING, _CMD_FIXCUR, "修复鼠标光标")
        win32gui.AppendMenu(m, win32con.MF_SEPARATOR, 0, "")
        win32gui.AppendMenu(m, win32con.MF_STRING, _CMD_QUIT, "退出")
        return m

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            return self._wndproc_inner(hwnd, msg, wparam, lparam)
        except Exception as e:
            try:
                import traceback as _tb
                print(f"[tray] 回调异常 {msg:#x}: {e}\n{_tb.format_exc()}", flush=True)
            except Exception:
                pass
            return 0

    def _wndproc_inner(self, hwnd, msg, wparam, lparam):
        import win32gui, win32con
        if msg == _WM_TRAY:
            if lparam in (win32con.WM_RBUTTONUP, win32con.WM_CONTEXTMENU):
                m = self._menu()
                try:
                    x, y = win32gui.GetCursorPos()
                    try:
                        win32gui.SetForegroundWindow(hwnd)
                    except Exception:
                        pass
                    win32gui.PostMessage(hwnd, 0x0000, 0, 0)      # WM_NULL
                    cmd = win32gui.TrackPopupMenu(
                        m, win32con.TPM_LEFTALIGN | win32con.TPM_RIGHTBUTTON,
                        x, y, 0, hwnd, None)
                    if cmd:
                        win32gui.PostMessage(hwnd, win32con.WM_COMMAND, cmd, 0)
                    win32gui.PostMessage(hwnd, 0x0000, 0, 0)
                finally:
                    try:
                        win32gui.DestroyMenu(m)
                    except Exception:
                        pass
            elif lparam == win32con.WM_LBUTTONDBLCLK:
                self._open_settings()
            return 0
        if msg == win32con.WM_COMMAND:
            cmd = wparam & 0xFFFF
            if cmd == _CMD_SETTINGS:
                self._open_settings()
            elif cmd == _CMD_TOGGLE:
                self._toggle()
            elif cmd == _CMD_FIXCUR:
                # 手动兜底：把系统光标表恢复成系统默认（3D 隐藏后万一残留）
                try:
                    import cursor_hide as _ch
                    _ch.restore()
                    print("[tray] 已恢复系统光标", flush=True)
                except Exception as e:
                    print(f"[tray] 恢复光标失败: {e}", flush=True)
            elif cmd == _CMD_QUIT:
                self._quit()
            return 0
        if msg == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        if msg == win32con.WM_DESTROY:
            try:
                import win32gui as w
                w.Shell_NotifyIcon(2, (hwnd, 0, 0, 0, 0))     # NIM_DELETE
            except Exception:
                pass
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _run(self):
        import win32gui, win32con, win32api
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "CubistTrayWnd"
        wc.lpfnWndProc = self._wndproc
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass
        self._hwnd = win32gui.CreateWindow(
            wc.lpszClassName, _TITLE, 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)

        hicon = 0
        try:
            hicon = win32gui.LoadImage(0, _ICON, win32con.IMAGE_ICON, 0, 0,
                                       win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE)
        except Exception:
            hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)

        flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
        nid = (self._hwnd, 0, flags, _WM_TRAY, hicon, _TITLE)
        win32gui.Shell_NotifyIcon(0, nid)                    # NIM_ADD
        print("[tray] 托盘图标已显示（双击=设置，右键=菜单）", flush=True)
        win32gui.PumpMessages()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tray", daemon=True)
        self._thread.start()


_INSTANCE = [None]


def stop():
    """在 os.execv 自重启前调用：先摘掉托盘图标，否则旧图标会残留
    （Shell_NotifyIcon 的注册不会随进程映像替换自动清理，要等鼠标划过
    资源管理器才回收 —— 表现为"每重启一次多一个图标"）。"""
    t = _INSTANCE[0]
    if t is None or not t._hwnd:
        return
    try:
        import win32gui
        win32gui.Shell_NotifyIcon(2, (t._hwnd, 0, 0, 0, 0, ""))   # NIM_DELETE
    except Exception:
        pass
    try:
        import win32gui
        win32gui.PostMessage(t._hwnd, 0x0010, 0, 0)               # WM_CLOSE
    except Exception:
        pass
    _INSTANCE[0] = None


def start(mode_file=None, on_quit=None):
    """启动托盘；pywin32 缺失或失败时静默跳过（不影响主功能）。"""
    try:
        import win32gui  # noqa: F401
    except Exception as e:
        print(f"[tray] 跳过（pywin32 不可用: {e}）", flush=True)
        return None
    t = _Tray(mode_file=mode_file, on_quit=on_quit)
    try:
        t.start()
        _INSTANCE[0] = t
        return t
    except Exception as e:
        print(f"[tray] 启动失败: {e}", flush=True)
        return None
