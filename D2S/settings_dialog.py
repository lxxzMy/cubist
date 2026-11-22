"""立体派 Cubist 设置窗口 —— 自绘现代深色 UI。

全部控件自绘（Canvas），不依赖 ttk 原生外观：
  _Switch 手机式开关 / _Button 圆角按钮 / _Slider 细滑轨 / _Stepper 步进器 / _Dropdown 下拉
"""
import json
import os
import sys
import tempfile
import time
import tkinter as tk
from tkinter import messagebox
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _stdio   # 零依赖、无副作用（见 _stdio.py：必须先于 utils 导入）
_stdio.ensure()

import utils  # noqa: E402

SETTINGS_PATH = os.path.join(HERE, "settings.yaml")
# ---------------- 2D / 3D 切换 ----------------
from mode_bus import MODE_FILE  # 路径单源（本模块无导入副作用）

# ---------------- 设置窗口单实例：已开着就直接激活 ----------------
# 托盘双击、启动器、再次双击 exe 都会起一个 settings_dialog；不加守卫就会
# 叠出好几个面板。命名 Mutex 判重，第二个实例把已有窗口拉到前台后退出。
_MUTEX_NAME = "Local\\Cubist-D2S-Settings"
_MUTEX_HANDLE = [None]


def _find_existing_dialog():
    """找另一个进程里的设置窗口（TkTopLevel + 标题含"立体派"）。"""
    import ctypes
    from ctypes import wintypes as _wt
    u32 = ctypes.windll.user32
    hits = []

    @ctypes.WINFUNCTYPE(_wt.BOOL, _wt.HWND, _wt.LPARAM)
    def _cb(hwnd, _lp):
        try:
            if not u32.IsWindowVisible(hwnd):
                return True
            cls = ctypes.create_unicode_buffer(64)
            u32.GetClassNameW(hwnd, cls, 64)
            if cls.value != "TkTopLevel":
                return True
            t = ctypes.create_unicode_buffer(128)
            u32.GetWindowTextW(hwnd, t, 128)
            if "立体派" in t.value:
                hits.append(hwnd)
        except Exception:
            pass
        return True

    u32.EnumWindows(_cb, 0)
    return hits[0] if hits else None


def _force_foreground(hwnd):
    """把窗口提到前台：后台进程直接 SetForegroundWindow 常被系统拒绝，
    用"置顶→取消置顶 + AttachThreadInput"两招绕过前台锁。"""
    import ctypes
    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    _SWP = 0x0001 | 0x0002 | 0x0040        # NOSIZE | NOMOVE | SHOWWINDOW
    try:
        u32.ShowWindow(hwnd, 9)            # SW_RESTORE
    except Exception:
        pass
    try:
        u32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, _SWP)   # HWND_TOPMOST
        u32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, _SWP)   # HWND_NOTOPMOST
    except Exception:
        pass
    fg = u32.GetForegroundWindow()
    t_fg = u32.GetWindowThreadProcessId(fg, None) if fg else 0
    t_me = k32.GetCurrentThreadId()
    attached = False
    if t_fg and t_fg != t_me:
        try:
            attached = bool(u32.AttachThreadInput(t_me, t_fg, True))
        except Exception:
            attached = False
    try:
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
        u32.SetFocus(hwnd)
    except Exception:
        pass
    if attached:
        try:
            u32.AttachThreadInput(t_me, t_fg, False)
        except Exception:
            pass


def claim_single_instance():
    """True = 可以继续创建窗口；False = 已有实例（已把它拉到前台，自己退出）。"""
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_wchar_p]
        h = k32.CreateMutexW(None, 0, _MUTEX_NAME)
        if not h:
            return True                    # 拿不到锁就放行，别把人挡在门外
        _MUTEX_HANDLE[0] = h               # 常驻：进程退出才释放
        if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
            return True
        # 已有一个设置窗口：它可能刚起、窗口还没建出来，短暂重试
        for _ in range(15):
            w = _find_existing_dialog()
            if w:
                _force_foreground(w)
                print("[settings] 已有设置窗口，已激活", flush=True)
                return False
            time.sleep(0.2)
        print("[settings] 已有实例但没找到窗口（可能正在退出）", flush=True)
        return False
    except Exception as e:
        print(f"[settings] 单实例检查失败（放行）: {e}", flush=True)
        return True


def current_mode():
    try:
        with open(MODE_FILE, encoding="utf-8") as f:
            return "3d" if f.read().strip() == "3d" else "2d"
    except OSError:
        return "2d"


def set_mode(mode):
    try:
        os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True)
        with open(MODE_FILE, "w", encoding="utf-8") as f:
            f.write("3d" if mode == "3d" else "2d")
        return True
    except OSError:
        return False


LOG_DIR = os.path.join(HERE, "logs")
PARAMS_REQ = os.path.join(LOG_DIR, "params.request")
# ---------------- 跟随主程序生命周期 ----------------
def _main_pid_alive():
    """主程序是否还在跑。None = 没有 pid 文件（判定不了，不跟随）。"""
    try:
        pid_file = os.path.join(HERE, "logs", "d2s.pid")
        if not os.path.exists(pid_file):
            return None
        with open(pid_file, encoding="utf-8") as f:
            pid = int(f.read().strip())
        if pid <= 0:
            return None
    except Exception:
        return None
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong(0)
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259                     # STILL_ACTIVE
    except Exception:
        return True


# ---------------- 开机自启动 ----------------
def _startup_dir():
    return os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                        "Start Menu", "Programs", "Startup")


def _startup_file():
    return os.path.join(_startup_dir(), "立体派Cubist.cmd")


def autostart_enabled():
    try:
        return os.path.exists(_startup_file())
    except Exception:
        return False


def _layout_paths():
    """(codedir, pythonw, runtime)——兼容两种布局：

    便携发布包（Cubist-1.x）：<root>/app + <root>/python + <root>/runtime；
    开发树：<root>/D2S + <root>/D2S/python3 + <root>/runtime。
    返回 None = 找不到（不能写坏启动脚本）。"""
    root = os.path.dirname(HERE)
    for codedir, pyw, rt in (
            (os.path.join(root, "app"),
             os.path.join(root, "python", "pythonw.exe"),
             os.path.join(root, "runtime")),
            (os.path.join(root, "D2S"),
             os.path.join(root, "D2S", "python3", "pythonw.exe"),
             os.path.join(root, "runtime"))):
        if os.path.isdir(codedir) and os.path.isfile(pyw):
            return codedir, pyw, rt
    return None


def set_autostart(on: bool):
    """写/删启动文件夹里的启动脚本。用 .cmd 而非快捷方式：不依赖 COM，路径可见可查。

    直接以 pythonw -m main 启动（托盘待命），不走 立体派.exe——启动器
    会顺带弹设置面板，开机自启动不该弹。"""
    f = _startup_file()
    if not on:
        try:
            if os.path.exists(f):
                os.remove(f)
            return True
        except Exception:
            return False
    lay = _layout_paths()
    if lay is None:
        return False
    codedir, pyw, runtime = lay
    try:
        os.makedirs(_startup_dir(), exist_ok=True)
        with open(f, "w", encoding="gbk", newline="") as fh:
            fh.write("@echo off\r\n")
            fh.write(f'cd /d "{codedir}"\r\n')
            fh.write(f'set PATH={runtime};%PATH%\r\n')
            fh.write(f'start "" "{pyw}" -m main\r\n')
        return True
    except Exception:
        return False


RESTART_REQ = os.path.join(LOG_DIR, "restart.request")
ICON_PATH = os.path.join(HERE, "icon.ico")

# 显示器 EDID 名称缓存（_edid 内部拉 PowerShell，显示器列表 2s 轮询不能重跑）
_EDID_CACHE = [None]

LIVE_KEYS = ("Parallax (px)", "Convergence Bias",
             "Anti-aliasing",
             "Show FPS", "VSync", "Interleave Phase")

BG = "#0B1220"
CARD = "#141E30"
CARD_HI = "#1B2942"
TRACK = "#233149"
FG = "#E8EEF9"
MUT = "#8296B0"
ACC = "#38BDF8"
ACC2 = "#0EA5E9"
FONT = ("Microsoft YaHei UI", 10)
FONT_S = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 10, "bold")
FONT_T = ("Microsoft YaHei UI", 15, "bold")


def _js(path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except OSError:
        pass


def _round(v, step):
    q = Decimal(str(step))
    r = Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP)
    if r == 0:
        r = abs(r)              # 归一化 -0.0 → 0.0（面板曾显示 "-0.00"）
    return float(r)


def _fmt(v, step):
    if v == 0:
        v = 0.0                 # -0.0 显示成 "0.00" 而不是 "-0.00"
    d = max(0, -Decimal(str(step)).as_tuple().exponent)
    return f"{v:.{d}f}"


_KS_NAMED = {"pause": 0x13, "scroll_lock": 0x91, "space": 0x20, "return": 0x0D,
             "tab": 0x09, "escape": 0x1B, "backspace": 0x08, "insert": 0x2D,
             "delete": 0x2E, "home": 0x24, "end": 0x23, "prior": 0x21,
             "next": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28}


def _keysym_to_vk(ks):
    """tkinter keysym -> Windows VK。"""
    if not ks:
        return 0
    k = ks.lower()
    if k in _KS_NAMED:
        return _KS_NAMED[k]
    if len(k) == 1 and k.isalnum():
        return ord(k.upper())
    if k.startswith("f") and k[1:].isdigit() and 1 <= int(k[1:]) <= 24:
        return 0x70 + int(k[1:]) - 1
    return 0


_VK_NAMES = {0x13: "Pause", 0x91: "ScrollLock", 0x20: "Space", 0x0D: "Enter",
             0x09: "Tab", 0x1B: "Esc", 0x08: "Backspace", 0x2D: "Insert",
             0x2E: "Delete", 0x24: "Home", 0x23: "End", 0x21: "PgUp",
             0x22: "PgDn", 0x25: "Left", 0x26: "Up", 0x27: "Right", 0x28: "Down"}


def _vk_label(vk):
    if vk in _VK_NAMES:
        return _VK_NAMES[vk]
    if 0x70 <= vk <= 0x87:
        return f"F{vk - 0x70 + 1}"
    try:
        return chr(vk)
    except Exception:
        return f"VK{vk}"


# --------------------------------------------------------------------- 基础控件
class _Round(tk.Canvas):
    """圆角矩形绘制工具。"""

    @staticmethod
    def rr(cv, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return cv.create_polygon(pts, smooth=True, **kw)


class _Button(_Round):
    def __init__(self, master, text, command, bg=CARD, fg=FG, fill=None,
                 width=110, height=34, accent=False):
        super().__init__(master, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._t = text
        self._cmd = command
        self._bg = bg
        self._accent = accent
        self._fg = "#06121F" if accent else fg
        self._base = fill or (ACC2 if accent else CARD_HI)
        self._hot = ACC if accent else "#26374F"
        self._cw, self._h = width, height
        self.bind("<Button-1>", lambda e: self._cmd())
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self._paint(False)

    def set_text(self, t):
        self._t = t
        self._paint(False)

    def _paint(self, hot):
        self.delete("all")
        self.rr(self, 1, 1, self._cw - 1, self._h - 1, 9,
                fill=self._hot if hot else self._base, outline="")
        self.create_text(self._cw // 2, self._h // 2, text=self._t,
                         fill=self._fg, font=FONT_B)


class _Switch(tk.Canvas):
    def __init__(self, master, var, bg=CARD):
        super().__init__(master, width=44, height=24, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._v = var
        self.bind("<Button-1>", self._click)
        try:
            self._v.trace_add("write", lambda *_: self._paint())
        except AttributeError:
            self._v.trace("w", lambda *_: self._paint())
        self._paint()

    def _click(self, _e=None):
        if getattr(self, "_locked", False):
            return          # 被规则锁定（例如 SBS 强制全屏）
        self._v.set(not bool(self._v.get()))

    def set_locked(self, locked):
        """锁定/解锁：锁定后点击无效，但外观保持正常（不是灰框）。"""
        self._locked = bool(locked)
        try:
            self.configure(cursor="arrow" if self._locked else "hand2")
        except Exception:
            pass
        self._paint()

    def _paint(self):
        self.delete("all")
        on = bool(self._v.get())
        # 锁定时用略暗的青/灰，但形状与开/关状态正常显示
        on_fill = ACC if not getattr(self, "_locked", False) else "#2A7FA8"
        off_fill = TRACK if not getattr(self, "_locked", False) else "#2A3444"
        _Round.rr(self, 2, 4, 42, 20, 8, fill=on_fill if on else off_fill, outline="")
        cx = 34 if on else 10
        self.create_oval(cx - 7, 5, cx + 7, 19, fill="#FFFFFF", outline="")


class _Slider(_Round):
    """细滑轨 + 圆钮；值由 var 驱动，支持拖动。"""

    def __init__(self, master, var, lo, hi, step, width=210, on_change=None):
        super().__init__(master, width=width, height=26, bg=CARD,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._v, self._lo, self._hi, self._step = var, lo, hi, step
        self._cw = width
        self._on_change = on_change
        self.bind("<Button-1>", self._set_from_x)
        self.bind("<B1-Motion>", self._set_from_x)
        try:
            self._v.trace_add("write", lambda *_: self._paint())
        except AttributeError:
            self._v.trace("w", lambda *_: self._paint())
        self._paint()

    def _set_from_x(self, ev):
        x = min(self._cw - 10, max(10, ev.x))
        frac = (x - 10) / (self._cw - 20)
        val = self._lo + frac * (self._hi - self._lo)
        try:
            self._v.set(_round(min(self._hi, max(self._lo, val)), self._step))
        except Exception:
            pass
        self._paint()
        if self._on_change:
            self._on_change()

    def _paint(self):
        self.delete("all")
        try:
            frac = (float(self._v.get()) - self._lo) / (self._hi - self._lo)
        except Exception:
            frac = 0.0
        frac = min(1.0, max(0.0, frac))
        y = 13
        _Round.rr(self, 10, y - 2, self._cw - 10, y + 2, 2, fill=TRACK, outline="")
        x = 10 + frac * (self._cw - 20)
        if x > 12:
            _Round.rr(self, 10, y - 2, x, y + 2, 2, fill=ACC, outline="")
        self.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#FFFFFF", outline="")


class _Stepper(_Round):
    def __init__(self, master, var, lo, hi, step, width=104):
        super().__init__(master, width=width, height=30, bg=CARD,
                         highlightthickness=0, bd=0)
        self._v, self._lo, self._hi, self._step = var, lo, hi, step
        self._cw = width
        self.tag_bind("minus", "<Button-1>", lambda e: self._bump(-1))
        self.tag_bind("plus", "<Button-1>", lambda e: self._bump(+1))
        # 变量变化要重画（_load 装载真实设置时，控件必须跟着更新）
        try:
            self._v.trace_add("write", lambda *_: self._paint())
        except AttributeError:
            self._v.trace("w", lambda *_: self._paint())
        self._paint()

    def _text(self):
        try:
            return _fmt(float(self._v.get()), self._step)
        except Exception:
            return "?"

    def _bump(self, d):
        try:
            v = float(self._v.get()) + d * self._step
            self._v.set(_round(min(self._hi, max(self._lo, v)), self._step))
        except Exception:
            pass
        self._paint()

    def _paint(self):
        self.delete("all")
        _Round.rr(self, 1, 1, self._cw - 1, 29, 8, fill=CARD_HI, outline="")
        self.create_text(14, 15, text="−", fill=ACC, font=("Segoe UI", 14, "bold"), tags="minus")
        self.create_text(self._cw - 14, 15, text="+", fill=ACC, font=("Segoe UI", 14, "bold"), tags="plus")
        # 数字必须每次重画：之前 delete("all") 把它删了且没人补，控件永远显示
        # 构造时的旧值（用户实测：抗锯齿显示 2 / 平滑显示 5，实际 0 / 10）
        self.create_text(self._cw // 2, 15, text=self._text(),
                         fill=FG, font=FONT_B, tags="val")


class _Dropdown(_Round):
    def __init__(self, master, items, index, on_pick, width=250, bg=CARD):
        super().__init__(master, width=width, height=32, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._items, self._idx, self._pick = list(items), index, on_pick
        self._cw = width
        self.bind("<Button-1>", self._open)
        self._paint()

    def _paint(self):
        self.delete("all")
        _Round.rr(self, 1, 1, self._cw - 1, 31, 8, fill=CARD_HI, outline="")
        txt = self._items[self._idx] if self._items else ""
        self.create_text(12, 16, text=txt, anchor="w", fill=FG, font=FONT)
        self.create_text(self._cw - 14, 16, text="▾", fill=ACC, font=FONT)

    def _open(self, _e=None):
        """点击展开；再点一次收回（含"点别处已自动收起"的那次点击）。"""
        import time as _t
        if getattr(self, "_popup", None) is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None
            self._closed_at = _t.time()
            return
        # FocusOut 会先于本次 Button-1 触发：刚关掉的那一下不要立刻又开
        if _t.time() - getattr(self, "_closed_at", 0.0) < 0.30:
            self._closed_at = 0.0
            return
        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.configure(bg=CARD_HI)
        x = self.winfo_rootx()
        y = self.winfo_rooty() + 34
        top.geometry(f"{self._cw}x{min(240, 32 * len(self._items))}+{x}+{y}")
        for i, it in enumerate(self._items):
            b = tk.Label(top, text=it, bg=CARD_HI, fg=FG, font=FONT, anchor="w",
                         padx=12, pady=6, cursor="hand2")
            b.pack(fill="x")
            b.bind("<Button-1>", lambda e, k=i: self._choose(k, top))
            b.bind("<Enter>", lambda e, w=b: w.configure(bg="#26374F"))
            b.bind("<Leave>", lambda e, w=b: w.configure(bg=CARD_HI))
        def _close(_e=None):
            self._popup = None
            self._closed_at = _t.time()
            try:
                top.destroy()
            except Exception:
                pass

        top.bind("<FocusOut>", _close)
        top.bind("<Escape>", _close)
        self._popup = top
        top.focus_force()

    def _choose(self, i, top):
        self._idx = i
        self._paint()
        self._popup = None
        import time as _t
        self._closed_at = _t.time()
        top.destroy()
        if self._pick:
            self._pick(i)

    def set_items(self, items, index=None):
        """更新候选项（显示器插拔/分辨率变化时实时刷新；保持当前选择）。"""
        self._items = list(items)
        if index is None:
            index = self._idx
        if self._items:
            self._idx = max(0, min(int(index), len(self._items) - 1))
        else:
            self._idx = 0
        self._paint()


# --------------------------------------------------------------------- 主窗口
class SettingsDialog(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("+6000+6000")   # 先挪出屏幕，再隐藏
        self.withdraw()
        self.title("立体派 Cubist")
        self.configure(bg=BG)
        # 正常窗口行为：最小化/最大化由系统标题栏提供；关闭=收起（程序本体在托盘里，
        # 这个进程只负责设置面板，所以直接销毁进程即可，托盘不受影响）。
        self.resizable(True, True)
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass
        try:
            self.bind("<F11>", lambda _e: self._toggle_max())
        except Exception:
            pass
        try:
            self.iconbitmap(ICON_PATH)
        except Exception:
            pass

        self._vars = {}
        self._entries = {}
        self._monitors = []

        self._build()
        self._load()
        # 显示器列表实时刷新（插拔/切分辨率后面板不用重开）
        try:
            self.after(2000, self._refresh_monitors)
        except Exception:
            pass
        try:
            self._yaml_mtime = os.path.getmtime(SETTINGS_PATH)
        except Exception:
            self._yaml_mtime = 0.0
        try:
            self.bind("<FocusIn>", self._refresh_if_changed)
        except Exception:
            pass
        try:
            self._vars["Auto Start"].set(autostart_enabled())
        except Exception:
            pass
        # 托盘"退出"只结束主程序；面板是独立进程，必须自己跟随退出，
        # 否则会留下一个孤儿窗口（实测托盘退出后面板还开着）。
        if _main_pid_alive() is not None:
            self.after(1000, self._watch_main)
        self.update_idletasks()
        w = 540
        h = self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(24, (sh - h) // 2 - 40)}")
        self.deiconify()

    # ---------------- 布局
    def _card(self, title):
        wrap = tk.Frame(self, bg=CARD)
        wrap.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(wrap, text=title, bg=CARD, fg=ACC, font=FONT_B,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 2))
        body = tk.Frame(wrap, bg=CARD)
        body.pack(fill="x", padx=14, pady=(0, 10))
        body.columnconfigure(1, weight=1)
        body._row = 0
        return body

    def _row(self, body, text):
        r = body._row
        body._row = r + 1
        tk.Label(body, text=text, bg=CARD, fg=FG, font=FONT,
                 anchor="w").grid(row=r, column=0, sticky="w", pady=6)
        return r

    def _slider_row(self, body, text, key, lo, hi, step, ends=None):
        r = self._row(body, text)
        holder = tk.Frame(body, bg=CARD)
        holder.grid(row=r, column=1, sticky="we", padx=(12, 0))
        col = 0
        if ends:
            tk.Label(holder, text=ends[0], bg=CARD, fg=MUT, font=FONT_S).grid(row=0, column=0, padx=(0, 6))
            col = 1
        var = self._vars[key]
        _Slider(holder, var, lo, hi, step, on_change=lambda: self._sync(key, step)).grid(
            row=0, column=col, sticky="w")
        ent = tk.Entry(holder, width=6, justify="center", bg=CARD_HI, fg=FG,
                       insertbackground=ACC, relief="flat", font=FONT)
        ent.grid(row=0, column=col + 1, padx=(10, 6 if ends else 0))
        if ends:
            tk.Label(holder, text=ends[1], bg=CARD, fg=MUT, font=FONT_S).grid(row=0, column=col + 2)
        self._entries[key] = (ent, step)

        def commit(_e=None):
            try:
                var.set(_round(min(hi, max(lo, float(ent.get()))), step))
            except (ValueError, tk.TclError):
                pass
            self._sync(key, step)

        ent.bind("<Return>", commit)
        ent.bind("<FocusOut>", commit)
        self._sync(key, step)

    def _sync(self, key, step):
        ent = self._entries.get(key)
        if not ent:
            return
        try:
            ent[0].delete(0, "end")
            ent[0].insert(0, _fmt(float(self._vars[key].get()), step))
        except Exception:
            pass

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 12))
        tk.Label(head, text="立体派", bg=BG, fg=ACC, font=FONT_T).pack(side="left")
        tk.Label(head, text="Cubist", bg=BG, fg=MUT, font=FONT).pack(side="left", padx=(8, 0), pady=(5, 0))

        c1 = self._card("立体")
        self._vars["Parallax (px)"] = tk.IntVar(value=24)
        self._slider_row(c1, "立体强度", "Parallax (px)", 0, 20, 1)
        self._vars["Convergence Bias"] = tk.DoubleVar(value=0.0)
        self._slider_row(c1, "深度位置", "Convergence Bias", -0.2, 0.2, 0.01, ends=("远", "近"))
        # Convergence Tau 已废弃（2026-11-18 收敛面直取每帧 P50）；
        # 变量仅为 yaml 兼容保留
        self._vars["Convergence Tau"] = tk.DoubleVar(value=2.0)

        # 「画质」卡片已整块隐藏：里面的抗锯齿/平滑帧数都已定值，没有可调项
        # 抗锯齿固定 3：语义是【两个方向里较大的 σ】（横屏→竖向=3），
        # 另一方向按每眼视口纵横比自动缩小（16:9 → 横向 1.69）
        self._vars["Anti-aliasing"] = tk.IntVar(value=3)
        # 深度 EMA 时间常数已定死 80ms（utils 常量），面板不再暴露



        c3 = self._card("操作")
        self._vars["Toggle Hotkey"] = tk.StringVar(value="pause")
        r = self._row(c3, "3D 切换热键")
        self._hk_btn = _Button(c3, self._hk_text(), self._capture_hotkey, bg=CARD)
        self._hk_btn.grid(row=r, column=1, sticky="e", padx=(12, 0))

        c4 = self._card("输出")

        # 输出格式：面板显示中文，落到 yaml 用引擎的英文枚举
        # 只保留已验证的三种：其余格式尚未实测，不对外暴露
        self._modes = [("行交错（偏振 3D 屏）", "Interleaved"),
                       ("Half-SBS（左右半宽）", "Half-SBS"),
                       ("Full-SBS（左右全宽）", "Full-SBS"),
                       ("Half-TAB（上下半高）", "Half-TAB"),
                       ("Full-TAB（上下全高）", "Full-TAB"),
                       ("Depth Map（深度图）", "Depth Map")]
        self._vars["Display Mode"] = tk.StringVar(value="Interleaved")
        cur_mode = str(self._vars["Display Mode"].get())
        mode_idx = next((i for i, (_l, v) in enumerate(self._modes) if v == cur_mode), 0)
        r = self._row(c4, "输出格式")
        self._mode_dd = _Dropdown(c4, [l for l, _v in self._modes], mode_idx, self._pick_mode,
                                  width=280)
        self._mode_dd.grid(row=r, column=1, sticky="e", padx=(12, 0))

        # 推理显卡：仅 GL 回退/离线工具有意义（DML 零拷贝跨卡会退读回）。
        # D3D 主路径固定"跟随采集屏所在卡"（adapter.pick 自动匹配），
        # 不再暴露手动选卡——选了也没用，只会误导。
        self._is_d3d = os.environ.get("D2S_RENDER", "d3d").strip().lower() \
            not in ("gl", "glfw", "opengl")
        self._vars["Inference GPU"] = tk.StringVar(value="Auto")
        self._gpus = [("Auto", "自动（跟随采集屏）")]
        if not self._is_d3d:
            self._gpus = self._list_adapters()
            r = self._row(c4, "推理显卡")
            cur_gpu = str(self._vars["Inference GPU"].get())
            gpu_idx = next((i for i, (_v, _l) in enumerate(self._gpus)
                            if _v == cur_gpu), 0)
            self._gpu_dd = _Dropdown(
                c4, [_l for _v, _l in self._gpus], gpu_idx,
                self._pick_gpu, width=280)
            self._gpu_dd.grid(row=r, column=1, sticky="e", padx=(12, 0))

        # 全屏输出开关已移除（2026-11-16 产品定案：WGC 窗口/皮肤模式退役，
        # 全屏输出恒开、由 utils 仲裁强制，不再暴露给用户）。

        c5 = self._card("显示")
        self._monitors = self._list_monitors()
        self._vars["Monitor Index"] = tk.IntVar(value=1)
        # 采集屏（AB 分离）：默认 = 输出屏；下拉只列与输出屏同卡的屏——
        # 零拷贝管线要求采集/渲染同卡（跨卡运行时弹框并回退同屏）。
        self._vars["Capture Monitor Index"] = tk.IntVar(value=1)
        cur = max(0, min(len(self._monitors) - 1,
                         int(self._vars["Monitor Index"].get()) - 1))
        _oids = [mid for mid, _l in self._monitors]
        # 行的顺序：采集屏在上、输出屏在下（用户直觉：源头在前）
        r = self._row(c5, "采集屏")
        self._cap_items = self._capture_choices(_oids[cur] if _oids else 1)
        self._cap_dd = _Dropdown(c5, [lab for _m, lab in self._cap_items], 0,
                                 self._pick_capture_monitor)
        self._cap_dd.grid(row=r, column=1, sticky="e", padx=(12, 0))

        r = self._row(c5, "输出屏")
        self._mon_dd = _Dropdown(c5, [lab for _, lab in self._monitors], cur, self._pick_monitor)
        self._mon_dd.grid(row=r, column=1, sticky="e", padx=(12, 0))

        self._vars["Interleave Phase"] = tk.BooleanVar(value=False)
        r = self._row(c5, "左右反转")
        _Switch(c5, self._vars["Interleave Phase"]).grid(row=r, column=1, sticky="e", padx=(12, 0))
        self._vars["Show FPS"] = tk.BooleanVar(value=False)
        r = self._row(c5, "显示帧率")
        _Switch(c5, self._vars["Show FPS"]).grid(row=r, column=1, sticky="e", padx=(12, 0))
        # 开机自启动：实际状态以启动文件夹里有没有我们的脚本为准
        self._vars["Auto Start"] = tk.BooleanVar(value=autostart_enabled())
        r = self._row(c5, "开机自启动")
        _Switch(c5, self._vars["Auto Start"]).grid(row=r, column=1, sticky="e", padx=(12, 0))

        # 隐藏项（保持 yaml 完整）
        self._vars.update({
            "Depth Model": tk.StringVar(value="d224"),
            "Depth Resolution": tk.IntVar(value=224),
            "Depth Fixed": tk.BooleanVar(value=True),
            "GPU Pipeline": tk.BooleanVar(value=True),
            "Edge Falloff": tk.BooleanVar(value=False),
            "VSync": tk.BooleanVar(value=False),
            "Capture Tool": tk.StringVar(value="DesktopDuplication"),
        })

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(2, 14))
        self._status = tk.Label(bar, text="", bg=BG, fg=MUT, font=FONT_S)
        self._status.pack(side="left")
        _Button(bar, "保存并应用", self._save, bg=BG, accent=True, width=120).pack(side="right")
        _Button(bar, "取消", self._cancel, bg=CARD, width=80).pack(side="right", padx=(0, 8))
        # 2D/3D 实时切换：写模式文件，主循环下一帧即生效（不需要重启、不需要记热键）
        self._mode_btn = _Button(bar, self._mode_btn_text(), self._toggle_2d3d,
                                  bg=CARD, width=120)
        self._mode_btn.pack(side="right", padx=(0, 8))
        self._refresh_mode_btn()

    def _cancel(self):
        """不保存直接关闭（yaml 只在 _save 里写，所以这里什么都不做即等于放弃修改）。"""
        try:
            self.destroy()
        except Exception:
            pass

    def _refresh_if_changed(self, _evt=None):
        """主程序重启后 yaml 可能已经变了，而面板还拿着旧快照 ——
        再点保存会误判"无变化"，用户看到的就是"保存没反应"。
        焦点回到面板时按 mtime 判定是否重载。"""
        try:
            mt = os.path.getmtime(SETTINGS_PATH)
        except Exception:
            return
        if mt == getattr(self, "_yaml_mtime", 0.0):
            return
        self._yaml_mtime = mt
        # 只重读文件并刷新 utils.settings，不做 importlib.reload
        # （reload 会重跑模块级代码，副作用不可控）
        try:
            _data = utils.read_yaml(SETTINGS_PATH) or {}
            if _data:
                utils.settings.clear()
                utils.settings.update(_data)
        except Exception:
            return
        self._load()
        try:
            self._status.config(text="已按最新配置刷新")
        except Exception:
            pass

    def _watch_main(self):
        import time as _t
        try:
            alive = _main_pid_alive()
        except Exception:
            alive = True
        now = _t.time()
        # alive is True=活着；False=进程已死；None=pid 文件不存在。
        # 托盘"退出"的清理会先删 d2s.pid，所以 None 必须按"已退出"处理，
        # 否则面板永远等不到 False，成为孤儿窗口（用户实测的 bug）。
        # 自重启时 pid 文件也会短暂消失：5 秒宽限期 + _restart_grace 兜住。
        if alive is not True:
            if getattr(self, "_dead_since", None) is None:
                self._dead_since = now
            if now - self._dead_since > 5.0 and now > getattr(self, "_restart_grace", 0.0):
                self.destroy()
                return
        else:
            self._dead_since = None
        try:
            self.after(1000, self._watch_main)
        except Exception:
            pass

    def _on_close(self):
        """点 X：收起面板（设置进程退出，程序本体仍在托盘）。"""
        try:
            self.destroy()
        except Exception:
            pass

    def _toggle_max(self):
        try:
            if getattr(self, "_maxed", False):
                self.state("normal")
                self._maxed = False
            else:
                self.state("zoomed")
                self._maxed = True
        except Exception:
            pass

    def _mode_btn_text(self):
        return "切换到 2D" if current_mode() == "3d" else "切换到 3D"

    def _mode_btn_paint(self):
        txt = self._mode_btn_text()
        try:
            if getattr(self._mode_btn, "_t", None) != txt:
                self._mode_btn.set_text(txt)
        except Exception:
            pass

    def _refresh_mode_btn(self):
        self._mode_btn_paint()
        try:
            self.after(800, self._refresh_mode_btn)
        except Exception:
            pass

    def _toggle_2d3d(self):
        # 防抖 1.5s：切 3D 后首帧有 ~0.5s 暖机（DML 首推理）+ 内容暗时会
        # 误以为"没反应"再点——而这是切换钮，第二击=切回关闭（用户实测
        # "来回几次才成功"的根因）。窗口内的重复点击直接忽略。
        import time as _t
        _now = _t.perf_counter()
        if _now - getattr(self, "_mode_click_t", 0.0) < 1.5:
            self._status.config(text="已切换，请稍候首帧（约1秒）…")
            return
        self._mode_click_t = _now
        want = "2d" if current_mode() == "3d" else "3d"
        ok = set_mode(want)
        self._status.config(text=(f"已切换到 {want.upper()}" if ok else "切换失败（模式文件不可写）"))
        self._mode_btn_paint()
        try:
            self.after(2000, lambda: self._status.config(text=""))
        except Exception:
            pass

    def _hk_text(self):
        raw = str(self._vars["Toggle Hotkey"].get()).strip()
        label = raw
        try:
            if ":" in raw:                      # 存的是 mods:vk
                mods, vk = raw.split(":", 1)
                label = ("Ctrl+" if int(mods) else "") + _vk_label(int(vk))
            else:                               # 旧的键名写法
                label = {"pause": "Pause", "scrolllock": "ScrollLock",
                         "f10": "F10", "f11": "F11", "f9": "F9", "f12": "F12",
                         "ctrl+f2": "Ctrl+F2", "ctrl+pause": "Ctrl+Pause"}.get(raw, raw.title())
        except Exception:
            pass
        return f"  {label}  ▾"

    def _pick_mode(self, i):
        try:
            _m = self._modes[i][1]
            self._vars["Display Mode"].set(_m)
        except Exception:
            pass

    def _pick_monitor(self, i):
        try:
            idx = self._monitors[i][0]
            self._vars["Monitor Index"].set(idx)
        except Exception:
            return
        # 采集屏必须与输出屏同卡：输出屏一改，采集屏下拉随即刷新
        # （原先选在别的卡上 → 自动落到新卡的第一块屏）。
        self._sync_capture_choices()

    def _pick_capture_monitor(self, i):
        try:
            self._vars["Capture Monitor Index"].set(self._cap_items[i][0])
        except Exception:
            pass

    def _capture_choices(self, out_i):
        """采集屏候选项 = 与输出屏同一块显卡的显示器（含自己）。

        零拷贝管线要求采集/渲染同卡，跨卡组合在 UI 层就不提供；枚举失败
        回退全量（不把列表清空——宁可见到也不让用户无从选择）。
        """
        mons = list(self._monitors)
        try:
            from gpu import placement as _plc
            if _plc.monitor_adapter(int(out_i)) is not None:
                same = set(_plc.monitors_same_adapter(int(out_i)))
                same.add(int(out_i))
                pick = [(mid, lab) for mid, lab in mons if mid in same]
                if pick:
                    return pick
        except Exception:
            pass
        return mons

    def _sync_capture_choices(self):
        """按当前输出屏重建采集屏下拉（越界的采集屏选择自动纠正）。"""
        dd = getattr(self, "_cap_dd", None)
        if dd is None:
            return
        try:
            out_i = int(self._vars["Monitor Index"].get())
        except Exception:
            out_i = self._monitors[0][0] if self._monitors else 1
        self._cap_items = self._capture_choices(out_i)
        ids = [mid for mid, _l in self._cap_items]
        try:
            cap_i = int(self._vars["Capture Monitor Index"].get())
        except Exception:
            cap_i = out_i
        if cap_i not in ids:
            cap_i = out_i if out_i in ids else ids[0]
            self._vars["Capture Monitor Index"].set(cap_i)
        if getattr(dd, "_popup", None) is None:
            dd.set_items([lab for _m, lab in self._cap_items], ids.index(cap_i))

    def _pick_gpu(self, i):
        try:
            self._vars["Inference GPU"].set(self._gpus[i][0])
        except Exception:
            pass

    # ---------------- 显示器 / 热键
    @staticmethod
    def _edid():
        # 结果缓存：内部要拉一次 PowerShell（最长 8s），显示器列表每 2s
        # 轮询刷新，不缓存会不停起 powershell 进程。显示器矩形变化时由
        # _refresh_monitors 清缓存重读。
        if _EDID_CACHE[0] is not None:
            return _EDID_CACHE[0]
        out = {}
        try:
            ps = ("Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorID | "
                  "ForEach-Object { [pscustomobject]@{ id = ($_.InstanceName -split '\\\\')[1]; "
                  "mfg = -join ($_.ManufacturerName | ? {$_ -gt 0} | % {[char]$_}); "
                  "model = -join ($_.UserFriendlyName | ? {$_ -gt 0} | % {[char]$_}) } } | "
                  "ConvertTo-Json -Compress")
            import subprocess
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=8,
                               creationflags=0x08000000)
            data = json.loads(r.stdout or "[]")
            if isinstance(data, dict):
                data = [data]
            for d in data:
                eid = (d.get("id") or "").strip()
                if eid:
                    out[eid] = " ".join(x for x in ((d.get("mfg") or "").strip(),
                                                    (d.get("model") or "").strip()) if x) or eid
        except Exception:
            pass
        _EDID_CACHE[0] = out
        return out

    def _list_adapters(self):
        """[(存储值, 显示标签)]：Auto + 各 adapter（序号 名称 显存）。

        序号是唯一判别键——本机就有两块同名的 GTX 1650，名字不可作键。
        枚举失败（无 DXGI/远程会话）时只剩 Auto 项，功能退化为不可选。
        """
        items = [("Auto", "自动（跟随显示卡）")]
        try:
            from gpu import adapter as _ad
            seen = set()
            for a in _ad.enum_adapters():
                if not a.vram:
                    continue   # Microsoft Basic Render Driver 之类软渲染器不列出
                label = "adapter%d  %s（%dMB）" % (a.index, a.name, a.vram >> 20)
                # 同名同显存的克隆条目不列出：远程/虚拟显示驱动（IddCx，
                # 实测向日葵 OrayIddDriver）在 DXGI 里冒充其渲染 GPU 的名字，
                # 且克隆的是哪个 GPU 会跨进程变——选到克隆上推理会话会失败
                key = (a.name, a.vram >> 20)
                if key in seen:
                    continue
                seen.add(key)
                items.append((str(a.index), label))
        except Exception:
            pass
        return items

    def _list_monitors(self):
        """显示器列表（1-based，按屏幕 (left, top) 排序）。

        排序口径与产品端 placement.monitor_rects() 完全一致——win32api /
        mss / DXGI 三套原生枚举顺序互不相同（本机双屏实测 win32api 第 1 个
        反而是右侧屏），不排序会导致"面板选 2 号屏、产品铺到 1 号屏"。
        """
        edid = self._edid()
        raw = []
        try:
            import win32api
            for hmon, _h, _r in win32api.EnumDisplayMonitors():
                try:
                    mi = win32api.GetMonitorInfo(hmon)
                    rr = mi.get("Monitor") or (0, 0, 0, 0)
                    dd = win32api.EnumDisplayDevices(mi.get("Device", ""), 0)
                    nice = ""
                    for part in (getattr(dd, "DeviceID", "") or "").split("\\"):
                        if part.upper() in edid:
                            nice = edid[part.upper()]
                            break
                    raw.append((int(rr[0]), int(rr[1]), nice,
                                f"{rr[2] - rr[0]}×{rr[3] - rr[1]}"))
                except Exception:
                    pass
        except Exception:
            pass
        raw.sort(key=lambda x: (x[0], x[1]))
        out = [(i, f"{i}  {nice + '  ' if nice else ''}{size}")
               for i, (_l, _t, nice, size) in enumerate(raw, start=1)]
        return out or [(1, "显示器 1")]

    def _refresh_monitors(self):
        """显示器插拔/分辨率变化后刷新列表（面板可能长期开着）。"""
        try:
            from gpu import placement as _plc
            _rects = tuple(_plc.monitor_rects())
            if _rects != getattr(self, "_mon_rects_seen", None):
                self._mon_rects_seen = _rects
                _EDID_CACHE[0] = None      # 显示器变了：EDID 名称重读一次
        except Exception:
            pass
        try:
            mons = self._list_monitors()
        except Exception:
            mons = None
        try:
            if mons and [lab for _, lab in mons] != \
                    [lab for _, lab in self._monitors]:
                self._monitors = mons
                try:
                    cur = int(self._vars["Monitor Index"].get())
                except Exception:
                    cur = 1
                ids = [i for i, _ in mons]
                sel = ids.index(cur) if cur in ids else 0
                self._vars["Monitor Index"].set(ids[sel])
                if getattr(self._mon_dd, "_popup", None) is None:
                    self._mon_dd.set_items([lab for _, lab in mons], sel)
                self._sync_capture_choices()
                print(f"[settings] 显示器列表更新: {len(mons)} 块", flush=True)
        except Exception:
            pass
        try:
            self.after(2000, self._refresh_monitors)
        except Exception:
            pass

    def _capture_hotkey(self):
        top = tk.Toplevel(self)
        top.title("热键")
        top.configure(bg=BG)
        top.resizable(False, False)
        top.transient(self)
        top.grab_set()
        tk.Label(top, text="按下要绑定的键", bg=BG, fg=FG, font=FONT).pack(pady=(18, 2))
        tk.Label(top, text="Pause 被程序独占，请用下面按钮", bg=BG, fg=MUT,
                 font=FONT_S).pack()
        bar = tk.Frame(top, bg=BG)
        bar.pack(pady=12)
        for k in ("pause", "f10", "scrolllock", "ctrl+f2"):
            def _pick(kk=k):
                self._vars["Toggle Hotkey"].set(kk)
                try:
                    top.destroy()
                except Exception:
                    pass
            _Button(bar, k, _pick, bg=BG, width=84, height=30).pack(side="left", padx=4)
        hint = tk.Label(top, text="等待按键…", bg=BG, fg=ACC, font=FONT_S)
        hint.pack(pady=(0, 16))
        top.geometry(f"420x190+{self.winfo_rootx() + 50}+{self.winfo_rooty() + 90}")

        def on_key(ev):
            vk = _keysym_to_vk(ev.keysym)
            if not vk:
                hint.config(text=f"不支持：{ev.keysym}")
                return
            mods = 2 if (ev.state & 0x4) else 0
            label = _vk_label(vk)
            self._vars["Toggle Hotkey"].set(f"{mods}:{vk}")
            self._hk_btn.set_text(f"  {('Ctrl+' if mods else '') + label}  ▾")
            top.destroy()

        top.bind("<KeyPress>", on_key)
        top.focus_force()

    # ---------------- 装载 / 保存
    def _load(self):
        s = dict(utils.settings)
        for key, var in self._vars.items():
            if key == "Auto Start":
                # 自启动以"启动文件夹里有没有我们的脚本"为准，
                # 不能被 yaml 覆盖：否则 yaml=false + 文件存在 时面板显示关闭，
                # 用户一保存就把真的自启动删掉了。
                continue
            cur = s.get(key, var.get())
            try:
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(cur))
                elif isinstance(var, tk.IntVar):
                    var.set(int(float(cur)))
                elif isinstance(var, tk.DoubleVar):
                    var.set(float(cur))
                else:
                    var.set(str(cur))
            except (TypeError, ValueError):
                pass
        for key, (ent, step) in self._entries.items():
            self._sync(key, step)
        self._hk_btn.set_text(self._hk_text())
        try:
            cur_mode = str(self._vars["Display Mode"].get())
            for i, (_l, v) in enumerate(self._modes):
                if v == cur_mode:
                    self._mode_dd._idx = i
                    self._mode_dd._paint()
                    break
        except Exception:
            pass
        try:
            cur_gpu = str(self._vars["Inference GPU"].get())
            for i, (_v, _l) in enumerate(self._gpus):
                if _v == cur_gpu:
                    self._gpu_dd._idx = i
                    self._gpu_dd._paint()
                    break
        except Exception:
            pass
        try:
            idx = int(self._vars["Monitor Index"].get())
            for i, (mid, _l) in enumerate(self._monitors):
                if mid == idx:
                    self._mon_dd._idx = i
                    self._mon_dd._paint()
                    break
        except Exception:
            pass
        self._sync_capture_choices()   # 采集屏下拉跟随输出屏（同卡过滤）

    def _collect(self):
        out = {}
        # 模型与分辨率不随面板保存写入：面板里的下拉可能还是旧值，
        # 一保存就会把用户手改到 yaml 的 448 覆盖回 224（实测踩过）。
        _NO_SAVE = ("Depth Model", "Depth Resolution")
        if getattr(self, "_is_d3d", False):
            # D3D 主路径自动跟随采集屏，保留 yaml 里的旧值（GL 回退仍可能用）
            _NO_SAVE += ("Inference GPU",)
        for key, var in self._vars.items():
            if key in _NO_SAVE:
                continue
            v = var.get()
            if key in ("Depth Resolution", "Parallax (px)", "Monitor Index",
                       "Capture Monitor Index",
                       "Anti-aliasing",
                       "Depth Close (px)", "Depth Grow (px)"):
                out[key] = int(v)
            elif key == "Cursor Scale":
                out[key] = round(float(v), 2)
            elif key in ("Convergence", "Depth Gamma", "Convergence Bias",
                         "Convergence Tau", "Depth Slope Limit"):
                _q = Decimal(str(v)).quantize(Decimal("0.001"),
                                              rounding=ROUND_HALF_UP)
                if _q == 0:
                    _q = abs(_q)    # -0.0 → 0.0（写盘不要负零）
                out[key] = float(_q)
            elif key in ("Show FPS", "VSync", "GPU Pipeline", "Edge Falloff",
                         "Depth Fixed", "Convergence Auto", "Interleave Phase"):
                out[key] = bool(v)
            else:
                out[key] = str(v)
        return out

    def _save(self):
        import time as _t
        self._dead_since = None
        self._restart_grace = _t.time() + 8.0     # 本次保存可能触发自重启
        updates = self._collect()
        before = {k: utils.settings.get(k) for k in updates}
        if not utils.write_yaml(SETTINGS_PATH, updates):
            messagebox.showerror("立体派 Cubist", "settings.yaml 写入失败")
            return
        # 开机自启动：按开关状态写/删启动项（失败只提示，不阻塞保存）
        try:
            _want_auto = bool(self._vars["Auto Start"].get())
            if _want_auto != autostart_enabled():
                if not set_autostart(_want_auto):
                    self._status.config(text="自启动设置失败（权限？）")
                self._vars["Auto Start"].set(autostart_enabled())
        except Exception:
            pass
        # 保存成功后立刻更新基线：否则面板还拿着"保存前"的快照，
        # 下次点保存会误判成"没有变化"→ 既不重启也不下发实时参数。
        try:
            utils.settings.update(updates)
        except Exception:
            pass
        live = {k: updates[k] for k in LIVE_KEYS
                if k in updates and updates[k] != before.get(k)}
        if live:
            _js(PARAMS_REQ, live)
        changed = [k for k in updates if updates[k] != before.get(k)]
        need = any(k not in LIVE_KEYS for k in changed)
        # 留痕：面板是独立进程，只能写 app.log；以后"改了没生效"可直接查
        try:
            print(f"[settings] 保存: 变化={changed or '无'} "
                  f"实时={list(live) or '无'} 重启={'是' if need else '否'}", flush=True)
        except Exception:
            pass
        if need:
            _js(RESTART_REQ, {"ts": time.time()})
            self._status.config(text="已保存 · 重启指令已发送")
        else:
            self._status.config(text="已保存 · 已生效")
        self.after(2500, lambda: self._status.config(text=""))
        # 不自动关闭：用户可以随时关，也可以再改


def main():
    if not claim_single_instance():
        return
    SettingsDialog().mainloop()


if __name__ == "__main__":
    main()
