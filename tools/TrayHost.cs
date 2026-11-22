// TrayHost/WinT/MonInfo - the complete tray-daemon C# (extracted from d2s-toggle.ps1).
// Precompiled into TrayHost.dll: the daemon loads the prebuilt DLL, bypassing
// Add-Type/csc at startup (a hung csc leaves the daemon stuck in compile:
// alive but no tray/hotkey/Timer; observed 2026-11-08). Single source of truth; rebuild via tools/build-trayhost.ps1.

using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;

public delegate IntPtr HookProc(int code, IntPtr wp, IntPtr lp);

public class TrayHost {
    [DllImport("user32.dll")] public static extern IntPtr SetWindowsHookExW(int id, HookProc cb, IntPtr mod, uint tid);
    [DllImport("user32.dll")] public static extern bool UnhookWindowsHookEx(IntPtr h);

    // --- Pause-key detection (low-level hook) ---
    // Key history: Right-Ctrl (IME) -> Alt+Space (system menu) ->
    // Ctrl+Space (also taken) -> Pause. Fn+Space is impossible: Fn is
    // intercepted by keyboard firmware and never reaches the OS.
    // Pause is a plain non-modifier key - fire on keydown edge.
    static HookProc _proc = HookCb;
    static Action _fire;
    static Form _sync;

    static IntPtr HookCb(int code, IntPtr wp, IntPtr lp) {
        if (code >= 0) {
            long kind = wp.ToInt64();
            int vk = Marshal.ReadByte(lp, 0);
            bool down = (kind == 0x0100 || kind == 0x0104);
            if (vk == 0x13 && down) {
                if (_fire != null && _sync != null && !_sync.IsDisposed) {
                    // Marshal to the pump thread (the PowerShell runspace
                    // thread): scriptblocks cannot run on hook/pool
                    // threads, and blocking the hook stalls all input.
                    try { _sync.BeginInvoke(_fire); } catch {}
                }
            }
        }
        return IntPtr.Zero;
    }

    public static void Run(string iconPath, string title, string restartReqPath,
                           Action onToggle, Action onMenuToggle, Action onExit, Action onRestart, Action onQuit, Action onSettings) {
        _fire = onToggle;
        _sync = new Form();
        _sync.ShowInTaskbar = false;
        _sync.Visible = false;
        var sh = _sync.Handle;   // marshaling target only, never shown
        IntPtr hook = SetWindowsHookExW(13, _proc, IntPtr.Zero, 0); // WH_KEYBOARD_LL
        if (hook == IntPtr.Zero) {
            try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "d2s-hook-fail.txt"),
                 "SetWindowsHookExW failed at " + DateTime.Now + "\r\n"); } catch {}
        }

        var menu = new ContextMenuStrip();
        menu.Items.Add("设置...", null, (s, e) => onSettings());
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("切换 3D / 2D", null, (s, e) => onMenuToggle());
        menu.Items.Add("重启立体派", null, (s, e) => onRestart());
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("退出立体派", null, (s, e) => onExit());
        

        using (var ni = new NotifyIcon()) {
            ni.Icon = new System.Drawing.Icon(iconPath);
            ni.Text = title;
            ni.ContextMenuStrip = menu;
            ni.Visible = true;
            // Left click = open settings (right click = menu, tray default).
            ni.MouseClick += (s, e) => { if (e.Button == MouseButtons.Left) onSettings(); };
            // The settings dialog's 'restart D2S' goes through a file request: 2s poll
            // (on the message pump, same blocking profile as the Restart menu item).
            // Delete BEFORE executing: a 15s blocking Restart cannot cause a restart storm.
            var poll = new System.Windows.Forms.Timer { Interval = 2000 };
            poll.Tick += (s, e) => {
                try {
                    if (restartReqPath != "" && System.IO.File.Exists(restartReqPath)) {
                        System.IO.File.Delete(restartReqPath);
                        onRestart();
                    }
                } catch {}
            };
            poll.Start();
            // Bare message loop pumps the LL hook + tray; no form is ever shown.
            // 图标"鼠标一悬停就消失"：NotifyIcon 的内部窗口在消息循环起来之前
            // 建立，shell 重绘时查询回调窗口失败就把图标删掉。消息循环启动后
            // 重新设置一次 Visible（false->true）强制重建回调窗口。
            var reAdd = new System.Windows.Forms.Timer { Interval = 300 };
            reAdd.Tick += (s2, e2) => {
                reAdd.Stop();
                try { ni.Visible = false; ni.Visible = true; } catch {}
            };
            reAdd.Start();
            // Application.Run() 不带参数：本环境实测会立即返回（守护起来就退出、
            // 托盘不出现）。用 ApplicationContext 起一个真正的消息循环。
            Application.Run(new ApplicationContext());
            ni.Visible = false;
        }
        UnhookWindowsHookEx(hook);
    }
}

public class WinT {
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr l);
    public delegate bool EnumWindowsProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int m);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int w, int hh, uint f);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
    public static IntPtr FindByTitle(string title) {
        IntPtr result = IntPtr.Zero;
        EnumWindows((h, l) => {
            var sb = new StringBuilder(128);
            GetWindowText(h, sb, 128);
            if (sb.ToString().StartsWith(title)) { result = h; return false; }
            return true;
        }, IntPtr.Zero);
        return result;
    }
}

public class MonInfo {
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct DISPLAY_DEVICE {
        public int cb; [MarshalAs(UnmanagedType.ByValTStr, SizeConst=32)] public string DeviceName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceString;
        public int StateFlags;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceID;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceKey;
    }
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern bool EnumDisplayDevicesW(string dev, uint i, ref DISPLAY_DEVICE d, uint flags);
    public static bool IsBuiltin(string deviceId) {
        string v = deviceId == null ? "" : deviceId.ToUpperInvariant();
        return v.Contains("BOE") || v.Contains("CMN") || v.Contains("AUO") || v.Contains("SDC") || v.Contains("SHARP");
    }
}
