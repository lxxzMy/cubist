// 立体派 Cubist 启动器（GUI，无控制台）
// 行为与 启动.bat 一致：
//   1) 已在运行（logs\d2s.pid 的进程还活着）→ 只弹设置面板
//   2) 未运行 → 启动主程序（托盘），等 3 秒让模型开始加载，再弹设置面板
// 目录约定（与发布包一致）：
//   立体派.exe
//   ├─ app\       程序本体
//   ├─ python\    内置 Python
//   └─ runtime\   模型与图像库
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows.Forms;

static class Launcher
{
    static string BaseDir
    {
        get { return Path.GetDirectoryName(Application.ExecutablePath); }
    }

    static int Main()
    {
        string root = BaseDir;
        string app = Path.Combine(root, "app");
        string py = Path.Combine(root, "python", "pythonw.exe");
        string runtime = Path.Combine(root, "runtime");

        if (!Directory.Exists(app) || !File.Exists(py))
        {
            MessageBox.Show(
                "找不到程序目录。\r\n\r\n请确认 立体派.exe 与 app\\、python\\、runtime\\ 在同一层。",
                "立体派 Cubist", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        RestoreCursors();
        bool running = IsAppRunning(app);

        if (!running)
        {
            if (!StartPython(py, app, runtime, "-m main"))
                return 2;
            Thread.Sleep(3000);          // 给模型加载留出时间，避免面板先弹、app 还没起
        }

        StartPython(py, app, runtime, "-m settings_dialog");
        return 0;
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    static extern bool SystemParametersInfo(uint uiAction, uint uiParam, IntPtr pvParam, uint fWinIni);

    /// <summary>把系统光标表恢复成系统默认（SPI_SETCURSORS）。
    /// 3D 输出时程序会把光标换成空白位图；若上次进程被强杀，
    /// 这里先恢复一次，避免"鼠标看不见/半透明"。</summary>
    static void RestoreCursors()
    {
        try { SystemParametersInfo(0x0057, 0, IntPtr.Zero, 0); } catch { }
    }

    static bool IsAppRunning(string appDir)
    {
        try
        {
            string pidFile = Path.Combine(appDir, "logs", "d2s.pid");
            if (!File.Exists(pidFile)) return false;
            int pid;
            if (!int.TryParse(File.ReadAllText(pidFile).Trim(), out pid)) return false;
            if (pid <= 0) return false;
            var p = Process.GetProcessById(pid);
            return p != null && !p.HasExited;
        }
        catch
        {
            return false;   // pid 过期/无权访问 → 当作没在跑
        }
    }

    static bool StartPython(string py, string appDir, string runtime, string args)
    {
        try
        {
            var psi = new ProcessStartInfo(py)
            {
                Arguments = args,
                WorkingDirectory = appDir,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            string oldPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            // 让 ORT/DirectML 能找到 runtime 下的依赖 DLL
            psi.EnvironmentVariables["PATH"] = runtime + ";" + oldPath;
            Process.Start(psi);
            return true;
        }
        catch (Exception e)
        {
            MessageBox.Show("启动失败：\r\n" + e.Message, "立体派 Cubist",
                            MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }
    }
}
