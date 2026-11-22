; Desktop2Stereo 安装器脚本（Inno Setup 6）
; 构建：build-installer.cmd（传 /DRepoRoot=仓库根 绝对路径）
; 原则：per-user 免管理员；TRT 为可选组件默认不装；Flet 面板不随包发布。

#define MyAppName "立体派 Cubist"
#define MyAppVersion "1.0"   ; 改版本时同步 D2S/utils.py 的 VERSION（单源约定）
#define MyAppExeName "settings_dialog.py"

#ifndef RepoRoot
; 未显式传 /DRepoRoot 时按本脚本位置推断仓库根（tools\installer\..\..）
#define RepoRoot SourcePath + "\..\.."
#endif

[Setup]
AppId={{7E3A1C42-9B5D-4E08-A6C2-D2S0PRODUCT01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\Cubist
PrivilegesRequired=lowest
Compression=lzma2/normal
SolidCompression=no
OutputDir={#RepoRoot}\dist
OutputBaseFilename=Cubist-setup
SetupIconFile={#RepoRoot}\D2S\icon.ico
UninstallDisplayIcon={app}\D2S\icon.ico
DisableProgramGroupPage=yes

[Types]
Name: "full"; Description: "标准安装（不含 TensorRT）"
Name: "compact"; Description: "基础安装"
Name: "custom"; Description: "自定义（可选装 TensorRT）"; Flags: iscustom

[Components]
Name: "core"; Description: "Desktop2Stereo 核心（引擎/运行库/托盘/设置）"; Types: full compact custom; Flags: fixed
Name: "trtruntime"; Description: "TensorRT 10.x 运行时（可选：448 深度模型提速用，约 1~2GB）"; Types: custom

[Files]
; ---- 核心：D2S（排除 Flet 面板与开发/运行时垃圾；settings.yaml 不覆盖——
;      用户配置升级保留，utils 的自愈机制会用 settings.yaml.default 初始化）----
Source: "{#RepoRoot}\D2S\*"; DestDir: "{app}\D2S"; Components: core; \
  Flags: recursesubdirs createallsubdirs ignoreversion; \
  Excludes: "rtmp,xr_viewer,flet_client,models,gui.py,logs,__pycache__,*.pyc,*.log,settings.yaml"
; ---- 运行库（CUDA/cuDNN/模型/转换器）----
Source: "{#RepoRoot}\runtime\*"; DestDir: "{app}\runtime"; Components: core; \
  Flags: recursesubdirs createallsubdirs ignoreversion; \
  Excludes: "trt-cache,parallaxrt_filter.log"
; ---- 工具（守护/看门狗/重启链）----
Source: "{#RepoRoot}\tools\*.ps1"; DestDir: "{app}\tools"; Components: core; Flags: ignoreversion
Source: "{#RepoRoot}\tools\*.vbs"; DestDir: "{app}\tools"; Components: core; Flags: ignoreversion
Source: "{#RepoRoot}\tools\TrayHost.dll"; DestDir: "{app}\tools"; Components: core; Flags: ignoreversion
Source: "{#RepoRoot}\tools\TrayHost.cs"; DestDir: "{app}\tools"; Components: core; Flags: ignoreversion
; ---- 文档 ----
Source: "{#RepoRoot}\README-先读我.md"; DestDir: "{app}"; Components: core; Flags: ignoreversion
Source: "{#RepoRoot}\3D系统使用手册.md"; DestDir: "{app}"; Components: core; Flags: ignoreversion
; ---- TensorRT 可选组件（构建机 _tools\trt-redist 有预取载荷才编译进去）----
#if FileExists(RepoRoot + "\\_tools\\trt-redist\\*")
Source: "{#RepoRoot}\_tools\trt-redist\*"; DestDir: "{app}\runtime\trt\bin"; Components: trtruntime; \
  Flags: recursesubdirs createallsubdirs ignoreversion
#endif

[Icons]
Name: "{userprograms}\立体派 Cubist 设置"; Filename: "{app}\D2S\python3\pythonw.exe"; \
  Parameters: "settings_dialog.py"; WorkingDir: "{app}\D2S"; \
  IconFilename: "{app}\D2S\icon.ico"; Comment: "打开 Desktop2Stereo 设置"
Name: "{userstartup}\立体派 Cubist 托盘"; Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\tools\d2s-toggle.ps1"""; \
  Comment: "Desktop2Stereo 托盘守护（开机自启）"

[Run]
; 看门狗任务：用安装后路径重注册（-Force 覆盖旧路径注册，阻断 5）
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\register-watchdog-task.ps1"""; \
  Flags: runhidden runasoriginaluser
; 启动守护（托盘出现）
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""{app}\tools\d2s-toggle.ps1"""; \
  Flags: runhidden runasoriginaluser nowait
; 装后打开设置（可见即成功）
Filename: "{app}\D2S\python3\pythonw.exe"; \
  Parameters: "settings_dialog.py"; WorkingDir: "{app}\D2S"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; 精确清场：按 pid/命令行杀（禁止 taskkill /im python.exe 误杀），阻断 4
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\uninstall-cleanup.ps1"""; \
  Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{app}\D2S\logs"
