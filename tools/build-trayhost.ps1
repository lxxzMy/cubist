# Precompile the tray daemon C# -> TrayHost.dll. Removes the Add-Type/csc
# dependency from daemon startup: a hung csc leaves the daemon alive but
# stuck before Run() - no tray, no hotkey, no Timer (observed 2026-11-08).
# Daemon load rule: TrayHost.dll newer than TrayHost.cs -> load the DLL;
# otherwise fall back to source compile (dev mode). ASCII comments only
# (PS5.1 misparses BOM-less UTF-8 Chinese comments as ANSI).
param([string]$RepoRoot = (Get-Location).Path)

$cs = Join-Path $RepoRoot 'tools\TrayHost.cs'
$dll = Join-Path $RepoRoot 'tools\TrayHost.dll'
$src = Get-Content $cs -Raw -Encoding UTF8   # 中文菜单：UTF-8 读源（ANSI 会吞行）
Add-Type -TypeDefinition $src `
    -ReferencedAssemblies @('System.Windows.Forms.dll', 'System.Drawing.dll') `
    -OutputAssembly $dll -OutputType Library
Write-Host ("TrayHost.dll built: " + (Get-Item $dll).Length + " bytes")
