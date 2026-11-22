# 编译发布包用的启动器 EXE（立体派.exe）
# 用法： .\tools\build-launcher.ps1 -Package "release\Cubist-1.0-224"
param(
    [string]$Package = "release\Cubist-1.0-224"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Package)) {
    $Package = Join-Path $root $Package
}
$src = Join-Path $PSScriptRoot "launcher.cs"
$icon = Join-Path $Package "app\icon.ico"
$out = Join-Path $Package "立体派.exe"

$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    $csc = (Get-ChildItem (Join-Path $env:WINDIR "Microsoft.NET\Framework64") -Recurse -Filter csc.exe |
            Select-Object -First 1).FullName
}
if (-not $csc) { throw "找不到 csc.exe（需要 .NET Framework 4.x）" }
if (-not (Test-Path $src)) { throw "找不到 launcher.cs" }
if (-not (Test-Path $Package)) { throw "发布包目录不存在: $Package" }

$args = @("/nologo", "/target:winexe", "/out:$out", "/r:System.Windows.Forms.dll")
if (Test-Path $icon) { $args += "/win32icon:$icon" }
$args += $src

& $csc @args
if ($LASTEXITCODE -ne 0) { throw "编译失败（exit $LASTEXITCODE）" }
$size = [math]::Round((Get-Item $out).Length / 1KB, 0)
Write-Host "已生成: $out  ($size KB)"
