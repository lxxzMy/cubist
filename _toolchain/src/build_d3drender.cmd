@echo off
setlocal enabledelayedexpansion
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSP=%%i
set MSVC=%VSP%\VC\Tools\MSVC
for /f "delims=" %%d in ('dir /b /ad "%MSVC%" ^| sort') do set MSVCT=%%d
set MSVCF=%MSVC%\%MSVCT%
set CLP=%MSVCF%\bin\Hostx64\x64\cl.exe
set SDK=%~dp0..\dl\sdk\cpp\c\Include\10.0.26100.0
set SDKLIB=%~dp0..\dl\sdk\x64\c
set INCLUDE=%MSVCF%\include;%SDK%\um;%SDK%\ucrt;%SDK%\shared
set LIB=%MSVCF%\lib\x64;%SDKLIB%\um\x64;%SDKLIB%\ucrt\x64
"%CLP%" /nologo /W3 /utf-8 /MT /EHsc /O2 /LD d3drender.cpp /Fe:d3drender.dll /Fo:d3drender.obj /link d3d11.lib dxgi.lib d3d12.lib d3dcompiler.lib user32.lib gdi32.lib advapi32.lib
if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)
del d3drender.obj d3drender.exp d3drender.lib 2>nul
echo [build] d3drender.dll done
