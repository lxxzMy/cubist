@echo off
rem Portable launcher: resolve everything relative to this script so the
rem package works from any drive/folder. CUDA/cuDNN runtime + DepthART ONNX
rem models live in ..\runtime.
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "D2S_ROOT=%%~fI"
set "PATH=%D2S_ROOT%\runtime;%PATH%"
Set "PYTHON_EXE=.\python3\python.exe"
%PYTHON_EXE% "%D2S_ROOT%	ools\dev_gui.py"
exit /b 0
