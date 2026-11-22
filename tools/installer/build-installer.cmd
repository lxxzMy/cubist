@echo off
rem Desktop2Stereo installer build. Requires Inno Setup 6 (ISCC on PATH or
rem in the default install dir). ASCII only.
setlocal
set REPO=%~dp0..\..
for %%I in ("%REPO%") do set REPO=%%~fI

set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
  echo Inno Setup 6 not found. Install it first:
  echo   winget install JRSoftware.InnoSetup
  echo   (or download https://jrsoftware.org/isdl.php)
  exit /b 1
)

echo Building Desktop2Stereo setup from %REPO% ...
%ISCC% /DRepoRoot=%REPO% "%~dp0d2s.iss"
if errorlevel 1 exit /b 1
echo Done: %REPO%\dist\Desktop2Stereo-setup.exe
