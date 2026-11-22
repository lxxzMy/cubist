#!/bin/bash
# Build p1_interop_test.exe with the VS BuildTools MSVC directly (no vcvars).
# Same nuget-SDK layout as build_p0.sh; C++ /EHsc; links d3d11/dxgi/d3d12/opengl32.
set -e
TC=/e/Projects/Desktop2Stereo/_toolchain
MSVC="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
SDKINC="$TC/dl/sdk/cpp/c/Include/10.0.26100.0"
SDKLIB="$TC/dl/sdk/x64/c"

WIN() { cygpath -w "$1"; }
export INCLUDE="$(WIN "$MSVC/include");$(WIN "$SDKINC/um");$(WIN "$SDKINC/ucrt");$(WIN "$SDKINC/shared")"
export LIB="$(WIN "$MSVC/lib/x64");$(WIN "$SDKLIB/um/x64");$(WIN "$SDKLIB/ucrt/x64")"

cd "$TC/src"
export MSYS2_ARG_CONV_EXCL='*'
"$MSVC/bin/Hostx64/x64/cl.exe" /nologo /W3 /MT /EHsc /O2 /Zi p1_interop_test.cpp \
  /Fe:p1_interop_test.exe /Fo:p1_interop_test.obj \
  /link d3d11.lib dxgi.lib d3d12.lib opengl32.lib user32.lib gdi32.lib
echo "built: $TC/src/p1_interop_test.exe"
