#!/bin/bash
# Build p0_dml_probe.exe with the VS BuildTools MSVC directly (no vcvars needed).
# Uses the nuget Windows SDK (26100.9457) under _toolchain/dl/sdk for UCRT/D3D headers+libs,
# and the sparse-cloned ORT v1.24.4 headers under _toolchain/dl/ort-src.
set -e
TC=/e/Projects/Desktop2Stereo/_toolchain
MSVC="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
SDKINC="$TC/dl/sdk/cpp/c/Include/10.0.26100.0"
SDKLIB="$TC/dl/sdk/x64/c"
ORTINC="$TC/dl/ort-src/include"

WIN() { cygpath -w "$1"; }
export INCLUDE="$(WIN "$MSVC/include");$(WIN "$SDKINC/um");$(WIN "$SDKINC/ucrt");$(WIN "$SDKINC/shared");$(WIN "$ORTINC");$(WIN "$ORTINC/onnxruntime/core/session")"
export LIB="$(WIN "$MSVC/lib/x64");$(WIN "$SDKLIB/um/x64");$(WIN "$SDKLIB/ucrt/x64")"

cd "$TC/src"
export MSYS2_ARG_CONV_EXCL='*'
"$MSVC/bin/Hostx64/x64/cl.exe" /nologo /W3 /MT /O2 /Zi p0_dml_probe.c \
  /Fe:p0_dml_probe.exe /Fo:p0_dml_probe.obj \
  /link kernel32.lib user32.lib
echo "built: $TC/src/p0_dml_probe.exe"
