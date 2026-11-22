#!/bin/bash
# Build d2s_dml_shim.dll (P3 zero-copy shim) with MSVC.
#
# 产物：<repo>/runtime/d2s_dml_shim.dll（/MT 静态 CRT，无外部运行时依赖；
#       ORT/DirectML 的 DLL 路径由 Python 侧在 create 时传入，不在这里链接）。
#
# 依赖（三者都要能找到，脚本会先自动探测，找不到就报清楚缺什么）：
#   1. MSVC  x64 工具链（VS 2022 BuildTools / Community 的 C++ 工作负载）
#   2. Windows SDK 10 的 um/ucrt/shared 头 + x64 lib（d3d11/dxgi/d3d12/opengl32）
#   3. ONNX Runtime 1.24.x 的头文件（onnxruntime_c_api.h 等）
#      推荐：_toolchain/dl/ort-src/include（官方源码包或 nuget 包解出来的 include）
#
# 用法：
#   bash _toolchain/src/build_shim.sh
#   MSVC_DIR=... SDK_DIR=... ORT_INC=... bash _toolchain/src/build_shim.sh   # 手工覆盖
set -e

TC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$TC/_toolchain/src"
WIN() { cygpath -w "$1"; }

fail() { echo "[x] $*" >&2; exit 1; }

# ---- 1) MSVC ----
if [ -n "$MSVC_DIR" ]; then
  MSVC="$MSVC_DIR"
else
  VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  VSPATH=""
  if [ -x "$VSWHERE" ]; then
    VSPATH="$("$VSWHERE" -latest -products '*' \
      -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
      -property installationPath 2>/dev/null | tr -d '\r')"
  fi
  [ -n "$VSPATH" ] || VSPATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
  MSVC="$(ls -d "$VSPATH"/VC/Tools/MSVC/* 2>/dev/null | sort -V | tail -1)"
fi
[ -n "$MSVC" ] && [ -d "$MSVC/include" ] || fail "找不到 MSVC 工具链（装 VS 2022 的“使用 C++ 的桌面开发”，或设 MSVC_DIR）"
CL="$MSVC/bin/Hostx64/x64/cl.exe"
[ -x "$CL" ] || fail "MSVC 里没有 x64 的 cl.exe：$CL"

# ---- 2) Windows SDK ----
if [ -n "$SDK_DIR" ]; then
  SDKINC="$SDK_DIR/Include"
  SDKLIB="$SDK_DIR/Lib"
else
  if [ -d "$TC/_toolchain/dl/sdk/cpp/c/Include" ]; then
    SDKINC="$TC/_toolchain/dl/sdk/cpp/c/Include"
    SDKLIB="$TC/_toolchain/dl/sdk/x64/c"
  else
    SDKINC="/c/Program Files (x86)/Windows Kits/10/Include"
    SDKLIB="/c/Program Files (x86)/Windows Kits/10/Lib"
  fi
fi
SDKVER="$(ls -d "$SDKINC"/* 2>/dev/null | sort -V | tail -1)"
[ -n "$SDKVER" ] || fail "找不到 Windows SDK 头文件（$SDKINC）"
[ -f "$SDKVER/um/d3d11.h" ] || fail "Windows SDK 里没有 um/d3d11.h：$SDKVER"
[ -d "$SDKLIB/um/x64" ] || fail "Windows SDK 里没有 x64 lib（$SDKLIB/um/x64）"

# ---- 3) ORT 头 ----
if [ -n "$ORT_INC" ]; then
  ORTINC="$ORT_INC"
else
  ORTINC="$TC/_toolchain/dl/ort-src/include"
fi
[ -f "$ORTINC/onnxruntime_c_api.h" ] || fail "找不到 ORT 头（$ORTINC/onnxruntime_c_api.h）。把 onnxruntime 1.24.x 的 include/ 解到这里，或设 ORT_INC"

echo "MSVC    = $MSVC"
echo "SDK     = $SDKVER"
echo "ORT_INC = $ORTINC"

export INCLUDE="$(WIN "$MSVC/include");$(WIN "$SDKVER/um");$(WIN "$SDKVER/ucrt");$(WIN "$SDKVER/shared");$(WIN "$ORTINC");$(WIN "$ORTINC/onnxruntime/core/session")"
export LIB="$(WIN "$MSVC/lib/x64");$(WIN "$SDKLIB/um/x64");$(WIN "$SDKLIB/ucrt/x64")"

cd "$SRC"
export MSYS2_ARG_CONV_EXCL='*'
"$CL" /nologo /W3 /utf-8 /MT /EHsc /O2 /LD d2s_dml_shim.cpp \
  /Fe:d2s_dml_shim.dll /Fo:d2s_dml_shim.obj \
  /link d3d11.lib dxgi.lib d3d12.lib opengl32.lib /IMPLIB:d2s_dml_shim.lib
rm -f d2s_dml_shim.exp d2s_dml_shim.lib d2s_dml_shim.obj
cp -f d2s_dml_shim.dll "$TC/runtime/d2s_dml_shim.dll"
echo "built: $TC/runtime/d2s_dml_shim.dll"
echo "next:  python tools/package_shim.py   # 装进 release/*/runtime 并重编相关 pyc"
