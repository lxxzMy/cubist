# -*- coding: utf-8 -*-
"""HLSL 编译器块（迁移 P0）：DXC/SM6 首选（dxcompiler.dll + dxil.dll，随
SDK bin 分发），FXC/d3dcompiler_47（系统 DLL，SM5_0）兜底；产物按源码 hash
缓存 .cso 到磁盘（DIBR 大着色器不逐启动重编）。

反馈定案：着色器编译走 DXC/SM6；P0 实测编译耗时。编译器选择记录在
Cso.compiler（"dxc" | "fxc"），parity 测试两侧可对照。
"""
import ctypes
import hashlib
import os

from .rtcom import ComPtr, guid_ptr
from .d3d12 import (DXC_DIRS, CLSID_DxcCompiler, IID_IDxcCompiler3,
                    IID_IDxcUtils, IID_IDxcBlob, S_DXC3, S_DXCUTILS,
                    S_DXCRESULT, S_BLOB)
from .d3d12 import IID_IDxcOperationResult, IID_IDxcResult

k32 = ctypes.WinDLL("kernel32")

_CACHE_DIRS = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "logs", "dxc-cache"),
)

CLSID_DxcValidator = "8ca3e215-f728-4cf3-8cdd-88af917587a1"   # dxcapi.h
IID_IDxcValidator = "a6e82bd2-1fd7-4826-9811-2857e797f49a"    # dxcapi.h L875


class DxcBuffer(ctypes.Structure):
    """dxcapi.h DxcBuffer/DxcText：{Ptr, Size, Encoding(0=自动/ANSI)}。"""
    _fields_ = [("Ptr", ctypes.c_void_p), ("Size", ctypes.c_size_t),
                ("Encoding", ctypes.c_uint)]


def _load_dll(cands, name):
    for d in DXC_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return ctypes.WinDLL(p)
    return ctypes.WinDLL(name)   # 系统 PATH 兜底


_dxc = None


def _dxc_dll():
    global _dxc
    if _dxc is None:
        _dxc = _load_dll(DXC_DIRS, "dxcompiler.dll")
    return _dxc


def _cache_dir():
    d = _CACHE_DIRS[0]
    os.makedirs(d, exist_ok=True)
    return d


class Cso:
    """编译产物：DXIL/DXBC 字节 + 元数据。data_ptr/nbytes 供 PSO_DESC。"""

    __slots__ = ("_copy", "compiler", "profile", "data_ptr", "nbytes")

    def __init__(self, blob_ptr, size, compiler, profile):
        self._copy = bytes((ctypes.c_ubyte * size).from_address(blob_ptr))
        self.compiler = compiler
        self.profile = profile
        self.data_ptr = ctypes.cast(self._copy, ctypes.c_void_p).value
        self.nbytes = len(self._copy)


# ---------------------------------------------------------------------------
# DXC 路径（IDxcCompiler3.Compile → IDxcResult.GetResult）
# ---------------------------------------------------------------------------
CLSID_DxcUtils = "6245d6af-66e0-48fd-80b4-4d271796748c"   # = CLSID_DxcLibrary


def _dxc_compile(src, profile, entry):
    """IDxcCompiler3 编译（SM6，D1 修复版）。

    官方签名 6 参数（dxcapi.h L846-856）：Compile(DxcBuffer*, LPCWSTR*,
    UINT32, IDxcIncludeHandler*, REFIID, LPVOID*)。旧实现传 8 个参数 +
    pSource 用 IDxcBlob* → riid=0/ppResult=NULL → E_INVALIDARG 实锤。
    """
    dll = _dxc_dll()
    out = ctypes.c_void_p()
    hr = dll.DxcCreateInstance(guid_ptr(CLSID_DxcCompiler),
                               guid_ptr(IID_IDxcCompiler3), ctypes.byref(out))
    if hr:
        raise OSError(hr, "DxcCreateInstance(IDxcCompiler3)")
    comp = ComPtr(out.value)
    sbytes = src.encode() if isinstance(src, str) else bytes(src)
    sb = (ctypes.c_ubyte * len(sbytes)).from_buffer_copy(sbytes)
    buf = DxcBuffer(ctypes.cast(sb, ctypes.c_void_p), len(sbytes), 0)
    # 不加 -Vd：dxil.dll 已在位（runtime/），DXC 走正常 DXIL 验证+签名——
    # 未签名 DXIL 会被 D3D12 CreateGraphicsPipelineState 拒收（E_INVALIDARG 实锤）
    args = ["-T", profile, "-E", entry]
    arr = (ctypes.c_wchar_p * len(args))(*args)
    res = ctypes.c_void_p()
    hr = comp.call(S_DXC3["Compile"], ctypes.c_long,
                   (ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p),
                    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p)),
                   ctypes.byref(buf), arr, len(args), None,
                   guid_ptr(IID_IDxcResult), ctypes.byref(res))
    comp.release()
    if hr:
        raise OSError(hr, "IDxcCompiler3.Compile")
    rop = ComPtr(res.value)
    status = ctypes.c_int()
    rop.call(S_DXCRESULT["GetStatus"], ctypes.c_long,
             (ctypes.POINTER(ctypes.c_int),), ctypes.byref(status))
    if status.value != 0:
        errp = ctypes.c_void_p()
        rop.call(S_DXCRESULT["GetErrorBuffer"], ctypes.c_long,
                 (ctypes.POINTER(ctypes.c_void_p),), ctypes.byref(errp))
        msg = ""
        if errp.value:
            eb = ComPtr(errp.value)
            p = eb.call(S_BLOB["GetBufferPointer"], ctypes.c_void_p, ())
            n = eb.call(S_BLOB["GetBufferSize"], ctypes.c_size_t, ())
            msg = ctypes.string_at(p, n).decode("utf-8", "replace")[:400]
            eb.release()
        rop.release()
        raise RuntimeError(f"DXC 编译失败({profile}): {msg}")
    obj = ctypes.c_void_p()
    hr = rop.call(S_DXCRESULT["GetResult"], ctypes.c_long,
                  (ctypes.POINTER(ctypes.c_void_p),), ctypes.byref(obj))
    rop.release()
    if hr or not obj.value:
        raise OSError(hr or -1, "OperationResult.GetResult")
    ob = ComPtr(obj.value)
    ptr = ob.call(S_BLOB["GetBufferPointer"], ctypes.c_void_p, ())
    n = ob.call(S_BLOB["GetBufferSize"], ctypes.c_size_t, ())
    return ptr, n, ob   # ob 生存期覆盖调用方拷贝


# ---------------------------------------------------------------------------
# FXC 兜底（d3dcompiler_47，SM5_0）
# ---------------------------------------------------------------------------
def _fxc_compile(src, profile, entry):
    d3dc = ctypes.WinDLL("d3dcompiler_47")
    pfn = d3dc.D3DCompile
    pfn.restype = ctypes.HRESULT
    pfn.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p))
    blob = ctypes.c_void_p()
    err = ctypes.c_void_p()
    hr = pfn(src.encode(), len(src), None, None, None, entry.encode(),
             profile.encode(), 0, 0, ctypes.byref(blob), ctypes.byref(err))
    if hr:
        msg = ""
        if err.value:
            from .rtcom import ComPtr as _C
            eb = _C(err.value)
            p = eb.call(3, ctypes.c_void_p, ())
            n = eb.call(4, ctypes.c_size_t, ())
            msg = ctypes.string_at(p, n).decode("utf-8", "replace")[:400]
        raise RuntimeError(f"FXC 编译失败({profile}): {msg}")
    bc = ComPtr(blob.value)
    ptr = bc.call(3, ctypes.c_void_p, ())   # ID3DBlob::GetBufferPointer
    n = bc.call(4, ctypes.c_size_t, ())     # ID3DBlob::GetBufferSize
    return ptr, n, bc


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
_dxc_ver = None


def _dxc_version_tag():
    """dxcompiler.dll 的 size+mtime（缓存 key 维度：SDK 升级自动失效旧 CSO）。"""
    global _dxc_ver
    if _dxc_ver is None:
        for d in DXC_DIRS:
            p = os.path.join(d, "dxcompiler.dll")
            if os.path.exists(p):
                st = os.stat(p)
                _dxc_ver = f"{st.st_size}:{int(st.st_mtime)}"
                break
        else:
            _dxc_ver = "none"
    return _dxc_ver


def compile_shader(name, src, kind="ps", compiler="dxc", force=False):
    """编译并缓存。返回 Cso。kind: ps|vs；profile 自动选择。

    compiler="dxc"（默认，SM6.x，-Vd 跳过签名验证——未签名 DXIL 在
    release 运行时合法，debug layer 仅警告）；dxcompiler.dll 不可用 →
    记录一行并 FXC SM5_0 兜底（系统 d3dcompiler_47）。编译错误
    （GetStatus!=0）不兜底——真问题必须可见。compiler="fxc" 显式可选
    （A/B 对照）。
    """
    # key 维度：compiler + 版本（dxcompiler size:mtime）+ 编译 args（-Vd 等
    # 参数变更必须失效旧 CSO——未签名 DXIL 会被 D3D12 拒收的实锤）
    ver = _dxc_version_tag() if compiler == "dxc" else ""
    flags = "T{E}:Nodef" if compiler == "dxc" else ""
    h = hashlib.sha1(f"{compiler}:{ver}:{flags}:{kind}:{src}"
                     .encode()).hexdigest()[:16]
    cache = os.path.join(_cache_dir(), f"{name}.{h}.{compiler}.cso")
    if not force and os.path.exists(cache):
        data = open(cache, "rb").read()
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        return Cso(ctypes.cast(buf, ctypes.c_void_p).value, len(data),
                   compiler, "")
    if compiler == "dxc":
        profile = {"ps": "ps_6_0", "vs": "vs_6_0", "cs": "cs_6_0"}[kind]
        try:
            ptr, n, keep = _dxc_compile(src, profile, entry_name(kind))
            c = Cso(ptr, n, "dxc", profile)
        except RuntimeError:
            raise    # 编译错误不兜底（真问题必须可见）
        except OSError as e:   # dxcompiler.dll 加载/创建失败 → FXC 兜底
            print(f"[dxc] 不可用（{e}）→ FXC/SM5_0 兜底: {name}", flush=True)
            return compile_shader(name, src, kind, "fxc", force)
    else:
        profile = {"ps": "ps_5_0", "vs": "vs_5_0", "cs": "cs_5_0"}[kind]
        ptr, n, keep = _fxc_compile(src, profile, entry_name(kind))
        c = Cso(ptr, n, "fxc", profile)
    open(cache, "wb").write(bytes(c._copy))
    return c


def entry_name(kind):
    return {"vs": "VSMain", "ps": "PSMain", "cs": "CSMain"}[kind]


# 全屏三角 VS（SM6/SM5 通用；SV_VertexID 生成三顶点，免 VB/IB）
FULLSCREEN_VS = """
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VOut VSMain(uint vid : SV_VertexID) {
    VOut o;
    float2 p = float2((vid << 1) & 2u, vid & 2u);
    o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = p;
    return o;
}
"""
