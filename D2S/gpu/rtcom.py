# -*- coding: utf-8 -*-
"""纯 ctypes 的 COM/WinRT/D3D11 最小运行时（M0 尖桩验证过的产品化拷贝）。

坑（详见 tools/gpu-spikes/ 与记忆）：
- WinRT 互操作接口只继承 IUnknown（槽位从 3 起）
- 行索引列（mlist/CA parent）与 GUID 全部来自 winmd 提取器，勿凭记忆改
"""
import ctypes
import uuid

# ---- DXGI HRESULT 唯一权威表（全库不得再地定义；历史勘误见 64d0699）----
# 2026-11-16 统一：dxgi_dup.py 曾把 WAIT_TIMEOUT 写成 0x887A0001（实为
# INVALID_CALL）。本调用点所有错误路径都 return None，错值无行为影响，
# 修正后仍逐位等价。
DXGI_ERROR_INVALID_CALL            = 0x887A0001
DXGI_ERROR_DEVICE_REMOVED          = 0x887A0005
DXGI_ERROR_DEVICE_HUNG             = 0x887A0006
DXGI_ERROR_DEVICE_RESET            = 0x887A0007
DXGI_ERROR_WAS_STILL_DRAWING       = 0x887A000A
DXGI_ERROR_NOT_CURRENTLY_AVAILABLE = 0x887A0022
DXGI_ERROR_WAIT_TIMEOUT            = 0x887A0027
DXGI_ERROR_ACCESS_LOST             = 0x887A0026

combase = ctypes.WinDLL("combase")
d3d11 = ctypes.WinDLL("d3d11")


class GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16),
                ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_str(cls, s):
        b = uuid.UUID(s).bytes_le
        return cls(int.from_bytes(b[0:4], "little"),
                   int.from_bytes(b[4:6], "little"),
                   int.from_bytes(b[6:8], "little"),
                   (ctypes.c_ubyte * 8)(*b[8:16]))


def guid_ptr(s):
    return ctypes.byref(GUID.from_str(s))


_PVP = ctypes.POINTER(ctypes.c_void_p)
_VFN = {}
_FNP = {}   # (vtable, slot, restype, atypes) → 已绑定的函数指针实例


def _vfn(restype, atypes):
    """按签名缓存 WINFUNCTYPE 类型。

    每次 ComPtr.call() 现建 WINFUNCTYPE 都会生成一个**新类型**（PyCFuncPtrType
    + StgDict + 字段描述符，彼此成环）→ 每帧十几次 COM 调用就是上千个循环
    垃圾，gen2 每次 20~40ms（2026-11-22 实测，GC 探针抓到的 dict/c_void_p/
    CField 大头就是它）。类型与参数表绑定，缓存后行为逐位不变。
    """
    key = (restype, atypes)
    t = _VFN.get(key)
    if t is None:
        t = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *atypes)
        _VFN[key] = t
    return t


class ComPtr:
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = ctypes.c_void_p(p)

    def _vt(self):
        vt_raw = ctypes.cast(self.p, _PVP)[0]
        return ctypes.cast(ctypes.c_void_p(vt_raw), _PVP)

    def call(self, slot, restype, atypes, *args):
        # 函数指针实例也缓存（key 含 vtable 地址）：vtable 是 COM DLL 里的
        # 静态只读数组，地址唯一且随进程存活 → 按它缓存不会串类；同时
        # 免掉每帧 20+ 个 CFuncPtr 实例及其 _objects 字典（GC 探针里
        # dict+c_void_p 12 万对的大头）。
        vt = int(ctypes.cast(self.p, _PVP)[0])
        key = (vt, slot, restype, atypes)
        fn = _FNP.get(key)
        if fn is None:
            fn = _vfn(restype, atypes)(self._vt()[slot])
            _FNP[key] = fn
        return fn(self.p, *args)

    def qi(self, iid_str):
        out = ctypes.c_void_p()
        hr = self.call(0, ctypes.c_long,
                       (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                       guid_ptr(iid_str), ctypes.byref(out))
        if hr:
            raise OSError(hr, f"QueryInterface({iid_str})")
        return out.value

    def release(self):
        return self.call(2, ctypes.c_uint32, ())


# ---------------- WinRT 基础 ----------------
def ro_init():
    return combase.RoInitialize(1)   # RO_INIT_MULTITHREADED


def hstr(s):
    out = ctypes.c_void_p()
    hr = combase.WindowsCreateString(s, len(s), ctypes.byref(out))
    if hr:
        raise OSError(hr, f"WindowsCreateString({s})")
    return out


def hstr_free(h):
    combase.WindowsDeleteString(h)


def activation_factory(runtime_name, iid_str):
    h = hstr(runtime_name)
    out = ctypes.c_void_p()
    hr = combase.RoGetActivationFactory(h, guid_ptr(iid_str), ctypes.byref(out))
    hstr_free(h)
    if hr:
        raise OSError(hr, f"RoGetActivationFactory({runtime_name}, {iid_str})")
    return ComPtr(out.value)


# ---------------- D3D11（vtable 槽位来自 mingw d3d11.h，勿凭记忆改）-----
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_DRIVER_TYPE_HARDWARE = 1
DXGI_FORMAT_B8G8R8A8_UNORM = 87
D3D11_USAGE_DEFAULT = 0
D3D11_CPU_ACCESS_READ = 0x20000


class TexDesc(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_uint), ("Height", ctypes.c_uint),
                ("MipLevels", ctypes.c_uint), ("ArraySize", ctypes.c_uint),
                ("Format", ctypes.c_uint),
                ("SampleCount", ctypes.c_uint), ("SampleQuality", ctypes.c_uint),
                ("Usage", ctypes.c_uint), ("BindFlags", ctypes.c_uint),
                ("CPUAccessFlags", ctypes.c_uint), ("MiscFlags", ctypes.c_uint)]


class Box(ctypes.Structure):
    """D3D11_BOX（CopySubresourceRegion 的源裁剪盒）"""
    _fields_ = [("left", ctypes.c_uint), ("top", ctypes.c_uint),
                ("front", ctypes.c_uint), ("right", ctypes.c_uint),
                ("bottom", ctypes.c_uint), ("back", ctypes.c_uint)]


class InitData(ctypes.Structure):
    _fields_ = [("pSysMem", ctypes.c_void_p), ("SysMemPitch", ctypes.c_uint),
                ("SysMemSlicePitch", ctypes.c_uint)]


def create_device(adapter_ptr=None):
    """建 D3D11 设备。

    adapter_ptr：IDXGIAdapter*（None = 默认 adapter，保持旧行为）。
    显式给 adapter 时 driver_type 必须传 UNKNOWN，否则 D3D11 会忽略
    adapter 参数（这是 D3D11CreateDevice 的硬性要求）。
    """
    dev = ctypes.c_void_p()
    fl = ctypes.c_uint()
    ctx = ctypes.c_void_p()
    if adapter_ptr is None:
        hr = d3d11.D3D11CreateDevice(
            None, D3D11_DRIVER_TYPE_HARDWARE, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None, 0, 7, ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
    else:
        p = adapter_ptr if isinstance(adapter_ptr, ctypes.c_void_p) \
            else ctypes.c_void_p(int(adapter_ptr))
        hr = d3d11.D3D11CreateDevice(
            p, 0, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,   # 0 = D3D_DRIVER_TYPE_UNKNOWN
            None, 0, 7, ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
        if hr:
            # 指定 adapter 失败：退回默认，别把整个采集打死
            hr = d3d11.D3D11CreateDevice(
                None, D3D11_DRIVER_TYPE_HARDWARE, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None, 0, 7, ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
    if hr:
        raise OSError(hr, "D3D11CreateDevice")
    return ComPtr(dev.value), ComPtr(ctx.value), fl.value


def create_tex(dev, desc, init=None):
    out = ctypes.c_void_p()
    hr = dev.call(5, ctypes.c_long,                     # CreateTexture2D
                  (ctypes.POINTER(TexDesc), ctypes.c_void_p,
                   ctypes.POINTER(ctypes.c_void_p)),
                  ctypes.byref(desc), init, ctypes.byref(out))
    if hr:
        raise OSError(hr, "CreateTexture2D")
    return ComPtr(out.value)


def ctx_copy_resource(ctx, dst, src):
    ctx.call(47, None, (ctypes.c_void_p, ctypes.c_void_p), dst.p, src.p)


def ctx_copy_subresource(ctx, dst, src, box, dst_x=0, dst_y=0):
    """CopySubresourceRegion（槽 46）：src 的 box 区域拷到 dst 的 (dst_x,dst_y)。"""
    ctx.call(46, None,
             (ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
              ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
              ctypes.POINTER(Box)),
             dst.p, 0, dst_x, dst_y, 0, src.p, 0, ctypes.byref(box))


def device_to_winrt(dev):
    """ID3D11Device -> (IDirect3DDevice IInspectable*, IDXGIDevice*)。"""
    dxgi = dev.qi("54ec77fa-1377-44e6-8c32-88fd5f44c84c")  # IDXGIDevice
    out = ctypes.c_void_p()
    hr = d3d11.CreateDirect3D11DeviceFromDXGIDevice(
        ctypes.c_void_p(dxgi), ctypes.byref(out))
    if hr:
        raise OSError(hr, "CreateDirect3D11DeviceFromDXGIDevice")
    return out.value, dxgi
