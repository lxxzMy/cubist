# -*- coding: utf-8 -*-
"""DXGI adapter 枚举与选择：让 D3D11 设备与 OpenGL 上下文落在【同一块显卡】上。

为什么需要：
    wglDXOpenDeviceNV / wglDXRegisterObjectNV（WGC 换色纹理）和
    cudaGraphicsGLRegisterImage（CUDA 后端）都要求 D3D11 设备与 GL 上下文
    同 adapter。跨卡（双显卡笔记本、SBS 输出屏与被采屏接在不同卡上、
    Windows 给 pythonw.exe 的 GPU 偏好和 GL 不一致）时互操作直接失败，
    表现就是"有 SBS 但视差为 0 = 退化 2D"。

选择规则（顺序）：
    1. 有 GL renderer 信息 → 按名字匹配（GL 用哪块，D3D 就跟哪块）；
    2. 给了显示器矩形 → 选驱动这块屏的 adapter；
    3. 都没有 → 选 DedicatedVideoMemory 最大的（"最强"启发式）。
    匹配不到时退化为 None，由调用方回落到默认 adapter（保持旧行为）。
"""
import ctypes
from ctypes import wintypes as wt

from .rtcom import ComPtr, d3d11, guid_ptr

dxgi = ctypes.WinDLL("dxgi")

IID_IDXGIFactory1 = "770aae78-f26f-4dba-a829-253c83d1b387"
D3D11_DRIVER_TYPE_UNKNOWN = 0
D3D11_DRIVER_TYPE_HARDWARE = 1
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]


class ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", wt.WCHAR * 128),
                ("VendorId", wt.UINT), ("DeviceId", wt.UINT),
                ("SubSysId", wt.UINT), ("Revision", wt.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID), ("Flags", wt.UINT)]


class OUTPUT_DESC(ctypes.Structure):
    _fields_ = [("DeviceName", wt.WCHAR * 32),
                ("DesktopCoordinates", wt.RECT),
                ("AttachedToDesktop", wt.BOOL), ("Rotation", wt.UINT),
                ("Monitor", ctypes.c_void_p)]


class AdapterInfo:
    """一个 DXGI adapter（ComPtr 常驻，进程生命周期内不释放）。"""

    __slots__ = ("index", "name", "luid", "vram", "outputs", "_ptr")

    def __init__(self, index, ptr, desc, outputs):
        self.index = index
        self._ptr = ptr
        self.name = (desc.Description or "").strip()
        self.luid = ((desc.AdapterLuid.HighPart << 32) | desc.AdapterLuid.LowPart)
        self.vram = int(desc.DedicatedVideoMemory)
        self.outputs = outputs          # [(device_name, (l,t,r,b), attached)]

    @property
    def ptr(self):
        return self._ptr.p

    def rects(self):
        return [rc for (_n, rc, att) in self.outputs if att]

    def __repr__(self):
        return (f"AdapterInfo({self.index}, {self.name!r}, "
                f"vram={self.vram//(1<<20)}MB, outs={len(self.outputs)})")


_factory = None
_adapters = None
_gl_renderer = None
_capture_name = None        # 采集/互操作侧 D3D11 设备实际落在哪块卡


def _ensure_factory():
    global _factory
    if _factory is not None:
        return
    p = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(guid_ptr(IID_IDXGIFactory1), ctypes.byref(p))
    if hr:
        raise OSError(hr & 0xFFFFFFFF, "CreateDXGIFactory1")
    _factory = ComPtr(p.value)


def enum_adapters(refresh=False):
    """枚举所有 adapter（含输出显示器矩形）。首次调用后缓存。"""
    global _adapters
    if _adapters is not None and not refresh:
        return _adapters
    _ensure_factory()
    out = []
    i = 0
    while True:
        ap = ctypes.c_void_p()
        hr = _factory.call(12, ctypes.c_long,             # IDXGIFactory1::EnumAdapters1
                           (wt.UINT, ctypes.POINTER(ctypes.c_void_p)),
                           i, ctypes.byref(ap))
        if hr or not ap.value:
            break
        ad = ComPtr(ap.value)                             # 常驻：后面建设备要用
        desc = ADAPTER_DESC1()
        ad.call(10, ctypes.c_long,                        # IDXGIAdapter1::GetDesc1
                (ctypes.POINTER(ADAPTER_DESC1),), ctypes.byref(desc))
        outs = []
        j = 0
        while True:
            op = ctypes.c_void_p()
            hr2 = ad.call(7, ctypes.c_long,               # IDXGIAdapter::EnumOutputs
                          (wt.UINT, ctypes.POINTER(ctypes.c_void_p)),
                          j, ctypes.byref(op))
            if hr2 or not op.value:
                break
            o = ComPtr(op.value)
            od = OUTPUT_DESC()
            o.call(7, ctypes.c_long,                      # IDXGIOutput::GetDesc
                   (ctypes.POINTER(OUTPUT_DESC),), ctypes.byref(od))
            r = od.DesktopCoordinates
            outs.append(((od.DeviceName or "").strip(),
                         (r.left, r.top, r.right, r.bottom),
                         bool(od.AttachedToDesktop)))
            o.release()
            j += 1
        out.append(AdapterInfo(i, ad, desc, outs))
        i += 1
    _adapters = out
    return out


def set_gl_renderer(s):
    """viewer 建完 GL 上下文后调用（glGetString(GL_RENDERER)）。"""
    global _gl_renderer
    _gl_renderer = (s or "").strip()


def gl_renderer():
    return _gl_renderer


def note_capture_device(name):
    """采集/互操作侧建完 D3D11 设备后登记它实际落在哪块卡。"""
    global _capture_name
    if name:
        _capture_name = str(name).strip()


def capture_device_name():
    return _capture_name


def _norm_gl(r):
    r = (r or "").strip()
    for cut in ("/PCIe", "/SSE2", "/OpenGL", " Direct3D", "/D3D"):
        k = r.find(cut)
        if k > 0:
            r = r[:k]
    return r.strip().lower()


def _match_by_gl(ads):
    g = _norm_gl(_gl_renderer)
    if not g:
        return None
    for a in ads:
        n = a.name.lower()
        if n and (n in g or g in n):
            return a
    # 名字对不上时按厂商关键字兜底（带虚拟屏驱动/驱动改写名字的情况）
    for a in ads:
        n = a.name.lower()
        if "nvidia" in n and "nvidia" in g:
            return a
        if ("radeon" in n or "advanced micro" in n or "amd" in n) and \
                ("amd" in g or "radeon" in g):
            return a
        if "intel" in n and "intel" in g:
            return a
    return None


def _match_by_rect(ads, rect):
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    for a in ads:
        for rc in a.rects():
            if rc[0] <= cx < rc[2] and rc[1] <= cy < rc[3]:
                return a
    return None


def pick(monitor_rect=None, refresh=False):
    """按"GL 同卡 → 指定屏所在卡 → 屏所在卡 → 显存最大"选 adapter。

    D3D-only（无 GL 锚）时必须追随【拥有活动输出的 adapter】——采集
    （IDXGIOutputDuplication）只能建在屏所在的卡上；选显存最大的卡会在
    "屏接核显、独显无输出"的机器上直接取不到输出（D3D 环初始化失败）。
    refresh=True：显示器插拔/换屏后刷新枚举缓存（默认命中缓存）。
    """
    try:
        ads = [a for a in enum_adapters(refresh=refresh)
               if a.vram or a.outputs]
    except Exception:
        return None
    if not ads:
        return None
    a = _match_by_gl(ads)
    if a is not None:
        return a
    if monitor_rect:
        a = _match_by_rect(ads, monitor_rect)
        if a is not None:
            return a
    # 无 GL：屏所在卡优先。多个带输出的卡时取包含桌面原点 (0,0) 的主屏卡，
    # 否则取第一块有输出的卡；全无输出（headless）才回落显存最大。
    with_out = [x for x in ads if x.rects()]
    if with_out:
        for x in with_out:
            for rc in x.rects():
                if rc[0] <= 0 < rc[2] and rc[1] <= 0 < rc[3]:
                    return x
        return with_out[0]
    return max(ads, key=lambda x: x.vram)


def pick_reason(chosen):
    if chosen is None:
        return "未匹配（用默认 adapter）"
    g = _norm_gl(_gl_renderer)
    n = chosen.name.lower()
    if g and (n in g or g in n):
        return "与 GL 同卡"
    if chosen.rects():
        return "屏所在卡"
    return "显存最大（无输出）"


def create_device(adapter_info):
    """在指定 adapter 上建 D3D11 设备；adapter_info=None 时等同旧行为。"""
    dev = ctypes.c_void_p()
    fl = ctypes.c_uint()
    ctx = ctypes.c_void_p()
    if adapter_info is None:
        hr = d3d11.D3D11CreateDevice(
            None, D3D11_DRIVER_TYPE_HARDWARE, None,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, None, 0, 7,
            ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
    else:
        hr = d3d11.D3D11CreateDevice(
            adapter_info.ptr, D3D11_DRIVER_TYPE_UNKNOWN, None,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, None, 0, 7,
            ctypes.byref(dev), ctypes.byref(fl), ctypes.byref(ctx))
    if hr:
        raise OSError(hr & 0xFFFFFFFF, "D3D11CreateDevice")
    return ComPtr(dev.value), ComPtr(ctx.value), fl.value


IID_ID3D12Device = "189819f1-1db6-4b57-be54-1821339b85f7"
D3D_FEATURE_LEVEL_11_0 = 0xB000


def d3d12_support(adapter_info=None):
    """DirectML 依赖 DirectX 12。返回 (ok, hr, name)。

    显卡太老 / 驱动没装好 / 远程桌面虚拟显卡时 D3D12CreateDevice 会失败，
    这正是"DirectML 不可用"的最常见原因。
    """
    name = adapter_info.name if adapter_info is not None else None
    try:
        d3d12 = ctypes.WinDLL("d3d12")
        dev = ctypes.c_void_p()
        # AdapterInfo.ptr 本身就是 c_void_p，别再套一层（会 TypeError）
        _ap = adapter_info.ptr if adapter_info is not None else None
        hr = d3d12.D3D12CreateDevice(
            _ap, D3D_FEATURE_LEVEL_11_0,
            guid_ptr(IID_ID3D12Device), ctypes.byref(dev))
        ok = (hr == 0 and bool(dev.value))
        if dev.value:
            try:
                ComPtr(dev.value).release()
            except Exception:
                pass
        return ok, hr, name
    except Exception as e:
        return False, None, f"{name or '默认 adapter'}: {e}"


def describe_device(dev):
    try:
        dxdev = ComPtr(dev.qi("54ec77fa-1377-44e6-8c32-88fd5f44c84c"))  # IDXGIDevice
        ap = ctypes.c_void_p()
        dxdev.call(7, ctypes.c_long,                       # IDXGIDevice::GetAdapter
                   (ctypes.POINTER(ctypes.c_void_p),), ctypes.byref(ap))
        ad = ComPtr(ap.value)
        desc = ADAPTER_DESC1()
        ad.call(10, ctypes.c_long,
                (ctypes.POINTER(ADAPTER_DESC1),), ctypes.byref(desc))
        name = (desc.Description or "").strip()
        ad.release()
        dxdev.release()
        return name
    except Exception:
        return None


def summary_lines():
    """启动自检日志：GL / 各 adapter / 选中的 adapter。"""
    lines = []
    try:
        ads = enum_adapters()
        if _gl_renderer:
            lines.append(f"[3D] GL renderer   = {_gl_renderer}")
        for a in ads:
            rects = ", ".join(f"{rc[0]},{rc[1]} {rc[2]-rc[0]}x{rc[3]-rc[1]}"
                              for rc in a.rects()) or "-"
            lines.append(f"[3D] adapter[{a.index}]  = {a.name}  "
                         f"VRAM {a.vram // (1 << 20)}MB  LUID {a.luid:#x}  "
                         f"输出: {rects}")
        ch = pick()
        if ch is not None:
            lines.append(f"[3D] 选中 adapter   = [{ch.index}] {ch.name} "
                         f"（{pick_reason(ch)}）")
        if _capture_name:
            lines.append(f"[3D] 采集设备 adapter = {_capture_name}")
            if _gl_renderer:
                g = _norm_gl(_gl_renderer)
                n = _capture_name.lower()
                if n and not (n in g or g in n):
                    lines.append(
                        f"[3D] 警告：GL（{_gl_renderer}）与采集设备（{_capture_name}）"
                        f"不在同一块显卡 —— D3D11↔GL 互操作会失败，"
                        f"SBS 会退化成 2D。")
    except Exception as e:
        lines.append(f"[3D] adapter 枚举失败: {e}")
    return lines
