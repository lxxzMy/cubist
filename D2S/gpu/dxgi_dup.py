"""DXGI Desktop Duplication：取帧 + 光标形状（指针元数据）。

为什么需要它
------------
系统光标一旦被 SetSystemCursor 换成"几乎透明"的替身：
  - GetCursorInfo/GetIconInfo 读回来的是规范化副本（实测：颜色/尺寸/mask 全被重置）
  - 于是"看不见"和"知道形状"在 Win32 光标 API 层面互斥

而桌面复制（IDXGIOutputDuplication）的光标信息来自【桌面合成器】：
  GetFramePointerShape() 直接给出当前光标的位图 + 热点，
  与光标表无关 —— 隐藏了也能拿到正确形状。

本模块只做两件事：
  1. 建立 duplication 会话（一个输出只能有一个会话）
  2. 轮询"当前光标位图 + 位置"，转成 RGBA（非预乘）
"""
import ctypes
import ctypes.wintypes as wt

from .rtcom import ComPtr, GUID, guid_ptr, d3d11, DXGI_ERROR_WAIT_TIMEOUT, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, DXGI_ERROR_ACCESS_LOST

dxgi = ctypes.WinDLL("dxgi")

D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_SDK_VERSION = 7
DXGI_FORMAT_B8G8R8A8_UNORM = 87

IID_IDXGIDevice = "54EC77FA-1377-44E6-8C32-88FD5F44C84C"
IID_IDXGIOutput1 = "00CDDEA8-939B-4B83-A340-A685226666CC"

# DXGI_OUTDUPL_POINTER_SHAPE_TYPE
SHAPE_MONOCHROME = 1
SHAPE_COLOR = 2
SHAPE_MASKED_COLOR = 4


class DXGI_OUTDUPL_POINTER_POSITION(ctypes.Structure):
    _fields_ = [("Position", wt.POINT), ("Visible", wt.BOOL)]


class DXGI_OUTDUPL_FRAME_INFO(ctypes.Structure):
    _fields_ = [("LastPresentTime", ctypes.c_longlong),
                ("LastMouseUpdateTime", ctypes.c_longlong),
                ("AccumulatedFrames", wt.UINT),
                ("RectsCoalesced", wt.BOOL),
                ("ProtectedContentMaskedOut", wt.BOOL),
                ("PointerPosition", DXGI_OUTDUPL_POINTER_POSITION),
                ("TotalMetadataBufferSize", wt.UINT),
                ("PointerShapeBufferSize", wt.UINT)]


class DXGI_OUTDUPL_POINTER_SHAPE_INFO(ctypes.Structure):
    _fields_ = [("Type", wt.UINT), ("Width", wt.UINT), ("Height", wt.UINT),
                ("Pitch", wt.UINT), ("HotSpot", wt.POINT)]


class PointerFrame:
    """一次轮询的结果。"""
    __slots__ = ("visible", "x", "y", "rgba", "hotspot", "shape_type", "w", "h")

    def __init__(self, visible, x, y, rgba=None, hotspot=(0, 0),
                 shape_type=0, w=0, h=0):
        self.visible = visible
        self.x, self.y = x, y
        self.rgba = rgba
        self.hotspot = hotspot
        self.shape_type, self.w, self.h = shape_type, w, h

    def __repr__(self):
        return (f"PointerFrame(visible={self.visible}, pos=({self.x},{self.y}), "
                f"shape={self.shape_type} {self.w}x{self.h}, hotspot={self.hotspot})")


def _mono_to_rgba(buf, w, h, pitch):
    """单色光标：上半是 AND mask，下半是 XOR mask（各 h/2 行）。"""
    import numpy as np
    n = h // 2
    rgbs = np.zeros((n, w, 4), dtype=np.uint8)
    for y in range(n):
        and_row = buf[y * pitch:(y + 1) * pitch]
        xor_row = buf[(n + y) * pitch:(n + y + 1) * pitch]
        for x in range(w):
            a_bit = (and_row[x >> 3] >> (7 - (x & 7))) & 1
            x_bit = (xor_row[x >> 3] >> (7 - (x & 7))) & 1
            if a_bit and not x_bit:
                continue                      # 透明
            if a_bit and x_bit:
                rgbs[y, x] = (255, 255, 255, 255)   # 反色 → 用白色近似
            elif not a_bit and x_bit:
                rgbs[y, x] = (255, 255, 255, 255)
            else:
                rgbs[y, x] = (0, 0, 0, 255)
    return rgbs


def _color_to_rgba(buf, w, h, pitch, masked=False):
    import numpy as np
    raw = np.frombuffer(bytes(buf), dtype=np.uint8)
    need = pitch * h
    if raw.size < need:
        return None
    img = raw[:need].reshape(h, pitch)[:, :w * 4].reshape(h, w, 4).copy()
    # BGRA -> RGBA
    img = img[:, :, [2, 1, 0, 3]]
    if masked:
        # MASKED_COLOR：alpha=0 的像素表示"反色"，这里按不透明黑处理
        a = img[:, :, 3]
        img[a == 0] = (0, 0, 0, 255)
    return img


class DxgiPointerSource:
    """一个显示输出上的 duplication 会话，用来读光标形状与位置。"""

    def __init__(self, monitor_index=0):
        self._dev = None
        self._ctx = None
        self._dup = None
        self._monitor = monitor_index
        self._shape_cache = {}
        self._open()

    # ------------------------------------------------------------- 建立会话
    def _open(self):
        # 选和 GL 同一块显卡（拿不到 GL 信息时按显存最大/输出所在卡），
        # 让后续的 D3D11↔GL 互操作（wglDXOpenDeviceNV）能成立。
        # 跨卡时互操作必失败 → 表现为"有 SBS 但视差为 0"。
        try:
            from . import placement as _pl
            _adv = _pl.pick_adapter()
        except Exception:
            _adv = None
        dev = ctypes.c_void_p()
        ctx = ctypes.c_void_p()
        feat = ctypes.c_uint(0)
        hr = 1
        if _adv is not None:
            hr = d3d11.D3D11CreateDevice(
                _adv.ptr, 0, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,   # 0 = UNKNOWN
                None, 0, D3D11_SDK_VERSION, ctypes.byref(dev),
                ctypes.byref(feat), ctypes.byref(ctx))
        if _adv is None or hr:
            hr = d3d11.D3D11CreateDevice(
                None, D3D_DRIVER_TYPE_HARDWARE, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None, 0, D3D11_SDK_VERSION, ctypes.byref(dev),
                ctypes.byref(feat), ctypes.byref(ctx))
        if hr:
            raise OSError(hr, "D3D11CreateDevice")
        self._dev = ComPtr(dev.value)
        self._ctx = ComPtr(ctx.value)
        try:
            from .adapter import describe_device as _desc, \
                note_capture_device as _note
            self.adapter_name = _desc(self._dev)
            _note(self.adapter_name)
            print(f"[3D] DXGI 采集设备 adapter = {self.adapter_name}", flush=True)
        except Exception:
            self.adapter_name = None

        dxdev = ComPtr(self._dev.qi(IID_IDXGIDevice))
        adapter_pp = ctypes.c_void_p()
        dxdev.call(7, ctypes.c_long, (ctypes.POINTER(ctypes.c_void_p),),
                   ctypes.byref(adapter_pp))          # IDXGIDevice::GetAdapter
        adapter = ComPtr(adapter_pp.value)
        # 目标 adapter 的输出个数可能和 mss 的序号对不上（多卡/虚拟屏），
        # 先按给定序号，失败再顺序找第一个可用输出。
        out_pp = ctypes.c_void_p()
        _cand = [self._monitor] + [k for k in range(4) if k != self._monitor]
        hr = 1
        for _k in _cand:
            out_pp = ctypes.c_void_p()
            hr = adapter.call(7, ctypes.c_long,            # IDXGIAdapter::EnumOutputs
                              (wt.UINT, ctypes.POINTER(ctypes.c_void_p)),
                              _k, ctypes.byref(out_pp))
            if not hr and out_pp.value:
                break
        if hr:
            raise OSError(hr, "EnumOutputs")
        out1 = ComPtr(ComPtr(out_pp.value).qi(IID_IDXGIOutput1))
        dup_pp = ctypes.c_void_p()
        # IDXGIOutput1 的 DuplicateOutput 槽位 = 22（实测逐个试探确认：
        # 19 是 GetDisplayModeList1，20 崩溃，21 超时，22 成功拿到对象）
        hr = out1.call(22, ctypes.c_long,              # DuplicateOutput
                       (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                       self._dev.p, ctypes.byref(dup_pp))
        if hr:
            raise OSError(hr, "DuplicateOutput")
        self._dup = ComPtr(dup_pp.value)
        for x in (out1, adapter, dxdev):
            try:
                x.release()
            except Exception:
                pass

    # ------------------------------------------------------------- 轮询
    def poll(self, timeout_ms=0):
        """取一次帧信息（并释放帧）。返回 PointerFrame 或 None。"""
        if self._dup is None or not getattr(self._dup, "p", None) or not self._dup.p.value:
            return None
        try:
            return self._poll_inner(timeout_ms)
        except OSError as e:
            # ACCESS_LOST / 设备丢失：重建会话，本次返回 None
            if getattr(e, "errno", 0) == DXGI_ERROR_ACCESS_LOST:
                self._reopen()
            return None

    def _poll_inner(self, timeout_ms):
        info = DXGI_OUTDUPL_FRAME_INFO()
        res_pp = ctypes.c_void_p()
        # IDXGIOutputDuplication 继承 IDXGIObject：
        # GetDesc(7) AcquireNextFrame(8) DirtyRects(9) MoveRects(10)
        # PointerShape(11) MapDesktopSurface(12) UnMap(13) ReleaseFrame(14)
        hr = self._dup.call(8, ctypes.c_long,          # AcquireNextFrame
                            (wt.UINT, ctypes.POINTER(DXGI_OUTDUPL_FRAME_INFO),
                             ctypes.POINTER(ctypes.c_void_p)),
                            timeout_ms, ctypes.byref(info), ctypes.byref(res_pp))
        if hr:
            if hr in (DXGI_ERROR_WAIT_TIMEOUT, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE):
                return None
            if hr == DXGI_ERROR_ACCESS_LOST:
                self._reopen()
            return None
        try:
            if res_pp.value:
                ComPtr(res_pp.value).release()
            if info.PointerShapeBufferSize == 0:
                return PointerFrame(bool(info.PointerPosition.Visible),
                                    info.PointerPosition.Position.x,
                                    info.PointerPosition.Position.y)
            return self._read_shape(info)
        finally:
            self._dup.call(14, ctypes.c_long, ())      # ReleaseFrame

    def _read_shape(self, info):
        need = info.PointerShapeBufferSize
        buf = (ctypes.c_ubyte * need)()
        got = wt.UINT(0)
        w, h = wt.UINT(0), wt.UINT(0)
        sh = DXGI_OUTDUPL_POINTER_SHAPE_INFO()
        hr = self._dup.call(11, ctypes.c_long,         # GetFramePointerShape
                            (wt.UINT, ctypes.c_void_p, ctypes.POINTER(wt.UINT),
                             ctypes.POINTER(wt.UINT), ctypes.POINTER(wt.UINT),
                             ctypes.POINTER(DXGI_OUTDUPL_POINTER_SHAPE_INFO)),
                            need, ctypes.cast(buf, ctypes.c_void_p),
                            ctypes.byref(got), ctypes.byref(w), ctypes.byref(h),
                            ctypes.byref(sh))
        if hr:
            return PointerFrame(bool(info.PointerPosition.Visible),
                                info.PointerPosition.Position.x,
                                info.PointerPosition.Position.y)
        key = (sh.Type, sh.Width, sh.Height, bytes(buf[:got.value]))
        rgba = self._shape_cache.get(key)
        if rgba is None:
            if sh.Type in (SHAPE_COLOR, SHAPE_MASKED_COLOR):
                rgba = _color_to_rgba(buf, sh.Width, sh.Height, sh.Pitch,
                                      masked=(sh.Type == SHAPE_MASKED_COLOR))
            elif sh.Type == SHAPE_MONOCHROME:
                rgba = _mono_to_rgba(buf, sh.Width, sh.Height, sh.Pitch)
            if rgba is not None:
                if len(self._shape_cache) > 16:
                    self._shape_cache.clear()
                self._shape_cache[key] = rgba
        return PointerFrame(bool(info.PointerPosition.Visible),
                            info.PointerPosition.Position.x,
                            info.PointerPosition.Position.y,
                            rgba=rgba, hotspot=(sh.HotSpot.x, sh.HotSpot.y),
                            shape_type=sh.Type, w=sh.Width, h=sh.Height)

    def _reopen(self):
        try:
            self._dup.release()
        except Exception:
            pass
        self._dup = None
        try:
            self._open()
        except Exception:
            pass

    def close(self):
        for x in (self._dup, self._ctx, self._dev):
            try:
                if x is not None:
                    x.release()
            except Exception:
                pass
        self._dup = self._ctx = self._dev = None
