# -*- coding: utf-8 -*-
"""DDA 采集桥（迁移 T1）：D3D11 DDA 环纹理 → D3D12 管线消费。

producer 侧（dxgi_gpu_grab，D2S_RENDER=d3d 门控）：环纹理加
MISC_SHARED_NTHANDLE|KEYEDMUTEX(0x800|0x100，D3D11 值)，建环时对每个槽
AcquireSync(0) 一次并一直持有，每帧只做 拷贝 + ctx4.Signal(帧序号)。
consumer 侧（本模块）：D3D11 SHARED fence（0x2）创建 + CreateSharedHandle
（槽 7）交给采集侧；D3D12 侧 OpenSharedHandle（槽 32）一次性缓存
（逐帧重 open ~0.03ms/帧不可接受）；LUID 校验 fail-loud；queue.Wait
门控（GPU 端顺序，CPU 不等）。

坑位背书（全部生产实锤，见 tools/d3d_microbench.py 探针 E 与
docs/zero-copy-migration.md）：
- D3D11/D3D12 枚举不同源：KM 0x100 vs 0x1000、fence SHARED 0x2 vs 0x1
- IDXGIResource1::CreateSharedHandle = 槽 13（槽 7 是 GetSharedHandle）
- KM 纹理不 AcquireSync 就写 = 拷贝被静默吞掉（每槽的第一次）；IDXGIKeyedMutex
  的虚表槽位实测 7=GetSharedHandle、8=AcquireSync、9=ReleaseSync
  （dxgi_gpu_grab.S_KEYED_*，2026-11-28 解跳板核对）
- OpenSharedHandle 打开的资源用隐式状态提升，发显式屏障曾致 Close
  E_INVALIDARG（d3drender.wrap_external 的 no_barrier 语义）
- D3D11/D3D12 设备必须同 LUID（跨卡共享句柄无效）
"""
import ctypes
import os

from .rtcom import ComPtr, guid_ptr, create_device
from . import placement


IID_IDXGIResource1 = "30961379-4609-4a41-998e-54fe567ee0c1"
IID_D3D11Device5 = "8ffde202-a0e7-45df-9e01-e837801b5ea0"
IID_D3D11Fence = "affde9d1-1df7-4bb7-8a34-0f46251dab80"
IID_IDXGIDevice = "54ec77fa-1377-44e6-8c32-88fd5f44c84c"
IID_IDXGIAdapter = "2411e7e1-12ac-4ccf-bd14-9798e8534fc0"
CTX4_IID = "917600da-f58c-4c33-98d8-3e15b390fa24"
IID_D3D12Fence = "0a753dcf-c4d8-4b91-adf6-be5a60d95a76"
IID_D3D12Resource = "696442be-a72e-4059-bc79-5b5c98040fad"

DXGI_SHARED_RESOURCE_READ = 0x80000000
DXGI_SHARED_RESOURCE_WRITE = 0x1
GENERIC_ALL = 0x10000000

# D3D11 侧值（与 D3D12 不同源！）
D3D11_MISC_NTHANDLE = 0x800
D3D11_MISC_KEYEDMUTEX = 0x100
D3D11_FENCE_FLAG_SHARED = 0x2

S_DEV5_CREATE_FENCE = 68     # ID3D11Device5::CreateFence
S_FENCE_CREATE_SHARED = 7    # ID3D11Fence::CreateSharedHandle
S_RES_CREATE_SHARED = 13     # IDXGIResource1::CreateSharedHandle
S_DEV12_OPEN_SHARED = 32     # ID3D12Device::OpenSharedHandle
S_QI_IDXGIRESOURCE1 = 0      # QueryInterface（IUnknown 槽 0，ComPtr.qi 封装）


def d3d_ring_misc():
    """DDA 环纹理 MiscFlags（D3D11 值）——仅 D2S_RENDER=d3d 时带共享标志。

    GL 默认路径保持 0：NTHANDLE|KM 纹理直接注册 wglDX interop 在全新
    会话必败（P1 实锤），加标志会破坏生产 GL 路径。
    """
    if os.environ.get("D2S_RENDER", "").strip().lower() == "d3d":
        return D3D11_MISC_NTHANDLE | D3D11_MISC_KEYEDMUTEX
    return 0


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", ctypes.c_uint), ("LuidHigh", ctypes.c_int),
                ("Flags", ctypes.c_uint)]


def dev11_luid(dev11):
    """D3D11Device → LUID（int；IDXGIDevice→GetAdapter→GetDesc）。

    槽位：IDXGIDevice::GetAdapter=7；IDXGIAdapter::GetDesc=10。
    """
    dxdev = ctypes.c_void_p(dev11.qi(IID_IDXGIDevice))
    try:
        ap = ctypes.c_void_p()
        hr = ComPtr(dxdev.value).call(7, ctypes.c_long,
                                      (ctypes.POINTER(ctypes.c_void_p),),
                                      ctypes.byref(ap))
        if hr:
            return 0
        ad = ComPtr(ap.value)
        desc = DXGI_ADAPTER_DESC1()
        hr = ad.call(10, ctypes.c_long, (ctypes.POINTER(DXGI_ADAPTER_DESC1),),
                     ctypes.byref(desc))
        ad.release()
        if hr:
            return 0
        return ((desc.LuidHigh << 32) | desc.LuidLow) & 0xFFFFFFFFFFFFFFFF
    finally:
        ComPtr(dxdev.value).release()


class DdaBridge:
    """DDA 环纹理的 D3D12 消费桥。

    初始化（d3d 管线启动时；zcc = DxgiGpuColor，先 build_inline）：
        # 【必须】把采集设备和它的 immediate context 传进来：
        # producer_handles() 交出去的 ctx4 要发"拷贝完成"的 Signal、收"写槽前
        # 等读完"的 Wait，而环纹理的拷贝在 zcc._ctx 上执行。让桥自建设备时
        # 两者不在同一设备的队列里、没有先后保证 → 渲染队列会读到拷贝尚未落地
        # 的旧内容（帧号在涨、画面往回跳，2026-11-28）。
        bridge = DdaBridge(ren, dev11=zcc._dev, ctx11=zcc._ctx)
        f11, ctx4 = bridge.producer_handles()  # 交 DxgiGpuColor 每帧 signal
        zcc.attach_bridge_fence(f11, ctx4, fcons11)   # 会核对 ctx4 是否同源
        ring_texes = bridge.open_ring(zcc._ring, w, h)   # 一次性 open
    每帧消费：
        bridge.queue_wait(v)                   # 录命令前（GPU 端门控）
        ring_texes[idx] 采样（no_barrier Tex）
    生产端每帧拷贝协议（dxgi_gpu_grab d3d 模式）：
        CopyResource → ctx4.Signal(f11, v)
        （v 与 bridge._seq 对账由 glue 层持有；环槽的带锁互斥体在建环时
        各 AcquireSync(0) 一次并一直持有——不取锁的那次拷贝会被丢）
    """

    def __init__(self, ren, dev11=None, ctx11=None, adv=None):
        self.ren = ren
        # adv 必须与渲染器同卡（主循环传 window.adapter）；None 才自选
        adv = adv if adv is not None else placement.pick_adapter()
        self.expect_luid = int(adv.luid) if adv is not None else 0
        if dev11 is None:
            self.dev11, self.ctx11, _fl = create_device(
                adv.ptr if adv is not None else None)
            built_here = True
        else:
            self.dev11, self.ctx11 = dev11, ctx11
            built_here = False
        if self.expect_luid:
            got = dev11_luid(self.dev11)
            if got != (self.expect_luid & 0xFFFFFFFFFFFFFFFF):
                raise RuntimeError(
                    f"[bridge] LUID 不一致: D3D11={got:#x} "
                    f"期望={self.expect_luid:#x}（跨卡共享句柄无效）")
        # 渲染器卡也必须一致（换屏/双卡时最易踩：环纹理 open 进另一块卡）
        _rl = int(getattr(ren, "luid", 0) or 0)
        if _rl and self.expect_luid and _rl != (self.expect_luid & 0xFFFFFFFFFFFFFFFF):
            raise RuntimeError(
                f"[bridge] 渲染器 LUID={_rl:#x} 与采集卡 "
                f"{self.expect_luid:#x} 不同（需重启换卡）")
        self.built_device = built_here

        # producer fence（D3D11 SHARED）→ 共享句柄 → D3D12 open（懒）
        dev5 = ctypes.c_void_p(self.dev11.qi(IID_D3D11Device5))
        d5 = ComPtr(dev5.value)
        f11 = ctypes.c_void_p()
        hr = d5.call(S_DEV5_CREATE_FENCE, ctypes.c_long,
                     (ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                      ctypes.POINTER(ctypes.c_void_p)),
                     0, D3D11_FENCE_FLAG_SHARED,
                     guid_ptr(IID_D3D11Fence), ctypes.byref(f11))
        if hr:
            raise OSError(hr, "[bridge] D3D11 fence 创建失败")
        self.fence11 = ComPtr(f11.value)
        hnd = ctypes.c_void_p()
        hr = self.fence11.call(S_FENCE_CREATE_SHARED, ctypes.c_long,
                               (ctypes.c_void_p, ctypes.c_uint,
                                ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p)),
                               None, GENERIC_ALL, None, ctypes.byref(hnd))
        if hr:
            raise OSError(hr, "[bridge] fence CreateSharedHandle 失败")
        self._fence_hnd = hnd.value
        self.ctx4 = ComPtr(self.ctx11.qi(CTX4_IID))
        self._fence12 = None          # 懒 open
        # 反向 fence（消费者→生产者，槽回收）：同样 D3D11 SHARED → 共享句柄
        # → D3D12 懒 open。消费者每帧 signal（按槽记录值），采集写该槽前
        # Wait 到该值，确保 GPU 已不再采样它（否则一帧闪/撕裂）。
        f11c = ctypes.c_void_p()
        hr = d5.call(S_DEV5_CREATE_FENCE, ctypes.c_long,
                     (ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                      ctypes.POINTER(ctypes.c_void_p)),
                     0, D3D11_FENCE_FLAG_SHARED,
                     guid_ptr(IID_D3D11Fence), ctypes.byref(f11c))
        if hr:
            raise OSError(hr, "[bridge] consumer fence 创建失败")
        self.fence_cons11 = ComPtr(f11c.value)
        hnd2 = ctypes.c_void_p()
        hr = self.fence_cons11.call(S_FENCE_CREATE_SHARED, ctypes.c_long,
                                    (ctypes.c_void_p, ctypes.c_uint,
                                     ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p)),
                                    None, GENERIC_ALL, None,
                                    ctypes.byref(hnd2))
        if hr:
            raise OSError(hr, "[bridge] consumer fence CreateSharedHandle 失败")
        self._fence_cons_hnd = hnd2.value
        self._fence_cons12 = None     # 懒 open
        self._seq = 0                 # producer 信号值（glue 层对账）

    def producer_handles(self):
        """(fence11 ComPtr, ctx4 ComPtr)——采集侧 signal 用。"""
        return self.fence11, self.ctx4

    def consumer_handles(self):
        """(fence_cons11 ComPtr, ctx4 ComPtr)——采集侧写槽前 Wait 用。"""
        return self.fence_cons11, self.ctx4

    def consumer_fence12(self):
        """D3D12 侧打开的 consumer fence 裸指针（消费者 signal 用）。"""
        if self._fence_cons12 is None:
            from .d3d12 import S_DEVICE
            dev = ComPtr(self.ren.get_dev_ptr().value)
            f12 = ctypes.c_void_p()
            hr = dev.call(S_DEV12_OPEN_SHARED, ctypes.c_long,
                          (ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p)),
                          self._fence_cons_hnd, guid_ptr(IID_D3D12Fence),
                          ctypes.byref(f12))
            if hr:
                raise OSError(hr,
                              "[bridge] consumer fence OpenSharedHandle 失败")
            self._fence_cons12 = f12.value
        return self._fence_cons12

    def next_seq(self):
        """递增并返回帧序号（与 signal 值/queue_wait 值一一对应）。"""
        self._seq += 1
        return self._seq

    def fence12(self):
        """D3D12 侧打开的 producer fence（裸指针，缓存）。"""
        if self._fence12 is None:
            from .d3d12 import S_DEVICE
            dev = ComPtr(self.ren.get_dev_ptr().value)
            f12 = ctypes.c_void_p()
            hr = dev.call(S_DEV12_OPEN_SHARED, ctypes.c_long,
                          (ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p)),
                          self._fence_hnd, guid_ptr(IID_D3D12Fence),
                          ctypes.byref(f12))
            if hr:
                raise OSError(hr, "[bridge] fence OpenSharedHandle 失败")
            self._fence12 = f12.value
        return self._fence12

    def open_ring(self, ring_comptrs, w, h, fmt=87):
        """D3D12 侧一次性 open 环纹理并包装成 no_barrier Tex。

        流程：每纹理 IDXGIResource1::CreateSharedHandle（槽 13，NT 句柄）
        → ID3D12Device::OpenSharedHandle（槽 32）。句柄 CloseHandle 由
        OS 收尾（进程内 NT 句柄复制语义已由 open 承接）。
        """
        from .d3d12 import S_DEVICE
        dev = ComPtr(self.ren.get_dev_ptr().value)
        out = []
        for t in ring_comptrs:
            r1 = ctypes.c_void_p(t.qi(IID_IDXGIResource1))
            r1c = ComPtr(r1.value)
            hnd = ctypes.c_void_p()
            hr = r1c.call(S_RES_CREATE_SHARED, ctypes.c_long,
                          (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p)),
                          None, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                          None, ctypes.byref(hnd))
            r1c.release()
            if hr:
                raise OSError(hr, "[bridge] 环纹理 CreateSharedHandle 失败")
            res12 = ctypes.c_void_p()
            hr = dev.call(S_DEV12_OPEN_SHARED, ctypes.c_long,
                          (ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p)),
                          hnd.value, guid_ptr(IID_D3D12Resource),
                          ctypes.byref(res12))
            ctypes.windll.kernel32.CloseHandle(hnd.value)
            if hr:
                raise OSError(hr, f"[bridge] 环纹理 OpenSharedHandle 失败 "
                              f"hr={hr & 0xFFFFFFFF:#x}")
            out.append(self.ren.wrap_external(res12.value, w, h, fmt))
        return out

    def queue_wait(self, v):
        """GPU 端等 producer signal（录进当前命令列表前调用；CPU 不阻塞）。"""
        from .d3d12 import S_QUEUE
        q = ComPtr(self.ren.get_queue_ptr().value)
        q.call(15, ctypes.c_long, (ctypes.c_void_p, ctypes.c_uint64),
               self.fence12(), v)   # S_QUEUE["Wait"] = 15

    def close(self):
        """释放自建设备（外部传入的不归本对象管）。"""
        try:
            self.ctx4.release()
        except Exception:
            pass
        try:
            self.fence11.release()
        except Exception:
            pass
        if self.built_device:
            try:
                self.ctx11.release()
                self.dev11.release()
            except Exception:
                pass
