# -*- coding: utf-8 -*-
"""D3D12 最小渲染运行时（GL→D3D 迁移 P0 基建，2026-11-21）。

设计要点（迁移计划定案）：
- 管线主体单一 D3D12 设备；全屏 pass 用 graphics PSO（像素着色器），
  与 GL 语义 1:1（RTV↔FBO、sampler↔sampler）。
- 屏障策略：可渲染纹理长期驻留组合态
  RENDER_TARGET|PIXEL_SHADER_RESOURCE|COPY_SRC|COPY_DST（0x684）——
  深度链内零屏障；swapchain backbuffer 在 Present 后隐式回 COMMON，
  画前 COMMON→RENDER_TARGET（Swapchain 负责）。
- 描述符：SRV 主堆（shader-visible）单纹理直用；多输入 pass 的连续区间
  用每帧 arena（CopyDescriptors 拷贝，fence 同步后复位）。cbuffer 走
  root CBV（upload 堆 GPUVA+offset，256B 对齐槽）。
- 槽位/IID 全部由 tools/d3d_slotgen.py 与 SDK 头文件 MIDL_INTERFACE 机器
  提取，并与 rtcom 生产值交叉验证（CreateTexture2D=5/Map=14/Unmap=15/
  GetDeviceRemovedReason=39/GetImmediateContext=40）——严禁凭记忆手改。
- 生命周期：_track/_untrack 统计（显存泄漏冒烟），ComPtr 复用 rtcom。
"""
import ctypes
import struct
import os

import numpy as np

from .rtcom import ComPtr, GUID, guid_ptr

# ---------------------------------------------------------------------------
# DLL
# ---------------------------------------------------------------------------
d3d12 = ctypes.WinDLL("d3d12")
dxgi = ctypes.WinDLL("dxgi")
k32 = ctypes.WinDLL("kernel32")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DXC_DIRS = (
    os.path.join(_ROOT, "_toolchain", "dl", "sdk", "cpp", "c", "bin",
                 "10.0.26100.0", "x64"),
    os.path.join(_ROOT, "runtime"),
)

# ---------------------------------------------------------------------------
# IIDs / CLSIDs —— 机器提取自 _toolchain SDK 头文件（MIDL_INTERFACE /
# CROSS_PLATFORM_UUIDOF），勿手改
# ---------------------------------------------------------------------------
IID_D3D12Device          = "189819f1-1db6-4b57-be54-1821339b85f7"
IID_D3D12CommandQueue    = "0ec870a6-5d7e-4c22-8cfc-5baae07616ed"
IID_D3D12CommandAllocator = "6102dee4-af59-4b09-b999-b44d73f09b24"
IID_D3D12GraphicsCommandList = "5b160d0f-ac1b-4185-8ba8-b3ae42a5a455"
IID_D3D12Fence           = "0a753dcf-c4d8-4b91-adf6-be5a60d95a76"
IID_D3D12Resource        = "696442be-a72e-4059-bc79-5b5c98040fad"
IID_D3D12DescriptorHeap  = "8efb471d-616c-4f49-90f7-127bb763fa51"
IID_D3D12RootSignature   = "c54a6b66-72df-4ee8-8be5-a946a1429214"
IID_ID3D12PipelineState  = "bf2872c0-93cf-4eaa-af23-8bcb9aec4bbb"
IID_IDXGIFactory2        = "50c83a1c-e072-4c48-87b0-3630fa36a6d0"
IID_IDXGISwapChain1      = "790a45f7-0d42-4876-983a-0a55cfe6f4aa"
IID_IDXGIKeyedMutex      = "9d8e1289-d7b3-465f-8126-250e349af85d"
IID_ID3D11Device5        = "8ffde202-a0e7-45df-9e01-e837801b5ea0"
IID_ID3D11Fence          = "affde9d1-1df7-4bb7-8a34-0f46251dab80"
IID_ID3D11DeviceContext4 = "917600da-f58c-4c33-98d8-3e15b390fa24"  # d3d11_3.h
CLSID_DxcCompiler        = "73e22d93-e6ce-47f3-b5bf-f0664f39c1b0"   # dxcapi.h
IID_IDxcCompiler3        = "228b4687-5a6a-4730-900c-9702b2203f54"   # dxcapi.h:829
IID_IDxcUtils            = "4605c4cb-2019-492a-ada4-65f20bb7d67f"   # dxcapi.h:510
IID_IDxcBlob             = "8ba5fb08-5195-40e2-ac58-0d989c3a0102"   # dxcapi.h:147
IID_IDxcResult           = "58346cda-dde7-4497-9461-6f87af5e0659"   # dxcapi.h:755

# ---------------------------------------------------------------------------
# 常量（d3d12.h/dxgi.h 枚举）
# ---------------------------------------------------------------------------
FMT_R32F      = 41
FMT_RGBA32F   = 2
FMT_R8_UNORM  = 61
FMT_R8G8B8A8  = 28
FMT_B8G8R8A8  = 87
FMT_UNKNOWN   = 0

HEAP_DEFAULT  = 1
HEAP_UPLOAD   = 2
HEAP_READBACK = 3

RES_FLAG_ALLOW_RENDER_TARGET = 0x1
RES_MISC_SHARED_NTHANDLE     = 0x800
RES_MISC_SHARED_KEYED_MUTEX  = 0x1000

STATE_COMMON        = 0
STATE_GENERIC_READ  = 0xAC3   # VCB|IB|INDIRECT_ARG|COPY_SRC|SRV×2（SDK 头文件原式，含 0x200）
STATE_RENDER_TARGET = 0x4
STATE_PIXEL_SRV     = 0x80
STATE_COPY_SRC      = 0x800   # COPY_SOURCE=0x800（SDK 头文件核实；0x200=INDIRECT_ARGUMENT！）
STATE_COPY_DST      = 0x400
STATE_PRESENT       = 0x800   # 与 COPY_SOURCE 同值（合法）
STATE_REST          = STATE_PIXEL_SRV | STATE_COPY_SRC | STATE_COPY_DST
STATE_REST_RT       = STATE_REST | STATE_RENDER_TARGET

HEAP_TYPE_CBV_SRV_UAV = 0
HEAP_TYPE_RTV         = 2
HEAP_FLAG_SHADER_VISIBLE = 1

RTV_DIM_TEXTURE2D = 4
SRV_DIM_TEXTURE2D = 4

FILTER_LINEAR = 0x50   # MIN_MAG_MIP_LINEAR
FILTER_POINT  = 0x00   # MIN_MAG_MIP_POINT
ADDR_CLAMP    = 3
CMP_NEVER     = 1
COMP_MAP_DEFAULT = 0x1688   # D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING

ROOT_PARAM_TABLE = 0
ROOT_PARAM_CBV   = 2
SHADER_VIS_ALL   = 0
RANGE_TYPE_SRV   = 0
TOPOLOGY_TRIANGLESTRIP = 5
QUEUE_DIRECT  = 0

DXGI_SWAP_EFFECT_DISCARD      = 0
DXGI_SWAP_EFFECT_FLIP_DISCARD = 4
DXGI_ALPHA_MODE_IGNORE        = 3
DXGI_USAGE_RENDER_TARGET      = 0x2

FEATURE_LEVEL_11_0 = 0xB000

# ---------------------------------------------------------------------------
# vtable 槽位（0-based；tools/d3d_slotgen.py 机器提取 + rtcom 生产值交叉验证）
# ---------------------------------------------------------------------------
S_DEVICE = {"CreateCommandQueue": 8, "CreateCommandAllocator": 9,
            "CreateGraphicsPipelineState": 10, "CreateCommandList": 12,
            "CheckFeatureSupport": 13, "CreateDescriptorHeap": 14,
            "GetDescriptorHandleIncrementSize": 15, "CreateRootSignature": 16,
            "CreateShaderResourceView": 18, "CreateUnorderedAccessView": 19,
            "CreateRenderTargetView": 20, "CreateSampler": 22,
            "CopyDescriptors": 23, "CreateCommittedResource": 27,
            "OpenSharedHandle": 32, "CreateFence": 36,
            "GetDeviceRemovedReason": 37, "GetAdapterLuid": 43}
# 注意：CreateCommittedResource=27 / CreateFence=36 / OpenSharedHandle=32 /
# GetAdapterLuid=43——d3d12.h 的 `#if !defined(_WIN32)/#else` 双分支
# （GetResourceAllocationInfo/GetCustomHeapProperties/GetAdapterLuid 三处）
# 曾让朴素计数膨胀 +2，CreateFence 38=GetCopyableFootprints 调用即 segfault
# （tools/d3d_slotgen.py 已带预处理器感知，见 2026-11-21 P0）。
S_CL = {"Close": 9, "Reset": 10, "DrawInstanced": 12, "Dispatch": 14,
        "CopyTextureRegion": 16, "CopyResource": 17,
        "IASetPrimitiveTopology": 20, "RSSetViewports": 21,
        "RSSetScissorRects": 22, "OMSetBlendFactor": 23, "SetPipelineState": 25,
        "ResourceBarrier": 26, "SetDescriptorHeaps": 28,
        "SetGraphicsRootSignature": 30, "SetGraphicsRootDescriptorTable": 32,
        "SetGraphicsRoot32BitConstants": 36,
        "SetGraphicsRootConstantBufferView": 38, "OMSetRenderTargets": 46,
        "ClearRenderTargetView": 48}
S_QUEUE = {"ExecuteCommandLists": 10, "Signal": 14, "Wait": 15}
S_FENCE = {"GetCompletedValue": 8, "SetEventOnCompletion": 9, "Signal": 10}
S_RES = {"Map": 8, "Unmap": 9, "GetDesc": 10, "GetGPUVirtualAddress": 11}
S_HEAP = {"GetCPUDescriptorHandleForHeapStart": 9,
          "GetGPUDescriptorHandleForHeapStart": 10}
S_SC = {"Present": 8, "GetBuffer": 9, "ResizeBuffers": 13}
S_FACTORY2 = {"CreateSwapChainForHwnd": 15, "MakeWindowAssociation": 8}
S_KEYED = {"AcquireSync": 7, "ReleaseSync": 8}
# D3D11 fence 相关（rtcom 已有 5/14/15/39/40/46/47/48/111；补齐）：
# ID3D11Fence 无 CPU Signal（ABI 事实，d3d11_3.h Vtbl 核实）——D3D11 侧
# fence 只能经 ID3D11DeviceContext4::Signal(147)/Wait(148) GPU 令。
S_D3D11_DEV5 = {"CreateFence": 68, "OpenSharedFence": 67}
S_D3D11_FENCE = {"CreateSharedHandle": 7, "GetCompletedValue": 8,
                 "SetEventOnCompletion": 9}
S_D3D11_CTX4 = {"Signal": 147, "Wait": 148}
IID_ID3D11DeviceContext4 = "917600da-f58c-4c33-98d8-3e15b390fa24"
S_DXC3 = {"Compile": 3}
S_DXCUTILS = {"CreateBlobFromPinned": 4}
# IDxcOperationResult 继承 IUnknown：GetStatus/GetResult/GetErrorBuffer=3/4/5
# （旧值 6/7/8 是 IDxcResult 追加段，手写错值——dxcapi.h L373-392 权威）
S_DXCRESULT = {"GetStatus": 3, "GetResult": 4, "GetErrorBuffer": 5}
IID_IDxcOperationResult = "cedb484a-d4e9-445a-b991-ca21ca157dc2"
S_BLOB = {"GetBufferPointer": 3, "GetBufferSize": 4}

# ---------------------------------------------------------------------------
# 结构体
# ---------------------------------------------------------------------------
class DXGI_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", ctypes.c_uint), ("Quality", ctypes.c_uint)]


class DXGI_RATIONAL(ctypes.Structure):
    _fields_ = [("Numerator", ctypes.c_uint), ("Denominator", ctypes.c_uint)]


class D3D12_COMMAND_QUEUE_DESC(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("Priority", ctypes.c_int),
                ("Flags", ctypes.c_uint), ("NodeMask", ctypes.c_uint)]


class D3D12_HEAP_PROPERTIES(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("CPUPageProperty", ctypes.c_int),
                ("MemoryPoolPreference", ctypes.c_int),
                ("CreationNodeMask", ctypes.c_uint), ("VisibleNodeMask", ctypes.c_uint)]


class D3D12_RESOURCE_DESC(ctypes.Structure):
    _fields_ = [("Dimension", ctypes.c_uint), ("Alignment", ctypes.c_uint64),
                ("Width", ctypes.c_uint64), ("Height", ctypes.c_uint),
                ("DepthOrArraySize", ctypes.c_uint16), ("MipLevels", ctypes.c_uint16),
                ("Format", ctypes.c_uint), ("SampleDesc", DXGI_SAMPLE_DESC),
                ("Layout", ctypes.c_int), ("Flags", ctypes.c_uint)]


class D3D12_DESCRIPTOR_HEAP_DESC(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int), ("NumDescriptors", ctypes.c_uint),
                ("Flags", ctypes.c_uint), ("NodeMask", ctypes.c_uint)]


class D3D12_CPU_HANDLE(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_size_t)]


class D3D12_GPU_HANDLE(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64)]


class D3D12_DESCRIPTOR_RANGE(ctypes.Structure):
    _fields_ = [("RangeType", ctypes.c_uint), ("NumDescriptors", ctypes.c_uint),
                ("BaseShaderRegister", ctypes.c_uint), ("RegisterSpace", ctypes.c_uint),
                ("OffsetInDescriptorsFromTableStart", ctypes.c_uint)]


class D3D12_ROOT_DESCRIPTOR_TABLE(ctypes.Structure):
    _fields_ = [("NumDescriptorRanges", ctypes.c_uint),
                ("pDescriptorRanges", ctypes.c_void_p)]


class D3D12_ROOT_DESCRIPTOR(ctypes.Structure):
    _fields_ = [("ShaderRegister", ctypes.c_uint), ("RegisterSpace", ctypes.c_uint)]


class D3D12_ROOT_PARAM_U(ctypes.Union):
    _fields_ = [("Table", D3D12_ROOT_DESCRIPTOR_TABLE),
                ("Descriptor", D3D12_ROOT_DESCRIPTOR)]


class D3D12_ROOT_PARAMETER(ctypes.Structure):
    _fields_ = [("ParameterType", ctypes.c_uint), ("Visibility", ctypes.c_uint),
                ("u", D3D12_ROOT_PARAM_U)]


class D3D12_STATIC_SAMPLER_DESC(ctypes.Structure):
    _fields_ = [("Filter", ctypes.c_uint), ("AddressU", ctypes.c_int),
                ("AddressV", ctypes.c_int), ("AddressW", ctypes.c_int),
                ("MipLODBias", ctypes.c_float), ("MaxAnisotropy", ctypes.c_uint),
                ("ComparisonFunc", ctypes.c_uint), ("BorderColor", ctypes.c_uint),
                ("MinLOD", ctypes.c_float), ("MaxLOD", ctypes.c_float),
                ("ShaderRegister", ctypes.c_uint), ("RegisterSpace", ctypes.c_uint),
                ("ShaderVisibility", ctypes.c_uint)]


class D3D12_ROOT_SIGNATURE_DESC(ctypes.Structure):
    _fields_ = [("NumParameters", ctypes.c_uint), ("pParameters", ctypes.c_void_p),
                ("NumStaticSamplers", ctypes.c_uint),
                ("pStaticSamplers", ctypes.c_void_p), ("Flags", ctypes.c_uint)]


class D3D12_SHADER_BYTECODE(ctypes.Structure):
    _fields_ = [("pShaderBytecode", ctypes.c_void_p),
                ("BytecodeLength", ctypes.c_size_t)]


class D3D12_SO_DECLARATION_ENTRY(ctypes.Structure):
    _fields_ = [("Stream", ctypes.c_uint), ("SemanticName", ctypes.c_void_p),
                ("SemanticIndex", ctypes.c_uint), ("StartComponent", ctypes.c_ubyte),
                ("ComponentCount", ctypes.c_ubyte), ("OutputSlot", ctypes.c_ubyte)]


class D3D12_STREAM_OUTPUT_DESC(ctypes.Structure):
    _fields_ = [("pSODeclaration", ctypes.c_void_p), ("NumEntries", ctypes.c_uint),
                ("pRasterizedStream", ctypes.c_void_p),
                ("NumRasterizedStreams", ctypes.c_uint), ("BufferFlags", ctypes.c_uint)]


class D3D12_RENDER_TARGET_BLEND_DESC(ctypes.Structure):
    _fields_ = [("BlendEnable", ctypes.c_int), ("LogicOpEnable", ctypes.c_int),
                ("SrcBlend", ctypes.c_int), ("DestBlend", ctypes.c_int),
                ("BlendOp", ctypes.c_int), ("SrcBlendAlpha", ctypes.c_int),
                ("DestBlendAlpha", ctypes.c_int), ("BlendOpAlpha", ctypes.c_int),
                ("LogicOp", ctypes.c_int), ("RenderTargetWriteMask", ctypes.c_ubyte)]


class D3D12_BLEND_DESC(ctypes.Structure):
    _fields_ = [("AlphaToCoverageEnable", ctypes.c_int),
                ("IndependentBlendEnable", ctypes.c_int),
                ("RenderTarget", D3D12_RENDER_TARGET_BLEND_DESC * 8)]


class D3D12_RASTERIZER_DESC(ctypes.Structure):
    _fields_ = [("FillMode", ctypes.c_int), ("CullMode", ctypes.c_int),
                ("FrontCounterClockwise", ctypes.c_int), ("DepthBias", ctypes.c_int),
                ("DepthBiasClamp", ctypes.c_float),
                ("SlopeScaledDepthBias", ctypes.c_float),
                ("DepthClipEnable", ctypes.c_int), ("MultisampleEnable", ctypes.c_int),
                ("AntialiasedLineEnable", ctypes.c_int),
                ("ForcedSampleCount", ctypes.c_uint), ("ConservativeRaster", ctypes.c_int)]


class D3D12_DEPTH_STENCIL_DESC(ctypes.Structure):
    _fields_ = [("DepthEnable", ctypes.c_int), ("DepthWriteMask", ctypes.c_uint),
                ("DepthFunc", ctypes.c_uint), ("StencilEnable", ctypes.c_int),
                ("StencilReadMask", ctypes.c_ubyte), ("StencilWriteMask", ctypes.c_ubyte)]


class D3D12_INPUT_ELEMENT_DESC(ctypes.Structure):
    _fields_ = [("SemanticName", ctypes.c_char_p), ("SemanticIndex", ctypes.c_uint),
                ("Format", ctypes.c_uint), ("InputSlot", ctypes.c_uint),
                ("AlignedByteOffset", ctypes.c_uint), ("InputSlotClass", ctypes.c_int),
                ("InstanceDataStepRate", ctypes.c_uint)]


class D3D12_INPUT_LAYOUT_DESC(ctypes.Structure):
    _fields_ = [("pInputElementDescs", ctypes.c_void_p), ("NumElements", ctypes.c_uint)]


class D3D12_CACHED_PSO(ctypes.Structure):
    _fields_ = [("pCachedBlob", ctypes.c_void_p),
                ("CachedBlobSizeInBytes", ctypes.c_size_t),
                ("pCachedPSOPipelineLibrary", ctypes.c_void_p)]


class D3D12_GRAPHICS_PIPELINE_STATE_DESC(ctypes.Structure):
    _fields_ = [("pRootSignature", ctypes.c_void_p), ("VS", D3D12_SHADER_BYTECODE),
                ("PS", D3D12_SHADER_BYTECODE), ("GS", D3D12_SHADER_BYTECODE),
                ("DS", D3D12_SHADER_BYTECODE), ("HS", D3D12_SHADER_BYTECODE),
                ("StreamOutput", D3D12_STREAM_OUTPUT_DESC),
                ("BlendState", D3D12_BLEND_DESC), ("SampleMask", ctypes.c_uint),
                ("RasterizerState", D3D12_RASTERIZER_DESC),
                ("DepthStencilState", D3D12_DEPTH_STENCIL_DESC),
                ("InputLayout", D3D12_INPUT_LAYOUT_DESC),
                ("IBStripCutValue", ctypes.c_int),
                ("PrimitiveTopologyType", ctypes.c_int),
                ("NumRenderTargets", ctypes.c_uint),
                ("RTVFormats", ctypes.c_uint * 8), ("DSVFormat", ctypes.c_uint),
                ("SampleDesc", DXGI_SAMPLE_DESC), ("NodeMask", ctypes.c_uint),
                ("CachedPSO", D3D12_CACHED_PSO), ("Flags", ctypes.c_uint)]


class D3D12_RES_TRANSITION(ctypes.Structure):
    _fields_ = [("pResource", ctypes.c_void_p), ("StateBefore", ctypes.c_uint),
                ("StateAfter", ctypes.c_uint), ("Subresource", ctypes.c_uint)]


class D3D12_RESOURCE_BARRIER(ctypes.Structure):
    _anonymous_ = ("t",)
    _fields_ = [("Type", ctypes.c_int), ("Flags", ctypes.c_uint),
                ("t", D3D12_RES_TRANSITION)]


class D3D12_VIEWPORT(ctypes.Structure):
    _fields_ = [("TopLeftX", ctypes.c_float), ("TopLeftY", ctypes.c_float),
                ("Width", ctypes.c_float), ("Height", ctypes.c_float),
                ("MinDepth", ctypes.c_float), ("MaxDepth", ctypes.c_float)]


class D3D12_RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _PLACED_FOOTPRINT(ctypes.Structure):
    _fields_ = [("Offset", ctypes.c_uint64), ("Format", ctypes.c_uint),
                ("Width", ctypes.c_uint), ("Height", ctypes.c_uint),
                ("Depth", ctypes.c_uint), ("RowPitch", ctypes.c_uint)]


class D3D12_TEXTURE_COPY_LOCATION(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("pResource", ctypes.c_void_p), ("Type", ctypes.c_int),
                ("u", _PLACED_FOOTPRINT)]
    # sizeof=48（C++ 同）；SubresourceIndex(Type=0) 经 _pad 覆盖（union@12）


class DXGI_SWAP_CHAIN_DESC1(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_uint), ("Height", ctypes.c_uint),
                ("Format", ctypes.c_uint), ("Stereo", ctypes.c_int),
                ("SampleDesc", DXGI_SAMPLE_DESC), ("BufferUsage", ctypes.c_uint),
                ("BufferCount", ctypes.c_uint), ("Scaling", ctypes.c_int),
                ("SwapEffect", ctypes.c_int), ("AlphaMode", ctypes.c_int),
                ("Flags", ctypes.c_uint)]


class D3D12_RANGE(ctypes.Structure):
    _fields_ = [("Begin", ctypes.c_size_t), ("End", ctypes.c_size_t)]


# ---------------------------------------------------------------------------
# 生命周期统计
# ---------------------------------------------------------------------------
_LIVE = {}
_ALLOC_N = [0]
_FREE_N = [0]


def _track(p, tag):
    _ALLOC_N[0] += 1
    _LIVE[tag] = _LIVE.get(tag, 0) + 1


def _release(com, tag):
    if com and com.p:
        com.release()
        _FREE_N[0] += 1
        _LIVE[tag] = max(0, _LIVE.get(tag, 1) - 1)


def live_stats():
    return dict(_LIVE), _ALLOC_N[0], _FREE_N[0]


_FMT_NP = {FMT_R32F: np.float32, FMT_RGBA32F: np.float32,
           FMT_R8_UNORM: np.uint8, FMT_R8G8B8A8: np.uint8,
           FMT_B8G8R8A8: np.uint8}


class Tex:
    """GPU 纹理：驻留态 REST（采样/拷贝源/目标），可渲染版含 RENDER_TARGET。"""

    def __init__(self, ctx, res, w, h, fmt, renderable, is_upload=False,
                 is_readback=False, map_ptr=None):
        self.ctx = ctx
        self.res = res
        self.w, self.h, self.fmt = w, h, fmt
        self.state = STATE_COMMON
        self.rest = (STATE_REST_RT if renderable else STATE_REST)
        self.renderable = renderable
        self.is_upload = is_upload
        self.is_readback = is_readback
        self.map_ptr = map_ptr              # upload/readback 常驻映射
        self.rtv_cpu = None
        self.srv_cpu = None                 # shader-visible 主堆槽
        self._init_done = False             # COMMON→REST 首次推进标记

    # ---- 屏障 -------------------------------------------------------------
    def transition(self, cl, to):
        if self.state == to:
            return
        b = D3D12_RESOURCE_BARRIER(Type=0, Flags=0)
        b.t.pResource = self.res.p
        b.t.StateBefore = self.state
        b.t.StateAfter = to
        b.t.Subresource = 0xFFFFFFFF
        cl.call(S_CL["ResourceBarrier"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_RESOURCE_BARRIER)),
                1, ctypes.byref(b))
        self.state = to

    def _init_rest_if_needed(self, cl):
        if not self._init_done:
            self.transition(cl, self.rest)
            self._init_done = True

    def to_read(self, cl):
        self._init_rest_if_needed(cl)
        self.transition(cl, STATE_PIXEL_SRV)

    def to_rest(self, cl):
        self.transition(cl, self.rest)

    # ---- 数据 -------------------------------------------------------------
    def upload_np(self, arr):
        """UPLOAD 纹理写内容（CPU memcpy；arr 形状 (h,w) dtype 匹配格式）。"""
        if not self.is_upload:
            raise RuntimeError("upload_np 仅 UPLOAD 纹理")
        src = np.ascontiguousarray(arr, dtype=_FMT_NP[self.fmt])
        bytes_w = self.w * np.dtype(_FMT_NP[self.fmt]).itemsize
        if src.nbytes < bytes_w * self.h:
            raise ValueError(f"上传数据不足: {src.nbytes} < {bytes_w * self.h}")
        ctypes.memmove(self.map_ptr, src.ctypes.data, bytes_w * self.h)

    def readback_np(self, dtype=None, shape=None):
        """READBACK 资源读内容（Map 常驻）。shape=None = 1D 字节流。"""
        dt = dtype or _FMT_NP[self.fmt]
        n = self.w * np.dtype(dt).itemsize
        buf = (ctypes.c_ubyte * n).from_address(self.map_ptr)
        flat = np.frombuffer(buf, dtype=dt)
        return flat.reshape(shape) if shape else flat

    def gpu_readback(self, h, w, dtype=None):
        """GPU 纹理 → READBACK → numpy（经命令提交+fence 同步）。"""
        dt = dtype or _FMT_NP[self.fmt]
        rb = self.ctx.make_readback(w, h, self.fmt)
        cl = self.ctx.begin_frame()
        self._init_rest_if_needed(cl)
        self.transition(cl, STATE_COPY_SRC)
        rb.transition(cl, STATE_COPY_DST)
        ls = D3D12_TEXTURE_COPY_LOCATION(pResource=self.res.p, Type=0)
        ld = D3D12_TEXTURE_COPY_LOCATION(pResource=rb.res.p, Type=0)
        cl.call(S_CL["CopyResource"], None,
                (ctypes.c_void_p, ctypes.c_void_p), ld.p, ls.p)
        self.to_rest(cl)
        self.ctx.submit()
        self.ctx.wait_gpu()
        arr = rb.readback_np(dtype)
        rb.free()
        return arr


class UploadTex(Tex):
    """UPLOAD 堆资源（buffer 形态）：CPU 写 + GPU CopyTextureRegion 源。"""

    def __init__(self, ctx, res, w, h, fmt, map_ptr):
        super().__init__(ctx, res, w, h, fmt, renderable=False, is_upload=True,
                         map_ptr=map_ptr)
        self.rest = STATE_COMMON   # UPLOAD 恒 COMMON

    def upload_np(self, arr):
        """buffer 写入（1D 字节序，PlacedFootprint 拷贝的源）。"""
        src = np.ascontiguousarray(arr, dtype=_FMT_NP[self.fmt]).view(np.uint8).ravel()
        ctypes.memmove(self.map_ptr, src.ctypes.data, src.nbytes)


class ReadbackTex(Tex):
    """READBACK 堆纹理：GPU 写 + CPU 读。"""

    def __init__(self, ctx, res, w, h, fmt, map_ptr):
        super().__init__(ctx, res, w, h, fmt, renderable=False, is_readback=True,
                         map_ptr=map_ptr)
        self.rest = STATE_COPY_DST

    def free(self):
        _release(self.res, "res")


# ---------------------------------------------------------------------------
# 主上下文
# ---------------------------------------------------------------------------
class D3D12Context:
    """单一 D3D12 设备 + 直连队列 + fence + 描述符堆 + PSO 缓存。"""

    SRV_HEAP_N = 4096
    RTV_HEAP_N = 256
    CB_SLOT_STRIDE = 64        # floats/pass（256B 对齐）
    CB_SLOTS = 512
    ARENA_N = 1024

    def __init__(self, adapter_ptr=None, debug=False):
        if debug:
            dbg = ctypes.c_void_p()
            hr = d3d12.D3D12GetDebugInterface(
                guid_ptr("344488b7-6846-474b-b989-f0274484810f"), ctypes.byref(dbg))
            if hr == 0 and dbg:
                ComPtr(dbg.value).call(3, None, ())   # ID3D12Debug::EnableDebugLayer
        out = ctypes.c_void_p()
        if adapter_ptr is None:
            ap = None
        elif isinstance(adapter_ptr, ctypes.c_void_p):
            ap = adapter_ptr
        else:
            ap = ctypes.c_void_p(int(adapter_ptr))
        hr = d3d12.D3D12CreateDevice(
            ap, FEATURE_LEVEL_11_0, guid_ptr(IID_D3D12Device), ctypes.byref(out))
        if hr:
            raise OSError(hr, f"D3D12CreateDevice(adapter={adapter_ptr:#x})")
        self.dev = ComPtr(out.value)
        _track(out.value, "dev")
        self.queue = self._create_queue()
        self.fence = self._create_fence()
        self._fence_val = 0
        self._event = k32.CreateEventW(None, 0, 0, None)
        self.allocator = self._create_allocator()
        self.cl = self._create_list()
        self.srv_heap = self._create_heap(HEAP_TYPE_CBV_SRV_UAV, self.SRV_HEAP_N, True)
        self.rtv_heap = self._create_heap(HEAP_TYPE_RTV, self.RTV_HEAP_N, False)
        self.srv_inc = self.dev.call(S_DEVICE["GetDescriptorHandleIncrementSize"],
                                     ctypes.c_uint, (ctypes.c_int,),
                                     HEAP_TYPE_CBV_SRV_UAV)
        self.rtv_inc = self.dev.call(S_DEVICE["GetDescriptorHandleIncrementSize"],
                                     ctypes.c_uint, (ctypes.c_int,), HEAP_TYPE_RTV)
        self._srv_next = 0
        self._rtv_next = 0
        self._arena_heap = self._create_heap(HEAP_TYPE_CBV_SRV_UAV, self.ARENA_N, True)
        # 结构体返回值在 C Vtbl 是 RetVal 指针模式（同 GetAdapterLuid）：
        # CPU 句柄槽 9 / GPU 句柄槽 10，实参=out 指针，返回=同指针
        self._arena_base = self._heap_cpu_start(self._arena_heap)
        self._arena_gpu = self._heap_gpu_start(self._arena_heap)
        self._arena_ptr = 0
        self.luid = self._adapter_luid()
        self._pso_cache = {}
        self._rs = None
        self.cb_res, self.cb_map = self._make_upload_buffer(
            self.CB_SLOT_STRIDE * 4 * self.CB_SLOTS)
        self._cb_next = 0
        self._tex_n = 0

    # ---- 创建 ------------------------------------------------------------
    def _create_queue(self):
        out = ctypes.c_void_p()
        qd = D3D12_COMMAND_QUEUE_DESC(Type=QUEUE_DIRECT)
        hr = self.dev.call(S_DEVICE["CreateCommandQueue"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_COMMAND_QUEUE_DESC), ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(qd), guid_ptr(IID_D3D12CommandQueue),
                           ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateCommandQueue")
        _track(out.value, "queue")
        return ComPtr(out.value)

    def _create_fence(self):
        out = ctypes.c_void_p()
        hr = self.dev.call(S_DEVICE["CreateFence"], ctypes.c_long,
                           (ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           0, 0, guid_ptr(IID_D3D12Fence), ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateFence")
        _track(out.value, "fence")
        return ComPtr(out.value)

    def _create_allocator(self):
        out = ctypes.c_void_p()
        hr = self.dev.call(S_DEVICE["CreateCommandAllocator"], ctypes.c_long,
                           (ctypes.c_int, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           QUEUE_DIRECT, guid_ptr(IID_D3D12CommandAllocator),
                           ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateCommandAllocator")
        _track(out.value, "alloc")
        return ComPtr(out.value)

    def _create_list(self):
        out = ctypes.c_void_p()
        hr = self.dev.call(S_DEVICE["CreateCommandList"], ctypes.c_long,
                           (ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           0, QUEUE_DIRECT, self.allocator.p, None,
                           guid_ptr(IID_D3D12GraphicsCommandList), ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateCommandList")
        _track(out.value, "cl")
        cl = ComPtr(out.value)
        # 新建命令列表必须先 Close 一次才能 Reset（D3D12 规则）
        cl.call(S_CL["Close"], ctypes.c_long, ())
        return cl

    def _create_heap(self, htype, n, shader_visible):
        out = ctypes.c_void_p()
        d = D3D12_DESCRIPTOR_HEAP_DESC(Type=htype, NumDescriptors=n,
                                       Flags=(HEAP_FLAG_SHADER_VISIBLE
                                              if shader_visible else 0))
        hr = self.dev.call(S_DEVICE["CreateDescriptorHeap"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_DESCRIPTOR_HEAP_DESC), ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(d), guid_ptr(IID_D3D12DescriptorHeap),
                           ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateDescriptorHeap(t={htype},n={n})")
        _track(out.value, f"heap{htype}")
        return ComPtr(out.value)

    def _make_upload_buffer(self, nbytes):
        out = ctypes.c_void_p()
        hp = D3D12_HEAP_PROPERTIES(Type=HEAP_UPLOAD)
        d = D3D12_RESOURCE_DESC(Dimension=1, Width=nbytes, Height=1,
                                DepthOrArraySize=1, MipLevels=1, Format=FMT_UNKNOWN,
                                SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=1)
        hr = self.dev.call(S_DEVICE["CreateCommittedResource"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_HEAP_PROPERTIES), ctypes.c_uint,
                            ctypes.POINTER(D3D12_RESOURCE_DESC), ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(hp), 0, ctypes.byref(d),
                           STATE_GENERIC_READ,
                           None, guid_ptr(IID_D3D12Resource), ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateCommittedResource(upload)")
        _track(out.value, "cbres")
        res = ComPtr(out.value)
        rng = D3D12_RANGE(0, 0)
        ptr = ctypes.c_void_p()
        hr = res.call(S_RES["Map"], ctypes.c_long,
                      (ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                      0, ctypes.byref(rng), ctypes.byref(ptr))
        if hr:
            raise OSError(hr, "Map(upload)")
        return res, ptr.value

    def _heap_cpu_start(self, heap):
        out = D3D12_CPU_HANDLE()
        p = heap.call(S_HEAP["GetCPUDescriptorHandleForHeapStart"], ctypes.c_void_p,
                      (ctypes.POINTER(D3D12_CPU_HANDLE),), ctypes.byref(out))
        return ctypes.cast(p or ctypes.byref(out), ctypes.POINTER(D3D12_CPU_HANDLE)
                           ).contents.ptr

    def _heap_gpu_start(self, heap):
        out = D3D12_GPU_HANDLE()
        p = heap.call(S_HEAP["GetGPUDescriptorHandleForHeapStart"], ctypes.c_void_p,
                      (ctypes.POINTER(D3D12_GPU_HANDLE),), ctypes.byref(out))
        return ctypes.cast(p or ctypes.byref(out), ctypes.POINTER(D3D12_GPU_HANDLE)
                           ).contents.ptr

    def _adapter_luid(self):
        class _LUID(ctypes.Structure):
            _fields_ = [("LowPart", ctypes.c_uint), ("HighPart", ctypes.c_long)]

        lu = _LUID()
        self.dev.call(S_DEVICE["GetAdapterLuid"], None,
                      (ctypes.POINTER(_LUID),), ctypes.byref(lu))
        return (lu.HighPart << 32) | lu.LowPart

    # ---- 描述符 ----------------------------------------------------------
    def _srv_cpu_at(self, slot):
        return D3D12_CPU_HANDLE(self._heap_cpu_start(self.srv_heap)
                                + slot * self.srv_inc)

    def _rtv_cpu_at(self, slot):
        return D3D12_CPU_HANDLE(self._heap_cpu_start(self.rtv_heap)
                                + slot * self.rtv_inc)

    def alloc_srv(self, res, fmt):
        slot = self._srv_next
        self._srv_next += 1
        h = self._srv_cpu_at(slot)
        d = _srv_desc(fmt)
        rp = res.p if hasattr(res, "p") else ctypes.c_void_p(res)
        # DestDescriptor 是【按值】的 8 字节结构（win64 寄存器传参）——
        # 传 byref 指针会把"句柄地址"当"句柄值"，驱动写垃圾描述符槽 →
        # 设备 INVALID_CALL 移除（P0 实锤）
        self.dev.call(S_DEVICE["CreateShaderResourceView"], None,
                      (ctypes.c_void_p, ctypes.c_void_p, D3D12_CPU_HANDLE),
                      rp, ctypes.byref(d), h)
        return h

    def alloc_rtv(self, res, fmt):
        slot = self._rtv_next
        self._rtv_next += 1
        h = self._rtv_cpu_at(slot)
        d = _rtv_desc(fmt)
        rp = res.p if hasattr(res, "p") else ctypes.c_void_p(res)
        self.dev.call(S_DEVICE["CreateRenderTargetView"], None,
                      (ctypes.c_void_p, ctypes.c_void_p, D3D12_CPU_HANDLE),
                      rp, ctypes.byref(d), h)
        return h

    # ---- 资源 ------------------------------------------------------------
    def make_texture(self, w, h, fmt, heap=HEAP_DEFAULT, state=STATE_COMMON,
                     flags=0, renderable=True, cls=Tex):
        out = ctypes.c_void_p()
        hp = D3D12_HEAP_PROPERTIES(Type=heap)
        d = D3D12_RESOURCE_DESC(Dimension=3, Width=w, Height=h, DepthOrArraySize=1,
                                MipLevels=1, Format=fmt,
                                SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=0,
                                Flags=flags | (RES_FLAG_ALLOW_RENDER_TARGET
                                               if renderable else 0))
        hr = self.dev.call(S_DEVICE["CreateCommittedResource"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_HEAP_PROPERTIES), ctypes.c_uint,
                            ctypes.POINTER(D3D12_RESOURCE_DESC), ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(hp), 0, ctypes.byref(d), state, None,
                           guid_ptr(IID_D3D12Resource), ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateCommittedResource({w}x{h} fmt={fmt} heap={heap})")
        _track(out.value, "res")
        if os.environ.get("D2S_D3D_DEBUG"):
            hr2 = self.dev.call(S_DEVICE["GetDeviceRemovedReason"], ctypes.c_long, ())
            print(f"[d3d12] committed 后 removed={hex(hr2 & 0xFFFFFFFF)}", flush=True)
        tex = cls(self, ComPtr(out.value), w, h, fmt, renderable)
        self._tex_n += 1
        if renderable:
            tex.rtv_cpu = self.alloc_rtv(out.value, fmt)
            if os.environ.get("D2S_D3D_DEBUG"):
                hr2 = self.dev.call(S_DEVICE["GetDeviceRemovedReason"], ctypes.c_long, ())
                print(f"[d3d12] RTV 后 removed={hex(hr2 & 0xFFFFFFFF)}", flush=True)
        if heap == HEAP_DEFAULT:
            tex.srv_cpu = self.alloc_srv(out.value, fmt)
            if os.environ.get("D2S_D3D_DEBUG"):
                hr2 = self.dev.call(S_DEVICE["GetDeviceRemovedReason"], ctypes.c_long, ())
                print(f"[d3d12] SRV 后 removed={hex(hr2 & 0xFFFFFFFF)}", flush=True)
        return tex

    def make_upload(self, w, h, fmt):
        # UPLOAD 堆 2D 纹理创建被运行时拒绝（R32F+ROW_MAJOR 组合，本机实测）——
        # 降级为 buffer（Width=字节数，Dimension=1）；P1 上传走 PlacedFootprint
        import numpy as np
        nbytes = w * h * np.dtype(_FMT_NP[fmt]).itemsize
        return self.make_upload_buffer(nbytes, w, h, fmt)


    def make_upload_buffer(self, nbytes, w, h, fmt):
        out = ctypes.c_void_p()
        hp = D3D12_HEAP_PROPERTIES(Type=HEAP_UPLOAD)
        d = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=nbytes, Height=1,
                                DepthOrArraySize=1, MipLevels=1, Format=FMT_UNKNOWN,
                                SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=1)
        hr = self.dev.call(S_DEVICE["CreateCommittedResource"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_HEAP_PROPERTIES), ctypes.c_uint,
                            ctypes.POINTER(D3D12_RESOURCE_DESC), ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(hp), 0, ctypes.byref(d), STATE_GENERIC_READ,
                           None, guid_ptr(IID_D3D12Resource), ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateCommittedResource(upload buffer)")
        _track(out.value, "res")
        rng = D3D12_RANGE(0, 0)
        ptr = ctypes.c_void_p()
        hr = ComPtr(out.value).call(S_RES["Map"], ctypes.c_long,
                                    (ctypes.c_uint, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p)),
                                    0, ctypes.byref(rng), ctypes.byref(ptr))
        if hr:
            raise OSError(hr, "Map(upload tex)")
        return UploadTex(self, ComPtr(out.value), w, h, fmt, ptr.value)

    def make_readback(self, w, h, fmt):
        # READBACK 堆 buffer（Width=字节数）；纹理拷贝走 CopyTextureRegion→buffer
        import numpy as np
        nbytes = w * h * np.dtype(_FMT_NP[fmt]).itemsize
        out = ctypes.c_void_p()
        hp = D3D12_HEAP_PROPERTIES(Type=HEAP_READBACK)
        d = D3D12_RESOURCE_DESC(Dimension=1, Alignment=0, Width=nbytes, Height=1,
                                DepthOrArraySize=1, MipLevels=1, Format=FMT_UNKNOWN,
                                SampleDesc=DXGI_SAMPLE_DESC(1, 0), Layout=1)
        hr = self.dev.call(S_DEVICE["CreateCommittedResource"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_HEAP_PROPERTIES), ctypes.c_uint,
                            ctypes.POINTER(D3D12_RESOURCE_DESC), ctypes.c_uint,
                            ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(hp), 0, ctypes.byref(d), STATE_COPY_DST,
                           None, guid_ptr(IID_D3D12Resource), ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateCommittedResource(readback buffer)")
        _track(out.value, "res")
        rng = D3D12_RANGE(0, 0)
        ptr = ctypes.c_void_p()
        hr = ComPtr(out.value).call(S_RES["Map"], ctypes.c_long,
                                    (ctypes.c_uint, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_void_p)),
                                    0, ctypes.byref(rng), ctypes.byref(ptr))
        if hr:
            raise OSError(hr, "Map(readback tex)")
        return ReadbackTex(self, ComPtr(out.value), w, h, fmt, ptr.value)

    # ---- 根签名 / PSO ----------------------------------------------------
    def root_sig(self):
        if self._rs:
            return self._rs
        # 根签名 blob（C++ D3D12SerializeRootSignature 生成 →
        # _toolchain/src/nthandle_probe 输出固化；ctypes ROOT_PARAM_U union
        # 对齐差异导致直接调用 E_INVALIDARG——绕开用预生成 blob）
        _rs_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "rootsig.blob")
        _RS_BLOB = open(_rs_path, "rb").read()
        blob_buf = (ctypes.c_ubyte * len(_RS_BLOB)).from_buffer_copy(_RS_BLOB)
        out = ctypes.c_void_p()
        _rs_ptr = ctypes.cast(blob_buf, ctypes.c_void_p).value
        _rs_size = len(_RS_BLOB)
        hr = self.dev.call(S_DEVICE["CreateRootSignature"], ctypes.c_long,
                           (ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                           0, _rs_ptr, _rs_size,
                           guid_ptr(IID_D3D12RootSignature), ctypes.byref(out))
        if hr:
            raise OSError(hr, "CreateRootSignature")
        _track(out.value, "rs")
        self._rs = ComPtr(out.value)
        return self._rs

    def pso(self, name, vs_cso, ps_cso, rtv_fmt, blend=False):
        key = (name, rtv_fmt, blend)
        if key in self._pso_cache:
            return self._pso_cache[key]
        d = D3D12_GRAPHICS_PIPELINE_STATE_DESC()
        d.pRootSignature = self.root_sig().p
        d.VS = D3D12_SHADER_BYTECODE(vs_cso.data_ptr, vs_cso.nbytes)
        d.PS = D3D12_SHADER_BYTECODE(ps_cso.data_ptr, ps_cso.nbytes)
        d.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF
        if blend:
            d.BlendState.RenderTarget[0].BlendEnable = 1
            d.BlendState.RenderTarget[0].SrcBlend = 5
            d.BlendState.RenderTarget[0].DestBlend = 6
            d.BlendState.RenderTarget[0].BlendOp = 1
            d.BlendState.RenderTarget[0].SrcBlendAlpha = 5
            d.BlendState.RenderTarget[0].DestBlendAlpha = 6
            d.BlendState.RenderTarget[0].BlendOpAlpha = 1
        rs = d.RasterizerState
        rs.FillMode = 3
        rs.CullMode = 0
        rs.DepthClipEnable = 1
        d.PrimitiveTopologyType = 1
        d.NumRenderTargets = 1
        d.RTVFormats[0] = rtv_fmt
        d.SampleDesc = DXGI_SAMPLE_DESC(1, 0)
        out = ctypes.c_void_p()
        hr = self.dev.call(S_DEVICE["CreateGraphicsPipelineState"], ctypes.c_long,
                           (ctypes.POINTER(D3D12_GRAPHICS_PIPELINE_STATE_DESC),
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                           ctypes.byref(d), guid_ptr(IID_ID3D12PipelineState),
                           ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateGraphicsPipelineState({name})")
        _track(out.value, "pso")
        self._pso_cache[key] = ComPtr(out.value)
        return self._pso_cache[key]

    # ---- 命令 ------------------------------------------------------------
    def begin_frame(self):
        hr = self.cl.call(S_CL["Reset"], ctypes.c_long,
                          (ctypes.c_void_p, ctypes.c_void_p), self.allocator.p, None)
        if hr:
            raise OSError(hr, "CommandList.Reset")
        self._arena_ptr = 0
        return self.cl

    def submit(self):
        hr = self.cl.call(S_CL["Close"], ctypes.c_long, ())
        if hr:
            raise OSError(hr, "CommandList.Close")
        lists = (ctypes.c_void_p * 1)(self.cl.p)
        self.queue.call(S_QUEUE["ExecuteCommandLists"], None,
                        (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
                        1, ctypes.cast(lists, ctypes.POINTER(ctypes.c_void_p)))
        self._fence_val += 1
        hr = self.queue.call(S_QUEUE["Signal"], ctypes.c_long,
                             (ctypes.c_void_p, ctypes.c_uint64),
                             self.fence.p, self._fence_val)
        if hr:
            raise OSError(hr, "Queue.Signal")
        return self._fence_val

    def wait_gpu(self, timeout_ms=500):
        cur = self.fence.call(S_FENCE["GetCompletedValue"], ctypes.c_uint64, ())
        if cur < self._fence_val:
            hr = self.fence.call(S_FENCE["SetEventOnCompletion"], ctypes.c_long,
                                 (ctypes.c_uint64, ctypes.c_void_p),
                                 self._fence_val, self._event)
            if hr:
                raise OSError(hr, "SetEventOnCompletion")
            r = k32.WaitForSingleObject(self._event, timeout_ms)
            if r != 0:
                raise TimeoutError(f"GPU fence 超时 {timeout_ms}ms")

    def copy_gpu(self, dst, src):
        """GPU 内 CopyResource（向既有 command list 追加，无屏障——
        调用方负责状态；fence 同步由 submit/wait_gpu 负责）。"""
        cl = self.cl
        ls = D3D12_TEXTURE_COPY_LOCATION(pResource=dst.res.p, Type=0)
        ld = D3D12_TEXTURE_COPY_LOCATION(pResource=src.res.p, Type=0)
        cl.call(S_CL["CopyResource"], None,
                (ctypes.c_void_p, ctypes.c_void_p), ld.p, ls.p)

    def draw_pass5(self, cl, pso, dst, t0, ring):
        """5 输入全屏 pass（中值 W5：t0 + ring4 → depth）。"""
        dst._init_rest_if_needed(cl)
        dst.transition(cl, STATE_RENDER_TARGET)
        for t in [t0] + ring:
            t.to_read(cl)
        self._cb_next = (self._cb_next + 1) % self.CB_SLOTS
        slot = self._cb_next
        floats = (ctypes.c_float * self.CB_SLOT_STRIDE)()
        floats[0] = float(dst.w)
        ctypes.memmove(self.cb_map + slot * self.CB_SLOT_STRIDE * 4,
                       floats, self.CB_SLOT_STRIDE * 4)
        cl.call(S_CL["SetPipelineState"], None, (ctypes.c_void_p,), pso.p)
        cl.call(S_CL["SetGraphicsRootSignature"], None, (ctypes.c_void_p,),
                self.root_sig().p)
        cl.call(S_CL["SetGraphicsRootConstantBufferView"], None,
                (ctypes.c_uint, ctypes.c_uint64), 0,
                self.cb_res.call(S_RES["GetGPUVirtualAddress"], ctypes.c_uint64, ())
                + slot * self.CB_SLOT_STRIDE * 4)
        n = 5
        arena_slot = self._arena_ptr
        self._arena_ptr += n
        if self._arena_ptr > self.ARENA_N:
            raise RuntimeError("arena 溢出")
        dst_h = D3D12_CPU_HANDLE(self._arena_base + arena_slot * self.srv_inc)
        src_h = (D3D12_CPU_HANDLE * n)(
            *(D3D12_CPU_HANDLE(t.srv_cpu.ptr + i * self.srv_inc)
              for i, t in enumerate([t0] + ring)))
        sizes1 = (ctypes.c_uint * 1)(n)
        dsts1 = (D3D12_CPU_HANDLE * 1)(dst_h)
        self.dev.call(S_DEVICE["CopyDescriptors"], None,
                      (ctypes.c_uint, ctypes.POINTER(D3D12_CPU_HANDLE),
                       ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
                       ctypes.POINTER(D3D12_CPU_HANDLE),
                       ctypes.POINTER(ctypes.c_uint), ctypes.c_int),
                      1, dsts1, sizes1, n, src_h, sizes1, HEAP_TYPE_CBV_SRV_UAV)
        gpu = self._arena_gpu + arena_slot * self.srv_inc
        cl.call(S_CL["SetGraphicsRootDescriptorTable"], None,
                (ctypes.c_uint, ctypes.c_uint64), 0, gpu)
        vp = D3D12_VIEWPORT(0, 0, float(dst.w), float(dst.h), 0.0, 1.0)
        cl.call(S_CL["RSSetViewports"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_VIEWPORT)), 1, ctypes.byref(vp))
        r = D3D12_RECT(0, 0, dst.w, dst.h)
        cl.call(S_CL["RSSetScissorRects"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_RECT)), 1, ctypes.byref(r))
        rtv = D3D12_CPU_HANDLE(dst.rtv_cpu.ptr)
        cl.call(S_CL["OMSetRenderTargets"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_CPU_HANDLE), ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_uint),
                1, ctypes.byref(rtv), 0, None, 0)
        cl.call(S_CL["DrawInstanced"], None,
                (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint),
                3, 1, 0, 0)
        dst.to_rest(cl)
        for t in [t0] + ring:
            t.to_rest(cl)

    # ---- 全屏 pass -------------------------------------------------------
    def draw_pass(self, cl, pso, dst, srvs=(), consts=()):
        """全屏像素 pass：dst 渲染目标；srvs 输入纹理（≤8，arena 连续拷贝）；
        consts ≤64 floats（root CBV b0）。"""
        dst._init_rest_if_needed(cl)
        dst.transition(cl, STATE_RENDER_TARGET)
        for t in srvs:
            t.to_read(cl)
        self._cb_next = (self._cb_next + 1) % self.CB_SLOTS
        slot = self._cb_next
        floats = (ctypes.c_float * self.CB_SLOT_STRIDE)()
        for i, v in enumerate(consts[:self.CB_SLOT_STRIDE]):
            floats[i] = v
        ctypes.memmove(self.cb_map + slot * self.CB_SLOT_STRIDE * 4,
                       floats, self.CB_SLOT_STRIDE * 4)
        cl.call(S_CL["SetPipelineState"], None, (ctypes.c_void_p,), pso.p)
        cl.call(S_CL["SetGraphicsRootSignature"], None, (ctypes.c_void_p,),
                self.root_sig().p)
        cl.call(S_CL["SetGraphicsRootConstantBufferView"], None,
                (ctypes.c_uint, ctypes.c_uint64), 0,
                self.cb_res.call(S_RES["GetGPUVirtualAddress"], ctypes.c_uint64, ())
                + slot * self.CB_SLOT_STRIDE * 4)
        if srvs:
            n = len(srvs)
            arena_slot = self._arena_ptr
            self._arena_ptr += n
            if self._arena_ptr > self.ARENA_N:
                raise RuntimeError("arena 溢出（每帧描述符 > ARENA_N）")
            dst_h = D3D12_CPU_HANDLE(self._arena_base + arena_slot * self.srv_inc)
            src_h = (D3D12_CPU_HANDLE * n)(
                *(D3D12_CPU_HANDLE(t.srv_cpu.ptr) for t in srvs))
            sizes1 = (ctypes.c_uint * 1)(n)
            dsts1 = (D3D12_CPU_HANDLE * 1)(dst_h)
            self.dev.call(S_DEVICE["CopyDescriptors"], None,
                          (ctypes.c_uint, ctypes.POINTER(D3D12_CPU_HANDLE),
                           ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
                           ctypes.POINTER(D3D12_CPU_HANDLE),
                           ctypes.POINTER(ctypes.c_uint), ctypes.c_int),
                          1, dsts1, sizes1, n, src_h, sizes1, HEAP_TYPE_CBV_SRV_UAV)
            gpu = self._arena_gpu + arena_slot * self.srv_inc
            cl.call(S_CL["SetGraphicsRootDescriptorTable"], None,
                    (ctypes.c_uint, ctypes.c_uint64), 0, gpu)
        vp = D3D12_VIEWPORT(0, 0, float(dst.w), float(dst.h), 0.0, 1.0)
        cl.call(S_CL["RSSetViewports"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_VIEWPORT)), 1, ctypes.byref(vp))
        r = D3D12_RECT(0, 0, dst.w, dst.h)
        cl.call(S_CL["RSSetScissorRects"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_RECT)), 1, ctypes.byref(r))
        rtv = D3D12_CPU_HANDLE(dst.rtv_cpu.ptr)
        cl.call(S_CL["OMSetRenderTargets"], None,
                (ctypes.c_uint, ctypes.POINTER(D3D12_CPU_HANDLE), ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_uint),
                1, ctypes.byref(rtv), 0, None, 0)
        cl.call(S_CL["DrawInstanced"], None,
                (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint),
                3, 1, 0, 0)
        dst.to_rest(cl)
        for t in srvs:
            t.to_rest(cl)


class Swapchain:
    """HWND swapchain。P0 探针矩阵定模型：flip 优先 / BLT 兜底。"""

    def __init__(self, ctx, hwnd, w, h, model="flip", buffer_count=2,
                 fmt=FMT_B8G8R8A8):
        f = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory2(0, guid_ptr(IID_IDXGIFactory2), ctypes.byref(f))
        if hr:
            raise OSError(hr, "CreateDXGIFactory2")
        self.factory = ComPtr(f.value)
        _track(f.value, "factory")
        self.ctx = ctx
        d = DXGI_SWAP_CHAIN_DESC1(
            Width=w, Height=h, Format=fmt, Stereo=0,
            SampleDesc=DXGI_SAMPLE_DESC(1, 0),
            BufferUsage=DXGI_USAGE_RENDER_TARGET, BufferCount=buffer_count,
            Scaling=0,
            SwapEffect=(DXGI_SWAP_EFFECT_FLIP_DISCARD if model == "flip"
                        else DXGI_SWAP_EFFECT_DISCARD),
            AlphaMode=DXGI_ALPHA_MODE_IGNORE, Flags=0)
        out = ctypes.c_void_p()
        hr = self.factory.call(S_FACTORY2["CreateSwapChainForHwnd"], ctypes.c_long,
                               (ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(DXGI_SWAP_CHAIN_DESC1), ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                               ctx.queue.p, hwnd, ctypes.byref(d), None, None,
                               ctypes.byref(out))
        if hr:
            raise OSError(hr, f"CreateSwapChainForHwnd(model={model})")
        _track(out.value, "sc")
        self.sc = ComPtr(out.value)
        self.w, self.h, self.fmt = w, h, fmt
        self.model = model
        self._bb = []
        self._idx = 0
        for i in range(buffer_count):
            bb = ctypes.c_void_p()
            hr = self.sc.call(S_SC["GetBuffer"], ctypes.c_long,
                              (ctypes.c_uint, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_void_p)),
                              i, guid_ptr(IID_D3D12Resource), ctypes.byref(bb))
            if hr:
                raise OSError(hr, "GetBuffer")
            _track(bb.value, "bb")
            tex = Tex(ctx, ComPtr(bb.value), w, h, fmt, renderable=True)
            tex.rtv_cpu = ctx.alloc_rtv(bb.value, fmt)
            self._bb.append(tex)

    def next_rt(self):
        t = self._bb[self._idx % len(self._bb)]
        return t

    def present(self, sync=1):
        # FLIP 模型下 Present 后 backbuffer 隐式回 COMMON
        for t in self._bb:
            t.state = STATE_COMMON
            t._init_done = False
        self._idx += 1
        return self.sc.call(S_SC["Present"], ctypes.c_long,
                            (ctypes.c_uint, ctypes.c_uint), sync, 0)

    def resize(self, w, h):
        self.w, self.h = w, h
        for t in self._bb:
            _release(t.res, "bb")
        self._bb = []
        hr = self.sc.call(S_SC["ResizeBuffers"], ctypes.c_long,
                          (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_uint),
                          0, w, h, FMT_UNKNOWN, 0)
        if hr:
            raise OSError(hr, "ResizeBuffers")
        for i in range(len(self._bb) if self._bb else 2):
            bb = ctypes.c_void_p()
            hr = self.sc.call(S_SC["GetBuffer"], ctypes.c_long,
                              (ctypes.c_uint, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_void_p)),
                              i, guid_ptr(IID_D3D12Resource), ctypes.byref(bb))
            if hr:
                raise OSError(hr, "GetBuffer(resize)")
            tex = Tex(self.ctx, ComPtr(bb.value), w, h, self.fmt, renderable=True)
            tex.rtv_cpu = self.ctx.alloc_rtv(bb.value, self.fmt)
            self._bb.append(tex)


# ---------------------------------------------------------------------------
# 小件
# ---------------------------------------------------------------------------
# 模块级：ctypes Structure 每次定义都是新类型（类型↔StgDict↔字段描述符
# 成环 = 循环垃圾）。仅建纹理/视图时触发，但也没必要每次造一个新类型。
class _SRV(ctypes.Structure):
    _fields_ = [("Format", ctypes.c_uint), ("ViewDimension", ctypes.c_uint),
                ("Shader4ComponentMapping", ctypes.c_uint), ("_pad", ctypes.c_uint),
                ("u", ctypes.c_ubyte * 24)]


class _RTV(ctypes.Structure):
    _fields_ = [("Format", ctypes.c_uint), ("ViewDimension", ctypes.c_uint),
                ("u", ctypes.c_ubyte * 24)]


def _srv_desc(fmt):
    """D3D12_SHADER_RESOURCE_VIEW_DESC——SDK 尺寸 40B（union 含 UINT64 成员
    对齐 8；截短版会被驱动越界读 → 设备移除 INVALID_CALL，2026-11-21 P0）。"""
    d = _SRV(Format=fmt, ViewDimension=SRV_DIM_TEXTURE2D,
             Shader4ComponentMapping=COMP_MAP_DEFAULT)
    # Texture2D 联合成员：MostDetailedMip=0, MipLevels=0xFFFFFFFF（全 mip）,
    # PlaneSlice=0, ResourceMinLODClamp=0.0f
    struct.pack_into("<IIIf", d, 16, 0, 0xFFFFFFFF, 0, 0.0)
    return d


def _rtv_desc(fmt):
    """D3D12_RENDER_TARGET_VIEW_DESC——SDK 尺寸 32B。"""
    d = _RTV(Format=fmt, ViewDimension=RTV_DIM_TEXTURE2D)
    struct.pack_into("<II", d, 8, 0, 0)   # TEX2D_RTV: MipSlice=0, PlaneSlice=0
    return d





COMP_MAP_DEFAULT = 0x1688   # D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
