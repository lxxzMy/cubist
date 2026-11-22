# -*- coding: utf-8 -*-
"""D3D12 渲染运行时块（迁移 R1/R2）：C 助手 DLL（d3drender.dll，shim 同款
模式）的 Python 封装。DLL 侧封装全部 D3D12 ABI 敏感操作（PSO 创建/命令
录制/屏障/描述符/swapchain），Python 只持句柄 + 字节缓冲。

分层关系：
- d3d12.py：纯 ctypes D3D12（P0 基建/微基准/探针，保留用于对账与槽位提取）
- d3drender.py：生产封装（本文件），状态跟踪 + 屏障在 Python 侧显式管理

状态模型（与 GL 语义 1:1）：非 simultaneous-access 纹理在 ExecuteCommandLists
边界自动 decay 回 COMMON；RENDER_TARGET/COPY_DEST 等写状态不能从 COMMON
隐式提升 —— 故本封装 Tex.state 显式跟踪，每帧开头视为 COMMON（上帧 submit
后 decay），pass 序列中显式 transition。buf（UPLOAD/READBACK）为
simultaneous-access 语义，恒 COMMON 免屏障。
"""
import ctypes
import os

import numpy as np

from .adapter import pick as _pick_adapter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))  # D2S/gpu -> D2S -> 仓库根

_occluded_logged = False    # Present 返回 OCCLUDED 只提示一次

# DLL 搜索顺序与 dml_shim.py 同款：runtime（部署目录）→ 仓库根 → 构建产物
_DLL_CANDIDATES = (
    os.path.join(_REPO, "runtime", "d3drender.dll"),
    os.path.join(_REPO, "d3drender.dll"),
    os.path.join(_REPO, "_toolchain", "src", "d3drender.dll"),
)


def find_dll():
    for p in _DLL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


# DXGI_FORMAT（与 d3d12.py 一致）
FMT_BGRA8 = 87   # DXGI_FORMAT_B8G8R8A8_UNORM
FMT_R32F = 41    # DXGI_FORMAT_R32_FLOAT
FMT_R16U = 57    # DXGI_FORMAT_R16_UNORM

# D3D12_HEAP_TYPE（SDK 枚举：DEFAULT=1，0 是无效值）
HEAP_DEFAULT = 1
HEAP_UPLOAD = 2
HEAP_READBACK = 3

# D3D12_RESOURCE_STATES（Python 侧状态跟踪用；SDK 枚举实值）
# 0x1=VERTEX_CB 0x2=INDEX_BUFFER（buffer-only）；纹理 SRV 只走 PIXEL_SRV
ST_COMMON = 0
ST_PIXEL_SRV = 0x80          # PIXEL_SHADER_RESOURCE
ST_ALL_SRV = 0xC0            # NON_PIXEL|PIXEL（仅 buffer 用）
ST_RENDER_TARGET = 0x4
ST_COPY_DEST = 0x400
ST_COPY_SOURCE = 0x800

# D3D12_TEXTURE_DATA_PITCH_ALIGNMENT：buffer footprint RowPitch 须 256 对齐
PITCH_ALIGN = 256


def row_pitch(w, bytes_per_px=4):
    return ((w * bytes_per_px + PITCH_ALIGN - 1) // PITCH_ALIGN) * PITCH_ALIGN


def _setup(dll):
    d = dll
    p, n = ctypes.c_void_p, ctypes.c_uint32
    u64 = ctypes.c_uint64
    i32 = ctypes.c_int32

    d.d3dr_create.restype = p
    d.d3dr_create.argtypes = [u64, ctypes.c_int]
    d.d3dr_destroy.restype = None
    d.d3dr_destroy.argtypes = [p]
    d.d3dr_get_err.restype = ctypes.c_char_p
    d.d3dr_get_err.argtypes = [p]
    d.d3dr_create_pso.restype = p
    d.d3dr_create_pso.argtypes = [p, ctypes.c_char_p, p, n, p, n, n]
    d.d3dr_create_pso_blend.restype = p
    d.d3dr_create_pso_blend.argtypes = [p, ctypes.c_char_p, p, n, p, n, n,
                                        ctypes.c_int]
    d.d3dr_create_tex.restype = p
    d.d3dr_create_tex.argtypes = [p, n, n, n, ctypes.c_int, ctypes.c_int]
    d.d3dr_release_tex.restype = None
    d.d3dr_release_tex.argtypes = [p]
    d.d3dr_alloc_rtv.restype = p
    d.d3dr_alloc_rtv.argtypes = [p, p, n]
    d.d3dr_alloc_srv.restype = u64
    d.d3dr_alloc_srv.argtypes = [p, p, n]
    d.d3dr_make_upload_buf.restype = p
    d.d3dr_make_upload_buf.argtypes = [p, n]
    d.d3dr_make_readback_buf.restype = p
    d.d3dr_make_readback_buf.argtypes = [p, n]
    d.d3dr_buf_map.restype = p
    d.d3dr_buf_map.argtypes = [p]
    d3dr_buf_ptr = d.d3dr_buf_ptr
    d3dr_buf_ptr.restype = p
    d3dr_buf_ptr.argtypes = [p]
    d.d3dr_copy_buf_to_tex.restype = None
    d.d3dr_copy_buf_to_tex.argtypes = [p, p, p, n, n, n, n]
    d.d3dr_copy_tex_to_buf.restype = i32
    d.d3dr_copy_tex_to_buf.argtypes = [p, p, p, n, n, n, n]
    d.d3dr_copy_tex_to_tex.restype = i32
    d.d3dr_copy_tex_to_tex.argtypes = [p, p, p]
    d.d3dr_rb_begin.restype = i32
    d.d3dr_rb_begin.argtypes = [p]
    d.d3dr_rb_barrier.restype = i32
    d.d3dr_rb_barrier.argtypes = [p, p, n, n]
    d.d3dr_rb_copy_tex_to_buf.restype = i32
    d.d3dr_rb_copy_tex_to_buf.argtypes = [p, p, p, n, n, n, n]
    d.d3dr_rb_exec.restype = i32
    d.d3dr_rb_exec.argtypes = [p]
    d.d3dr_rb_copy_buf_to_buf.restype = i32
    d.d3dr_rb_copy_buf_to_buf.argtypes = [p, p, p, n]
    d.d3dr_get_dev.restype = u64
    d.d3dr_get_dev.argtypes = [p]
    d.d3dr_get_queue.restype = u64
    d.d3dr_get_queue.argtypes = [p]
    d.d3dr_get_desc_counts.restype = u64
    d.d3dr_get_desc_counts.argtypes = [p]
    d.d3dr_create_cs_pso.restype = p
    d.d3dr_create_cs_pso.argtypes = [p, ctypes.c_char_p, p, n]
    d.d3dr_dispatch.restype = i32
    d.d3dr_dispatch.argtypes = [p, p, ctypes.POINTER(u64), n,
                                ctypes.POINTER(u64), n,
                                ctypes.POINTER(ctypes.c_float), n, n, n, n]
    d.d3dr_create_uav_buf.restype = p
    d.d3dr_create_uav_buf.argtypes = [p, n]
    d.d3dr_uav_barrier.restype = i32
    d.d3dr_uav_barrier.argtypes = [p, p]
    d.d3dr_wait_flip.restype = i32
    d.d3dr_wait_flip.argtypes = [p, n]
    d.d3dr_signal_fence.restype = i32
    d.d3dr_signal_fence.argtypes = [p, p, ctypes.c_uint64]
    d.d3dr_alloc_uav.restype = u64
    d.d3dr_alloc_uav.argtypes = [p, p, n]
    d.d3dr_alloc_srv_buf.restype = u64
    d.d3dr_alloc_srv_buf.argtypes = [p, p, n]
    d.d3dr_begin_frame.restype = i32
    d.d3dr_begin_frame.argtypes = [p]
    d.d3dr_draw_pass.restype = i32
    d.d3dr_draw_pass.argtypes = [p, p, ctypes.POINTER(u64), n,
                                 ctypes.POINTER(ctypes.c_float), n, p, n, n,
                                 i32, i32, i32, i32, ctypes.c_int,
                                 i32, i32, i32, i32]
    d.d3dr_clear_rtv.restype = i32
    d.d3dr_clear_rtv.argtypes = [p, p, ctypes.c_float, ctypes.c_float,
                                 ctypes.c_float, ctypes.c_float]
    d.d3dr_submit.restype = i32
    d.d3dr_submit.argtypes = [p]
    d.d3dr_wait.restype = i32
    d.d3dr_wait.argtypes = [p, n]
    d.d3dr_transition.restype = i32
    d.d3dr_transition.argtypes = [p, p, n, n]
    d.d3dr_create_swapchain.restype = i32
    d.d3dr_create_swapchain.argtypes = [p, p, n, n, ctypes.c_int]
    d.d3dr_destroy_swapchain.restype = i32
    d.d3dr_destroy_swapchain.argtypes = [p]
    d.d3dr_present.restype = i32
    d.d3dr_present.argtypes = [p, ctypes.c_int]
    # 非阻塞呈现：flags 透传给 IDXGISwapChain::Present
    d.d3dr_present_flags.restype = i32
    d.d3dr_present_flags.argtypes = [p, ctypes.c_int, ctypes.c_uint]
    # 内部 fence 查询（非阻塞）：流水线判断"某次提交完成没有"
    d.d3dr_completed_fence.restype = ctypes.c_ulonglong
    d.d3dr_completed_fence.argtypes = [p]
    d.d3dr_cur_bb.restype = ctypes.c_uint
    d.d3dr_cur_bb.argtypes = [p]
    d.d3dr_last_fence.restype = ctypes.c_ulonglong
    d.d3dr_last_fence.argtypes = [p]
    d.d3dr_get_backbuffer_rtv.restype = p
    d.d3dr_get_backbuffer_rtv.argtypes = [p]
    d.d3dr_get_backbuffer_tex.restype = p
    d3dr_get_backbuffer_tex = d.d3dr_get_backbuffer_tex
    d3dr_get_backbuffer_tex.restype = p
    d3dr_get_backbuffer_tex.argtypes = [p]
    d.d3dr_get_srv_handle.restype = u64
    d.d3dr_get_srv_handle.argtypes = [p, p, n]
    d.d3dr_resize_swapchain.restype = i32
    d.d3dr_resize_swapchain.argtypes = [p, n, n]
    return d


_dll = None


def dll():
    global _dll
    if _dll is None:
        path = find_dll()
        if not path:
            raise RuntimeError(
                "[d3dr] d3drender.dll 不存在（runtime/ 或 _toolchain/src/）")
        _dll = _setup(ctypes.WinDLL(path))
    return _dll


class Buf:
    """UPLOAD/READBACK buffer（simultaneous-access 语义）或 UAV buffer。

    uav buf（V2 compute 目标）：state 显式跟踪（UAV 写状态不能从 COMMON
    提升，dispatch 前 transition(UNORDERED_ACCESS)；读回前 COPY_SOURCE）。
    """

    __slots__ = ("ctx", "res", "nbytes", "map", "state", "_uav", "_srv")

    def __init__(self, ctx, res, nbytes, map_ptr, state=ST_COMMON):
        self.ctx = ctx
        self.res = res
        self.nbytes = nbytes
        self.map = map_ptr
        self.state = state
        self._uav = None
        self._srv = None

    @property
    def uav(self):
        if self._uav is None:
            self._uav = dll().d3dr_alloc_uav(self.ctx, self.res, self.nbytes)
        return self._uav

    @property
    def srv(self):
        if self._srv is None:
            self._srv = dll().d3dr_alloc_srv_buf(self.ctx, self.res,
                                                 self.nbytes)
        return self._srv

    def transition(self, to_state):
        if to_state == self.state:
            return
        dll().d3dr_transition(self.ctx, self.res, self.state, to_state)
        self.state = to_state

    def uav_barrier(self):
        """UAV 执行屏障（同资源在前后 dispatch 间读写依赖时必需）。"""
        dll().d3dr_uav_barrier(self.ctx, self.res)


class Tex:
    """D3D12 纹理（句柄）+ 显式状态跟踪。

    state 语义：每次 submit 后 GPU 端 decay 回 COMMON —— 本封装在
    begin_frame 时不自动重置，由调用方（pass 序列构建者）保证序列自洽；
    单帧内 transition 后 state 即更新。buf 类资源恒 COMMON。
    no_barrier=True（外部共享资源）：不发显式屏障（OpenSharedHandle 打开
    的资源靠隐式状态提升，显式屏障曾致 Close E_INVALIDARG——探针 E 实锤），
    仅记录 state。
    """

    __slots__ = ("ctx", "res", "w", "h", "fmt", "is_buf", "no_barrier",
                 "state", "_rtv", "_srv", "_upbuf", "_upmap", "_rbbuf")

    def __init__(self, ctx, res, w, h, fmt, is_buf=False, no_barrier=False):
        self.ctx = ctx
        self.res = res
        self.w = w
        self.h = h
        self.fmt = fmt
        self.is_buf = is_buf
        self.no_barrier = no_barrier
        self.state = ST_COMMON
        self._rtv = None
        self._srv = None
        self._upbuf = None
        self._upmap = None
        self._rbbuf = None

    @property
    def rtv(self):
        if self._rtv is None:
            self._rtv = dll().d3dr_alloc_rtv(self.ctx, self.res, self.fmt)
        return self._rtv

    @property
    def srv(self):
        """SRV 的 GPU 句柄（arena 固定槽；root table 直接绑定）。"""
        if self._srv is None:
            self._srv = dll().d3dr_alloc_srv(self.ctx, self.res, self.fmt)
        return self._srv

    def transition(self, to_state):
        if self.is_buf or to_state == self.state:
            return
        if not self.no_barrier:
            dll().d3dr_transition(self.ctx, self.res, self.state, to_state)
        self.state = to_state

    def release(self):
        d = dll()
        if self._upbuf is not None:
            d.d3dr_release_tex(self._upbuf)
            self._upbuf = None
        if self._rbbuf is not None:
            d.d3dr_release_tex(self._rbbuf)
            self._rbbuf = None
        if self.res is not None:
            d.d3dr_release_tex(self.res)
            self.res = None


class D3DRenderer:
    """d3drender.dll RenderCtx 封装：设备/队列/fence/描述符堆/PSO/swapchain。"""

    def __init__(self, luid=0, debug=False):
        d = dll()
        if luid == 0:
            info = _pick_adapter()
            luid = info.luid if info is not None else 0
        self.luid = int(luid) & 0xFFFFFFFFFFFFFFFF   # 创建卡（换屏检测用）
        # adapter.py 的 luid 为 int64（HighPart<<32 | LowPart）；d3dr_create
        # 的 adapter_luid_match=0 表示"取第一个硬件卡"
        self.ctx = d.d3dr_create(
            ctypes.c_uint64(luid & 0xFFFFFFFFFFFFFFFF), 1 if debug else 0)
        if not self.ctx:
            raise RuntimeError(
                "[d3dr] d3dr_create 失败: " + err_text(d, self.ctx))
        self._vs_keep = []   # PSO 期间 shader 字节持有（DLL 不拷贝）
        self._ps_keep = []

    # ---- PSO ----
    def create_pso(self, name, vs, ps, rtv_fmt=FMT_BGRA8):
        """vs/ps: Cso（hlsl.compile_shader 产物）或 (ptr, nbytes) 元组。"""
        d = dll()
        vsp, vsn = _blob(vs)
        psp, psn = _blob(ps)
        self._vs_keep.append(vs)
        self._ps_keep.append(ps)
        pso = d.d3dr_create_pso(self.ctx, name.encode(), vsp, vsn, psp, psn,
                                rtv_fmt)
        if not pso:
            raise RuntimeError("[d3dr] create_pso(%s) 失败: %s"
                               % (name, self.last_err()))
        return pso

    def create_pso_blend(self, name, vs, ps, rtv_fmt=FMT_BGRA8):
        """alpha 混合 PSO（SrcAlpha/InvSrcAlpha；overlay/cursor 用）。"""
        d = dll()
        vsp, vsn = _blob(vs)
        psp, psn = _blob(ps)
        self._vs_keep.append(vs)
        self._ps_keep.append(ps)
        pso = d.d3dr_create_pso_blend(self.ctx, name.encode(), vsp, vsn, psp,
                                      psn, rtv_fmt, 1)
        if not pso:
            raise RuntimeError("[d3dr] create_pso_blend(%s) 失败: %s"
                               % (name, self.last_err()))
        return pso

    # ---- 纹理 ----
    def create_tex(self, w, h, fmt=FMT_BGRA8, heap=HEAP_DEFAULT,
                   renderable=True):
        res = dll().d3dr_create_tex(self.ctx, w, h, fmt, heap,
                                    1 if renderable else 0)
        if not res:
            raise RuntimeError("[d3dr] create_tex(%dx%d fmt=%d) 失败: %s"
                               % (w, h, fmt, self.last_err()))
        return Tex(self.ctx, res, w, h, fmt)

    def upload_tex(self, w, h, fmt=FMT_R32F, renderable=False):
        """捆绑便捷对：目标纹理 + 常驻 upload buffer（自拷贝场景）。"""
        tex = self.create_tex(w, h, fmt, HEAP_DEFAULT, renderable)
        buf = self.make_upload_buffer(row_pitch(w, _bpp(fmt)) * h)
        tex._upbuf = buf.res
        tex._upmap = buf.map
        return tex

    def make_upload_buffer(self, nbytes):
        """裸 upload buffer（目标纹理由调用方在 copy_buf_to_tex 指定）。"""
        d = dll()
        res = d.d3dr_make_upload_buf(self.ctx, nbytes)
        if not res:
            raise RuntimeError("[d3dr] upload buf 失败: " + self.last_err())
        m = d.d3dr_buf_map(res)
        if not m:
            raise RuntimeError("[d3dr] upload buf map 失败: " + self.last_err())
        return Buf(self.ctx, res, nbytes, m)

    def upload_to_buf(self, buf, np_data, bytes_per_px=None):
        """np (h,w) → CPU 侧 upload buffer（256 对齐 pitch，逐行补 padding）。
        bytes_per_px：按【纹理格式】的字节宽（如 BGRA8=4）；缺省取
        np_data.itemsize（仅 (h,w) 紧凑布局正确——(h,w,ch) u8 必须显式传）。
        【性能】UPLOAD heap 映射是 WriteCombining——必须走 memmove 顺序
        SIMD 写；ctypes slice 赋值逐小片拷贝实测 7.75ms/帧（200KB）。"""
        h, w = np_data.shape[:2]
        bpp = bytes_per_px if bytes_per_px is not None else np_data.itemsize
        pitch = row_pitch(w, bpp)
        rowb = w * bpp
        if pitch == rowb:
            src = np.ascontiguousarray(np_data).ravel()
            ctypes.memmove(buf.map, src.ctypes.data, pitch * h)
        else:
            src = np.ascontiguousarray(np_data).ravel()
            saddr = src.ctypes.data
            for y in range(h):
                ctypes.memmove(buf.map + y * pitch,
                               saddr + y * rowb, rowb)

    def copy_buf_to_tex(self, buf, tex):
        """upload buffer → 纹理（录制期间调用；COMMON→COPY_DEST→PIXEL_SRV）。"""
        d = dll()
        pitch = row_pitch(tex.w, _bpp(tex.fmt))
        tex.transition(ST_COPY_DEST)
        d.d3dr_copy_buf_to_tex(self.ctx, buf.res, tex.res,
                               tex.w, tex.h, tex.fmt, pitch)
        tex.transition(ST_PIXEL_SRV)

    def copy_upload(self, tex):
        """捆绑对自拷贝：Tex._upbuf → tex 自身（upload_tex 场景便捷入口）。"""
        d = dll()
        pitch = row_pitch(tex.w, _bpp(tex.fmt))
        tex.transition(ST_COPY_DEST)
        d.d3dr_copy_buf_to_tex(self.ctx, tex._upbuf, tex.res,
                               tex.w, tex.h, tex.fmt, pitch)
        tex.transition(ST_PIXEL_SRV)

    def upload(self, tex, np_data, bytes_per_px=None):
        """np_data (h,w[,c]) → 写 CPU 侧 upload buffer（256 对齐 pitch，
        逐行补 padding）。pitch 按【纹理格式】字节宽（bytes_per_px 缺省
        _bpp(tex.fmt)，np.itemsize 仅对紧凑单值布局正确）。实际 GPU 拷贝
        在 copy_upload(tex) 时录进当前命令列表（必须在 begin_frame 后）。
        【性能】同 upload_to_buf：WC 内存必须 memmove。"""
        h, w = np_data.shape[:2]
        bpp = bytes_per_px if bytes_per_px is not None else _bpp(tex.fmt)
        pitch = row_pitch(w, bpp)
        rowb = w * bpp
        src = np.ascontiguousarray(np_data).ravel()
        saddr = src.ctypes.data
        if pitch == rowb:
            ctypes.memmove(tex._upmap, saddr, pitch * h)
        else:
            for y in range(h):
                ctypes.memmove(tex._upmap + y * pitch,
                               saddr + y * rowb, rowb)

    def readback(self, tex, dtype):
        """一站式读回：录 tex→readback buf 拷贝（独立帧）→ submit → wait →
        返回 numpy。dtype：np.uint8 → BGRA (h,w,4)；np.float32 → R32F (h,w)。
        必须在前一帧 submit+wait 之后调用（begin_frame 会 Reset 命令列表）。"""
        import numpy as np

        d = dll()
        pitch = row_pitch(tex.w, _bpp(tex.fmt))
        if tex._rbbuf is None:
            tex._rbbuf = d.d3dr_make_readback_buf(self.ctx, pitch * tex.h)
            if not tex._rbbuf:
                raise RuntimeError("[d3dr] readback buf 失败: "
                                   + self.last_err())
        # readback 专用列表（旁路 per-frame 轮换——不打乱帧级 allocator 对账）
        if d.d3dr_rb_begin(self.ctx) != 0:
            raise RuntimeError("[d3dr] rb_begin 失败")
        if tex.state != ST_COPY_SOURCE:
            d.d3dr_rb_barrier(self.ctx, tex.res, tex.state, ST_COPY_SOURCE)
        d.d3dr_rb_copy_tex_to_buf(self.ctx, tex.res, tex._rbbuf,
                                  tex.w, tex.h, tex.fmt, pitch)
        if d.d3dr_rb_exec(self.ctx) != 0:
            raise RuntimeError("[d3dr] rb_exec 失败")
        self.wait()
        tex.state = ST_COMMON   # rb 屏障后回 COMMON（记录对齐）
        ptr = d.d3dr_buf_ptr(tex._rbbuf)
        # dtype 决定解读：单字节 dtype → 按 fmt 的 bpp 解成多通道
        per_px = _bpp(tex.fmt) if np.dtype(dtype).itemsize == 1 else 1
        rowb = tex.w * per_px * np.dtype(dtype).itemsize
        buf = (ctypes.c_char * (pitch * tex.h)).from_address(ptr)
        if pitch == rowb:
            arr = np.frombuffer(buf, dtype=dtype,
                                count=tex.w * tex.h * per_px)
        else:   # 逐行剥 pitch padding
            arr = np.empty(tex.h * tex.w * per_px, dtype=dtype)
            for y in range(tex.h):
                arr[y * tex.w * per_px:(y + 1) * tex.w * per_px] = \
                    np.frombuffer(buf[y * pitch:y * pitch + rowb],
                                  dtype=dtype)
        out = arr.reshape(tex.h, tex.w, per_px)
        return out.copy() if per_px > 1 else out[:, :, 0].copy()

    # ---- 帧录制 ----
    def begin_frame(self):
        dll().d3dr_begin_frame(self.ctx)

    def draw_pass(self, pso, srvs, consts, rtv, dst_w, dst_h,
                  viewport=None, scissor=None, stats=None):
        """全屏 pass。srvs: [Tex, ...]（≤8，取其 .srv GPU 句柄）；
        consts: float 数组（≤64）；rtv: Tex（用其 .rtv）或裸 handle；
        viewport/scissor: (x, y, w, h) 像素（缺省=全 dst）；
        stats: 判据 Buf → 绑到 t7（root slot7；空位用 srvs[0] 填充）。"""
        d = dll()
        hs = [_srv_of(t) for t in (srvs or [])]
        if stats is not None:
            filler = hs[0] if hs else 0
            hs = (hs + [filler] * 8)[:7] + [
                getattr(stats, "srv", stats)]
        handles = [ctypes.c_uint64(h) for h in hs]
        arr = (ctypes.c_uint64 * len(handles))(*handles) if handles else None
        cb = (ctypes.c_float * len(consts))(*consts) if consts else None
        rtv_h = rtv.rtv if isinstance(rtv, Tex) else rtv
        vx, vy, vw, vh = viewport if viewport else (0, 0, dst_w, dst_h)
        if scissor:
            ux, uy, uw, uh = scissor
            use_sc, sargs = 1, (ux, uy, uw, uh)
        else:
            use_sc, sargs = 0, (0, 0, dst_w, dst_h)
        hr = d.d3dr_draw_pass(self.ctx, pso, arr, len(handles), cb,
                              len(consts) if consts else 0, rtv_h, dst_w,
                              dst_h, vx, vy, vw, vh, use_sc, *sargs)
        if hr != 0:
            raise RuntimeError("[d3dr] draw_pass 失败 hr=%s: %s"
                               % (hr, self.last_err()))

    def clear_rtv(self, rtv, rgba=(0.0, 0.0, 0.0, 1.0)):
        """清渲染目标（GL ctx.clear 对应物）。TAB/深度图有黑边，必须先清，
        否则视口外残留轮换缓冲里的旧帧内容。rtv: Tex 或裸 CPU 句柄。"""
        rtv_h = rtv.rtv if isinstance(rtv, Tex) else rtv
        hr = dll().d3dr_clear_rtv(self.ctx, rtv_h,
                                  float(rgba[0]), float(rgba[1]),
                                  float(rgba[2]), float(rgba[3]))
        if hr != 0:
            raise RuntimeError("[d3dr] clear_rtv 失败 hr=%s: %s"
                               % (hr, self.last_err()))

    def submit(self):
        hr = dll().d3dr_submit(self.ctx)
        if hr != 0:
            raise RuntimeError("[d3dr] submit 失败 hr=%s: %s"
                               % (_hr(hr), self.last_err()))

    def wait(self, timeout_ms=10000):
        hr = dll().d3dr_wait(self.ctx, timeout_ms)
        if hr != 0:
            raise RuntimeError("[d3dr] wait 失败 hr=%s" % _hr(hr))

    def copy_tex_to_tex(self, dst, src):
        """GPU 内整资源拷贝（CopyResource；录制期间调用）。含屏障：
        src→COPY_SOURCE / dst→COPY_DEST（提交后 decay 回 COMMON）。"""
        d = dll()
        src.transition(ST_COPY_SOURCE)
        dst.transition(ST_COPY_DEST)
        d.d3dr_copy_tex_to_tex(self.ctx, dst.res, src.res)
        src.state = ST_COMMON
        dst.state = ST_COMMON

    # ---- swapchain ----
    def create_swapchain(self, hwnd, w, h, use_flip=True):
        self._hwnd = hwnd
        self._use_flip = use_flip
        hr = dll().d3dr_create_swapchain(self.ctx, hwnd, w, h,
                                         1 if use_flip else 0)
        if hr != 0:
            raise RuntimeError("[d3dr] create_swapchain 失败 hr=%s: %s"
                               % (_hr(hr), self.last_err()))
        self._sc_w, self._sc_h = w, h

    @property
    def backbuffer(self):
        """当前 backbuffer（Tex 包装，含状态跟踪；PRESENT 后回 COMMON）。"""
        d = dll()
        res = d.d3dr_get_backbuffer_tex(self.ctx)
        rtv_h = d.d3dr_get_backbuffer_rtv(self.ctx)
        t = Tex(self.ctx, res, self._sc_w, self._sc_h, FMT_BGRA8)
        t._rtv = rtv_h
        t.state = ST_COMMON
        return t

    def present(self, sync_interval=1, nowait=False):
        """上屏。nowait=True 走 DXGI_PRESENT_DO_NOT_WAIT：队列满时不阻塞，
        返回 WAS_STILL_DRAWING 视为"本拍跳过"（返回 False）。"""
        d = dll()
        if nowait:
            # 非阻塞呈现：先用 frame-latency waitable 探一下有没有呈现槽。
            # 没有任何槽 → 本拍只提交不上屏（丢这一帧），绝不阻塞。
            # 注：DXGI_PRESENT_DO_NOT_WAIT(0x400) 在本机的"layered + flip +
            # waitable"组合下会返回 INVALID_CALL，所以走探针 + 跳过这条路。
            try:
                _ok = self.wait_flip(0)
            except Exception:
                _ok = True
            if not _ok:
                d.d3dr_submit(self.ctx)     # 录好的显示命令照常提交
                return False
            hr = d.d3dr_present(self.ctx, sync_interval)
        else:
            hr = d.d3dr_present(self.ctx, sync_interval)
        if hr == 0:
            return True
        # DXGI_STATUS_OCCLUDED（0x087A0001）是成功码：窗口被遮挡/不可见，
        # 帧不出光而已，不是失败（AMD 机器实测会被当异常刷屏）。
        if (hr & 0xFFFFFFFF) == 0x087A0001:
            global _occluded_logged
            if not _occluded_logged:
                _occluded_logged = True
                print("[d3dr] Present 返回 OCCLUDED（窗口被遮挡，跳过）",
                      flush=True)
            return True
        raise RuntimeError("[d3dr] present 失败 hr=%s: %s"
                           % (_hr(hr), self.last_err()))

    def completed_fence(self):
        """内部 fence 已完成值（非阻塞）。"""
        return int(dll().d3dr_completed_fence(self.ctx))

    def last_fence(self):
        """最近一次提交分配到的内部 fence 值。"""
        return int(dll().d3dr_last_fence(self.ctx))

    def cur_bb(self):
        """当前后缓冲序号（探针：画面回退时对照 bb）。"""
        return int(dll().d3dr_cur_bb(self.ctx))

    def wait_flip(self, timeout_ms=0):
        """等翻转落地（waitable object）。timeout_ms=0 → 非阻塞探测；
        返回 True = 已落地（Present 的帧已被合成/显示）。"""
        return dll().d3dr_wait_flip(self.ctx, int(timeout_ms)) == 0

    def signal_fence(self, fence, value):
        """渲染队列 signal 一张外部共享 fence（裸 D3D12Fence*）。

        槽回收用：帧末 signal，采集侧写环槽前 Wait 到该值，保证 GPU 已经
        不再采样该槽。失败不致命（最多退回旧行为）。
        """
        hr = dll().d3dr_signal_fence(self.ctx,
                                     ctypes.c_void_p(int(fence)),
                                     int(value))
        if hr != 0:
            raise RuntimeError("[d3dr] signal_fence 失败 hr=%s" % _hr(hr))

    def resize_swapchain(self, w, h):
        # 重建式 resize：ResizeBuffers 在本环境 flip swapchain 上 access
        # violation（WinError 2173），绕开——销毁+重建语义等价。
        # in-flight flip 由 swapchain waitable object 排空（sleep 不可靠）。
        self.wait()
        hr = dll().d3dr_destroy_swapchain(self.ctx)
        if hr != 0:
            raise RuntimeError("[d3dr] destroy_swapchain 失败 hr=%s" % _hr(hr))
        hr = dll().d3dr_create_swapchain(self.ctx, self._hwnd, w, h,
                                         self._use_flip)
        if hr != 0:
            raise RuntimeError("[d3dr] resize(create) 失败 hr=%s: %s"
                               % (_hr(hr), self.last_err()))
        self._sc_w, self._sc_h = w, h

    def wrap_external(self, res_ptr, w, h, fmt=FMT_BGRA8):
        """包装外部导入的 D3D12Resource（DDA 桥 OpenSharedHandle 产物）。

        no_barrier：opened 共享资源用隐式状态提升，不发显式屏障。
        release 责任在桥（外部资源生命周期归 OpenSharedHandle 侧管理）。
        """
        if not res_ptr:
            raise RuntimeError("[d3dr] wrap_external: res_ptr 为空")
        return Tex(self.ctx, res_ptr, w, h, fmt, no_barrier=True)

    # ---- compute（V2 GPU 判据） -----------------------------------------
    def create_cs_pso(self, name, cs):
        d = dll()
        psp, psn = _blob(cs)
        self._ps_keep.append(cs)
        pso = d.d3dr_create_cs_pso(self.ctx, name.encode(), psp, psn)
        if not pso:
            raise RuntimeError("[d3dr] create_cs_pso(%s) 失败: %s"
                               % (name, self.last_err()))
        return pso

    def make_uav_buf(self, nbytes):
        """UAV 目标 buffer（DEFAULT+ALLOW_UNORDERED_ACCESS；UAVStateBuf）。"""
        res = dll().d3dr_create_uav_buf(self.ctx, nbytes)
        if not res:
            raise RuntimeError("[d3dr] uav buf 失败: " + self.last_err())
        b = Buf(self.ctx, res, nbytes, None)
        b.state = ST_COMMON
        return b

    def dispatch(self, pso, uavs, srvs, consts, gx, gy=1, gz=1):
        """compute dispatch（录进当前帧列表）。uavs/srvs: [Buf|Tex]。"""
        d = dll()
        uav_h = [ctypes.c_uint64(getattr(b, "uav", b))
                 for b in (uavs or [])]
        srv_h = [ctypes.c_uint64(_srv_of(t)) for t in (srvs or [])]
        ua = (ctypes.c_uint64 * len(uav_h))(*uav_h) if uav_h else None
        sa = (ctypes.c_uint64 * len(srv_h))(*srv_h) if srv_h else None
        cb = (ctypes.c_float * len(consts))(*consts) if consts else None
        hr = d.d3dr_dispatch(self.ctx, pso, ua, len(uav_h), sa, len(srv_h),
                             cb, len(consts) if consts else 0, gx, gy, gz)
        if hr != 0:
            raise RuntimeError("[d3dr] dispatch 失败 hr=%s: %s"
                               % (hr, self.last_err()))

    def get_dev_ptr(self):
        """裸 ID3D12Device*（shim v2 DML 注入；引用计数不增，勿 Release）。"""
        d = dll()
        d.d3dr_get_dev.restype = ctypes.c_uint64
        d.d3dr_get_dev.argtypes = [ctypes.c_void_p]
        v = d.d3dr_get_dev(self.ctx)
        if not v:
            raise RuntimeError("[d3dr] get_dev_ptr 失败")
        return ctypes.c_void_p(v)

    def get_queue_ptr(self):
        """裸 ID3D12CommandQueue*（shim v2 DML1 队列注入；同上）。"""
        d = dll()
        d.d3dr_get_queue.restype = ctypes.c_uint64
        d.d3dr_get_queue.argtypes = [ctypes.c_void_p]
        v = d.d3dr_get_queue(self.ctx)
        if not v:
            raise RuntimeError("[d3dr] get_queue_ptr 失败")
        return ctypes.c_void_p(v)

    def last_err(self):
        return err_text(dll(), self.ctx)

    def destroy(self):
        if getattr(self, "ctx", None):
            dll().d3dr_destroy(self.ctx)
            self.ctx = None


def _bpp(fmt):
    return 4 if fmt in (FMT_BGRA8, FMT_R32F) else 2 if fmt == FMT_R16U else 4


def _srv_of(t):
    """Tex/Buf → SRV GPU 句柄；int（裸句柄）透传。"""
    if isinstance(t, (Tex, Buf)):
        return t.srv
    return t


def _blob(sh):
    """Cso 或 (ptr, n) → (void_p, n)。"""
    if hasattr(sh, "data_ptr"):
        return sh.data_ptr, sh.nbytes
    return sh[0], sh[1]


def _hr(hr):
    return "0x%08x" % (hr & 0xFFFFFFFF)


def err_text(d, ctx):
    try:
        m = d.d3dr_get_err(ctx)
        return (m or b"").decode(errors="replace")
    except Exception:
        return "(no err)"
