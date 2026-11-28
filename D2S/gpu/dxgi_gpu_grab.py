# -*- coding: utf-8 -*-
"""颜色零拷贝（全屏 SBS）v3。

分工（每样资源只由一个线程创建/使用）：
  采集线程（bootstrap）：创 D3D11 设备 → 取桌面尺寸 → 建 duplication →
                        建 3 张环纹理 → 置 ready → 等 registered → 开始搬帧
  渲染线程：等 ready → wglDXRegisterObjectNV 一次性注册（此时采集还没开跑）
           → 置 registered → 之后每帧 lock / render / unlock

为什么设备也要在采集线程建：DXGI duplication 对创建线程敏感，v2 里
设备在渲染线程建、duplication 在采集线程建，AcquireNextFrame 会直接卡死
（14s 零帧、超时不生效）。
"""
import ctypes
import ctypes.wintypes as wt
import os
import threading
import time

from .dxgi_dup import DXGI_OUTDUPL_FRAME_INFO
from .rtcom import (ComPtr, TexDesc, create_device, create_tex,
                    ctx_copy_resource, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_WAIT_TIMEOUT, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
from . import dxgi_dup as _dd
from .dda_bridge import d3d_ring_misc

_IID_TEX2D = "6f15aaf2-d208-4e89-9ab4-489535d34f9c"

# QPC（DDA 的 LastPresentTime 与它同源，端到端延迟仪表用）
_k32_qpc = ctypes.WinDLL("kernel32", use_last_error=True)


def qpc():
    c = ctypes.c_longlong()
    _k32_qpc.QueryPerformanceCounter(ctypes.byref(c))
    return c.value


def qpc_freq():
    f = ctypes.c_longlong()
    _k32_qpc.QueryPerformanceFrequency(ctypes.byref(f))
    return f.value or 1
_IID_KEYEDMUTEX = "9d8e1289-d7b3-465f-8126-250e349af85d"
# IDXGIKeyedMutex 的虚表槽位（2026-11-28 解虚表跳板实测，别凭记忆改）：
# 7=GetSharedHandle（与 IDXGIResource1 第 7 槽是同一个函数）、8=AcquireSync、
# 9=ReleaseSync。原先写成 7/8，每帧实际调的是 GetSharedHandle(NULL) 和
# AcquireSync(0, 超时=寄存器残值)——第一次 AcquireSync 成功后锁一直被持有，
# 之后的拷贝才落地，所以每个槽的第一次拷贝被丢（进 3D 的前 3 帧全黑，
# poll 却报有新帧）。
S_KEYED_ACQUIRE = 8
S_KEYED_RELEASE = 9
S_CTX4_SIGNAL = 147
S_CTX4_WAIT = 148


class Mapped(ctypes.Structure):
    _fields_ = [("pData", ctypes.c_void_p), ("RowPitch", ctypes.c_uint),
                ("DepthPitch", ctypes.c_uint)]
# IDXGIOutput5：DuplicateOutput1（槽 26）支持指定格式，能应对 HDR/硬件叠加场景
_IID_OUTPUT5 = "80a07424-ab52-42eb-833c-0c42fd282d98"
RING = 3
DXGI_FORMAT_R16G16B16A16_FLOAT = 10


# 模块级：ctypes Structure 每次定义都是新类型（类型↔StgDict↔字段描述符
# 成环 = 循环垃圾），output_size_now 每 0.5s 调一次，不能在里面建。
class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _OUT_DESC(ctypes.Structure):
    _fields_ = [("DeviceName", ctypes.c_wchar * 32),
                ("DesktopCoordinates", _RECT),
                ("AttachedToDesktop", wt.BOOL),
                ("Rotation", wt.UINT),
                ("Monitor", ctypes.c_void_p)]


def _output_size(dev, monitor_index):
    """IDXGIOutput::GetDesc 的 DesktopCoordinates → 桌面尺寸。"""
    dxdev = ComPtr(dev.qi(_dd.IID_IDXGIDevice))
    adapter_pp = ctypes.c_void_p()
    dxdev.call(7, ctypes.c_long, (ctypes.POINTER(ctypes.c_void_p),),
               ctypes.byref(adapter_pp))
    adapter = ComPtr(adapter_pp.value)

    for _k in [monitor_index] + [k for k in range(4) if k != monitor_index]:
        out_pp = ctypes.c_void_p()
        hr = adapter.call(7, ctypes.c_long,
                          (wt.UINT, ctypes.POINTER(ctypes.c_void_p)),
                          _k, ctypes.byref(out_pp))
        if hr or not out_pp.value:
            continue
        out = ComPtr(out_pp.value)
        desc = _OUT_DESC()
        if not out.call(7, ctypes.c_long, (ctypes.POINTER(_OUT_DESC),),
                        ctypes.byref(desc)):
            w = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left
            h = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top
            if w > 0 and h > 0:
                return w, h
    raise RuntimeError("取不到输出尺寸")


class DxgiGpuColor:
    def __init__(self, gpu_q, monitor_index, tag=None, adapter=None):
        self.gpu_q = gpu_q
        # monitor_index = **该 adapter 内**的输出序号（placement.pick_for_monitor
        # 解析；双屏分属两块 GPU 时不能再用全局序号）
        self.monitor = int(monitor_index)
        self.adapter = adapter      # AdapterInfo（None → build_inline 自选）
        self.tag = tag
        self.gen = 1
        self.gl_locked = [False] * RING
        self.inline = False
        self._last_pub = None
        self._pst = {"n": 0, "ok": 0, "accsum": 0, "acc0": 0, "wait": 0, "na": 0,
                     "other": 0, "multi": 0}
        self._last_acq_t = 0.0
        self._acq_gap_max = 0.0
        # 取帧节奏诊断（只记录，不改行为；30s 随 poll30s 打一行）：轮询间隔、
        # 取帧间隔、"桌面呈现时刻 → 我们取到"的延迟、攥帧时长。
        self._qpf = qpc_freq()
        self._prev_poll_end = 0.0
        self._prev_ok_t = 0.0
        self._acq_lat_ms = 0.0
        self._acc_frames = 0
        self._acq_log = []        # (前一轮询间隔ms, 取帧间隔ms, 取到延迟ms, 累计帧数)
        self._hold_log = []       # 每次释放时攥了多久（ms）
        self._pgap_n = [0, 0, 0]  # 轮询间隔 ≥8 / ≥16 / ≥33 ms 的次数
        self._pgap_max = 0.0
        # DDA 到达率遥测窗：30s 打一行（旧版每 60 次 poll 打一行——3D 空转
        # 时 poll 上千次/秒，11 分钟刷 5000+ 行）
        self._pst_t = time.perf_counter()
        self._pst_n = 0
        # 当前正在显示的槽：采集绝不写它（否则会覆盖正在采样的纹理 → 画面撕裂/闪烁）
        self.avoid = -1
        # 第二个受保护的槽（流水线：上一对已上屏、还会被重呈现反复采样的
        # 那一帧；avoid 留给在飞的那一帧）。见 _free_slot。
        self.avoid2 = -1
        self.writing = [False] * RING      # 采集线程正在写该槽（渲染侧等它写完）
        self.ready = threading.Event()       # 采集线程：资源就绪
        self.registered = threading.Event()  # 渲染线程：GL 注册完成
        self._stop = threading.Event()
        self._dev = None
        self._ctx = None
        self._dup = None
        self._ring = []
        self._w = self._h = 0
        self._ring_i = 0
        self._dropped = 0
        self.dx = None
        self.ring_gl = None
        # D3D12 桥模式（D2S_RENDER=d3d）：NTHANDLE|KM 环 + producer fence。
        # bridge_fence/bridge_ctx4 由 gpu/dda_bridge.DdaBridge 注入；
        # bridge_kms = 每 slot 的 IDXGIKeyedMutex（建环时各取锁一次、一直
        # 持有，见 _setup_bridge_kms）；bridge_seq 与消费端 DdaBridge.next_seq
        # 一一对应。
        self.bridge_fence = None
        self.bridge_ctx4 = None
        self.bridge_kms = None
        self.bridge_seq = 0
        # 反向（消费者→生产者）fence：D3D12 每帧 signal，采集写某槽前
        # Wait 该槽的最后 signal 值——保证 GPU 已经不再采样它才覆盖。
        self.bridge_fence_cons = None
        self.slot_done = [0] * RING
        # 每个环槽最后一次被写入时的采集序号（bridge_seq）。消费端显示前
        # 用它核对"槽里还是不是我的那一帧"。
        self.slot_seq = [0] * RING
        # DDA 帧所有权：发出 GPU 拷贝后必须等拷贝完成（producer fence
        # 完成值）再 ReleaseFrame——否则 DXGI 可把同一桌面表面交给下一帧，
        # DWM 的写入与我们未完成的拷贝并发 → 拷出"上半旧帧/下半新帧"
        # 的撕裂画面（播放视频时最明显，暂停即消失）。
        # acquire 持续失败检测（非 WAIT_TIMEOUT/NA）：AMD 890M 实测会话会
        # 进入永久失败态（只重建能救）。_err_t0 = 连续错误起点。
        # DDA 帧所有权：拷贝发出后先攥着，帧末再 ReleaseFrame（此时拷贝
        # 早已落地，既满足 DDA 契约又不花 CPU 等待；见 _release_held）。
        self._frame_held = False
        self._held_seq = 0
        self._held_t = 0.0
        self._err_t0 = 0.0
        self._err_reopen_n = 0
        self._err_reopen_t = 0.0
        self._last_err_hr = 0
        # 延迟仪表（QPC）：frame_qpc = 桌面最后一次 present（DDA 提供），
        # acq_qpc = 本次 acquire 返回时刻；两者差 = 采集到达延迟
        self.frame_qpc = 0
        self.acq_qpc = 0
        # 上一轮 poll 的结果——主循环要区分"会话活着、只是没有新帧"和
        # "会话坏了"（后者绝不能沿用环里的旧帧）：
        #   ok     取到新帧并拷进环（返回槽号）
        #   wait   会话活着但没有新帧（WAIT_TIMEOUT）
        #   acc0   只有鼠标类更新，桌面图像没变
        #   na     暂时不可用   error 会话错误（ACCESS_LOST 等）
        #   nodup  会话不存在   held  上一帧拷贝未落地
        #   noslot 取到帧但没有空闲环槽（帧被丢弃）
        self.poll_state = "init"
        # 卡点看门狗：记录当前阶段与进入时刻，独立线程每秒检查一次
        self._phase = "init"
        self._phase_t = time.perf_counter()
        self._wd_started = False
        # 会话失效检测（纯日志，不自动处置）：_poll_t 证明消费端活跃
        self._poll_t = 0.0        # 最近一次 poll() 调用
        self._last_alive_t = time.perf_counter()   # 最近一次 hr==0（会话活着）

    def _set_phase(self, name):
        self._phase = name
        self._phase_t = time.perf_counter()

    def _watchdog(self):
        last_logged = 0.0
        _reason_logged = 0.0
        while not self._stop.is_set():
            time.sleep(0.5)
            _now = time.perf_counter()
            _held = _now - self._phase_t
            # 只有 poll 还在被消费（3D 渲染中）才报"卡"——2D 态（转换关闭）
            # 渲染循环不调 poll，phase 停在最后一次的值上，此前每秒误打一行
            # "卡在 idle"（发布包日志 786 行全是这个：暂停噪音被当成故障）
            _consuming = bool(self._poll_t and _now - self._poll_t < 1.0)
            if _consuming and _held > 1.0 and _now - last_logged > 1.0:
                last_logged = _now
                print(f"[zcc] 卡在 {self._phase} 已 {_held:.1f}s", flush=True)
            # 会话失效检测（纯检测不自动重建——40fa771 的自动重建建立在
            # 被推翻的"被抢"理论上，真实触发从未出现）：poll 活跃（3D 渲染
            # 中）但持续 >30s 拿不到任何成功 acquire（WAIT_TIMEOUT 也不算
            # 活）且设备 S_OK = 异常态，尖叫日志留证；处置留给人工/后续。
            if (_consuming and _now - self._last_alive_t > 30.0
                    and _now - getattr(self, "_stolen_log_t", 0.0) > 60.0):
                self._stolen_log_t = _now
                print(f"[zcc] 异常：3D 消费中但 {_now - self._last_alive_t:.0f}s "
                      f"无帧（设备若 S_OK 且桌面在动=会话失效，需查）", flush=True)
                self._last_alive_t = _now   # 退避
            # 卡住超过 2s：查设备是否被移除（槽 39 = GetDeviceRemovedReason）
            if _consuming and _held > 2.0 and _now - _reason_logged > 2.0 and self._dev is not None:
                _reason_logged = _now
                try:
                    hr = self._dev.call(39, ctypes.c_long, ())
                    _hr = hr & 0xFFFFFFFF
                    _txt = {0: "S_OK", 0x887A0005: "DEVICE_REMOVED",
                            0x887A0006: "DEVICE_HUNG",
                            0x887A0007: "DEVICE_RESET"}.get(_hr, "?")
                    print(f"[zcc] 设备状态 GetDeviceRemovedReason = {_hr:#x} ({_txt})",
                          flush=True)
                except Exception as e:
                    print(f"[zcc] 查设备状态异常: {type(e).__name__}: {e}",
                          flush=True)

    # ------------------------------------------------------------------ 采集线程
    def start(self):
        threading.Thread(target=self._bootstrap, daemon=True,
                         name="dxgi-gpu").start()
        if not self._wd_started:
            self._wd_started = True
            threading.Thread(target=self._watchdog, daemon=True,
                             name="dxgi-wd").start()

    def stop(self):
        self._stop.set()

    @property
    def size(self):
        return self._w, self._h

    def output_size_now(self):
        """DDA 目标输出当前尺寸（分辨率/显示器切换后与环尺寸比对用）。"""
        try:
            return _output_size(self._dev, self.monitor)
        except Exception:
            return self._w, self._h

    def close(self):
        """显式释放 DDA 会话/环/设备（ComPtr 不随 GC 释放）。

        采集环重建/退出前必须调：旧会话攥着输出会让新 DuplicateOutput
        被拒（0x887a0022），表现为永久无帧。
        """
        self._stop.set()
        try:
            if self._dup is not None:
                self._dup.release()
        except Exception:
            pass
        self._dup = None
        for _t in (self._ring or []):
            try:
                _t.release()
            except Exception:
                pass
        self._ring = []
        for _k in (self.bridge_kms or []):
            try:
                _k.release()
            except Exception:
                pass
        self.bridge_kms = None
        try:
            self._ctx.release()
        except Exception:
            pass
        try:
            self._dev.release()
        except Exception:
            pass

    def ring_ptrs(self):
        return [t.p for t in self._ring]

    def build_inline(self, gl_ctx=None):
        """在调用线程（渲染线程）内联建设备/duplication/环；可选立即注册。

        跨线程版本在 app 里会让 AcquireNextFrame 永久阻塞（看门狗实证），
        而独立进程里同代码正常 —— 改为全部在渲染线程完成，去掉跨线程交互。
        """
        self._set_phase("inline-device")
        adv_ptr = None
        if getattr(self, "adapter", None) is not None:
            adv_ptr = getattr(self.adapter, "ptr", None)
            self.adapter_name = getattr(self.adapter, "name", None)
        else:
            try:
                from . import placement as _pl
                _adv = _pl.pick_adapter()
                self.adapter = _adv
                adv_ptr = getattr(_adv, "ptr", None)
                self.adapter_name = getattr(_adv, "name", None)
            except Exception:
                adv_ptr = None
        self._dev, self._ctx, _ = create_device(adv_ptr)
        self._set_phase("inline-size")
        self._w, self._h = _output_size(self._dev, self.monitor)
        self._set_phase("inline-dup")
        for _ in range(60):
            try:
                if self._open_dup():
                    break
            except Exception as e:
                print(f"[zcc] 建 duplication 异常: {type(e).__name__}: {e}",
                      flush=True)
            time.sleep(0.05)
        if self._dup is None:
            raise RuntimeError("duplication 建立失败")
        self._set_phase("inline-ring")
        _misc = d3d_ring_misc()
        self._ring = [
            create_tex(self._dev, TexDesc(self._w, self._h, 1, 1,
                                          DXGI_FORMAT_B8G8R8A8_UNORM,
                                          1, 0, 0, 8, 0, _misc))
            for _ in range(RING)
        ]
        if _misc:
            self._setup_bridge_kms()
        self.ready.set()
        if gl_ctx is not None:
            self.register_gl(gl_ctx)
        self._set_phase("idle")
        self.inline = True
        if not self._wd_started:
            self._wd_started = True
            threading.Thread(target=self._watchdog, daemon=True,
                             name="dxgi-wd").start()
        print(f"[zcc] 内联采集就绪 {self._w}x{self._h}", flush=True)

    def _probe_gl(self, idx):
        """D2S_ZC_PROBE：把环纹理经 GL 采样到 32×32 读回，验证互操作视图。"""
        fbo = getattr(self, "_probe_fbo", None)
        if fbo is None:
            return
        import numpy as _np3
        gltex = self.ring_gl[idx]
        self.dx.lock(gltex)
        try:
            gltex.use(0)
            self._probe_prog['t'].value = 0
            fbo.use()
            fbo.viewport = (0, 0, 32, 32)
            self._probe_vao.render(3)      # TRIANGLE_STRIP = 3 in moderngl
            raw = fbo.read(components=3, alignment=1)
        finally:
            self.dx.unlock(gltex)
        mean = float(_np3.frombuffer(raw, _np3.uint8).mean())
        if mean < 1.0:
            self._probe_gl_black = getattr(self, "_probe_gl_black", 0) + 1
            if self._probe_gl_black <= 10:
                print(f"[probe] GL 采样全黑 slot={idx} mean={mean:.2f}",
                      flush=True)
        else:
            self._probe_gl_ok = getattr(self, "_probe_gl_ok", 0) + 1

    def _probe_stage(self, idx):
        """D2S_ZC_PROBE：把环纹理拷到 staging 并 Map，检查拷贝结果是否全黑。"""
        if getattr(self, "_stg", None) is None:
            self._stg = create_tex(
                self._dev, TexDesc(self._w, self._h, 1, 1,
                                   DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0,
                                   3, 0, 0x20000, 0))     # STAGING + CPU_READ
            self._mapped = Mapped()
        stage = self._stg
        ctx_copy_resource(self._ctx, stage, self._ring[idx])
        m = self._mapped
        hr = self._ctx.call(14, ctypes.c_long,
                            (ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint, ctypes.POINTER(Mapped)),
                            stage.p, 0, 1, 0, ctypes.byref(m))
        if hr:
            return
        rows = max(1, self._h // 8)
        buf = ctypes.string_at(m.pData, m.RowPitch * rows)
        mean = sum(buf) / max(1, len(buf))
        try:
            self._ctx.call(15, None, (ctypes.c_void_p, ctypes.c_uint),
                           stage.p, 0)
        except Exception:
            pass
        if mean < 1.0:
            self._probe_black = getattr(self, "_probe_black", 0) + 1
            if self._probe_black <= 10:
                print(f"[probe] D3D 拷贝结果全黑 slot={idx} mean={mean:.2f}",
                      flush=True)
        else:
            self._probe_ok = getattr(self, "_probe_ok", 0) + 1

    def _reopen_dup(self, tries=12, base_delay=0.02):
        """重建会话。tries/base_delay 控制重试强度：渲染线程（poll）里的
        恢复路径用 tries=1（单次、不阻塞）；启动/后台线程用默认 12 次带
        退避（最多 ~2s）。"""
        now = time.perf_counter()
        # 1.0s 冷却会让连续丢失期间长时间无帧（画面停住/跳变 = 用户看到的闪）。
        # 缩短到 50ms：重建本身只需 ~10ms，没必要等一整秒。
        if now - getattr(self, "_dup_lost_t", 0.0) < 0.05:
            return False
        self._dup_lost_t = now
        # 旧会话的帧所有权随 _dup 一起作废（新会话没有"待释放帧"）
        self._frame_held = False
        _t_rebuild0 = now
        print("[zcc] duplication 丢失（ACCESS_LOST），重建…", flush=True)
        try:
            self._dup.release()
        except Exception:
            pass
        self._dup = None
        self._set_phase("reopen-dup")
        backoff = max(0.0, float(base_delay))
        for _ in range(max(1, int(tries))):
            try:
                if self._open_dup():
                    print(f"[zcc] duplication 已重建（耗时 "
                          f"{(time.perf_counter() - _t_rebuild0) * 1000:.0f}ms）",
                          flush=True)
                    self._set_phase("idle")
                    return True
            except Exception:
                pass
            if backoff > 0:
                time.sleep(backoff)
                backoff = min(0.3, backoff * 1.6)
        print("[zcc] duplication 重建失败（下次再试）", flush=True)
        self._set_phase("idle")
        return False

    def _release_held(self):
        """帧末释放：拷贝已落地才 ReleaseFrame（不等待、不阻塞）。

        立即释放（96898cc）会让 DXGI 把同一桌面表面交给下一帧，DWM 的写入
        与我们未完成的拷贝并发 → 采到"上半旧帧/下半新帧"的撕裂。帧末统一
        释放时拷贝已过去数毫秒，fence 必然完成，所以既守契约又不吃 0.2ms。
        返回 True=已释放或本就无帧；False=拷贝仍未完成（下一轮 poll 再试）。
        """
        if not self._frame_held:
            return True
        _done = True
        if self.bridge_fence is not None:
            try:
                _done = (int(self.bridge_fence.call(
                    8, ctypes.c_uint64, ())) >= self._held_seq)
            except Exception:
                _done = True
        if not _done and (time.perf_counter() - self._held_t) < 0.1:
            # 帧末都还没落地：说明拷贝被 GPU 队列拖住了（疑点日志，限频）
            _n = getattr(self, "_late_n", 0) + 1
            self._late_n = _n
            _t = time.perf_counter()
            self._hold_blk = getattr(self, "_hold_blk", 0) + 1
            self._hold_max = max(getattr(self, "_hold_max", 0.0),
                                 (_t - self._held_t) * 1000.0)
            if _n <= 3 or _t - getattr(self, "_late_t", 0.0) >= 5.0:
                self._late_t = _t
                print(f"[zcc] 拷贝未落地 ({(_t - self._held_t) * 1000:.1f}ms,"
                      f" 累计 {_n})", flush=True)
            return False
        _h = (time.perf_counter() - self._held_t) * 1000.0
        self._hold_log.append(_h)
        self._hold_max = max(getattr(self, "_hold_max", 0.0), _h)
        self._frame_held = False
        return self._release_frame()

    def _release_frame(self):
        """ReleaseFrame；失败 = 会话已坏（之后 Acquire 会永久 INVALID_CALL，
        AMD 实测），必须重建，否则画面永久冻结。返回 True=释放成功。"""
        if self._dup is None:
            return False
        try:
            _rel = self._dup.call(14, ctypes.c_long, ())
        except Exception:
            _rel = -1
        if _rel:
            print(f"[zcc] ReleaseFrame 失败 hr={_rel & 0xFFFFFFFF:#x}"
                  f" → 重建 duplication", flush=True)
            self.poll_state = "error"     # 会话已坏：主循环不得沿用旧帧
            self._reopen_dup(tries=1)      # 单次尝试，失败由 poll 顶部节流重试
            return False
        return True

    def poll(self):
        """渲染线程每帧调用：取一帧并拷进空闲环槽，返回槽号（无帧返回 -1）。
        这一层只做计时（取帧节奏诊断），行为全在 _poll_impl 里。"""
        _t_in = time.perf_counter()
        _g = 0.0
        if self._prev_poll_end:
            _g = (_t_in - self._prev_poll_end) * 1000.0
            if _g > self._pgap_max:
                self._pgap_max = _g
            for _i, _th in enumerate((8.0, 16.0, 33.0)):
                if _g >= _th:
                    self._pgap_n[_i] += 1
        _ok = self._poll_impl()
        self._prev_poll_end = time.perf_counter()
        if _ok >= 0:
            self._acq_log.append(
                (_g, (_t_in - self._prev_ok_t) * 1000.0
                 if self._prev_ok_t else 0.0, self._acq_lat_ms,
                 self._acc_frames))
            self._prev_ok_t = _t_in
        return _ok

    def _poll_impl(self):
        """渲染线程每帧调用：取一帧并拷进空闲环槽，返回槽号（无帧返回 -1）。"""
        self._poll_t = time.perf_counter()
        if self._dup is None:
            # 重建失败后（典型：显示器模式切换中，DuplicateOutput 返回
            # DXGI_ERROR_MODE_CHANGE_IN_PROGRESS）旧代码会直接在这里
            # AttributeError 闪退。改为节流重试：每 250ms 试建一次会话。
            _now_r = time.perf_counter()
            if _now_r - getattr(self, "_dup_retry_t", 0.0) >= 0.25:
                self._dup_retry_t = _now_r
                try:
                    if self._open_dup():
                        print("[zcc] duplication 已重建", flush=True)
                        self._set_phase("idle")
                except Exception:
                    pass
            self.poll_state = "nodup"
            return -1
        if self._frame_held and not self._release_held():
            self.poll_state = "held"
            return -1            # 拷贝仍未落地（异常态）：本轮先不取新帧
        t_pre = time.perf_counter()
        self._set_phase("acquire")
        # 非阻塞（timeout=0）：DDA 原生 AcquireNextFrame 只要 timeout>0
        # 就额外背 ~15ms（本仓库 capture.py 实测：0→0.01ms、2/6→15.5ms、
        # 16→30.8ms）。内联消费端每帧 poll 一次，8ms 会把循环压到
        # ~7.5+15ms/帧 = 30fps（内容端 60fps 也吃不满）。无帧时返回 -1，
        # 调用方 sleep 0.5ms 后再来（见 main 的 D3D 循环）。
        hr, info, res_pp = self._acquire(0)
        self._set_phase("idle")
        if hr == 0:
            self._last_alive_t = t_pre   # acquire 成功（含 acc0）= 会话活着
        st = self._pst
        st["n"] += 1
        _hr = hr & 0xFFFFFFFF
        if hr:
            if _hr == DXGI_ERROR_WAIT_TIMEOUT:
                st["wait"] += 1
                self._err_t0 = 0.0
                self.poll_state = "wait"
            elif _hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE:
                st["na"] += 1
                self._err_t0 = 0.0
                self.poll_state = "na"
            else:
                st["other"] += 1
                self.poll_state = "error"
                if _hr != self._last_err_hr:
                    self._last_err_hr = _hr
                    print(f"[zcc] acquire 错误码 = {_hr:#x} "
                          f"(ACCESS_LOST={DXGI_ERROR_ACCESS_LOST:#x})",
                          flush=True)
                self._err_t0 = self._err_t0 or time.perf_counter()
                # 非超时错误持续 >2s：DDA 会话已坏（ACCESS_LOST 之外也有
                # 永久失败态，实测 AMD）——退避重建，否则画面永远冻结。
                if time.perf_counter() - self._err_t0 > 2.0:
                    self._err_t0 = 0.0
                    self._err_reopen_n += 1
                    _bo = min(8.0, 2.0 ** self._err_reopen_n)
                    if time.perf_counter() - self._err_reopen_t >= _bo:
                        self._err_reopen_t = time.perf_counter()
                        print(f"[zcc] acquire 持续失败 hr={_hr:#x}"
                              f"（other={st['other']}）→ 重建 duplication",
                              flush=True)
                        self._reopen_dup(tries=1)   # 单次不阻塞，失败由 poll 顶部重试
        elif info.AccumulatedFrames:
            st["ok"] += 1
            self._err_t0 = 0.0
            self._err_reopen_n = 0
            self.poll_state = "ok"
            if info.AccumulatedFrames >= 2:
                st["multi"] += 1          # 一次拿到 2+ 帧 = 消费间隔 > 一个刷新
            _nowa = time.perf_counter()
            if self._last_acq_t:
                self._acq_gap_max = max(self._acq_gap_max,
                                        _nowa - self._last_acq_t)
            self._last_acq_t = _nowa
            st["accsum"] += info.AccumulatedFrames   # 真实内容变化次数（可合并）
            # 累计（诊断）：DDA 自报的桌面真实变化次数，跨 poll60 不丢
            self._af_total = getattr(self, "_af_total", 0) + \
                int(info.AccumulatedFrames)
        else:
            st["acc0"] += 1
            self._err_t0 = 0.0
            self._err_reopen_n = 0
            self.poll_state = "acc0"
        _nowp = time.perf_counter()
        if _nowp - self._pst_t >= 30.0:
            # DDA 到达率常驻暴露（30s 窗）。accsum=窗口内 AccumulatedFrames
            # 总和=桌面真实变化次数；ok<accsum 说明消费慢于内容（合并发生）
            # ——"内容慢" vs "消费慢"一眼分辨。
            _npolls = st["n"] - self._pst_n
            self._pst_t = _nowp
            self._pst_n = st["n"]
            print(f"[zcc] poll30s: dda={st['ok'] + st['acc0']}/{_npolls} "
                  f"(ok={st['ok']} accsum={st['accsum']} acc0={st['acc0']} "
                  f"wait={st['wait']} na={st['na']} other={st['other']} "
                  f"multi={st['multi']} gapmax={self._acq_gap_max * 1000:.1f}ms "
                  f"holdmax={getattr(self, '_hold_max', 0.0):.1f}ms "
                  f"holdblk={getattr(self, '_hold_blk', 0)} "
                  f"hr={self._last_err_hr:#x})", flush=True)
            print("[zcc] " + self._acq_stats(), flush=True)
            self._acq_gap_max = 0.0
            self._hold_max = 0.0
            self._hold_blk = 0
            if os.environ.get("D2S_ZC_DEBUG"):
                print(f"[zcc] poll {st['n']} 次: ok={st['ok']} acc0={st['acc0']} "
                      f"wait={st['wait']} na={st['na']} other={st['other']} "
                      f"(last hr={_hr:#x})", flush=True)
            for _k in ("ok", "accsum", "acc0", "wait", "na", "other",
                       "multi"):
                st[_k] = 0
        if hr or not res_pp.value:
            # 会话丢失（全屏切换/视频进出全屏必现）→ 重建；否则永远拿不到帧。
            # 单次尝试：渲染线程不阻塞，失败后由 poll 顶部 250ms 节流重试。
            if _hr == DXGI_ERROR_ACCESS_LOST:
                self._reopen_dup(tries=1)
            return -1
        idx = -1
        try:
            if info.AccumulatedFrames:
                src = ComPtr(ComPtr(res_pp.value).qi(_IID_TEX2D))
                try:
                    idx = self._free_slot()
                    if idx >= 0:
                        self.writing[idx] = True
                        try:
                            self._set_phase("copy")
                            # 桥模式（环带 NTHANDLE|KM）：拷贝后要发信号。
                            # 锁在建环时已对每个槽取过并一直持有，这里不碰
                            bridge_on = bool(self.bridge_kms)
                            # 槽回收：等 GPU 用完这个槽的上一次显示/重呈现
                            # （D3D12 消费者 signal 的共享 fence）再覆盖
                            if (self.bridge_fence_cons is not None
                                    and self.slot_done[idx] > 0):
                                self.bridge_ctx4.call(
                                    S_CTX4_WAIT, ctypes.c_long,
                                    (ctypes.c_void_p, ctypes.c_uint64),
                                    self.bridge_fence_cons.p,
                                    self.slot_done[idx])
                            ctx_copy_resource(self._ctx, self._ring[idx], src)
                            if bridge_on:
                                self.bridge_signal_after_copy(idx)
                                # 先攥着这一帧：帧末 _release_held() 在拷贝
                                # 早已落地后 ReleaseFrame（DDA 契约），既不
                                # 花 CPU 等待，也不跳过任何一轮 acquire。
                                self._frame_held = True
                                self._held_seq = self.bridge_seq
                                self._held_t = time.perf_counter()
                            self._set_phase("idle")
                            if os.environ.get("D2S_ZC_FINISH"):
                                # 诊断：D3D 拷贝后强制 GPU 同步，验证是否为
                                # "拷贝尚未可见时就采样"的同步问题
                                try:
                                    import OpenGL.GL as _glf
                                    _glf.glFinish()
                                except Exception:
                                    pass
                        finally:
                            self.writing[idx] = False
                        if os.environ.get("D2S_ZC_PROBE"):
                            try:
                                if idx >= 0:
                                    self._probe_gl(idx)
                            except Exception as e:
                                print(f"[probe] gl err: {type(e).__name__}: {e}",
                                      flush=True)
                            try:
                                self._probe_stage(idx)
                            except Exception as e:
                                print(f"[probe] stage err: {type(e).__name__}: {e}",
                                      flush=True)
                finally:
                    src.release()
        finally:
            if not self._frame_held:
                self._release_frame()
        if idx >= 0:
            try:
                self.frame_qpc = (int(info.LastPresentTime)
                                  if info.LastPresentTime else 0)
                self.acq_qpc = qpc()
            except Exception:
                self.frame_qpc = 0
            self._last_pub = (self.gen, idx, t_pre, time.perf_counter(),
                              (self._w, self._h), None, self.tag)
            self.slot_seq[idx] = self.bridge_seq
            # 取帧节奏诊断：桌面这帧的呈现时刻（DDA 给的 QPC）→ 我们取到的延迟
            try:
                self._acq_lat_ms = ((qpc() - int(info.LastPresentTime))
                                    * 1000.0 / self._qpf
                                    if info.LastPresentTime else 0.0)
            except Exception:
                self._acq_lat_ms = 0.0
            self._acc_frames = int(info.AccumulatedFrames)
        elif self.poll_state == "ok":
            self.poll_state = "noslot"
        return idx

    def _free_slot(self):
        """挑一个可以被新帧覆盖的环槽；没有返回 -1。

        绝不覆盖两个槽：正在推理/刚提交的那一帧（avoid），和上一对已经
        上屏、还会被重呈现反复采样的那一帧（avoid2）。环是 3 槽，所以
        永远给采集留着一个槽放最新帧。少了任何一个保护，GPU 被拖慢时
        采集就会把"更新的画面"写进正在被显示的槽：画面先闪到未来、再随
        下一对跳回去（用户看到的"疯狂回退"，2026-11-27 全屏游戏实测）。
        """
        for _ in range(RING):
            cand = self._ring_i
            self._ring_i = (self._ring_i + 1) % RING
            if (not self.gl_locked[cand] and cand != self.avoid
                    and cand != self.avoid2):
                return cand
        return -1

    def frame_unchanged(self):
        """上一轮 poll 没给帧，而且原因是"会话活着、只是没有新帧"
        （wait / acc0）——只有这时环里的上一帧才仍然是当前桌面，可以沿用。
        会话错误、重建中、拷贝未落地都不算：那一帧可能早已过时。"""
        return self.poll_state in ("wait", "acc0")

    @property
    def copy_pending(self):
        """刚采到的那一帧还在往环槽里拷（GPU 上没落地）。"""
        return self._frame_held

    def _acq_stats(self):
        """取帧节奏一行（本窗），打完清零。用来分清"取帧率低"是主循环没来取、
        还是来了但 DDA 迟迟不给、还是取到了在等拷贝落地。"""
        import numpy as _np
        log, hold = self._acq_log, self._hold_log
        pg, pgm = self._pgap_n, self._pgap_max
        self._acq_log, self._hold_log = [], []
        self._pgap_n, self._pgap_max = [0, 0, 0], 0.0
        if not log:
            return "取帧节奏: 本窗没取到帧"
        a = _np.array(log, _np.float64)
        h = _np.array(hold, _np.float64) if hold else _np.zeros(1)

        def p(x, q):
            return float(_np.percentile(x, q)) if x.size else float("nan")

        one = a[a[:, 3] == 1]
        many = a[a[:, 3] >= 2]
        return ("取帧节奏: 轮询间隔 max=%.1fms（≥8ms %d 次, ≥16ms %d, ≥33ms %d）| "
                "取帧间隔 p50=%.1f p90=%.1f max=%.1f ms | "
                "桌面呈现→取到 p50=%.1f p90=%.1f max=%.1f ms | "
                "单帧 n=%d 取到延迟 p50=%.1f | 合并帧 n=%d 取到延迟 p50=%.1f "
                "其前一轮询间隔 p50=%.1f | 攥帧 p50=%.1f max=%.1f ms（%d 次）"
                % (pgm, pg[0], pg[1], pg[2],
                   p(a[:, 1], 50), p(a[:, 1], 90), float(a[:, 1].max()),
                   p(a[:, 2], 50), p(a[:, 2], 90), float(a[:, 2].max()),
                   int(one.shape[0]), p(one[:, 2], 50),
                   int(many.shape[0]), p(many[:, 2], 50), p(many[:, 0], 50),
                   p(h, 50), float(h.max()), int(h.size)))

    def _open_dup(self):
        # 重试期内（模式切换中）每 250ms 会调一次：失败细节每秒最多打一行
        _now_o = time.perf_counter()
        _verbose = _now_o - getattr(self, "_open_log_t", 0.0) >= 1.0
        if _verbose:
            self._open_log_t = _now_o
        dxdev = ComPtr(self._dev.qi(_dd.IID_IDXGIDevice))
        adapter_pp = ctypes.c_void_p()
        dxdev.call(7, ctypes.c_long, (ctypes.POINTER(ctypes.c_void_p),),
                   ctypes.byref(adapter_pp))
        adapter = ComPtr(adapter_pp.value)
        # 只采集配置的输出：跨输出兜底会把会话静默开到另一块屏上——
        # 双屏实测全黑根因（目标 #0 瞬时 E_INVALIDARG → 滑到 #1=3D 输出屏，
        # DDA 只见 WDA 排除后的黑桌面 = 全黑 + 十秒一帧）。E_INVALIDARG 是
        # 过渡态瞬时错误，失败交给调用方的重试循环（同输出退避），绝不换屏。
        for _k in [self.monitor]:
            out_pp = ctypes.c_void_p()
            hr = adapter.call(7, ctypes.c_long,
                              (wt.UINT, ctypes.POINTER(ctypes.c_void_p)),
                              _k, ctypes.byref(out_pp))
            if hr or not out_pp.value:
                continue
            out1 = ComPtr(ComPtr(out_pp.value).qi(_dd.IID_IDXGIOutput1))
            if getattr(self, "_dup_of_logged", None) != _k:
                self._dup_of_logged = _k
                print(f"[zcc] duplication 目标输出 #{_k}（监控索引 "
                      f"{self.monitor}；双屏若采错屏=在此定位）", flush=True)
            # 优先 DuplicateOutput1（IDXGIOutput5）：视频走硬件叠加平面(MPO)
            # 或 HDR 时，老 DuplicateOutput 会持续报 NOT_CURRENTLY_AVAILABLE。
            try:
                out5 = ComPtr(ComPtr(out_pp.value).qi(_IID_OUTPUT5))
                dup_pp = ctypes.c_void_p()
                # 只传 BGRA8：环纹理是 BGRA8，若驱动真返回 FP16 帧，
                # CopyResource 会格式不匹配 → DXGI_ERROR_DEVICE_REMOVED
                # （实测设备被移除、采集永久失效）。FP16 支持需要环纹理
                # 同步改成 RGBA16F，属于后续独立改动。
                fmts = (ctypes.c_uint * 1)(DXGI_FORMAT_B8G8R8A8_UNORM)
                hr5 = out5.call(26, ctypes.c_long,
                                (ctypes.c_void_p, wt.UINT, wt.UINT,
                                 ctypes.POINTER(ctypes.c_uint),
                                 ctypes.POINTER(ctypes.c_void_p)),
                                self._dev.p, 0, 1, fmts, ctypes.byref(dup_pp))
                if not hr5 and dup_pp.value:
                    self._dup = ComPtr(dup_pp.value)
                    print("[zcc] 使用 DuplicateOutput1", flush=True)
                    return True
                if _verbose:
                    print(f"[zcc] DuplicateOutput1 不可用 "
                          f"hr={hr5 & 0xFFFFFFFF:#x}，回落 DuplicateOutput",
                          flush=True)
            except Exception as e:
                print(f"[zcc] DuplicateOutput1 异常: {type(e).__name__}: {e}",
                      flush=True)
            dup_pp = ctypes.c_void_p()
            hr = out1.call(22, ctypes.c_long,
                           (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                           self._dev.p, ctypes.byref(dup_pp))
            if not hr and dup_pp.value:
                self._dup = ComPtr(dup_pp.value)
                return True
            self._last_open_hr = hr & 0xFFFFFFFF   # 诊断：最后一次被拒的 hr
        if getattr(self, "_last_open_hr", 0) and _verbose:
            # 常见两类：0x887a0025 模式切换中（显示器/分辨率/刷新率在变，
            # 稍等即可）；0x887a0022 被其它 duplication 会话占用（dev/发布包
            # 残留实例互踩）。两类都应继续重试。
            print(f"[zcc] DuplicateOutput 被拒 hr={self._last_open_hr:#x}"
                  f"（0x887a0025=显示器模式切换中，0x887a0022=被其它实例"
                  f"占用），稍后重试", flush=True)
        return False

    def _bootstrap(self):
        try:
            adv_ptr = None
            try:
                from . import placement as _pl
                adv_ptr = getattr(_pl.pick_adapter(), "ptr", None)
            except Exception:
                adv_ptr = None
            self._set_phase("bootstrap-device")
            self._dev, self._ctx, _ = create_device(adv_ptr)
            self._set_phase("bootstrap-size")
            self._w, self._h = _output_size(self._dev, self.monitor)
            for _ in range(50):
                if self._stop.is_set():
                    return
                try:
                    if self._open_dup():
                        break
                except Exception as e:
                    print(f"[zcc] 建 duplication 异常: {type(e).__name__}: {e}",
                          flush=True)
                time.sleep(0.05)
            if self._dup is None:
                print("[zcc] 建 duplication 失败，颜色零拷贝停用", flush=True)
                return
            self._set_phase("bootstrap-ring")
            _misc = d3d_ring_misc()
            self._ring = [
                create_tex(self._dev, TexDesc(self._w, self._h, 1, 1,
                                              DXGI_FORMAT_B8G8R8A8_UNORM,
                                              1, 0, 0, 8, 0, _misc))
                for _ in range(RING)
            ]
            if _misc:
                self._setup_bridge_kms()
            self._set_phase("wait-register")
            print(f"[zcc] 采集线程就绪 {self._w}x{self._h}（设备/duplication/环）",
                  flush=True)
            self.ready.set()
        except Exception as e:
            import traceback
            print(f"[zcc] 采集线程初始化失败: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return
        # 等渲染线程完成 GL 注册（注册期间不能有拷贝）
        while not self._stop.is_set() and not self.registered.wait(0.05):
            pass
        if self._stop.is_set():
            return
        self._loop()

    # ------------------------------------------------------------------ D3D12 桥（T1）
    def attach_bridge_fence(self, fence11, ctx4, fence_cons=None):
        """注入 producer/consumer fence（gpu/dda_bridge.DdaBridge 建）——d3d 模式。

        fence_cons：反向（消费者→生产者）fence。采集写槽前 Wait 该槽最后
        的消费者 signal 值，避免覆盖还在被 GPU 采样的环纹理（一帧闪/撕裂）。

        ctx4 必须是【本采集设备 immediate context】的 ID3D11DeviceContext4
        （桥要用 dev11=self._dev, ctx11=self._ctx 建）：拷贝后的 Signal 和写槽前
        的 Wait 都发在它上面，而环纹理的拷贝在 self._ctx 上执行。不是同一个
        上下文（= 另一个设备）时两者之间没有先后保证——渲染队列等到信号时
        拷贝可能还没执行，显示读到环槽里上一轮的画面（帧号在涨、画面往回跳，
        2026-11-28）。这里直接报错，不让错误的接线带着竞态跑起来。
        """
        if ctx4 is not None and getattr(self, "_ctx", None) is not None:
            from .dda_bridge import CTX4_IID
            own = ctypes.c_void_p(self._ctx.qi(CTX4_IID))
            try:
                same = bool(ctx4.p and ctx4.p.value == own.value)
            finally:
                if own.value:
                    ComPtr(own.value).release()
            if not same:
                raise RuntimeError(
                    "[bridge] ctx4 不是采集设备的上下文：桥必须用 "
                    "DdaBridge(ren, dev11=<采集设备>, ctx11=<采集上下文>) 建，"
                    "否则'拷贝完成'信号与拷贝不在同一设备、没有先后保证")
        self.bridge_fence = fence11
        self.bridge_ctx4 = ctx4
        self.bridge_fence_cons = fence_cons

    def _setup_bridge_kms(self):
        """环纹理的每-slot IDXGIKeyedMutex（NTHANDLE|KM 模式）：各取锁一次，
        一直持有到环销毁。

        带锁共享的纹理没取锁就写，那次拷贝被静默丢掉（实测每个槽的第一次
        全丢：进 3D 前 3 帧全黑）。生产里其实一直是"已持有"的状态——只是靠
        每帧拷贝后那次（写错了槽位的）AcquireSync 在第一次使用时取上。这里
        改成建环时显式取锁，每帧不再碰它。先后保证来自桥的 fence（拷贝后
        Signal / 写槽前 Wait）；D3D12 那头从不碰这个锁。
        """
        self.bridge_kms = []
        for i, t in enumerate(self._ring):
            km = ComPtr(ctypes.c_void_p(t.qi(_IID_KEYEDMUTEX)).value)
            # 超时 100ms：正常立刻返回（新建的锁没人持有）；真等不到就留日志
            hr = km.call(S_KEYED_ACQUIRE, ctypes.c_long,
                         (ctypes.c_uint64, ctypes.c_uint), 0, 100)
            if hr:
                print(f"[zcc] 槽{i} 取锁失败 hr={hr & 0xFFFFFFFF:#x}"
                      f"（该槽的拷贝会被丢）", flush=True)
            self.bridge_kms.append(km)
        print(f"[zcc] 桥模式：环 NTHANDLE|KM（{RING} 槽）", flush=True)

    def bridge_signal_after_copy(self, idx):
        """拷贝后桥协议尾段：ctx4.Signal(fence, seq) + Flush。

        Flush 必须有：D3D11 immediate ctx 命令是延迟提交的，不 Flush 则
        Signal 不达 GPU，消费端首帧 queue_wait 会读到未写入的环（实测
        首帧恒黑、其余正常——后续帧命令攒队顺带 flush 掩盖了问题）。
        """
        if self.bridge_fence is None:
            return None
        self.bridge_seq += 1
        self.bridge_ctx4.call(S_CTX4_SIGNAL, ctypes.c_long,
                              (ctypes.c_void_p, ctypes.c_uint64),
                              self.bridge_fence.p, self.bridge_seq)
        # ID3D11DeviceContext::Flush = 槽 111（SDK 头文件数法：
        # CopyResource=47 与 rtcom 生产值吻合校准；Flush 为接口内第 105 个
        # 虚方法 → 7 + 105 - 1 = 111）
        # D2S_BRIDGE_NO_FLUSH=1：诊断开关，测 Flush 的每帧代价
        if os.environ.get("D2S_BRIDGE_NO_FLUSH", "") not in (
                "1", "true", "True"):
            self._ctx.call(111, None, ())
        return self.bridge_seq

    # ------------------------------------------------------------------ 渲染线程
    def register_gl(self, gl_ctx):
        """渲染线程调用：把环纹理一次性注册进 GL（采集此时还没开跑）。

        DxInterop 延迟到此导入：D2S_RENDER=d3d 的进程走 DdaBridge 桥，
        完全不碰 GL（模块顶层不拉 opengl32）。
        """
        from .dx_interop import DxInterop
        self.dx = DxInterop(self._dev.p)
        self.ring_gl = []
        for t in self._ring:
            gltex = gl_ctx.texture((8, 8), 4, dtype='f1')
            gltex.repeat_x = False
            gltex.repeat_y = False
            gltex._glo_interop = True
            self.dx.register(gltex, t.p)
            self.ring_gl.append(gltex)
        self.registered.set()
        try:
            self._probe_tex = gl_ctx.texture((32, 32), 3, dtype='f1')
            self._probe_fbo = gl_ctx.framebuffer([self._probe_tex])
            self._probe_prog = gl_ctx.program(
                vertex_shader="#version 330\nin vec2 p; out vec2 uv;\n"
                              "void main(){uv=p*0.5+0.5; gl_Position=vec4(p,0,1);}",
                fragment_shader="#version 330\nin vec2 uv; out vec4 c;\n"
                                "uniform sampler2D t;\n"
                                "void main(){ c = vec4(texture(t, uv).rgb, 1.0); }")
            import numpy as _np2
            _qv = _np2.array([-1, -1, 1, -1, -1, 1, 1, 1], _np2.float32)
            self._probe_vao = gl_ctx.vertex_array(
                self._probe_prog, [(gl_ctx.buffer(_qv), "2f", "p")])
        except Exception as _e:
            print(f"[zcc] 探针资源创建失败: {type(_e).__name__}: {_e}", flush=True)
            self._probe_fbo = None
        print(f"[zcc] GL 注册完成（{RING} 张环纹理）", flush=True)

    def lock(self, idx):
        self.dx.lock(self.ring_gl[idx])

    def unlock(self, idx):
        try:
            self.dx.unlock(self.ring_gl[idx])
        except Exception:
            pass

    # ------------------------------------------------------------------ 搬帧循环
    def _acquire(self, timeout_ms):
        info = DXGI_OUTDUPL_FRAME_INFO()
        res_pp = ctypes.c_void_p()
        hr = self._dup.call(8, ctypes.c_long,
                            (wt.UINT, ctypes.POINTER(DXGI_OUTDUPL_FRAME_INFO),
                             ctypes.POINTER(ctypes.c_void_p)),
                            timeout_ms, ctypes.byref(info), ctypes.byref(res_pp))
        return hr, info, res_pp

    def _loop(self):
        perf = time.perf_counter
        n = 0
        tries = 0
        hr_last = 0
        st = {"wait": 0, "na": 0, "other": 0, "ok": 0, "acc0": 0}
        print("[zcc] 搬帧循环开始", flush=True)
        _hb_t = perf()
        _hb_n = 0
        while not self._stop.is_set():
            t_pre = perf()
            _hb_n += 1
            if t_pre - _hb_t >= 1.0:
                print(f"[zcc] 采集心跳 {_hb_n}/s（发布累计 {n}）", flush=True)
                _hb_t, _hb_n = t_pre, 0
            try:
                # timeout=0 非阻塞：实测桌面静止时传 16ms 也会一直阻塞到有变化，
                # 主循环拿不到控制权（无法响应 stop/重建）。改为轮询 + 1ms 睡眠。
                self._set_phase("acquire")
                hr, info, res_pp = self._acquire(0)
                self._set_phase("idle")
            except Exception as e:
                print(f"[zcc] acquire 异常: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.005)
                continue
            if hr or not res_pp.value:
                tries += 1
                hr_last = hr & 0xFFFFFFFF
                if hr_last == DXGI_ERROR_WAIT_TIMEOUT:
                    st["wait"] += 1
                elif hr_last == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE:
                    st["na"] += 1
                else:
                    st["other"] += 1
                # 全屏切换/独占会让 duplication 会话失效（0x887A0026）——
                # 必须重建，否则永远拿不到帧（实测全屏 GL 窗口必现）
                if hr_last == DXGI_ERROR_ACCESS_LOST:
                    print("[zcc] duplication 丢失（全屏切换），重建…", flush=True)
                    try:
                        self._dup.release()
                    except Exception:
                        pass
                    self._dup = None
                    # 退避重试直到成功或停止（全屏切换期间可能连续失败）
                    backoff = 0.05
                    while not self._stop.is_set():
                        try:
                            if self._open_dup():
                                break
                        except Exception:
                            pass
                        time.sleep(backoff)
                        backoff = min(0.5, backoff * 1.5)
                    if self._dup is None:
                        return
                    print("[zcc] duplication 已重建", flush=True)
                    continue
                if tries % 300 == 0:
                    print(f"[zcc] acquire 统计 wait={st['wait']} na={st['na']} "
                          f"other={st['other']} ok={st['ok']} acc0={st['acc0']} "
                          f"last={hr_last:#x}", flush=True)
                    st = {"wait": 0, "na": 0, "other": 0, "ok": 0, "acc0": 0}
                time.sleep(0.001)
                continue
            idx = -1
            try:
                if info.AccumulatedFrames:
                    src = ComPtr(ComPtr(res_pp.value).qi(_IID_TEX2D))
                    try:
                        if not getattr(self, "_fmt_logged", False):
                            self._fmt_logged = True
                            _d = TexDesc()
                            src.call(10, None, (ctypes.POINTER(TexDesc),),
                                     ctypes.byref(_d))
                            print(f"[zcc] 帧纹理格式 = {_d.Format}"
                                  f"（87=BGRA8, 10=FP16）", flush=True)
                        idx = self._free_slot()
                        if idx >= 0:
                            self.writing[idx] = True
                            try:
                                self._set_phase("copy")
                                # 锁在建环时已取过并一直持有（_setup_bridge_kms）
                                ctx_copy_resource(self._ctx, self._ring[idx], src)
                                if self.bridge_kms:
                                    self.bridge_signal_after_copy(idx)
                                    self.slot_seq[idx] = self.bridge_seq
                                self._set_phase("idle")
                            finally:
                                self.writing[idx] = False
                    finally:
                        src.release()
            finally:
                try:
                    self._set_phase("release")
                    self._dup.call(14, ctypes.c_long, ())   # ReleaseFrame
                except Exception:
                    pass
            if idx < 0:
                if info.AccumulatedFrames:
                    self._dropped += 1
                else:
                    st["acc0"] += 1
                time.sleep(0.001)
                continue
            t_post = perf()
            st["ok"] += 1
            self.gpu_q.publish((self.gen, idx, t_pre, t_post,
                                (self._w, self._h), None, self.tag))
            n += 1
            _now_t = perf()
            if _now_t - getattr(self, "_pub_t", _now_t) >= 1.0:
                print(f"[zcc] 发布/秒 = {n - getattr(self, '_pub_n', 0)}"
                      f"（累计 {n}，丢 {self._dropped}）", flush=True)
                self._pub_t, self._pub_n = _now_t, n
