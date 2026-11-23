# -*- coding: utf-8 -*-
"""GPU 判据链编排（V2）：raw 深度 → 直方图 → 分位/Otsu 锚 → z_sub → cut。

全部 compute/PS（d3drender compute 基建 + 图形根签名 t7），链 shader 直读
stats buffer（NORM_STATS_FS/PP_STATS_FS）。唯一 CPU 读回 = stats 4 floats
（收敛锚 UI 语义，延迟一帧）。

stats buffer 布局（uint[8] structured，float 位模式）：
  [0] lo（P2 映射回 raw 值域）
  [1] hi（P98）
  [2] anchor（近峰中位，归一化 [0,1]）
  [3] 历史 cut 面积（深度判据已删，恒 0，仅元组兼容）
  [4] P50（归一化域，本帧；收敛面直取，无 EMA）
  [8] RGB TV（遥测）；[9] cut 决策（纯 RGB）；[10] 本帧自适应阈值（遥测）
  [0]/[1] 是**时域平滑后**的 lo/hi（链 shader 直接用）；[6]/[7] 是原始分位

直方图量程（range 缓冲，2 floats）：**每帧 raw 的 min/max**（minmax_cs，
::4 子采样域，与直方图同一批样本）。直方图按 [min, max] 分 256 bin、分位
再映射回 raw 值——不再有按模型手填的固定区间（2026-11-24 用户定案：标定
全自动）。手填区间的两个毛病都由它消掉：太短 → 越界样本挤进末格、P98 顶死
在端点；太长 → 数据只占少数 bin、分位变粗。

z 乒乓：z_cur 写本帧 z_sub（PS 采样 raw ::4 + 本帧 lo/hi 归一化），
z_prev 读上帧（cut 对比）；帧末交换，无拷贝。
"""
import ctypes
import os

import numpy as np

from . import d3drender as dr
from .hlsl import compile_shader
from .hlsl_stats import (ANCHOR_CS, HIST_CS, MINMAX_CS, QUANT_CS, RGB_HIST_CS,
                         ZERO_CS)

ST_COMMON = 0x0
ST_UAV = 0x8             # D3D12_RESOURCE_STATE_UNORDERED_ACCESS
ST_RENDER_TARGET = 0x4
ST_NON_PIXEL_SRV = 0x40
ST_ALL_SRV = 0xC0        # NON_PIXEL|PIXEL
ST_COPY_SOURCE = 0x800

_RB128 = ctypes.c_char * 128   # stats 读回视图（每帧用，别新建数组类型）

CUT_RES = 0.35      # 旧逐像素口径（仅 depth_norm CPU 参照保留）
# RGB 场景切换（4-bit/通道 = 4096 bins 直方图 TV；2026-11-22 调研定值）：
# 同镜头运动 0.02~0.12 / 摇镜 0.12~0.30 / 硬切 0.35~0.75；
# 固定阈值必在某类片上失效 → 自适应 thr = max(floor, mean + K·std)。
CUT_RGB = 0.40       # 硬切下限（FFmpeg scene 0.4 平衡档 / 论文 0.35~0.5）
CUT_ADAPT_K = 3.0    # 自适应系数（30 帧窗，样本 <8 只用下限）
CUT_MIN_SCENE = 15   # 最小场景长（帧；PySceneDetect 默认 15，防抖）
RGB_SAMPLE_STEP = 2  # 颜色直方图采样步长（224/2=112² 网格；成本 ~0.16ms）
# 收敛锚：显示深度的【近侧质量分位】——从近端累计 q·N 质量的深度。
# 单一连续估计器（无峰检测/Otsu/回退切换）。2026-11-22 用户定案 q=10%
# （原 5%：更多近侧质量进入锚，收敛面更贴主体前表面）。
ANCHOR_Q = 0.10
# 归一化留双端余量（用户定案 2026-11-21）：P2/P98 → [0.02, 0.98]，
# 两端各留 2% —— 收缩/扩张期间超出当前范围的少量内容先吃余量，
# 实在超过才被截断
DEPTH_RANGE_LO = float(os.environ.get("D2S_DEPTH_RANGE_LO", "0.02"))
DEPTH_RANGE_HI = float(os.environ.get("D2S_DEPTH_RANGE_HI", "0.98"))
DEPTH_RANGE_SPAN = DEPTH_RANGE_HI - DEPTH_RANGE_LO
DEPTH_RANGE_MAX = DEPTH_RANGE_HI   # 兼容旧引用


class D3DStats:
    """GPU 判据块（宿主 = D3DRenderer）。"""

    def __init__(self, ren, res=224):
        self.ren = ren
        # res：int = 方形；(w, h) = 非方形（16:9 宽屏模型）
        if isinstance(res, (tuple, list)):
            self.res_w, self.res_h = int(res[0]), int(res[1])
        else:
            self.res_w = self.res_h = int(res)
        self.res = self.res_w
        # ::4 子采样域（depth_norm 同款）
        self.sub_w = (self.res_w + 3) // 4
        self.sub_h = (self.res_h + 3) // 4
        self.sub = self.sub_w           # 兼容旧调用
        self.hist = ren.make_uav_buf(256 * 4)
        self.rgb_prev = ren.make_uav_buf(4096 * 4)
        self.tv_hist = ren.make_uav_buf(64 * 4)   # 自适应 TV 环（GPU 常驻状态）
        self.stats = ren.make_uav_buf(32 * 4)
        self.range = ren.make_uav_buf(2 * 4)      # 本帧直方图量程 [min, max]
        self._p_zero = ren.create_cs_pso(
            "stats_zero", compile_shader("stats_zero", ZERO_CS, "cs"))
        self._p_minmax = ren.create_cs_pso(
            "stats_minmax", compile_shader("stats_minmax", MINMAX_CS, "cs"))
        self._p_hist = ren.create_cs_pso(
            "stats_hist", compile_shader("stats_hist", HIST_CS, "cs"))
        self._p_rgb = ren.create_cs_pso(
            "stats_rgb", compile_shader("stats_rgb", RGB_HIST_CS, "cs"))
        self._p_quant = ren.create_cs_pso(
            "stats_quant", compile_shader("stats_quant", QUANT_CS, "cs"))
        # 显示域收敛锚：最终 depth（标定→高斯→EMA 之后）的近峰中位
        self._p_anchor = ren.create_cs_pso(
            "stats_anchor", compile_shader("stats_anchor", ANCHOR_CS, "cs"))
        # 缓存的上帧结果（CPU 侧副本，供 cbuffer 与遥测）
        self.lo = 0.0
        self.hi = 1.0
        self.anchor = 0.0
        self.cut_ratio = 0.0
        # 读回字段先给默认值：流水线模式下"还没有任何一帧读完"时（静态桌面
        # 没帧可采），30s 遥测会直接引用它们——不初始化就会 AttributeError。
        self.rgb_tv = 0.0
        self.cut_flag = 0.0
        self.cut_thr = 0.0
        self.p50 = 0.0
        self.lo_raw = 0.0
        self.hi_raw = 0.0
        self.anchor_disp = 0.0
        self._rb = None
        # 归一化范围平滑时间常数（ms）。默认 0 = **每帧快照**：
        # 6 物体等间距实验实锤——模型尺度抖动（W 3~5 随机）在快照归一化下
        # 被仿射不变性完全吃掉（z 峰峰 ≤0.006），而 80ms 平滑反而把尺度
        # 抖动变成位置抖动（最外物体 z 摆 0.77~1.20、被截断）。
        # 想要"物体移出缓慢回缩"可显式设 D2S_RANGE_TAU_MS=80，代价如上。
        self.range_tau_ms = float(os.environ.get("D2S_RANGE_TAU_MS", "0"))
        # 范围更新模式：percentile（默认，无时域）＝每帧快照 P2/P98；
        # decile＝P10/P90；ratchet/lerp 保留对照
        self.range_mode = os.environ.get(
            "D2S_RANGE_MODE", "percentile").strip().lower()
        self._range_t = 0.0
        _bl, _bs = self._band(self.range_mode)
        print(f"[stats] 归一化口径 = {self.range_mode}"
              f"（映射带 [{_bl:g}, {_bl + _bs:g}]）", flush=True)
        # RGB 切变判据参数（env 可调；确定性测试可直接改这三个属性）
        self.cut_floor = float(os.environ.get("D2S_CUT_RGB", str(CUT_RGB)))
        self.cut_k = float(os.environ.get("D2S_CUT_ADAPT_K",
                                          str(CUT_ADAPT_K)))
        self.cut_min_scene = float(os.environ.get("D2S_CUT_MIN_SCENE",
                                                  str(CUT_MIN_SCENE)))
        # 收敛锚近侧质量分位（0.05 = 最近的 5% 质量所在深度）
        self.anchor_q = float(os.environ.get("D2S_ANCHOR_Q",
                                             str(ANCHOR_Q)))

    def _range_alpha(self, a_range):
        """返回范围平滑 α：显式给值则直用（测试）；否则按真实帧间隔算。
        首帧返回 1.0：直接快照（shader 里未播种也会快照，双保险）。"""
        if a_range is not None:
            return float(a_range)
        if self.range_tau_ms <= 0.0:
            return 1.0          # 快照（默认）：每帧直接用本帧分位
        import math as _m
        import time as _t
        now = _t.perf_counter()
        if self._range_t <= 0.0:
            self._range_t = now
            return 1.0
        dt = now - self._range_t
        self._range_t = now
        return 1.0 - _m.exp(-dt * 1000.0 / max(1.0, self.range_tau_ms))

    def _cb(self, cut_res=CUT_RES, a_range=1.0, mode=None):
        """compute cbuffer：P[0] 的 x/y 预留（量程在 range 缓冲里，由
        minmax_cs 每帧写，不走 cbuffer），P[1]=(sub_res, step, sub_h,
        range_alpha)，P[2]=(0, range_lo, range_span, range_mode)。
        cut 阈值按 range_span 缩放 → 与 [0,1] 域判定等价。"""
        m = (mode or self.range_mode)
        band_lo, band_span = self._band(m)
        return [0.0, 0.0, float(self.res),
                float(cut_res) * band_span, float(self.sub_w), 4.0,
                float(self.sub_h), float(a_range), 0.0,
                float(band_lo), float(band_span),
                {"lerp": 0.0, "percentile": 1.0, "ratchet": 2.0,
                 "decile": 5.0}.get(m, 1.0)]

    def record(self, raw_tex, cut=True, alpha_range=None, range_mode=None,
               color_tex=None, color_size=None):
        """录判据链到当前帧（须在 begin_frame 之后，raw 已就绪）。

        raw_tex: R32F res×res（raw 深度）。链内完成：min/max(量程，::4) →
        hist(::4) → RGB 场景切换判据（需 color_tex）→ 分位/Otsu 锚。
        cut=False 时 RGB 仍算 TV（遥测）但不触发切变决策（供确定性测试）。
        color_tex/color_size: 采集帧 BGRA8 + (w,h)；缺失时无 cut 判据
        （stats[8]/[9] 被 ZERO_CS 清零）。
        RGB 判据 = TV(4096 bins) > max(cut_floor, mean + cut_k·std)
        （30 帧滑窗，均不含本帧）且距上次切 ≥ cut_min_scene 帧。
        alpha_range=归一化范围的平滑系数（None=按 τ 与真实 dt 算，
        显式值供确定性门禁）。扩张/收缩统一。
        """
        ren = self.ren
        g_w = (self.sub_w + 15) // 16
        g_h = (self.sub_h + 15) // 16
        # ECL 后资源实际状态恒 COMMON：记录先对齐（槽位可能留上帧值）
        for b in (self.hist, self.stats, self.rgb_prev, self.tv_hist,
                  self.range):
            b.state = ST_COMMON
        raw_tex.transition(ST_ALL_SRV)   # minmax / hist(CS) 读
        self.hist.transition(ST_UAV)
        self.stats.transition(ST_UAV)
        ren.dispatch(self._p_zero, [self.hist, self.stats], [], self._cb(), 1)
        # UAV→UAV 依赖必须显式执行屏障（zero→hist / zero→quant 共用缓冲）
        self.hist.uav_barrier()
        self.stats.uav_barrier()
        # 量程：本帧 raw 的 min/max（::4 域）→ range[2]，直方图按它分格
        self.range.transition(ST_UAV)
        ren.dispatch(self._p_minmax, [self.range], [raw_tex], self._cb(), 1)
        self.range.transition(ST_NON_PIXEL_SRV)     # 写完 → 供 hist/quant 读
        # 直方图（::4 域，对齐 depth_norm.percentile_stats）
        ren.dispatch(self._p_hist, [self.hist], [raw_tex, self.range],
                     self._cb(), g_w, g_h, 1)
        self.hist.uav_barrier()
        if color_tex is not None:
            w, h = (color_size or (self.res, self.res))
            color_tex.transition(ST_ALL_SRV)
            self.rgb_prev.transition(ST_UAV)
            self.tv_hist.transition(ST_UAV)
            ren.dispatch(self._p_rgb,
                         [self.rgb_prev, self.stats, self.tv_hist],
                         [color_tex],
                         [float(self.res), float(w), float(h),
                          float(self.cut_floor),
                          1.0 if cut else 0.0, float(self.cut_k),
                          float(RGB_SAMPLE_STEP), 0.0,
                          float(self.cut_min_scene), 0.0, 0.0, 0.0], 1)
            self.stats.uav_barrier()
        # 分位 + Otsu 锚（hist UAV→NON_PIXEL_SRV 直转，compute 读态）
        self.hist.transition(ST_NON_PIXEL_SRV)
        _a_r = self._range_alpha(alpha_range)
        _mode = (self.range_mode if range_mode is None
                 else str(range_mode).lower())
        ren.dispatch(self._p_quant, [self.stats], [self.hist, self.range],
                     self._cb(a_range=_a_r, mode=_mode), 1)
        # stats 给 PS 读（t7；COMMON 读隐式提升）
        self.stats.transition(ST_COMMON)
        self.hist.transition(ST_COMMON)
        self.rgb_prev.transition(ST_COMMON)
        self.tv_hist.transition(ST_COMMON)
        self.range.transition(ST_COMMON)

    @staticmethod
    def _band(mode):
        """归一化映射带：decile = [0.1,0.9]，其余 = [0.02,0.98]。"""
        if mode == "decile":
            return 0.1, 0.8
        return DEPTH_RANGE_LO, DEPTH_RANGE_SPAN

    @property
    def band_lo(self):
        return self._band(self.range_mode)[0]

    @property
    def band_span(self):
        return self._band(self.range_mode)[1]

    def anchor_from(self, depth_tex):
        """对【最终显示深度】（标定→高斯→EMA 之后）算近侧质量分位 → stats[31]。

        收敛面的唯一基准：必须是实际显示出来的深度，而不是 raw。
        单一估计器：从近端累计 anchor_q·N 质量的深度（无峰检测/无回退切换）。
        每帧在链内 submit 之前调用（同一条命令列表，零额外提交）。
        """
        ren = self.ren
        depth_tex.transition(ST_ALL_SRV)
        self.stats.transition(ST_UAV)
        cb = [float(self.res_w), 4.0, float(self.res_h), 0.0,
              float(self.anchor_q), 0.0, 0.0, 0.0]
        ren.dispatch(self._p_anchor, [self.stats], [depth_tex], cb, 1)
        self.stats.uav_barrier()

    def readback_start(self):
        """提交 stats 读回（不等待）。返回本次提交的内部 fence 值。

        流水线用：本拍提交，下一拍用 `readback_poll(v)` 非阻塞取值。
        `d3dr_rb_exec` 会分配一个新的内部 fence 值（不 Signal 外部 fence）。
        """
        ren = self.ren
        d = dr.dll()
        if self._rb is None:
            self._rb = d.d3dr_make_readback_buf(ren.ctx, 128)
        if d.d3dr_rb_begin(ren.ctx) != 0:
            raise RuntimeError("[stats] rb_begin 失败")
        if self.stats.state != ST_COPY_SOURCE:
            d.d3dr_rb_barrier(ren.ctx, self.stats.res, self.stats.state,
                              ST_COPY_SOURCE)
        d.d3dr_rb_copy_buf_to_buf(ren.ctx, self.stats.res, self._rb, 128)
        rc = d.d3dr_rb_exec(ren.ctx)
        if rc != 0:
            raise RuntimeError(f"[stats] rb_exec 失败 rc={rc:#x}: "
                               + ren.last_err())
        self.stats.state = ST_COMMON
        return int(d.d3dr_last_fence(ren.ctx))

    def readback_poll(self, v):
        """`v` 之后才取；未完成返回 None（不阻塞、不改任何字段）。"""
        d = dr.dll()
        if int(d.d3dr_completed_fence(self.ren.ctx)) < int(v):
            return None
        ptr = d.d3dr_buf_ptr(self._rb)
        # (c_char * 128) 每帧新建数组类型会成环 → 模块级复用
        vals = np.frombuffer(_RB128.from_address(ptr),
                             dtype=np.float32).copy()
        # [0]/[1] raw 域 lo/hi；[2] Otsu 锚(raw 窗)；
        # [3] 历史 cut 面积（深度判据已删，恒 0，仅为元组兼容保留）；
        # [8] RGB TV；[9] cut 决策（纯 RGB）；[10] 自适应阈值；[31] 显示域收敛锚
        self.lo = float(vals[0])
        self.hi = float(vals[1])
        self.anchor = float(vals[2])
        self.cut_ratio = float(vals[3])
        self.rgb_tv = float(vals[8])
        self.cut_flag = float(vals[9])
        self.cut_thr = float(vals[10])   # 本帧自适应阈值（floor 或 mean+K·std）
        self.p50 = float(vals[4])
        self.lo_raw = float(vals[6])     # 本帧原始 P2
        self.hi_raw = float(vals[7])     # 本帧原始 P98
        self.anchor_disp = float(vals[31])
        return (self.lo, self.hi, self.anchor, self.cut_ratio, self.p50,
                self.anchor_disp)

    def readback(self):
        """读回 stats 4 floats（一站式；独立帧）。返回 (lo, hi, anchor, cut)。

        lo/hi 已按本帧量程（range 缓冲的 min/max）还原为 raw 值。
        """
        ren = self.ren
        v = self.readback_start()
        ren.wait()
        return self.readback_poll(v)
