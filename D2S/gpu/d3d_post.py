# -*- coding: utf-8 -*-
"""D3D12 深度后处理链（迁移 R2）：depth_post.DepthPost.spatial_chain 的
D3D12 移植，宿主 = d3drender.D3DRenderer（d3drender.dll 生产后端）。

pass 序列与 GL 版逐式对应（归一化+dilate → 闭运算 → 坡度限幅 → 分离高斯 →
EMA/中值），公式零改动（HLSL 移植见 gpu/hlsl_depth.py）。
状态跟踪/屏障由 Tex.transition 显式管理（RENDER_TARGET/COPY_* 写状态不
可从 COMMON 提升；ECL 边界自动 decay 回 COMMON）。

离线语义：spatial_chain 每次调用内含多次 submit+wait（parity/逐帧处理用，
全部 GPU 任务完成后返回）；实时路径的帧间流水线编排属 R4 显示链集成。
"""
import struct

import numpy as np

from . import d3drender as dr
from .d3drender import FMT_R32F, ST_COMMON, ST_PIXEL_SRV, ST_RENDER_TARGET
from .hlsl import compile_shader, FULLSCREEN_VS


class D3DDepthPost:
    """D3D12 深度后处理链（宿主 = D3DRenderer）。

    纹理组（R32F res×res）：raw(输入) / dil / sl[2]（slope+gauss ping-pong）/
    ema(时域状态) / depth(最终输出) / med_ring[4]（中值模式）。
    cbuffer 打包见 gpu/hlsl_depth.pack_consts（同布局）。
    """

    def __init__(self, ren, res=224):
        self.ren = ren
        self.res = res
        mk = lambda: ren.create_tex(res, res, FMT_R32F, renderable=True)
        self.raw = ren.create_tex(res, res, FMT_R32F, renderable=True)
        self.dil = mk()
        self.sl = [mk(), mk()]
        self.ema = mk()
        self.depth = mk()
        self.med_ring = [mk() for _ in range(4)]
        self._med_ptr = 0
        self._ema_first = True
        # 成对流水（defer=True）由主循环显式开启（main.py 的 D2S_PIPELINE）。
        # 默认 False = 旧语义（链尾 wait），离屏工具/parity 门禁不受影响。
        self.defer = False
        self.disp_depth = [mk(), mk()]
        self._disp_i = 0
        self.last_disp_depth = self.disp_depth[0]
        # raw 输入上传 buffer：双缓冲 + 每缓冲"上次被 GPU 读完"的 fence 值。
        # 去掉 wait 之后，下一帧 memmove 不能踩还在被 GPU 读的那一张。
        self._pitch = dr.row_pitch(res, 4)
        self.ups = [ren.make_upload_buffer(self._pitch * res),
                    ren.make_upload_buffer(self._pitch * res)]
        self._up_i = 0
        self._up_done = [0, 0]
        self._pso = {}
        self._vs = compile_shader("depth_vs", FULLSCREEN_VS, "vs")

    def _pso_for(self, name, ps_src):
        if name in self._pso:
            return self._pso[name]
        ps = compile_shader(f"depth_{name}", ps_src, "ps")
        pso = self.ren.create_pso(f"depth_{name}", self._vs, ps, FMT_R32F)
        self._pso[name] = pso
        return pso

    @staticmethod
    def _cb(res, lo=0.0, hi=1.0, w=0.0, alpha=0.0, aa=0.0, sigma=0.0, m=0.0,
            target=0.0, dirx=0.0, diry=0.0, k=0.0, cut_area=0.0,
            range_lo=0.0, range_span=1.0):
        b = struct.pack("<15f", res, lo, hi, w, alpha, aa, sigma, m,
                        target, dirx, diry, k, cut_area,
                        range_lo, range_span) + b"\x00" * 49 * 4
        return list(np.frombuffer(b, dtype=np.float32))

    # ---- 上传 ------------------------------------------------------------
    def upload_raw(self, arr):
        """raw 深度 numpy (res,res) f32 → CPU upload buffer（写地址）。

        双缓冲：优先用"上次 GPU 读完"的那张（fence 已完成），否则换另一张；
        两张都在飞（异常）才退化成等一次——流水线里不该发生。
        """
        if self.defer:
            _done = self.ren.completed_fence()
            _nxt = 1 - self._up_i
            if _done < self._up_done[_nxt] and _done >= self._up_done[self._up_i]:
                self._up_i = _nxt
            elif _done < self._up_done[self._up_i]:
                self.ren.wait()          # 异常态兜底（不乱写正在被读的缓冲）
        self.ren.upload_to_buf(self.ups[self._up_i],
                               np.ascontiguousarray(arr, np.float32))

    def _copy_upload(self):
        """upload buffer → raw 纹理（含 COMMON→COPY_DEST→PIXEL_SRV 屏障）。
        上帧主链已 submit（ECL decay），raw 实际 COMMON——先对齐状态记录。"""
        self.raw.state = ST_COMMON
        self.ren.copy_buf_to_tex(self.ups[self._up_i], self.raw)

    # ---- pass 组 ---------------------------------------------------------
    def _pass(self, name, ps_src, dst, srcs, consts, stats=None):
        """单全屏 pass：dst→RENDER_TARGET 屏障、draw、发真实 RT→COMMON 屏障
        （供同帧后续 copy/读取——state 记录与实际一致）。
        stats: D3DStats（其 .stats Buf 绑 t7；V2 生产链用）。"""
        pso = self._pso_for(name, ps_src)
        dst.transition(ST_RENDER_TARGET)
        self.ren.draw_pass(pso, srcs, consts, dst, self.res, self.res,
                           stats=(stats.stats if stats is not None else None))
        dst.transition(ST_COMMON)

    # ---- 主链 ------------------------------------------------------------
    def spatial_chain(self, lo, hi, aa_texels=3.0, aa_v=None, grow=0,
                      close_px=0,
                      slope_target=0.0, slope_m=0.0, iters=6,
                      temporal="ema", alpha=0.19, cut=False, stats=None,
                      color_tex=None, color_size=None, raw_tex=None,
                      raw_buf=None):
        """执行完整空间+时域链：raw → normalize/dil → close → slope →
        gauss h/v → EMA/中值 → depth。返回 depth Tex。

        aa_v: 垂直方向 σ（texel 空间）。默认 None = aa_texels×2/3（旧行为，
        供 parity/离线工具）；产品按当前每眼视口纵横比传值：深度 224² 被
        拉伸到视口后，屏幕空间各向同性要求 σ_v = σ_h·(vp_w/vp_h)
        （Interleaved 16:9 → ×1.78；Half-SBS 非 Fill 每眼 960×1080 → ×0.89）。

        stats: D3DStats（V2 生产路径）——给出时 lo/hi 由 GPU 判据直读
        （忽略入参），EMA 的 cut 重置由 stats[9]（RGB 决策）驱动；lo/hi
        仍保留参数供 parity 参照（GL 语义 CPU lo/hi）。

        raw_tex: 外部 raw 深度纹理（GPU 输出的推理结果）——给出时跳过
        upload+拷贝，直接拿它当链输入（省掉 CPU 往返与一拍流水）。
        """
        from .hlsl_depth import (DIL_FS, ERO_FS, GAUSS_FS, MEDIAN_FS,
                                 NORM_DIL_FS, PP_FS, PP_STATS_FS, SLOPE_FS)
        res = self.res
        self.ren.begin_frame()
        # 本帧显示用哪一张 depth 乒乓（与上一帧交替）
        self._disp_i ^= 1
        if raw_tex is None:
            if raw_buf is not None:
                # GPU 输出的推理结果（显存 buffer）→ raw 纹理（同帧一个 PS）
                self.buf_to_tex(raw_buf, self.raw)
            else:
                self._copy_upload()   # COMMON→COPY_DEST→PIXEL_SRV（屏障内含）
            _raw = self.raw
        else:
            # GPU 输出直入：推理结果已在显存（同队列 FIFO 保证已完成）
            _raw = raw_tex
        # EMA 系数：首帧 1.0 播种；P50 的 EMA 与深度链共用同一个 alpha
        a_eff = 1.0 if self._ema_first else alpha
        if stats is not None:
            stats.record(_raw, cut=cut, color_tex=color_tex,
                         color_size=color_size)

        # 1) normalize+dilate（生产：lo/hi 读 stats t7）
        cur = self.dil
        if stats is not None:
            self._pass("dil_stats", NORM_DIL_FS, cur, [_raw],
                       self._cb(res, w=grow,
                                range_lo=stats.band_lo,
                                range_span=stats.band_span), stats=stats)
        else:
            self._pass("dil", DIL_FS, cur, [_raw],
                       self._cb(res, lo, hi, grow))

        # 2) 闭运算（可选）
        if close_px and close_px > 0:
            k = float(min(6, int(close_px)))
            tmp = self.sl[0]
            self._pass("close_max", DIL_FS, tmp, [cur], self._cb(res, 0, 1, k))
            cur2 = self.sl[1]
            self._pass("close_min", ERO_FS, cur2, [tmp],
                       self._cb(res, 0, 0, 0, k=k))
            cur = cur2

        # 3) 坡度限幅（可选）
        if slope_target > 0.0 and slope_m > 0.0:
            cands = (self.dil, self.sl[0], self.sl[1])
            for i in range(max(1, int(iters))):
                dst = next(f for f in cands if f is not cur)
                self._pass(f"slope{i}", SLOPE_FS, dst, [cur],
                           self._cb(res, m=slope_m, target=slope_target))
                cur = dst

        # 4) 分离高斯 h/v（各向异性：水平 σ=AA / 垂直 ×2/3 —— 2026-11-20 定案）
        aa = aa_texels
        aa_y = (aa * (2.0 / 3.0)) if aa_v is None else float(aa_v)
        if aa and aa > 0.0:
            cands = (self.dil, self.sl[0], self.sl[1])
            gh = next(f for f in cands if f is not cur)
            self._pass("gauss_h", GAUSS_FS, gh, [cur],
                       self._cb(res, aa=aa, sigma=aa, dirx=1.0, diry=0.0))
            gv = next(f for f in cands if f is not gh)
            self._pass("gauss_v", GAUSS_FS, gv, [gh],
                       self._cb(res, aa=aa_y,
                                sigma=aa_y, dirx=0.0, diry=1.0))
            cur = gv

        # 5) 时域 EMA / 中值 W5（EMA/中值 pass 随主链同一列表录制——列表
        # 在 submit+wait 后才可 begin_frame 复用，顺序不可颠倒）
        # 首帧 alpha=1.0 直通（GL _first / CPU _ema_first 同语义：ema 状态
        # 未播种时 mix(prev=0, cur, 0.19) 会把首帧压到 19%）
        if temporal == "ema":
            dst = self.depth
            self.ema.transition(ST_PIXEL_SRV)
            dst.transition(ST_RENDER_TARGET)
            if stats is not None:
                pso = self._pso_for("ema_pp_stats", PP_STATS_FS)
                self.ren.draw_pass(
                    pso, [cur, self.ema],
                    self._cb(res, alpha=a_eff),
                    dst, res, res, stats=stats.stats)
            else:
                pso = self._pso_for("ema_pp", PP_FS)
                self.ren.draw_pass(pso, [cur, self.ema],
                                   self._cb(res, lo=0.0, hi=1.0, alpha=a_eff),
                                   dst, res, res)
            dst.transition(ST_COMMON)     # 真实 RT→COMMON 屏障（同帧回拷）
            # EMA 状态 ← 输出：同列表紧接着录（单提交，省一次 fence 往返）
            self.ren.copy_tex_to_tex(self.ema, self.depth)
            # 收敛锚 = 最终显示深度（本帧 EMA 之后）的近峰中位
            if stats is not None:
                stats.anchor_from(self.depth)
            self._tail_submit()
        elif temporal == "median":
            ring = self.med_ring
            if self._ema_first:
                # 首帧播种：环 4 槽全部 ← cur、输出 = cur（CPU 语义对齐）
                for f in ring:
                    self.ren.copy_tex_to_tex(f, cur)
                self.ren.copy_tex_to_tex(self.depth, cur)
                if stats is not None:
                    stats.anchor_from(self.depth)
                self._tail_submit()
                self._med_ptr = 0
            else:
                dst = self.depth
                for f in ring:
                    f.transition(ST_PIXEL_SRV)
                dst.transition(ST_RENDER_TARGET)
                pso = self._pso_for("median", MEDIAN_FS)
                self.ren.draw_pass(pso, [cur] + ring, self._cb(res), dst,
                                   res, res)
                # ring 保持 PIXEL_SRV 记录（同帧内 SRV→COPY_DEST 由
                # copy_tex_to_tex 的 transition 发真实屏障）
                dst.transition(ST_COMMON)
                # 环滚动：最老槽 ← cur（同列表单提交）
                self.ren.copy_tex_to_tex(ring[self._med_ptr], cur)
                if stats is not None:
                    stats.anchor_from(self.depth)
                self._tail_submit()
                self._med_ptr = (self._med_ptr + 1) % 4
        else:
            raise ValueError(temporal)
        self._ema_first = False
        return self.depth

    def buf_to_tex(self, buf, dst_tex):
        """GPU 输出的推理结果（StructuredBuffer<float>）→ R32F 纹理。

        须在 begin_frame 之后、同帧提交之前调用（与链同一命令列表）。
        """
        from .hlsl_depth import BUF_TO_TEX_FS
        pso = self._pso_for("buf_to_tex", BUF_TO_TEX_FS)
        dst_tex.transition(ST_RENDER_TARGET)
        self.ren.draw_pass(pso, [buf], self._cb(self.res), dst_tex,
                           self.res, self.res)
        dst_tex.transition(ST_COMMON)

    def _tail_submit(self):
        """链尾：最终深度 → 显示乒乓拷贝 + 提交。

        - 拷贝与链同一命令列表（一次提交），显示侧只读 disp_depth，下一帧
          链写 self.depth 不会覆盖正在显示的那张（颜色/深度同帧配对）。
        - `defer=True`（流水线）时不等 GPU，结果下一拍再用；upload 缓冲的
          占用值同时记账，供下一帧 upload_raw 判断能否复写。
        """
        self.ren.copy_tex_to_tex(self.disp_depth[self._disp_i], self.depth)
        self.last_disp_depth = self.disp_depth[self._disp_i]
        self.ren.submit()
        self._up_done[self._up_i] = self.ren.last_fence()
        if not self.defer:
            self.ren.wait()

    # ---- 读回 ------------------------------------------------------------
    def readback_depth(self):
        """depth → numpy (res,res) f32（一站式：录拷贝+submit+wait+解析）。"""
        return self.ren.readback(self.depth, np.float32)
