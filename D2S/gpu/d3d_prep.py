# -*- coding: utf-8 -*-
"""D3D12 深度预处理（迁移 R3）：viewer.gpu_preprocess 的 D3D12 移植。

BGRA8 采集帧 → R32F CHW 平铺 + ImageNet 归一化。生产几何 = flat
（W=192、H=3R²/192，线性序 CHW、行距 768B=256 对齐且无 padding——
与 GL _prep_flat 零拷贝布局同款，shim v2 按此读入）。
宿主 = d3drender.D3DRenderer。
"""
import numpy as np

from . import d3drender as dr
from .d3drender import FMT_BGRA8, FMT_R32F, ST_PIXEL_SRV, ST_RENDER_TARGET
from .hlsl import compile_shader, FULLSCREEN_VS
from .hlsl_depth import PREP_FS

FLAT_W = 192   # 与 viewer.gpu_depth_init _prep_flat 同款


class D3DDepthPrep:
    """D3D12 预处理块（宿主 = D3DRenderer）。

    color 纹理（BGRA8 采集帧，任意尺寸）→ prep 纹理（flat 192×784 R32F）。
    color 纹理换代（尺寸变化）时 set_frame 内重建。
    """

    def __init__(self, ren, res=224, flat=True, normalize=True):
        self.ren = ren
        # res：int = 方形 res×res；(w, h) = 非方形（16:9 宽屏模型，如 672×384）
        if isinstance(res, (tuple, list)):
            self.res_w, self.res_h = int(res[0]), int(res[1])
        else:
            self.res_w = self.res_h = int(res)
        self.res = self.res_w          # 兼容旧调用（当方形用的那些地方）
        self.flat = flat
        # normalize=True：prep 做 ImageNet mean/std（DepthART 等"外部归一化"
        # 模型）；False：原样输出 0..1 RGB（ZipDepth 等 mean/std 在图内的
        # 模型，再做一次就双重标准化）。
        self.normalize = bool(normalize)
        self.flat_w = FLAT_W if flat else 0
        if flat:
            n = self.res_w * self.res_h * 3
            if n % FLAT_W:
                raise ValueError(f"prep flat 布局要求 3*W*H 能被 {FLAT_W} 整除"
                                 f"（{self.res_w}x{self.res_h}）")
            self.prep_w = FLAT_W
            self.prep_h = n // FLAT_W
        else:
            self.prep_w = self.res_w * 3
            self.prep_h = self.res_h
        self.prep = ren.create_tex(self.prep_w, self.prep_h, FMT_R32F,
                                   renderable=True)
        self._color = None
        self._color_buf = None
        self._vs = compile_shader("depth_vs", FULLSCREEN_VS, "vs")
        self._pso = ren.create_pso(
            "depth_prep", self._vs, compile_shader("depth_prep", PREP_FS),
            FMT_R32F)

    def consts(self):
        """prep pass 的 cbuffer：[res_w, flat_w, normalize_flag, res_h]。"""
        return [float(self.res_w), float(self.flat_w),
                1.0 if self.normalize else 0.0, float(self.res_h)]

    def set_frame(self, bgra_u8):
        """BGRA HWC uint8 帧上传（尺寸变化时换代重建 upload 对）。
        接受 3 通道（补 A=255 凑 BGRA8 4B/px——upload 按 itemsize 布局）。"""
        h, w = bgra_u8.shape[:2]
        if bgra_u8.shape[2] == 3:
            b4 = np.empty((h, w, 4), np.uint8)
            b4[:, :, :3] = bgra_u8
            b4[:, :, 3] = 255
            bgra_u8 = b4
        if self._color is None or self._color.w != w or self._color.h != h:
            if self._color is not None:
                self._color.release()
            self._color = self.ren.create_tex(w, h, FMT_BGRA8,
                                              renderable=False)
            self._color_buf = self.ren.make_upload_buffer(
                dr.row_pitch(w, 4) * h)
        self.ren.upload_to_buf(self._color_buf, bgra_u8, bytes_per_px=4)
        self.ren.begin_frame()
        self._color.state = dr.ST_COMMON
        self.ren.copy_buf_to_tex(self._color_buf, self._color)
        self.ren.submit()
        self.ren.wait()

    def run(self):
        """prep pass：color → prep（CHW 平铺）。返回 prep Tex。"""
        self.ren.begin_frame()
        self._color.transition(ST_PIXEL_SRV)
        self.prep.state = dr.ST_COMMON
        self.prep.transition(ST_RENDER_TARGET)
        self.ren.draw_pass(self._pso, [self._color], self.consts(),
                           self.prep, self.prep_w, self.prep_h)
        self.prep.state = dr.ST_COMMON
        self.ren.submit()
        self.ren.wait()
        return self.prep

    def readback(self):
        """prep 纹理 → numpy (prep_h, prep_w) f32（ravel 后即 CHW 线性序）。"""
        return self.ren.readback(self.prep, np.float32)
