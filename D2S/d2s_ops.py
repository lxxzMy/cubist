# -*- coding: utf-8 -*-
"""双后端热路径算子（去 torch 瘦身计划 S1）。

torch 在本管线里只做四件事：颜色转换、缩放、深度后处理、EMA。这里把
热路径的 12 个操作做成双实现，统一以 numpy 数组为输入/输出约定：

    RGB 图像:  HWC uint8
    模型输入:  CHW float32（0..1 或已归一化）
    深度图:    HW float32，值域 [0,1]

D2S_BACKEND=torch   镜像原 torch/CUDA 实现（from_numpy -> cuda -> op -> numpy）
D2S_BACKEND=numpy   纯 numpy + cv2，不 import torch

torch 分支的存在只是为了 S1/S2 的逐像素对拍（数值上必须是 depth.py 的
逐 op 镜像，镜像处注明行号）；它的输入/输出仍走 numpy 边界，因此不追求
性能 —— 性能对拍在管线级（改前管线 vs 改后管线）做。S3 之后默认 numpy，
torch 分支随 torch 一起删除。

约定：
- 所有浮点计算保持 float32（与管线现状一致，避免 float64 悄悄变慢）。
- "1 LSB" 容差 = 1/255（uint8 域 / 深度 [0,1] 域的同量级）。
"""
import os
import numpy as np
import cv2

BACKEND = os.environ.get("D2S_BACKEND", "numpy").strip().lower()
if BACKEND not in ("torch", "numpy"):
    raise ValueError(f"D2S_BACKEND must be 'torch' or 'numpy', got {BACKEND!r}")

# ---- torch 惰性加载（numpy 后端永远不 import torch，冷启动 ~40s -> ~10s 的关键）----
_torch = None
_dev = None


def _t():
    """惰性 import torch 并选定设备。只在 BACKEND == 'torch' 时会被调用。"""
    global _torch, _dev
    if _torch is None:
        import torch
        _torch = torch
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _torch


def _to_t(a: np.ndarray):
    return _torch.from_numpy(a).to(_dev)


def _to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy()


# ============================================================================
# 1. 颜色转换：BGRA/BGR -> RGB（depth.py process() CUDA 分支 line 550）
#    整数域运算，两后端位级一致；torch 分支留作对拍基准。
# ============================================================================
def bgra_to_rgb(bgrx: np.ndarray) -> np.ndarray:
    if BACKEND == "numpy":
        if bgrx.shape[2] == 4:
            return cv2.cvtColor(bgrx, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(bgrx, cv2.COLOR_BGR2RGB)
    torch = _t()
    t = _to_t(bgrx)
    t = t[..., :3].flip(-1)  # 切 3 通道再翻转（镜像原实现的避免 fancy-index 写法）
    return _to_np(t.contiguous())


# ============================================================================
# 2. 工作分辨率缩放（depth.py process()：本部署 1080p->1080 为恒等变换）
#    numpy 分支不再升 float32 —— 全程 uint8（计划决策 1）。
# ============================================================================
def resize_work(rgb_hwc_u8: np.ndarray, target_height: int) -> np.ndarray:
    h0, w0 = rgb_hwc_u8.shape[:2]
    if target_height >= h0:
        return rgb_hwc_u8
    new_h = (target_height // 2) * 2
    new_w = (int(w0 * target_height / h0) // 2) * 2
    if BACKEND == "numpy":
        return cv2.resize(rgb_hwc_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    torch = _t()
    t = _to_t(rgb_hwc_u8).permute(2, 0, 1).float().unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(new_h, new_w), mode="bilinear", align_corners=False,
        antialias=True)  # 镜像 process() 的 antialias 下采样
    return _to_np(t[0]).transpose(1, 2, 0).round().clip(0, 255).astype(np.uint8)


# ============================================================================
# 3. 模型输入缩放 + 归一化（depth.py predict_depth line 2343-2368 +
#    _resize_patch_aligned_t line 676-706）。输出 float32 CHW。
#    几何（目标尺寸）用纯 python 计算，两后端共用，保证形状一致。
# ============================================================================
def model_input_geometry(h: int, w: int, target: int, patch):
    if patch is None:
        return target, target
    longest = max(h, w)
    scale = target / float(longest) if longest != target else 1.0
    sh = max(1, int(round(h * scale)))
    sw = max(1, int(round(w * scale)))

    def nearest_multiple(x, p):
        down = (x // p) * p
        up = down + p
        return up if abs(up - x) <= abs(x - down) else down

    return max(1, nearest_multiple(sh, patch)), max(1, nearest_multiple(sw, patch))


def resize_model_input(rgb_hwc_u8: np.ndarray, target: int, patch,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                       normalize: bool = True) -> np.ndarray:
    """HWC uint8 RGB -> float32 CHW（缩放 + /255 + 可选 ImageNet 归一化）。

    模型是 DepthART（MobileNetV4-M-slim-SPF, 224），官方输入约定是 RGB 图像；
    mean/std 是标准 ImageNet 的 **RGB** 统计量。ONNX 里没有通道顺序的元数据，
    所以顺序由上游约定保证，并靠 viewer 启动时的已知色自检兜底 —— 改这里之前
    先确认通道顺序，R/B、R/G、G/B 任一错位都会让深度明显劣化（实测相关系数
    0.07~0.96 不等）。

    patch=None  方形缩放到 (target, target)（方形缩放路径，bilinear 无 antialias）
    patch=int   长边缩放后对齐 patch 网格（DA3 等；torch 现状 bicubic+antialias，
                numpy 用 INTER_AREA —— 该路径在 S3 后随 torch 模型一起失效，
                仅作占位）

    numpy 侧在 uint8 域做 LINEAR 缩放（0.3ms；float 域要 9.3ms，其中
    astype(f32) 就占 8.7ms）。插值结果量化回 uint8 引入 ≤0.75 LSB 输入差，
    实测端到端（ORT 推理 + 2/98 归一化后）：深度差 max 2.5%、mean 0.16%，
    折合 4px 视差预算下 max 0.10px / mean 0.007px —— 低于可感知阈值。
    """
    h0, w0 = rgb_hwc_u8.shape[:2]
    new_h, new_w = model_input_geometry(h0, w0, target, patch)
    if BACKEND == "numpy":
        if (new_h, new_w) != (h0, w0):
            if rgb_hwc_u8.dtype == np.uint8:
                interp = cv2.INTER_AREA if patch is not None else cv2.INTER_LINEAR
                x = cv2.resize(rgb_hwc_u8, (new_w, new_h), interpolation=interp)
            else:
                x = cv2.resize(rgb_hwc_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            x = rgb_hwc_u8
        x = x.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)
    else:
        torch = _t()
        t = _to_t(rgb_hwc_u8).permute(2, 0, 1).float().unsqueeze(0)
        if (new_h, new_w) != (h0, w0):
            if patch is None:
                t = torch.nn.functional.interpolate(
                    t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            else:
                t = torch.nn.functional.interpolate(
                    t, size=(new_h, new_w), mode="bicubic",
                    align_corners=False, antialias=True)
        x = _to_np(t[0]) / np.float32(255.0)
    if normalize:
        m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        x = (x - m) / s
    return np.ascontiguousarray(x, dtype=np.float32)


# ============================================================================
# 4. 分位数归一化（depth.py normalize() line 1037-1088）
#    2/98 百分位 clip + min-max，输出 [0,1]。次序统计量精确镜像 topk 无 lerp。
# ============================================================================
def percentile_normalize(d_hw_f32: np.ndarray, percentile: float = 2.0,
                         subsample_cap: int = 6144,
                         metric: bool = False) -> np.ndarray:
    if BACKEND == "numpy":
        if metric:  # metric 模型先做 1/depth 反转（valid = d>0）
            valid = d_hw_f32 > 0
            inv = np.where(valid, 1.0 / np.maximum(d_hw_f32, np.float32(1e-12)), d_hw_f32)
            v = inv[valid].ravel()
        else:
            inv = d_hw_f32
            v = inv.ravel()  # C 序 == torch.flatten

        n = v.size
        if n <= 10:
            dmin = dmax = np.float32(0.0)
        else:
            if n > subsample_cap:  # 与 torch 相同的降采样步长（确定性，可对拍）
                step = (n + subsample_cap - 1) // subsample_cap
                v = v[::step]
            m = v.size
            lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
            tail = min(m, max(1, int(round(lo_q * (m - 1))) + 1))
            if tail == m:
                dmin, dmax = v.min(), v.max()
            else:  # == topk(smallest).max() / topk(largest).min()
                part = np.partition(v, (tail - 1, m - tail))
                dmin, dmax = part[tail - 1], part[m - tail]

        denom = max(dmax - dmin, np.float32(1e-6))  # torch clamp_min(1e-6)
        out = (inv - dmin) / denom
        return np.clip(out, np.float32(0.0), np.float32(1.0), out=out)

    # ---- torch 镜像（normalize + _percentile_bounds_no_lerp line 969-979）----
    torch = _t()
    d = _to_t(d_hw_f32)
    if metric:
        valid = d > 0
        inv = torch.where(valid, 1.0 / d.clamp(min=1e-12), d)
        v = inv[valid]
    else:
        inv = d
        v = inv.flatten()
    n = v.numel()
    if n <= 10:
        dmin = dmax = torch.zeros((), device=d.device)
    else:
        vv = v
        if vv.numel() > subsample_cap:
            step = (vv.numel() + subsample_cap - 1) // subsample_cap
            vv = vv[::step]
        lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
        tail = min(vv.numel(), max(1, int(round(lo_q * (vv.numel() - 1))) + 1))
        if tail == vv.numel():
            dmin, dmax = vv.min(), vv.max()
        else:
            dmin = torch.topk(vv, tail, largest=False, sorted=False).values.max()
            dmax = torch.topk(vv, tail, largest=True, sorted=False).values.min()
    denom = (dmax - dmin).clamp_min(1e-6)
    out = ((inv - dmin) / denom).clamp(0.0, 1.0)
    return _to_np(out)


# ============================================================================
# 5. Gamma（depth.py apply_gamma line 960；gamma==1.0 恒等）
# ============================================================================
def apply_gamma(d_hw_f32: np.ndarray, gamma: float = 1.45) -> np.ndarray:
    if gamma == 1.0:
        return d_hw_f32
    if BACKEND == "numpy":
        return np.power(d_hw_f32, np.float32(gamma), dtype=np.float32)
    torch = _t()
    return _to_np(torch.pow(_to_t(d_hw_f32), gamma))


# ============================================================================
# 6. 前景对比度（depth.py apply_foreground_scale line 709-736）
# ============================================================================
def foreground_scale(d_hw_f32: np.ndarray, scale: float, mid: float = 0.5,
                     eps: float = 1e-6) -> np.ndarray:
    if BACKEND == "numpy":
        d = np.clip(d_hw_f32, 0.0, 1.0)
        if abs(scale) < eps:
            return d
        exponent = 1.0 / (1.0 + scale)
        dist = d - np.float32(mid)
        out = np.float32(mid) + np.sign(dist) * np.power(np.abs(dist), np.float32(exponent))
        return np.clip(out, 0.0, 1.0)
    torch = _t()
    d = _to_t(d_hw_f32).clamp(0.0, 1.0)
    if abs(scale) < eps:
        return _to_np(d)
    exponent = 1.0 / (1.0 + scale)
    dist = d - mid
    out = mid + torch.sign(dist) * torch.pow(torch.abs(dist), exponent)
    return _to_np(out.clamp(0.0, 1.0))


# ============================================================================
# 7. 空域平滑 AA（depth.py anti_alias line 740-765）
#     可分离高斯，k = 3*strength|1，sigma = 0.5*strength，零边界（F.conv2d 默认）。
# ============================================================================
def anti_alias(d_hw_f32: np.ndarray, strength: float = 1.0,
                v_ratio: float = 2.0 / 3.0) -> np.ndarray:
    """各向异性分离高斯（2026-11-20 用户定案 水平3/垂直2）：σh=0.5·strength，
    σv=σh·v_ratio——与 GPU depth_post.spatial_chain 同口径（parity 在岗）。"""
    k = int(3 * strength) | 1
    if k < 3:
        return d_hw_f32
    if BACKEND == "numpy":
        sigma = 0.5 * strength
        sigma_v = sigma * v_ratio
        kv = max(3, int(3 * strength * v_ratio) | 1)
        coords = np.arange(k, dtype=np.float32) - (k // 2)
        gauss = np.exp(-(coords ** 2) / np.float32(2.0 * sigma * sigma))
        gauss = (gauss / gauss.sum()).astype(np.float32)
        coords_v = np.arange(kv, dtype=np.float32) - (kv // 2)
        gauss_v = np.exp(-(coords_v ** 2) / np.float32(2.0 * sigma_v * sigma_v))
        gauss_v = (gauss_v / gauss_v.sum()).astype(np.float32)
        # sepFilter2D 先 x 后 y，与 depth.py 先 kx 后 ky 同序。边界=REPLICATE
        #（2026-11-20 parity 定案：原 BORDER_CONSTANT(0) 零填充把边带深度拉向
        # 远、且与 GPU 侧纹理钳制分叉 0.023 MAE——两侧统一为边缘复制）。
        # 注意 torch 分支的 conv2d 零 padding 与此不再对拍（detorch 后 numpy
        # 为唯一产品后端）。
        return cv2.sepFilter2D(d_hw_f32, -1, gauss.reshape(1, -1), gauss_v.reshape(-1, 1),
                               borderType=cv2.BORDER_REPLICATE)
    torch = _t()
    d = _to_t(d_hw_f32)[None, None]
    sigma = 0.5 * strength
    coords = torch.arange(k, device=d.device, dtype=d.dtype) - k // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kx = gauss.view(1, 1, 1, -1)
    ky = gauss.view(1, 1, -1, 1)
    F = torch.nn.functional
    d = F.conv2d(d, kx, padding=(0, k // 2))
    d = F.conv2d(d, ky, padding=(k // 2, 0))
    return _to_np(d[0, 0])


# ============================================================================
# 8. 前景膨胀（depth.py DEPTH_GROW line 2405-2409：replicate pad + max_pool2d）
#    max 滤波下，越界样本不参与（cv2 dilate 默认边界）与复制边缘等价：
#    任何越界窗口必然覆盖到边缘像素，重复取 max 不改变结果。
# ============================================================================
def depth_grow(d_hw_f32: np.ndarray, g: int) -> np.ndarray:
    if g <= 0:
        return d_hw_f32
    if BACKEND == "numpy":
        return cv2.dilate(d_hw_f32, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    torch = _t()
    d = _to_t(d_hw_f32)[None, None]
    d = torch.nn.functional.pad(d, (g, g, g, g), mode="replicate")
    d = torch.nn.functional.max_pool2d(d, 2 * g + 1, stride=1)
    return _to_np(d[0, 0])


# ============================================================================
# 9. 边缘贴合（depth.py joint_bilateral_filter line 991-1020，DEPTH_EDGE_SNAP）
#    guide = 模型输入分辨率的 0..1 RGB（CHW float32）。
# ============================================================================
def edge_snap(depth_hw_f32: np.ndarray, guide_chw_f32: np.ndarray,
              radius: int = 2, sigma_color: float = 0.08) -> np.ndarray:
    if BACKEND == "numpy":
        from numpy.lib.stride_tricks import sliding_window_view
        H, W = depth_hw_f32.shape
        k = 2 * radius + 1
        lum = (guide_chw_f32[0] * np.float32(0.299) + guide_chw_f32[1] * np.float32(0.587)
               + guide_chw_f32[2] * np.float32(0.114))
        gp = sliding_window_view(np.pad(lum, radius, mode="edge"), (k, k))
        dp = sliding_window_view(np.pad(depth_hw_f32, radius, mode="edge"), (k, k))
        # (H, W, k, k) -> (k*k, H, W)，对应 F.unfold 的布局
        gp = gp.reshape(H, W, k * k).transpose(2, 0, 1)
        dp = dp.reshape(H, W, k * k).transpose(2, 0, 1)

        yy, xx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        r2 = ((yy - radius) ** 2 + (xx - radius) ** 2).astype(np.float32)
        w = np.exp(-r2.reshape(k * k, 1, 1) / np.float32(2.0 * radius * radius)
                   - ((gp - lum[None]) ** 2) / np.float32(2.0 * sigma_color * sigma_color))
        out = (w * dp).sum(0) / np.maximum(w.sum(0), np.float32(1e-6))
        return np.clip(out, 0.0, 1.0, out=out.astype(np.float32))

    torch = _t()
    F = torch.nn.functional
    depth = _to_t(depth_hw_f32)
    guide = _to_t(guide_chw_f32)
    H, W = depth.shape[-2:]
    k = 2 * radius + 1
    pad = (radius, radius, radius, radius)
    lum = (guide[0] * 0.299 + guide[1] * 0.587 + guide[2] * 0.114)
    gp = F.unfold(F.pad(lum.unsqueeze(0).unsqueeze(0), pad, mode="replicate"), k)[0]
    gp = gp.view(k * k, H, W)
    dp = F.unfold(F.pad(depth.unsqueeze(0).unsqueeze(0), pad, mode="replicate"), k)[0]
    dp = dp.view(k * k, H, W)
    yy, xx = torch.meshgrid(torch.arange(k, device=depth.device),
                            torch.arange(k, device=depth.device), indexing="ij")
    r2 = ((yy - radius) ** 2 + (xx - radius) ** 2).to(depth.dtype)
    w = torch.exp(-r2.reshape(k * k, 1, 1) / (2.0 * radius * radius)
                  - ((gp - lum.unsqueeze(0)) ** 2) / (2.0 * sigma_color * sigma_color))
    out = (w * dp).sum(0) / w.sum(0).clamp_min(1e-6)
    return _to_np(out.clamp_(0.0, 1.0).to(depth.dtype))


# ============================================================================
# 10. 时间平滑 EMA（depth.py DepthStabilizer line 2193-2290）
#     fixed / fixed+valve / adaptive 三模式，逐 op 镜像，原地更新。
# ============================================================================
class StabilizerNumpy:
    """与 DepthStabilizer 数值语义一致；输入输出均为 HW float32 numpy。"""

    def __init__(self, alpha=0.99, min_alpha=0.15, diff_lo=0.12, diff_hi=0.32,
                 valve_lo=0.35, valve_hi=0.60, mode="adaptive"):
        self.alpha = alpha
        self.valve_lo = valve_lo
        self.valve_hi = max(valve_hi, valve_lo + 1e-6)
        self.min_alpha = min_alpha
        self.diff_lo = diff_lo
        self.diff_hi = max(diff_hi, diff_lo + 1e-6)
        self.mode = mode
        self.prev = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return depth
        if self.prev is None or self.prev.shape != depth.shape:
            self.prev = depth.copy()
            return depth
        p = self.prev
        if self.mode == "smooth":
            # 连续动态响应（2026-11-10 用户需求"动态大更新快/动态小更新慢"）：
            # alpha = base + (min-base)*smoothstep(0, D, |diff|)，D=valve_lo。
            # 相比 fixed+valve：无硬阈值断崖（跨阈值突变）、无死区（阈值内
            # 恒速爬=无区分度）——diff 单调连续映射到跟随速度。
            a = np.abs(depth - p, dtype=np.float32)
            t = a * np.float32(1.0 / max(1e-6, self.valve_lo))
            np.clip(t, 0.0, 1.0, out=t)
            ss = t * t * (np.float32(3.0) - np.float32(2.0) * t)  # 3t²-2t³
            ss *= np.float32(self.min_alpha - self.alpha)
            ss += np.float32(self.alpha)
            p *= ss
            p += (np.float32(1.0) - ss) * depth
            return p
        if self.mode in ("fixed+valve", "adaptive"):
            # a = clamp((|depth-prev| - lo) / (hi - lo), 0, 1)
            # alpha(x) = base + a * (min - base)；prev = a*prev + (1-a)*depth
            a = np.abs(depth - p, dtype=np.float32)
            lo = self.valve_lo if self.mode == "fixed+valve" else self.diff_lo
            hi = self.valve_hi if self.mode == "fixed+valve" else self.diff_hi
            a -= np.float32(lo)
            a /= np.float32(hi - lo)
            np.clip(a, 0.0, 1.0, out=a)
            a *= np.float32(self.min_alpha - self.alpha)
            a += np.float32(self.alpha)
            p *= a
            p += (np.float32(1.0) - a) * depth  # (1-a)*depth，与 torch 同序
            return p
        p *= np.float32(self.alpha)
        p += depth * np.float32(1.0 - self.alpha)
        return p


class StabilizerTorch:
    """torch 镜像（对拍用）。状态保存在 CUDA tensor 上，边界只做 numpy 转换。"""

    def __init__(self, alpha=0.99, min_alpha=0.15, diff_lo=0.12, diff_hi=0.32,
                 valve_lo=0.35, valve_hi=0.60, mode="adaptive"):
        self.alpha = alpha
        self.valve_lo = valve_lo
        self.valve_hi = max(valve_hi, valve_lo + 1e-6)
        self.min_alpha = min_alpha
        self.diff_lo = diff_lo
        self.diff_hi = max(diff_hi, diff_lo + 1e-6)
        self.mode = mode
        self.prev = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return depth
        torch = _t()
        d = _to_t(depth)
        if self.prev is None or self.prev.shape != d.shape:
            self.prev = d.clone()
            return depth
        p = self.prev
        if self.mode in ("fixed+valve", "adaptive"):
            lo = self.valve_lo if self.mode == "fixed+valve" else self.diff_lo
            hi = self.valve_hi if self.mode == "fixed+valve" else self.diff_hi
            a = torch.sub(d, p)
            a = a.abs_()
            a = a.sub_(lo).div_(hi - lo).clamp_(0.0, 1.0)
            a = a.mul_(self.min_alpha - self.alpha).add_(self.alpha)
            p.mul_(a)
            tmp = a.neg().add_(1.0).mul_(d)
            p.add_(tmp)
        else:
            p.mul_(self.alpha).add_(d * (1.0 - self.alpha))
        return _to_np(p)


def make_stabilizer(**kw):
    """工厂：按 D2S_BACKEND 返回 numpy 或 torch 实现的 EMA。"""
    if BACKEND == "numpy":
        return StabilizerNumpy(**kw)
    return StabilizerTorch(**kw)


# ============================================================================
# 11. 双线性放大（depth.py predict_depth _bilinear_upsample line 2430-2437）
#     S3 之后此操作交给 shader（深度保持 224，采样时双线性放大），CPU 不再做。
# ============================================================================
def bilinear_upsample(d_hw_f32: np.ndarray, out_hw) -> np.ndarray:
    if d_hw_f32.shape == tuple(out_hw):
        return d_hw_f32
    if BACKEND == "numpy":
        return cv2.resize(d_hw_f32, (out_hw[1], out_hw[0]),
                          interpolation=cv2.INTER_LINEAR)
    torch = _t()
    t = _to_t(d_hw_f32)[None, None]
    t = torch.nn.functional.interpolate(t, size=tuple(out_hw), mode="bilinear",
                                        align_corners=False)
    return _to_np(t[0, 0])


# ============================================================================
# 12. CHW<->HWC 布局工具（替代 permute/contiguous 链）
# ============================================================================
def chw_to_hwc(x_chw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x_chw.transpose(1, 2, 0))
