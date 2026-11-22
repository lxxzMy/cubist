# -*- coding: utf-8 -*-
"""深度管线（去 torch 版，S3）。

torch 已从热路径移除：颜色转换 / 缩放 / 深度后处理 / EMA 全部走
d2s_ops（numpy + cv2，逐 op 与原 torch 实现对拍 <1 LSB，见 ab_ops_check /
ab_pipeline），推理走 ONNX Runtime CUDA（numpy 直进直出，不再经过
torch tensor 的 host<->device 往返）。

模型只支持本项目的 ONNX 家族（settings.yaml 的 d*/m4* 档位；ONNX 文件由
D2S_ONNX 指定，CUDA/cuDNN DLL 由 ../runtime 提供）。

深度图不再在 CPU 放大：predict_depth 返回模型分辨率（224²）的 HW
float32，立体渲染由 shader 采样时双线性放大（viewer 的 u_depth_resolution）。
"""
import os
import time
import threading
import warnings

import numpy as np
import cv2

from utils import (MODEL_ID, DEPTH_RESOLUTION, AA_STRENGTH,
                   CAPTURE_MODE, DEBUG, USE_JBU, DEPTH_UPSAMPLER, DEPTH_EDGE_SNAP,
                   DEPTH_SMOOTHING, DEPTH_SMOOTHING_FRAMES,
                   INFER_BACKEND, DEPTH_VALVE_LO, DEPTH_TREND_GAIN)
import d2s_ops

warnings.filterwarnings("ignore", category=UserWarning)

# ---- CUDA/cuDNN DLL 目录（启动脚本已设 PATH，这里兜底任意启动方式） ----
_RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runtime")
if os.path.isdir(_RUNTIME):
    os.environ["PATH"] = _RUNTIME + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_RUNTIME)
    except (AttributeError, OSError):
        pass


# ============================================================================
# 模型输入几何
# ============================================================================
def get_patch_size():
    """固定方形输入 -> None（predict_depth 走方形缩放分支）。"""
    if CAPTURE_MODE == "Window":
        return None
    if "infinidepth" in MODEL_ID.lower():
        return 16
    if "da3" in MODEL_ID.lower() or "any" in MODEL_ID.lower() or "dinov2" in MODEL_ID.lower():
        return 14
    if MODEL_ID.lower().endswith("dt"):
        # 轻量蒸馏档（ViT-Tiny）：模型 native 用 patch 对齐长边缩放，
        # 但该路径是去 torch 后的占位，且导出的 ONNX 是固定 224×224 方形
        # （224=16×14 ✓）。与 d224 同走方形缩放，保证 A/B 同待遇。
        # 该档位名不含其他档位关键字，勿与上面合并。
        return None
    return None


# ============================================================================
# ONNX 引擎（numpy 直进直出）
# ============================================================================
class OrtDepthModel:
    """ONNX Runtime GPU engine.

    输入 [3,H,W] float32（ImageNet 归一化），输出 [H,W] float32 disparity
    （近=高）。相比去 torch 前的实现：删除了 448 IOBinding 分支与
    torch tensor 的 host<->device 往返 —— d224 本来就走 numpy 便携分支，
    现在那是唯一分支。
    """

    def __init__(self, onnx_path=None):
        import onnxruntime as _ort
        path = onnx_path or _resolve_onnx_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Depth ONNX not found: {path} (set D2S_ONNX to override)")
        self.onnx_path = path
        self._lock = threading.Lock()

        backend = (INFER_BACKEND or "auto").lower()
        _pascal = _is_pascal_gpu()
        trt_ok = _trt_runtime_available() if backend in ("tensorrt", "auto") else False
        if trt_ok and _pascal:
            print("[TRT] 检测到 10 系（Pascal）显卡：TensorRT 10.x 已移除该架构支持，"
                  "已跳过 TRT", flush=True)
            trt_ok = False
        cache_hit = _trt_cache_hit(path) if trt_ok else False

        # 显式选 TensorRT 但运行时缺失 → fail-loud，不静默回落（TRT加速计划 §3.1）
        # 10 系是硬件限制，不 raise，直接回落 CUDA EP 并说明原因。
        if backend == "tensorrt" and not trt_ok:
            if _pascal:
                print("[TRT] 推理后端=TensorRT，但当前是 10 系（Pascal）显卡："
                      "TRT 10.x 已不再支持，本次回落 CUDA EP。"
                      "建议把推理后端改为 auto/cuda。", flush=True)
            else:
                raise RuntimeError(
                    "[TRT] 推理后端=TensorRT 但运行时缺失（nvinfer_10.dll 不在 PATH/"
                    "runtime\\trt\\bin）。请安装 TRT 组件或把后端改回 auto/cuda。")

        trt_providers = [("TensorrtExecutionProvider", _trt_opts(path)),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        # cuDNN 选项：建 session 时穷举卷积算法（比默认启发式快约 8%），
        # 并允许 cuDNN 用更大 workspace 以启用更快的算法。代价是首次建会话慢几秒、
        # 多占一点显存；这两个都是"一次付出、长期收益"。
        cuda_providers = [("CUDAExecutionProvider", {
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
        }), "CPUExecutionProvider"]
        # DirectML：不依赖任何 CUDA/cuDNN 库（发布包可小一个数量级），
        # 224 上实测 4.1ms，输出与 CUDA 数值一致（MAE 1e-6）。
        # 显式指定 device_id：多卡机器（核显 + 独显）上固定到"和 GL 同一块 /
        # 显存最大"的 adapter，别让 DML 跑到核显上。Python API 不能传 D3D12
        # 设备/队列（DirectML #535），device_id 是唯一可用的设备选择手段。
        _dml_dev_id = None
        try:
            from gpu import placement as _pl
            _dml_dev_id = _pl.inference_device_id()
        except Exception:
            _dml_dev_id = None
        if _dml_dev_id is not None:
            dml_providers = [("DmlExecutionProvider", {"device_id": _dml_dev_id}),
                             "CPUExecutionProvider"]
        else:
            dml_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

        if backend == "tensorrt" and trt_ok:
            # 先用 CUDA EP 出画面（首次 engine 构建 18s+ 不能挡启动），
            # 后台线程构建 TRT engine，构建成功后原子切换（TRT加速计划 §3.2）
            self.session = _ort.InferenceSession(path, providers=cuda_providers)
            self.input_name = self.session.get_inputs()[0].name
            self._trt_ready = False
            threading.Thread(target=self._build_trt_session, daemon=True).start()
        elif backend == "auto" and trt_ok and cache_hit:
            # auto：有运行时且 engine 缓存命中才直接用 TRT（无缓存命中不触发
            # 构建，保持默认行为不变——TRT加速计划 §3.4）
            self.session = _ort.InferenceSession(path, providers=trt_providers)
            self.input_name = self.session.get_inputs()[0].name
            self._trt_ready = "TensorrtExecutionProvider" in self.session.get_providers()
            if not self._trt_ready:
                print("[TRT] auto: TRT EP 未生效，已回落 CUDA EP", flush=True)
        elif backend == "dml":
            try:
                self.session = _ort.InferenceSession(path, providers=dml_providers)
            except Exception as _de:
                if _dml_dev_id is None:
                    raise
                print(f"[DML] device_id={_dml_dev_id} 建会话失败（{_de}），"
                      f"回退默认设备", flush=True)
                self.session = _ort.InferenceSession(
                    path, providers=["DmlExecutionProvider",
                                     "CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self._trt_ready = False
            if "DmlExecutionProvider" not in self.session.get_providers():
                print("[DML] DmlExecutionProvider 不可用，已回落 CPU", flush=True)
            elif _dml_dev_id is not None:
                print(f"[DML] device_id={_dml_dev_id}（已指定显卡）", flush=True)
        else:
            self.session = _ort.InferenceSession(path, providers=cuda_providers)
            self.input_name = self.session.get_inputs()[0].name
            self._trt_ready = False

        want = self.session.get_inputs()[0].shape
        if len(want) == 4 and isinstance(want[-1], int) and want[-1] != DEPTH_RESOLUTION:
            print(f"[Depth] warning: session expects {want[-1]}, "
                  f"settings Depth Resolution = {DEPTH_RESOLUTION}")
        print(f"[Depth] ONNX Runtime engine loaded: {path} "
              f"(providers: {self.session.get_providers()})")

    def _build_trt_session(self):
        """后台构建 TRT engine：成功则原子切换 self.session；失败大声记录、
        保持 CUDA EP（显式选择时的失败也只记日志——进程已用 CUDA 出画）。"""
        import onnxruntime as _ort
        try:
            print("[TRT] 后台构建 engine（首次 18s+，之后走缓存）……", flush=True)
            trt_providers = [("TensorrtExecutionProvider", _trt_opts(self.onnx_path)),
                             "CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = _ort.InferenceSession(self.onnx_path, providers=trt_providers)
            if "TensorrtExecutionProvider" not in sess.get_providers():
                raise RuntimeError("TRT EP 未生效（DLL/版本矩阵问题）")
            with self._lock:
                self.session = sess
                self.input_name = sess.get_inputs()[0].name
                self._trt_ready = True
            print("[TRT] engine 就绪，已切换到 TensorRT", flush=True)
        except Exception as e:
            print(f"[TRT] engine 构建失败，保持 CUDA EP: {type(e).__name__}: {e}",
                  flush=True)

    def __call__(self, x_chw_f32: np.ndarray) -> np.ndarray:
        with self._lock:
            sess, name = self.session, self.input_name
        out = sess.run(None, {name: x_chw_f32[None]})[0]
        return out.reshape(out.shape[-2], out.shape[-1]).astype(np.float32)


# TRT EP 选项（TRT加速计划 §3）：engine 缓存按 模型文件名+模型mtime 建子目录
# （模型更新即失效重建）；GPU 名/驱动/TRT 版本维度由 ORT 自身的缓存文件名承担。
def _trt_opts(onnx_path):
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    try:
        mt = int(os.path.getmtime(onnx_path))
    except OSError:
        mt = 0
    _root = _RUNTIME
    cache_dir = os.path.join(_root, "trt-cache", f"{stem}-{mt:x}")
    os.makedirs(cache_dir, exist_ok=True)
    return {
        # ORT 1.24 的 TRT EP 选项值要 'True'/'False'（'1' 会整个 EP 拒载）
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": cache_dir,
        "trt_fp16_enable": "True",        # 1650 有 FP16 CUDA Core
        "trt_timing_cache_enable": "True",
        "trt_builder_optimization_level": "3",
    }


def _is_pascal_gpu():
    """10 系（Pascal, SM 6.1）判断。

    TensorRT 10.x 已移除 Pascal 支持（8.6 是最后一个支持的版本），所以 10 系
    不再尝试 TRT。用显卡名匹配 GTX 10x0 / GT 1030；拿不到名字时返回 False
    （放行，让运行时自己报错）。
    """
    try:
        import re as _re
        from gpu import adapter as _ad
        a = _ad.pick()
        name = (getattr(a, "name", "") or "") if a is not None else ""
        return bool(_re.search(r"\b(GTX|GT)\s*10\d{2}\b", name, _re.I))
    except Exception:
        return False


def _trt_runtime_available():
    """nvinfer_10.dll 是否可达（PATH 或 runtime\\trt\\bin）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trt_bin = os.path.join(root, "runtime", "trt", "bin")
    if os.path.isdir(trt_bin) and trt_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_bin + ";" + os.environ.get("PATH", "")
    for d in os.environ.get("PATH", "").split(";"):
        d = d.strip('"')
        if d and (os.path.isfile(os.path.join(d, "nvinfer_10.dll"))
                  or os.path.isfile(os.path.join(d, "nvinfer.dll"))):
            return True
    return False


def _trt_cache_hit(onnx_path):
    """engine 缓存目录是否存在且非空（auto 档只在命中时启用 TRT）。"""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    try:
        mt = int(os.path.getmtime(onnx_path))
    except OSError:
        return False
    _root = _RUNTIME
    cache_dir = os.path.join(_root, "trt-cache", f"{stem}-{mt:x}")
    try:
        return os.path.isdir(cache_dir) and any(
            f.endswith((".engine", ".plan")) for f in os.listdir(cache_dir))
    except OSError:
        return False


def _resolve_onnx_path():
    """ONNX 解析顺序：env 覆盖 → settings (Depth Model, Depth Resolution)
    经 DEPTH_MODEL_FILES 表 → 默认 d224。"""
    env = os.environ.get("D2S_ONNX")
    if env:
        return env
    try:
        import utils as _u
        p = _u.resolve_depth_model_onnx()
        if p:
            return p
    except Exception:
        pass
    for _sub in ("models", "runtime"):
        _p = os.path.join(os.path.dirname(_RUNTIME), _sub, "depth-224.onnx")
        if os.path.isfile(_p):
            return _p
    return os.path.join(_RUNTIME, "depth-224.onnx")


class _LazyDepthModel:
    """D3D 主路径不用的 legacy 会话：首次真正调用时才加载。

    D3D 推理走 d2s_dml_shim v2（渲染器 device 上的零拷贝会话），此对象仅供
    GL 回退/离线工具使用。D2S_SKIP_LEGACY_MODEL=1（main 在 d3d 模式下设置）
    时 import 不再白加载一份模型（省 3~5s 启动 + 一份显存）。
    """

    def __init__(self):
        self._m = None

    def _get(self):
        if self._m is None:
            self._m = OrtDepthModel()
        return self._m

    def __call__(self, *a, **k):
        return self._get()(*a, **k)

    def __getattr__(self, name):
        if name == "_m":
            raise AttributeError(name)
        return getattr(self._get(), name)


depth_model = _LazyDepthModel()
# 兼容旧名（ab_pipeline 等工具按 model_wraper 取 session）
model_wraper = type("_ModelWrapperAlias", (), {"model": depth_model})()


def _probe_dml_status():
    """探测 GPU 推理后端是否真的可用，并给出人话原因。

    典型失败：显卡/驱动不支持 D3D12（老卡、远程桌面、虚拟机）→
    DirectML 不可用 → 会话回落到 CPU → GPU 直通每帧失败 → SBS 退化成 2D。
    """
    st = {"backend": (INFER_BACKEND or "auto").lower(), "providers": [],
          "dml_ok": False, "reason": "", "adapter": None, "d3d12": None}
    if os.environ.get("D2S_SKIP_LEGACY_MODEL") == "1":
        # D3D 主路径：推理由 shim v2 自带 ORT+DML 承担，与 legacy 会话无关。
        # 不触碰 depth_model（否则会把懒加载又拉起来）。
        st.update(backend="d3d-shim2", dml_ok=True,
                  reason="D3D 主路径由 d2s_dml_shim v2 推理")
        return st
    try:
        st["providers"] = list(depth_model.session.get_providers())
    except Exception:
        pass
    st["dml_ok"] = bool(st["providers"]) and st["providers"][0] in (
        "DmlExecutionProvider", "CUDAExecutionProvider",
        "TensorrtExecutionProvider")
    if st["dml_ok"]:
        return st
    try:
        import onnxruntime as _ort
        avail = _ort.get_available_providers()
        if "DmlExecutionProvider" not in avail and st["backend"] in ("dml", "auto"):
            st["reason"] = "onnxruntime 未提供 DmlExecutionProvider（安装/打包缺组件）"
        else:
            st["reason"] = ("推理后端回落到 "
                            f"{st['providers'][0] if st['providers'] else '未知'}")
    except Exception as e:
        st["reason"] = f"onnxruntime 探测失败：{e}"
    try:
        from gpu import adapter as _ad
        _adv = _ad.pick()
        ok, hr, name = _ad.d3d12_support(_adv)
        st["d3d12"] = bool(ok)
        st["adapter"] = name
        if _adv is not None and not ok:
            _hr = "?" if hr is None else hex(hr)
            st["reason"] += (f"；{name} 创建 D3D12 设备失败（hr={_hr}），"
                             f"DirectML 需要 DirectX 12 支持（显卡/驱动太旧？）")
    except Exception:
        pass
    return st


DML_STATUS = _probe_dml_status()
try:
    if DML_STATUS["providers"]:
        _ok = "GPU 可用" if DML_STATUS["dml_ok"] else "不可用"
        print(f"[DML] 后端={DML_STATUS['backend']} "
              f"providers={DML_STATUS['providers']} → {_ok}", flush=True)
        if not DML_STATUS["dml_ok"]:
            print(f"[DML] 原因：{DML_STATUS['reason']}", flush=True)
except Exception:
    pass

_warmup_done = False


def _warmup(steps: int = 3):
    global _warmup_done
    if _warmup_done:
        return
    dummy = np.zeros((3, DEPTH_RESOLUTION, DEPTH_RESOLUTION), np.float32)
    for _ in range(steps):
        depth_model(dummy)
    _warmup_done = True


# ============================================================================
# 帧预处理：BGRA/BGR -> RGB，缩到工作分辨率（全程 uint8）
# ============================================================================
def process(img_bgrx: np.ndarray, target_height: int) -> np.ndarray:
    """capture 的 BGRA/BGR numpy -> HWC uint8 RGB（工作分辨率）。"""
    return d2s_ops.resize_work(d2s_ops.bgra_to_rgb(img_bgrx), target_height)


# ============================================================================
# 时间平滑（参数语义与原 DepthStabilizer 一致，实现走 d2s_ops）
# ============================================================================
# "fixed" 以时间常数（帧数）参数化：alpha = 0.37 ** (1 / frames)
#   10 帧 -> 0.90（原版）   5 帧 -> 0.82   3 帧 -> 0.72
_FIXED_FRAMES = max(1.0, float(DEPTH_SMOOTHING_FRAMES))
_FIXED_ALPHA = 0.37 ** (1.0 / _FIXED_FRAMES)

depth_stabilizer = d2s_ops.make_stabilizer(
    alpha=0.99 if DEPTH_SMOOTHING == "adaptive" else _FIXED_ALPHA,
    min_alpha=0.30,
    diff_lo=0.12,
    diff_hi=0.32,
    # 阈值经 settings "Depth Valve Threshold" 可调（默认 0.35）：
    # 近似场景差值 < 阈值走全平滑，只有真镜头切换（差值 > 阈值）快跟随
    valve_lo=DEPTH_VALVE_LO,
    valve_hi=max(0.60, DEPTH_VALVE_LO + 0.05),
    mode=DEPTH_SMOOTHING,
)

_IS_METRIC = any(k in MODEL_ID.lower() for k in ('metric', 'kitti', 'nyu', 'depth-ai', 'da3'))


def is_metric():
    return _IS_METRIC


# ============================================================================
# 深度推理 + 后处理（模型分辨率输出，不放大）
# ============================================================================
# ---- 固定标定（方案 A）--------------------------------------------------
# 逐帧 percentile 归一化会让同一物体在不同帧映射到不同的 0..1 值（实测跨度
# 在 2.24~4.45 间变化，即"深度呼吸"）。改用固定区间，跨帧绝对稳定。
# 默认值由 40 段壁纸视频 / 200 帧统计得出（p1 中位 1.22、p99 中位 5.34）。
#   D2S_DEPTH_FIXED=0     关闭，退回逐帧 percentile
#   D2S_DEPTH_FIXED_LO/HI 覆盖区间
try:
    from utils import (DEPTH_FIXED as _DF, DEPTH_FIXED_LO as _DFL,
                       DEPTH_FIXED_HI as _DFH)
except Exception:
    _DF, _DFL, _DFH = True, 1.22, 5.34
# 每个模型的原始深度范围差异可达 8 个数量级（d22x: 0.4~5，m4m: 1e7~1e8，
# dt: 0.9~1.05），一套固定区间不可能通用 —— 按模型名匹配。
_DEPTH_FIXED_RANGES = (
    # 必须 (模型名关键字, 输入分辨率) 同时匹配：224 与 448 的原始分布差别显著，
    # 混用会把中位数映射到完全不同的位置（用户实测：物体全凸 或 全凹）。
    ("d448", 448, (1.22, 5.34)),   # p1 1.219 / p50 2.956 / p99 5.342
    ("d224", 224, (0.20, 5.40)),   # 宽静态区间：覆盖实拍动画(min 0.21)与壁纸(p99 5.23)
    ("m4m", 448, (4.4e7, 1.01e8)),
    ("m4s", 448, (4.4e7, 1.01e8)),
    ("dt", 224, (0.926, 1.045)),
)


def _pick_fixed_range(model_id: str):
    """按 (模型名, 输入分辨率) 双键匹配；分辨率不符则返回 None（退回逐帧自适应）。"""
    mid = (model_id or "").lower()
    try:
        res = int(globals().get("DEPTH_RESOLUTION", 0))
    except Exception:
        res = 0
    for key, want_res, rng in _DEPTH_FIXED_RANGES:
        if key in mid and int(want_res) == res:
            return rng
    return None


_DEPTH_FIXED_ON = os.environ.get("D2S_DEPTH_FIXED", "1" if _DF else "0") != "0"
_DEPTH_FIXED_LO = float(os.environ.get("D2S_DEPTH_FIXED_LO", str(_DFL)))
_DEPTH_FIXED_HI = float(os.environ.get("D2S_DEPTH_FIXED_HI", str(_DFH)))


# 最近一帧"EMA 之前"的深度中位数（0..1，近=1）。自动收敛面用它，
# 避免与逐像素 EMA 形成二次平滑导致跟踪过慢。
LAST_MEDIAN = 0.5
LAST_P30 = 0.35   # 30 分位：自动收敛面用它（比中位数低 → 整体凸出）


# 固定区间也自适应：慢速跟踪每帧的 p1/p99，避免"宽动态影片被截断"。
# 时间常数大（8s）→ 不会像逐帧归一化那样呼吸，但能跟上片源切换。

# ============================================================================
# Holt 双指数（level + trend）：把"位置"和"变化速度"分开估
#   抖动 = 高频零均值 → trend 估出≈0，输出被平均掉
#   平移 = 持续同向 → trend 跟住速度，输出几乎无滞后
# 因果（只看过去帧），每像素 4 次乘加，224² 约 0.1ms
# ============================================================================
_HOLT = {"lvl": None, "trd": None, "n": 0}
_HOLT_STATS = {"n": 0, "jit": 0.0, "prev": None, "prev2": None}
_HOLT_LOG = os.environ.get("D2S_DEPTH_STATS", "0") == "1"


# ---- 自动标定（方案 B）：p1/p99 慢跟随 -------------------------------------
# 纯逐帧 p1/p99 会呼吸（上面的注释就是当年的实测），所以这里给标定值本身
# 加一个 3 秒量级的 EMA：画面内容缓慢变化时跟上，逐帧噪声被吃掉。
_AUTO_RANGE = {"lo": None, "hi": None, "t": 0.0}
try:
    from utils import (DEPTH_AUTO_RANGE as _DAR, DEPTH_AUTO_RANGE_TAU as _DART,
                       DEPTH_NORM_MODE as _DNM,
                       DEPTH_PCT_LOW, DEPTH_PCT_HIGH)
except Exception:
    _DAR, _DART, _DNM = True, 3.0, "da3"
    DEPTH_PCT_LOW, DEPTH_PCT_HIGH = 0.5, 99.5
_NORM_MODE = _DNM
_AUTO_RANGE_ON = os.environ.get("D2S_AUTO_RANGE", "1" if _DAR else "0") != "0"
_AUTO_RANGE_TAU = float(os.environ.get("D2S_AUTO_RANGE_TAU", str(_DART)))


def _auto_range_update(raw, now=None):
    """返回 (lo, hi)：p1/p99 的慢跟随版本。首帧直接对齐。"""
    import time as _t
    now = _t.perf_counter() if now is None else now
    lo_f = float(np.percentile(raw, 0.5))
    hi_f = float(np.percentile(raw, 99.5))
    if hi_f - lo_f < 1e-6:
        hi_f = lo_f + 1e-6
    st = _AUTO_RANGE
    if st["lo"] is None or st["hi"] is None:
        st["lo"], st["hi"], st["t"] = lo_f, hi_f, now
        return st["lo"], st["hi"]
    dt = max(1e-3, min(1.0, now - st["t"]))
    a = 1.0 - np.exp(-dt / max(0.1, _AUTO_RANGE_TAU))
    # 区间必须保持有效（lo < hi），并且不允许反转
    new_lo = st["lo"] + (lo_f - st["lo"]) * a
    new_hi = st["hi"] + (hi_f - st["hi"]) * a
    if new_hi - new_lo < 1e-6:
        new_hi = new_lo + 1e-6
    st["lo"], st["hi"], st["t"] = new_lo, new_hi, now
    return new_lo, new_hi


def _holt_apply(d, a, b, valve=0.35):
    st = _HOLT
    if st["lvl"] is None or st["lvl"].shape != d.shape:
        st["lvl"] = d.copy()
        st["trd"] = np.zeros_like(d)
        st["n"] = 0
        return d
    lvl, trd = st["lvl"], st["trd"]
    pred = lvl + trd
    # 场景切换（大面积偏离预测）→ 直接重置，避免拖影
    if float(np.mean(np.abs(d - pred))) > valve:
        st["lvl"] = d.copy()
        st["trd"] = np.zeros_like(d)
    else:
        new_lvl = a * d + (1.0 - a) * pred
        new_trd = b * (new_lvl - lvl) + (1.0 - b) * trd
        st["lvl"], st["trd"] = new_lvl, new_trd
    st["n"] += 1
    out = st["lvl"]

    return out


# 原样模式：不 gamma / 不 AA / 不 grow / 不 edge_snap / 不平滑 / 不自适应
# 只做 0~6 → 0~1 的固定线性映射（调试"模型原貌"用）
try:
    from utils import DEPTH_RAW_OUTPUT as _DRO
except Exception:
    _DRO = False
_RAW_MODE = [bool(_DRO)]
try:
    from utils import DEPTH_PLAIN_NORM as _DPN
except Exception:
    _DPN = False
_PLAIN_NORM = [bool(_DPN)]


def set_plain_norm(on):
    _PLAIN_NORM[0] = bool(on)
_RAW_LO, _RAW_HI = 0.0, 6.0


def set_raw_mode(on):
    _RAW_MODE[0] = bool(on)



def _da3_normalize(raw):
    """原版 Desktop2Stereo 的做法：2%/98% 分位裁剪 + min-max（逐帧，无 EMA）。

    对应 DA3 官方 visualize_depth：度量模型先取 1/depth，视差模型直接用。
    返回 [H,W] float32，近=1 远=0。
    """
    d = np.asarray(raw, np.float32)
    if _IS_METRIC:
        d = np.where(d > 1e-12, 1.0 / np.maximum(d, 1e-12), d)
    # 与 GPU depth_norm.percentile_stats 同口径 ::4 子采样：P5/P95 的分位
    # 方差比 P0.5/P99.5 大一个量级（p(1-p) 项），全量 vs ::4 曾差 0.0126
    # （parity A3 实锤）——同采样两侧逐位一致
    sub = d[::4, ::4]
    lo = float(np.percentile(sub, DEPTH_PCT_LOW))
    hi = float(np.percentile(sub, DEPTH_PCT_HIGH))
    span = max(1e-6, hi - lo)
    return np.clip((d - lo) / span, 0.0, 1.0).astype(np.float32)



# ============================================================================
# 纯 EMA（与 tools/filter_mae_bench.py / dibr_motion_truth.py 里验证的一致）
#   a = 1 - 0.37^(1/N)：N 帧后旧值权重剩 0.37
#   基准结论：所有场景都优于 Holt（trend 项会放大噪声）
# ============================================================================
_EMA_STATE = {"prev": None}

# 时域中值 W5（2026-11-20 用户定案，产品时域级；与 GPU depth_post.MEDIAN_FS
# 对拍）。播种语义与 GPU 一致：首帧环 5 槽全为首帧 → 帧 2-3 输出旧帧中位、
# 帧 4 起满窗（启动 3 帧收敛）。D2S_TEMPORAL=ema 回退 legacy EMA。
_TEMPORAL = os.environ.get("D2S_TEMPORAL", "ema")
_MED_STATE = {"ring": []}
# 切阀前帧 z（::4 归一化 pre-AA 子样本；判据在 postprocess_depth 内，
# 与 main._gpu_depth_step 同函数 depth_norm.cut_hit——内层旧状态阀已拆，
# 双重阀会让两侧在"切换后第二帧"类场景决策分叉）
_CUT_PREVZ = [None]


def _median_apply(d, w=5):
    st = _MED_STATE
    if len(st["ring"]) != w or st["ring"][0].shape != d.shape:
        st["ring"] = [d.copy() for _ in range(w)]
        return d.copy()
    st["ring"].append(d.copy())
    st["ring"].pop(0)
    return np.median(np.stack(st["ring"]), axis=0).astype(np.float32)


def _ema_apply(d, a):
    """纯 EMA（切阀在 postprocess_depth 上游，此处不再内嵌）。

    a = 1 - 0.37^(1/N)：N 帧后旧值权重剩 0.37。
    """
    st = _EMA_STATE
    if st["prev"] is None or st["prev"].shape != d.shape:
        st["prev"] = d.copy()
        return d
    out = a * d + (1.0 - a) * st["prev"]
    st["prev"] = out
    return out


def _publish_stats(d):
    """把 LAST_MEDIAN / LAST_P30 发布给自动收敛面（平滑之后的深度）。"""
    global LAST_MEDIAN, LAST_P30
    try:
        from depth_norm import near_peak_median
        LAST_MEDIAN = near_peak_median(d)   # 双峰取近景峰中位（与 GPU 同步）
        LAST_P30 = float(np.percentile(d, 30))
    except Exception:
        pass


def postprocess_depth(raw, use_temporal_smooth: bool = True, guide=None):
    """模型原始输出 -> 管线契约深度（后处理链，CPU 侧，GPU/CPU 路径共用）。

    percentile 归一化 -> gamma -> 前景对比度 -> AA -> grow -> 边缘贴合 -> EMA。
    """
    # ---- 原版纯净模式：只做逐帧 P2/P98 归一化，其余全部跳过 ----
    if _PLAIN_NORM[0]:
        d = _da3_normalize(raw)
        _publish_stats(d)
        _depth_stats_update(d)
        return d

    # ---- 原样模式：直接线性映射，后续所有处理全部跳过 ----
    if _RAW_MODE[0]:
        d = np.clip((np.asarray(raw, np.float32) - _RAW_LO) / (_RAW_HI - _RAW_LO), 0.0, 1.0)
        _publish_stats(d)
        _depth_stats_update(d)
        return d

    _rng = _pick_fixed_range(globals().get("MODEL_ID", ""))
    if _NORM_MODE == "da3" and not _IS_METRIC:
        # 原版行为：逐帧 P2/P98 + min-max（无慢跟随、无延迟）
        d = _da3_normalize(raw)
    elif _AUTO_RANGE_ON and not _IS_METRIC:
        # 自动标定：p1/p99 慢跟随（3s），比固定区间自适应、比逐帧稳定
        _lo, _hi = _auto_range_update(raw)
        _span = max(1e-6, _hi - _lo)
        d = np.clip((raw - _lo) / _span, 0.0, 1.0).astype(np.float32)
    elif _DEPTH_FIXED_ON and not _IS_METRIC:
        _lo, _hi = _rng if _rng else (_DEPTH_FIXED_LO, _DEPTH_FIXED_HI)
        _span = max(1e-6, _hi - _lo)
        d = np.clip((raw - _lo) / _span, 0.0, 1.0).astype(np.float32)
    else:
        d = d2s_ops.percentile_normalize(raw, metric=_IS_METRIC)
    # 残差切阀（2026-11-21 CPU 判据）：z_t vs z_{t−1}，归一化 pre-AA 域
    # （与 main._gpu_depth_step 同函数 depth_norm.cut_hit/同采样——parity
    # 逐位）。命中 → 跳过本帧时域（状态=新帧），切换无渐变。
    try:
        from depth_norm import cut_hit as _cut_hit2
        _z = d[::4, ::4]
        _cutnow = _cut_hit2(_z, _CUT_PREVZ[0])
        _CUT_PREVZ[0] = _z
    except Exception:
        _cutnow = False
    # 用户要求删除 gamma / 前景增强 / grow / edge_snap：链路只留
    #   归一化 → AA（可选）→ 时域 EMA
    if not USE_JBU:  # JBU 模式下由引导上采样负责平滑，预模糊会产生光晕
        d = d2s_ops.anti_alias(d, AA_STRENGTH)
    if use_temporal_smooth and not _cutnow:
        if _TEMPORAL == "median":
            d = _median_apply(d)
        elif DEPTH_SMOOTHING == "ema":
            _a = 1.0 - 0.37 ** (1.0 / max(1.0, float(DEPTH_SMOOTHING_FRAMES)))
            d = _ema_apply(d, _a)
        elif DEPTH_SMOOTHING == "holt":
            # 时间常数语义：N 帧后旧值权重剩 0.37 → a = 1 - 0.37^(1/N)
            # （原写法 0.37^(1/N) 是反的：N=10 时 a=0.905 几乎不滤波，
            #  实测抖动只降 12%；修正后同参数降 93%）
            _a = 1.0 - 0.37 ** (1.0 / max(1.0, float(DEPTH_SMOOTHING_FRAMES)))
            _b = max(0.02, float(DEPTH_TREND_GAIN))   # 趋势增益（设置项）
            d = _holt_apply(d, _a, _b, DEPTH_VALVE_LO)
        else:
            d = depth_stabilizer(d)
    if _cutnow:
        # 状态=本帧（下帧从新场景起 EMA/中值满环）——跳过时域但不复位状态
        # 的话，下帧会拿旧场景状态混出新鬼影
        _EMA_STATE["prev"] = d.copy()
        _MED_STATE["ring"] = [d.copy() for _ in range(5)]
    # 统计量必须在【深度平滑之后】采样：平滑前采到的是逐帧噪声，收敛面会
    # 跟着抖（用户实测：p30 在 0.16~0.29 之间来回跳 → 画面整块抽动）。
    _publish_stats(d)
    _depth_stats_update(d)
    return d


def _depth_stats_update(d):
    """诊断：二阶差分均值 = 抖动能量。D2S_DEPTH_STATS=1 时每 120 帧打印。"""
    if not _HOLT_LOG:
        return
    ss = _HOLT_STATS
    if ss["prev"] is not None and ss["prev2"] is not None and ss["prev"].shape == d.shape:
        ss["jit"] += float(np.mean(np.abs(d - 2.0 * ss["prev"] + ss["prev2"])))
        ss["n"] += 1
    ss["prev2"], ss["prev"] = ss["prev"], d.copy()
    if ss["n"] >= 120:
        print(f"[depth-stats] 抖动能量={ss['jit']/ss['n']:.5f} mode={DEPTH_SMOOTHING}", flush=True)
        ss["jit"], ss["n"] = 0.0, 0


def predict_depth(image_rgb, return_tuple=False, use_temporal_smooth: bool = True,
                  dtype=None, _backend_retry: bool = False):
    """HWC uint8 RGB -> HW float32 [0,1] 深度（模型分辨率，近=1 远=0）。

    return_tuple/dtype/_backend_retry 仅为旧签名兼容，忽略。
    阶段顺序镜像原 predict_depth：缩放/归一化 -> ORT -> 百分位归一化 ->
    gamma -> 前景对比度 -> AA -> grow -> 边缘贴合 -> EMA。
    """
    t0 = time.perf_counter() if DEBUG else 0
    _warmup()

    x = d2s_ops.resize_model_input(image_rgb, DEPTH_RESOLUTION, get_patch_size())
    guide = None
    if DEPTH_EDGE_SNAP or USE_JBU:
        # 边缘贴合 / JBU 的引导图 = 未归一化的 0..1 模型输入
        guide = d2s_ops.resize_model_input(image_rgb, DEPTH_RESOLUTION,
                                           get_patch_size(), normalize=False)
    raw = depth_model(x)
    if MODEL_ID.lower().endswith("dt"):
        # 该档位输出大=远；管线契约是 disparity（近=高）。取负后
        # percentile_normalize 的 min-max 自动重标定，方向即翻转。
        raw = -raw

    d = postprocess_depth(raw, use_temporal_smooth=use_temporal_smooth, guide=guide)

    if DEBUG:
        print(f"[depth] predict {1000 * (time.perf_counter() - t0):.1f} ms")
    return d


# === ParallaxRT frame-skip shim ============================================
# Desktop/video depth changes slowly; re-running the full depth pass every
# other frame is visually equivalent and halves the per-frame cost on GPUs
# that can't keep full rate. Tune via D2S_INFER_EVERY (1 = default).
_D2S_INFER_EVERY = max(1, int(os.environ.get("D2S_INFER_EVERY", "1")))
_d2s_parity = 0
_d2s_cached = None

_predict_depth_orig = predict_depth


def predict_depth(image_rgb, return_tuple=False, use_temporal_smooth: bool = True,
                  dtype=None, _backend_retry: bool = False):
    global _d2s_parity, _d2s_cached
    if (_D2S_INFER_EVERY > 1 and not return_tuple
            and MODEL_ID.lower().startswith("local/") and _d2s_cached is not None):
        _d2s_parity = (_d2s_parity + 1) % _D2S_INFER_EVERY
        if _d2s_parity != 0:
            return _d2s_cached
    result = _predict_depth_orig(image_rgb, return_tuple=return_tuple,
                                 use_temporal_smooth=use_temporal_smooth,
                                 dtype=dtype, _backend_retry=_backend_retry)
    if not return_tuple and MODEL_ID.lower().startswith("local/"):
        _d2s_cached = result
        _d2s_parity = 0
    return result
# === end ParallaxRT shim ==================================================


# ============================================================================
# MJPEG 流的 SBS 合成（遗留路径，需要 torch；本地查看器不经过这里）
# ============================================================================
