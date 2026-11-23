# -*- coding: utf-8 -*-
"""深度模型目录：MODEL_MAPPING / DEPTH_MODEL_FILES / 解析。

从 utils.py 拆出（2026-11-16）。resolve 的默认档位由 utils 的包装
注入（原实现读 utils 的 globals()）。
"""
import os

# ---------------------------------------------------------------------------
# 本地 ONNX 模型注册表（唯一来源）——2026-11-24 收口
# 几个模型只是三样东西不同：文件、输入尺寸 (W,H)、预处理方式；后面的管线
# （prep → DML → 深度链 → 显示）完全一致。加模型 = 加一行。
# 模型的输出值域**不在这里写**：直方图量程每帧按 raw 的 min/max 自动取
# （D3DStats 的 minmax pass），所以模型输出什么尺度都不用配。
#   label      设置面板显示名
#   file       ONNX 文件名（models/ 或 runtime/ 下）
#   size       模型输入 (W, H)
#   prep_norm  True：prep 做 ImageNet mean/std（模型要外部归一化）
#              False：原样 0..1 RGB（mean/std 在 ONNX 图里，再归一化=双重标准化）
#   license    权重许可（随包分发时写 NOTICE）
# ---------------------------------------------------------------------------
LOCAL_MODELS = {
    "d224": dict(label="DepthART 224",
                 file="depth-224.onnx", size=(224, 224),
                 prep_norm=True, license="CC BY-NC 4.0"),
    "d448": dict(label="DepthART 448",
                 file="depth-448.onnx", size=(448, 448),
                 prep_norm=True, license="CC BY-NC 4.0"),
    "zip384": dict(label="ZipDepth 384",
                   file="zipdepth-384.onnx", size=(384, 384),
                   prep_norm=False, license="MIT"),
    "zipw512": dict(label="ZipDepth 宽屏 512×288",
                    file="zipdepth_wide_512x288.onnx", size=(512, 288),
                    prep_norm=False, license="MIT"),
    "zipw672": dict(label="ZipDepth 宽屏 672×384",
                    file="zipdepth_wide_672x384.onnx", size=(672, 384),
                    prep_norm=False, license="MIT"),
}

# 产品深度模型（2026-11-24 用户定案："ZipDepth 宽屏 512，就这个了"）。
# 设置面板不再提供模型选择，utils.MODEL 固定取它、忽略 yaml 里的旧键；
# 开发/对比实验用环境变量 D2S_DEPTH_MODEL 临时切换到注册表里的其它键。
PRODUCT_MODEL = "zipw512"

# Model Mapping Dict
MODEL_MAPPING = {
    # 本地 ONNX 档位：由注册表生成（内部 id = local/<key>）
    **{k: f"local/{k}" for k in LOCAL_MODELS},
    # 旧本地档位（未随包，保留映射兼容老 yaml）
    "m4s": "local/m4s",
    "m4m": "local/m4m",
    # 轻量蒸馏档（ViT-Tiny，约 6.2M 参数）。
    # 2026-11-08 实测（1650 CUDA，224²，真实照片）：21.7-22.5ms ≈ d224 的
    # 2.5 倍，且样例中把近处物体判成"远"（物体级错误）；大结构平滑
    # 但速度不达标，留作可选档。ONNX: models/depth-tiny-224.onnx，
    # 用 D2S_ONNX 指向即可；输出大=远，predict_depth 已按此翻转。
    "dt": "local/dt",
    # Depth-Anything V2
    "Depth-Anything-V2-Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth-Anything-V2-Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth-Anything-V2-Large": "depth-anything/Depth-Anything-V2-Large-hf",

    # InfiniDepth
    "InfiniDepth-Small": "lc700x/InfiniDepth-Small",
    "InfiniDepth-SmallPlus": "lc700x/InfiniDepth-SmallPlus",
    "InfiniDepth-Base": "lc700x/InfiniDepth-Base",
    "InfiniDepth-Large": "lc700x/InfiniDepth-Large",
    
    # Video-Depth-Anything
    "Video-Depth-Anything-Small": "depth-anything/Video-Depth-Anything-Small",
    "Video-Depth-Anything-Base": "depth-anything/Video-Depth-Anything-Base",
    "Video-Depth-Anything-Large": "depth-anything/Video-Depth-Anything-Large",
    
    # DA3
    "DA3-SMALL": "depth-anything/DA3-SMALL",
    "DA3-BASE": "depth-anything/DA3-BASE",
    # "DA3-LARGE": "depth-anything/DA3-LARGE",
    # "DA3-GIANT": "depth-anything/DA3-GIANT",
    "DA3-LARGE": "depth-anything/DA3-LARGE-1.1",
    "DA3-GIANT": "depth-anything/DA3-GIANT-1.1",
    "DA3METRIC-LARGE": "depth-anything/DA3METRIC-LARGE",
    # "DA3NESTED-GIANT-LARGE": "depth-anything/DA3NESTED-GIANT-LARGE",
    "DA3NESTED-GIANT-LARGE": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "DA3MONO-LARGE": "depth-anything/DA3MONO-LARGE",
    
    # Depth-Anything-V2 Metric Outdoor
    "Depth-Anything-V2-Metric-Outdoor-Small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth-Anything-V2-Metric-Outdoor-Base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Depth-Anything-V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    
    # Depth-Anything-V2 Metric Indoor
    "Depth-Anything-V2-Metric-Indoor-Small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Depth-Anything-V2-Metric-Indoor-Base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Depth-Anything-V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    
    # Metric-Video-Depth-Anything
    "Metric-Video-Depth-Anything-Small": "depth-anything/Metric-Video-Depth-Anything-Small",
    "Metric-Video-Depth-Anything-Base": "depth-anything/Metric-Video-Depth-Anything-Base",
    "Metric-Video-Depth-Anything-Large": "depth-anything/Metric-Video-Depth-Anything-Large",
    
    # LiheYoung/depth-anything
    "depth-anything-small": "LiheYoung/depth-anything-small-hf",
    "depth-anything-base": "LiheYoung/depth-anything-base-hf",
    "depth-anything-large": "LiheYoung/depth-anything-large-hf",
    "depth-anything-indoor-large": "lc700x/depth-anything-indoor-large-hf",
    "depth-anything-outdoor-large": "lc700x/depth-anything-outdoor-large-hf",

    # Distill-Any-Depth
    "Distill-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
    "Distill-Any-Depth-Base": "lc700x/Distill-Any-Depth-Base-hf",
    "Distill-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
    
    # DPT-DINOv2 KITTI
    "dpt-dinov2-small-kitti": "facebook/dpt-dinov2-small-kitti",
    "dpt-dinov2-base-kitti": "lc700x/dpt-dinov2-base-kitti-hf",
    "dpt-dinov2-large-kitti": "lc700x/dpt-dinov2-large-kitti-hf",
    "dpt-dinov2-giant-kitti": "lc700x/dpt-dinov2-giant-kitti-hf",
    
    # DPT-DINOv2 NYU
    "dpt-dinov2-small-nyu": "lc700x/dpt-dinov2-small-nyu-hf",
    "dpt-dinov2-base-nyu": "lc700x/dpt-dinov2-base-nyu-hf",
    "dpt-dinov2-large-nyu": "lc700x/dpt-dinov2-large-nyu-hf",
    "dpt-dinov2-giant-nyu": "facebook/dpt-dinov2-giant-nyu",
    
    # Other
    "depth-ai": "lc700x/depth-ai-hf",
    "dpt-hybrid-midas": "lc700x/dpt-hybrid-midas-hf",
    
    # Intel/DPT
    "dpt-beit-base-384": "Intel/dpt-beit-base-384",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
    "dpt-large": "Intel/dpt-large",
    "dpt-large-redesign": "lc700x/dpt-large-redesign-hf",
    
    # Intel/ZoeDepth
    "zoedepth-nyu-kitti": "Intel/zoedepth-nyu-kitti",
    "zoedepth-nyu": "Intel/zoedepth-nyu",
    "zoedepth-kitti": "Intel/zoedepth-kitti",

    # Apple/DepthPro
    "DepthPro-Large": "apple/DepthPro-hf"
}

# ---- 由注册表派生的兼容表（旧调用方仍按这些名字取）-----------------------
# prep 不能做 ImageNet 归一化的模型（mean/std 在图内）
PREP_RAW_MODELS = {f"local/{k}" for k, s in LOCAL_MODELS.items()
                   if not s["prep_norm"]}
# 输入尺寸 (W, H)；不在表里的模型按 Depth Resolution × 同值（方形）
MODEL_SIZES = {f"local/{k}": tuple(s["size"]) for k, s in LOCAL_MODELS.items()}

# 本地模型文件：注册表生成 + 旧档位。键 (模型, 输入高)——老代码按
# (Depth Model, Depth Resolution) 查；值为 None = 该组合未导出。
DEPTH_MODEL_FILES = {
    **{(k, s["size"][1]): s["file"] for k, s in LOCAL_MODELS.items()},
    ("m4s", 448): "depth-m4s-448.onnx",
    ("m4m", 224): None,
    ("m4m", 448): "depth-m4m-448.onnx",
    # 轻量蒸馏档：输出大=远（方向反），未进注册表/面板
    ("dt", 224): "depth-tiny-224.onnx",
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_file_exists(fname):
    return any(os.path.isfile(os.path.join(_ROOT, sub, fname))
               for sub in ("models", "runtime"))


def local_model_spec(key):
    """注册表 key → spec dict；不在注册表返回 None。"""
    return LOCAL_MODELS.get(key)


def model_spec(model=None):
    """当前模型的 spec：model 省略时读 utils.MODEL（经 globals 兜底）。"""
    m = model if model is not None else globals().get("MODEL")
    return LOCAL_MODELS.get(m)


def available_local_models():
    """注册表里有 ONNX 文件的档位 key 列表（设置面板下拉用，按注册顺序）。"""
    return [k for k, s in LOCAL_MODELS.items() if _model_file_exists(s["file"])]


def available_depth_models(resolution=None):
    """当前 Depth Resolution 下有本地 ONNX 的模型名列表（旧接口，保留）。

    注册表模型不再看 resolution（尺寸随模型走）；旧档位仍按显式
    resolution 过滤。
    """
    res = int(resolution) if resolution is not None else None
    out = []
    for (m, r), fname in DEPTH_MODEL_FILES.items():
        if not fname:
            continue
        if m not in LOCAL_MODELS and res is not None and r != res:
            continue
        if _model_file_exists(fname):
            out.append(m)
    return out


def resolve_depth_model_onnx(model=None, resolution=None):
    """模型 → ONNX 绝对路径（models/ 或 runtime/）；无文件返回 None。

    注册表模型：文件随模型走，忽略 resolution（兼容老签名）；
    旧档位：按 (model, resolution) 查 DEPTH_MODEL_FILES。
    """
    m = model if model is not None else globals().get('MODEL')
    r = resolution if resolution is not None else globals().get('DEPTH_RESOLUTION')
    if m is None:
        return None
    spec = LOCAL_MODELS.get(m)
    if spec is not None:
        fname = spec["file"]
    else:
        if r is None:
            return None
        fname = DEPTH_MODEL_FILES.get((m, int(r)))
    if not fname:
        return None
    for sub in ("models", "runtime"):
        path = os.path.join(_ROOT, sub, fname)
        if os.path.isfile(path):
            return path
    return None

