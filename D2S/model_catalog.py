# -*- coding: utf-8 -*-
"""深度模型目录：MODEL_MAPPING / DEPTH_MODEL_FILES / 解析。

从 utils.py 拆出（2026-11-16）。resolve 的默认档位由 utils 的包装
注入（原实现读 utils 的 globals()）。
"""
import os

# Model Mapping Dict
MODEL_MAPPING = {
    # 本地 ONNX 档位（由 depth.py 的 OrtDepthModel 加载）
    # 本地 ONNX 档位（内部 id 保持中性）
    "d224": "local/d224",
    "d448": "local/d448",
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

# 本地可用的深度模型文件：显式 (Depth Model, Depth Resolution) -> runtime/ 文件名。
# 文件名带分辨率后缀，禁止 basename 猜测（审阅阻断 6）。值为 None = 该组合未导出。
DEPTH_MODEL_FILES = {
    ("d224", 224): "depth-224.onnx",
    ("d448", 448): "depth-448.onnx",
    ("m4s", 448): "depth-m4s-448.onnx",
    ("m4m", 224): None,
    ("m4m", 448): "depth-m4m-448.onnx",
    ("dt", 224): "depth-tiny-224.onnx",
}


def available_depth_models(resolution=None):
    """当前 Depth Resolution 下有本地 ONNX 的模型名列表（设置界面下拉用）。

    在 MODEL/DEPTH_RESOLUTION 定义之前调用时只按显式 resolution 过滤。
    """
    # 默认档位由 utils 的兼容包装注入（原实现读 utils.globals()）
    res = int(resolution) if resolution is not None else None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for (m, r), fname in DEPTH_MODEL_FILES.items():
        if fname is None:
            continue
        if res is not None and r != res:
            continue
        if any(os.path.isfile(os.path.join(root, sub, fname))
               for sub in ("models", "runtime")):
            out.append(m)
    return out


def resolve_depth_model_onnx(model=None, resolution=None):
    """(Depth Model, Depth Resolution) -> runtime ONNX 绝对路径；无对应文件返回 None。"""
    m = model if model is not None else globals().get('MODEL')
    r = resolution if resolution is not None else globals().get('DEPTH_RESOLUTION')
    if m is None or r is None:
        return None
    fname = DEPTH_MODEL_FILES.get((m, int(r)))
    if not fname:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sub in ("models", "runtime"):
        path = os.path.join(root, sub, fname)
        if os.path.isfile(path):
            return path
    return None

