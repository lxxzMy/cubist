import requests
import yaml, threading, time
import os, platform, socket
import functools
import tempfile

DEBUG = False
# App Version
# 版本号唯一来源：改这里，并同步 tools/installer/d2s.iss 的 MyAppVersion
VERSION = "2.5.0"
# OS_NAME 单源在 platform_probe.py（下方导入）
# Define StereoMix devices
STEREO_MIX_NAMES = [
# English
"stereo mix", "what you hear", "loopback", "system audio", "wave out mix", "mixed output",
# Chinese
"立体声混音", "您听到的声音", "环路", "系统音频", "波形输出混合", "混合输出",
# Japanese
"ステレオ ミックス", "ステレオミックス", "ループバック", "システムオーディオ", "ミックス出力",
# Spanish
"mezcla estéreo", "lo que escuchas", "bucle", "audio del sistema", "salida mixta",
# French
"mixage stéréo", "bouclage", "audio système", "sortie mixte",
# German
"stereomix", "was du hörst", "loopback", "systemaudio", "gemischte ausgabe",
# macOS specific
"blackhole", "loopback", "aggregate device", "multi-output device", "virtual desktop speakers", "remote sound",
# Linux specific
"monitor"
]

# Models with Disabled TRT 
DISABLE_TRT_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    # "video-depth-anything",
]

# Models with Disabled TRT 
DISABLE_MIGRAPHX_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    # "video-depth-anything",
]

TRT_FIX_KEYWORDS = [
    # Intel/zoedepth
    "Intel/zoedepth-nyu-kitti",
]

FORCE_FP32_KEYWORDS = [
    # ZoeDepth models
    "Intel/zoedepth-nyu",
    "Intel/zoedepth-kitti",
]


# Models with Disabled CoreML 
DISABLE_COREML_KEYWORDS = [
    "video-depth-anything",
    "da3-",
    "da3nested",
    "dpt-beit",
    "zoedepth",
    "depthpro",
    "infinidepth",
]

# Models with Disabled OpenVINO 
DISABLE_OPENVINO_KEYWORDS = [
    "dpt-hybrid-midas-hf",
]

# Disable CuDNN for RX 6000 and 5000 series GPUs
DISABLE_CUDNN_GFX = [
    "gfx1030",  # RX 6950, 6900, 6850, 6800 (Navi 21)
    "gfx1031",  # RX 6750, 6700 (Navi 22)
    "gfx1032",  # RX 6650, 6600 (Navi 23)
    "gfx1033",  # Van Goh (Navi 23)
    "gfx1034",  # RX 6550, 6500, 6400, 6300 (Navi 24)
    "gfx1010",  # RX 5700, 5600 (Navi 10)
    "gfx1012",  # RX 5500, 5400, 5300 (Navi 14)
    "gfx1011",  # BC160, Radeon Pro V520 (Navi 12)
]
DISABLE_TRITON_GFX = []

# Global shutdown event
shutdown_event = threading.Event()


# HF 镜像设置（2026-11-08）：模型全为本地 ONNX，HF 下载路径已废弃，
# 静态设置即可（原 is_cn_ip() 同步探测实测吃冷启动 5.1s，已移除）。
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 拆分（2026-11-16）：配置 IO → config.py；平台探测/WDA/设备 →
# platform_probe.py；模型目录 → model_catalog.py。本模块保留
# 常量管线并 re-export 全部旧名（importer 兼容）。
from config import (settings, read_yaml, write_yaml, _quarantine_corrupt_yaml,
                    _SETTINGS_PATH)
from platform_probe import (OS_NAME, get_font_type, get_fps, get_monitor_size,
                           compute_output_resolution, _get_device_name_from_mss_monitor,
                           _get_fps_windows, crop_icon, _gpu_name, get_device,
                           user32, SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
                           hide_window_from_capture, show_window_in_capture)
from model_catalog import (MODEL_MAPPING, DEPTH_MODEL_FILES,
                          available_depth_models)

# Get settings
RUN_MODE = settings["Run Mode"]
# Add for 3D monitor
USE_3D_MONITOR = False

# Add for FrameGen
LOSSLESS_SCALING_SUPPORT = False
MODEL = settings["Depth Model"]
MODEL_ID = MODEL_MAPPING[MODEL]
ALL_MODELS = settings["Model List"]
CACHE_PATH = "./models"
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Computing Device"]
FP16 = settings["FP16"]
# 推理后端：auto / cuda / tensorrt（TRT加速计划 §3.4：auto=有 TRT 运行时且
# engine 缓存命中才启用，否则保持 CUDA EP；显式选 tensorrt 时失败即报错）。
# 与旧 "TensorRT" 键（torch 时代遗留）无关。
INFER_BACKEND = str(settings.get("Infer Backend", "auto")).strip().lower()
MONITOR_INDEX,  DISPLAY_MODE = settings["Monitor Index"], settings["Display Mode"]
# 采集屏（AB 分离）：独立于输出屏 MONITOR_INDEX。缺省 = 输出屏（老配置无感）。
# 零拷贝管线要求两者在同一块显卡上——不同卡时 main 启动会弹框提示。
CAPTURE_MONITOR_INDEX = int(settings.get("Capture Monitor Index",
                                        MONITOR_INDEX) or MONITOR_INDEX)
STEREO_DISPLAY_INDEX = settings.get("Stereo Output")
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
# 同屏自投射：viewer 盖在捕获屏上（Stereo Output 清空或与 Monitor Index 相同）。
# 此时 2D 镜像自己没有意义：2D 隐藏 viewer 原生显示，3D 才显窗转换；
# 指针由真实硬件光标承担（采集绘制会同屏双光标，且画的光标带视差）。
try:
    SAME_SCREEN = (not STEREO_DISPLAY_INDEX) or         int(STEREO_DISPLAY_INDEX) == int(MONITOR_INDEX)
except (TypeError, ValueError):
    SAME_SCREEN = False
OUTPUT_RESOLUTION = compute_output_resolution(
    settings.get("Processing Resolution", "Auto"),
    DISPLAY_MODE,
    MONITOR_INDEX,
    STEREO_DISPLAY_INDEX,
    RUN_MODE,
)
SHOW_FPS = settings["Show FPS"]
IPD = 0.064   # 常量（位移公式相消，已删设置暴露；保留供旧分支 import）
CONVERGENCE = float(settings.get("Convergence", 0.35))   # 自动收敛的起始平面
# 自动收敛面（跟随画面深度中位数）+ 偏置（正=更凸出/更靠前）
CONV_AUTO = True    # 收敛面恒为自动（只剩偏置可调）
CONV_BIAS = float(settings.get("Convergence Bias", 0.0))
# 自动收敛面的跟踪时间常数（秒）：越小跟得越快
# 深度归一化：固定标定（True）或逐帧自适应（False）——设置面板可切换
DEPTH_FIXED = bool(settings.get("Depth Fixed", True))
DEPTH_FIXED_LO = float(settings.get("Depth Fixed Lo", 1.22))
DEPTH_FIXED_HI = float(settings.get("Depth Fixed Hi", 5.34))
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"] if CAPTURE_MODE == "Window" else None
FPS = get_fps(WINDOW_TITLE, MONITOR_INDEX)

# Image Processing Parameters
FOREGROUND_SCALE = settings["Foreground Scale"] / 10 # 0-10
AA_STRENGTH = settings["Anti-aliasing"] * 2

# Experimental Settings





CAPTURE_TOOL = "DesktopDuplication"
FILL_16_9 = settings["Fill 16:9"]
# Interleaved row phase: which row parity the panel's polarizer film maps to
# the left eye (0 = even rows from the top, 1 = swapped). Viewer hotkeys were
# removed - edit the value here, then restart. The renderer applies row-parity
# auto-alignment (window-y parity vs the monitor origin) on top of this base.
# SBS/TAB 需要整屏 1:1 输出（窗口铺满显示器、WDA 排除、显示器捕获）
_SBS_OUTPUT = str(DISPLAY_MODE).strip() in (
    "Full-SBS", "Half-SBS", "Full-TAB", "Half-TAB")
# 全屏输出：SBS/TAB 强制；行交错由设置决定（窗口贴合 or 整屏）
FULLSCREEN_OUTPUT = _SBS_OUTPUT or bool(settings.get("Fullscreen Output", False))
# 产品定案（2026-11-16，用户拍板）：WGC 窗口/皮肤模式退役（相关代码已删，
# 回滚 = git tag pre-refactor）。只保留全屏输出 + DXGI 采集。
if not FULLSCREEN_OUTPUT:
    FULLSCREEN_OUTPUT = True
    print("[Settings] 产品定案：仅全屏输出模式（WGC 窗口/皮肤模式已退役）",
          flush=True)

# 调试专用：D2S_NO_FULLSCREEN=1 强制窗口化（联调开关，产品定案之后评估，
# 只影响显式设置本变量的调试会话，不改变产品默认行为）
if __import__("os").environ.get("D2S_NO_FULLSCREEN") == "1":
    FULLSCREEN_OUTPUT = False
_ip_raw = settings.get("Interleave Phase", 0)
# 面板已改为勾选框：兼容 true/false 与 0/1
INTERLEAVE_PHASE = (1 if (_ip_raw is True or str(_ip_raw).strip().lower() in ("1", "true")) else 0)
# Stereo parallax budget in screen pixels per eye at the ends of the depth
# range (when Convergence = 0.5). One number: the offset is linear in depth.
# Cursor handling: DXGI capture has no cursor in the frame (it lives on a
# separate overlay plane), so it has to be drawn back.
# 不再提供"显示鼠标"开关：按【显示模式】自动决定。
#   行交错/单眼：一只系统光标在两只眼的行里天然正确 → 完全交给系统
#   SBS/TAB   ：真实光标只有 1 个，而两半各要一个 → 空白替换 + 自绘两份
# 采集端合成（原 frame 模式）已废弃：SBS 下必然被复制成两份，非 SBS 也不需要。
_SBS_LIKE = str(settings.get("Display Mode", "")).strip() in (
    "Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB")
CURSOR_MODE = "overlay" if _SBS_LIKE else "off"
DRAW_CURSOR = False
import os as _os_cur
if _os_cur.environ.get("D2S_CURSOR_OFF", "0") == "1":
    # 调试：即使 SBS 也不自绘（用来判断多出来的光标是不是我们画的）
    CURSOR_MODE = "off"
    print("[Settings] D2S_CURSOR_OFF=1 → 不画光标（调试隔离）")
elif CURSOR_MODE == "overlay":
    print(f"[Settings] 光标策略：SBS/TAB → 自绘（模式={settings.get('Display Mode')}）")
else:
    print("[Settings] 光标策略：非 SBS → 交还系统（形状/空闲隐藏原生正确）")
# 自绘光标尺寸已按"系统光标实际可见高度"归一，不再需要手工倍数
CURSOR_SCALE = 1.0
PARALLAX_PX = float(settings.get("Parallax (px)", 7.0))
# Depth -> parallax shape: "linear" (offset proportional to depth) or "legacy"
# (the original non-linear curve, -d*(1+0.35*(1-d)) + C, which pushes the
# near-mid ground further than a straight line does).
PARALLAX_CURVE = str(settings.get("Parallax Curve", "legacy")).strip().lower()
if PARALLAX_CURVE not in ("linear", "legacy"):
    PARALLAX_CURVE = "legacy"
# Depth upsampler: JBU (colour-guided, half-res working grid), BGU (bilateral
# guided upsampling: local affine fit at 224, evaluated on the full-res guide)
# or BILINEAR. Both guided variants skip the low-res gaussian pre-blur.
DEPTH_UPSAMPLER = "BILINEAR"    # 深度上采样恒为双线性（JBU 未实现）
USE_JBU = DEPTH_UPSAMPLER in ("JBU", "BGU")
# Snap the soft depth ramp onto the real image edges (joint bilateral at the
# model resolution). Without it, a few px of parallax drags the background
# colour onto foreground silhouettes (e.g. the green chalkboard onto a cheek).
DEPTH_EDGE_SNAP = bool(settings.get("Depth Edge Snap", False))
# Temporal depth smoothing: fixed (original, alpha 0.9) | fixed+valve | adaptive | off
# Gamma applied to the normalised depth before it reaches the shader.
# 1.45 is the look-oriented default; 1.0 makes the pipeline linear in the
# model's output (which is affine in 1/z), i.e. physically correct disparity.
DEPTH_GAMMA = min(3.0, max(0.1, float(settings.get("Depth Gamma", 1.45))))
# 注意：YAML 1.1 会把 off/on/yes/no 解析成布尔值！
# `Depth Smoothing: off` 读出来是 False，str(False).lower()="false" 不在白名单，
# 于是静默回退成 "fixed"——用户以为关了平滑，实际换了另一种平滑。
DEPTH_SMOOTHING = "ema"    # 固定纯 EMA（其余算法实测都不如它）
# Time constant of the fixed smoother。2026-11-22 定案：固定 80ms，面板已移除，
# 不再从 yaml 读（旧配置里的 Depth Smoothing Tau (ms)/Frames 一律忽略）。
DEPTH_SMOOTHING_TAU_MS = 80.0
DEPTH_SMOOTHING_FRAMES = DEPTH_SMOOTHING_TAU_MS * 60.0 / 1000.0
# fixed+valve 的快跟随触发阈值：像素差低于它走全平滑，高于它快速切新深度。
# 0.35=只放行真镜头切换（近似场景跳变实测治理值，2026-11-10）。
DEPTH_VALVE_LO = min(0.60, max(0.05, float(settings.get("Depth Valve Threshold", 0.35))))
# Holt 趋势增益（b）：越大跟运动越快、也更容易放噪声进来
DEPTH_TREND_GAIN = min(0.60, max(0.05, float(settings.get("Depth Trend Gain", 0.30))))
# 自适应平滑：整体帧间变化达到此值时，alpha 提到 1（立即跟随）
# 0.0 = 关闭自适应（此前误写成 max(0.01,..)，导致设为 0 时变成 0.01 → 每帧都判为大变化 → 完全不平滑）
DEPTH_ADAPT_STEP = min(1.0, max(0.0, float(settings.get("Depth Adaptive Step", 0.0))))
# 深度自动标定：p1/p99 慢跟随（替代固定区间）
DEPTH_AUTO_RANGE = bool(settings.get("Depth Auto Range", True))
DEPTH_AUTO_RANGE_TAU = min(10.0, max(0.5, float(settings.get("Depth Auto Range Tau", 3.0))))
# 深度原样输出：完全不做处理，仅按 0~6 线性归一化（调试对照用）
DEPTH_RAW_OUTPUT = bool(settings.get("Depth Raw Output", False))
# 归一化模式：da3=P2/P98 逐帧 min-max（原版）/ auto=P1/P99 慢跟随 / fixed=固定区间
DEPTH_NORM_MODE = str(settings.get("Depth Normalize Mode", "da3")).strip().lower()
# 原版纯净模式：只做逐帧 P2/P98 归一化，跳过 gamma/AA/grow/edge_snap/时域平滑
DEPTH_PLAIN_NORM = bool(settings.get("Depth Plain Normalize", False))
# 逐帧归一化用的分位（原版是 2/98）
DEPTH_PCT_LOW = min(45.0, max(0.0, float(settings.get("Depth Percentile Low", 0.5))))
DEPTH_PCT_HIGH = min(100.0, max(55.0, float(settings.get("Depth Percentile High", 99.5))))
# 边缘渐隐（DIBR 出框抑制）：true=0.01 宽 smoothstep，false=0 关
_ef = settings.get("Edge Falloff", True)
EDGE_FALLOFF = 0.0 if str(_ef).strip().lower() in ("false", "0", "no") else 0.01
TOGGLE_HOTKEY = str(settings.get("Toggle Hotkey", "ctrl+f2")).strip().lower()
if TOGGLE_HOTKEY not in ("ctrl+f2", "pause", "scrolllock", "f10", "ctrl+f10"):
    TOGGLE_HOTKEY = "ctrl+f2"
# （2026-11-23 删除 DEPTH_GROW / DEPTH_SLOPE_LIMIT / DEPTH_CLOSE：膨胀、
#   坡度限幅、闭运算三步已从深度链整体移除，相关设置键一并删除。）
# ParallaxRT-style silhouette handling (ported from dibr_renderer_cuda.cu):
#   edge snap  - at a strong depth edge, pull the rim pixel's depth onto the
#                background side so it stops sampling across the layer
#   relief     - add local image contrast into the depth so it follows the
#                real edge instead of the model's 30-60 px ramp
#   iterations - one fixed-point refinement of the warp position
DIBR_EDGE_SNAP = False    # 已移除：AA 平滑后阈值永不触发
DEPTH_RELIEF = 0.0        # 已移除
DIBR_WARP_ITERATIONS = False   # 已移除：重采样导致双边缘
# Spatial disocclusion fill. With the current soft depth map it paints flat
# background blocks INSIDE silhouettes (measured on a real frame), which is the
# artefact the user described as "blocks from outside the person coming in".
# Off by default; the pre-fix app effectively never ran it either.
DISOCLUSION_FILL = False  # 未启用
VSYNC = settings.get("VSync", False)
FIX_VIEWER_ASPECT = settings["Fix Viewer Aspect"]
LANG = settings["Language"]

# Handheld Controller Operation Guide for OpenXR Link, can be easily extended for in-game usage
if LANG == "CN":
    ROWS = [
        ("# 手柄操作指南", "", "", True),
        ("", "", "", False),

        ("[屏幕位置与姿态]", "", "", True),
        ("左握持 + 激光指屏幕", "按住移动或旋转", "屏幕垂直平移或90度旋转", False),
        ("右握持 + 激光指屏幕", "按住移动或旋转", "头部球面旋转或任意旋转", False),
        ("双握持 + 激光指屏幕", "按住移动", "双手中心点移动", False),
        ("左握持 + 左摇杆 左右", "按住推动", "屏幕偏摆旋转", False),
        ("左握持 + 左摇杆 前后", "按住推动", "屏幕俯仰旋转", False),
        ("右握持 + 右摇杆 左右", "按住推动", "屏幕尺寸调整", False),
        ("右握持 + 右摇杆 前后", "按住推动", "屏幕距离调整", False),
        ("", "", "", False),

        ("[预设与杂项]", "", "", True),
        ("左 Menu 键", "短按", "显示/隐藏状态与快捷键面板", False),
        ("左 Y 键", "短按", "默认环境重置屏幕 / 房间重置座位高度", False),
        ("左 Y 键", "长按 1s", "默认环境循环屏幕预设 / 房间循环座位或灯光", False),
        ("左 X 键", "短按", "显示/隐藏虚拟键盘", False),
        ("左 X 键", "长按 1-5s 后松开", "切换灯光/发光模式", False),
        ("左 X 键", "按住 5s", "切换透视绿幕", False),
        ("左扳机(激光不指屏幕)", "按住 3s", "循环裁剪模式 自动/手动/关闭", False),
        ("左扳机(激光不指屏幕,手动模式)", "双击", "开关裁剪调整暂停", False),
        ("左摇杆(裁剪调整中,不握持)", "左右/前后", "裁剪左右(X)或上下(Y),主轴", False),
        ("右 A 键", "激光指屏幕时短按", "鼠标左键单击", False),
        ("右 B 键", "激光指屏幕时短按", "鼠标右键单击", False),
        ("左摇杆按下", "短按", "切换环境模型", False),
        ("", "", "", False),

        ("[深度与视觉]", "", "", True),
        ("右握持 + 左摇杆 前后", "按住推动", "调整深度强度并同步设置", False),
        ("右握持 + 左摇杆 左右", "按住推动", "调整光效透明度", False),
        ("右握持 + 右摇杆按下", "短按", "重置深度强度为 2.0", False),
        ("右摇杆按下", "短按/长按", "切换曲面屏", False),
        ("", "", "", False),

        ("[鼠标与快捷键(激光指向屏幕时)]", "", "", True),
        ("任一扳机", "全按单击", "屏幕触摸/鼠标单击", False),
        ("任一扳机", "持续按住", "拖动/长按触摸", False),
        ("双扳机", "同时按住", "双指平移/缩放手势", False),
        ("扳机指状态面板", "全按单击", "显示/隐藏快捷键面板", False),
        ("右摇杆 前后(无握持)", "按住推动", "鼠标滚轮滚动", False),
        ("右摇杆 左右(无握持)", "按住推动", "鼠标水平滚动", False),
        ("左摇杆 前后(无握持)", "按住推动", "键盘上下方向键", False),
        ("左摇杆 左右(无握持)", "按住推动", "键盘左右方向键", False),
        ("左摇杆按下", "短按", "切换环境模型", False),
        ("左摇杆按下", "长按 1s", "显示/隐藏状态与快捷键面板", False),
        ("右摇杆按下", "短按", "切换曲面/平面屏幕", False),
        ("右摇杆按下", "长按 1s", "重置屏幕朝向", False),
        ("", "", "", False),

        ("[虚拟键盘(仅键盘显示时)]", "", "", True),
        ("双握持 + 激光指键盘", "按住移动", "键盘绕头球面移动", False),
        ("右握持 + 左摇杆 左右", "按住推动", "键盘宽度缩放", False),
        ("右握持 + 左摇杆 前后", "按住推动", "键盘推拉距离", False),
        ("左握持 + 右摇杆 左右", "按住推动", "键盘偏摆偏移", False),
        ("左握持 + 右摇杆 前后", "按住推动", "键盘俯仰偏移", False),
        ("左握持 + 左摇杆按下", "按住移动", "键盘环绕头部位移", False),
        ("左/右扳机", "半按", "触发键盘按键", False),
        ("", "", "", False),

        ("[手柄模型校准(开发者)]", "", "", True),
        ("右 A+B 键", "同时按住 0.5s", "切换手柄模型并显示环境/手柄指示", False),
        ("右 A+B 键", "同时按住 5s", "进入/退出校准模式", False),
        ("右 B 键", "校准模式中单击", "保存校准并退出", False),
    ]
else:
    ROWS = [
        ("# Controller Operation Guide", "", "", True),
        ("", "", "", False),

        ("[Screen Position & Orientation]", "", "", True),
        ("Left Grip + laser on screen", "Hold & move or twist", "Translate vertically or 90deg rotate", False),
        ("Right Grip + laser on screen", "Hold & move or twist", "Sphere-orbit or free rotate", False),
        ("Both Grips + laser on screen", "Hold & move", "Move by center of both hands", False),
        ("Left Grip + left stick L/R", "Hold & push", "Screen yaw rotation", False),
        ("Left Grip + left stick U/D", "Hold & push", "Screen pitch rotation", False),
        ("Right Grip + right stick L/R", "Hold & push", "Screen width adjustment", False),
        ("Right Grip + right stick U/D", "Hold & push", "Screen distance adjustment", False),
        ("", "", "", False),

        ("[Presets & Misc]", "", "", True),
        ("Left Menu button", "Short press", "Show / hide status + shortcuts panel", False),
        ("Left Y button", "Short press", "Reset screen / room seat height", False),
        ("Left Y button", "Long press 1s", "Cycle screen presets / room seat-light presets", False),
        ("Left X button", "Quick tap (<1s)", "Show/hide virtual keyboard", False),
        ("Left X button", "Hold >1s, release", "Toggle light / glow mode", False),
        ("Left X button", "Hold >4s", "Toggle passthrough green", False),
        ("Left trigger (laser off screen)", "Hold 3s", "Cycle Crop: Auto/Manual/Off", False),
        ("Left trigger (off screen, Manual)", "Double-tap", "Toggle crop-adjust pause", False),
        ("Left stick (crop-adjust, no grip)", "L/R or U/D", "Crop sides (X) or top/bottom (Y), dominant axis", False),
        ("Right Grip + left stick L/R", "Hold & push", "Adjust veil transparency", False),
        ("Right A button", "Laser on screen click", "Left mouse click", False),
        ("Right B button", "Laser on screen click", "Right mouse click", False),
        ("Left stick press", "Short press", "Cycle environment model", False),
        ("", "", "", False),

        ("[Depth & Visual]", "", "", True),
        ("Right Grip + left stick U/D", "Hold & push", "Adjust depth strength", False),
        ("Right Grip + left stick L/R", "Hold & push", "Adjust light effect opacity", False),
        ("Right Grip + right stick press", "Short press", "Reset depth strength to 2.0", False),
        ("Right stick press", "Short/long press", "Toggle curved / reset screen direction", False),
        ("", "", "", False),

        ("[Mouse & Shortcuts (laser on screen)]", "", "", True),
        ("Either trigger", "Full press click", "Screen touch / mouse click", False),
        ("Either trigger", "Hold", "Drag / touch hold", False),
        ("Both triggers", "Hold together", "Two-finger pan / zoom gesture", False),
        ("Trigger on status panel", "Full press click", "Show / hide shortcuts panel", False),
        ("Right stick U/D (no grip)", "Hold & push", "Mouse wheel scroll", False),
        ("Right stick L/R (no grip)", "Hold & push", "Horizontal mouse scroll", False),
        ("Left stick U/D (no grip)", "Hold & push", "Keyboard Up/Down arrows", False),
        ("Left stick L/R (no grip)", "Hold & push", "Keyboard Left/Right arrows", False),
        ("Left stick press", "Short press", "Cycle environment model", False),
        ("Left stick press", "Long press 1s", "Show / hide status + shortcuts panel", False),
        ("Right stick press", "Short press", "Toggle curved / flat screen", False),
        ("Right stick press", "Long press 1s", "Reset screen direction", False),
        ("", "", "", False),

        ("[Virtual Keyboard (keyboard visible only)]", "", "", True),
        ("Both Grips + laser on keyboard", "Hold & move", "Orbit keyboard around head", False),
        ("Right Grip + left stick L/R", "Hold & push", "Keyboard width resize", False),
        ("Right Grip + left stick U/D", "Hold & push", "Keyboard push/pull distance", False),
        ("Left Grip + right stick L/R", "Hold & push", "Keyboard yaw offset", False),
        ("Left Grip + right stick U/D", "Hold & push", "Keyboard pitch offset", False),
        ("Left Grip + left stick press", "Hold & move", "Keyboard sphere orbit", False),
        ("Left/Right trigger", "Half press", "Trigger key on keyboard", False),
        ("", "", "", False),

        ("[Controller Calibration (Developer)]", "", "", True),
        ("Right A+B buttons", "Hold both 0.5s", "Switch controller model and indicator", False),
        ("Right A+B buttons", "Hold both 5s", "Enter/exit calibration mode", False),
        ("Right B button", "Click in calibration mode", "Save calibration & exit", False),
    ]

# Determin the run mode and stream mode
if RUN_MODE == "Local Viewer":
    RUN_MODE = "Viewer"
elif RUN_MODE == "3D Monitor" and OS_NAME == "Windows":
    RUN_MODE = "Viewer"
    USE_3D_MONITOR = True
elif RUN_MODE in ("MJPEG Streamer", "RTMP Streamer"):
    # 推流面已随 2c 重构移除（2026-11-16）：老配置名兼容映射到本地 Viewer
    RUN_MODE = "Viewer"
elif RUN_MODE == "OpenXR Link":
    RUN_MODE = "OpenXR"
else:
    RUN_MODE = "Streamer"

# Specify the Stereo Display for output

STEREO_DISPLAY_INDEX = settings["Stereo Output"]
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
CONTROLLER_MODEL = settings["Controller Model"]
ENVIRONMENT_MODEL = settings.get("Environment Model", "Default")
XR_PREVIEW_WINDOW = settings.get("XR Preview", True)
CROP_MODE = settings.get("Crop Mode", "manual")

# Initialize Device —— S3 去 torch：设备探测不再依赖 torch。推理事实以
# ONNX Runtime 的执行后端为准（CUDA/cuDNN DLL 由 ../runtime 提供）；名字
# 只用于日志与 GUI 展示（nvidia-smi -> WMI 兜底）。DEVICE 是后端字符串
# （"cuda"/"cpu"），不再是 torch.device。

DEVICE, DEVICE_INFO = get_device(DEVICE_ID)


def resolve_depth_model_onnx(model=None, resolution=None):
    """兼容包装：默认档位来自本模块常量（原实现读 utils.globals()）。"""
    import model_catalog as _mc
    m = model if model is not None else globals().get("MODEL")
    r = resolution if resolution is not None else globals().get("DEPTH_RESOLUTION")
    return _mc.resolve_depth_model_onnx(m, r)
