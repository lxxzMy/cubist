# main.py
import time as _time_boot
import builtins as _builtins_boot
_builtins_boot._D2S_T0 = _time_boot.perf_counter()   # 进程内最早锚（builtins 全局可见）
_T0 = _time_boot.perf_counter()
import threading
import queue
import time
import signal
import sys
import subprocess
import io
import os

# GLFW 惰性导入：D2S_RENDER=d3d（1.1-d3d 主路径）的进程完全不拉 GL 栈
# （PyOpenGL/moderngl/glfw 三个包导入 GIL 密集，且 d3d 路径零 GL 依赖）。
# GL 回退（D2S_RENDER=gl）在 main() 内 `global glfw; import glfw`。
glfw = None
RENDER_BACKEND = os.environ.get("D2S_RENDER", "d3d").strip().lower()
# D3D 主路径的推理走 d2s_dml_shim v2（渲染器 device 上的零拷贝会话），
# depth.py 的 legacy ONNX 会话完全用不到 —— 别在 import 时白加载一份模型
# （会占用另一块卡并拖慢启动 3~5s）。GL 回退/离线工具照常懒加载。
if RENDER_BACKEND not in ("gl", "glfw", "opengl"):
    os.environ.setdefault("D2S_SKIP_LEGACY_MODEL", "1")

# DPI 感知必须先于任何窗口创建（外部用户日志实锤的老 bug：此前在窗口创建
# 之后才 SetProcessDPIAware，缩放非 100% + 多屏负坐标（origin -1920）下
# 坐标被系统虚化 → 全屏定位/采集原点错位 = 黑屏；用户手动设"DPI 缩放=
# 程序"才恢复，管理员只是让兼容性设置留存的巧合）。提前到导入期，
# 兼容性设置不再依赖。幂等：后面的第二次调用无害。
try:
    import ctypes as _ct_dpi
    _ct_dpi.windll.user32.SetProcessDPIAware()
except Exception:
    pass

from collections import deque
from depth_norm import percentile_stats, z_norm

# Force UTF-8 stdout/stderr on Windows
try:
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# When launched via GUI stdout/stderr are PIPEs: tqdm uses \r to redraw
# the bar in-place, but the GUI reads lines (readline), so \r updates
# buffer until the final \n and the bar "suddenly appears" complete.
# Wrap stdout to convert \r -> \n so each tqdm tick becomes a new line,
# and spoof isatty() so tqdm/HF Hub actually render bars at all.
class _CrToLf:
    def __init__(self, stream):
        self._s = stream
    def write(self, text):
        self._s.write(text.replace('\r', '\n'))
    def flush(self):
        self._s.flush()
    def isatty(self):
        return True
    def __getattr__(self, name):
        return getattr(self._s, name)

if sys.stdout is not None and not sys.stdout.isatty():
    sys.stdout = _CrToLf(sys.stdout)
if sys.stderr is not None and not sys.stderr.isatty():
    sys.stderr = _CrToLf(sys.stderr)

# PID file for the hotkey daemon: a CIM-free liveness/exit channel. WMI
# queries issued from the daemon's message-pump thread can stall and take
# both hotkeys down with them, so the daemon reads this file and uses plain
# Get-Process instead. Written BEFORE the heavy imports: depth init can
# crawl for minutes during a first TRT engine build, and a stale pid in
# that window makes the daemon think D2S is not running.
_PID_DIR_EARLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
try:
    os.makedirs(_PID_DIR_EARLY, exist_ok=True)
except Exception:
    pass

# ---- 单实例守卫 -------------------------------------------------------------
# pid 文件只写不校验，直接再跑一次 main（或双击两次启动器）就会起第二个实例：
# 两个实例抢同一个 DXGI 桌面复制会话（一个输出只允许一个会话），并且互相
# 翻 %TEMP%\d2s-3d-mode → 3D 画面反复退回 2D、采集看门狗反复重建。
# 用命名 Mutex 在导入重模块之前就拦掉，并弹一句人话提示。
_SINGLE_MUTEX = None


def _single_instance_guard():
    """True = 本进程是唯一实例；False = 已有实例在跑。"""
    global _SINGLE_MUTEX
    try:
        import ctypes
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.CreateMutexW.restype = ctypes.c_void_p
        _k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_wchar_p]
        h = _k32.CreateMutexW(None, 0, "Local\\Cubist-D2S-SingleInstance")
        if not h:
            return True                      # 拿不到锁就放行，别把人挡在门外
        _SINGLE_MUTEX = h                    # 句柄常驻：进程退出才释放
        return ctypes.get_last_error() != 183   # ERROR_ALREADY_EXISTS
    except Exception:
        return True


if not _single_instance_guard():
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            "立体派已在运行，请检查托盘。\n\n"
            "（如果托盘里没有图标，请在任务管理器里结束残留的 pythonw.exe 后重试）",
            "立体派 Cubist — 已在运行",
            0x00000040 | 0x00040000 | 0x00010000)   # 信息 | 置顶 | 前台
    except Exception:
        pass
    raise SystemExit(0)

try:
    with open(os.path.join(_PID_DIR_EARLY, "d2s.pid"), "w") as _pf:
        _pf.write(str(os.getpid()))
except Exception:
    pass

# pythonw 下 sys.stdout/stderr 为 None：任何 print 都会 AttributeError，
# 用户完全看不到报错（双击启动"毫无反应"的真因）。必须在导入 utils 之前
# 兜底——utils 在导入期就会 print("[Settings] ...")。
import _stdio
_stdio.ensure()

from utils import (FULLSCREEN_OUTPUT, OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, MONITOR_INDEX, SHOW_FPS, XR_PREVIEW_WINDOW, CROP_MODE, FPS, WINDOW_TITLE, CONVERGENCE, RUN_MODE, LOSSLESS_SCALING_SUPPORT, USE_3D_MONITOR, FILL_16_9, FIX_VIEWER_ASPECT, CAPTURE_MODE, STEREO_DISPLAY_SELECTION, STEREO_DISPLAY_INDEX, shutdown_event, CONTROLLER_MODEL, ENVIRONMENT_MODEL, VSYNC, INTERLEAVE_PHASE, PARALLAX_PX, PARALLAX_CURVE, DEPTH_UPSAMPLER, AA_STRENGTH, DIBR_EDGE_SNAP, DEPTH_RELIEF, DIBR_WARP_ITERATIONS, DISOCLUSION_FILL, CURSOR_MODE, SAME_SCREEN, CAPTURE_MONITOR_INDEX)
from depth import process
import depth as _depth_mod   # 深度 EMA/参数的运行时热改入口

# CUDA-GL interop (CUDART/PBO) 上传路径已随去 torch 移除：viewer 直接
# glTexSubImage2D 上传 numpy 帧，不再需要 GPU 端驻留 tensor。
# Global process references

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_REQUEST_FILE = os.path.join(BASE_DIR, "logs", "stop.request")

# PID file path (the write itself happens above, before the heavy imports).
PID_FILE = os.path.join(BASE_DIR, "logs", "d2s.pid")

# Track current stream size + restart lock
current_stream_size = None
ffmpeg_restart_lock = threading.Lock()

# Use precise frame interval
TIME_SLEEP = 1.0 / FPS

# 注意：utils.FPS 是历史命名，含义是【显示器刷新率】——
# platform_probe.get_fps → EnumDisplaySettings.DisplayFrequency（失败回落 60），
# **不是** DDA/内容帧率。内容/深度管线完全跟 DDA 新帧走，不看它。
REFRESH_HZ = max(1.0, float(FPS))

# 无新帧时"光标/HUD 重呈现"的节拍：按刷新周期稍提前（0.968×，
# 60Hz → 16.13ms ≈ 62Hz；120Hz → 8.07ms ≈ 124Hz）。略快于刷新率是
# 为了每次 vblank 前都有一帧待翻，避免调度抖动错过 vblank 掉成半速。
# 它只决定光标/overlay 的重画频率，不限制内容处理。
REPRESENT_DT = 0.968 / REFRESH_HZ

# Queues with size=1 (latest-frame-only logic)
class LatestSlot:
    """单槽 hold-latest 管道段（延迟优化：44ms -> ~25ms）。

    生产者 publish 永不阻塞（覆盖旧值 + bump 版本）：处理线程死了采集照跑、
    渲染卡了不再反压处理（旧 FIFO 队列的 put 阻塞曾把整条管线冻死）。
    消费者 take_new 只在有更新版本时返回（可限时等待），take 无视版本拿最新。

    版本号防静止空转：静态桌面没有新帧时，深度线程在 take_new 里休眠
    （100ms 超时兜底轮询键盘/模式），不再对同一帧反复推理
    （实测旧管线静态桌面 1.9 核 CPU + 33% GPU 全是重复劳动）。

    发布合约：槽里的 numpy 数组视为只读 —— 采集的 repeat 路径会复用同一
    buffer，任何消费者就地改写会串数据。需要可变的自己 copy。
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._version = 0
        self._seen = 0          # 本消费者已取走的版本
        self._closed = False

    def publish(self, item):
        with self._cond:
            self._item = item
            self._version += 1
            self._cond.notify_all()

    def take_new(self, timeout=0.0):
        """有比上次新的才返回；等到超时仍无则 None。close 后恒 None。"""
        with self._cond:
            if self._closed:
                return None
            if self._version <= self._seen and timeout > 0:
                self._cond.wait(timeout)
            if self._closed or self._version <= self._seen:
                return None
            self._seen = self._version
            return self._item

    def take(self):
        with self._cond:
            return self._item

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


# viewer 隐藏（同屏 2D / 手动隐藏）期间采集休眠的共享开关；
# 恢复时 capture_loop 强制发布一帧，避免显窗后黑屏等画面变化。
CAPTURE_IDLE = [False]
GPU_COLOR = [None]     # 全屏 SBS 颜色零拷贝采集器（DXGI 环纹理）
ZCC_STATE = [None]     # None=等渲染线程初始化 / 'ok' / 'failed'（失败→CPU 回落）
GPU_ACTIVE = [False]   # M1 GPU 直通：渲染线程 runner 就绪后置 True，process
                       # 线程据此跳过 CPU 推理；init 失败保持 False=CPU 兜底
                       # 开会时尺寸，目标 resize 后帧不跟随（皮肤跟着活矩形
                       # 走 -> 黑柱 letterbox，用户实测截图反推确认）；
                       # 尺寸对不上时由探靶对账重开会话

raw_q = LatestSlot()      # 采集 -> 处理（名字沿用，语义已换）
depth_q = LatestSlot()    # 处理 -> 渲染
gpu_q = LatestSlot()      # M2：WinRT 采集 -> 渲染（GPU 帧句柄，numpy 退场）
import utils as _utils_mod   # GPU Pipeline 属重启组（模块导入时读一次）

import time as _time_hdr
_stdio.ensure(header=f"===== 启动 {_time_hdr.strftime('%Y-%m-%d %H:%M:%S')} =====")
GPU_PIPELINE_CFG = bool(_utils_mod.settings.get("GPU Pipeline", False))
# GPU 深度归一化百分位 → depth_norm.py（P2 解耦，常量与实现均在彼处）
try:
    _tau_ms = float(getattr(_depth_mod, "DEPTH_SMOOTHING_TAU_MS",
                            float(getattr(_depth_mod, "DEPTH_SMOOTHING_FRAMES", 0))
                            * 1000.0 / 60.0) or 0)
    if _tau_ms > 0:
        print(f"[depth] 深度 EMA：τ = {_tau_ms:.0f}ms", flush=True)
except Exception:
    pass
# GPU 直通要求推理会话里真的有 GPU EP（DML/CUDA/TRT）。没有的话继续开只会
# 每帧失败、画面退化成 2D。这里提前关掉并把原因告诉用户（pythonw 没控制台，
# 日志用户看不到，所以除了日志还弹一个置顶提示）。
try:
    _dml_st = getattr(_depth_mod, "DML_STATUS", None) or {}
    _gpu_eps = {"DmlExecutionProvider", "CUDAExecutionProvider",
                "TensorrtExecutionProvider"}
    if RENDER_BACKEND in ("gl", "glfw", "opengl"):
        _gpu_ok = bool(_gpu_eps & set(_dml_st.get("providers") or []))
    else:
        # D3D 主路径：推理走 d2s_dml_shim v2 自带的 ORT+DML（capi 目录
        # runtime/ 或 _toolchain/dl），与产品自己的 onnxruntime 会话无关
        # ——会话回落 CPU 不再是否决项（shim 建不起来时由采集环报错）。
        _gpu_ok = True
    if GPU_PIPELINE_CFG and not _gpu_ok:
        _why = _dml_st.get("reason") or "未检测到可用的 GPU 推理后端"
        print("=" * 62, flush=True)
        print(f"[3D] GPU 深度不可用：{_why}", flush=True)
        print("[3D] 已改用 CPU 计算深度：3D 仍可用，但帧率会下降", flush=True)
        print("=" * 62, flush=True)
        try:
            _utils_mod.settings["GPU Pipeline"] = False
        except Exception:
            pass
        GPU_PIPELINE_CFG = False
        try:
            import notify as _notify
            _logp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "logs", "app.log")
            _notify.warn(
                "立体派：3D 加速不可用",
                "检测到当前显卡/驱动无法使用 GPU 深度加速：\n\n"
                f"  {_why}\n\n"
                "已自动改用 CPU 计算深度：3D 仍然可用，但帧率会下降。\n\n"
                "建议：\n"
                "  · 把显卡驱动更新到最新版（需要支持 DirectX 12）\n"
                "  · 远程桌面 / 虚拟机里没有真实 GPU 时属于正常现象\n"
                f"\n详细日志：{_logp}",
                once_key="gpu-ep-unsupported")
        except Exception:
            pass
except Exception as _e:
    print(f"[3D] GPU 深度自检失败: {_e}", flush=True)

# Thread latency tracking dictionaries
# Diagnostics: set D2S_FPS_LOG=1 to print fps / stage timings every 5s, which
# is how a stutter gets traced after the fact (the log is otherwise silent).
_FPS_LOG = os.environ.get("D2S_FPS_LOG", "") not in ("", "0", "false", "False")
# Depth stages that missed the 60 Hz budget, plus the worst one, per window.
_diag = {"over": 0, "max": 0.0}

thread_latencies = {
    'capture': 0.0,
    'resize': 0.0,
    'depth': 0.0,
    'render': 0.0,
    'total': 0.0
}

# Initialize capture
from capture import DesktopGrabber
def capture_loop():
    # 颜色零拷贝（默认开启）：DXGI 环纹理直达渲染线程，
    # 省掉 8MB CPU 拷贝 + 6MB 上传（4K 下一帧 33MB，必须走这条路）。
    # 需要 GPU 深度管线（深度从 color_tex 采样，CPU 分支拿不到 rgb）。
    # 显式关闭：D2S_CAPTURE_GPU=0/false/off/no
    _zcvv = os.environ.get("D2S_CAPTURE_GPU", "1").strip().lower()
    _zcc = _zcvv not in ("", "0", "false", "off", "no")
    if _zcc and not GPU_PIPELINE_CFG:
        print("[capture] 颜色零拷贝需要 GPU Pipeline，已忽略 D2S_CAPTURE_GPU",
              flush=True)
        _zcc = False
    if _zcc:
        # 颜色零拷贝由渲染线程初始化（viewer 起来之后再建 duplication：
        # 实测“先建再进全屏”会被 ACCESS_LOST 打废、重建后几乎不可用
        # （300 次 NOT_CURRENTLY_AVAILABLE 才对 24 帧）；而在全屏稳定态
        # 里创建则能稳定出帧）。
        #
        # 产品语义（用户定案 2026-11-16）：GPU-only，永不回落 CPU 采集。
        # 此前的 8s 超时回落会造成"降级会话"（CPU 采集 = 帧率 50 波动
        # 形态、画质降、还和 GPU 环互踩）；环建不起来就等——渲染线程
        # 的指数退避（1/2/4/8s 循环）会在过渡窗/泄漏会话自愈后建起。
        # 期间画面等待（黑 = 未就绪，可观测），每 30s 打一条状态。
        # D2S_CAPTURE_GPU=0 显式关闭仍走下方 CPU 路径（诊断专用）。
        print("[capture] 颜色零拷贝：等待渲染线程初始化环（GPU-only，"
              "无 CPU 回落）", flush=True)
        _zw0 = time.time()
        _zstat_t = time.time()
        while not shutdown_event.is_set():
            if ZCC_STATE[0] == "ok":
                break
            _znow = time.time()
            if _znow - _zstat_t > 30.0:
                _zstat_t = _znow
                print(f"[capture] GPU 环未就绪已 {_znow - _zw0:.0f}s"
                      f"（状态={ZCC_STATE[0]}，渲染线程退避重试中），"
                      f"继续等待——CPU 回落已按产品语义禁用", flush=True)
            time.sleep(0.1)
        while not shutdown_event.is_set():
            time.sleep(0.5)
        if GPU_COLOR[0] is not None:
            try:
                GPU_COLOR[0].stop()
            except Exception:
                pass
        return
    # 【CPU 采集路径：仅 D2S_CAPTURE_GPU=0 显式诊断用】产品语义 GPU-only
    # （见上方等待循环），自动回落已按用户定案移除（2026-11-16）。
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX)
    _grab_times = []
    _idle_was = False
    _last_live = time.perf_counter()   # 看门狗用
    _grab_err_t = 0.0   # 看门狗：上次拿到真帧/异常确认存活的时刻
    while not shutdown_event.is_set():
        try:
            # （迟到让位机制已随 GPU-only 定案退役：自动回落不再发生，
            #  本路径仅 D2S_CAPTURE_GPU=0 显式诊断时进入，无交还对象。）
            if CAPTURE_IDLE[0]:
                _idle_was = True
                _last_live = time.perf_counter()
                time.sleep(0.2)
                continue
            # 采集看门狗：外部程序在同一屏开第二个 duplication 会话会把
            # 本会话挤掉（实测诊断脚本触发过：viewer 冻结、CPU 空转、
            # GPU 全零）。内部重建只覆盖可识别的 HRESULT；静默 >3s 且
            # 非休眠 => 无差别重建整个 grabber。
            _acq_ok = getattr(cap, "_acq_ok_ts", None)
            if _acq_ok is not None and time.perf_counter() - _acq_ok > 3.0:
                print("[capture] 静默>3s，重建采集会话（看门狗）", flush=True)
                try:
                    cap.stop()
                except Exception:
                    pass
                cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX)
                _last_live = time.perf_counter()
            t_pre = time.perf_counter()
            got = cap.grab()
            frame_raw, size, is_new = got   # DXGI: repeat/缓存/兜底帧 is_new=False
            t_post = time.perf_counter()
            if _FPS_LOG and is_new:
                _grab_times.append(t_post - t_pre)
                if len(_grab_times) >= 60:
                    import statistics as _st
                    print(f"[cap] grab {_st.median(_grab_times)*1000:.1f}ms "
                          f"(p95 {sorted(_grab_times)[int(len(_grab_times)*0.95)]*1000:.1f}ms, "
                          f"n={len(_grab_times)})", flush=True)
                    _grab_times = []

            if shutdown_event.is_set():
                break

            # 只发布真新帧：静态桌面不 bump 版本，深度线程得以休眠。
            # 空闲恢复例外：强制发布一帧（哪怕是 repeat 缓存帧），否则
            # 同屏 2D->3D 显窗后要等第一次画面变化才有内容（黑屏）。
            if is_new or _idle_was:
                raw_q.publish((frame_raw, size, t_pre, t_post))
            _idle_was = False
        except queue.Empty:
            continue
        except Exception as e:
            # 限流打印 + 让出 CPU：会话被外部挤占时 grab 每次调用都抛
            # （实测 6 秒刷百万行告警、单核空转），错误处理本身不能成为
            # 故障放大器。看门狗（>3s 无 acquire 应答）负责重建。
            _now = time.perf_counter()
            if _now - _grab_err_t > 5.0:
                _grab_err_t = _now
                print(f"[Warning] capture error: {e}")
            time.sleep(0.05)
            continue

# Combined processing + depth thread (replaces process_loop and depth_loop)
import tempfile as _tf
import mode_bus
_MODE_FILE = mode_bus.MODE_FILE
_boot_perf_t = time.perf_counter()   # "开机→环就绪"耗时打点的锚（GPU-only 观测）

# 开机默认不运行（产品语义：打开 = 待命，用户切 3D 才开始转换）。
# 模式文件不再跨启动记忆上一次状态；托盘/热键读同一文件，自动一致。
try:
    mode_bus.write_mode(False)
except OSError:
    pass


def _zcc_ensure_inline(window, want):
    """颜色零拷贝内联环懒初始化（2D/3D 通用，幂等）。

    2D 态也必须建好：此前初始化只在 3D 渲染路径里跑，2D 的隐藏分支
    提前 continue 根本走不到 —— 切 3D 那一刻才开始建 duplication
    （E_INVALIDARG 过渡重试）+注册 GL，首个 3D 画面要等数秒
    （用户实测"之前瞬间、现在很久"）。开机在 2D 就绪，切换即响应。
    """
    if not want or GPU_COLOR[0] is not None:
        return
    import time as _t_zi
    _now_zi = _t_zi.perf_counter()
    # 退避只约束 2D 后台预热；3D 消费中缺环 = 最高优先，无视退避立即重试
    # （失败的尝试自身含 ~3s 内部重试窗，实际频率 ~3s/次，不会打爆）。
    # 此前 8s 退避档会让"切 3D"被自己的退避表压住 → 3D 启动 20s+。
    if _now_zi < globals().get("_zcc_retry_at", 0.0):
        if not want:
            return
        globals()["_zcc_retry_at"] = 0.0
        globals()["_zcc_retry_n"] = 0
    try:
        from gpu.dxgi_gpu_grab import DxgiGpuColor
        # 采集屏（AB 分离）：按 placement 口径解析该卡内的输出序号
        # （MONITOR_INDEX-1 只是全局序号，双卡双屏时会对不上输出）。
        _adv_i = None
        try:
            from gpu import placement as _pl_i
            _adv_i = _pl_i.pick_adapter()
            _oi_i = _pl_i.output_index_for(CAPTURE_MONITOR_INDEX, _adv_i)
        except Exception:
            _oi_i = None
        if _oi_i is None:
            _oi_i = max(0, int(CAPTURE_MONITOR_INDEX) - 1)
        _gc = DxgiGpuColor(gpu_q, _oi_i, adapter=_adv_i)
        _gc.build_inline()
        _gc.register_gl(window.ctx)
        window._dx = _gc.dx
        window._dx_ring_gl = _gc.ring_gl
        window._dx_gen = _gc.gen
        GPU_COLOR[0] = _gc
        ZCC_STATE[0] = "ok"
        globals()["_zcc_shown_slot"] = -1
        globals()["_zcc_retry_n"] = 0
        globals()["_zcc_retry_at"] = 0.0
        print(f"[capture] 颜色零拷贝内联就绪（开机→就绪 "
              f"{_now_zi - globals().get('_boot_perf_t', _now_zi):.1f}s）",
              flush=True)
    except Exception as _e:
        # 失败不永久判死：开机预建常撞 viewer 全屏初始化的 E_INVALIDARG
        # 过渡窗（重试 3s 不够），永久失败会让整个会话降级 CPU 采集
        # （帧率 50 波动的实测来源）。指数退避后再试；环迟到就绪时
        # 采集线程的"迟到让位"会自动把 CPU 交还给 GPU 环。
        _n = min(2, globals().get("_zcc_retry_n", 0) + 1)
        globals()["_zcc_retry_n"] = _n
        # 退避上限 2s（原 8s）：尝试本身含 ~3s 内部重试窗，8s 档会让
        # "堵块清除→环就绪"拖到 20s+（用户实测）。2s 档下周期 ≤5s。
        globals()["_zcc_retry_at"] = _now_zi + 1.0 * (2 ** (_n - 1))   # 1/2s
        if ZCC_STATE[0] != "failed":
            ZCC_STATE[0] = "failed"   # 采集线程先回落 CPU（迟到让位待命）
            print(f"[capture] 颜色零拷贝初始化失败（{type(_e).__name__}），"
                  f"回落 CPU 采集并 {_n and 2 ** (_n - 1)}s 后重试", flush=True)

# 自愈：上一次进程被强杀（没走 atexit）时，系统光标可能还停留在我们
# 替换的空白位图上 —— 表现为"鼠标变成几乎看不见的半透明"。
# 启动时无条件恢复一次系统光标表，再由正常流程按需重新隐藏。
try:
    import cursor_hide as _ch_heal
    _ch_heal.restore()
except Exception:
    pass
# 托盘图标：pythonw 无控制台，没有它用户"双击后毫无反应"。
# 双击图标开设置、右键有 切 2D/3D 与 退出。
try:
    import tray as _tray_mod
    _TRAY = _tray_mod.start(mode_file=_MODE_FILE, on_quit=shutdown_event.set)
except Exception as _te:
    print(f"[tray] 未启动: {_te}", flush=True)
import ctypes as _ct_mod
_u32 = _ct_mod.windll.user32
_SUPPRESS_FILE = _tf.gettempdir() + '/d2s-suppress-3d-toggle'


# 帧数按 60Hz 折算成时间常数：N=3 → τ=50ms。刷新率变化时用 dt 重算 alpha，
# 保证不同刷新率下 EMA 的拖尾时长一致（否则 120Hz 下 3 帧只有 25ms）。
EMA_BASE_HZ = 60.0
_LAST_DEPTH_T = [0.0]


def ema_alpha(frames: float, now: float) -> float:
    """按真实帧间隔算 alpha；frames 以 60Hz 为基准解释。"""
    import math as _m
    tau = max(1e-3, float(frames) / EMA_BASE_HZ)
    prev = _LAST_DEPTH_T[0]
    dt = (now - prev) if prev > 0 else (1.0 / EMA_BASE_HZ)
    _LAST_DEPTH_T[0] = now
    dt = min(max(dt, 1.0 / 240.0), 0.2)      # 夹住异常间隔（暂停/卡顿）
    return 1.0 - _m.exp(-dt / tau)


def _render_frame(window):
    """所有渲染调用都走这里：顺便把覆盖层要的 FPS/Delay 注入（GPU 路径
    不经过 update_frame，别处注入会漏）。"""
    try:
        window.actual_fps = float(globals().get("_cur_fps", getattr(window, "actual_fps", 0.0) or 0.0))
        window.total_latency = float(globals().get("_e2e_ms", 0.0))
    except Exception:
        pass
    try:
        # 强制回默认帧缓冲：探针/诊断路径可能留下别的 FBO 绑定，
        # 导致 SBS 画进小 FBO、屏幕全黑（颜色零拷贝下实测）。
        window.ctx.screen.use()
    except Exception:
        pass
    _dbg = None
    if os.environ.get("D2S_ZC_PROBE"):
        try:
            import ctypes as _ctp
            from OpenGL import GL as _GLP
            _dxo = getattr(window, "_dx", None)
            _lk = getattr(_dxo, "last_lock", None) if _dxo is not None else None
            _unit = None
            _pgm = getattr(window, "prog", None)
            try:
                if _pgm is not None and 'tex_color' in _pgm:
                    _unit = int(_pgm['tex_color'].value)
            except Exception:
                pass
            _bnd = _GLP.glGetIntegerv(_GLP.GL_TEXTURE_BINDING_2D)
            _dbg = (f"lock_ok={_lk['ok'] if _lk else '?'} "
                    f"retries={_lk['retries'] if _lk else '?'} "
                    f"unit={_unit} bound_glo={_bnd} "
                    f"tex_glo={getattr(getattr(window, 'color_tex', None), 'glo', '?')}")
        except Exception as _e0:
            _dbg = f"dbg_err:{type(_e0).__name__}"
    window.render()
    # 黑帧探针（D2S_ZC_PROBE=1）：最终帧缓冲中心 32×32 若接近全黑，
    # 打印渲染状态，定位"纹理有内容但画出来是黑"的原因。
    if os.environ.get("D2S_ZC_PROBE"):
        try:
            import numpy as _np
            _w, _h = glfw.get_framebuffer_size(window.window)
            _vw = ((_w - 32) // 2, (_h - 32) // 2, 32, 32)
            _raw = window.ctx.screen.read(components=3, viewport=_vw)
            _m = float(_np.frombuffer(_raw, _np.uint8).mean())
            if _m < 1.0:
                _n = globals().get("_blk_n", 0) + 1
                globals()["_blk_n"] = _n
                if _n <= 20:
                    try:
                        _cm = window.gpu_color_mean()
                        _cms = ("None" if _cm is None else
                                f"{float(__import__('numpy').atleast_1d(_cm).ravel()[0]):.1f}")
                    except Exception as _ce:
                        _cms = f"err:{type(_ce).__name__}:{_ce}"
                    try:
                        _vis = bool(glfw.get_window_attrib(window.window, glfw.VISIBLE))
                    except Exception:
                        _vis = "?"
                    try:
                        _ds = getattr(window, "_u_depth_strength", None)
                        _dsv = None if _ds is None else round(float(_ds.value), 3)
                        _pp = getattr(window, "_u_p_parallax", None)
                        _ppv = None if _pp is None else round(float(_pp.value), 4)
                        _eo = getattr(window, "_u_eye_offset", None)
                        _eov = None if _eo is None else round(float(_eo.value), 4)
                        _uni = (f"ds={_dsv} ppar={_ppv} eye={_eov} "
                                f"parallax={getattr(window, 'parallax_px', '?')} "
                                f"mode={getattr(window, 'display_mode', '?')}")
                    except Exception as _ue:
                        _uni = f"uni_err:{type(_ue).__name__}"
                    # FBO 复绘：把同一帧画进 64×64 FBO 再读，区分"渲染画不出"
                    # 与"present 路径黑"
                    _fbm = None
                    try:
                        _ctxp = window.ctx
                        _pf = globals().get("_fbo_probe")
                        if _pf is None:
                            _ptex = _ctxp.texture((64, 64), 3)
                            _pf = _ctxp.framebuffer([_ptex])
                            globals()["_fbo_probe"] = _pf
                            globals()["_fbo_probe_tex"] = _ptex
                        _pf.use()
                        _pf.viewport = (0, 0, 64, 64)
                        window.render()
                        _rawb = _pf.read(components=3, alignment=1)
                        _fbm = float(__import__("numpy").frombuffer(_rawb,
                                    __import__("numpy").uint8).mean())
                        _ctxp.screen.use()
                    except Exception as _fe:
                        _fbm = f"err:{type(_fe).__name__}"
                    try:
                        _wr2, _hr2 = glfw.get_framebuffer_size(window.window)
                        _dtx = getattr(window, "depth_tex", None)
                        _fg = getattr(window, "_frame_has_depth", None)
                        _win_info = (f"win={_wr2}x{_hr2} depth_glo="
                                     f"{getattr(_dtx, 'glo', '?')} has_depth={_fg}")
                    except Exception as _we:
                        _win_info = f"win_err:{type(_we).__name__}"
                    print(f"[blk] #{_n} mean={_m:.2f} {_win_info} fbo_mean={_fbm} {_dbg} {_uni} visible={_vis} tex_mean={_cms} "
                          f" has_depth={getattr(window, '_frame_has_depth', '?')}"
                          f" use_3d="
                          f"{getattr(window, 'use_3d', '?')} tex="
                          f"{getattr(getattr(window, 'color_tex', None), 'glo', '?')} "
                          f"size={getattr(window, '_texture_size', '?')} "
                          f"has_depth={getattr(window, '_frame_has_depth', '?')}",
                          flush=True)
            else:
                globals()["_blk_ok"] = globals().get("_blk_ok", 0) + 1
        except Exception:
            pass



# ---- Pause 键 2D/3D toggle --------------------------------------------------
# 现代方案：RegisterHotKey 全局热键（事件驱动，WM_HOTKEY 消息投递，零轮询
# 零漏按零 CPU；一次按压恰好一条消息）。注册失败（被别的程序占用）则
# 回退到双沿轮询。
# settings "Toggle Hotkey": pause / scrolllock / ctrl+<字母> / 面板捕获的 mods:vk
HOTKEY_EVENT = [False]      # 热键线程 -> 处理线程的一次性握手
HOTKEY_OK = [False]         # 注册成功后处理线程跳过轮询
_WM_HOTKEY = 0x0312
_MOD_CONTROL = 0x0002
_VK_SCROLL, _VK_F10 = 0x91, 0x79

# 桌面/任务栏等 shell 表面：不是"内容窗口"。前台一旦是它们（点托盘/任务
# 栏、桌面获得焦点、托盘溢出窗/任务栏缩略图短暂拿到前台），内容区域会缩成
# 一条/一小块（外部用户日志实测 3D 区域 = (0,1008)-(1920,1032)；本机实测
# NotifyIconOverflowWindow 120x120），此时应回退整屏。
_SHELL_CLASSES = frozenset((
    "Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
    "NotifyIconOverflowWindow", "TaskListThumbnailWnd", "TaskListOverlayWnd",
    "tooltips_class32", "ForegroundStaging",
    "XamlExplorerHostIslandWindow",      # Win11 托盘/开始弹出层
))


def _is_shell_window(hwnd):
    """hwnd 是否 shell 表面（桌面/任务栏/托盘弹出层）。异常一律 False。"""
    try:
        import ctypes as _c
        _u = _c.windll.user32
        _u.GetClassNameW.argtypes = [_c.c_void_p, _c.c_wchar_p, _c.c_int]
        _b = _c.create_unicode_buffer(128)
        _u.GetClassNameW(_c.c_void_p(hwnd), _b, 128)
        return _b.value in _SHELL_CLASSES
    except Exception:
        return False


def _hotkey_spec():
    """settings -> (mods, vk, 名字)。未知值回退 ctrl+f2。

    注意：按住 Ctrl 时 Pause 会被系统翻译成 VK_CANCEL(0x03)（=Ctrl+Break），
    而不是 VK_PAUSE+MOD_CONTROL，所以 "Ctrl+Pause" 这类组合拿不到消息。
    """
    import utils as _u
    h = str(_u.settings.get("Toggle Hotkey", "ctrl+space")).strip().lower()
    import re as _re
    _m = _re.match(r"^(\d+):(\d+)$", h)
    if _m:                                   # 面板捕获的任意键：mods:vk
        _mods, _vk = int(_m.group(1)), int(_m.group(2))
        _names = {0x13: "Pause", 0x91: "Scroll Lock", 0x78: "F9", 0x79: "F10",
                  0x7A: "F11", 0x7B: "F12", 0x20: "Space", 0x0D: "Enter"}
        _label = _names.get(_vk)
        if _label is None:
            _label = (chr(_vk) if (0x41 <= _vk <= 0x5A or 0x30 <= _vk <= 0x39)
                      else f"VK{_vk}")
        _pre = "".join(("Ctrl+" if _mods & 0x2 else "", "Alt+" if _mods & 0x1 else "",
                        "Shift+" if _mods & 0x4 else ""))
        _nm = f"{_pre}{_label}" if _pre else _label
        return (_mods, _vk, _nm)
    # 支持单键（pause / scrolllock / f9~f12 / a~z）与前缀 ctrl+
    _single = {"pause": (0x13, "Pause"),
               "scrolllock": (_VK_SCROLL, "Scroll Lock"),
               "f9": (0x78, "F9"),
               "f10": (0x79, "F10"),
               "f11": (0x7A, "F11"),
               "f12": (0x7B, "F12")}
    _mods = 0
    _name = h
    if h.startswith("ctrl+"):
        _mods, h = _MOD_CONTROL, h[5:]
        _name = "Ctrl+" + h.upper()
    if h in _single:
        _vk, _label = _single[h]
        return (_mods, _vk, (f"Ctrl+{_label}" if _mods else _label))
    if len(h) == 1 and ("a" <= h <= "z" or "0" <= h <= "9"):
        return (_mods, ord(h.upper()), (f"Ctrl+{h.upper()}" if _mods else h.upper()))
    return (_MOD_CONTROL, 0x71, "Ctrl+F2")


def _hotkey_thread():
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    _MSG = wintypes.MSG
    _mods, _vk, _name = _hotkey_spec()
    _ok = False
    for _i in range(5):
        if u32.RegisterHotKey(None, 1, _mods, _vk):
            _ok = True
            break
        time.sleep(1.0)
    if not _ok:
        print(f"[hotkey] RegisterHotKey({_name}) 失败（重试5次），回退轮询检测",
              flush=True)
        return
    HOTKEY_OK[0] = True
    print(f"[hotkey] {_name} 全局热键已注册（事件驱动）", flush=True)
    msg = _MSG()
    while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == _WM_HOTKEY:
            HOTKEY_EVENT[0] = True
    u32.UnregisterHotKey(None, 1)

_VK_PAUSE = 0x13
_HK_LAST = [0.0]


def _toggle_poll(pause_prev, _unused=None):
    """一帧热键检测 -> (key_prev, None, fired)。

    两路沿：0x8000（当前按住）跨采样边沿 + LSB（自上次调用后按过，
    键盘快按落在两次轮询之间也能捕捉）。首帧只建基线不吃 LSB。
    """
    import ctypes as _ct
    _u32 = _ct.windll.user32
    _mods, _vk, _ = _hotkey_spec()
    _st = _u32.GetAsyncKeyState(_vk)
    p = bool(_st & 0x8000)
    fired = (p and pause_prev is False) or (pause_prev is not None and bool(_st & 0x0001))
    if fired:
        # 按住不放时键盘自动重复会不断置 LSB → 一个实例就能把 2D/3D 翻成抖振。
        # 两次翻转间隔 <300ms 一律丢掉（人手不可能这么快）。
        _now = time.perf_counter()
        if _now - _HK_LAST[0] < 0.3:
            fired = False
        else:
            _HK_LAST[0] = _now
    return p, None, fired


def process_depth_loop():
    _is_set = shutdown_event.is_set
    _latencies = thread_latencies
    _perf = time.perf_counter
    _last_err_t = 0.0
    import ctypes as _ct
    _isvisible = _ct.windll.user32.IsWindowVisible
    _rc_prev = False   # Pause 按下状态（上一帧）
    _want3d = False
    _last_3d = None       # 上次发布时的 3D 状态（翻转时强制重处理最新帧）
    _was_hidden = False
    _stage_log = []
    _PROC_GATE = 0.96 / max(1, FPS)   # 发布节拍：略低于刷新间隔（见循环内注释）
    _next_proc_t = 0.0
    # 自适应深度节流状态（详见循环内注释）；D2S_INFER_EVERY 作为
    # 初始/下限（设 2/3 = 永不低于该间隔）
    import os as _os_mod
    from collections import deque as _deque
    _ad_every = max(1, int(_os_mod.environ.get("D2S_INFER_EVERY", "1")))
    _ad_ema = 8.0        # depth 阶段耗时 EMA（ms，仅日志参考）
    _ad_win = _deque(maxlen=40)   # 最近 40 次推理的超时标志（>14ms）
    _ad_cool = 0
    _ad_hold = 0.0       # 升档冷却（ms 时间戳）
    _ad_t0 = _perf()
    _ad_n = 0            # 帧计数（模 _ad_every）
    _last_depth = None
    _warm_t = 0.0      # ORT 保温计时
    while not _is_set():
        try:
            # 热键事件（RegisterHotKey 线程投递）；注册失败才走轮询兜底
            if HOTKEY_EVENT[0]:
                HOTKEY_EVENT[0] = False
                _fired = True
                _src = "hotkey_event"
            elif not HOTKEY_OK[0]:
                _rc_prev, _, _fired = _toggle_poll(_rc_prev)
                _src = "poll_fallback"
            else:
                _fired = False
                _src = ""
            if _fired:
                _new_mode = '2d' if _want3d else '3d'
                try:
                    mode_bus.write_mode(not _want3d)
                    # Stamp the press so the tray daemon does not flip it back.
                    with open(_SUPPRESS_FILE, 'w') as _sf:
                        _sf.write(repr(time.time()))
                    print(f"[Main] pause -> {_new_mode} @ {time.strftime('%H:%M:%S')} "
                          f"[src={_src} hotkey_ok={HOTKEY_OK[0]}]", flush=True)
                except OSError as _e:
                    print(f"[Main] pause: cannot write mode file ({_e})")

            # 模式文件读取（外部变更检测：非热键写入的模式翻转在此现形）。
            # 节流到 150ms 一读：文件只在用户按 Pause 时变化，逐拍 open+read
            # 在动画场景（循环空转快）吃掉渲染线程 ~55% CPU（py-spy 实测），
            # 是颜色零拷贝后帧率掉到 50 的真凶；150ms 检测延迟无感。
            _new_want3d = mode_bus.read_mode_throttled(
                time.perf_counter(), "proc")
            if _new_want3d != _want3d and not _fired:
                print(f"[Main] 模式被外部翻转 {_want3d}->{_new_want3d} "
                      f"@ {time.strftime('%H:%M:%S')}", flush=True)
            _want3d = _new_want3d

            # Pause: while the viewer window is hidden, drop frames without
            # running resize/inference/render. hold-latest 槽自行覆盖，
            # 不会像旧队列那样把采集线程堵在 put() 上。采集也一并休眠
            #（同屏 2D / 手动隐藏 = 没有消费者）。
            _hwnd = globals().get('_VIEWER_HWND')
            _hidden = bool(_hwnd and not _isvisible(_hwnd))
            if _hidden:
                _was_hidden = True
                CAPTURE_IDLE[0] = True
                if _perf() - _warm_t > 5.0:
                    _warm_t = _perf()
                time.sleep(0.2)
                continue
            CAPTURE_IDLE[0] = False

            # 强制重处理：模式翻转 / 窗口重现。静态桌面没有新帧可等，
            # 但 2D/3D 语义变了必须重跑最新帧，否则切换会等到画面变化才生效。
            _force = (_want3d != _last_3d) or _was_hidden
            _was_hidden = False
            if _force:
                _item = raw_q.take()                # 最新帧，无视版本
            else:
                _item = raw_q.take_new(timeout=0.1)  # 有新帧就醒，100ms 兜底
            if _item is None:
                continue

            # 发布节拍（速率帽）：DXGI 对 DirectFlip 游戏（PvZ 这类老引擎
            # 不限帧）会把游戏每次 Present 直接送进 duplication，实测 78fps
            # 灌进 60Hz 镜面 -> 无 vsync swap 中途换帧 = 撕裂+卡顿。旧管线
            # 的 16.6ms tick 恰好是这顶帽子。栅格取刷新间隔的 96%：真 60Hz
            # 合成内容（帧距 16.67ms > 栅格）永远直通、零额外延迟；超速内
            # 容压回 ~62fps，等待期到达的更新帧覆盖槽，醒来后拿最新的处理。
            _now = _perf()
            _slack = _next_proc_t - _now
            if _slack > 0:
                time.sleep(_slack)
                _fresher = raw_q.take()
                if _fresher is not None:
                    _item = _fresher
            _next_proc_t = max(_next_proc_t + _PROC_GATE,
                               _perf() - 0.05)  # 静止后不追赶补帧

            frame_raw, size, t_pre, t_post = _item[:4]
            _frame_tag = _item[4] if len(_item) > 4 else None
            _last_3d = _want3d

            process_start = _perf()
            _queue_wait = process_start - t_post
            frame_rgb = process(frame_raw, size)
            _proc_time = _perf() - process_start
            # 只用 GPU 推理：process 线程不再计算深度（CPU 路径已移除）。
            # 深度由渲染线程的 GPU 直通产生，这里只传色帧。
            depth = None
            depth_latency = 0.0
            _latencies['capture'] = _queue_wait
            _latencies['resize'] = _proc_time
            _latencies['depth'] = depth_latency
            if depth_latency > 0.0166:
                _diag["over"] += 1
                _diag["max"] = max(_diag["max"], depth_latency)

            # EMA 的 prev buffer 会被下一帧就地改写：发布前 copy（224²，~25µs）
            if depth is not None:
                depth = depth.copy()
            depth_q.publish((frame_rgb, depth, t_pre, t_post, _frame_tag))
            # 端到端延迟（覆盖层 Delay 显示）：采集开始 → 本帧处理完成
            try:
                globals()["_e2e_ms"] = max(0.0, (time.perf_counter() - t_pre) * 1000.0)
            except Exception:
                pass

            if _FPS_LOG:
                _stage_log.append((_queue_wait, _proc_time))
                if len(_stage_log) >= 60:
                    import statistics as _st
                    _q = _st.median([x[0] for x in _stage_log]) * 1000
                    _p = _st.median([x[1] for x in _stage_log]) * 1000
                    print(f"[depth-loop] queue(t_post) {_q:.1f}ms | process {_p:.1f}ms | "
                          f"n={len(_stage_log)}", flush=True)
                    _stage_log.clear()

        except Exception:
            # A dead depth thread silently freezes the whole pipeline. Back off
            # and retry instead of letting this thread die（take_new 的 100ms
            # 兜底也保证不会永久卡死在等帧上）.
            _now = _perf()
            if _now - _last_err_t > 5.0:
                _last_err_t = _now
                import traceback
                traceback.print_exc()
            time.sleep(0.5)
            continue

def cleanup_all_resources():
    """Global cleanup function"""
    print("[Cleanup] Shutting down all resources...")

    # 退出前恢复系统光标：3D 时我们把光标表换成了空白位图，
    # 不恢复的话进程没了、光标还停在"看不见"的状态。
    try:
        import cursor_hide as _ch_clean
        _ch_clean.restore()
    except Exception:
        pass

    # 先摘托盘图标：正常退出若直接 sys.exit，窗口不销毁，
    # 资源管理器要等鼠标划过才回收 → 看起来"退出后图标还在"。
    try:
        import tray as _tray_cleanup
        _tray_cleanup.stop()
    except Exception:
        pass

    # Drop the PID file first so the hotkey daemon stops tracking us.
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass
    # Stop capture
    except Exception as e:
        print(f"[Cleanup] Error stopping capture: {e}")
    # 唤醒所有在 take_new 里等待的线程（LatestSlot 无积压可排）。
    for q in (raw_q, depth_q):
        q.close()
    
    print("[Cleanup] All resources cleaned up")

def _force_exit_watchdog(delay):
    """Fallback: hard-exit if a graceful shutdown hangs.

    OpenXR / GLFW / GPU-driver native threads sometimes refuse to join, which
    would leave this process alive as an orphan still holding the GPU context +
    model caches (e.g. ~/.miopen) - that makes the *next* launch of any mode
    crash until a reboot.  If the main loop hasn't unwound and exited within
    `delay` seconds of a shutdown signal, force the process to die.
    """
    time.sleep(delay)
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)

_shutdown_watchdog_started = False

def _arm_shutdown_watchdog():
    """Start the 8s force-exit watchdog once per process shutdown.

    Shared by the signal handler and the stop.request watcher: a graceful
    unwind that ever hangs (e.g. blocked on a queue or a stuck native thread)
    must still terminate, otherwise the caller's taskkill fallback fires and
    we lose the GPU cleanup we were trying to protect.
    """
    global _shutdown_watchdog_started
    if not _shutdown_watchdog_started:
        _shutdown_watchdog_started = True
        threading.Thread(target=_force_exit_watchdog, args=(8.0,),
                         name="ShutdownWatchdog", daemon=True).start()

def _watch_stop_request_file():
    """GUI-to-child graceful stop channel that avoids Windows CTRL_BREAK."""
    while not shutdown_event.is_set():
        try:
            if os.path.exists(STOP_REQUEST_FILE):
                print("[Signal] Stop requested by GUI, shutting down gracefully...")
                shutdown_event.set()
                _arm_shutdown_watchdog()
                try:
                    os.remove(STOP_REQUEST_FILE)
                except Exception:
                    pass
                break
        except Exception:
            pass
        time.sleep(0.1)

def signal_handler(signum, frame):
    """Handle Ctrl+C / CTRL_BREAK / termination signals.

    We intentionally do NOT exit here.  Instead we set shutdown_event and let
    the main render/stream loop unwind naturally so the viewer's own cleanup
    runs - in OpenXR mode that means xr.end_session/destroy_session plus release
    of the D3D11/GL GPU interop.  Skipping that (a hard kill mid-GPU-frame) can
    wedge the GPU/compute driver until reboot and break the next launch of any
    mode.  A watchdog thread force-exits if the graceful unwind ever hangs.
    """
    print(f"\n[Signal] Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    _arm_shutdown_watchdog()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if OS_NAME == "Windows":
    # CTRL_BREAK_EVENT (sent by the GUI to this process group) arrives as
    # SIGBREAK on Windows: route it through the same graceful handler so the
    # OpenXR/GPU teardown runs before we exit.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


def _resolve_capture_output(window):
    """采集屏（CAPTURE_MONITOR_INDEX）→ 渲染卡内的 DDA 输出序号。

    AB 分离语义：窗口/上屏在 MONITOR_INDEX，采集可在另一块屏。但零拷贝
    管线把采集环纹理直接建在渲染设备上，所以两块屏必须同一块显卡。
    同卡 → 返回采集屏在该卡里的输出序号；不同卡 → 弹一次提示框（运行时
    告知用户），并回退为"采集输出屏"，避免静默黑屏/采集永远建不起。
    返回 (ok, output_index)，ok=False 表示已回退。
    """
    out_i = int(getattr(window, "output_index", 0))
    try:
        cap_i = int(CAPTURE_MONITOR_INDEX)
        out_m = int(MONITOR_INDEX)
    except (TypeError, ValueError):
        return True, out_i
    try:
        from gpu import placement as _plc
        _n = len(_plc.monitor_rects())
    except Exception:
        return True, out_i
    # 越界值（拆屏后屏数变少/手改 yaml）不能静默套到第一块屏上
    if _n and not (1 <= cap_i <= _n):
        print(f"[capture] 采集屏设置 {cap_i} 越界（共 {_n} 块屏）→ "
              f"用输出屏 {out_m}", flush=True)
        return True, out_i
    if cap_i == out_m:
        return True, out_i
    try:
        k = _plc.output_index_for(cap_i, getattr(window, "adapter", None))
        if k is not None:
            print(f"[capture] 采集屏=显示器 {cap_i}（本卡输出 {k}），"
                  f"输出屏=显示器 {out_m}", flush=True)
            return True, k
        ca = _plc.monitor_adapter(cap_i)
        oa = getattr(window, "adapter", None) or _plc.monitor_adapter(out_m)
        cn = getattr(ca, "name", "未知")
        on = getattr(oa, "name", "未知")
    except Exception as _e:
        cn = on = f"枚举失败（{_e}）"
    try:
        import ctypes as _ct
        _ct.windll.user32.MessageBoxW(
            None,
            f"采集屏（显示器 {cap_i}）与输出屏（显示器 {out_m}）"
            f"不在同一块显卡：\n\n"
            f"  采集：{cn}\n"
            f"  输出：{on}\n\n"
            "零拷贝管线要求两块屏接在同一块显卡上。\n"
            "本次已回退为采集输出屏；请把两块屏接到同一块卡，\n"
            "或在设置里把采集屏/输出屏选到同一块卡。",
            "立体派：采集屏与输出屏不在同一显卡", 0x00040030)  # TOPMOST|WARNING
    except Exception:
        pass
    print(f"[capture] 警告：采集屏 {cap_i}（{cn}）与输出屏 {out_m}（{on}）"
          f"不同卡 → 回退采集输出屏", flush=True)
    return False, out_i


def _main_d3d(mode="Viewer"):
    """D3D12 产品主循环（Cubist-1.1-d3d 主路径；GL 回退 D2S_RENDER=gl）。

    事件驱动（跟随 DDA 新帧交付率，静止不空转推理），零 GL 依赖：
      DDA 桥环（NTHANDLE|KM）→ prep → DML shim v2（同队列 FIFO）→
      GPU 判据链（hist/分位/Otsu 锚/cut，链 shader 直读 stats）→
      立体显示（Half-SBS/Interleaved/Passthrough/DepthMap）→ flip present。

    产品语义与 GL 路径一致：热键/tray/mode 文件、设置实时调参（params.request）、
    WDA 排除、光标自绘、同屏 2D 原生/3D 覆盖显隐、自动收敛（GPU 锚延迟一帧）。
    """
    import ctypes as _ctd
    from collections import deque as _dq

    # POINTER(c_float) 每次调用都会新建一个指针类型（类型↔StgDict 成环 =
    # 循环垃圾），每帧一次 → 必须复用。
    _PF32 = _ctd.POINTER(_ctd.c_float)

    import numpy as np      # 模块级无 np（GL 分支内部才 import）

    from gpu import d3drender as _dr
    from gpu import dml_shim as _shim_mod
    from gpu import dxgi_gpu_grab as _zcc_mod
    from gpu.d3d_stats import DEPTH_RANGE_HI as _RANGE_HI
    from gpu.d3d_stats import DEPTH_RANGE_LO as _RANGE_LO
    from gpu.d3d_stats import D3DStats
    from gpu.d3d_window import D3DStereoWindow
    from gpu.dda_bridge import DdaBridge
    from utils import get_monitor_size

    threading.Thread(target=_hotkey_thread, daemon=True).start()
    os.environ.setdefault("D2S_RENDER", "d3d")

    # ---- 窗口（SAME_SCREEN：全屏隐藏覆盖层；否则窗口化查看器） ----
    # 显示器尺寸按 placement 的统一口径（(left,top) 排序，与设置面板一致）
    try:
        from gpu import placement as _plc0
        _mr0 = _plc0.monitor_rect(MONITOR_INDEX)
    except Exception:
        _mr0 = None
    if _mr0:
        _mw, _mh = _mr0[2] - _mr0[0], _mr0[3] - _mr0[1]
    else:
        _mw, _mh = get_monitor_size(MONITOR_INDEX)
    if SAME_SCREEN:
        _w, _h = int(_mw), int(_mh)
    else:
        _w = 1280
        _h = int(round(_w * _mh / max(1, _mw)))
    window = D3DStereoWindow(
        capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX,
        display_mode=DISPLAY_MODE, fill_16_9=FILL_16_9,
        convergence=CONVERGENCE, parallax_px=PARALLAX_PX,
        interleave_phase=INTERLEAVE_PHASE,
        cursor_overlay=(CURSOR_MODE == "overlay"),
        show_fps=bool(SHOW_FPS),
        frame_size=(_w, _h), vsync=bool(VSYNC), same_screen=SAME_SCREEN)
    window.set_title(WINDOW_TITLE)   # 与 GL 路径同标题（托盘守护按标题找窗口）
    # 采集屏/输出屏（AB 分离）：window.output_index 始终是"DDA 采集输出
    # 序号"，这里按 CAPTURE_MONITOR_INDEX 解析（不同卡 → 弹框 + 回退）。
    _cap_ok, _cap_oi = _resolve_capture_output(window)
    window.output_index = _cap_oi
    print(f"[Main] D3D 产品窗口就绪 {_w}x{_h} display={DISPLAY_MODE} "
          f"same_screen={SAME_SCREEN} res={window._res} "
          f"cap_mon={CAPTURE_MONITOR_INDEX} cap_oi={_cap_oi}"
          f"{'' if _cap_ok else '（不同卡回退）'}", flush=True)
    if FULLSCREEN_OUTPUT and os.environ.get("D2S_WDA", "1") != "0":
        try:
            print(f"[viewer] WDA 已设: {window.set_wda(True)}", flush=True)
        except Exception as _e:
            print(f"[viewer] WDA 设置失败: {_e}", flush=True)
    _sbs_full = FULLSCREEN_OUTPUT and DISPLAY_MODE in (
        "Full-SBS", "Half-SBS", "Full-TAB", "Half-TAB")
    if FULLSCREEN_OUTPUT and (CURSOR_MODE != "off" or _sbs_full):
        # 句柄表必须在光标被隐藏【之前】建（GL 路径同序；否则缓存的是空白形状）
        window.prepare_cursor_sprite(force_arrow=bool(FULLSCREEN_OUTPUT))
        try:
            import cursor_hide as _ch
            _ch.start_watcher(_MODE_FILE)
            print("[cursor] 全屏 3D 隐藏光标已启用", flush=True)
        except Exception as _ce:
            print(f"[cursor] 启动失败: {_ce}", flush=True)
    if not SAME_SCREEN:
        window.position_on_monitor()

    ren = window.ren
    _LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    _PARAMS_REQ = os.path.join(_LOGDIR, "params.request")
    # 呈现探针（logs/present_watch.on）：每次上屏打 seq/slot/bb/是否真上屏，
    # 用来判定"画面回退"是槽复用（seq 正常、内容旧）还是 flip 队列（bb 换）。
    _PRESENT_PROBE = os.path.join(_LOGDIR, "present_watch.on")
    _params_mtime = 0.0
    _params_t = 0.0

    # ---- 采集环（懒建 + 指数退避；桥模式的消费端全部在此） ----
    _cap = None
    _cap_chk_t = 0.0        # 输出尺寸巡检（分辨率切换后重建采集环）
    _adp_n = 0              # 连续检测到"屏所在卡换了"的次数（≥4 才自重启）
    _mon_seen = [None]      # 上一次的显示器矩形集合（变了才刷新 adapter 缓存）
    _retry_at = 0.0
    _retry_n = 0

    def _ensure_capture():
        nonlocal _cap, _retry_at, _retry_n
        if _cap is not None:
            return True
        _now = time.perf_counter()
        if _now < _retry_at:
            return False
        _zcc = _bridge = None
        try:
            class _Q:
                def publish(self, item):
                    pass

            _zcc = _zcc_mod.DxgiGpuColor(
                _Q(), int(getattr(window, "output_index", 0)),
                adapter=getattr(window, "adapter", None))
            _zcc.build_inline(gl_ctx=None)
            _bridge = DdaBridge(ren, adv=getattr(window, "adapter", None))
            _f11, _ctx4 = _bridge.producer_handles()
            _fcons11, _ = _bridge.consumer_handles()
            _zcc.attach_bridge_fence(_f11, _ctx4, _fcons11)
            ring = _bridge.open_ring(_zcc._ring, _zcc._w, _zcc._h)
            stats = D3DStats(ren, window._res)
            shim = _shim_mod.open_shim2(ren, window.prep.prep, window._res)
            out = np.empty(window._res * window._res, np.float32)
            out2 = np.empty(window._res * window._res, np.float32)
            _cap = dict(zcc=_zcc, bridge=_bridge, ring=ring, stats=stats,
                        shim=shim, out=out, outs=[out, out2],
                        fcons12=_bridge.consumer_fence12())
            _retry_n = 0
            _retry_at = 0.0
            print(f"[capture] D3D 采集环就绪 {_zcc._w}x{_zcc._h} "
                  f"(DDA 桥 + DML shim v2) GPU={getattr(_zcc, 'adapter_name', '?')}",
                  flush=True)
            return True
        except Exception as _e:
            # 失败必须释放已建的 DDA 会话：同进程第二次 DuplicateOutput
            # 会 E_INVALIDARG（单会话独占输出）——首败后不清理 = 永久失败。
            # stop() 只是信号；对象销毁（refcount）才真正 Release 设备/会话，
            # 清引用后强制 gc，并留一小段释放窗口再允许重试。
            try:
                if _zcc is not None:
                    _zcc.stop()
            except Exception:
                pass
            _zcc = None
            _bridge = None
            try:
                import gc as _gc
                _gc.collect()
            except Exception:
                pass
            time.sleep(0.35)
            _retry_n = min(4, _retry_n + 1)
            _retry_at = _now + min(8.0, 1.0 * (2 ** (_retry_n - 1)))
            print(f"[capture] D3D 环初始化失败（{type(_e).__name__}: {_e}），"
                  f"{_retry_at - _now:.0f}s 后重试", flush=True)
            return False

    # ---- 槽回收（consumer→producer）：GPU 用完环槽后才允许采集覆盖它 ----
    _cons_seq = [0]

    def _signal_consumed(slot):
        """帧末在渲染队列 signal 该槽的回收值（采集侧写槽前 Wait 到它）。

        avoid 只挡"当前正在显示"的槽，挡不住 1~2 帧前仍在 GPU 上飞的槽；
        采集覆盖在飞槽 = 一帧整屏错帧。signal 失败不记录（退回旧行为，
        宁可不保护也不让采集永久挂死）。
        """
        if slot is None or slot < 0 or _cap is None:
            return
        _f = _cap.get("fcons12")
        if _f is None:
            return
        try:
            _cons_seq[0] += 1
            _v = _cons_seq[0]
            ren.signal_fence(_f, _v)
            _cap["zcc"].slot_done[slot] = _v
        except Exception as _e:
            if not globals().get("_cons_err_logged"):
                globals()["_cons_err_logged"] = True
                print(f"[capture] 槽回收 signal 失败（退回旧行为）: {_e}",
                      flush=True)

    def _drop_capture():
        """丢弃当前采集环（分辨率/显示器切换后重建用）。

        旧 DDA 会话必须显式 close：ComPtr 不随 GC 释放，会话攥着输出会
        让新 DuplicateOutput 被拒。
        """
        nonlocal _cap, _prev_idx, _switch_force
        cap0 = _cap
        _cap = None
        _prev_idx = -1
        _switch_force = False
        if cap0 is not None:
            try:
                cap0["zcc"].close()
            except Exception:
                pass
            try:
                cap0["bridge"].close()
            except Exception:
                pass
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

    _ema_first = True
    _restart_mtime = 0.0
    _restart_t = 0.0
    # 30s 遥测窗：60fps → ~1800 帧，maxlen 必须覆盖整窗。原来是 300（≈5s），
    # p99 只统计最近 5 秒 —— 偶发卡顿在打印前的滚动窗口里已经被挤掉了。
    _proc = _dq(maxlen=2048)
    _lat_e2e = _dq(maxlen=2048)   # 桌面 present → 本帧 present 提交（ms）
    _lat_cap = _dq(maxlen=2048)   # 桌面 present → DDA acquire 返回（ms）
    _st_inf = _dq(maxlen=2048)    # 推理（prep pass + DML run）
    _st_run = _dq(maxlen=2048)    # DML Run 本体（shim 自报，ms）
    _st_anchor = _dq(maxlen=2048)  # 收敛锚读回（含一次 GPU 同步）
    _st_chain = _dq(maxlen=2048)  # GPU 判据 + 深度链
    _st_disp = _dq(maxlen=2048)   # 显示 + present
    _loop_gap = _dq(maxlen=2048)  # 相邻两帧处理起点间隔（真实管线节拍）
    _hb = [0, 0, 0, 0, 0]         # 节拍直方图（30s 窗，见 _process）
    _e2e_over = [0]               # 本窗端到端延迟 >=18ms 的帧数
    _poll_use = _dq(maxlen=2048)  # 单次 zcc.poll() 耗时
    # 非每帧操作的窗口 max：它们不进 _proc 的 deque（稀疏），但正是"偶发
    # 卡顿"的头号嫌疑 → 单独按 30s 窗记 max/sum/n。
    _win = {}
    _fops = {}      # 本帧窗口（非每帧操作 → 本次耗时），卡顿时随帧打印

    def _wm(name, t0):
        """记一笔非每帧开销（max/sum/n，30s 清零）；返回本次耗时 ms。"""
        d = (time.perf_counter() - t0) * 1000.0
        b = _win.get(name)
        if b is None:
            _win[name] = [d, d, 1]
        else:
            if d > b[0]:
                b[0] = d
            b[1] += d
            b[2] += 1
        _fops[name] = d
        return d

    _qpf = _zcc_mod.qpc_freq()
    _fps_n = 0
    _fps_t = time.perf_counter()
    _fps_n1 = 0            # HUD 用的 1s 滚动 FPS（window.actual_fps）
    _fps_t1 = time.perf_counter()
    _rc_prev = False
    _want3d = False
    _vis = None
    _prev_idx = -1
    _switch_force = False
    _style_t = 0.0
    # ---- 内容区域（GL 同语义）：只对前台窗口做视差，桌面/任务栏原样 ----
    _u32c = _ctd.windll.user32
    _dwmapi = _ctd.WinDLL("dwmapi")
    _dwmapi.DwmGetWindowAttribute.restype = _ctd.c_long
    _dwmapi.DwmGetWindowAttribute.argtypes = [
        _ctd.c_void_p, _ctd.c_uint, _ctd.c_void_p, _ctd.c_uint]

    class _CRECT(_ctd.Structure):
        _fields_ = [("L", _ctd.c_long), ("T", _ctd.c_long),
                    ("R", _ctd.c_long), ("B", _ctd.c_long)]

    class _MONINFO(_ctd.Structure):
        _fields_ = [("cbSize", _ctd.c_uint), ("rcMonitor", _CRECT),
                    ("rcWork", _CRECT), ("dwFlags", _ctd.c_uint)]

    _u32c.MonitorFromWindow.restype = _ctd.c_void_p
    _u32c.MonitorFromWindow.argtypes = [_ctd.c_void_p, _ctd.c_uint]
    _u32c.GetMonitorInfoW.argtypes = [_ctd.c_void_p, _ctd.c_void_p]
    _u32c.GetMonitorInfoW.restype = _ctd.c_int

    _crect_t = 0.0

    def _update_content_rect(_now):
        nonlocal _crect_t
        # 30ms 节流（GL 是 120ms，拖窗口明显滞后；取矩形本身只是两三个
        # 十微秒级 Win32 查询，30ms 无感），拖拽/缩放时 3D 区域实时跟随
        if not FULLSCREEN_OUTPUT or _now - _crect_t < 0.03:
            return
        _crect_t = _now
        try:
            _fg = _u32c.GetForegroundWindow()
            if (not _fg or _fg == window.hwnd
                    or _fg == _u32c.GetShellWindow()
                    or _u32c.IsIconic(_ctd.c_void_p(_fg))
                    or _is_shell_window(_fg)
                    or not _u32c.IsWindowVisible(_ctd.c_void_p(_fg))):
                window.set_content_rect()
                return
            _cr = _CRECT()
            _ok = _dwmapi.DwmGetWindowAttribute(
                _ctd.c_void_p(_fg), 9, _ctd.byref(_cr),
                _ctd.sizeof(_CRECT)) == 0
            if not _ok:
                _u32c.GetWindowRect(_ctd.c_void_p(_fg), _ctd.byref(_cr))
            if not (_cr.R > _cr.L + 16 and _cr.B > _cr.T + 16):
                window.set_content_rect()
                return
            # 最大化窗口四边各伸出工作区 ~8px（不可见 resize 边框，底部
            # 那一截被任务栏盖住）——直接拿窗口矩形会把任务栏算进 3D 区域。
            # 规则：覆盖整块屏 = 全屏窗口（任务栏被盖住）→ 整屏都 3D；
            # 否则裁到工作区 rcWork（排除任务栏）。
            _mi = _MONINFO()
            _mi.cbSize = _ctd.sizeof(_MONINFO)
            _hm = _u32c.MonitorFromWindow(_ctd.c_void_p(_fg), 2)
            if _hm and _u32c.GetMonitorInfoW(_ctd.c_void_p(_hm),
                                             _ctd.byref(_mi)):
                _mon = _mi.rcMonitor
                _mw = max(1, _mon.R - _mon.L)
                _mh = max(1, _mon.B - _mon.T)
                _covw = (_cr.R - _cr.L) / _mw
                _covh = (_cr.B - _cr.T) / _mh
                if _covw >= 0.99 and _covh >= 0.99:
                    window.set_content_rect()          # 全屏：整屏做 3D
                else:
                    _wk = _mi.rcWork
                    window.set_content_rect(
                        max(_cr.L, _wk.L), max(_cr.T, _wk.T),
                        min(_cr.R, _wk.R), min(_cr.B, _wk.B))
            else:
                window.set_content_rect(_cr.L, _cr.T, _cr.R, _cr.B)
        except Exception:
            try:
                window.set_content_rect()
            except Exception:
                pass
    # 诊断/门禁转储：
    #   D2S_FRAME_DUMP=<dir> + D2S_FRAME_DUMP_HZ → frame_<ms>.png（e2e 契约）
    #   D2S_D3D_DUMP=<dir> → 前 20 帧 depth_XX.png + stereo_XX.png（排查用）
    _dump_dir = os.environ.get("D2S_D3D_DUMP", "")
    _fdump_dir = os.environ.get("D2S_FRAME_DUMP", "")
    _fdump_hz = float(os.environ.get("D2S_FRAME_DUMP_HZ", "4") or 4)
    _fdump_t = 0.0
    _dump_n = 0
    _dump_out = [None]
    for _dd in (_dump_dir, _fdump_dir):
        if _dd:
            os.makedirs(_dd, exist_ok=True)

    # 逐帧撕裂探针（诊断）：logs/tear_watch.on 存在时启用（无需重启开关）。
    # 每帧把显示同样的内容重画到离屏纹理 → 读回 → 统计"整行突变"。
    # 撕裂缝 = 某一行上下两半差异覆盖整宽；正常内容只有局部边缘。
    _tear_flag = os.path.join(_LOGDIR, "tear_watch.on")
    _tear_flag_t = 0.0
    _tear_on = [False]

    def _tear_probe(window, src, cap, ren, out_tex):
        """返回 True=检出**新出现**的整行断层（打印并保存该帧）。

        半宽盲区：Half-SBS 一只眼的撕裂只覆盖半幅（~50% 列），全宽阈值
        0.55 会漏掉——所以按左右两半分别统计。撕裂是瞬态的（下一帧就换
        位置），用"上一帧同一行还不是断层"把静态整宽边缘滤掉。
        """
        import cv2 as _cv
        window.disp.draw(window._mode, src, window.post.depth,
                         src_w=cap["zcc"].size[0],
                         src_h=cap["zcc"].size[1], out=out_tex,
                         overlay=window._draw_overlay)
        arr = ren.readback(out_tex, np.uint8)
        g = arr[:, :, 1].astype(np.int16)
        d = np.abs(np.diff(g, axis=0))
        W = d.shape[1]
        cur = [(d[:, 0:W // 2] > 24).mean(axis=1),
               (d[:, W // 2:W] > 24).mean(axis=1)]
        prev = globals().get("_tear_prev")
        hit = None
        if prev is not None and len(prev) == 2 and \
                prev[0].shape == cur[0].shape:
            for k, f in enumerate(cur):
                lo, hi = 60, max(61, f.shape[0] - 60)
                reg = f[lo:hi]
                pre = prev[k][lo:hi]
                new = np.where((reg > 0.55) & (pre < 0.25), reg, 0.0)
                if new.size and float(new.max()) > 0.0:
                    y = int(np.argmax(new)) + lo
                    hit = (k, y, float(new.max()), float(prev[k][y]))
                    break
        globals()["_tear_prev"] = cur
        if hit is not None:
            k, y, mx, pv = hit
            n = globals().get("_tear_n", 0) + 1
            globals()["_tear_n"] = n
            print(f"[tear] {'L' if k == 0 else 'R'} y={y} frac={mx:.2f}"
                  f"（上一帧 {pv:.2f}）第 {n} 次", flush=True)
            try:
                _cv.imwrite(os.path.join(
                    _LOGDIR, f"tear_{int(time.time() * 1000)}.png"), arr)
            except Exception:
                pass
            return True
        return False


    def _self_restart(reason=""):
        """进程内自重启（关互斥体后 execv；模式文件保留 → 3D 自动恢复）。"""
        print(f"[Main] 自重启（{reason}）…", flush=True)
        try:
            if _cap is not None:
                _cap["zcc"].stop()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass
        try:
            import tray as _tray_stop
            _tray_stop.stop()      # 先摘托盘图标，否则重启后残留
        except Exception:
            pass
        try:
            import cursor_hide as _ch2
            _ch2.restore()         # 空白光标表恢复（新进程导入期也会自愈）
        except Exception:
            pass
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # execv = 先拉新进程再退旧进程：先关单实例互斥体，子进程出生即是持有人
        try:
            import ctypes as _ct_rm
            if globals().get("_SINGLE_MUTEX"):
                _ct_rm.WinDLL("kernel32").CloseHandle(
                    _ct_rm.c_void_p(globals()["_SINGLE_MUTEX"]))
                globals()["_SINGLE_MUTEX"] = None
        except Exception:
            pass
        os.execv(sys.executable,
                 [sys.executable, "-u", "-m", "main"] + sys.argv[1:])

    def _handle_restart(_now):
        """设置面板"保存并应用"→ logs/restart.request。GL 主循环有自重启，
        D3D 之前漏了 → 改显示模式后"保存了却不生效"。这里同款处理。"""
        nonlocal _restart_mtime, _restart_t
        if _now - _restart_t < 0.1:
            return
        _restart_t = _now
        _rr = os.path.join(_LOGDIR, "restart.request")
        try:
            _st = os.stat(_rr)
        except FileNotFoundError:
            return
        if _st.st_mtime == _restart_mtime:
            return
        _restart_mtime = _st.st_mtime
        try:
            os.remove(_rr)
        except OSError:
            pass
        print("[settings] 收到重启请求，自重启…", flush=True)
        _self_restart("设置面板请求")

    def _process(idx):
        """单帧：桥门控 → prep → DML → GPU 判据链 → 显示 → present。"""
        nonlocal _ema_first, _fdump_t, _tear_flag_t
        t0 = time.perf_counter()
        # 管线节拍：进程内两帧处理起点间隔（proc 之外的空档 = poll/消息泵/
        # overlay 维系等）。>1s 的是 2D 待命后回首帧，丢弃。
        _lg = t0 - globals().get("_last_proc_t", 0.0)
        globals()["_last_proc_t"] = t0
        if 0.0 < _lg < 1.0:
            _loop_gap.append(_lg * 1000.0)
            # 节拍直方图（30s 窗）：<18ms=赶上 60Hz；24-34=丢 1 个 vsync；
            # 34-60=丢 2 个。用来看"51fps"到底是抖动还是成片丢帧。
            if _lg < 0.018:
                _hb[0] += 1
            elif _lg < 0.024:
                _hb[1] += 1
            elif _lg < 0.034:
                _hb[2] += 1
            elif _lg < 0.060:
                _hb[3] += 1
            else:
                _hb[4] += 1
        cap = _cap
        cap["bridge"].queue_wait(cap["zcc"].bridge_seq)
        src = cap["ring"][idx]
        # 1) prep（桥纹理直接采样，无 CPU 上传）
        ren.begin_frame()
        src.state = _dr.ST_COMMON
        src.transition(_dr.ST_PIXEL_SRV)
        window.prep.prep.state = _dr.ST_COMMON
        window.prep.prep.transition(_dr.ST_RENDER_TARGET)
        ren.draw_pass(window.prep._pso, [src],
                      [float(window._res), float(window.prep.flat_w),
                       0.0, 0.0],
                      window.prep.prep, window.prep.prep_w,
                      window.prep.prep_h)
        window.prep.prep.state = _dr.ST_COMMON
        ren.submit()
        # 2) DML 推理（内部 prep→rBuf 拷贝 + Run，阻塞至 CPU）
        cap["shim"].run(
            _ctd.cast(cap["out"].ctypes.data, _PF32),
            cap["out"].nbytes)
        try:
            _st_run.append(float(cap["shim"].times6()[1]))
        except Exception:
            pass
        raw = cap["out"].reshape(window._res, window._res)
        _t1 = time.perf_counter()
        _ta1 = time.perf_counter()
        # 4) 深度链（GPU 判据：lo/hi/cut 全在 GPU；EMA α 由 cut 覆写）
        _smooth = str(getattr(_depth_mod, "DEPTH_SMOOTHING", "ema"))
        alpha = (ema_alpha(float(_depth_mod.DEPTH_SMOOTHING_FRAMES),
                           time.perf_counter()) if _smooth != "off" else 1.0)
        if _ema_first:
            alpha = 1.0
        aa = (float(getattr(_depth_mod, "AA_STRENGTH", 0.0)) * 0.5
              * (int(_depth_mod.DEPTH_RESOLUTION) / 224.0))
        # aa = 旋钮值 = 【较大方向的 σ】（横屏时就是竖向）。224² 深度拉伸到
        # 每眼视口后，屏幕空间各向同性要求：短边对应方向的 texel σ 更大。
        # 横屏（vp_w≥vp_h）：V = aa，H = aa·(vp_h/vp_w)
        #   16:9 → V=aa，H=aa/1.778；Half-SBS 非Fill 每眼 960×1080 → V=aa，
        #   H=aa·0.889。
        # 竖屏反过来（H=aa，V=aa·(vp_w/vp_h)）。
        _m = getattr(window, "_mode", "interleaved")
        _sw, _sh = cap["zcc"].size
        try:
            if _m in ("half_sbs", "full_sbs", "half_tab", "full_tab"):
                _vp = window.disp.sbs_viewports(_m, _sw, _sh)[0]
            else:
                _vp = window.disp.single_viewport(_sw, _sh)
            _ar = float(_vp[2]) / max(1.0, float(_vp[3]))
            if _ar >= 1.0:
                _aa_h, _aa_v = aa / _ar, aa
            else:
                _aa_h, _aa_v = aa, aa * _ar
        except Exception:
            _vp, _aa_h, _aa_v = None, aa, aa
        _aa_key = (round(_aa_h, 3), round(_aa_v, 3))
        if globals().get("_aa_key_last") != _aa_key:
            globals()["_aa_key_last"] = _aa_key
            print(f"[depth] AA σ: 大方向={aa:.2f} → H={_aa_h:.2f} "
                  f"V={_aa_v:.2f}（视口 "
                  f"{'%dx%d' % _vp[2:4] if _vp else '?'}）", flush=True)
        window.post.upload_raw(raw)
        window.post.spatial_chain(
            0.0, 0.0, aa_texels=_aa_h, aa_v=_aa_v,
            temporal="ema", alpha=alpha, cut=True, stats=cap["stats"],
            color_tex=src, color_size=cap["zcc"].size)
        _ema_first = False
        _t2 = time.perf_counter()
        # 3b) 收敛面：默认每帧自动（近侧主体峰锚定，见 ANCHOR_CS）。
        # D2S_CONV_FIXED 设为数值时临时固定（观测用）；auto/空 = 自动。
        _tan0 = time.perf_counter()
        _conv_fixed = os.environ.get("D2S_CONV_FIXED", "auto").strip().lower()
        if _conv_fixed not in ("", "auto"):
            try:
                _cv = min(1.0, max(0.0, float(_conv_fixed)))
                window.apply_live_params({"Convergence": _cv})
                globals()["_APPLIED_CONV"] = float(_cv)
                # 仍读回一次只为遥测 lo/hi/cut（0.3ms；不参与收敛面）
                cap["stats"].readback()
                if not globals().get("_CONV_FIXED_LOGGED"):
                    globals()["_CONV_FIXED_LOGGED"] = True
                    print(f"[conv] 固定收敛面 = {_cv:.3f}"
                          f"（D2S_CONV_FIXED；设 auto/空 恢复自动）", flush=True)
            except Exception:
                pass
        elif _auto_conv_on():
            try:
                _st = cap["stats"]
                (_lo, _hi, _anchor, _cutr,
                 _p50, _anchor_disp) = _st.readback()
                # 逐帧异常探针：一帧闪烁时看是"锚尖峰"还是"CUT 重置"
                _ad = float(_anchor_disp)
                _pv = globals().get("_anchor_prev")
                if _pv is not None and abs(_ad - _pv) > 0.15:
                    print(f"[conv-spike] 锚 {_pv:.3f} → {_ad:.3f} "
                          f"(rgbTV={getattr(_st, 'rgb_tv', 0):.2f} "
                          f"cut={getattr(_st, 'cut_flag', 0):.0f})",
                          flush=True)
                globals()["_anchor_prev"] = _ad
                _cut_now = float(getattr(_st, "cut_flag", 0.0)) > 0.5
                if _cut_now and not globals().get("_cut_prev"):
                    print(f"[cut] RGB 切变帧 rgbTV={getattr(_st, 'rgb_tv', 0):.2f} "
                          f"thr={getattr(_st, 'cut_thr', 0):.2f}", flush=True)
                globals()["_cut_prev"] = _cut_now
                _depth_mod.LAST_MEDIAN = min(
                    1.0, max(0.0, float(_anchor_disp)))
                _depth_mod.LAST_P30 = _depth_mod.LAST_MEDIAN
                auto_convergence_update(None)
                # 收敛面直接锚在近侧主体峰上（不再向中心半幅阻尼）；
                # 固定下移 0.1（给滑块留上端余量），设置里的偏移仍扣。
                _cv = min(1.0, max(0.0,
                                   _AUTO_CONV[0] - AUTO_CONV_OFFSET
                                   - _auto_conv_bias()))
                window.apply_live_params({"Convergence": _cv})
                globals()["_APPLIED_CONV"] = float(_cv)
            except Exception:
                pass
        _tan1 = time.perf_counter()
        # 深度转储（验收诊断；D2S_D3D_DUMP 目录，前 20 帧）
        nonlocal _dump_n
        if _dump_dir and _dump_n < 20:
            try:
                _d = window.post.readback_depth()
                _v = (_d - _d.min()) / max(1e-9, _d.max() - _d.min()) * 255.0
                try:
                    import cv2 as _cv
                    _cv.imwrite(os.path.join(
                        _dump_dir, f"depth_{_dump_n:02d}.png"),
                        _v.astype(np.uint8))
                except Exception:
                    np.save(os.path.join(_dump_dir,
                                         f"depth_{_dump_n:02d}.npy"), _d)
                # 离屏立体输出（WDA 不影响自身 readback；验证显示链）
                _ot = _dump_out[0]
                if _ot is None:
                    _ot = ren.create_tex(window.disp.w, window.disp.h,
                                         _dr.FMT_BGRA8, renderable=True)
                    _dump_out[0] = _ot
                window.disp.draw(window._mode, src, window.post.depth,
                                 src_w=cap["zcc"].size[0],
                                 src_h=cap["zcc"].size[1], out=_ot,
                                 overlay=window._draw_overlay)
                import cv2 as _cv2
                _cv2.imwrite(os.path.join(_dump_dir,
                                          f"stereo_{_dump_n:02d}.png"),
                             ren.readback(_ot, np.uint8))
                print(f"[dump] depth_{_dump_n:02d} min={_d.min():.3f} "
                      f"max={_d.max():.3f} mean={_d.mean():.3f}", flush=True)
            except Exception as _de:
                print(f"[dump] 失败: {_de}", flush=True)
            _dump_n += 1
        # e2e 帧转储（D2S_FRAME_DUMP 契约：frame_<epoch_ms>.png @ HZ）
        if _fdump_dir:
            _now_f = time.perf_counter()
            if _now_f - _fdump_t >= 1.0 / max(0.1, _fdump_hz):
                _fdump_t = _now_f
                try:
                    _ot = _dump_out[0]
                    if _ot is None:
                        _ot = ren.create_tex(window.disp.w, window.disp.h,
                                             _dr.FMT_BGRA8, renderable=True)
                        _dump_out[0] = _ot
                    window.disp.draw(window._mode, src, window.post.depth,
                                     src_w=cap["zcc"].size[0],
                                     src_h=cap["zcc"].size[1], out=_ot,
                                     overlay=window._draw_overlay)
                    import cv2 as _cv3
                    _cv3.imwrite(os.path.join(
                        _fdump_dir, f"frame_{int(time.time() * 1000)}.png"),
                        ren.readback(_ot, np.uint8))
                except Exception as _fe:
                    print(f"[dump] frame 失败: {_fe}", flush=True)
        # 逐帧撕裂探针（logs/tear_watch.on 存在时启用）：同一帧内容重画到
        # 离屏 → 读回 → 判"整行突变"（撕裂=整宽两段错位；正常内容只有局部边）
        _twn = time.perf_counter()
        if _twn - _tear_flag_t > 1.0:
            _tear_flag_t = _twn
            try:
                _tear_on[0] = os.path.exists(_tear_flag)
            except Exception:
                _tear_on[0] = False
        if _tear_on[0]:
            try:
                _ot = _dump_out[0]
                if _ot is None:
                    _ot = ren.create_tex(window.disp.w, window.disp.h,
                                         _dr.FMT_BGRA8, renderable=True)
                    _dump_out[0] = _ot
                _tear_probe(window, src, cap, ren, _ot)
            except Exception as _te:
                if not globals().get("_tear_err_logged"):
                    globals()["_tear_err_logged"] = True
                    print(f"[tear] 探针异常: {_te!r}", flush=True)
        # 5) 显示 + present（同帧一条龙）
        window.attach_color(src, *cap["zcc"].size)
        window.render()
        window.present()
        globals()["_present_t"] = time.perf_counter()
        _signal_consumed(idx)
        _t3 = time.perf_counter()
        _st_inf.append((_t1 - t0) * 1000.0)
        _st_anchor.append((_tan1 - _tan0) * 1000.0)
        _st_chain.append((_t2 - _ta1) * 1000.0)
        _st_disp.append((_t3 - _t2) * 1000.0)
        # 端到端仪表：DDA 的 LastPresentTime（桌面最后一次 present，QPC）
        # → 本帧 present 提交；cap = 其中采集到达段。均不含最终合成出光时间
        _zq = cap["zcc"].frame_qpc
        _e2e_v = 0.0
        if _zq:
            _now_q = _zcc_mod.qpc()
            _e2e_v = (_now_q - _zq) * 1000.0 / _qpf
            _lat_e2e.append(_e2e_v)
            if cap["zcc"].acq_qpc:
                _lat_cap.append((cap["zcc"].acq_qpc - _zq) * 1000.0 / _qpf)
        _dt = (time.perf_counter() - t0) * 1000.0
        _proc.append(_dt)
        # 端到端延迟尖峰计数（30s 窗口统计，不逐帧打日志）
        if _e2e_v >= 18.0:
            _e2e_over[0] += 1
        # 卡顿逐帧归因：proc 慢、端到端 L 拉长、或两帧间隔被撑开时打一行
        # （限频 1 条/2s）——带阶段分解和本帧跑过的非每帧操作。
        if _dt > 18.0 or _e2e_v >= 22.0 or 24.0 < _lg < 1.0:
            _tsp = time.perf_counter()
            if _tsp - globals().get("_stall_t", 0.0) >= 2.0:
                globals()["_stall_t"] = _tsp
                _fo = " ".join(f"{_k}={_v:.1f}"
                               for _k, _v in sorted(_fops.items()) if _v >= 0.2)
                print(f"[stall] L={_e2e_v:.1f} "
                      f"gap={(_lg * 1000.0 if _lg < 1.0 else 0.0):.1f} "
                      f"proc={_dt:.1f} [inf {(_t1 - t0) * 1000.0:.1f} "
                      f"run {(_st_run[-1] if _st_run else 0.0):.1f} "
                      f"anchor {(_tan1 - _tan0) * 1000.0:.1f} "
                      f"chain {(_t2 - _ta1) * 1000.0:.1f} "
                      f"disp {(_t3 - _t2) * 1000.0:.1f}] ops: {_fo}",
                      flush=True)
        _fops.clear()
        return raw

    # ---- 成对流水线（D2S_PIPELINE=1，默认） --------------------------------
    # 每拍只做"提交"，不阻塞等 GPU：
    #   a) 上一拍投递的推理完成 → 上传 raw + 录深度链 + 提交（链内不 wait）
    #   b) 用刚出的对（颜色槽 + disp_depth）显示并 present(DO_NOT_WAIT)
    #   c) 若有新采集帧 → prep + 投递推理（工作线程）
    # 颜色与深度始终同帧配对；代价是输出比采集晚约 1 拍。
    # 成对流水线（D2S_PIPELINE=1，默认）
    # 诊断转储/撕裂探针内部有阻塞读回，与流水互斥 → 它们开着时自动退回串行。
    _PL_ON = (os.environ.get("D2S_PIPELINE", "1") != "0"
              and not os.environ.get("D2S_D3D_DUMP")
              and not os.environ.get("D2S_FRAME_DUMP"))
    _pl = {"tag": None, "pair": None, "last_pair": None, "out_i": 0,
           "rb_v": None, "seq": 0, "n": 0, "n0": 0, "n1": 0}
    _pl_gpu = None      # None=未判定；True=推理输出留在显存；False=CPU 输出
    # 深度链的 defer 语义只由本循环开启（离屏工具/parity 保持旧的 wait 语义）
    window.post.defer = _PL_ON
    _pl_ticked = [False]      # 本拍是否已由 idle 分支推进（防重复 tick）

    def _pl_aa():
        """当前每眼视口的各向异性 σ（与 _process 同口径）。"""
        aa = (float(getattr(_depth_mod, "AA_STRENGTH", 0.0)) * 0.5
              * (int(_depth_mod.DEPTH_RESOLUTION) / 224.0))
        _m = getattr(window, "_mode", "interleaved")
        _sw, _sh = cap["zcc"].size
        try:
            if _m in ("half_sbs", "full_sbs", "half_tab", "full_tab"):
                _vp = window.disp.sbs_viewports(_m, _sw, _sh)[0]
            else:
                _vp = window.disp.single_viewport(_sw, _sh)
            _ar = float(_vp[2]) / max(1.0, float(_vp[3]))
            return (aa / _ar, aa) if _ar >= 1.0 else (aa, aa * _ar)
        except Exception:
            return aa, aa

    def _tick_pipeline(idx):
        """流水线一拍（idx=-1 表示这一拍没有新采集帧）。

        GPU 输出模式（默认，shim 把推理结果留在显存）：
          新帧 → prep → 推理入队 → 深度链入队 → 显示这一对，全部同一拍
          入队、CPU 全程不等待（同队列 FIFO 保证先推理后链）。
        CPU 输出模式（D2S_DML_GPU_OUT=0）：工作线程 + 下一拍取结果。
        """
        nonlocal _ema_first, _pl_gpu
        cap = _cap
        # GPU 输出可用性只判定一次
        if _pl_gpu is None:
            _ptr = 0
            try:
                _ptr = int(cap["shim"].out_buf_ptr())
            except Exception as _e:
                _ptr = 0
                print(f"[pl] out_buf_ptr 调用失败: {_e!r}", flush=True)
            if _ptr:
                _pl["raw_buf"] = _dr.Buf(ren.ctx, _ptr,
                                        window._res * window._res * 4, 0)
                _pl_gpu = True
                print(f"[pl] GPU 输出模式（推理结果留在显存，链直读；"
                      f"buf={_ptr:#x}）", flush=True)
            else:
                _pl_gpu = False
                _err = ""
                try:
                    _err = cap["shim"].gpu_error()
                except Exception:
                    pass
                print(f"[pl] CPU 输出模式（env={os.environ.get('D2S_DML_GPU_OUT')!r}"
                      f"，buf=0，gpuErr={_err!r}）", flush=True)
        t0 = time.perf_counter()
        _did_work = False
        _st = cap["stats"]
        # 呈现探针开关（1s 节流读文件；logs/present_watch.on）
        if t0 - _pl.get("probe_t", 0.0) >= 1.0:
            _pl["probe_t"] = t0
            try:
                _pl["probe"] = os.path.exists(_PRESENT_PROBE)
            except Exception:
                _pl["probe"] = False
        # 0) 收敛锚：读上一拍提交的读回（非阻塞；没完成就沿用旧值）
        if _pl["rb_v"] is not None:
            _vals = _st.readback_poll(_pl["rb_v"])
            if _vals is not None:
                _pl["rb_v"] = None
                try:
                    _ad = float(_vals[5])
                    _pv = globals().get("_anchor_prev")
                    if _pv is not None and abs(_ad - _pv) > 0.15:
                        print(f"[conv-spike] 锚 {_pv:.3f} → {_ad:.3f}",
                              flush=True)
                    globals()["_anchor_prev"] = _ad
                    _depth_mod.LAST_MEDIAN = min(1.0, max(0.0, _ad))
                    _depth_mod.LAST_P30 = _depth_mod.LAST_MEDIAN
                    auto_convergence_update(None)
                    _cv = min(1.0, max(0.0, _AUTO_CONV[0] - AUTO_CONV_OFFSET
                                    - _auto_conv_bias()))
                    window.apply_live_params({"Convergence": _cv})
                    globals()["_APPLIED_CONV"] = float(_cv)
                except Exception:
                    pass
        # 1) 推理完成 → 链（不等待）【CPU 输出模式】
        if (not _pl_gpu and _pl["tag"] is not None
                and cap["shim"].poll(_pl["tag"][0])):
            _tag, _raw, _pidx = _pl["tag"]
            _pl["tag"] = None
            _seq = _tag
            _src = cap["ring"][_pidx]
            window.post.upload_raw(_raw.reshape(window._res, window._res))
            _alpha = (ema_alpha(float(_depth_mod.DEPTH_SMOOTHING_FRAMES),
                                time.perf_counter())
                      if str(getattr(_depth_mod, "DEPTH_SMOOTHING", "ema"))
                      != "off" else 1.0)
            if _ema_first:
                _alpha = 1.0
            _ah, _av = _pl_aa()
            window.post.spatial_chain(
                0.0, 0.0, aa_texels=_ah, aa_v=_av,
                temporal="ema", alpha=_alpha, cut=True, stats=_st,
                color_tex=_src, color_size=cap["zcc"].size)
            _ema_first = False
            # 链已提交（同队列）→ 紧接着提交 stats 读回，不等待
            if _pl["rb_v"] is None:
                _pl["rb_v"] = _st.readback_start()
            _pl["pair"] = (_pidx, window.post.last_disp_depth, _seq)
            _pl["chains"] = _pl.get("chains", 0) + 1
        # 1b) GPU 输出模式：新帧 → prep 提交 → 推理入队 → 深度链入队。
        #     同一拍里全部入队（同队列 FIFO：推理→链），CPU 不等待，
        #     也不需要工作线程/流水级。
        if _pl_gpu and idx >= 0:
            _src = cap["ring"][idx]
            # 桥协议：D3D11 拷贝是异步的，采样环纹理前必须在渲染队列上
            # 等 producer signal（串行路径同款 gate）——流水线这里曾漏掉，
            # 4K 帧（33MB 拷贝）会读到未落地的槽 = 推理吃到旧帧/半帧。
            cap["bridge"].queue_wait(cap["zcc"].bridge_seq)
            ren.begin_frame()
            _src.state = _dr.ST_COMMON
            _src.transition(_dr.ST_PIXEL_SRV)
            window.prep.prep.state = _dr.ST_COMMON
            window.prep.prep.transition(_dr.ST_RENDER_TARGET)
            ren.draw_pass(window.prep._pso, [_src],
                          [float(window._res), float(window.prep.flat_w),
                           0.0, 0.0],
                          window.prep.prep, window.prep.prep_w,
                          window.prep.prep_h)
            window.prep.prep.state = _dr.ST_COMMON
            ren.submit()
            if cap["shim"].run_gpu():
                _alpha = (ema_alpha(float(_depth_mod.DEPTH_SMOOTHING_FRAMES),
                                    time.perf_counter())
                          if str(getattr(_depth_mod, "DEPTH_SMOOTHING", "ema"))
                          != "off" else 1.0)
                if _ema_first:
                    _alpha = 1.0
                _ah, _av = _pl_aa()
                window.post.spatial_chain(
                    0.0, 0.0, aa_texels=_ah, aa_v=_av,
                    temporal="ema", alpha=_alpha, cut=True, stats=_st,
                    color_tex=_src, color_size=cap["zcc"].size,
                    raw_buf=_pl["raw_buf"])
                _ema_first = False
                if _pl["rb_v"] is None:
                    _pl["rb_v"] = _st.readback_start()
                _pl["seq"] += 1
                _pl["pair"] = (idx, window.post.last_disp_depth, _pl["seq"])
                cap["zcc"].avoid = idx
                _pl["chains"] = _pl.get("chains", 0) + 1
                _pl["caps"] = _pl.get("caps", 0) + 1
                try:
                    _pl.setdefault("pair_qpc", {})[_pl["seq"]] = \
                        cap["zcc"].frame_qpc
                except Exception:
                    pass
                _did_work = True
        # 2) 显示这一对（颜色槽 + 同帧 disp_depth）。准备就绪就立刻上屏：
        #    能不能被 DWM 消费交给 flip 队列/waitable 判定，不在这里节流。
        if _pl["pair"] is not None:
            _didx, _ddepth, _dseq = _pl["pair"]
            _pl["last_pair"] = _pl["pair"]
            _pl["pair"] = None
            _src = cap["ring"][_didx]
            window.attach_color(_src, *cap["zcc"].size)
            window.render(depth_tex=_ddepth)
            _pbb = ren.cur_bb()
            _pok = window.present(nowait=True)
            if _pl.get("probe"):
                print(f"[pp] seq={_dseq} slot={_didx} bb={_pbb} "
                      f"ok={1 if _pok else 0}", flush=True)
            _tp2 = time.perf_counter()
            globals()["_present_t"] = _tp2
            _signal_consumed(_didx)
            _pl["n"] += 1
            _pl["presents"] = _pl.get("presents", 0) + 1
            # e2e：这一对的"桌面 present → 我们的 present 提交"
            try:
                _q0 = _pl["pair_qpc"].get(_dseq)
                if _q0:
                    _lv = (_zcc_mod.qpc() - _q0) * 1000.0 / _qpf
                    _lat_e2e.append(_lv)
                    if _lv >= 18.0:
                        _e2e_over[0] += 1
            except Exception:
                pass
            _did_work = True
            # 真实节拍 = 相邻两次"显示一对"的间隔（不是循环迭代间隔）
            _pg = _tp2 - globals().get("_pl_show_t", 0.0)
            globals()["_pl_show_t"] = _tp2
            if 0.0 < _pg < 1.0:
                _loop_gap.append(_pg * 1000.0)
                if _pg < 0.018:
                    _hb[0] += 1
                elif _pg < 0.024:
                    _hb[1] += 1
                elif _pg < 0.034:
                    _hb[2] += 1
                elif _pg < 0.060:
                    _hb[3] += 1
                else:
                    _hb[4] += 1
        elif (_pl["last_pair"] is not None
              and t0 - globals().get("_present_t", 0.0) >= REPRESENT_DT):
            # 静态内容：按刷新率重呈上一对（光标/HUD 跟随）。这一对已经
            # signal 过，不再重复 signal；用 avoid 钉住槽防止被采集覆盖。
            _didx, _ddepth, _ = _pl["last_pair"]
            cap["zcc"].avoid = _didx
            window.attach_color(cap["ring"][_didx], *cap["zcc"].size)
            window.render(depth_tex=_ddepth)
            window.present(nowait=True)
            globals()["_present_t"] = time.perf_counter()
        # 3)【CPU 输出模式】新采集帧 → prep + 投递推理（不等待）。只在
        #    worker 还忙时跳过（流水线级数决定的吞吐），不按时间节流。
        if (not _pl_gpu) and idx >= 0 and _pl["tag"] is None:
            _src = cap["ring"][idx]
            # 桥协议 gate（见上：串行路径 1576 行同款；流水线曾漏）
            cap["bridge"].queue_wait(cap["zcc"].bridge_seq)
            ren.begin_frame()
            _src.state = _dr.ST_COMMON
            _src.transition(_dr.ST_PIXEL_SRV)
            window.prep.prep.state = _dr.ST_COMMON
            window.prep.prep.transition(_dr.ST_RENDER_TARGET)
            ren.draw_pass(window.prep._pso, [_src],
                          [float(window._res), float(window.prep.flat_w),
                           0.0, 0.0],
                          window.prep.prep, window.prep.prep_w,
                          window.prep.prep_h)
            window.prep.prep.state = _dr.ST_COMMON
            ren.submit()
            _o = cap["outs"][_pl["out_i"]]
            _pl["out_i"] ^= 1
            _pl["seq"] += 1
            _tag = _pl["seq"]
            cap["shim"].submit(_tag, _o.ctypes.data, _o.nbytes)
            _pl["tag"] = (_tag, _o, idx)
            cap["zcc"].avoid = idx
            _did_work = True
            _pl["caps"] = _pl.get("caps", 0) + 1
            try:
                _pl.setdefault("pair_qpc", {})[_tag] = cap["zcc"].frame_qpc
                if len(_pl["pair_qpc"]) > 64:
                    _pl["pair_qpc"].clear()
            except Exception:
                pass
        _st_inf.append(0.0)
        _st_anchor.append(0.0)
        _st_chain.append(0.0)
        _st_disp.append(0.0)
        try:
            _st_run.append(float(getattr(cap["shim"], "last_times", [0, 0])[1]))
        except Exception:
            pass
        # 只在"这一拍真干了活"时计一帧（空转的 tick 不进 proc/fps 统计）
        if _did_work:
            _proc.append((time.perf_counter() - t0) * 1000.0)
        else:
            time.sleep(0.0005)   # 空转让出 CPU（否则 8k/s 空转吃满一核）
        return

    # GC 治理（实测 gen2 全堆回收 14~35ms/次——代价≈遍历活对象集，与回收多少
    # 无关；落在帧间就是一次可见卡顿）：
    #   1) 启动完成后 freeze()：现存 4.7 万对象进永久代，全堆回收不再遍历；
    #   2) 第三阈值抬到 10 万：gen2 基本不触发（gen0/1 保留，微秒级）。
    # 前提是环≈0——2026-11-22 已把运行期造环源清掉（pump 里的 POINT/MSG、
    # identify 里的 _BM、COM 函数指针/POINTER 每调用现建 → 全部模块级/缓存）。
    # gen0 阈值【不动】：抬到 3 万实测反使每次 gen0 遍历 3.3 万对象（17~19ms）。
    try:
        import gc as _gc2
        _gc2.collect()
        _gc2.freeze()
        _gc2.set_threshold(700, 10, 100000)
    except Exception:
        pass

    print(f"[Main] D3D 主循环开始（热键 {_hotkey_spec()[2]} 切 2D/3D）",
          flush=True)
    try:
        while not shutdown_event.is_set():
            _pl_ticked[0] = False
            _tp = time.perf_counter()
            window.pump()
            _wm("pump", _tp)
            # 显示模式变化（切分辨率/换屏/拔插）后的覆盖层几何重建
            _tg = time.perf_counter()
            window.apply_pending_geometry()
            _wm("geom", _tg)
            _tt = time.perf_counter()
            # 重启请求必须在循环【顶部】轮询：2D 同屏时下面走 sleep/continue
            # 分支（不处理帧），放后面会永远收不到——"保存了却不生效"的真因
            _handle_restart(time.perf_counter())
            if window.should_close():
                print("[Main] 窗口关闭 → 退出", flush=True)
                break

            # ---- 热键 / 模式文件（事件优先，注册失败轮询兜底） ----
            if HOTKEY_EVENT[0]:
                HOTKEY_EVENT[0] = False
                _want3d = not _want3d
                try:
                    mode_bus.write_mode(_want3d)
                    with open(_SUPPRESS_FILE, "w") as _sf:
                        _sf.write(repr(time.time()))
                    print(f"[Main] pause -> {'3d' if _want3d else '2d'} @ "
                          f"{time.strftime('%H:%M:%S')} [hotkey_event]",
                          flush=True)
                except OSError as _e:
                    print(f"[Main] pause: cannot write mode file ({_e})")
            elif not HOTKEY_OK[0]:
                _rc_prev, _, _fired = _toggle_poll(_rc_prev)
                if _fired:
                    _want3d = not _want3d
                    try:
                        mode_bus.write_mode(_want3d)
                        with open(_SUPPRESS_FILE, "w") as _sf:
                            _sf.write(repr(time.time()))
                        print(f"[Main] pause(poll) -> "
                              f"{'3d' if _want3d else '2d'}", flush=True)
                    except OSError:
                        pass
            else:
                _new = mode_bus.read_mode_throttled(time.perf_counter(), "d3d")
                if _new != _want3d:
                    print(f"[Main] 模式被外部翻转 {_want3d}->{_new} "
                          f"@ {time.strftime('%H:%M:%S')}", flush=True)
                    _want3d = _new
            _wm("top", _tt)

            # ---- 同屏显隐（2D = 原生桌面，3D = 覆盖层） ----
            if _want3d != _vis:
                _vis = _want3d
                if SAME_SCREEN:
                    if _want3d:
                        window.show()
                        window._reassert_overlay_style()
                        window.keep_topmost()
                        print("[Main] 3D 覆盖层显示", flush=True)
                    else:
                        window.hide()
                        print("[Main] 2D 收回覆盖层", flush=True)
                        # 安全时刻：gen2 平时不自动触发，这里补一次全量回收
                        try:
                            import gc as _gci
                            _gci.collect()
                        except Exception:
                            pass
                _switch_force = bool(_want3d)

            if not _want3d and SAME_SCREEN:
                time.sleep(0.05)      # 桌面原生显示：零开销待命
                continue

            if not _ensure_capture():
                time.sleep(0.2)
                continue
            cap = _cap

            # ---- 分辨率/显示器切换：环纹理尺寸必须跟随 DDA 输出 ----
            # CopyResource 尺寸不匹配会丢帧/花屏；变了就整环重建（旧会话
            # 显式 close，模式切换期间 DXGI 拒建由 _ensure_capture 退避重试）。
            _now_c = time.perf_counter()
            if _now_c - _cap_chk_t > 0.5:
                _cap_chk_t = _now_c
                _tpt = time.perf_counter()
                # 显示器切到另一块 GPU 的输出？采集与渲染设备必须与新屏同卡，
                # 进程内换不了 D3D12 渲染器 → 自重启（模式文件保留，3D 自动
                # 恢复）。要求连续 4 次（≈2s）都不同，避开模式切换瞬间的抖动。
                _lu = 0
                _oi = None
                try:
                    from gpu import placement as _plc
                    _rects = tuple(_plc.monitor_rects())
                    _fresh = _rects != _mon_seen[0]
                    _mon_seen[0] = _rects
                    _a, _oi, _ = _plc.pick_for_monitor(MONITOR_INDEX,
                                                       refresh=_fresh)
                    # 采集输出以采集屏为准（AB 分离；None=不在本卡 → 保持原值）
                    _ci = _plc.output_index_for(CAPTURE_MONITOR_INDEX, _a)
                    if _ci is not None:
                        _oi = _ci
                    _lu = int(_a.luid) & 0xFFFFFFFFFFFFFFFF if _a else 0
                except Exception:
                    _lu, _oi = 0, None
                if _lu and _lu != getattr(ren, "luid", 0):
                    _adp_n += 1
                    if _adp_n >= 4:
                        _self_restart(f"显示器换到另一块 GPU "
                                      f"({getattr(ren, 'luid', 0):#x}→{_lu:#x})")
                else:
                    _adp_n = 0
                # 同一块卡内输出序号变了（多屏重排）→ 采集环要换输出
                if (_oi is not None
                        and _oi != getattr(window, "output_index", _oi)):
                    window.output_index = _oi
                    if cap["zcc"].monitor != _oi:
                        print(f"[capture] 输出序号 {cap['zcc'].monitor} → "
                              f"{_oi}，重建采集环", flush=True)
                        _drop_capture()
                        time.sleep(0.1)
                        continue
                try:
                    _csz = cap["zcc"].output_size_now()
                except Exception:
                    _csz = cap["zcc"].size
                if tuple(_csz) != tuple(cap["zcc"].size):
                    print(f"[capture] 输出尺寸 {cap['zcc'].size} → {_csz}，"
                          f"重建采集环", flush=True)
                    _drop_capture()
                    time.sleep(0.1)
                    continue
                _wm("patrol", _tpt)

            # ---- 新帧驱动（静态桌面：切换瞬间强制重呈上一帧） ----
            _tpl = time.perf_counter()
            idx = cap["zcc"].poll()
            _poll_use.append((time.perf_counter() - _tpl) * 1000.0)
            if os.environ.get("D2S_ZC_DEBUG"):
                _pn = globals().get("_d3d_pn", 0) + 1
                globals()["_d3d_pn"] = _pn
                if idx >= 0:
                    globals()["_d3d_fn"] = globals().get("_d3d_fn", 0) + 1
                _pt_d = time.perf_counter()
                if _pt_d - globals().get("_d3d_ppt", 0.0) >= 2.0:
                    globals()["_d3d_ppt"] = _pt_d
                    _pn0 = globals().get("_d3d_pnp", 0)
                    _fn0 = globals().get("_d3d_fnp", 0)
                    _fn = globals().get("_d3d_fn", 0)
                    _af = getattr(cap["zcc"], "_af_total", 0)
                    _af0 = globals().get("_d3d_afp", 0)
                    globals()["_d3d_pnp"] = _pn
                    globals()["_d3d_fnp"] = _fn
                    globals()["_d3d_afp"] = _af
                    print(f"[poll2s] polls={_pn - _pn0} frames={_fn - _fn0} "
                          f"af={_af - _af0}", flush=True)
            if idx >= 0:
                _prev_idx = idx
                _pl["newf"] = _pl.get("newf", 0) + 1
                # 告知采集：这个槽正在显示/会被重呈现反复采样，不要覆盖
                # （内联 poll 会跳过 avoid；否则 GPU 采样中被拷新帧 = 一帧闪）
                cap["zcc"].avoid = idx
            elif _want3d and _switch_force and _prev_idx >= 0:
                idx = _prev_idx
            elif _PL_ON and _want3d:
                # 流水线：没有新帧也要推进（链/显示/推理都在飞）
                _tick_pipeline(-1)
                _pl_ticked[0] = True
            else:
                # GL 同款：无新帧时用上一帧 color/depth 重画 display+overlay
                # 并 present。内容/深度管线完全跟 DDA 走（这里不产生新内容）；
                # 这个分支只保证【光标/HUD】按显示器刷新率更新——否则内容
                # 30fps 时光标也只能 30Hz。节拍 = REPRESENT_DT（按实际刷新率）。
                # 闸门必须看【最近一次 present】（内容帧也算）：只看上次重呈现
                # 会让每次内容帧后又补一次重呈现 → 实际 ~98 present/s（60Hz
                # 屏），DWM 合成被自己的重复帧挤占，交付帧率反被拖到 47~51。
                _pt = time.perf_counter()
                if (window.post.depth is not None
                        and _pt - globals().get("_present_t", 0.0)
                        >= REPRESENT_DT):
                    globals()["_d3d_repr_t"] = _pt
                    _tr = time.perf_counter()
                    try:
                        window.render()
                        window.present()
                        globals()["_present_t"] = time.perf_counter()
                        _signal_consumed(cap["zcc"].avoid)
                        if SAME_SCREEN:
                            window.keep_topmost()
                        if os.environ.get("D2S_ZC_DEBUG"):
                            _rn = globals().get("_d3d_repr_n", 0) + 1
                            globals()["_d3d_repr_n"] = _rn
                            _rt = globals().get("_d3d_repr_pt", 0.0)
                            if _pt - _rt >= 2.0:
                                globals()["_d3d_repr_pt"] = _pt
                                print(f"[repr] 2s 重呈现 "
                                      f"{_rn - globals().get('_d3d_repr_pn', 0)} 次",
                                      flush=True)
                                globals()["_d3d_repr_pn"] = _rn
                    except Exception as _re_err:
                        if not globals().get("_d3d_repr_err_logged"):
                            globals()["_d3d_repr_err_logged"] = True
                            print(f"[repr] 重呈现失败: {_re_err}", flush=True)
                    _wm("repr", _tr)
                else:
                    _ti = time.perf_counter()
                    time.sleep(0.0005)
                    _wm("idle", _ti)
                continue
            _switch_force = False

            # ---- 设置面板实时调参（消费即删） ----
            _now = time.perf_counter()
            if _now - _params_t > 0.05:
                _params_t = _now
                _tpa = time.perf_counter()
                try:
                    _st = os.stat(_PARAMS_REQ)
                    if _st.st_mtime != _params_mtime:
                        _params_mtime = _st.st_mtime
                        import json as _json
                        with open(_PARAMS_REQ, encoding="utf-8-sig") as _f:
                            _req = _json.load(_f)
                        if "Convergence Bias" in _req:
                            try:
                                _AUTO_CONV_BIAS[0] = float(
                                    _req["Convergence Bias"])
                            except Exception:
                                pass
                        if "Depth Plain Normalize" in _req:
                            _depth_mod.set_plain_norm(
                                bool(_req["Depth Plain Normalize"]))
                        if "Anti-aliasing" in _req:
                            try:
                                _depth_mod.AA_STRENGTH = \
                                    2.0 * float(_req["Anti-aliasing"])
                            except Exception:
                                pass
                        window.apply_live_params(_req)
                        print(f"[settings] 实时调参已应用 {sorted(_req)}",
                              flush=True)
                        os.remove(_PARAMS_REQ)   # 消费即删
                except FileNotFoundError:
                    pass
                except Exception as _pe:
                    print(f"[settings] params.request 处理失败: {_pe}",
                          flush=True)
                    try:
                        os.remove(_PARAMS_REQ)
                    except OSError:
                        pass
                _wm("params", _tpa)

            if _want3d:
                if _PL_ON:
                    if not _pl_ticked[0]:
                        _tick_pipeline(idx)
                else:
                    _process(idx)
            else:
                # 2D 窗口化查看器：passthrough 直显（同屏 2D 走上面休眠分支）
                window.attach_color(cap["ring"][idx], *cap["zcc"].size)
                window.disp.draw("passthrough", cap["ring"][idx],
                                 src_w=cap["zcc"].size[0],
                                 src_h=cap["zcc"].size[1],
                                 overlay=window._draw_overlay)
                window.present(0)
                globals()["_present_t"] = time.perf_counter()
                _signal_consumed(idx)

            # ---- 覆盖层维系（同屏 3D：topmost 每帧 / 样式 2s） ----
            if _want3d and SAME_SCREEN:
                # GL 同款：每帧重申 HWND_TOPMOST。PotPlayer 等会在 topmost
                # band 内"后到者胜"压住输出；实测 SetWindowPos 中位
                # 0.089ms（p99 0.7~2ms），用户定案视觉优先。
                _to = time.perf_counter()
                window.keep_topmost()
                # 逐帧 Z 序探针：谁跑到我们上面（一帧整屏闪的定位用）
                try:
                    _u2 = _ct_mod.windll.user32
                    _prv = _u2.GetWindow(ctypes.c_void_p(window.hwnd),
                                         3)          # GW_HWNDPREV
                    if _prv != globals().get("_zord_prev"):
                        globals()["_zord_prev"] = _prv
                        _cbz = _ct_mod.create_unicode_buffer(128)
                        _u2.GetClassNameW(_ct_mod.c_void_p(_prv), _cbz, 128)
                        _ttz = _ct_mod.create_unicode_buffer(160)
                        _u2.GetWindowTextW(_ct_mod.c_void_p(_prv), _ttz, 160)
                        _cz = _cbz.value
                        _benign = ("", "tooltips_class32", "ForegroundStaging",
                                   "IME", "MSCTFIME UI",
                                   "QianwenIMEVoiceInputBar",
                                   "QianwenIMECandidateWindow")
                        if _cz not in _benign:
                            print(f"[zorder] 我们上方: {_cz} {_ttz.value!r}",
                                  flush=True)
                except Exception:
                    pass
                _wm("ovl", _to)
                if _now - _style_t > 2.0:
                    _style_t = _now
                    _ts = time.perf_counter()
                    window._reassert_overlay_style()
                    _wm("style", _ts)
                # 内容区域 ~8Hz：只对前台窗口做 3D（GL 同语义）
                _tc = time.perf_counter()
                _update_content_rect(_now)
                _wm("crect", _tc)

            # ---- 遥测（5s） ----
            if not _PL_ON:
                _fps_n += 1
                _fps_n1 += 1
            if _now - _fps_t1 >= 1.0:      # HUD 数值 1s 刷新（GL 同一口径）
                if _PL_ON:
                    window.actual_fps = ((_pl["n"] - _pl["n1"])
                                         / max(1e-3, _now - _fps_t1))
                    _pl["n1"] = _pl["n"]
                else:
                    window.actual_fps = _fps_n1 / max(1e-3, _now - _fps_t1)
                _fps_n1 = 0
                _fps_t1 = _now
            if _now - _fps_t >= 30.0:
                _el = max(1e-3, _now - _fps_t)
                if _PL_ON:
                    _fps = (_pl["n"] - _pl["n0"]) / _el
                    _pl["n0"] = _pl["n"]
                else:
                    _fps = _fps_n / _el
                _fps_n = 0
                _fps_t = _now
                _p = np.asarray(_proc, np.float32) if _proc else np.zeros(1)
                _e = np.asarray(_lat_e2e, np.float32) if _lat_e2e \
                    else np.zeros(1)
                _c = np.asarray(_lat_cap, np.float32) if _lat_cap \
                    else np.zeros(1)
                _si = np.asarray(_st_inf, np.float32) if _st_inf \
                    else np.zeros(1)
                _sa = np.asarray(_st_anchor, np.float32) if _st_anchor \
                    else np.zeros(1)
                _sc = np.asarray(_st_chain, np.float32) if _st_chain \
                    else np.zeros(1)
                _sd = np.asarray(_st_disp, np.float32) if _st_disp \
                    else np.zeros(1)
                _sr = np.asarray(_st_run, np.float32) if _st_run \
                    else np.zeros(1)
                _lg = np.asarray(_loop_gap, np.float32) if _loop_gap \
                    else np.zeros(1)
                _pu = np.asarray(_poll_use, np.float32) if _poll_use \
                    else np.zeros(1)

                def _q(a, q):
                    return float(np.percentile(a, q))

                print(f"[d3d] fps={_fps:.1f} dda_seq={cap['zcc'].bridge_seq} "
                      f"proc avg={float(_p.mean()):.1f} "
                      f"p50={_q(_p, 50):.1f} "
                      f"p99={_q(_p, 99):.1f} max={float(_p.max()):.1f}ms "
                      f"[inf {_q(_si, 50):.1f} run {_q(_sr, 50):.1f} "
                      f"anchor {_q(_sa, 50):.1f} "
                      f"chain {_q(_sc, 50):.1f} "
                      f"disp {_q(_sd, 50):.1f}] "
                      f"e2e avg={float(_e.mean()):.1f} "
                      f"p50={_q(_e, 50):.1f} "
                      f"p99={_q(_e, 99):.1f} max={float(_e.max()):.1f}ms "
                      f"std={float(_e.std()):.2f} over18={_e2e_over[0]} "
                      f"(cap p50={_q(_c, 50):.1f}ms) "
                      f"lo={cap['stats'].lo:.2f} hi={cap['stats'].hi:.2f} "
                      f"rgbTV={cap['stats'].rgb_tv:.2f} "
                      f"thr={cap['stats'].cut_thr:.2f} "
                      f"cut={cap['stats'].cut_flag:.0f} "
                      f"shim={cap['shim'].times6()}"
                      f"{'' if not _PL_ON else (' pl(new=%d cap=%d chain=%d show=%d job=%.1fms)'
                       % (_pl.get('newf', 0) - _pl.get('newf0', 0),
                          _pl.get('caps', 0) - _pl.get('caps0', 0),
                          _pl.get('chains', 0) - _pl.get('chain0', 0),
                          _pl.get('presents', 0) - _pl.get('show0', 0),
                          float(getattr(cap['shim'], 'last_job_ms', 0.0))))}",
                      flush=True)
                _pl["newf0"] = _pl.get("newf", 0)
                _pl["caps0"] = _pl.get("caps", 0)
                _pl["chain0"] = _pl.get("chains", 0)
                _pl["show0"] = _pl.get("presents", 0)
                # 阶段尾部分解 + 非每帧开销的窗口 max（卡顿归因用）
                _wn = " ".join(f"{_k}={_v[0]:.1f}/{_v[1] / max(1, _v[2]):.2f}"
                               f"x{_v[2]}" for _k, _v in sorted(_win.items()))
                print(f"[d3d] 尾部分解 p99/max: inf {_q(_si, 99):.1f}/"
                      f"{float(_si.max()):.1f} run {_q(_sr, 99):.1f}/"
                      f"{float(_sr.max()):.1f} anchor {_q(_sa, 99):.1f}/"
                      f"{float(_sa.max()):.1f} chain {_q(_sc, 99):.1f}/"
                      f"{float(_sc.max()):.1f} disp {_q(_sd, 99):.1f}/"
                      f"{float(_sd.max()):.1f} | loop gap p50={_q(_lg, 50):.1f} "
                      f"p99={_q(_lg, 99):.1f} max={float(_lg.max()):.1f} "
                      f"poll p99={_q(_pu, 99):.1f} max={float(_pu.max()):.1f} "
                      f"| gap <18:{_hb[0]} 18-24:{_hb[1]} 24-34:{_hb[2]} "
                      f"34-60:{_hb[3]} >60:{_hb[4]} | max/avg(ms): {_wn}",
                      flush=True)
                for _i in range(5):
                    _hb[_i] = 0
                _e2e_over[0] = 0
                _win.clear()
                window.actual_fps = _fps
    finally:
        try:
            if _cap is not None:
                _cap["zcc"].stop()
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            pass
        print("[Main] D3D 主循环退出", flush=True)


def main(mode="Viewer"):
    print(f"[boot] main() 进入 +{_time_boot.perf_counter() - _T0:.1f}s"
          f"（含全部导入 + DML 模型加载）", flush=True)
    # ---- 渲染后端分派（Cubist-1.1-d3d）：d3d = 主路径，gl = 回退 ----
    if RENDER_BACKEND not in ("gl", "glfw", "opengl"):
        if mode == "Viewer":
            return _main_d3d(mode)
        print(f"[boot] D3D 后端暂只支持 Viewer 模式（mode={mode}）", flush=True)
    global glfw
    import glfw as _glfw_mod
    glfw = _glfw_mod
    # viewer（拖带 PyOpenGL 的 GL 包，纯 Python 导入 ~0.2s）必须在三线程
    # 孵化【之前】导入：线程一旦开跑，GIL 争抢会把这次导入饿成 21.6s
    # （cProfile 实测，= "切 3D 半天不出画面"的主分量）。单线程时导入。
    from viewer import StereoWindow
    # Start capture and processing threads —— capture/depth 线程挪到 viewer
    # 窗口创建【之后】（窗口创建 + moderngl ctx 也是 GIL 密集段，实测被
    # 饿成 2.4+4.2s）；hotkey 先行保证切换热键尽早上线。
    threading.Thread(target=_hotkey_thread, daemon=True).start()
    print(f"[boot] hotkey 孵化 +{_time_boot.perf_counter() - _T0:.1f}s", flush=True)
    
    # FPS tracking variables
    frame_count = 0
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_fps_update_time = time.perf_counter()
    current_fps = 0.0
    total_frames = 0
    window = None
    
    # FPS statistics tracking
    fps_values = deque(maxlen=300)  # Recent FPS values (O(1) bounded; for 1% low)
    max_fps_history = 300  # Keep last 300 FPS values (5 seconds at 60 FPS)
    avg_fps = 0.0
    low_fps_1_percent_avg = float('inf')  # Average of FPS below 1% percentile
    fps_update_interval = 5.0  # Update statistics every 5 seconds
    _last_fps_log = time.perf_counter()

    # Latency statistics tracking.
    # Use a deque + running sum so per-frame collection is O(1) (the old list
    # used .append()+.pop(0), and .pop(0) is O(n) on every single frame), and the
    # sliding-window average is O(1) instead of summing the whole history.
    max_latency_history = 300  # Keep same amount as FPS
    latency_history = deque()   # bounded manually to keep the running sum in sync
    latency_sum = 0.0           # running sum of latency_history for O(1) average
    avg_total_latency = 0.0
    
    try:
        if mode == "Viewer":
            # 颜色环需求（原定义在渲染循环段——先用后定义，被 SAME_SCREEN
            _gpu_pipeline = False
            try:
                import utils as _u
                _gpu_pipeline = bool(_u.settings.get("GPU Pipeline", False))
            except Exception:
                pass

            # 短路掩盖的暗雷，ruff F821 抓出后上移）
            _zcvv_r = os.environ.get("D2S_CAPTURE_GPU", "1").strip().lower()
            _zcc_want = (_zcvv_r not in ("", "0", "false", "off", "no")
                         and _gpu_pipeline)
            if SAME_SCREEN or _zcc_want:
                # 同屏 2D 启动：viewer 隐藏、采集休眠，永远等不到首帧 ——
                # 窗口尺寸直接取捕获屏几何（OUTPUT_RESOLUTION 与之同源）。
                # GPU-only（_zcc_want）同样必须走这里：采集线程在等环、环由
                # 渲染循环建——若此处阻塞等首帧，渲染循环永不开工 = 循环等待
                # （实测开机 26s 卡在 depth_q.take_new，等某个兜底碰巧解环；
                # rgb=None 起步 viewer 本就支持，首帧由 GPU 路径 feed）。
                from utils import get_monitor_size
                w, h = get_monitor_size(MONITOR_INDEX)
                rgb = depth = None
            else:
                # Get initial frame to determine window size (block until first frame arrives)
                _item0 = None
                while _item0 is None and not shutdown_event.is_set():
                    _item0 = depth_q.take_new(timeout=5.0)
                rgb, depth = _item0[0], _item0[1]
                w, h = rgb.shape[1], rgb.shape[0]  # HWC uint8 numpy
            # （同屏分支）w/h 来自显示器；跨屏分支来自帧 shape
            if DISPLAY_MODE == "Full-SBS":
                w = 2 * w
            if not SAME_SCREEN:
                # For local viewer only
                h = int(1280 / w * h)
                w = 1280
                
            viewer_kwargs = dict(
                capture_mode=CAPTURE_MODE,
                monitor_index=MONITOR_INDEX,
                depth_ratio=1.0,
                convergence=CONVERGENCE,
                display_mode=DISPLAY_MODE,
                fill_16_9=FILL_16_9,
                show_fps=SHOW_FPS,
                use_3d=USE_3D_MONITOR,
                fix_aspect=FIX_VIEWER_ASPECT,
                lossless_scaling=LOSSLESS_SCALING_SUPPORT,
                specify_display=STEREO_DISPLAY_SELECTION,
                stereo_display_index=STEREO_DISPLAY_INDEX,
                frame_size=(w, h),
                vsync=VSYNC,
                interleave_phase=INTERLEAVE_PHASE,
                parallax_px=PARALLAX_PX,
                parallax_curve=PARALLAX_CURVE,
                dibr_edge_snap=DIBR_EDGE_SNAP,
                depth_relief=DEPTH_RELIEF,
                warp_iterations=DIBR_WARP_ITERATIONS,
                disocclusion_fill=DISOCLUSION_FILL,
                cursor_overlay=(CURSOR_MODE == "overlay"),
                fullscreen_output=FULLSCREEN_OUTPUT,
            )
            try:
                window = StereoWindow(**viewer_kwargs)
                # capture/depth 线程在窗口+GL 就绪后孵化（GIL 争抢规避，见 main 顶部注释）
                threading.Thread(target=capture_loop, daemon=True).start()
                threading.Thread(target=process_depth_loop, daemon=True).start()
                # 隐藏光标会连形状一起丢掉（GetCursorInfo 返回 None），
                # 在拿到 DXGI 指针元数据之前先不隐藏
                # D2S_WDA=0 时跳过（调试用：让系统截图能拍到我们的输出）
                if FULLSCREEN_OUTPUT and os.environ.get("D2S_WDA", "1") != "0":
                    try:
                        import utils as _uu
                        _uu.hide_window_from_capture(window.window)   # WDA：全屏输出不入镜
                        print('[viewer] WDA 已设：全屏输出不入镜', flush=True)
                    except Exception as _e:
                        print(f'[viewer] WDA 设置失败: {_e}', flush=True)
                # 隐藏系统光标：mask 有"要画"的位（图像非空→句柄可取），
                # 但 color 的 alpha 全 0（合成后不可见）
                # SBS/TAB 全屏：无论是否自绘，都必须把系统光标藏掉——
                # 单支光标横跨两个半屏，比没有光标更糟。
                _sbs_full = FULLSCREEN_OUTPUT and DISPLAY_MODE in (
                    "Full-SBS", "Half-SBS", "Full-TAB", "Half-TAB")
                if FULLSCREEN_OUTPUT and (CURSOR_MODE != "off" or _sbs_full):
                    try:
                        import cursor_hide as _ch
                        _ch.start_watcher(_MODE_FILE)   # 与主循环同一个模式文件
                        print("[cursor] 全屏 3D 隐藏光标已启用", flush=True)
                    except Exception as _ce:
                        print(f"[cursor] 启动失败: {_ce}", flush=True)
                try:
                    import glfw as _glfw
                    globals()['_VIEWER_HWND'] = _glfw.get_win32_window(window.window)
                except Exception:
                    pass
            except Exception as e:
                raise
            # 构建标记：确认当前进程加载的是哪一版代码（execv 重启不改 PID/创建时间，
            # 光看进程启动时间判断不了版本，曾经因此误判过）
            try:
                import hashlib as _hl
                _base = os.path.dirname(os.path.abspath(__file__))
                _vp = os.path.join(_base, "viewer.py")
                if not os.path.exists(_vp):     # 发布包只有 .pyc
                    _vp = os.path.join(_base, "viewer.pyc")
                _mt = os.path.getmtime(_vp)
                _h = _hl.md5(io.open(_vp, "rb").read()).hexdigest()[:8]
                print(f"[build] {os.path.basename(_vp)} mtime="
                      f"{time.strftime('%H:%M:%S', time.localtime(_mt))} "
                      f"md5={_h} pid={os.getpid()} +"
                      f"{time.perf_counter() - _T0:.1f}s", flush=True)
            except Exception as _be:
                print(f"[build] 构建标记失败: {_be}", flush=True)
            print(f"[Main] Local Viewer Started +"
                  f"{time.perf_counter() - _boot_perf_t:.1f}s")
            _aa_note = "" if DEPTH_UPSAMPLER == "BILINEAR" else " (skipped: guided upsample)"
            print(f"[Main] display={DISPLAY_MODE} | upsampler={DEPTH_UPSAMPLER} | "
                  f"parallax={PARALLAX_PX:g}px | plane={CONVERGENCE} | "
                  f"AA={AA_STRENGTH}{_aa_note} | phase={INTERLEAVE_PHASE} | vsync={VSYNC}")
            
            # Process the first frame we already retrieved（同屏 2D 无首帧，跳过）
            render_start_time = time.perf_counter()
            total_latency = 0.0
            render_latency = 0.0
            if rgb is not None:
                window.update_frame(rgb, depth, current_fps, 0.0)  # initial latency unknown
                render_latency = time.perf_counter() - render_start_time
                total_latency = (render_start_time - _item0[3]) + render_latency  # t_post 起算
            thread_latencies['render'] = render_latency
            thread_latencies['total'] = total_latency
            
            # Main render loop
            _last_render_t = time.perf_counter()
            # Variables for latency update at 1Hz
            last_latency_display = 0.0    # last latency value shown
            last_fps_time = time.perf_counter()
            _win = window.window
            _should_close = glfw.window_should_close
            _is_set = shutdown_event.is_set
            _perf = time.perf_counter
            _latencies = thread_latencies
            _stereo_display_mode = DISPLAY_MODE
            # 渲染卡死自检钩子（验收用）：D2S_TEST_RENDER_STALL_S=3 时，第 120 帧
            # 后阻塞渲染线程 3 秒 —— 管线应继续推进并在阻塞结束后拿最新帧自愈。
            _stall_after = int(os.environ.get("D2S_TEST_RENDER_STALL_S", "0") or 0)
            _stall_frames = 0
            import ctypes as _ct2
            _isvisible2 = _ct2.windll.user32.IsWindowVisible
            _hwnd_main = globals().get('_VIEWER_HWND')
            import ctypes as _ct_r
            _u32r = _ct_r.windll.user32
            _u32r.SetProcessDPIAware()
            from winstructs import RECT as _RECT, POINTXY as _POINT, MONITORINFO as _MONINFO
            _same_screen = SAME_SCREEN
            _vis3d = None       # viewer 当前可见性跟随的 3D 模式（None=未初始化）
            _fs_done = False
            _last_style_t = 0.0  # overlay 样式重申节流（2s）
            _winfo_dbg = bool(os.environ.get("D2S_DEBUG_WINFO"))
            _fb_dbg = bool(os.environ.get("D2S_DEBUG_FB"))
            _fb_t = 0.0
            # 设置对话框的实时调参请求，每 10 帧查一次。
            # （对话框与皮肤零联动：ui.request 已废除，对话框是普通默认窗口）
            _LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            _PARAMS_REQ = os.path.join(_LOGDIR, "params.request")
            _params_mtime = 0.0
            _loop_n = 0
            # ---- Present 对齐实验：DWM 合成节拍钟 + swap 保护窗 ----
            # 渲染节奏由新帧驱动（相位随机），swap 落点相对合成边界随机分布
            # ——"赶上截止"凭运气。节拍钟给出边界预测；guard>0 时 swap 若
            # 落在下次边界前 guard 毫秒内，睡过边界再换（赌截止→稳上下一班）。
            # 时钟 2D 静止时自动陈旧失效（next_boundary=None，零行为变化）。
            import utils as _u
            try:
                _guard_ms = float(_u.settings.get("Present Guard (ms)", 0) or 0)
            except (TypeError, ValueError):
                _guard_ms = 0.0
            from viewer import DwmComposeClock
            _compose_clock = DwmComposeClock()
            _lead_hist = []      # swap→下次合成边界的提前量（ms）
            _guard_hits = 0
            # guard 以 60Hz 为基准配置；刷新率越高，帧预算越短，等比缩小
            if _guard_ms > 0:
                try:
                    _hz = float(FPS) or 60.0
                except Exception:
                    _hz = 60.0
                _guard_ms = _guard_ms * (60.0 / max(1.0, _hz))
                print(f"[present] 对齐开启：guard={_guard_ms:.2f}ms "
                      f"(基准 60Hz，当前 {_hz:g}Hz)", flush=True)
            # ---- M1 GPU 直通（深度 GPU 直通，色路仍 CPU）----
            # 颜色零拷贝开关（需要 GPU 深度管线：深度从 color_tex 采样）。
            # 默认开启，与采集侧同一套语义；显式关闭：D2S_CAPTURE_GPU=0
            _gpu_runner = None
            _gpu_keepalive_t = 0.0
            _gpu_tot_hist = []    # GPU 直通真全链（t_post→swap）滑动窗
            _bind_err_t = 0.0     # bind 失败日志限流
            import numpy as np   # GPU 分支的 ascontiguousarray 用（模块级无 np）
            if _gpu_pipeline:
                print("[gpu] GPU 直通管线启用（推理将移入渲染线程 GL 串行）",
                      flush=True)
            _locked_slot = None   # 本帧已上锁的环槽（swap-skip/异常路径的保底锚）
            _last_frame_gen = -1  # 最后成功处理帧的 gen（换代间隙跳过保活）
            # ---- 卡死看门狗：渲染循环 3 秒无心跳 → faulthandler 打全部线程栈 ----
            def _hang_watchdog():
                import faulthandler as _fh
                import time as _tw
                last, last_t = -1, _tw.time()
                while not _is_set():
                    _tw.sleep(1.0)
                    b = globals().get("_loop_beat", 0)
                    if b != last:
                        last, last_t = b, _tw.time()
                    elif _tw.time() - last_t > 3.0:
                        try:
                            print(f"[hang] 渲染循环 {_tw.time() - last_t:.1f}s 无心跳，"
                                  f"dump 线程栈:", flush=True)
                            _fh.dump_traceback(file=sys.stdout)
                        except Exception:
                            pass
                        _tw.sleep(5.0)
                        last_t = _tw.time()
            try:
                import threading as _thw
                if not globals().get("_hang_wd_started"):
                    globals()["_hang_wd_started"] = True
                    _thw.Thread(target=_hang_watchdog, daemon=True,
                                name="hang-watchdog").start()
            except Exception:
                pass

            def _gpu_depth_step(rich):
                """渲染线程 GPU 深度一步（颜色环帧/普通帧两分支共用核心）：
                懒建 runner → preprocess → infer → 分位归一 → GPU 后处理。
                rich=True 保留 SBS 分支的分段计时/转储/统计探针。"""
                nonlocal _gpu_runner
                if _gpu_runner is None:
                    _dr = int(_depth_mod.DEPTH_RESOLUTION)
                    _sess = _depth_mod.depth_model.session
                    _is_dml = (_sess.get_providers()[0] == "DmlExecutionProvider")
                    # 预处理纹理用 R32F：R16F 虽然回读少 0.12ms，但
                    # fp16→fp32 转换让 pack 多 0.23ms，实测净亏，不采用
                    window.gpu_depth_init(_dr)
                    if _is_dml:
                        from gpu.dml_gl import DmlDepthRunner
                        _gpu_runner = DmlDepthRunner(
                            _sess, None, None, _dr,
                            tex_prep=window._prep_tex,
                            tex_raw=window._raw_tex)
                    else:
                        from gpu.cuda_gl import GpuDepthRunner
                        _gpu_runner = GpuDepthRunner(
                            _sess, window._prep_tex.glo,
                            window._raw_tex.glo, _dr)
                    GPU_ACTIVE[0] = True
                    try:
                        from gpu import placement as _pl
                        _pl.note_inference(_pl.inference_note(bool(_is_dml)))
                        for _ln in _pl.report():
                            print(_ln, flush=True)
                    except Exception:
                        pass
                    # 2026-11-20 用户定案：MC 运动补偿默认停用（检测问题链
                    # 多于收益：假切/匹配失败/局部突变），产品回归纯 EMA——
                    # postprocess 走 legacy ema_pass_legacy。D2S_MC=1 可重新
                    # 开启（代码与离线证据链 tools/mc_*_test.py 全保留）。
                    window._mc_on = (os.environ.get("D2S_MC", "0") == "1")
                    if window._mc_on:
                        print(f"[mc] 运动补偿EMA实验性启用（D2S_MC=0 关闭，"
                              f"离线证据 tools/mc_*_test.py）", flush=True)
                    else:
                        print("[mc] 运动补偿停用，纯 EMA（D2S_MC=1 可开启）",
                              flush=True)
                    print(f"[gpu] runner 就绪({'SBS' if rich else 'SBS-2'}) "
                          f"res={_dr}", flush=True)
                _ts0 = _perf()
                _t_pre0 = _t_pre1 = _t_inf1 = _t_post0 = _t_post1 = None
                # 收敛锚读回【延迟一帧】（2026-11-21 优化）：在此刻上一帧的
                # GL 队列已被 swap（vsync）排空，读回同步等待≈0，只付 200KB
                # 拷贝+Otsu ~0.3ms——原位置（本帧 postprocess 之后）要等整条
                # 深度链跑完，2.4ms/帧。depth_tex 此刻=上一帧时域后输出，
                # 锚滞后一帧对慢统计量+死区零影响。首帧 _post 未建→None 跳过。
                try:
                    _t_conv0 = _perf()
                    if _auto_conv_on():
                        _dm = window.read_depth_median()
                        if _dm is not None:
                            _depth_mod.LAST_MEDIAN = min(1.0, max(0.0, float(_dm)))
                            _depth_mod.LAST_P30 = _depth_mod.LAST_MEDIAN
                    auto_convergence_update(None)
                    _t_conv1 = _perf()
                except Exception:
                    _t_conv0 = _t_conv1 = None
                if rich:
                    _t_pre0 = _perf()
                window.gpu_preprocess()
                if rich:
                    _t_pre1 = _perf()
                raw = _gpu_runner.infer(window._raw_tex)
                if rich:
                    _t_inf1 = _perf()
                    _ddir = os.environ.get("D2S_DEPTH_DUMP", "")
                    if _ddir:
                        try:
                            import cv2 as _cv2
                            import time as _t2
                            _v2 = (raw if raw.ndim == 2 else raw[0])
                            _lo_ = float(_v2.min())
                            _hi_ = float(_v2.max())
                            _n2 = (_v2 - _lo_) / max(1e-9, _hi_ - _lo_) * 255.0
                            os.makedirs(_ddir, exist_ok=True)
                            _cv2.imwrite(
                                os.path.join(_ddir, f"depth_{int(_t2.time()*1000)}.png"),
                                _n2.astype(np.uint8))
                        except Exception:
                            pass
                lo, hi, _med_raw = percentile_stats(raw)
                # 残差切阀（CPU 判据，2026-11-21）：z_t vs z_{t−1} 在推理回读
                # numpy 上算——GPU 1x1 pass 版的 4 字节读回会把 GL+DML 流水线
                # 拽成同步（+7ms/帧，fps 60→47 实锤），此版零新增读回。
                # 与 depth.postprocess_depth 同函数同域（parity 逐位）。
                try:
                    from depth_norm import z_sub as _z_sub, cut_hit as _cut_hit
                    _z = _z_sub(raw, lo, hi)
                    _cut = _cut_hit(_z, globals().get("_cut_prev_z"))
                    globals()["_cut_prev_z"] = _z
                    if _cut and _perf() - globals().get("_cut_t", 0.0) > 0.5:
                        globals()["_cut_t"] = _perf()
                        print("[cut] 残差面积超阈 → 时域重置", flush=True)
                except Exception:
                    _cut = False
                # LAST_MEDIAN 不再用 raw 中位数：与 CPU 路径（depth.py
                # _publish_median 用平滑后深度）不一致，且 raw 单帧噪声/闪变
                # 直通收敛面——改由下方 postprocess 之后按时域后深度发布。
                smooth = str(getattr(_depth_mod, "DEPTH_SMOOTHING", "ema"))
                alpha = (ema_alpha(float(_depth_mod.DEPTH_SMOOTHING_FRAMES), _perf())
                         if smooth != "off" else 1.0)
                aa = (float(getattr(_depth_mod, "AA_STRENGTH", 0.0)) * 0.5
                      * (int(_depth_mod.DEPTH_RESOLUTION) / 224.0))
                if rich:
                    _t_post0 = _perf()
                # MC-EMA：postprocess 前跑运动补偿步（D2S_MC=0 关闭走旧路径）
                _mc_cut = False
                if getattr(window, "_mc_on", False):
                    try:
                        _zn = z_norm(raw, lo, hi)
                        _mc_cut, _mst = window.mc_step(window._raw_tex, lo, hi, _zn)
                        window._mc_cut_now = _mc_cut
                        if _mc_cut:
                            alpha = 1.0        # 切镜头：整幅直通
                        globals()["_mc_acc_n"] = globals().get("_mc_acc_n", 0) + 1
                        if _mst.get("n") and _perf() - globals().get("_mc_t", 0.0) > 5.0:
                            globals()["_mc_t"] = _perf()
                            print("[mc] gpu %.2fms cut=%s n=%d "
                                  "histL1=%.3f pix=%.3f trust=%.3f" %
                                  (_mst.get("gpu", 0), _mst.get("cut"),
                                   _mst.get("n", 0),
                                   _mst.get("hist_l1", -1),
                                   _mst.get("pix_area", -1),
                                   _mst.get("trust", -1)), flush=True)
                    except Exception as _mce:
                        if _perf() - globals().get("_mc_err_t", 0.0) > 5.0:
                            globals()["_mc_err_t"] = _perf()
                            print(f"[mc] 步失败（退旧EMA路径本帧）: "
                                  f"{type(_mce).__name__}", flush=True)
                            import traceback as _tb_mc
                            _tb_mc.print_exc()
                window.postprocess_gpu(window._raw_tex, lo, hi, alpha, aa,
                                       cut=_cut)
                # （收敛锚读回已延迟到下一步开头——见 _gpu_depth_step 顶部）
                if rich:
                    _t_post1 = _perf()
                    if os.environ.get("D2S_DH_MARK"):
                        print("[dh] depth->True", flush=True)
                window._frame_has_depth = True
                thread_latencies['gpu'] = _perf() - _ts0
                if rich:
                    _stage_ms = thread_latencies['gpu'] * 1000.0
                    _scn = globals().get("_zc_stage_n", 0) + 1
                    globals()["_zc_stage_n"] = _scn
                    if _scn % 10 == 1 and os.environ.get("D2S_ZC_DEBUG"):
                        print("[zcc] 分段 pre=%.1f infer=%.1f post=%.1f 总=%.1fms"
                              % ((_t_pre1 - _t_pre0) * 1000.0,
                                 (_t_inf1 - _t_pre1) * 1000.0,
                                 (_t_post1 - _t_post0) * 1000.0,
                                 _stage_ms), flush=True)
                    # 分段累计（5s [dml] 行输出）：post 再细切出收敛锚读回
                    _seg = globals().setdefault("_dml_seg",
                                                {"pre": 0.0, "infer": 0.0,
                                                 "conv": 0.0, "post": 0.0,
                                                 "n": 0})
                    _seg["pre"] += (_t_pre1 - _t_pre0) * 1000.0
                    _seg["infer"] += (_t_inf1 - _t_pre1) * 1000.0
                    _seg["conv"] += ((_t_conv1 - _t_conv0) * 1000.0
                                     if _t_conv0 is not None
                                     and _t_conv1 is not None else 0.0)
                    _seg["post"] += (_t_post1 - _t_post0) * 1000.0
                    _seg["n"] += 1
                    try:
                        globals()["_dml_sum"] = globals().get("_dml_sum", 0.0) + _stage_ms
                        _tm = getattr(_gpu_runner, "times", None)
                        if _tm:
                            _acc = globals().get("_dml_acc")
                            if _acc is None:
                                _acc = globals()["_dml_acc"] = {
                                    "read": 0.0, "pack": 0.0,
                                    "run": 0.0, "write": 0.0, "n": 0}
                            for _k in ("read", "pack", "run", "write"):
                                _acc[_k] += float(_tm.get(_k, 0.0))
                            _acc["n"] += 1
                        if _perf() - globals().get("_dml_t", 0.0) > 5.0:
                            globals()["_dml_t"] = _perf()
                            _acc2 = globals().get("_dml_acc") or {}
                            _n = max(1, _acc2.get("n", 0))
                            _frames = max(1, globals().get("_dml_frames", 0))
                            _avg = globals().get("_dml_sum", 0.0) / _frames
                            if _n:
                                _seg2 = globals().get("_dml_seg") or {}
                                _sn = max(1, _seg2.get("n", 0))
                                print("[dml] read %.2f pack %.2f run %.2f "
                                      "write %.2f | 平均 %.2f ms/帧 "
                                      "(%d/%d 帧跑了推理)"
                                      % (_acc2["read"] / _n * 1000,
                                         _acc2["pack"] / _n * 1000,
                                         _acc2["run"] / _n * 1000,
                                         _acc2["write"] / _n * 1000,
                                         _avg,
                                         _acc2["n"], _frames), flush=True)
                                if _seg2.get("n"):
                                    print("[dml] 分段 pre=%.1f infer=%.1f "
                                          "conv锚=%.1f post=%.1f ms/帧"
                                          % (_seg2["pre"] / _sn,
                                             _seg2["infer"] / _sn,
                                             _seg2["conv"] / _sn,
                                             _seg2["post"] / _sn), flush=True)
                                    for _k in ("pre", "infer", "conv", "post"):
                                        _seg2[_k] = 0.0
                                    _seg2["n"] = 0
                                for _k in ("read", "pack", "run", "write"):
                                    _acc2[_k] = 0.0
                                _acc2["n"] = 0
                                globals()["_dml_sum"] = 0.0
                                globals()["_dml_frames"] = 0
                    except Exception:
                        pass

            while not _should_close(_win) and not _is_set():
                globals()["_loop_beat"] = globals().get("_loop_beat", 0) + 1

                # ---- 锁槽保底：上一拍若因 swap 跳过/异常未解锁，这里必释放 ----
                # （首帧皮肤未显、2D/3D 切换、bind 后异常等路径都会跳过 swap 点
                #  的解锁；泄漏攒满 3 槽 = 采集端全锁丢帧 = 用户实测 fps<1）
                if _locked_slot is not None:
                    try:
                        window.gpu_unlock(_locked_slot)
                    except Exception:
                        pass
                    _g0u = GPU_COLOR[0]
                    if _g0u is not None and _locked_slot < len(_g0u.gl_locked):
                        _g0u.gl_locked[_locked_slot] = False
                    _locked_slot = None

                # ---- 自重启请求：不依赖托盘守护，自己 execv 换新进程 ----
                if _loop_n % 10 == 1:
                    try:
                        _rr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "restart.request")
                        _rs = os.stat(_rr)
                        if _rs.st_mtime != globals().get("_restart_mtime", 0):
                            globals()["_restart_mtime"] = _rs.st_mtime
                            try:
                                os.remove(_rr)
                            except OSError:
                                pass
                            print("[settings] 收到重启请求，自重启…", flush=True)
                            try:
                                window.update_frame(None)
                                glfw.terminate()
                            except Exception:
                                pass
                            # 先摘掉托盘图标，否则重启后旧图标残留
                            try:
                                import tray as _tray_stop
                                _tray_stop.stop()
                            except Exception:
                                pass
                            os.chdir(os.path.dirname(os.path.abspath(__file__)))
                            # execv 前先摘单实例互斥体：Windows 的 os.execv 是
                            # "先拉新进程再退旧进程"，重叠窗口里子进程会看到
                            # ERROR_ALREADY_EXISTS → 弹"已在运行"框后退出 =
                            # 设置切换时的"报错+闪退"（SBS→行交错实测复现）。
                            # 先关句柄，子进程出生即成为持有人。
                            try:
                                import ctypes as _ct_rm
                                if globals().get("_SINGLE_MUTEX"):
                                    _ct_rm.WinDLL("kernel32").CloseHandle(
                                        _ct_rm.c_void_p(globals()["_SINGLE_MUTEX"]))
                                    globals()["_SINGLE_MUTEX"] = None
                            except Exception:
                                pass
                            os.execv(sys.executable,
                                     [sys.executable, "-u", "-m", "main"] +
                                     [a for a in sys.argv[1:]])
                    except FileNotFoundError:
                        pass
                    except Exception as _re:
                        print(f"[settings] 重启请求处理失败: {_re}", flush=True)

                # ---- 设置对话框实时调参轮询 ----
                _loop_n += 1
                if _loop_n % 10 == 0:
                    try:
                        _st = os.stat(_PARAMS_REQ)
                        if _st.st_mtime != _params_mtime:
                            _params_mtime = _st.st_mtime
                            import json as _json
                            # utf-8-sig：容忍 PowerShell 写出的 BOM
                            with open(_PARAMS_REQ, encoding="utf-8-sig") as _f:
                                _req = _json.load(_f)
                            # 收敛相关：自动收敛读的是全局，必须在这里同步
                            if "Convergence Bias" in _req:
                                try:
                                    _AUTO_CONV_BIAS[0] = float(_req["Convergence Bias"])
                                except Exception:
                                    pass
                            if "Depth Plain Normalize" in _req:
                                try:
                                    import depth as _dm2
                                    _dm2.set_plain_norm(bool(_req["Depth Plain Normalize"]))
                                    print(f"[depth] 原版纯净 = {bool(_req['Depth Plain Normalize'])}",
                                          flush=True)
                                except Exception:
                                    pass
                            if "Depth Raw Output" in _req:
                                try:
                                    import depth as _dm
                                    _dm.set_raw_mode(bool(_req["Depth Raw Output"]))
                                    print(f"[depth] 原样输出 = {bool(_req['Depth Raw Output'])}",
                                          flush=True)
                                except Exception:
                                    pass
                            if "Anti-aliasing" in _req:
                                try:
                                    # 面板值 → AA_STRENGTH（= 2×面板），主循环每帧
                                    # getattr 读取，所以下一帧即生效
                                    _depth_mod.AA_STRENGTH = 2.0 * float(_req["Anti-aliasing"])
                                    print(f"[depth] AA sigma = {_depth_mod.AA_STRENGTH:g} texel",
                                          flush=True)
                                except Exception:
                                    pass
                            if "Depth Edge Snap Gain" in _req:
                                try:
                                    window.set_edge_snap_gain(float(_req["Depth Edge Snap Gain"]))
                                except Exception:
                                    pass
                            window.apply_live_params(_req)
                            print(f"[settings] live params applied: {_req}",
                                  flush=True)
                            os.remove(_PARAMS_REQ)   # 消费即删：陈旧请求不跨重启
                    except FileNotFoundError:
                        pass
                    except Exception as _e:
                        print(f"[settings] params.request 损坏已删除: {_e}", flush=True)
                        try:
                            os.remove(_PARAMS_REQ)
                        except OSError:
                            pass

                # 同屏自投射：2D 隐藏 viewer（原生显示，零开销），3D 显窗转换。
                # 显/隐必须在本线程（glfw 建窗线程）执行；显窗后重申位置/
                # 全屏/采集排除，隐藏后下面的暂停分支自动接管休眠。
                if _same_screen:
                    _want3d_now = mode_bus.read_mode()
                    if _want3d_now != _vis3d:
                        _vis3d = _want3d_now
                        if _want3d_now:
                            # 只负责显窗 + 清掉驻留期的陈旧帧：定位/全屏/
                            # 采集排除由首帧块（update_frame 首帧路径）完成，
                            # 这里绝不重复 toggle_fullscreen（会切回去）。
                            depth_q.take_new()   # 丢弃陈旧槽（防一帧旧画闪现）
                            glfw.show_window(_win)
                            window._reassert_overlay_style()
                            print("[Main] 同屏 3D：viewer 已上屏", flush=True)
                        else:
                            glfw.hide_window(_win)
                            print("[Main] 2D：转换关闭（viewer 隐藏，原生显示）", flush=True)

                # 周期重申 topmost（~2s）：别的窗口一点'置顶'就会压到
                # overlay 头上盖住 3D 画面（topmost band 内后到者胜）。
                # 全屏 SBS：WDA 会被样式/全屏重置清掉，周期重申（每 2s）
                if (FULLSCREEN_OUTPUT and os.environ.get("D2S_WDA", "1") != "0"
                        and _perf() - globals().get("_last_wda", 0.0) > 2.0):
                    globals()["_last_wda"] = _perf()
                    try:
                        _v = _ct_r.c_ulong(0)
                        _vhw2 = glfw.get_win32_window(_win)
                        _u32r.GetWindowDisplayAffinity(_ct_r.c_void_p(_vhw2), _ct_r.byref(_v))
                        if _v.value != 0x11:
                            if _u32r.SetWindowDisplayAffinity(_ct_r.c_void_p(_vhw2), 0x11):
                                print("[viewer] WDA 重申成功", flush=True)
                            else:
                                print(f"[viewer] WDA 重申失败（当前 0x{_v.value:X}）", flush=True)
                    except Exception as _we:
                        print(f"[viewer] WDA 重申异常: {_we}", flush=True)

                # 全屏输出：每帧把本窗口提到最前（HWND_TOPMOST）。
                # PotPlayer 等会把自己置顶，双方都在 topmost band 时
                # "后到者胜" → 压住我们的全屏输出。用户定案每帧重申
                # （2026-11-18：被压下去的视觉影响 > 节流省下的开销）。
                # 成本实测（tools/bench_topmost.py，GTX1650/19045，n=1000）：
                # 中位 0.089ms = 帧预算 0.54%；尾部 p99 0.7~2ms（DWM 忙时），
                # 已知并接受。若真实会话 1% low 回归，回退口径 =
                # 恢复 _ztop_t 50ms 节流（历史曾走 0.5s→200ms→每帧→50ms）。
                if FULLSCREEN_OUTPUT:
                    try:
                        _vhwz = glfw.get_win32_window(_win)
                        _u32r.SetWindowPos(_ct_r.c_void_p(_vhwz),
                                           _ct_r.c_void_p(-1),      # HWND_TOPMOST
                                           0, 0, 0, 0,
                                           0x0002 | 0x0001 | 0x0010)  # NOMOVE|NOSIZE|NOACTIVATE
                    except Exception:
                        pass

                # 内容区域：只对前台窗口做 3D，桌面/任务栏原样投射（~8Hz 更新）
                if FULLSCREEN_OUTPUT and _perf() - globals().get("_last_crect", 0.0) > 0.12:
                    globals()["_last_crect"] = _perf()
                    try:
                        if True:
                            _fg = _u32r.GetForegroundWindow()
                            _mine = glfw.get_win32_window(_win)
                            _desktop = _u32r.GetShellWindow()
                            if (_fg and _fg != _mine and _fg != _desktop
                                    and not _is_shell_window(_fg)
                                    and _u32r.IsWindowVisible(
                                        _ct_r.c_void_p(_fg))):
                                _cr = _RECT()
                                # 优先用 DWM 扩展边框（去掉不可见的调整边框）
                                _ok = _u32r.DwmGetWindowAttribute(
                                    _ct_r.c_void_p(_fg), 9, _ct_r.byref(_cr),
                                    _ct_r.sizeof(_RECT)) == 0
                                if not _ok:
                                    _u32r.GetWindowRect(_ct_r.c_void_p(_fg), _ct_r.byref(_cr))
                                if _cr.R > _cr.L + 16 and _cr.B > _cr.T + 16:
                                    window.set_content_rect(_cr.L, _cr.T, _cr.R, _cr.B)
                                else:
                                    window.set_content_rect()
                            else:
                                window.set_content_rect()   # 前台是桌面/我们自己 → 全屏都做
                    except Exception as _ce2:
                        window.set_content_rect()

                if _same_screen and _perf() - _last_style_t > 2.0:
                    # 样式周期重申：GLFW 的任何属性改写（全屏切换等）都可能
                    # 重置分层 alpha/穿透位，事件驱动的重申挡不住"最后一次
                    # churn 之后没人补"的时序 —— 周期性兜底，最多 2s 自愈。
                    # （置顶重申原在此处 2s 一次，已被全屏置顶 50ms 块覆盖，
                    #   2026-11-18 摘除 _pin_topmost 调用。）
                    _last_style_t = _perf()
                    window._reassert_overlay_style()
                    if _fb_dbg and _perf() - _fb_t > 3.0:
                        _fb_t = _perf()
                        try:
                            import numpy as _np, cv2 as _cv2
                            _fb = window.ctx.screen.read(components=3, alignment=1)
                            _fw, _fh = glfw.get_framebuffer_size(_win)
                            _im = _np.frombuffer(_fb, _np.uint8).reshape(_fh, _fw, 3)[::-1]
                            _cv2.imwrite(os.path.join(os.environ.get("TEMP", "."), "d2s-fb.png"), _im)
                            print(f"[fb-dump] {_fw}x{_fh} mean={_im.mean():.1f} std={_im.std():.1f}", flush=True)
                        except Exception as _e:
                            print(f"[fb-dump] failed: {_e}", flush=True)
                    if _winfo_dbg:
                        _mons = [(glfw.get_monitor_pos(m),
                                  glfw.get_video_mode(m).size.width,
                                  glfw.get_video_mode(m).size.height)
                                 for m in glfw.get_monitors()]
                        print(f"[winfo] win_pos={glfw.get_window_pos(_win)} "
                              f"win_size={glfw.get_window_size(_win)} "
                              f"fb={glfw.get_framebuffer_size(_win)} "
                              f"mons={_mons}", flush=True)

                # Pause rendering while the viewer window is hidden: drawing
                # the cached frame at full rate kept the GPU busy in the
                # background state.
                if _hwnd_main and not _isvisible2(_hwnd_main):
                    # 2D/隐藏态也要把颜色环建好（否则切 3D 才初始化 →
                    # 首个 3D 画面等数秒）；建好后此分支照旧休眠让出 CPU
                    _zcc_ensure_inline(window, _zcc_want)
                    time.sleep(0.2)
                    glfw.poll_events()
                    continue

                # 等新结果（最多 50ms 一段，让 300ms keepalive 检查仍然及时）。
                # 3D/2D 都由"有新帧才画"限速：采集端 vsync 60/s，这里不会空转。
                # M2：GPU 直通模式直接消费采集线程的 GPU 帧句柄
                # 颜色零拷贝（v4 内联）：设备/duplication/环全部在**渲染线程**
                # 建好并注册。跨线程版本在 app 里 AcquireNextFrame 会永久阻塞
                # （看门狗实证），独立进程同代码正常，所以彻底去掉跨线程。
                    _zcc_ensure_inline(window, _zcc_want)
                _gc0 = (GPU_COLOR[0]
                        if (GPU_COLOR[0] is not None
                        and getattr(GPU_COLOR[0], "inline", False)) else None)
                if _gc0 is not None:
                    # 内联取帧：acquire → CopyResource，在同一线程完成。
                    # avoid = 当前正在显示的槽：绝不覆盖它（否则画面撕裂/闪烁）
                    _gc0.avoid = globals().get("_zcc_shown_slot", -1)
                    _idx = _gc0.poll()
                    _item = _gc0._last_pub if _idx >= 0 else None
                else:
                    _item = depth_q.take_new(timeout=0.05)
                _got = _item is not None
                # D2S_BENCH=1：没有新帧时复用上一帧继续走完整帧路径（含深度），
                # 用来实测"逐帧深度 + 渲染"的真实上限，而不是靠耗时推算。
                if not _got and os.environ.get("D2S_BENCH") \
                        and globals().get("_bench_item") is not None:
                    _item = globals()["_bench_item"]
                    _got = True
                elif _got:
                    globals()["_bench_item"] = _item
                _tn = globals().get("_zc_take_n", 0) + (1 if _got else 0)
                globals()["_zc_take_n"] = _tn
                _tnow = _perf()
                if (os.environ.get("D2S_ZC_DEBUG")
                        and _tnow - globals().get("_zc_take_t", 0.0) >= 1.0):
                    print(f"[zc] take/秒 = {_tn - globals().get('_zc_take_prev', 0)}",
                          flush=True)
                    globals()["_zc_take_prev"] = _tn
                    globals()["_zc_take_t"] = _tnow
                # 空闲/恢复诊断：视频暂停时采集端不再给新帧（fps→0 属预期），
                # 但"短暂退出 SBS"必须能对上时间点，所以记录空转时长与恢复首帧。
                try:
                    _now_i = _perf()
                    if _got:
                        if globals().get("_idle_since"):
                            # 限频 1 条/秒：非阻塞 poll 下 got↔not-got 每秒翻转
                            # 25-50 次，原样打印=渲染线程 I/O 洪水（app.log 一晚
                            # 28,735 行），且每次 print 带 flush 阻塞渲染线程
                            if _now_i - globals().get("_idle_pr_t", 0.0) > 1.0:
                                globals()["_idle_pr_t"] = _now_i
                                print(f"[idle] 恢复：空转 "
                                      f"{_now_i - globals()['_idle_since']:.2f}s，"
                                      f"首帧处理 {(_perf() - _now_i) * 1000:.0f}ms",
                                      flush=True)
                            globals()["_idle_since"] = 0.0
                    else:
                        if not globals().get("_idle_since"):
                            globals()["_idle_since"] = _now_i
                        elif _now_i - globals()["_idle_since"] > 0.8:
                            if not globals().get("_idle_logged"):
                                globals()["_idle_logged"] = True
                                print("[idle] 无新帧开始（采集端停止推帧）", flush=True)
                        if globals().get("_idle_since") and _now_i - globals()["_idle_since"] > 0.8:
                            globals()["_idle_logged"] = True
                except Exception:
                    pass
                # 端到端延迟（覆盖层显示用）：从【采集时刻】到【即将上屏】
                if _got and len(_item) > 3:
                    try:
                        globals()["_e2e_ms"] = max(0.0, (_perf() - _item[2]) * 1000.0)
                    except Exception:
                        pass
                if not _got:
                    # 无新帧也重新呈现上一帧：暂停、或视频本身 24fps 时，
                    # 采集端不会给 60 帧/秒；这里不补就会出现"输出跟着掉到
                    # 24fps"= 用户看到的 FPS 骤降/顿挫。重呈现不重算深度。
                    try:
                        _pt_now = _perf()
                        # D2S_FREE_RUN=1：去掉节流，测渲染循环的最高帧率
                        # 正常节拍 = REPRESENT_DT（按显示器实际刷新率稍提前）
                        _min_dt = (0.0 if (os.environ.get("D2S_FREE_RUN")
                                           or os.environ.get("D2S_BENCH"))
                                   else REPRESENT_DT)
                        _dt = _pt_now - globals().get("_last_present_t", 0.0)
                        if _dt >= _min_dt:
                            globals()["_last_present_t"] = _pt_now
                            # D2S_NO_REPRESENT_DRAW=1：诊断用——无新帧时
                            # 只交换缓冲、不重画（验证闪烁是否来自重呈现路径）
                            if not os.environ.get("D2S_NO_REPRESENT_DRAW"):
                                window._zc_from = "represent"
                                _render_frame(window)
                            glfw.swap_buffers(_win)
                            if _min_dt == 0.0:
                                # 自由运行：统计实际呈现帧率（不受采集 60Hz 限制）
                                _fn = globals().get("_free_n", 0) + 1
                                globals()["_free_n"] = _fn
                                _ft = _perf()
                                if _ft - globals().get("_free_t", 0.0) >= 1.0:
                                    print(f"[free] 呈现/秒 = "
                                          f"{_fn - globals().get('_free_p', 0)}",
                                          flush=True)
                                    globals()["_free_p"] = _fn
                                    globals()["_free_t"] = _ft
                        else:
                            # 去掉 16ms 睡眠：取帧等待已由 acquire(8ms) 承担，
                            # 再睡 16ms 会把周期拖到 ~24ms（实测只有 42fps）。
                            time.sleep(0.001)
                    except Exception:
                        pass
                    continue
                if _got:
                    _dbg_n = globals().get("_dbg_n", 0) + 1
                    globals()["_dbg_n"] = _dbg_n
                    if _dbg_n % 120 == 0:
                        try:
                            print(f"[lat] len={len(_item)} types={[type(x).__name__ for x in _item]}",
                                  flush=True)
                        except Exception as _de:
                            print(f"[lat] 诊断失败: {_de}", flush=True)
                if _got:
                    if _gpu_pipeline and (_gc0 is not None):
                        _gen, _ridx, t_pre, t_post, _wh, _cpu, _ftag = _item
                        rgb = depth = None
                        _g0 = _gc0
                        if _g0 is None or _g0.gen != _gen:
                            _got = False    # 换代在途帧（重开会话瞬间）
                        else:
                            _last_frame_gen = _gen  # 成功处理：记录 gen
                            if getattr(window, "_dx", None) is None:
                                from gpu.dx_interop import DxInterop
                                window._dx = DxInterop(_g0.dev.p)
                            # 等采集线程把该槽写完之后再加锁（否则
                            # wglDXLockObjectsNV 与 CopyResource 竞态）
                            _wr = getattr(_g0, "writing", None)
                            if _wr is not None:
                                for _ in range(200):
                                    if not _wr[_ridx]:
                                        break
                                    time.sleep(0.0002)
                            _g0.gl_locked[_ridx] = True   # 采集线程跳过此槽
                            _locked_slot = _ridx
                            _tb0 = _perf()
                            try:
                                window.gpu_bind_color(
                                    window._dx, _gen, _ridx, _g0.ring_ptrs(),
                                    _wh[0], _wh[1])
                                globals()["_zcc_shown_slot"] = _ridx
                                if os.environ.get("D2S_ZC_PROBE"):
                                    try:
                                        _cm = window.gpu_color_mean()
                                        if _cm is not None and float(_cm[0]) < 2.0:
                                            print(f"[probe] 黑帧 slot={_ridx} "
                                                  f"mean={float(_cm[0]):.2f}",
                                                  flush=True)
                                    except Exception:
                                        pass
                                _bm = globals().setdefault("_bind_stat", [0.0, 0])
                                _bm[0] += (_perf() - _tb0) * 1000.0
                                _bm[1] += 1
                                if _bm[1] >= 60:
                                    if os.environ.get("D2S_ZC_DEBUG"):
                                        print(f"[zcc] bind 平均 "
                                              f"{_bm[0]/_bm[1]:.2f}ms", flush=True)
                                    _bm[0] = 0.0
                                    _bm[1] = 0
                            except Exception as _be:
                                # 丢帧不炸线程：换代瞬间的互操作异常
                                # 若向上传播会杀死整个渲染循环（实测主进程退出）
                                _g0.gl_locked[_ridx] = False
                                _got = False
                                if _perf() - _bind_err_t > 5.0:
                                    _bind_err_t = _perf()
                                    import traceback as _tb2
                                    print(f"[gpu] bind 失败丢帧 gen={_gen}: "
                                          f"{type(_be).__name__}", flush=True)
                                    _tb2.print_exc()
                    else:
                        rgb, depth, t_pre, t_post = _item[:4]

                    if _stall_after and _stall_frames == 120:
                        print(f"[render-stall-test] blocking render {_stall_after}s",
                              flush=True)
                        time.sleep(_stall_after)
                    _stall_frames += 1

                    current_time = _perf()
                    total_latency = current_time - t_post      # 像素落地起算（真延迟）
                    _lat_pre = current_time - t_pre            # 旧口径（含 grab 阻塞+4ms 轮询），过渡期对照

                    latency_history.append(total_latency)
                    latency_sum += total_latency
                    if len(latency_history) > max_latency_history:
                        latency_sum -= latency_history.popleft()
                    frame_count += 1
                    total_frames += 1
                    elapsed = current_time - last_time
                    if elapsed >= 1.0:
                        current_fps = frame_count / elapsed
                        globals()["_cur_fps"] = current_fps
                        frame_count = 0
                        last_time = current_time

                        # Measured pipeline fps for tuning - lands in
                        # %TEMP%\d2s-main-out.log. Disable via _FPS_LOG.
                        if _FPS_LOG and current_time - _last_fps_log >= 5.0:
                            _last_fps_log = current_time
                            # Present 相位：swap 提前量分布 + 危险区占比 + guard 命中
                            _ptail = ""
                            if _lead_hist:
                                _ls = sorted(_lead_hist)
                                _late2 = sum(1 for x in _ls if x < 2.0)
                                _ptail = (f" | lead med {_ls[len(_ls)//2]:.1f}"
                                          f"/min {_ls[0]:.1f}ms"
                                          f" late<2ms {int(_late2 * 100 / len(_ls))}%"
                                          f" guard {_guard_hits}/{len(_ls)}")
                                _guard_hits = 0
                                _lead_hist = []
                            if _gpu_tot_hist:
                                _gs = sorted(_gpu_tot_hist)
                                _ptail += (f" | gpu {_latencies.get('gpu', 0) * 1000:.1f}ms"
                                           f" gtotal {_gs[len(_gs)//2] * 1000:.1f}ms")
                            # 中断检测：上一拍 fb 正常但本拍骤降 → 精确打点
                            # （用户报告暂停瞬间画面断，但 5s fps 窗口看不到）
                            try:
                                _cm, _fbm, _swz = window.gpu_color_mean()
                                _last_good_fb = globals().get("_last_good_fb", 0.0)
                                if _fbm > 5.0:
                                    _last_good_fb = _perf()
                                    globals()["_last_good_fb"] = _last_good_fb
                                    _last_good_col = _cm[:3].copy()
                                elif _fbm < 2.0 and _perf() - _last_good_fb < 2.0:
                                    print(f"[INTERRUPT] fb骤降 {_last_good_col}→"
                                          f"fb={_fbm:.0f} col={_cm[:3]} "
                                          f"@{_perf()-_last_good_fb:.3f}s后 "
                                          f"t={time.strftime('%H:%M:%S.%f')[:12]}",
                                          flush=True)
                                _ptail += (f" col({int(_cm[0])},{int(_cm[1])},"
                                           f"{int(_cm[2])},a{int(_cm[3])})"
                                           f" fb{_fbm:.0f}")
                                if _swz is not None:
                                    _ptail += f" swz{_swz}"
                            except Exception as _cme:
                                # 诊断探针自身异常不许再静默——col/fb 消失
                                # 正是黑屏会话失去眼睛的原因（发布包日志实证）
                                if _perf() - globals().get("_colerr_t", 0.0) > 10.0:
                                    globals()["_colerr_t"] = _perf()
                                    print(f"[fps] col 探针异常: "
                                          f"{type(_cme).__name__}: {_cme}",
                                          flush=True)
                            print(f"[fps] {current_fps:.1f} | depth {_latencies['depth']*1000:.0f}ms | "
                                  f"resize {_latencies['resize']*1000:.0f}ms | "
                                  f"render {_latencies['render']*1000:.0f}ms | "
                                  f"total {total_latency*1000:.0f}ms | "
                                  f"total(t_pre) {_lat_pre*1000:.0f}ms | "
                                  f"over {_diag['over']} (max {_diag['max']*1000:.0f}ms)"
                                  f"{_ptail}",
                                  flush=True)
                            _diag["over"] = 0
                            _diag["max"] = 0.0

                        fps_values.append(current_fps)

                        if current_time - last_fps_update_time >= fps_update_interval:
                            if fps_values:
                                avg_fps = sum(fps_values) / len(fps_values)
                            if fps_values and len(fps_values) >= 20:
                                sorted_fps = sorted(fps_values)
                                one_percent_index = max(1, int(len(sorted_fps) * 0.1))
                                fps_below_1_percent = sorted_fps[:one_percent_index]
                                low_fps_1_percent_avg = sum(fps_below_1_percent) / len(fps_below_1_percent) if fps_below_1_percent else (sorted_fps[0] if sorted_fps else 0.0)

                            if latency_history:
                                avg_total_latency = latency_sum / len(latency_history)

                            last_fps_update_time = current_time

                        last_latency_display = total_latency

                        if SHOW_FPS:
                            title_text = (
                                f"{current_fps:.1f}FPS | "
                                f"Avg: {avg_fps:.1f} | "
                                f"1% Low Avg: {low_fps_1_percent_avg:.1f} | "
                                f"Latency: {last_latency_display*1000:.0f}ms | "
                                f"Avg Latency: {avg_total_latency*1000:.0f}ms "
                                f"(Capture:{_latencies['capture']*1000:.0f}ms "
                                f"Resize:{_latencies['resize']*1000:.0f}ms "
                                f"Depth:{_latencies['depth']*1000:.0f}ms "
                                f"Render:{render_latency*1000:.0f}ms)"
                            )
                        else:
                            title_text = f"{current_fps:.0f}FPS {last_latency_display*1000:.0f}ms"


                        # Clean title for the local viewer; the fullscreen
                        # window shows no FPS/latency.
                        # 只在标题变化时设置：每帧 SetWindowTextW 会让窗口/标题栏
                        # 以 60Hz 反复重绘，肉眼就是"Stereo Viewer 窗口闪烁"。
                        _wtitle = "Stereo Viewer"
                        if _wtitle != globals().get("_win_title_last"):
                            globals()["_win_title_last"] = _wtitle
                            glfw.set_window_title(_win, _wtitle)

                        # GPU 互操作帧（rgb=None）跳过上传：color_tex 已在
                        # gpu_bind_color 换装；静态帧/常规帧走原路径。
                        # 颜色零拷贝帧（_gc0 非空、rgb=None）也必须进深度块，
                        # 否则 gpu 段为 0、只剩 2D。
                    if rgb is not None or _gc0 is not None:
                        if rgb is not None:
                            window.update_frame(rgb, depth, current_fps, last_latency_display)
                        # SBS/普通窗口模式也走 GPU 深度（此前仅皮肤
                        # 模式下跑，所以切到 Half-SBS 就没有深度了）
                        if _gpu_pipeline:
                            _den = globals().get("_zc_depth_n", 0) + 1
                            globals()["_zc_depth_n"] = _den
                            if _den % 30 == 1 and os.environ.get("D2S_ZC_DEBUG"):
                                print(f"[zc] 深度块进入 #{_den} rgb={rgb is not None} "
                                      f"gc0={_gc0 is not None}", flush=True)
                            _want3d_sbs = mode_bus.read_mode_throttled(
                                _perf(), "rmf_sbs", default=True)
                            # D2S_DEPTH_EVERY_N：每 N 帧推理 1 帧（A/B 定责用，
                            # GPU 预算挤占 DWM 合成的实验开关）。跳过的帧整块
                            # 深度跳过，渲染沿用 _raw_tex 上一张深度（EMA
                            # τ=50ms 天然平滑，画面不跳变）。
                            _dep_n = int(os.environ.get("D2S_DEPTH_EVERY_N", "1") or 1)
                            if _dep_n > 1:
                                _dep_i = globals().get("_dep_every_i", 0) + 1
                                globals()["_dep_every_i"] = _dep_i
                                _want3d_sbs = _want3d_sbs and (_dep_i % _dep_n == 1)
                            if _want3d_sbs:
                                try:
                                    globals()["_dml_frames"] = \
                                        globals().get("_dml_frames", 0) + 1
                                except Exception:
                                    pass
                                try:
                                    _gpu_depth_step(True)
                                except Exception as _ge:
                                    # 不静默：GPU 深度链路失败必须可见（含 DML 回读路径）
                                    if _perf() - globals().get("_gpu_err_t", 0.0) > 5.0:
                                        globals()["_gpu_err_t"] = _perf()
                                        import traceback as _tb3
                                        print(f"[gpu] 深度链路失败: {type(_ge).__name__}: {_ge}",
                                              flush=True)
                                        _tb3.print_exc()
                        if _auto_conv_on():
                            window.apply_live_params({"Convergence": min(1.0, max(0.0, 0.5 * _AUTO_CONV[0] + 0.25 - _auto_conv_bias()))})
                    else:
                        # 颜色零拷贝帧（rgb=None、_gc0 有值）也必须进深度块——
                        # 否则 has_depth 一直是 False，画面绝大多数时间是 2D，
                        # 偶尔蹦一帧 3D = 用户看到的闪烁（实测 1290 bind 对 22 次深度）。
                        if rgb is not None or _gc0 is not None:
                            if rgb is not None:
                                window.update_frame(rgb, depth)
                            # SBS/普通窗口模式：GPU 深度
                            if _gpu_pipeline:
                                _want3d_b = mode_bus.read_mode_throttled(
                                    _perf(), "rmf_b", default=True)
                                if _want3d_b:
                                    try:
                                        _gpu_depth_step(False)
                                    except Exception:
                                        pass
                            if _auto_conv_on():
                                window.apply_live_params({"Convergence": min(1.0, max(0.0, 0.5 * _AUTO_CONV[0] + 0.25 - _auto_conv_bias()))})
                        else:
                            # 覆盖层的 FPS/Delay 必须更新，否则一直显示 0
                            try:
                                window.actual_fps = current_fps
                                # 用端到端延迟（采集→上屏），不是渲染耗时
                                window.total_latency = float(globals().get("_e2e_ms", 0.0))
                            except Exception:
                                pass

                    # 覆盖层（FPS/Delay）：两条分支都要更新，且用统一的端到端延迟
                    try:
                        window.actual_fps = current_fps
                        window.total_latency = float(globals().get("_e2e_ms", 0.0))
                    except Exception:
                        pass

                    # 记录真正写进着色器的收敛面（来自 viewer 的 self.convergence）
                    try:
                        globals()["_APPLIED_CONV"] = float(getattr(window, "convergence", -1.0))
                    except Exception:
                        pass



                    render_latency = _perf() - render_start_time
                    _latencies['render'] = render_latency
                    _latencies['total'] = total_latency

                # Render only when a fresh frame arrived (or every 300ms to
                # keep the surface alive): re-drawing the same texture at full
                # rate was saturating the GPU even in the idle 2D mirror.
                # 节奏由 take_new 的等待保证（无新帧即阻塞，不空转），2D/3D 统一。
                if _got or (_perf() - _last_render_t) > 0.3:
                    _last_render_t = _perf()
                    # GPU 直通保温：2D 闲置期由本线程定期假推理（process
                    # 线程的保温在 GPU_ACTIVE 时跳过，避免并发 run）
                    if _gpu_pipeline and _gpu_runner is not None \
                            and _perf() - _gpu_keepalive_t > 5.0:
                        try:
                            _gpu_runner.keepalive(window.depth_tex)
                        except Exception:
                            pass
                        _gpu_keepalive_t = _perf()
                    # 收敛面每帧写入（原来挂在每秒一次的 FPS 块里 → 1Hz 才更新）
                    if _auto_conv_on():
                        _cv = min(1.0, max(0.0, 0.5 * _AUTO_CONV[0] + 0.25 - _auto_conv_bias()))
                        window.apply_live_params({"Convergence": _cv})
                        globals()["_APPLIED_CONV"] = _cv
                    try:
                        # GPU 直通路径不走 update_frame()，文字覆盖层（FPS/Delay）
                        # 必须在这里刷新，否则永远不更新（内部有缓存，开销可忽略）
                        try:
                            window._update_overlay_texture()
                        except Exception:
                            pass
                        _rt0 = _perf()
                        window._zc_from = "new"
                        _render_frame(window)
                        _rt = (_perf() - _rt0) * 1000
                        _rs = globals().setdefault("_render_stat", [0.0, 0, 0.0])
                        _rs[0] += _rt
                        _rs[1] += 1
                        _rs[2] = max(_rs[2], _rt)
                        if _rs[1] >= 60:
                            print(f"[render] 平均 {_rs[0]/_rs[1]:.2f}ms 最大 {_rs[2]:.1f}ms n={_rs[1]}",
                                  flush=True)
                            _rs[0] = 0.0
                            _rs[1] = 0
                            _rs[2] = 0.0
                    except Exception as _re:
                        # A render exception must not kill this thread: a dead
                        # render loop blocks the depth queue and freezes the
                        # whole pipeline (observed on 2D->3D mode switches).
                        import traceback
                        traceback.print_exc()
                    # Present 对齐：swap 落在下次合成边界前 guard 毫秒内
                    # 则睡过边界（+1.5ms 余量）再换。
                    if _guard_ms > 0:
                        _nb = _compose_clock.next_boundary()
                        if _nb is not None:
                            _ph = _nb - _perf()
                            if 0.0 < _ph < _guard_ms / 1000.0:
                                _guard_hits += 1
                                time.sleep(_ph + 0.0015)
                    try:
                        window._frame_dump()   # GL 内部读帧，避开 WDA 截图限制
                    except Exception:
                        pass
                    # NV_DX_interop 规约：被锁的 D3D 对象不应跨越
                    # SwapBuffers——锁着 swap 会让 present 端闪烁
                    # （背缓冲内容稳定、屏幕上却闪）。先解锁再换页。
                    if _gpu_pipeline and (_gc0 is not None) \
                            and _got and _ridx is not None:
                        window.gpu_unlock(_ridx)
                        _g0u2 = _gc0
                        if _g0u2 is not None:
                            _g0u2.gl_locked[_ridx] = False
                        _locked_slot = None
                    glfw.swap_buffers(_win)
                    # M2：解环纹理锁（采集线程才能再写该槽）。
                    # 颜色零拷贝同样要在这里解：否则深度预处理会在本帧
                    # 内采样这把锁、而解锁被推到下一轮 → 整帧死锁
                    # （实测循环跑 10 轮后卡死）。
                    if _gpu_pipeline and (_gc0 is not None) \
                            and _got and _ridx is not None:
                        window.gpu_unlock(_ridx)
                        _g0 = _gc0
                        if _g0 is not None:
                            _g0.gl_locked[_ridx] = False
                        _locked_slot = None
                    # GPU 直通真全链（t_post→swap）：2048 行的打点在
                    # GPU 推理之前，量不到渲染线程内的推理——swap 后
                    # 单独累计，[fps] 行以 gtotal 报中位数
                    if _gpu_pipeline and _got:
                        _gpu_tot_hist.append(_perf() - t_post)
                        if len(_gpu_tot_hist) > 300:
                            del _gpu_tot_hist[:150]
                    # 相位统计：swap 距下次合成边界的提前量（越小越险）
                    _tsw = _perf()
                    _nb2 = _compose_clock.next_boundary(now=_tsw)
                    if _nb2 is not None:
                        _lead_hist.append((_nb2 - _tsw) * 1000.0)
                        if len(_lead_hist) > 300:
                            del _lead_hist[:150]
            glfw.poll_events()

            glfw.terminate()

        elif mode == "OpenXR":
            from xrviewer import OpenXRViewer, OPENXR_AVAILABLE
            if not OPENXR_AVAILABLE:
                raise ImportError("pyopenxr not installed - run: pip install pyopenxr")
            _item = depth_q.take_new(timeout=5.0)
            rgb, depth = (_item or (None, None))[:2]
            w, h = rgb.shape[1], rgb.shape[0]  # HWC uint8 numpy
            print(f"[Main] Capture source: {CAPTURE_MODE}")
            if CAPTURE_MODE == 'Monitor':
                print(f"[Main] Input monitor: {MONITOR_INDEX}")
            print(f"[Main] Desktop resolution: {w}x{h}")
            print(f"[Main] Environment: {ENVIRONMENT_MODEL}")
            try:
                viewer = OpenXRViewer(
                        depth_ratio=1.0,
                    convergence=CONVERGENCE,
                    frame_size=(w, h),
                    fps=FPS,
                    depth_q=depth_q,
                    show_fps=SHOW_FPS,
                    controller_model=CONTROLLER_MODEL,
                    capture_mode=CAPTURE_MODE,
                    monitor_index=MONITOR_INDEX,
                    environment_name=ENVIRONMENT_MODEL,
                    show_preview_window=XR_PREVIEW_WINDOW,
                    crop_mode=CROP_MODE,
                )
                viewer.run(first_rgb=rgb, first_depth=depth)
            except Exception as e:
                import traceback
                print(f"[Main] OpenXR Link error: {e}")
                print(traceback.format_exc())

                    
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received, shutting down...")
    # except Exception as e:
    #     print(f"[Main] Error: {e}")
    finally:
        # Ensure cleanup happens
        shutdown_event.set()
        cleanup_all_resources()
        
        if SHOW_FPS:
            total_time = time.perf_counter() - start_time
            overall_avg_fps = total_frames / total_time if total_time > 0 else 0
            print(f"Overall Average FPS: {overall_avg_fps:.2f}")
            if fps_values:
                print(f"Recent Average FPS: {avg_fps:.1f}")
                print(f"Recent 1% Low Average FPS: {low_fps_1_percent_avg:.1f}")
        print(f"[Main] Stopped")

    return h


def capture_monitor_mode() -> bool:
    """显示器捕获：环境变量强制，或 SBS/TAB 全屏输出（要任务栏/光标）。"""
    if os.environ.get("D2S_CAPTURE_MONITOR", "") == "1":
        return True
    try:
        return bool(FULLSCREEN_OUTPUT)
    except NameError:
        return False

# --- 自动收敛面（auto convergence）-----------------------------------------
# 用每帧深度中位数慢速跟踪屏幕平面：换片源/换游戏都不用手调 convergence。
# 关闭：D2S_AUTO_CONV=0
_AUTO_CONV = [0.5]
# 实时可改的收敛参数（滑块改动要立刻生效；原来读 utils 常量 = 启动后改不动）
_AUTO_CONV_BIAS = [0.0]
# 收敛面固定下移量（用户定案 2026-11-22：0.10 → 0.20，滑块范围同步 ±0.2）：
# 锚常贴近 1.0，固定先减 0.2 给"深度位置"滑块整段留出上端余量——滑块
# -0.2(远) 时 plane=锚 ≤1.0，不被 clamp 到 1.0 产生死区/跳变。
AUTO_CONV_OFFSET = 0.20
def _auto_conv_on():
    """自动收敛面是否启用：环境变量优先，其次 settings。"""
    if os.environ.get("D2S_AUTO_CONV", "1") == "0":
        return False
    try:
        import utils as _u
        return bool(getattr(_u, "CONV_AUTO", True))
    except Exception:
        return True


def _auto_conv_bias():
    """收敛面偏移：实时滑块优先（_AUTO_CONV_BIAS），否则用启动时的 settings。"""
    try:
        if abs(_AUTO_CONV_BIAS[0]) > 1e-9:
            return float(_AUTO_CONV_BIAS[0])
        import utils as _u
        return float(getattr(_u, "CONV_BIAS", 0.0))
    except Exception:
        return 0.0


def auto_convergence_update(depth_arr, now=None):
    """显示域近侧主体峰锚 -> 收敛面（每帧直取，无 EMA/无死区）。"""
    if not _auto_conv_on():
        return _AUTO_CONV[0]
    try:
        import numpy as _np
        import time as _t
        try:
            import depth as _dmod
            med = float(getattr(_dmod, "LAST_MEDIAN", 0.5))
        except Exception:
            med = float(_np.median(depth_arr)) if depth_arr is not None else 0.5
        if not (0.0 <= med <= 1.0):
            return _AUTO_CONV[0]
        # 收敛锚 = 最终显示深度（EMA 之后）的近侧主体峰（ANCHOR_CS）：
        # 每帧直取、不做任何再平滑/死区（用户定案）。
        _AUTO_CONV[0] = med
        now = _t.perf_counter() if now is None else now
        if now - globals().get('_AUTO_CONV_LOG_T', 0.0) > 30.0:
            globals()['_AUTO_CONV_LOG_T'] = now
            print(f'[auto-conv] base={_AUTO_CONV[0]:.3f} '
                      f'off={AUTO_CONV_OFFSET:.2f} '
                      f'bias={_auto_conv_bias():+.2f} '
                      f'→ applied='
                      f'{min(1.0, max(0.0, _AUTO_CONV[0] - AUTO_CONV_OFFSET - _auto_conv_bias())):.3f} '
                      f'shader={globals().get("_APPLIED_CONV", -1):.3f} '
                      f'(锚={med:.3f})', flush=True)
    except Exception:
        pass
    return _AUTO_CONV[0]


if __name__ == "__main__":
    threading.Thread(target=_watch_stop_request_file,
                     name="StopRequestWatcher", daemon=True).start()
    main(mode=RUN_MODE)
    # main() has returned, so cleanup_all_resources() already ran in its finally block.  
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)



