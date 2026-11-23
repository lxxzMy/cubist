# Cubist（立体派）

把普通 2D 桌面/视频/游戏实时转换成 3D 画面，输出到 3D 显示器、偏光屏和 AR 眼镜。
Windows 10/11 · Direct3D 12 · DirectML，纯 GPU 零拷贝管线（采集纹理 → 推理 → 深度链 → 上屏全程不出显存）。

> **免费的纯非商业开源项目**：代码 Apache-2.0；随仓库分发的深度模型为 **CC BY-NC 4.0（禁止商用）**，
> 详见 `models/README.md`。仓库含源码 + 模型，不含 Python 运行时和预编译 DLL（见「运行前提」）。

## 能力

- **输入**：整屏 DXGI Desktop Duplication（DDA），零拷贝共享纹理；可同时指定「采集屏」与「输出屏」（AB 屏分离，适合 AR 眼镜）。
- **深度**：单目深度模型（ONNX）经 DirectML 推理，输入 224×224，输出 GPU 端直接进深度后处理链（归一化 / 保边 AA / 时域 EMA / 切镜检测）。
- **输出格式**：

  | 格式 | 说明 |
  |---|---|
  | `Interleaved` | 行交错（偏振/快门 3D 屏主力档） |
  | `Half-SBS` | 左右半宽（每眼占输出画布一半宽） |
  | `Full-SBS` | 左右全宽（每眼一份完整画面并排） |
  | `Half-TAB` | 上下半高 |
  | `Full-TAB` | 上下全高 |
  | `Depth Map` | 深度图调试输出：直接显示**模型原始深度**（224²，线性上采样到当前分辨率，按本帧 P2/P98 归一化 + spectral 上色） |

- **呈现**：flip 模型交换链（3 缓冲）+ 分层覆盖层；`PIPELINE` 成对流水线把每帧 3 个 GPU 同步点压掉，颜色与深度严格同帧配对。
- **切换**：托盘菜单 / 设置面板随时切 2D↔3D；2D 时窗口隐藏（原生桌面）。

## 架构（D3D12 主路径）

```
DDA（IDXGIOutputDuplication）
  └─ 环纹理 ×3（NTHANDLE|KEYEDMUTEX 共享）
       └─ prep pass（BGRA → 224² CHW，ImageNet 归一化，192×784 flat 布局）
            └─ DirectML（onnxruntime-directml，同队列/同设备，零拷贝输入）
                 └─ 深度后处理链（dilate → slope → 各向异性高斯 → EMA → 归一化）
                      └─ 显示 pass（DIBR/Interleaved，逐眼视口 + scissor）
                           └─ flip swapchain present
```

关键文件：

| 路径 | 作用 |
|---|---|
| `D2S/main.py` | 主循环：2D/3D 状态机、成对流水线、遥测 |
| `D2S/gpu/d3drender.py` + `_toolchain/src/d3drender.cpp` | D3D12 渲染器 C 助手（PSO/屏障/交换链/显示 pass） |
| `D2S/gpu/d3d_prep.py` / `d3d_post.py` / `d3d_stats.py` | prep、深度链、GPU 判据 |
| `D2S/gpu/dxgi_gpu_grab.py` + `dda_bridge.py` | DDA 采集与跨 API 环纹理桥 |
| `D2S/gpu/dml_shim.py` + `_toolchain/src/d2s_dml_shim.cpp` | DirectML 推理 shim（同队列、可选 GPU 输出） |
| `D2S/gpu/placement.py` | 显示器 ↔ 显卡映射（采集屏/输出屏同卡校验） |
| `D2S/gpu/d3d_view.py` | 视口、着色器常量、overlay/光标 |
| `D2S/gpu/hlsl_*.py` | HLSL 着色器源码（运行时经 DXC 编译） |

## 运行前提

| 项 | 说明 |
|---|---|
| 系统 | Windows 10 2004+ / Windows 11（依赖 D3D12 + DDA + DirectML） |
| 显卡 | 支持 DirectX 12 的 GPU（DirectML 路径实测 NVIDIA GTX 1650 / RTX 2060） |
| Python | 3.12（`D2S/python3` 为发布包内的可选便携运行时，不在仓库中） |
| 模型 | `models/depth-224.onnx`（**已随仓库**，CC BY-NC 4.0，非商用） |
| ONNX Runtime | `_toolchain/dl/ort-dml/`（onnxruntime.dll + DirectML.dll），或用 pip 安装 `onnxruntime-directml` |
| 原生 DLL | `runtime/d3drender.dll`、`runtime/d2s_dml_shim.dll`（可由 `_toolchain/src` 构建） |

Python 依赖（D3D 主路径）：`numpy opencv-python pillow PyYAML requests pywin32 mss`；
GL 回退/工具额外需要 `moderngl glfw PyOpenGL PyOpenGL-accelerate`。见 `D2S/requirements.txt`。

## 从源码运行

1. 安装 Python 依赖：
   ```bat
   pip install numpy opencv-python pillow PyYAML requests pywin32 mss
   ```
2. 准备 ONNX Runtime（DirectML）：把 `onnxruntime-directml` 的 `capi` 目录放到
   `_toolchain/dl/ort-dml/`（需含 `onnxruntime.dll` 与 `DirectML.dll`），或装进 Python 环境。
3. 模型已随仓库：`models/depth-224.onnx`（程序按 `models/` → `runtime/` 顺序查找；也可用
   `D2S_ONNX=<路径>` 指定）。注意它是 **CC BY-NC 4.0**，仅限非商业使用。
4. 构建原生库：
   ```bat
   cd _toolchain\src
   build_d3drender.cmd        :: 需要 VS Build Tools + Windows SDK（见脚本内路径）
   ```
   `d2s_dml_shim.dll` 的构建见 `_toolchain/src/build_shim.sh`。
   产物拷到 `runtime/`。
5. 启动：
   ```bat
   set D2S_RENDER=d3d
   D2S\python3\python.exe -u D2S\main.py
   ```
   或直接双击发布包里的 `立体派.exe`（源码树里没有）。

## 本仓工具

- 门禁/冒烟：`tools/d3d_parity_test.py`、`d3d_dibr_parity.py`、`d3d_interleaved_parity.py`、`d3d_view_smoke.py`、`d3drender_smoke.py`、`dda_bridge_smoke.py`、`shim2_sanity.py`、`smoke.py`
- 实时：`tools/d3d_live.py`（真实屏幕流 + `--gate` 门禁）
- 端到端：`tools/e2e_d3d.py`、`e2e_d3d_live.py`、`e2e_playback.py`
- 打包：`tools/make_release_d3d.py` + `tools/package_shim.py`（重编 pyc、同步 DLL、重编启动器）
- 调参/分析：`tools/filter_mae_bench.py`、`preblur_bakeoff.py`、`pan_algorithm_bakeoff.py` 等

## 已知限制

- 仅 Windows；D3D12 + DirectML 是唯一主力路径，GL 旧管线保留在历史分支、不在本仓。
- TensorRT/CUDA 运行时不随仓库分发，需要自行安装。
- 深度模型（DepthART MobileNetV4-M-slim-SPF）为 **CC BY-NC 4.0**：可随本免费项目分发，但**不得商用**；需要商用请自行替换为许可允许的模型（如 Depth-Anything-V2-Small Apache-2.0 / Distill-Any-Depth-Small MIT）。
- 「采集屏 / 输出屏」必须位于**同一块显卡**（零拷贝前提）；跨卡会在启动时弹窗提示并回退为同屏采集。
- AR 眼镜的「3840×2160 渲染、眼镜纵向压回 3840×1080」用法需要显示端支持 2160p 输入；此时输出格式应选 `Half-SBS`。
- `Depth Map` 是调试输出：显示的是**模型原始深度**（224×224，线性上采样，未经归一化/AA/EMA 链处理），spectral 四段色表；暗色文字界面上会呈现大块色块，属正常现象。
- 环境变量开关见 `docs/env-flags.md`；排障日志在 `D2S/logs/`（不入库）。

## 许可证与致谢

- 本项目：**Apache License 2.0**（见 `LICENSE`），定位为**免费的纯非商业开源项目**。
- `D2S/` 源自 **Desktop2Stereo**（MIT，Copyright (c) 2026 LC700X），原始许可保留在 `D2S/LICENSE`。
- 第三方组件清单与许可声明见 `NOTICE`。
- 深度模型 `models/depth-224.onnx`：**CC BY-NC 4.0（非商用）**，来自 **DepthART**
  （MobileNetV4-M-slim-SPF，权重由 Depth Anything V2-L 监督训练）。
  署名、修改说明与完整条款见 `models/README.md` 和 `models/LICENSE-CC-BY-NC-4.0.txt`。
  代码的 Apache-2.0 与模型的 CC BY-NC 4.0 相互独立：**重新分发本仓库（含模型）视为非商业分发**，
  不得用于商业目的。

---

### English

Cubist converts ordinary 2D Windows desktops, videos and games into real-time stereoscopic 3D
for 3D monitors, polarized displays and AR glasses. Windows 10/11, Direct3D 12 + DirectML,
fully GPU-resident zero-copy pipeline (DDA capture → shared ring textures → prep → DML inference →
depth post-chain → per-eye display pass → flip present).

This is a **free, non-commercial** open-source project. The source code is licensed under
Apache-2.0; the bundled depth model (`models/depth-224.onnx`, DepthART MobileNetV4-M-slim-SPF,
trained with Depth Anything V2-L supervision) is licensed **CC BY-NC 4.0 — non-commercial use only**
(see `models/README.md`). The Python runtime and prebuilt DLLs are not included.
The `D2S` application base is derived from Desktop2Stereo (MIT, LC700X).
