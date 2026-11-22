# D2S_* 环境开关清单（2026-11-16 重构 P5 盘点）

约定：全部以 `D2S_` 为前缀（原 `PARALLAXRT_D2S_*` 已于本轮统一改名：
`D2S_ONNX` / `D2S_INFER_EVERY`）。产品路径（不设任何变量）不受任何
开关影响；下列均为诊断/联调用途。

## 常用（保留）

| 变量 | 作用 | 生效点 |
|---|---|---|
| `D2S_NO_FULLSCREEN=1` | 强制窗口化联调（产品定案"恒全屏"之后评估，只影响显式设置的会话） | utils.py |
| `D2S_ZEROCOPY=1` | 启用 DML 零拷贝链路（发布包验收用，激活时日志出现 `[zc] MAE=0`） | gpu/dml_shim.py |
| `D2S_DUMP_FRAME=1` | 转储 SBS 输出帧（verify_sbs_parallax.py 的输入） | viewer.py |
| `D2S_FRAME_DUMP=<dir>` / `D2S_FRAME_DUMP_HZ=<n>` | 周期转储渲染帧 | viewer.py |
| `D2S_CURSOR_OFF=1` | 隔离自绘光标（判断多出来的光标是否我们画的） | utils.py |
| `D2S_WDA=0` | 关闭 WDA 采集排除（让系统截图能拍到输出） | main.py |
| `D2S_DEPTH_EVERY_N=<n>` | 每 N 帧推理 1 帧（GPU 预算挤占 DWM 的定责实验） | main.py |
| `D2S_ZC_DEBUG=1` | 零拷贝链路分帧统计 | main.py / gpu/* |
| `D2S_FPS_LOG=1` | 采集/处理分段耗时 60 帧中位 | main.py |
| `D2S_ZC_PROBE=1` | 零拷贝帧绑定三件套探针 | main.py / viewer.py |
| `D2S_ONNX=<path>` | 覆盖深度模型路径 | depth.py |
| `D2S_INFER_EVERY=<n>` | infer-every 帧缓存（ParallaxRT 历史兼容 shim） | depth.py |
| `D2S_CAPTURE_GPU=0` | 强制 CPU 采集路径（诊断；产品语义 GPU-only 无回落） | main.py |
| `D2S_AUTO_CONV=0` | 关闭自动收敛面 | main.py |
| `D2S_BACKEND=numpy/torch` | d2s_ops 后端选择（A/B 用） | d2s_ops.py |
| `D2S_CAPTURE_MONITOR=1` | 采集靶改显示器 HMONITOR | main.py |

## 一次性/历史探针（保留但视为归档；均已 env 门控，关闭时零成本）

`D2S_IC_SEARCH`/`D2S_IC_DEBUG`（交错相位标定）、`D2S_DH_MARK`（深度标志打点）、
`D2S_TEST_RENDER_STALL_S`（渲染卡死自检钩子）、`D2S_ZC_PROBE2`（draw 时刻三件套）、
`D2S_UNIFORM_DEBUG`、`D2S_SBS_DEBUG`、`D2S_RENDER_DUMP`、`D2S_PREP_F32`、
`D2S_GRAB_LOG`、`D2S_FAKE_CURSOR_NULL`、`D2S_DEBUG_WINFO`、`D2S_DEBUG_FB`、
`D2S_ZC_MIRROR`、`D2S_ZC_FINISH`、`D2S_DEPTH_TRACE`、`D2S_DEPTH_STATS`、
`D2S_DEPTH_FIXED/_LO/_HI`、`D2S_AUTO_RANGE/_TAU`、`D2S_CURSOR_DEBUG/_ALPHA/_RGB`、
`D2S_FORCE_WIN`、`D2S_BENCH`、`D2S_FREE_RUN`、`D2S_NO_REPRESENT_DRAW`、
`D2S_DEPTH_DUMP`、`D2S_GPU_TIMING`。

> 这些探针每处仅 1-2 行、默认关闭时无任何运行时开销。本轮重构**未批量删除**
> （dev 门禁只覆盖启动链，渲染内部路径的删改验证不足——同 P4e 的取舍），
> 后续在增强门禁覆盖后可按此清单清理。

## 已消失（随功能面移除）

- `D2S_DUMP_PAIR` —— 随 WGC 皮肤模式 GPU 块删除（P2a）
- `PARALLAXRT_D2S_ONNX` / `PARALLAXRT_D2S_INFER_EVERY` —— 改名统一（P5）
- ORT profile JSON 转储 —— 无第一方代码启用（2026-11-15 的临时调试遗留，
  根目录 4 个 `onnxruntime_profile__*.json` 为其产物，已 gitignore）
