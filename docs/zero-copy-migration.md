# 零拷贝迁移计划：GL 纹理 → D3D11 → D3D12 → ORT DML

> 目标：把 DML 推理的输入侧从「GPU→CPU 回读 600KB」改成「GPU 资源直接喂 DML」，
> 省掉 `[dml] read` 的 1.3ms 和 `pack` 的 0.18ms。
> 输出侧仍走 CPU 回读（200KB，0.06ms），收益太小，不做反向零拷贝。

## 0. 结论先行

- **Python 做不到**：`onnxruntime.dll` 不导出 `OrtDmlCreateGPUAllocationFromD3DResource`；
  DML 零拷贝 API 在 `OrtDmlApi` 结构体里，Python 绑不到。而且 Python 的
  `InferenceSession` 不暴露原始 `OrtSession*`，没法把外部 `OrtValue` 绑进去。
- **所以 shim 必须自己持有 ORT 会话**：Python 只通过 ctypes 调
  `run(input_d3d12_resource, out_bytes)`，会话、设备、绑定全在 native 侧。
- 预期收益：深度阶段 5.5ms → **约 4.1ms**（read 1.3→≈0.05、pack 0.18→0）。

## 1. 现状链路

```
桌面帧 (RGB numpy, CPU)
  ↓ upload
color_tex (GL, 工作分辨率)
  ↓ gpu_preprocess()：224 shader（缩放 + ImageNet 归一化 + CHW 平铺）
_prep_tex (GL, R32F 672×224，行内 [R|G|B] 平铺)
  ↓ tex_prep.read()：glReadPixels 600KB，要等 GPU 画完 → 1.3ms
numpy (1,3,224,224)
  ↓ ORT DML（CPU 输入 → GPU）  run 4.0ms
_raw_tex (GL, R32F 224×224) ← tex_raw.write 0.06ms
```

瓶颈是 `glReadPixels`：纯拷贝 0.4ms，其余是等待。

## 2. 目标链路

关键事实：`wglDXRegisterObjectNV` 是「把 **D3D11 纹理** 注册进一个 **GL 纹理名**」，
不是把 GL 纹理转成 D3D11 纹理。现有 `D2S/gpu/dx_interop.py` 就是按这个方向用的。

> **P1 修正（2026-11-15）**：步骤 2-3 之间加一跳 staging——GL 拷进「无 misc 的
> 注册纹理」，D3D11 侧再带 keyed mutex 协议拷进 NTH|KM 共享纹理。原因见 P1 段
> 驱动事实 2/3。多花 ~0.12ms，换确定性。

```
1. shim 创建 D3D11 纹理 T11：DXGI_FORMAT_R32_FLOAT，224×672，
   D3D11_RESOURCE_MISC_SHARED_NTHANDLE | SHARED_KEYEDMUTEX
2. 把 T11 注册进一个 GL 纹理名（复用 dx_interop 的 wglDXRegisterObjectNV）
3. GL 侧把 _prep_tex 用 glCopyImageSubData 拷进 T11（GPU→GPU，约 0.05ms）
4. T11 → IDXGIResource::CreateSharedHandle(NT) → ID3D12Device::OpenSharedHandle
   → ID3D12Resource R12
5. OrtDmlApi::CreateGPUAllocationFromD3DResource(R12, &dml_alloc)
6. OrtApi::CreateTensorWithDataAsOrtValue(DML memory info, dml_alloc,
   bytes=602112, shape=[1,3,224,224], FLOAT) → OrtValue
7. IOBinding::BindInput → Run → 输出 CPU 回读 200KB（沿用现有 tex_raw.write）
```

### 2.1 必须先改的东西：prep 纹理布局

> **P2 修正（2026-11-15）**：最终几何是 **192×784**（平铺 flat CHW），不是下文
> 最初设想的 224×672——192×4=768B 行距恰好满足 D3D12 拷贝的 256 对齐且无
> padding，784×192 恰为 3×224×224（见 P2 段第 3 条）。原分析保留如下。

现在是 **672×224、行内 [R|G|B] 平铺**，读回后要 `reshape(224,3,224).transpose` 才是 CHW。
DML 张量要 **CHW 连续**，所以 shader 输出改成平铺 flat 布局；改完后现有回读
路径也变简单（直接 `reshape(3,224,224)`，省掉 transpose），`pack` 从 0.18ms 再降一点。

## 3. 分阶段计划

### P0 可行性验证 ✅ 完成（2026-11-15）

实测结论（probe 源码 `_toolchain/src/p0_dml_probe.c`，`p0_dml_probe.exe` PASS）：

- **工具链比预想的好**：VS BuildTools 2022 的 MSVC 14.44.35207 本来就在
  （`cl.exe`/`vcvars64` 齐，只是不在 PATH）；缺的只有 Windows SDK，用 nuget
  `Microsoft.Windows.SDK.CPP(.x64)` 10.0.26100.9457 落在 `_toolchain/dl/sdk`
  解决——免 admin、免安装器、可重建（`.gitignore` 已排除）。dxgi1_5.h 在
  `shared/`（新 SDK 合并了 dxgi 布局），`um/DirectML.h` 也在。
- **ORT 头文件**：sparse clone tag `v1.24.4`（`_toolchain/dl/ort-src`），commit
  `2d92497` 与 dll FileVersion `1.24.20260316.2.2d92497` 构建号一致，布局零风险。
  无需 import lib：probe 用 `LoadLibraryExW + GetProcAddress("OrtGetApiBase")`。
- **实测**：`GetExecutionProviderApi("DML", 1)`（注意是 **1**，不是 24）→
  `OrtDmlApi` 8/8 成员非空：`CreateGPUAllocationFromD3DResource`、
  `FreeGPUAllocation`、`GetD3D12ResourceFromAllocation`、`…_DML/DML1/DML2`、
  `GetDMLDevice`、`GetDMLCommandQueue` 全部在位。
- depth-224.onnx DML session（device 0）CPU 输入推理：best **4.15ms** /
  avg 4.54ms（与 Python 路径 ~4.0ms 同档），输出有限值、量纲正常。
- **意外收获**：`SessionOptionsAppendExecutionProvider_DML1` 支持注入自有
  `IDMLDevice + ID3D12CommandQueue`——适配器 LUID 对齐（§4 决策表）从
  "device_id 赌对卡"升级为**显式注入 + `GetDMLDevice` 反查自检**。

环境坑（备查）：git 系统配置 `http.sslCAInfo` 指向已不存在的 `F:\Git\...`，
clone 时需 `GIT_SSL_CAINFO` 覆盖；Git Bash 下直调 cl.exe 要 `MSYS2_ARG_CONV_EXCL='*'`
防 `/nologo` 被转成路径。

### P1 互操作 ✅ 完成（2026-11-15）

判据达成（`_toolchain/src/p1_interop_test.cpp`，三门 0/150528 差异，
含 ±0 / denormal / 最小正规数 / ±FLT_MAX 位模式逐字节存活）：

- **落地架构比 §2 计划多一跳（STAGING）**：GL 源纹理 → `glCopyImageSubData` →
  注册进 GL 的**无 misc 标志 D3D11 纹理**（staging）→ D3D11 侧
  `AcquireSync(0) → CopySubresourceRegion → Flush → ReleaseSync(0)` →
  NTHANDLE|KEYEDMUTEX 共享纹理 → `CreateSharedHandle(NT)` → D3D12
  `OpenSharedHandle` → `CopyTextureRegion` → READBACK。
- **三个驱动事实**（这台机器 / 当前 NV 驱动）：
  1. `NTHANDLE` 单独创建被拒（必须 `|KEYEDMUTEX`，符合 MSDN）；
  2. NTH|KM 纹理**直接注册进 GL 在全新 interop 会话上必败**（fresh session /
     plain 热身 / KM 热身四种姿势都救不了），但同进程后段同参数又能注册成功——
     未定位的驱动怪癖，架构上绕开（staging 无此问题，矩阵验证 2×5 组合可复现）；
  3. **KEYEDMUTEX 纹理不 `AcquireSync` 就写 = 拷贝被静默吞掉**，D3D12 读回全 0。
- 成本实测：staging 拷贝 **121~148µs/帧**（含 keyed mutex acquire/release 同步开销，
  两次运行有抖动，P3 全链 A/B 再定论）；D3D12 读回 RowPitch 896→1024
  （256 对齐），逐行跳 padding 比对。
- R32F 注册/拷贝无格式坑（此前风险表第 1 条解除）。

### P2 ORT DML 零拷贝推理 ✅ 完成（2026-11-15）

判据双双达成（`_toolchain/src/p2_zerocopy_test.cpp` + `p2_ref.py`）：

- **零拷贝 vs CPU 输入（同一 session）：MAE = 0，50176/50176 逐位相等**
- **与 app Python 路径（onnxruntime-directml, device 0）交叉验证：MAE = 0，逐位相等**
- Run(零拷贝) best **3.92ms**（判据 ≤4.0 ✓；CPU 输入 4.15ms），含 KM 拷贝+ fence
  + D3D12 拷贝的全链 best 4.25ms

源码侦察（`_toolchain/dl/ort-src` sparse）改了三处设计（相对 §2）：

1. **`CreateGPUAllocationFromD3DResource` 只吃 buffer**：大小取
   `GetDesc().Width`（配套 `DmlExternalBufferAllocator` 同样按 UAV buffer 建）。
   输入侧因此加一步 D3D12 `CopyTextureRegion(共享纹理→UAV buffer)`，与推理
   同队列 FIFO 排序；
2. **memory info 名 = `"DML"`**（大写，OrtDeviceAllocator，id 0）；
3. **prep 纹理几何 224×672 → 192×784**：`CopyTextureRegion` 进 buffer 强制
   RowPitch 256 对齐，224 宽（896B/行）拷出来必带 padding，DML 要稠密
   602112B；192×4=768B 恰好对齐，784×192=150528 = 3×224×224 一个不差，
   单次拷贝天然稠密。**§2.1 的布局结论据此更新**。

其它落地细节：session 用 `SessionOptionsAppendExecutionProvider_DML1` 注入
自有 `IDMLDevice`（`DMLCreateDevice` 加载 app capi 的 DirectML.dll）+ 自有
DIRECT 队列；跨 API 同步 = D3D11 共享 fence（NT handle）→ D3D12 队列 Wait。
测试坑：±FLT_MAX 输入会把深度网络打爆（激活溢出，CPU 参考 NaN）——测试图案
特殊值改 ±10（±0/denormal 保留）。

### P3 集成 ✅ 完成（2026-11-15，判据部分达成——见"认知修正"）

落地件（全部就位，`D2S_ZEROCOPY=1` 启用，失败自动永久回退）：

- `runtime/d2s_dml_shim.dll`（源码 `_toolchain/src/d2s_dml_shim.cpp`，/MT 静态 CRT，
  C ABI create/run/times6/destroy，P2 链路全封装 + LUID 选卡 + 6 段计时）；
- `D2S/gpu/dml_shim.py`：ctypes 封装（DLL 搜索 runtime→仓库根→_toolchain/src；
  DirectML.dll 随包优先、缺则用 system32）；
- `D2S/gpu/dml_gl.py`：DmlDepthRunner 零拷贝分支——零输入 sanity + **首帧实况
  对账**（同帧 shim 输出 vs python 回读+会话，MAE<1e-5 才续用，防同步竞态）+
  任何失败打印一次并永久回退；flat 几何回读路径（reshape(3,R,R) 免 transpose）；
- `D2S/viewer.py`：gpu_depth_init flat 几何（192×784）+ shader `flat_w` 变体 +
  通道自检/viewport 适配；**main.py 零改动**。

**应用内实测（GTX 1650，3D Half-SBS 60fps）**：

| 指标 | 基线（回读） | 零拷贝 | 判据 | 判定 |
|---|---|---|---|---|
| 输入链（read+pack） | 1.2~1.7ms | **~1.0ms** | ≤0.15ms | ✗（见下） |
| run | 3.5~5.6ms | 3.3~4.1ms | — | ✓ |
| 阶段均值 | 6.3~8.8ms | **5.5~6.7ms** | ≤4.3ms | ✗（但优于基线 ~1ms） |
| 数值一致 | — | sanity/实况 MAE=0 | MAE<1e-5 | ✓ |
| 帧率 | — | 57~60fps | 60fps | ✓ |

6 段分解（ms）：lock 0.31 / glCopy 0.05 / unlock 0.16 / km 0.15 / d3d12 0.34 / run 3.6。

**认知修正（本机实测定案）**：原 read 的 1.3ms 大头是"等 GPU 画完"而不是拷贝
（glReadPixels 1.07~1.92 vs 零拷贝全链 ~1.0）——零拷贝消灭了拷贝（0.6→0.05），
消灭不了等待：剩余 ~1ms 是 interop lock/unlock（0.5）+ D3D12 fence/submit（0.34）
的**每帧跨 API 固定同步成本**。read≤0.15ms 的判据在本机驱动上不可达；
深度阶段赢 ~1ms/帧是真实但有限的收益。

关键调优记录：`glFinish`→`glFlush`（interop unlock 自带 GPU 侧依赖插入，省 0.55ms，
正确性由首帧实况对账兜底——零输入 sanity 检不出同步竞态，静态内容无竞态）。

后续可选（未做）：双缓冲 pipelining（帧 N+1 的 lock/copy 与帧 N 推理重叠，
+1 帧深度延迟，理论再省 ~0.7ms）；或退回 PBO 异步回读（计划 §7）。

### P4 打包（1~2 天）

- 新 DLL 及其依赖（VC runtime / onnxruntime.dll / DirectML.dll）随包；
- CUDA 构建（`release/Cubist-1.0-224CUDA`）不受影响（shim 只在 DML 路径启用）；
- 干净机器验证（无 VS、无 ORT 头文件的环境）。

## 4. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 谁持有 ORT 会话 | **shim** | Python 拿不到 `OrtSession*`，别无选择 |
| 输入资源 | 共享 D3D11 纹理 T11 | `wglDXRegisterObjectNV` 只能这个方向 |
| 输出 | CPU 回读（200KB） | 反向零拷贝只省 0.06ms，不值 |
| prep 布局 | 改成 CHW 连续 | DML 张量要求；顺带简化回读路径 |
| 适配器 | shim 的 D3D11/D3D12/GL/ORT 全对齐同一 LUID | 跨卡必失败（已有 `gpu/adapter.pick`） |

## 5. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 驱动不接受 R32F 共享纹理 | P1 卡住 | 先做 P0 小验证；备选 RGBA32F（浪费 4 倍带宽） |
| ORT DML 对「外部资源 + IOBinding」支持有坑 | P2 卡住 | P0 里先用 CPU 内存模拟 OrtValue 走通，再换 D3D12 资源 |
| keyed mutex 同步死锁 | 画面卡死 | 先 `Flush` + 单线程串行，确保同一线程提交 |
| 工具链/ORT 头文件拿不到 | 计划无法启动 | P0 第一件事；拿不到就退回 PBO 方案 |
| 多 GPU / 混合显卡 | 运行时失败 | 启动自检 LUID，失败回退 |

## 6. 工期与人力

- **1.5~2 周**（1 名熟悉 D3D11/D3D12 + GL interop 的 C++ 工程师），不含客户现场验证。
- 分阶段可随时停：P0 失败就退回「PBO 异步回读」（省 1ms，代价 1 帧深度滞后）。

## 7. 备选（不写 C++ 的话）

| 方案 | 收益 | 代价 |
|---|---|---|
| PBO 异步回读 | ~1.0ms | 深度多滞后 1 帧（16.7ms） |
| R16F 预处理纹理 | −0.12ms（read） | +0.23ms（pack 转换），**净亏，已否** |
| CPU 预处理 | 无回读 | 7.3ms，**慢 5 倍，已否** |

## 8. 验收清单

- [x] 输出与现有路径 MAE < 1e-5（P2 双验证 + P3 应用内 sanity/首帧实况全 0）
- [ ] `[dml] read` ≤ 0.15ms，深度阶段 ≤ 4.3ms —— **不可达（P3 定案）**：read 地板
      ~1.0ms（跨 API 每帧同步成本），实际赢 ~1ms/帧；见 P3 段认知修正
- [x] 60fps 稳定（57~60fps，`D2S_ZEROCOPY=1` 实测）
- [ ] 2 小时长跑：显存不增长、无 TDR、无句柄泄漏（11-15 累计 ~5 分钟零拷贝运行
      零异常/零回退/MAE=0；连续长跑被「Cubist Tray Watchdog」计划任务换掉外来
      实例阻断——正式 2h soak 建议走看门狗托管部署开 `D2S_ZEROCOPY=1` 跑）
- [x] 失败回退路径可用（`D2S_ZEROCOPY=0` 或 shim 任何失败 → 行为与旧版一致）
- [x] CUDA 构建不受影响（flat 布局仅 D2S_ZEROCOPY=1 时生效，CUDA 路径原样）

## 9. P4 打包（零拷贝 shim）—— 落地步骤

### 构建（换机器也能跑）

`_toolchain/src/build_shim.sh` 现在会自动探测三件事，缺什么就报什么：

| 依赖 | 探测方式 | 兜底 |
|---|---|---|
| MSVC x64 工具链 | `vswhere` 找 VS 安装 → `VC/Tools/MSVC/*` 最新版 | 环境变量 `MSVC_DIR` |
| Windows SDK | `_toolchain/dl/sdk/...` 优先，否则系统 `Windows Kits/10` | 环境变量 `SDK_DIR` |
| ORT 头 | `_toolchain/dl/ort-src/include` | 环境变量 `ORT_INC` |

```bash
bash _toolchain/src/build_shim.sh     # 产物 runtime/d2s_dml_shim.dll（/MT，无外部依赖）
python tools/package_shim.py          # 装进 release/*/runtime/ 并重编相关 pyc
```

`package_shim.py` 做三件事：拷 DLL、重编 `gpu/dml_shim.pyc` / `gpu/dml_gl.pyc` /
`viewer.pyc`、逐项校验。**发布包里的 app 正在运行时 DLL 会被占用**，脚本会明确
提示先退出（或先结束 pythonw.exe），不会半途拷一半。

### 验证（发布包内）

```bash
D2S_ZEROCOPY=1  ./立体派.exe      # 日志应出现：
#   [zc] 零拷贝链路激活（sanity MAE=0，luid=0x...）
#   [zc] 实况对账通过（MAE=0）
#   [dml] read ~1.0 pack 0.00 run ~4.0 write ~0.06
D2S_ZEROCOPY=0  ./立体派.exe      # 不应出现任何 [zc] 行（回读路径）
```

### 已知测量口径

- `write` 必须**直接计时**：之前用「总时间 − read − run」当残差，会把周期对账
  帧（每 60s 一次，含 600KB 回读 + 一次参考推理）算进去，看起来像 2ms。
  现在 `write` 实测 0.06~0.09ms，对账耗时单独从诊断项里扣掉。

## 9. 收益评估（2026-11-15 深夜，1080p 同机同场景 A/B，各 30s）

| 指标 | 双零拷贝 | 双关（CPU 采集+回读） | 差值 |
|---|---|---|---|
| fps | 60.0~60.1 | 58.9~60.0 | ≈0（1080p 都不卡瓶颈） |
| t_pre（取帧→上屏） | 7~9ms | 7~9ms | **无可测差异** |
| 深度 run（推理调用） | 3.69ms | 4.89ms | **−1.2ms**（H2D 拷贝消除） |
| 深度输入链 read+pack | 0.98+0.00 | 1.05+0.11 | −0.2ms |
| 颜色采集 CPU 耗时 | bind ~1.2ms（GPU 侧） | grab 4.1ms（p95 6.6，纯 CPU） | **−3ms CPU/帧** |
| gtotal（帧预算占用） | 12.8~13.1 | 12.7~12.9 | ≈0 |

**结论**：1080p 下两条零拷贝对端到端延迟无可测收益（处理 7~9ms 远小于
16.7ms 预算，被流水线节奏吸收）；真实收益是——
1. 深度：推理调用内 −1.2ms（P2 单测 3.92 vs 4.15 同向）；
2. 颜色：每帧 −3ms 纯 CPU（不在延迟上体现，在占用/线程争用上体现）；
3. **4K 解锁**：CPU 采集 1080p 抓帧 4.1ms/8.3MB → 4K 线性外推 ~16ms/33MB
   （p95 26ms）必然掉帧；GPU 路径 4K 仍在预算内——当初的设计动机。
端到端组成（真实玻璃到玻璃）：桌面变化→DDA 交付 0~16.7 + 处理 7~9 +
DWM 合成 0~16.7 ≈ **典型 23~25ms / 最坏 ~40ms**（受两个合成周期支配，
处理侧已非主导）。样本为单轮 30s，±0.5ms 噪声。

今天最大的延迟改善其实不是零拷贝，是 f5388b8 模式文件节流（释放渲染线程
~55% CPU，动画场景 43-53→60fps）——零拷贝的价值在 CPU 减负、4K 余量与
架构纵深，不在 1080p 延迟数字。
