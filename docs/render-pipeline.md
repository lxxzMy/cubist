# 离线渲染管线（我的模拟）vs 程序实时管线

> 用途：把"我渲染"和"程序渲染"的每一步都写清楚，任何差异都能在这里对号入座。
> 参数取当前设置：强度 20px、收敛面 0.35、抗锯齿 2（σ=4 texel）、Depth Grow 0、Depth Close 0、Slope 0。

## A. 我的离线渲染（`eyeL_true.png` / `eyeR_true.png` 就是这条链）

| # | 步骤 | 具体实现 | 关键参数 |
|---|---|---|---|
| 1 | 读图 | `cv2.imread` | BGR, 1920×1080, uint8 |
| 2 | 缩到模型尺寸 | `cv2.resize(..., (448,448), INTER_AREA)` | 面积平均，抗摩尔纹 |
| 3 | 色彩空间/归一化 | BGR→RGB，`/255`，ImageNet 标准化 | mean `[.485,.456,.406]`，std `[.229,.224,.225]` |
| 4 | 张量布局 | HWC→CHW，加 batch | (1,3,448,448) float32 |
| 5 | 深度推理 | ONNX `DepthART-spf-448.onnx`（CPU EP 或 TRT） | 输出 raw 448×448，范围约 0.9~4.7 |
| 6 | 取分位 | `raw[::4,::4]`（112×112）→ P2 / P98 | 步长 4 的子采样 |
| 7 | 归一化 | `d = clip((raw − P2)/(P98 − P2), 0, 1)` | 硬编码 2%/98% |
| 8 | **AA（真高斯）** | `cv2.GaussianBlur(d, ksize=0, sigma=4.0)` | **σ = 4 texel @448**（≈17px@1080p）；OpenCV 自动核 ≈33×33；边界 reflect |
| 9 | 上采样到屏幕 | `cv2.resize(d, (1920,1080), INTER_LINEAR)` | 双线性 |
| 10 | 算位移 | 每眼 `shift = ∓2·P·(d − c)`（像素） | P=20，c=0.35；近处 d>c 时位移反向 |
| 11 | 采样 | `cv2.remap(img, x+shift, y, INTER_LINEAR, BORDER_REFLECT)` | 水平位移，垂直不变 |
| 12 | 输出 | PNG 1920×1080（每眼一张） | 无 EMA、无 edge snap |

### 我**没有**做的（所以比程序"干净"）

- 没有 EMA（单帧处理）
- 没有 edge snap（轮廓吸附）
- 没有 content mask（内容区域遮罩）
- 没有 alpha 边界淡出
- 没有 disocclusion fill
- 没有行交错（输出的是单眼完整图）

## B. 程序实时管线（GPU 直通）

| # | 步骤 | 实现 | 备注 |
|---|---|---|---|
| 1 | 采集 | WGC 零拷贝，目标窗口客户区 | 1920×1040 |
| 2 | GPU 预处理 | `gpu_preprocess()`：彩色 → 672×224 float 纹理（每行 `[R块|G块|B块]`） | GL shader |
| 3 | CUDA 映射 | GL 纹理 → CUDA array（`cudaGraphicsMapResources`） | ~0.5ms |
| 4 | 拷贝入 | 3 次 `cudaMemcpy2DFromArray`（每通道一个平面）→ CHW 缓冲 | ~0.1ms |
| 5 | TRT 推理 | DepthART-spf-448 | ~5.3ms |
| 6 | 拷贝出 | `cudaMemcpy2DToArray` → `_raw_tex` | ~0.05ms |
| 7 | 回读 | 读回 448 raw（用于算 P2/P98 + 中位） | ~0.65ms |
| 8 | 后处理 pass 图 | `normalize → [close(max→min)] → [slope] → gauss-h → gauss-v → EMA` | 见下 |
| 9 | 渲染 | 着色器：位移、edge snap、按行取眼、采样彩色、alpha 边界 | 交错输出 |
| 10 | 呈现 | `glfw.swap_buffers` + Present 对齐 | 60fps |

### 后处理 pass 图细节（`postprocess_gpu`）

| pass | 作用 | 参数 | 成本 |
|---|---|---|---|
| `normalize` | 归一化 + 可选 Grow（max 滤波） | lo/hi（P2/P98）、grow(0~6) | ~0.05ms |
| `close-max` / `close-min` | 闭运算填孤立凹陷 | close(0~4) | ~0.1ms |
| `slope{i}` | 坡度限幅（k·\|d'\|>阈值的像素才模糊） | target、6 次迭代 | ~0.2ms |
| **`gauss-h`** | 水平 17 tap 高斯 | σ = AA_STRENGTH（texel） | ~0.05ms |
| **`gauss-v`** | 垂直 17 tap 高斯 | 同上 | ~0.05ms |
| `ema` | 时域平滑 | α = 1 − 0.37^(1/N) | ~0.05ms |

> 2026-11-15 之前这里是**9 点近似**（±σ、±2σ，权重 0.6065/0.1353），
> 实测等效 σ 只有真高斯约 1/3，是"碎缝"的根因；现改为两遍分离式真高斯。

## C. 两条链的差异（对照表）

| 项 | 我的渲染 | 程序 |
|---|---|---|
| AA 实现 | 真高斯 σ=4（≈33 tap） | 两遍 17 tap 真高斯（≡） |
| EMA | 无 | N=10 → α≈0.095（当前设为 off） |
| edge snap | 无 | 开（阈值 0.12/增益） |
| 源图分辨率 | **448（模型输入上采样）** | **1920×1040 原始纹理** ← 主要视觉差 |
| 输出 | 单眼全幅 | 行交错 |
| content mask / alpha | 无 | 有 |

## D. 已知结论（用数据验证过）

| 检验 | 结果 |
|---|---|
| 程序 `dtex` vs 9 点近似 | MAE **0.0004**（就是它） |
| 程序 `dtex` vs 真高斯 σ=4 | MAE 0.0045，**最大差 0.357**，0.57% 像素 >0.1 |
| 程序 fb vs 用程序深度模拟 | MAE 1.06，>30 的像素 0.37% |
| 深度是否慢一帧 | 否（本帧深度 4.74 vs 上帧 14.01） |

## E. 参数速查

```
面板 抗锯齿 2  →  AA_STRENGTH=4  →  真高斯 σ = 4 texel（448 网格）
面板 抗锯齿 4  →  σ = 8 texel
面板 抗锯齿 6  →  σ = 12 texel
位移公式：shift_px = 2 · 强度 · (深度 − 收敛面)
跨层条件：shift 变化 > 1 texel 宽度 → 需要 σ ≳ 2·强度·Δd / 2.6 才能摊开
```


## F. 调试通道与禁忌（用过一次就记住）

| 需求 | 正确做法 | 禁忌 |
|---|---|---|
| 看程序真实输出 | `D2S_RENDER_DUMP=<dir>`（GL 内 `glReadPixels`） | ❌ `D2S_WDA=0` |
| 跑程序着色器看效果 | `tools/live_offline.py`（无头 GL） | ❌ 为了截图关 WDA |
| 确认加载了哪一版代码 | 启动日志 `[build] mtime/md5/pid` | ❌ 用进程启动时间判断（execv 不改 PID/时间） |

**为什么不能关 WDA**：全屏输出模式下程序改走 DesktopDuplication 抓屏幕，而屏幕上正是它自己的 3D 窗口 → 抓到自己 → 无限递归（画中画）。

**为什么不能用进程启动时间判断版本**：自重启走 `os.execv`，PID 与进程创建时间都不变。
