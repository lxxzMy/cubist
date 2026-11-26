# 深度模型说明

本目录只有一个模型文件：`zipdepth_wide_512x288.onnx`。它是 Cubist 使用的**唯一深度模型**——
单目相对深度估计：每帧把采集画面缩放到 512×288 送进去，在 GPU 上经 DirectML 推理，
得到的深度图再经重标定、平滑和时域滤波后用于 3D 合成。程序固定加载它，设置面板不提供模型选择。

## 文件

| 项 | 值 |
|---|---|
| 文件名 | `zipdepth_wide_512x288.onnx`（24,625,797 字节） |
| SHA-256 | `197d598f50ee3270b12831096dbab260c6bd5979bab13b87f01a28fad0268274` |
| 输入 | 名为 `image`，`float32 [1, 3, 288, 512]`（NCHW），RGB，取值 0..1；ImageNet 均值/方差已在模型图内，不要再做归一化 |
| 输出 | 名为 `depth`，`float32 [1, 1, 288, 512]`，相对逆深度：值越大越近，没有固定的尺度与零点 |
| 规模 | 约 610 万参数，纯卷积，无注意力算子 |
| 许可 | MIT，全文见同目录 `LICENSE-ZipDepth-MIT.txt` |

## 它来自哪里

- **ZipDepth**（基础网络与权重）：Fabio Tosi、Luca Bartolomei、Matteo Poggi、Stefano Mattoccia，
  University of Bologna，ECCV 2026，"ZipDepth: Bringing Lightweight Zero-Shot Monocular Depth
  Anywhere, on Any Device"。<https://github.com/fabiotosi92/ZipDepth>，MIT，
  Copyright (c) 2026 Fabio Tosi。
- **Widescreen EdgePad**（宽屏微调、图优化与 ONNX 导出）：Nightfall 团队。在 ZipDepth 的标准
  权重 `zipdepth_base` 上只训练了解码器（610 万参数中的 154,005 个，分两个阶段共 6000 步），
  让同一套权重适配包括 16:9 在内的多种输入尺寸、逐帧更稳；编码器仍是 ZipDepth 的。
  训练标签来自原版 EdgePad-384 自己的输出、Video Depth Anything Small 和 Marigold V2
  （后两者为 Apache-2.0，只用于生成标签，不随模型分发）。
  发布页：<https://github.com/tB0nE/nightfall/releases/tag/edgepad-models-2026-10-08>，MIT，
  Copyright (c) 2026 The Nightfall authors。

## 本项目做了什么、没做什么

- **没有修改模型文件**：这里的文件与上述发布页的同名资产逐字节一致（SHA-256 相同）。
- 只做推理。模型的原始输出每帧按 P2/P98 两个分位重新标定到 [0.02, 0.98]，再做各向异性高斯
  平滑与 80 ms 的时域 EMA，然后用于 3D 合成。这些处理都由 GPU 完成，不改动模型本身。
- 实测（GTX 1650，DirectML）：单帧推理约 7 ms；1080p 60 fps 内容下，从桌面合成完成到
  我们提交显示约 11 ms（中位数）。

## 许可

- 模型按 **MIT License** 分发，全文见 `LICENSE-ZipDepth-MIT.txt`。这份文本与发布页随附的
  `LICENSE` 逐字节一致，保留了其中的两行版权声明（ZipDepth / Nightfall 各一行）。
- MIT 允许使用、修改与分发（包括商业使用），条件是随附版权声明与许可全文。
  **再分发这个模型时，请保留本文件和 `LICENSE-ZipDepth-MIT.txt`。**
- 如实说明：ZipDepth 官方 README 写明，它的权重是用 Depth Anything V2 **Large** 生成的伪深度
  蒸馏训练的，而该教师模型的权重是 CC BY-NC 4.0。ZipDepth 作者以 MIT 许可在同一仓库发布
  权重，没有对权重附加任何限制。本项目照此分发，不对此另作主张。

## 给开发者

`D2S/model_catalog.py` 的 `LOCAL_MODELS` 里还登记了几个对比用的档位（DepthART 224/448、
ZipDepth 384、ZipDepth 宽屏 672×384），它们**不随仓库 / 发布包**。需要对比时，把对应的
ONNX 放进 `models/` 或 `runtime/`，再用环境变量 `D2S_DEPTH_MODEL=<键>` 临时切换；
不设置时固定使用本模型。

---

# Depth model

`zipdepth_wide_512x288.onnx` is the only depth model Cubist uses (monocular relative depth,
512×288 input, DirectML inference on the GPU). The application always loads this file.

- **Input** `image`: float32 `[1, 3, 288, 512]` (NCHW), RGB in 0..1 — ImageNet mean/std are
  applied inside the graph. **Output** `depth`: float32 `[1, 1, 288, 512]`, relative inverse
  depth (larger = nearer).
- **Origin**: [ZipDepth](https://github.com/fabiotosi92/ZipDepth) (University of Bologna,
  ECCV 2026; MIT, Copyright (c) 2026 Fabio Tosi) — **Widescreen EdgePad** fine-tune and ONNX
  export by the Nightfall authors
  ([release `edgepad-models-2026-10-08`](https://github.com/tB0nE/nightfall/releases/tag/edgepad-models-2026-10-08);
  MIT, Copyright (c) 2026 The Nightfall authors).
- **Unmodified**: byte-identical to the upstream asset,
  SHA-256 `197d598f50ee3270b12831096dbab260c6bd5979bab13b87f01a28fad0268274`.
- **License**: MIT. The full text, including both copyright lines, is in
  `LICENSE-ZipDepth-MIT.txt`; keep it together with the model when redistributing.
