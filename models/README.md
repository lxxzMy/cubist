# 深度模型（depth-224.onnx）

本目录的 `depth-224.onnx` 是 **DepthART** 项目 `MobileNetV4-M-slim-SPF`（224 输入）权重的
ONNX 导出，用于 Cubist 的单目深度推理。

## 许可：CC BY-NC 4.0（非商用）

- 本模型**不是**本项目的 Apache-2.0 代码，按 **Creative Commons Attribution-NonCommercial 4.0
  International（CC BY-NC 4.0）** 分发：
  <https://creativecommons.org/licenses/by-nc/4.0/>
  完整法律文本见同目录 `LICENSE-CC-BY-NC-4.0.txt`。
- **禁止商业使用**（NonCommercial）：不得将本模型用于主要指向商业利益或金钱报酬的用途，
  也不得随商业产品分发。Cubist 本项目为免费、非商业项目，因此可以随本仓库分发该模型。
- 本仓库的**源代码**仍是 Apache-2.0（见根目录 `LICENSE`）；该许可不适用于本模型文件。

## 署名（Attribution）

- DepthART: "DepthART: Depth Anything Rethought for Tiny Models"
  - 代码/项目主页：<https://github.com/xuefeng-cvr/DepthART>
  - 权重发布：<https://huggingface.co/Fengxue93/DepthART>
  - 原始文件：`onnx/relative_mnv4m_slim_spf_224_default.onnx`
  - Copyright (c) 2026 DepthART Authors；DepthART 自有材料为 CC BY 4.0。
- 教师模型：**Depth Anything V2-Large**（`depth-anything/Depth-Anything-V2-Large`），
  其权重为 CC BY-NC 4.0：<https://creativecommons.org/licenses/by-nc/4.0/>。
  DepthART 官方说明其 checkpoints 使用 Depth Anything V2-L 生成的监督训练，
  且不包含在 DepthART 的 CC BY 4.0 授权范围内。

## 修改说明（Indicate changes）

相对 DepthART 发布的原始权重/导出，本项目：
1. 直接使用官方 ONNX 导出 `relative_mnv4m_slim_spf_224_default.onnx`（未改权重）；
2. 以固定 224×224、RGB、ImageNet 归一化输入运行（与官方一致）；
3. 在本仓库中以文件名 `depth-224.onnx` 分发。

若你二次分发本模型，请保留本文件与 `LICENSE-CC-BY-NC-4.0.txt`，并保留上述署名与修改说明。
