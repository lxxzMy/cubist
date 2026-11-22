# TensorRT 支持矩阵（Windows）

## 决策

**放弃 10 系（Pascal, SM 6.1）的 TensorRT 支持**，覆盖 16/20/30/40/50 系。

原因：TensorRT 10 已移除 Pascal；保留 10 系意味着额外带一套 TRT 8.6 运行时，
而 50 系（Blackwell）又要求 TRT 10.8+，两套运行时无法合并。

10 系机器不受影响：检测到 Pascal 时跳过 TRT，回落 CUDA EP（`depth.py`
的 `_is_pascal_gpu()`，日志说明原因）。

## 引擎数量

| 方案 | 引擎数 | 说明 |
|---|---|---|
| **运行时构建（默认，推荐）** | **0** | ORT 的 TRT EP 按显卡自动建引擎并缓存（首次约 18s，后台） |
| 预编译（可选） | **2** | Turing 7.5 一个、Ampere+ HC 一个 |

为什么是 2 个而不是 4 个：开了硬件兼容（HC）的 “All Ampere and later” 引擎
可以同时跑 30/40/50 系；Turing 不在该范围内，必须单独一个。

| 引擎 | 构建机 | 覆盖 |
|---|---|---|
| Turing 7.5 | 16 系或 20 系 | 16 系、20 系 |
| Ampere+ HC | 30 系（Ampere） | 30、40、50 系 |

构建时两个引擎必须用**发布包里同一版 TRT**。

## 硬约束

- **不跨平台**：Windows 引擎只能 Windows 用；
- **绑定 TRT 版本**：发 TRT 10.x.y 就得用它构建（可开 version-compatible 放宽同大版本内的小版本）；
- **绑定架构**：无 HC 的引擎换 SM 就加载失败；
- **ORT 缓存命名**：`trt_engine_cache_path` 下按“模型 + 选项”哈希命名，手工预置容易对不上，所以不推荐预编译。

## 代码位置

- `D2S/depth.py`
  - `_is_pascal_gpu()`：显卡名匹配 `GTX 10x0` / `GT 1030`；
  - backend=tensorrt 且 Pascal：**不 raise**，回落 CUDA EP 并打印原因；
  - backend=tensorrt 且运行时缺失（非 Pascal）：保持 fail-loud。
- 引擎缓存目录：`_trt_opts()` 里的 `trt_engine_cache_path`（按模型文件名 + mtime 分目录）。
