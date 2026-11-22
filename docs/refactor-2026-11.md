# 重构报告（2026-11-16，pre-refactor → refactor-done）

## 动机与目标

产品面定案收窄（WGC 皮肤退役、Windows-only、RTMP/MJPEG 退出、XR 保留）后，
代码库背负全部历史：应用 2.4 万行挤在 5 个巨石文件、git 跟踪 1.9 万文件中
1.85 万为误提交 venv/转储、无 lint 无测试、多处已核实隐患（HRESULT 常量
一错一正、GPU 推理块×3、ctypes 结构体×4、模式协议×8、grab 元数不一致）。

目标：**行为完全保持**（60fps/6ms/启动时序），结构可维护化。

## 结果总览

| 指标 | 前 | 后 |
|---|---|---|
| D2S 应用代码 | 24,211 行 / 5 巨石 | ~10,300 行 / 20 内聚模块 |
| git 跟踪文件 | ~19,000 | ~880 |
| 仓库新增可复验门禁 | 无 | refactor_gate（30 项黄金基线）+ smoke.py + ruff |
| lint | 无 | ruff E9/F821/F811/E722 全绿 |

最大文件：viewer 5,391→4,023（shaders/overlay 抽出）；main 4,177→~2,300；
capture 1,930→~500；utils 1,367→~490（config/platform_probe/model_catalog 三分）。

## 阶段与提交

| 阶段 | 提交 | 内容 |
|---|---|---|
| P0 基线 | 36ab2e0 + tag pre-refactor | tools/refactor_gate.py：depth（生产色彩+参考深度链）/render（live_offline 无头 GL）/smoke（启动日志断言）三探针；30 项自洽全 PASS（mae=0.0） |
| P1 卫生 | d228d0b | 1.88 万误提交文件出索引（trainenv/_dml/调试转储）；.gitignore 补洞；根目录游离 settings.yaml 归档 docs/legacy |
| P2a WGC | 2d709dc | 皮肤子系统彻底摘除（capture/main/viewer/utils/gpu/wgc_winrt+dx_iids 删除）；boot 1.7s→0.8s |
| P2b 平台 | 73ad9cb + 2ee475e | Windows-only：metal_viewer 删、Darwin/Linux 分支删、utils 平台探测收窄 |
| P2c 推流 | 2903fa3 | RTMP/MJPEG 全链删（torch 遗迹清零）；gui.py→tools/dev_gui.py；A/B 工具→tools/ab/ |
| P2d 零散 | 40941b2 | torch 编译器块/后端开关×9/死白名单/keepalive 等 |
| P3 消重 | b90f98c | HRESULT 表入 gpu/rtcom.py（dxgi_dup 错值修正）；winstructs.py 结构体单源；GPU 推理块收编 _gpu_depth_step(rich)；mode_bus.py 协议单源（150ms 节流与 per-key 窗口语义保真）；grab 统一 3 元组；_stdio.py |
| P4 拆分 | 184b37f + b667575 | shaders.py（渲染探针逐位验证）；utils 三分+shim；overlay.py |
| P5 调试面 | dee45bc | PARALLAXRT_*→D2S_ONNX/D2S_INFER_EVERY；bat 陈旧模型覆盖清除（dev 启动即崩暗坑）；D2S_NO_FULLSCREEN 恢复；docs/env-flags.md（44 开关盘点）；版本号单源约定 |
| P6 工程化 | 2494f45 + tag refactor-done | ruff 全绿（抓出 3 个重构引入潜在崩溃+2 个原有暗雷）；smoke.py；package_shim 补 8 新模块并刷新 pyc 发布包 |

## 有意推迟项（与理由）

**门禁覆盖边界**是本轮的核心取舍依据：dev 环境（CUDA 版 ORT，无 DML EP）
冒烟只覆盖启动链——渲染循环内部、清理路径的改写缺乏逐位验证手段。据此：

1. **渲染循环模块化拆分（原 P4 主项）**：main() 渲染循环经 75+ 处 globals()
   与 cell-list 通信，拆模块需穿透全部状态键，而门禁验证不到循环内部。
   风险/收益不成立。已做替代：GPU 推理块收编（有 SBS 分支行为可对照）+
   _zcc_want/_gpu_pipeline 定义序修复。
2. **StereoWindow.render() 拆 per-mode 方法**：同上（渲染探针只验证 shader
   源码等价性，不执行 StereoWindow.render）。
3. **shutdown 显式句柄注册表**：清理路径仅在优雅退出时执行，冒烟用
   taskkill 强杀不经过。已删 cleanup 中的死 cap 嗅探分支。
4. **一次性调试探针批量删除**：各 1-2 行、env 门控、零关闭成本；清单见
   docs/env-flags.md"归档"节，待门禁覆盖增强后按单清理。

**解除条件**：把冒烟升级为"3D 会话级"验证（需要 DML EP 可用的解释器或
发布包内跑）后，上述各项可安全重启。

## 遗留与待办

- **安装器重建**：本机无 Inno Setup 6（上包 2026-11-08 建于别处）。
  d2s.iss 全量拷 D2S\*，新模块自动流入；装好 ISCC 后跑
  `tools\installer\build-installer.cmd` 即可。
- **手动验收清单**（docs/release-vs-dev-audit.md §5）：冷启 10 分钟无黑屏
  60fps、Pause 2D↔3D ×10、双实例踩踏 6s 恢复——需真实 3D 屏环境，由用户
  按需执行。
- **dev 解释器无 DML EP**：windows_capture 包的 DDA 在本机初始化失败
  （0x887A0004，pre-refactor 已有）；CPU 采集诊断路径不可用与此相关。
- **配置默认值四处分叉**（gui 选项表/settings_dialog 种子/yaml.default/代码
  fallback）：本轮未动（原 C 档内容），gui 已移 tools/ 后剩余两处。
- settings.yaml(.default) 中已失效的 streaming 组键保留未删（写入端兼容）。

## 门禁使用

```bat
:: 日常改动快速回归（仓库根）
D2S\python3\python.exe -I tools\smoke.py            :: compile+ruff+gate 全量
D2S\python3\python.exe -I tools\smoke.py --skip-smoke
D2S\python3\python.exe -I tools\refactor_gate.py capture   :: 行为大改后重采基线
```

黄金基线 `_refactor_baseline/`（gitignored，本机可重建）锚定于
pre-refactor；对渲染/深度**语义**的刻意变更（非重构）请重采基线而非对照旧基线。
