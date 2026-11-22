# 发布包 vs dev 树全量对比审计（2026-11-15）

> 起因：发布包（release/Cubist-1.0.224）持续黑屏，用户要求与测试程序（D2S/ dev 树）
> 全量对比。结论先行：**代码面完全一致（已证伪"pyc 陈旧"）；黑屏是运行面的
> 既有缺陷——DDA 会话被抢后永不恢复**，与当天的镜传/节流修复无关。

## 1. 代码面：23 个 pyc 逐一取证，零差异

方法：pyc 头部（timestamp-based, flags=0）的 offset8/offset12 记录源码 mtime/size，
与 dev 树当前源码逐一比对。

| 发布包 app/ | 数量 | 结论 |
|---|---|---|
| 顶层 pyc（main/viewer/capture/…） | 12 | 全部与源一致（含 22:20 刷新的 6 个） |
| gpu/ 包 pyc | 11 | 全部与源一致 |
| runtime/d2s_dml_shim.dll | 1 | 与仓库根 md5 一致（f818f389…） |

"其它模块陈旧、与刷新的 6 个 API 漂移"假设**证伪**。打包链路（第一代 Inno Setup
排除法 → 第三代 pyc 形态 + package_shim.py 直编目标）无源码改写。

## 2. 环境面：差异清单与影响判定

| 项 | dev | 发布包 | 判定 |
|---|---|---|---|
| onnxruntime | 跑 DML（`PYTHONPATH=_dml` overlay） | onnxruntime-directml 同二进制 | 一致 |
| dev python3 自带 | onnxruntime-gpu(CUDA12.8+TRT) | — | 未用于 DML 路径，无关 |
| 深度模型 | runtime/depth-224.onnx 原件 | models/ 匿名化副本（-11,110B，f459f4f） | 等价 |
| settings.yaml | Parallax 20、Interleaved 实验态 | Parallax 15、Half-SBS、缺 Draw Cursor 键 | 无害差异 |
| 启动器 | 命令行 pythonw | .NET exe：`-m main`、CWD=app/、PATH 前置 runtime/ | 路径解析全成立 |

## 3. 运行面：黑屏第一手证据（发布包 app/logs/app.log 四会话）

- `[zcc] 卡在 idle` 共 **786 行**，`GetDeviceRemovedReason=0x0 (S_OK)`——设备没挂，
  `AcquireNextFrame` 永久超时 = DDA 会话被抢/失效后**无恢复机制**
  （`_reopen_dup` 只认 ACCESS_LOST；被抢表现为无限 WAIT_TIMEOUT）
- 时间线实锤双实例互踩：dev `D2S/logs/d2s.pid`（22:25:46）与发布包会话
  （22:23 / 22:31）**重叠**——DXGI 桌面复制每输出全局唯一，代码注释早有记载
  （main.py 采集看门狗处："第二个 duplication 会话会把本会话挤掉→冻结黑屏"）
- 黑屏在旧版（viewer.pyc md5 7fea53b6）与新版（396cf337）同样复现——
  与当天修复无关；Pause 2D→3D 翻转后同样进入永久 idle

## 4. 修复（本次提交）

`gpu/dxgi_gpu_grab.py` 三处：

1. **会话被抢检测**（`_watchdog`）：poll 活跃但 >6s 无任何成功 acquire
   （WAIT_TIMEOUT 不算活）→ 置 `_need_reopen` 标记；`poll()` 在本线程消费并
   `_reopen_dup()`（跨线程不碰 D3D）。静态桌面误报代价 = 一次 ~10ms 例行重建
   （新会话首帧即当前桌面），日志限频 1 次/分钟
2. **DuplicateOutput 被拒提示**（`_open_dup`）：记录最后被拒 hr 并提示
   "另一实例持有桌面复制？"——方向性线索替代静默黑屏
3. 既有 `_reopen_dup` 冷却 50ms + 12 次指数退避（610ae02）不动，天然承担
   "抢回"重试

## 5. 复测方法

1. 单实例发布包冷启（先杀光 dev/看门狗实例）+ 真实内容 10 分钟：不黑、~60fps、
   无"卡在 idle"
2. 双实例踩踏：dev + 发布包同跑 2 分钟 → 观察被抢方 ≤6s 内"标记重建→已重建"
   恢复出帧
3. Pause 2D↔3D ×10：翻转后 ≤6s 恢复

## 6. 遗留账本（2026-11-15 全黑终案后，用户确认记入）

修复性质判定：0c72dc8 治本（消灭"降级静默改变目标语义"错误类——降级只降
质量不改含义，失败从不可观测变可观测）；但对以下三项是治标，待还：

1. **E_INVALIDARG 诱因未查明**：DuplicateOutput1 在全屏切换过渡态报
   0x80070057，现靠同输出重试消化（实测窗口 <3s），未直接测量/定位。
   若过渡窗口变长 → 退 CPU 采集（质量降但正确），不会黑屏。安全网在。
2. **监视器映射仍是枚举序号**（Monitor Index → 输出 #N）：换接线/换主屏/
   枚举序变化可能再指错屏——现在会安全地错（可见失败→CPU 回落）而非
   静默黑屏。根治方向：按窗口所在屏选输出（MonitorFromWindow）。
3. **防御机制膨胀**：镜传/节流/迟到让位/看门狗去噪/输出钉死等七层散在
   build_inline/_open_dup/_reopen_dup/看门狗四处。终态应把"采集会话
   生命周期"（选屏/建会话/重试/回落策略）收敛成一个模块、一处策略。

还账时机：下次动采集模块时一并处理；触发条件 2 为"用户改接显示器布局"。
