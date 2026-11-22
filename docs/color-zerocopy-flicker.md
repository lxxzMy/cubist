# 颜色零拷贝闪烁（持续全屏黑闪）诊断与修复

> 2026-11-15。现象：颜色零拷贝（`D2S_CAPTURE_GPU` 默认开）下画面持续全屏黑闪。
> 结论先行：**重呈现路径在锁外采样环纹理**——`wglDXUnlockObjectsNV` 之后、
> 无 `wglDXLockObjectsNV` 保护下用互操作纹理 draw，NV 规约未定义，本机驱动
> 返回全黑。修复：**镜传（draw-copy mirror）**，`gpu_bind_color` 锁内把环纹理
> 绘制式拷进普通 GL 镜像纹理后立即解锁，所有后续采样走镜像。

## 证据链

| 环节 | 验证 | 结果 |
|---|---|---|
| 黑帧仍在？ | `D2S_ZC_PROBE=1` 90s | [blk] 20/20（max 上限），持续黑帧确认 |
| present 端？ | [blk] 内 fbo_mean=0.0 | FBO 复绘也黑 → 排除 present，问题在 draw 之前 |
| 纹理内容？ | [zc2] `direct`（glGetTexImage 直读） | **恒亮**（40~43），所有帧、所有槽 |
| draw 能画出来？ | [zc2] `drawread`（当前绑定渲 32×32 FBO） | 分裂：新帧亮 / 部分帧黑 |
| 黑帧出身？ | `window._zc_from` 戳（new/represent/wash×3） | **100% 相关**：`new` 18 帧全亮、`represent` 6 帧全黑（两轮复现：18/6 与 19/6） |

机制：无新帧时的重呈现（main.py 62Hz 节流路径）重画上一帧，此时环槽已在
**上一拍 swap 前**解锁（unlock#1，NV 规约要求锁不得跨 SwapBuffers）；锁外
采样互操作纹理 → 驱动返回黑。静止桌面新帧稀疏 → 黑亮交替 = 持续闪。
方向与上一轮假设相反：**新帧亮、重呈现黑**（上轮以为是"新帧黑、重呈现亮"）。

## 修复（viewer.gpu_bind_color，默认开，`D2S_ZC_MIRROR=0` 对照）

```
dx.lock(ring) → ring.use(0) → 全屏 quad 渲进 [mirror] FBO → screen.use()
→ dx.unlock(ring)（锁只存在 ~0.1ms，不跨 swap）→ color_tex = mirror
```

- 绘制式拷贝而非 `glCopyImageSubData`：后者对 interop 挂名纹理 GLError
  （驱动按占位元数据校验尺寸，实测每帧必炸）；全屏 quad 走真实 D3D 尺寸，
  与既有 `_ctex_fbo` 同款（仓库已验证的做法）；
- RGBA 字节原样透传，shader 侧 `.bgra` 通道交换不受影响；
- 所有 draw（新帧/重呈现/黑洗重显）与 224 预处理全部采样普通镜像——
  锁外采样这一**类**问题整体消失，锁不再跨 SwapBuffers（160e078 的约束
  自动满足），`avoid`/锁占用语义不变；
- 代价：每新帧一次 w×h 全屏绘制拷贝（[fps] gpu 6.1ms 内含，无可观测增量）。

## 修复验证（90s，行交错全屏，静止桌面）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| represent drawread | 0.0（100% 黑） | 34.3（100% 亮） |
| new drawread | 36.8 | 34.3 |
| [blk] 黑帧 | 20（上限） | **1**（启动首帧，bind 前，无害） |
| 镜传异常 | — | 0 |

## 本次诊断新增的探针（均 env 门控，可复用）

- `D2S_ZC_PROBE2=1` → `[zc2]`（viewer.py `_zc_draw_probe`）：draw 时刻三件套
  ——unit0 真实绑定 / glGetTexImage 直读（带 GL 错误检查区分"黑"与"读不回"）/
  当前绑定渲 32×32 FBO 均值。四个插桩点：passthrough / main（Anaglyph/Interleaved/
  Interleaved-V）/ sbs（fill_16_9）/ sbs-nofill。限 24 帧。
- `window._zc_from` 出身戳（main.py 五个 `_render_frame` 调用点）：
  new / represent / wash-a/b/c——任何"帧出身 vs 帧状态"的相关性分析可用。
- 教训：插桩点必须先核对 render() 的**完整分支骨架**再落（本次 Half-SBS 实际
  走 `fill_16_9=False` 的非填充分支，前三个桩全空转了一轮）。

## 遗留

- 30 分钟长跑验收待做（黑帧率目标 0/min）；用户目视确认闪烁消失。
- `D2S/settings.yaml` 的 Display Mode 测试期间从 Half-SBS 切到了 Interleaved
  （根目录 settings.yaml 原本就是 Interleaved）——确认后按需还原。
- main.py 的出身戳与用户的颜色零拷贝 WIP 同文件未提交，随用户 WIP 一并处理。
