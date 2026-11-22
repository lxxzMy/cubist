# 颜色路径零拷贝：DXGI 采集 → D3D11 → GL（P1 验证记录）

目标：全屏模式下省掉「GPU→CPU 8MB → process → 上传 6MB」，预计 2~4ms。

## 已确认可行的部分

1. **vtable 槽位**（`D2S/gpu/dxgi_dup.py` 原来写错，已定位）：
   - `IDXGIOutputDuplication` 继承 `IDXGIObject`，所以
     - `AcquireNextFrame` = **槽 8**（原来写 4 → 访问违例，被 `except OSError` 静默吞掉）
     - `ReleaseFrame` = **槽 14**（原来写 7）
   - `GetDesc`=7、`GetFramePointerShape`=11、`MapDesktopSurface`=12、`UnMapDesktopSurface`=13。

2. **取帧规则**：
   - 会话刚建时的第一帧 `AccumulatedFrames == 0`（空帧，拷贝出来全黑），必须**跳过**；
   - **必须在 `ReleaseFrame` 之前完成 `CopyResource`**，否则拷贝拿到空白。
   - 正确顺序：Acquire → 判断 acc>0 → QI 到 `ID3D11Texture2D` → `CopyResource` 到自己的纹理 → ReleaseFrame。

3. **D3D11 拷贝验证通过**：`CopyResource` 到 staging 纹理后 `Map` 读回，
   均值与 GDI 截图一致（127.5 vs 127.5）。

4. **注册进 GL 成功**：`wglDXRegisterObjectNV` 对「普通 D3D11 纹理（bind=SHADER_RESOURCE，
   misc=0）」可以注册（和 WGC 环纹理同配置），lock/unlock 正常。

5. **shader 采样可用**：把互操作纹理绑到 sampler，渲染到普通 FBO 再读回，
   能拿到画面（说明 viewer 里 `color_tex` 的采样路径成立）。

## 还需要的适配

1. **通道顺序/朝向要校准**：采样得到的通道与 GDI 对照存在偏差（实测 R 通道不一致），
   接入时要用已知图案校准 swizzle 与是否需要上下翻转。
2. **互操作纹理不能 CPU 回读**：
   - `glGetTexImage` 返回全黑；
   - `glctx.framebuffer([interop_tex])` 报 `framebuffer is not complete (UNSUPPORTED)`。
   - 所以依赖 CPU 读回 color_tex 的路径要改：
     - `[fps]` 的 `gpu_color_mean()` 颜色探针；
     - `D2S_FRAME_DUMP` 的 color 导出；
     - 推流/录制（如果有）。
   - 这些路径在零拷贝模式下应改为「跳过」或「先 blit 到普通纹理再读」。

## 接入方案（A）

- 在 `capture.py` 的 DXGI 分支增加「GPU 纹理模式」（开关：`D2S_CAPTURE_GPU=1`）；
- 采集线程：Acquire → 跳过空帧 → CopyResource 到环纹理（3 张，参照 WGC 的 RING）→ ReleaseFrame；
- 渲染线程：`window.gpu_bind_color(dx, gen, ring_idx, ptrs, w, h)` 换成环纹理（复用 WGC 的逻辑）；
- 光标：GPU 模式下 CPU 帧不存在，光标要靠 viewer 的自绘覆盖层（SBS 路径本来就有）；
- 回退：任何一步失败（注册失败、LUID 不一致、拿不到帧）自动切回 CPU 路径。

## 验证 spike

临时脚本（不入库）：`%TEMP%\d2s_zero_spike*.py`、`d2s_spike_loop2.py`、`d2s_spike_sample.py`。
关键调用序列见上文第 2 条。
