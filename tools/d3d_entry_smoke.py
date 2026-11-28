# -*- coding: utf-8 -*-
"""3D 入口门禁：2D→3D 切换时不得显示本段开始之前的画面。

背景（2026-11-27 实测）：进 3D 的头两帧显示的是上一段 3D 的最后一帧
（e2e 29.8 分钟）。成因有三：DDA 会话在 2D 空档里失效，第一拍 poll 返回
ACCESS_LOST；主循环"切换瞬间沿用上一帧"的兜底分不清"没有新帧"和"会话
坏了"，把上一段的环槽当新帧送进管线；上一段留在流水线里的推理结果也被
照常消费；覆盖层在第一帧上屏前就已 show（露出交换链里的残留画面）。

本门禁覆盖其中可离线验证的两件：
  A) 采集端：poll() 如实报告"这一轮为什么没给帧"，且"可以沿用上一帧"只在
     会话活着、确实没有新帧时成立（假 COM 对象，纯 Python）。
  B) 覆盖层：窗口隐藏时首帧也能用 nowait 提交成功并计数——覆盖层"等本段
     首帧上屏后再显示"依赖这一点；被跳过的 present 不计数；keep_topmost()
     之后窗口确实带 TOPMOST（真实 D3D12，隐藏窗口，不 show）。
    C) 深度复用的显示帧平滑步（D3DDepthPost.ema_step）离屏数值验证。
  D) 推理提交时机（gpu.dml_shim.infer_pick）：采到帧的同一拍不交，拷贝落地之后
     才交；忙时不收不交；永远算最新的一帧。
主循环本身的入口时序只能在实机切换 2D/3D 时看日志验证。
用法：D2S\\python3\\python.exe -I tools\\d3d_entry_smoke.py
"""
import ctypes
import os
import sys

os.environ["D2S_RENDER"] = "d3d"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402

from gpu import dxgi_gpu_grab as _zg  # noqa: E402

W, H = 640, 360


class _FakeDup:
    """只需要 release()（真 COM 纹理走不到的路径才用它）。"""

    def release(self):
        pass


def _fake_acquire(zcc, hr, acc=0, res=0):
    def _acq(_timeout_ms):
        info = _zg.DXGI_OUTDUPL_FRAME_INFO()
        info.AccumulatedFrames = acc
        info.LastPresentTime = 12345 if acc else 0
        return hr, info, ctypes.c_void_p(res)
    zcc._acquire = _acq


def part_a(fails):
    def _mk():
        return _zg.DxgiGpuColor(None, 0)

    z = _mk()
    if z.poll_state != "init":
        fails.append(f"A0 初始 poll_state={z.poll_state}")

    # 会话不存在
    z = _mk()
    z._dup = None
    z._open_dup = lambda: False
    if z.poll() != -1 or z.poll_state != "nodup" or z.frame_unchanged():
        fails.append(f"A1 nodup: state={z.poll_state}")

    # 上一帧拷贝未落地
    z = _mk()
    z._dup = _FakeDup()
    z._frame_held = True
    z._release_held = lambda: False
    if z.poll() != -1 or z.poll_state != "held" or z.frame_unchanged():
        fails.append(f"A2 held: state={z.poll_state}")

    # 会话活着、没有新帧 —— 只有这两种可以沿用上一帧
    for name, hr, acc, res in (
            ("wait", _zg.DXGI_ERROR_WAIT_TIMEOUT, 0, 0),
            ("acc0", 0, 0, 0)):
        z = _mk()
        z._dup = _FakeDup()
        _fake_acquire(z, hr, acc, res)
        if z.poll() != -1 or z.poll_state != name:
            fails.append(f"A3 {name}: state={z.poll_state}")
        elif not z.frame_unchanged():
            fails.append(f"A3 {name}: frame_unchanged 应为 True")

    # 会话坏了 —— 一律不沿用
    for name, hr in (
            ("error", _zg.DXGI_ERROR_ACCESS_LOST),
            ("na", _zg.DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)):
        z = _mk()
        z._dup = _FakeDup()
        _fake_acquire(z, hr)
        z._reopen_dup = lambda tries=12, base_delay=0.02: False
        if z.poll() != -1 or z.poll_state != name:
            fails.append(f"A4 {name}: state={z.poll_state}")
        elif z.frame_unchanged():
            fails.append(f"A4 {name}: frame_unchanged 应为 False")

    # ReleaseFrame 失败 = 会话已坏
    z = _mk()
    z.poll_state = "wait"
    z._dup = type("D", (), {"call": lambda *a: -1})()
    z._reopen_dup = lambda tries=12, base_delay=0.02: False
    if z._release_frame() or z.poll_state != "error" or z.frame_unchanged():
        fails.append(f"A5 ReleaseFrame 失败: state={z.poll_state}")

    # 上一轮"可沿用"不能被下一轮"会话坏了"继承
    z = _mk()
    z._dup = _FakeDup()
    _fake_acquire(z, _zg.DXGI_ERROR_WAIT_TIMEOUT)
    z.poll()
    _fake_acquire(z, _zg.DXGI_ERROR_ACCESS_LOST)
    z._reopen_dup = lambda tries=12, base_delay=0.02: False
    z.poll()
    if z.frame_unchanged():
        fails.append("A6 wait 之后 ACCESS_LOST，frame_unchanged 仍为 True")
    print("[A] poll_state / frame_unchanged OK", flush=True)

    # 环槽选择：在飞的那一帧（avoid）和上一对（avoid2）都不会被新帧覆盖；
    # 3 槽里永远给采集留一个槽（GPU 被拖慢时同一槽会被反复写最新帧）
    z = _mk()
    z.gl_locked = [False, False, False]
    z.avoid, z.avoid2, z._ring_i = 1, 0, 0
    if [z._free_slot() for _ in range(3)] != [2, 2, 2]:
        fails.append("A7 avoid=1 avoid2=0 时唯一可写槽应是 2")
    z.avoid = z.avoid2 = 1
    got = [z._free_slot() for _ in range(3)]
    if sorted(set(got)) != [0, 2] or 1 in got:
        fails.append(f"A7 avoid=avoid2=1 时不应写槽 1: {got}")
    z.avoid, z.avoid2, z._ring_i = 0, 1, 0
    z.gl_locked = [False, False, True]
    if z._free_slot() != -1:
        fails.append("A7 三槽都不可写时应返回 -1")
    print("[A] 环槽选择（avoid/avoid2）OK", flush=True)


def part_b(fails):
    from gpu.d3d_window import D3DStereoWindow
    from gpu import d3drender as dr

    win = D3DStereoWindow(display_mode="Half-SBS", fill_16_9=False,
                          frame_size=(W, H), same_screen=True,
                          cursor_overlay=False, convergence=0.5,
                          parallax_px=12.0, monitor_index=1)
    try:
        if win.is_visible():
            fails.append("B0 同屏覆盖层创建后不应可见")
        if win.present_seq != 0:
            fails.append(f"B0 初始 present_seq={win.present_seq}")

        a = np.zeros((H, W, 4), np.uint8)
        a[:, : W // 2] = (60, 60, 220, 255)
        a[:, W // 2:] = (220, 60, 60, 255)
        up = win.ren.upload_tex(W, H, dr.FMT_BGRA8, renderable=False)
        win.ren.upload(up, a)
        win.ren.begin_frame(); win.ren.copy_upload(up); win.ren.submit()
        win.ren.wait()
        rw, rh = win._res_w, win._res_h
        win.post.upload_raw(np.full((rh, rw), 0.5, np.float32))
        win.post.spatial_chain(0.0, 1.0, aa_texels=0.0, temporal="ema",
                               alpha=1.0)
        win._frame_has_depth = True
        win.attach_color(up, W, H)

        # 先问槽、再提交（main._flip_slot_free 的用法）：waitable 是一次性信号，
        # 问过之后提交那一步不能再问第二次（checked=True）。隐藏窗口上首帧
        # 也要提交成功。
        pre = bool(win.ren.wait_flip(0))
        win.render()
        if not pre or not win.present(nowait=True, checked=True) \
                or win.present_seq != 1:
            fails.append(f"B1 先问槽再 checked 提交应成功 "
                         f"（pre={pre} seq={win.present_seq}）")
        # 紧接着的重呈现节拍同样要能提交
        win.render()
        if not win.present(nowait=True) or win.present_seq != 2:
            fails.append(f"B2 隐藏时第二次 nowait present 未成功 "
                         f"（seq={win.present_seq}）")
        # 被跳过的 present（翻转队列没槽）不计数
        real = win.disp.present
        win.disp.present = lambda _sync, nowait=False, checked=False: False
        try:
            skipped = win.present(nowait=True)
        finally:
            win.disp.present = real
        if skipped or win.present_seq != 2:
            fails.append(f"B3 跳过的 present 不应计数（seq={win.present_seq}）")
        if win.is_visible():
            fails.append("B4 全程不应 show")
        # keep_topmost：置顶后必须核对结果。系统会静默忽略这种交换链窗口的
        # 置顶请求（见 D3DStereoWindow.keep_topmost），所以这里不能只看调用
        # 返回值。隐藏窗口上置顶也应生效。
        win.keep_topmost()
        if not win.is_topmost():
            fails.append("B5 keep_topmost 后没有 TOPMOST")
        win.keep_topmost()
        if not win.is_topmost():
            fails.append("B5 连续调用后 TOPMOST 没保持")
        print("[B] 隐藏覆盖层 nowait present 计数 / 置顶生效 OK", flush=True)
    finally:
        try:
            win.close()
        except Exception:
            pass
        try:
            win.ren.destroy()
        except Exception:
            pass


def part_c(fails):
    """C) 深度复用的显示帧平滑步（D3DDepthPost.ema_step）：没有新推理结果的
    显示帧，用最近一次推理的 AA 后深度再走一步 EMA。离屏数值验证：
    原始深度从 0.2 跳到 0.8 后，链 α=0.5 → 0.5；每帧 ema_step(0.5) →
    0.65 → 0.725，逐步逼近（平滑按显示帧走，不是跳变）。"""
    from gpu import d3drender as dr
    from gpu import d3d_post as dp
    ren = dr.D3DRenderer(debug=False)
    try:
        post = dp.D3DDepthPost(ren, (64, 64))
        post.upload_raw(np.full((64, 64), 0.2, np.float32))
        post.spatial_chain(0.0, 1.0, aa_texels=0.0, temporal="ema", alpha=1.0)
        got = [float(ren.readback(post.depth, np.float32).mean())]
        post.upload_raw(np.full((64, 64), 0.8, np.float32))
        post.spatial_chain(0.0, 1.0, aa_texels=0.0, temporal="ema", alpha=0.5)
        got.append(float(ren.readback(post.depth, np.float32).mean()))
        tex = []
        for _ in range(2):
            tex.append(post.ema_step(0.5))
            got.append(float(ren.readback(post.depth, np.float32).mean()))
        want = (0.2, 0.5, 0.65, 0.725)
        for name, g, w in zip(("首帧", "链 α=0.5", "ema_step 1", "ema_step 2"),
                              got, want):
            if abs(g - w) > 0.01:
                fails.append(f"C {name}: 均值 {g:.3f}，应为 {w:.3f}")
        if tex[0] is None or tex[1] is None or tex[0] is tex[1]:
            fails.append("C ema_step 的显示深度应在两张乒乓纹理间交替")
        print("[C] 深度复用 显示帧 EMA 步 %s" % " ".join("%.3f" % g for g in got),
              flush=True)
    finally:
        ren.destroy()


def part_d(fails):
    """D) 推理提交时机（gpu.dml_shim.infer_pick，纯函数）。一次推理在 GPU 上
    是原子的，采到帧的同一拍就交会把它自己的采集拷贝和这一帧的桌面合成
    一起挡住，攥帧超过 15ms 就丢一次桌面刷新（2026-11-28 逐帧实测）。
    规则：先记成待提交，拷贝落地之后才交；忙（推理或它的输入读回在飞）时
    不收不交；永远算最新的一帧。"""
    from gpu.dml_shim import infer_pick
    a, b = (0, 11, 1000), (1, 12, 1017)      # (槽, 采集序号, 桌面呈现时刻)

    def chk(name, got, want):
        if got != want:
            fails.append(f"D {name}: 得到 {got}，应为 {want}")

    chk("采到帧的同一拍只记下不提交", infer_pick(a, None, False, True),
        (a, None, True))
    chk("拷贝落地后才提交", infer_pick(None, a, False, False),
        (None, a, False))
    chk("拷贝还在飞不提交", infer_pick(None, a, False, True),
        (a, None, False))
    chk("忙时不收新帧", infer_pick(b, None, True, False),
        (None, None, False))
    chk("忙时不提交", infer_pick(None, a, True, False),
        (a, None, False))
    chk("忙时新帧不顶掉待提交的", infer_pick(b, a, True, False),
        (a, None, False))
    chk("等待期间来了新帧就换成它", infer_pick(b, a, False, True),
        (b, None, True))
    chk("没有待提交帧", infer_pick(None, None, False, False),
        (None, None, False))
    print("[D] 推理提交时机 OK", flush=True)


def main():
    fails = []
    part_a(fails)
    part_b(fails)
    part_c(fails)
    part_d(fails)
    if fails:
        print(f"[entry-smoke] FAIL: {fails}")
        return 1
    print("[entry-smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
