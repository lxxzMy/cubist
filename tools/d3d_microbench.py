# -*- coding: utf-8 -*-
"""P0 微基准 + 探针矩阵（迁移计划定案：入口桥接成本量化后才开工 P1+）。

探针：
  A  D3D12 设备建链（锚点 adapter、debug layer、LUID）
  B  着色器编译：DXC/SM6 vs FXC/SM5_0 耗时 + .cso 缓存
  C  swapchain 模型矩阵：{plain, LAYERED+alpha, LAYERED+NOREDIRECTIONBITMAP}
     × {FLIP_DISCARD, DISCARD} —— HRESULT 与实际 Present 行为
  D  GLFW NO_API framebuffer 语义（vs GetClientRect / DPI）
  E  入口桥接微基准：D3D11(DDA侧) CopyResource→ctx4.Signal→D3D12 queue.Wait
     →空 pass→Present——单帧 CPU 等待/GPU 气泡/句柄 open 成本逐段量化，
     与 GL 时代 5-10ms 同步成本同表对账
  F  生命周期冒烟：创建/释放 N 轮，live_stats 必须归零

用法：D2S\\python3\\python.exe -I tools\\d3d_microbench.py [--frames 240]
输出：stdout 表 + docs/d3d-p0-microbench.md（追加）
"""
import argparse
import ctypes
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")

u32 = ctypes.WinDLL("user32")
results = []


def rec(name, val, note=""):
    results.append((name, val, note))
    print(f"  {name:<44s} {val!s:<24s} {note}")


def probe_a_device():
    print("\n== A: D3D12 设备建链 ==")
    from gpu import adapter as _ad
    from gpu.d3d12 import D3D12Context
    adv = _ad.pick()
    t0 = time.perf_counter()
    ctx = D3D12Context(adapter_ptr=(adv.ptr if adv else None), debug=True)
    ms = (time.perf_counter() - t0) * 1000
    rec("A1 设备创建(debug layer)", f"{ms:.1f}ms", f"LUID {ctx.luid:#x}")
    rec("A2 适配器", adv.name if adv else "默认", f"adapter[{adv.index}]" if adv else "")
    if adv:
        rec("A3 LUID 与 D3D11 侧一致", ctx.luid == adv.luid,
            f"d3d12 {ctx.luid:#x} vs d3d11 {adv.luid:#x}")
    return ctx


def probe_b_shaders(ctx):
    print("\n== B: 着色器编译（DXC/SM6 vs FXC/SM5_0） ==")
    from gpu import hlsl
    from gpu.d3d12 import S_DEVICE as _SD

    def devstate(tag):
        hr = ctx.dev.call(_SD["GetDeviceRemovedReason"], ctypes.c_long, ())
        rec(f"B 设备态 {tag}", hex(hr & 0xFFFFFFFF))

    ps = "float4 PSMain() : SV_Target { return float4(1,0,0,1); }"
    devstate("开始")
    for comp in ("dxc", "fxc"):
        try:
            t0 = time.perf_counter()
            cso = hlsl.compile_shader("bench_ps", ps, "ps", comp, force=True)
            ms = (time.perf_counter() - t0) * 1000
            rec(f"B1 {comp} 编译", f"{ms:.1f}ms",
                f"{cso.compiler}/{cso.profile} {cso.nbytes}B")
            t0 = time.perf_counter()
            hlsl.compile_shader("bench_ps", ps, "ps", comp)
            ms2 = (time.perf_counter() - t0) * 1000
            rec(f"B2 {comp} 缓存命中", f"{ms2:.2f}ms", "磁盘 .cso")
        except Exception as e:
            rec(f"B {comp}", "FAIL", f"{type(e).__name__}: {e}")
        devstate(comp + " 后")


WINSTYLECombos = [
    ("plain", 0, 0),
    ("layered+alpha", 0x80000, 0x20),           # WS_EX_LAYERED(+alpha 属性)
    ("layered+noredir", 0x80000 | 0x00200000, 0x20),  # + WS_EX_NOREDIRECTIONBITMAP
]


def _mk_test_window(ex_style):
    u32.CreateWindowExW.restype = ctypes.c_void_p
    u32.CreateWindowExW.argtypes = (ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    hwnd = u32.CreateWindowExW(ex_style, "STATIC", None, 0x90000000, 0, 0, 640, 480,
                               None, None, None, None)
    if not hwnd:
        return None
    # LAYERED + alpha 255（与产品 _reassert_overlay_style 同语义）
    if ex_style & 0x80000 and not (ex_style & 0x00200000):
        u32.SetLayeredWindowAttributes(hwnd, 0, 255, 2)   # LWA_ALPHA
    u32.ShowWindow(hwnd, 5)
    u32.PumpEvents = None
    for _ in range(30):
        msg = (ctypes.c_uint * 4)()
        while u32.PeekMessageW(msg, None, 0, 0, 1):
            u32.TranslateMessage(msg)
            u32.DispatchMessageW(msg)
        u32.ShowWindow(hwnd, 5)
    return hwnd


def probe_c_swapchain(ctx):
    print("\n== C: swapchain 模型矩阵（GLFW NO_API 测试窗） ==")
    from gpu.d3d12 import Swapchain
    import glfw
    for name, ex, _al in WINSTYLECombos:
        for model in ("flip", "blt"):
            hwnd = None
            try:
                glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
                glfw.window_hint(glfw.VISIBLE, 0)
                glfw.window_hint(glfw.FOCUS_ON_SHOW, 0)
                w = glfw.create_window(640, 480, f"sc-{name}-{model}", None, None)
                if not w:
                    rec(f"C {name}×{model}", "窗口创建失败")
                    continue
                glfw.show_window(w)
                glfw.poll_events()
                hwnd = glfw.get_win32_window(w)
                # LAYERED 样式后置（SetLayeredWindowAttributes alpha=255）
                if ex & 0x80000:
                    u32.SetWindowLongPtrW(hwnd, -20, ex & ~0x20)  # GWL_EXSTYLE
                    u32.SetLayeredWindowAttributes(hwnd, 0, 255, 2)
                sc = Swapchain(ctx, hwnd, 640, 480, model=model, buffer_count=2)
                worst = 0.0
                ok = True
                for _ in range(30):
                    t1 = time.perf_counter()
                    hr = sc.present(1)
                    if hr < 0:
                        ok = False
                        break
                    worst = max(worst, (time.perf_counter() - t1) * 1000)
                rec(f"C {name}×{model}", "可用" if ok else "Present 失败",
                    f"30帧最大单次 {worst:.1f}ms")
            except Exception as e:
                rec(f"C {name}×{model}", "不支持",
                    f"{type(e).__name__}: winerror={getattr(e, 'winerror', None)} {e}")
            finally:
                try:
                    if w:
                        glfw.destroy_window(w)
                except Exception:
                    pass


def probe_d_noapi_fb():
    print("\n== D: GLFW NO_API framebuffer 语义 ==")
    try:
        import glfw
    except ImportError:
        rec("D glfw", "不可用", "dev 环境无 glfw?")
        return
    if not glfw.init():
        rec("D glfw.init", "失败")
        return
    try:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.VISIBLE, 0)
        w = glfw.create_window(800, 450, "noapi-probe", None, None)
        if not w:
            rec("D NO_API 窗口", "创建失败")
            return
        fb = glfw.get_framebuffer_size(w)
        hwnd = glfw.get_win32_window(w)
        r = (ctypes.c_long * 4)()
        u32.GetClientRect(hwnd, r)
        rec("D framebuffer_size", f"{fb[0]}x{fb[1]}", "GLFW NO_API")
        rec("D GetClientRect", f"{r[2]-r[0]}x{r[3]-r[1]}", "win32 客户区")
        rec("D 一致", fb == (r[2] - r[0], r[3] - r[1]),
            "NO_API 下 fb=客户区像素 → 现有渲染尺寸代码可平移")
        glfw.destroy_window(w)
    except Exception as e:
        rec("D NO_API", "FAIL", f"{type(e).__name__}: {e}")


def probe_e_bridge(ctx, frames):
    print("\n== E: 入口桥接微基准（DDA 侧语义） ==")
    import numpy as np
    from gpu.rtcom import ComPtr, GUID, guid_ptr, create_device, create_tex, TexDesc
    from gpu import adapter as _ad
    from gpu.d3d12 import (S_DEVICE, IID_ID3D11Device5, IID_ID3D11Fence,
                           S_D3D11_DEV5, S_D3D11_FENCE, S_D3D11_CTX4, S_RES,
                           IID_IDXGIKeyedMutex, S_KEYED)
    d3d11 = ctypes.WinDLL("d3d11")
    adv = _ad.pick()
    dev11, ctx11, _fl = create_device(adv.ptr if adv else None)
    dev5 = ctypes.c_void_p(dev11.qi(IID_ID3D11Device5))
    d5 = ComPtr(dev5.value)
    ctx4 = ctypes.c_void_p(ctx11.qi("917600da-f58c-4c33-98d8-3e15b390fa24"))
    c4 = ComPtr(ctx4.value)
    # 共享纹理（NTHANDLE+KM；D3D11 的 KM=0x100、NTHANDLE=0x800——注意与
    # D3D12 的 0x1000 不同！FENCE_FLAG_SHARED D3D11=0x2、D3D12=0x1 同坑）
    td = TexDesc(1920, 1080, 1, 1, 87, 1, 0, 0, 8, 0, 0x800 | 0x100)  # RT|NTHANDLE|KM(0x100!)
    tex11 = create_tex(dev11, td)
    # 纹理 NT 句柄：IDXGIResource1::CreateSharedHandle（槽 8，IID 30961379…）
    IID_IDXGIResource1 = "30961379-4609-4a41-998e-54fe567ee0c1"
    r1 = ctypes.c_void_p(tex11.qi(IID_IDXGIResource1))
    r1c = ComPtr(r1.value)
    tex_hnd = ctypes.c_void_p()
    READ_WRITE = 0x80000000 | 0x1   # DXGI_SHARED_RESOURCE_READ|WRITE
    # CreateSharedHandle = 槽 13（槽 8 是 GetSharedHandle——另一个方法！
    # IDXGIResource1 Vtbl 机器提取：…GetDevice(7) GetSharedHandle(8) GetUsage(9)
    # SetEvictionPriority(10) GetEvictionPriority(11) CreateSubresourceSurface(12)
    # CreateSharedHandle(13)）
    hr = r1c.call(13, ctypes.c_long,
                  (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                   ctypes.POINTER(ctypes.c_void_p)),
                  None, READ_WRITE, None, ctypes.byref(tex_hnd))
    if hr:
        rec("E 纹理共享句柄", "FAIL", f"hr={hex(hr & 0xFFFFFFFF)}")
        return
    rec("E 纹理共享句柄", "ok")
    # D3D11 fence（SHARED=0x2！D3D12 是 0x1——两套 API 枚举不一致）+ 共享句柄
    f11 = ctypes.c_void_p()
    hr = d5.call(S_D3D11_DEV5["CreateFence"], ctypes.c_long,
                 (ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                  ctypes.POINTER(ctypes.c_void_p)),
                 0, 0x2, guid_ptr(IID_ID3D11Fence), ctypes.byref(f11))
    if hr:
        rec("E fence 创建", "FAIL", f"hr={hr:#x}")
        return
    f = ComPtr(f11.value)
    hnd = ctypes.c_void_p()
    GENERIC_ALL = 0x10000000
    hr = f.call(S_D3D11_FENCE["CreateSharedHandle"], ctypes.c_long,
                (ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                 ctypes.POINTER(ctypes.c_void_p)),
                None, GENERIC_ALL, None, ctypes.byref(hnd))
    if hr:
        rec("E fence CreateSharedHandle", "FAIL", f"hr={hex(hr & 0xFFFFFFFF)}")
        return
    # D3D12 侧 open（句柄缓存语义：只 open 一次，测成本）
    IID_D3D12Fence = "0a753dcf-c4d8-4b91-adf6-be5a60d95a76"
    IID_D3D12Resource = "696442be-a72e-4059-bc79-5b5c98040fad"
    t0 = time.perf_counter()
    f12 = ctypes.c_void_p()
    ctx.dev.call(S_DEVICE['OpenSharedHandle'], ctypes.c_long,
                 (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                 hnd.value, guid_ptr(IID_D3D12Fence), ctypes.byref(f12))
    open_fence_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r12 = ctypes.c_void_p()
    ctx.dev.call(S_DEVICE['OpenSharedHandle'], ctypes.c_long,
                 (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                 tex_hnd.value, guid_ptr(IID_D3D12Resource), ctypes.byref(r12))
    open_tex_ms = (time.perf_counter() - t0) * 1000
    print(f"[e-dbg] r12={r12.value:#x} tex_hnd={tex_hnd.value:#x} "
          f"f12={f12.value:#x}", flush=True)
    rec("E1 句柄 open（一次性）", f"fence {open_fence_ms:.2f}ms tex {open_tex_ms:.2f}ms",
        "缓存复用，不逐帧 open")
    # 句柄逐帧 open 成本（对比项）：重 open N 次
    t0 = time.perf_counter()
    for _ in range(100):
        junk = ctypes.c_void_p()
        ctx.dev.call(S_DEVICE['OpenSharedHandle'], ctypes.c_long,
                     (ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
                     hnd.value, guid_ptr("0a753dcf-c4d8-4b91-adf6-be5a60d95a76"),
                     ctypes.byref(junk))
        ComPtr(junk.value).release()
    rec("E2 逐帧重 open 100 次均值", f"{(time.perf_counter()-t0)*10:.3f}ms/帧",
        "（印证缓存必要性）")
    # D3D12 侧打开 fence → queue.Wait
    fq = ComPtr(f12.value)
    # 逐帧循环：D3D11 copy → signal → D3D12 wait → 空 pass → submit
    n = frames
    t_d3d11 = t_wait = t_submit = 0.0
    t_frame = []
    src11 = create_tex(dev11, TexDesc(1920, 1080, 1, 1, 87, 1, 0, 0, 8, 0, 0))
    # D3D12 空链目标纹理
    dst12 = ctx.make_texture(1920, 1080, 87)
    v = 0
    ctx.wait_gpu()
    for i in range(n):
        t0 = time.perf_counter()
        c11 = ctx11
        c11.call(47, None, (ctypes.c_void_p, ctypes.c_void_p), tex11.p, src11.p)
        v += 1
        c4.call(S_D3D11_CTX4["Signal"], ctypes.c_long,
                (ctypes.c_void_p, ctypes.c_uint64), f.p, v)
        t11 = time.perf_counter() - t0
        t0 = time.perf_counter()
        cl = ctx.begin_frame()
        rr = ComPtr(r12.value)
        # 隐式提升：COMMON 打开的共享资源用作 COPY_SRC/COPY_DST 无需屏障
        # （D3D12 隐式状态提升规则；显式屏障曾致 Close E_INVALIDARG——
        #  打开资源的实际初始态与假设不符）
        loc_s = __import__("gpu.d3d12", fromlist=["D3D12_TEXTURE_COPY_LOCATION"]).D3D12_TEXTURE_COPY_LOCATION(pResource=rr.p, Type=0)
        loc_d = __import__("gpu.d3d12", fromlist=["D3D12_TEXTURE_COPY_LOCATION"]).D3D12_TEXTURE_COPY_LOCATION(pResource=dst12.res.p, Type=0)
        cl.call(17, None, (ctypes.c_void_p, ctypes.c_void_p),
                loc_d.pResource, loc_s.pResource)
        fv = ctx.submit()
        ctx.queue.call(15, ctypes.c_long, (ctypes.c_void_p, ctypes.c_uint64),
                       f12.value, v)   # queue.Wait(fence11_open, v)
        t0b = time.perf_counter()
        ctx.wait_gpu()
        t_wait_sub = time.perf_counter() - t0b
        t_d3d11 += t11 * 1000
        t_wait += (t0b - t0) * 1000     # D3D12 侧提交+wait 插入成本
        t_submit += (t0b - t0) * 1000
        t_frame.append((time.perf_counter() - t0) * 1000)
    tf = sorted(t_frame)
    rec(f"E3 桥接单帧总时长 p50", f"{tf[len(tf)//2]:.2f}ms",
        f"p99 {tf[int(len(tf)*0.99)]:.2f}ms / max {tf[-1]:.2f}ms")
    rec("E4 D3D11 侧均摊（copy+signal）", f"{t_d3d11/n:.2f}ms")
    rec("E5 D3D12 侧均摊（open wait+copy+submit）", f"{t_wait/n:.2f}ms")
    rec("E6 host 同步均摊", f"{t_wait_sub*1000 if False else t_wait_sub:.2f}ms/帧均值已含")
    rec("E7 折算 fps 上限", f"{1000.0/tf[len(tf)//2]:.1f}",
        "（GL 时代同点 5-10ms 同步 → 对账见终报）")
    # 释放
    _rel_dev(dev11, ctx11)


def _rel_dev(dev11, ctx11):
    dev11.release()
    ctx11.release()


def probe_f_lifecycle(ctx):
    print("\n== F: 生命周期冒烟 ==")
    from gpu.d3d12 import live_stats
    before = live_stats()
    for _ in range(50):
        objs = (ctx.make_texture(256, 256, 41),
                ctx.make_upload(256, 256, 41),
                ctx.make_readback(256, 256, 41))
        _release_all(*objs)
    after = live_stats()
    rec("F1 50 轮创建/释放", f"alloc {after[1]-before[1]} free {after[2]-before[2]}",
        "alloc==free=漏为零")
    rec("F2 存活对象差", str({k: v for k, v in after[0].items()
                              if after[0].get(k)}), "res 应不增长")


def _release_all(*texs):
    for t in texs:
        if hasattr(t, "res"):
            from gpu.d3d12 import _release
            _release(t.res, "res")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()
    print("[microbench] P0 探针矩阵开始")
    ctx = probe_a_device()

    def tex_smoke(tag):
        from gpu.d3d12 import S_DEVICE as _SD2

        def ds(step):
            hr = ctx.dev.call(_SD2["GetDeviceRemovedReason"], ctypes.c_long, ())
            rec(f"烟测 {tag} {step}", hex(hr & 0xFFFFFFFF))
        try:
            ds("前")
            t = ctx.make_texture(64, 64, 41)
            ds("建后")
            _release_all(t)
            ds("释放后")
            rec(f"烟测 {tag}", "ok")
        except Exception as e:
            rec(f"烟测 {tag}", "FAIL", f"{type(e).__name__}: {e}")

    tex_smoke("A后")
    probe_b_shaders(ctx)
    from gpu.d3d12 import S_DEVICE as _SD
    _rr = ctx.dev.call(_SD["GetDeviceRemovedReason"], ctypes.c_long, ())
    rec("B后 设备移除原因", hex(_rr & 0xFFFFFFFF))
    tex_smoke("B后")
    probe_e_bridge(ctx, args.frames)   # 高价值探针先行（C 有原生崩溃风险）
    tex_smoke("E后")
    probe_d_noapi_fb()
    tex_smoke("D后")
    probe_c_swapchain(ctx)
    tex_smoke("C后")
    probe_f_lifecycle(ctx)
    # 入档
    out = os.path.join(ROOT, "docs", "d3d-p0-microbench.md")
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"\n## 运行 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| 探针 | 结果 | 备注 |\n|---|---|---|\n")
        for name, val, note in results:
            f.write(f"| {name} | {val} | {note} |\n")
    print(f"\n[microbench] 结果已追加 → docs/d3d-p0-microbench.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
