# -*- coding: utf-8 -*-
"""T2 门禁：shim v2 数值对账（DML 直入 vs CPU ONNX 会话）。

管线：合成 CHW 输入 → D3DDepthPrep 同款 flat 几何 → D3DRenderer 上传
prep 纹理 → shim v2（DML 直入同队列）→ 输出 CPU float vs
CPU ONNX 会话同输入 → MAE。

断言：
  S1 零输入 sanity：MAE < 1e-5
  S2 真帧 20 帧：MAE < 1e-3
  S3 times6 时序：copy+exec < 2ms、run < 15ms（DML 单帧）
用法：D2S\\python3\\python.exe -I tools\\shim2_sanity.py
"""
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "D2S"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["PATH"] = os.path.join(ROOT, "runtime") + os.pathsep + os.environ.get("PATH", "")
os.environ["D2S_ONNX"] = ""

import numpy as np

RES = 224
N = 20


def cpu_session():
    import depth as depth_mod
    sess = depth_mod.depth_model.session
    name = depth_mod.depth_model.input_name
    return sess, name


def main():
    fails = []
    from gpu import d3drender as dr
    from gpu import adapter as _ad
    from gpu.d3d_prep import D3DDepthPrep
    from gpu.dml_shim import DmlZeroCopyShim2

    import cv2
    frames = []
    for f in sorted(os.listdir(os.path.join(ROOT, "_refactor_baseline", "inputs"))):
        if f.endswith(".png"):
            frames.append(cv2.cvtColor(cv2.imread(
                os.path.join(ROOT, "_refactor_baseline", "inputs", f)),
                cv2.COLOR_BGR2RGB))
    frames = [cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA)
              for f in frames]

    adv = _ad.pick()
    ren = dr.D3DRenderer(luid=(adv.luid if adv else 0), debug=False)
    prep = D3DDepthPrep(ren, RES)   # flat 192×784 R32F

    model_path = os.path.join(ROOT, "runtime", "depth-224.onnx")
    # DML 需要 capi 的 onnxruntime.dll/DirectML.dll——优先工具链
    # ort-dml（T2 装的 onnxruntime-directml），回退 runtime 遍历
    capi = os.path.join(ROOT, "_toolchain", "dl", "ort-dml", "onnxruntime",
                        "capi")
    if not os.path.isfile(os.path.join(capi, "onnxruntime.dll")):
        capi = None
        for base, dirs, files in os.walk(os.path.join(ROOT, "runtime")):
            if "onnxruntime.dll" in files and "DirectML.dll" in files:
                capi = base
                break
    if capi is None:
        print("[shim2] 无 DML capi（onnxruntime.dll+DirectML.dll）")
        return 1
    print(f"[shim2] DML capi: {capi}", flush=True)

    shim = DmlZeroCopyShim2(capi, model_path, RES,
                            ren.get_dev_ptr(), ren.get_queue_ptr(),
                            prep.prep.res, prep.prep_w, prep.prep_h)
    print("[shim2] create OK", flush=True)

    sess, in_name = cpu_session()
    out_buf = np.empty(RES * RES, np.float32)

    # S1 零输入
    z = np.zeros((RES, RES, 3), np.uint8)
    bgra = np.empty((RES, RES, 4), np.uint8)
    bgra[:, :, 0] = z[:, :, 2]; bgra[:, :, 1] = z[:, :, 1]
    bgra[:, :, 2] = z[:, :, 0]; bgra[:, :, 3] = 255
    prep.set_frame(bgra)
    prep.run()
    chw = np.ascontiguousarray(prep.readback().ravel().reshape(3, RES, RES))
    shim.run(ctypes.cast(out_buf.ctypes.data, ctypes.POINTER(ctypes.c_float)),
             out_buf.nbytes)
    cpu_out = sess.run(None, {in_name: chw[None]})[0].reshape(-1)
    m0 = float(np.mean(np.abs(out_buf - cpu_out)))
    print(f"[S1] 零输入 MAE={m0:.2e}（期望<1e-5）", flush=True)
    if m0 >= 1e-5:
        fails.append(f"S1 零输入 {m0:.2e}")

    # S2 真帧
    maes = []
    for i in range(N):
        f = frames[i % len(frames)]
        bgra[:, :, 0] = f[:, :, 2]; bgra[:, :, 1] = f[:, :, 1]
        bgra[:, :, 2] = f[:, :, 0]; bgra[:, :, 3] = 255
        prep.set_frame(bgra)
        prep.run()
        chw = np.ascontiguousarray(prep.readback().ravel().reshape(3, RES, RES))
        shim.run(ctypes.cast(out_buf.ctypes.data,
                             ctypes.POINTER(ctypes.c_float)), out_buf.nbytes)
        cpu_out = sess.run(None, {in_name: chw[None]})[0].reshape(-1)
        maes.append(float(np.mean(np.abs(out_buf - cpu_out))))
    mmax = max(maes)
    mmed = sorted(maes)[len(maes) // 2]
    print(f"[S2] 真帧 {N} 帧 MAE 中位 {mmed:.2e} 最大 {mmax:.2e}（期望<1e-3）",
          flush=True)
    if mmax >= 1e-3:
        fails.append(f"S2 真帧 {mmax:.2e}")

    # S3 时序
    t6 = shim.times6()
    print(f"[S3] times6: copy+exec {t6[0]:.2f}ms run {t6[1]:.2f}ms", flush=True)
    if t6[0] >= 2.0:
        fails.append(f"S3 copy {t6[0]:.2f}")
    if t6[1] >= 15.0:
        fails.append(f"S3 run {t6[1]:.2f}")

    shim.close()
    ren.destroy()
    if fails:
        print(f"[shim2] FAIL: {fails}")
        return 1
    print("[shim2] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
