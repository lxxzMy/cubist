# -*- coding: utf-8 -*-
"""降采样核对抗：高分辨率 → 224 的算法选择（真实图片滑窗 + 真实模型）。

产品 D3D 现状：PREP_FS 单点双线性采样（无 mip / 无多抽头）→ 大倍率降采样
容易混叠。本工具对比：
  linear   单点双线性（＝产品 shader 现状）
  area     INTER_AREA（面积平均，正确抗混叠）
  cubic    INTER_CUBIC
  lanczos  INTER_LANCZOS4
  area2    两步 INTER_AREA（先到 1.5×，再到 224）

指标同 preblur：内容点 raw 时间波动 / 局部残差 / 过产品链 z 波动 /
与 area 参照的结构偏差 / 梯度能量。

用法：D2S\\python3\\python.exe -I tools\\resample_bakeoff.py [图片] [帧数]
"""
import hashlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "D2S"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from pan_algorithm_bakeoff import (bilinear, fit_affine,  # noqa: E402
                                   fit_affine_robust)
from preblur_bakeoff import (DEF_IMG, MODEL, RES, WIN,  # noqa: E402
                             chain_z, grad_energy, load_frames)

KERNELS = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
           "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def infer_img(fr, name):
    sm = cv2.resize(fr, (RES, RES), interpolation=KERNELS[name])
    return sm


def infer_area2(fr):
    h, w = fr.shape[:2]
    mid = (int(round(w / 1.5)), int(round(h / 1.5)))
    sm = cv2.resize(fr, mid, interpolation=cv2.INTER_AREA)
    return cv2.resize(sm, (RES, RES), interpolation=cv2.INTER_AREA)


def infer_variant(img_path, nframes, name):
    key = hashlib.sha1((img_path + f"|{os.path.getmtime(img_path)}|"
                        f"{nframes}|{name}|v1").encode("utf-8", "ignore")
                       ).hexdigest()[:16]
    cache = os.path.join(tempfile.gettempdir(),
                         f"d2s_resamp_{name}_{key}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return [d[f"r{i}"] for i in range(nframes)]
    frames, _ = load_frames(img_path, nframes)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL, so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    out = []
    for i, fr in enumerate(frames):
        sm = infer_area2(fr) if name == "area2" else infer_img(fr, name)
        x = sm.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None]
        out.append(np.squeeze(sess.run(None, {iname: x})[0]).astype(np.float32))
        print(f"\r  {name} 推理 {i + 1}/{len(frames)}", end="", flush=True)
    print()
    np.savez_compressed(cache, **{f"r{i}": r for i, r in enumerate(out)})
    return out


def load_full_frames(img_path, nframes, dx=8, dy=2):
    """真实倍率模式：整幅 1920x1080 平移（wrap）后直接降到 224（8.57×）。"""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise SystemExit(f"读图失败: {img_path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    return [np.roll(img, (-t * dx, -t * dy), axis=(1, 0))
            for t in range(nframes)], (W, H)


def infer_full(img_path, nframes, name):
    key = hashlib.sha1((img_path + f"|{os.path.getmtime(img_path)}|"
                        f"{nframes}|{name}|full|v1")
                       .encode("utf-8", "ignore")).hexdigest()[:16]
    cache = os.path.join(tempfile.gettempdir(),
                         f"d2s_resamp_full_{name}_{key}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return [d[f"r{i}"] for i in range(nframes)]
    frames, _ = load_full_frames(img_path, nframes)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL, so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    out = []
    for i, fr in enumerate(frames):
        if name == "area2":
            h, w = fr.shape[:2]
            mid = (int(round(w / 1.5)), int(round(h / 1.5)))
            sm = cv2.resize(fr, mid, interpolation=cv2.INTER_AREA)
            sm = cv2.resize(sm, (RES, RES), interpolation=cv2.INTER_AREA)
        else:
            sm = cv2.resize(fr, (RES, RES), interpolation=KERNELS[name])
        x = sm.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)[None]
        out.append(np.squeeze(sess.run(None, {iname: x})[0]).astype(np.float32))
        print(f"\r  full/{name} 推理 {i + 1}/{len(frames)}", end="", flush=True)
    print()
    np.savez_compressed(cache, **{f"r{i}": r for i, r in enumerate(out)})
    return out


def report(out, px, py, names):
    n = px.shape[0]
    ref = out["area"]
    ref_grad = np.mean([grad_energy(d) for d in ref])
    print("\n=== 降采样核对比（参照 = INTER_AREA）===")
    print(f"{'kernel':<9}{'raw std':>9}{'raw p2p':>9}{'local resid':>12}"
          f"{'vs area RMS':>12}{'corr':>8}{'grad':>7}{'chain z std':>12}")
    for nm in names:
        raws = out[nm]
        rp = np.stack([bilinear(raws[t], px[t], py[t]) for t in range(n)])
        std = float(rp.std(axis=0).mean())
        p2p = float((rp.max(axis=0) - rp.min(axis=0)).mean())
        res = []
        for t in range(1, n):
            p = bilinear(raws[t - 1], px[t - 1], py[t - 1])
            c = bilinear(raws[t], px[t], py[t])
            _, _, ratio = fit_affine_robust(p, c)
            res.append(ratio)
        rms, corr = [], []
        for t in range(n):
            a, b, _ = fit_affine(ref[t].ravel(), raws[t].ravel())
            resid = raws[t] - (a * ref[t] + b)
            lo, hi = np.percentile(ref[t], [2.0, 98.0])
            rms.append(float(np.sqrt(np.mean(resid ** 2)) /
                             max(1e-6, hi - lo)))
            x = ref[t].ravel() - ref[t].mean()
            y = raws[t].ravel() - raws[t].mean()
            corr.append(float(np.dot(x, y) /
                              max(1e-9, np.linalg.norm(x) *
                                  np.linalg.norm(y))))
        grad = float(np.mean([grad_energy(d) for d in raws]) / ref_grad)
        zz = chain_z(raws, px, py)
        zstd = float(zz[3:].std(axis=0).mean())
        print(f"{nm:<9}{std:>9.4f}{p2p:>9.4f}{np.mean(res):>12.3f}"
              f"{np.mean(rms):>12.4f}{np.mean(corr):>8.4f}{grad:>7.3f}"
              f"{zstd:>12.4f}")


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEF_IMG
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    full = len(sys.argv) > 3 and sys.argv[3] == "full"
    names = ["linear", "area", "cubic", "lanczos", "area2"]
    if full:
        _, (W, H) = load_full_frames(img_path, 1)
        n = nframes
        xs = np.arange(240, W - 240, 40, dtype=np.float64)
        ys = np.arange(160, H - 160, 40, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        sx_, sy_ = gx.ravel(), gy.ravel()
        px = np.zeros((n, sx_.size), np.float64)
        py = np.zeros((n, sy_.size), np.float64)
        for t in range(n):
            px[t] = (((sx_ - t * 8) % W) + 0.5) / W * RES - 0.5
            py[t] = (((sy_ - t * 2) % H) + 0.5) / H * RES - 0.5
        print(f"整幅 {W}x{H} → 224（8.57×）  {n} 帧  "
              f"probe {sx_.size} 点", flush=True)
        out = {nm: infer_full(img_path, nframes, nm) for nm in names}
    else:
        frames, offs = load_frames(img_path, nframes)
        n = len(frames)
        max_ox = max(o[0] for o in offs)
        min_ox = min(o[0] for o in offs)
        max_oy = max(o[1] for o in offs)
        min_oy = min(o[1] for o in offs)
        xs = np.arange(max_ox + 40, WIN + min_ox - 40, 35, dtype=np.float64)
        ys = np.arange(max_oy + 40, WIN + min_oy - 40, 35, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        sx_, sy_ = gx.ravel(), gy.ravel()
        px = np.zeros((n, sx_.size), np.float64)
        py = np.zeros((n, sy_.size), np.float64)
        for t, (ox, oy) in enumerate(offs):
            px[t] = (sx_ + 0.5 - ox) / WIN * RES - 0.5
            py[t] = (sy_ + 0.5 - oy) / WIN * RES - 0.5
        print(f"图片 {os.path.basename(img_path)}  {n} 帧  "
              f"probe {sx_.size} 点", flush=True)
        out = {nm: infer_variant(img_path, nframes, nm) for nm in names}
    report(out, px, py, names)


if __name__ == "__main__":
    main()
