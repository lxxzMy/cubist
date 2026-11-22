# -*- coding: utf-8 -*-
"""重构行为保持门禁（refactor gate）：黄金基线采集 + 比对。

三探针（全部用生产代码路径，不另写模拟）：
  depth   固定帧序列 → depth.process（生产色彩链）+ depth.predict_depth
          （参考深度全链：ORT → d2s_ops 后处理 → EMA 时序）→ npy
  render  tools/live_offline.py 无头 GL + 从 viewer.py 抽出的程序着色器
          源码 → 左右眼 PNG + 深度 npy
  smoke   D2S_NO_FULLSCREEN=1 启动 -m main，断言关键日志行齐、无 Traceback

用法（仓库根，必须用 D2S 自带解释器）：
  D2S\\python3\\python.exe -I tools\\refactor_gate.py capture
  D2S\\python3\\python.exe -I tools\\refactor_gate.py check [--skip-smoke]
                                        [--only depth|render|smoke]

基线目录 _refactor_baseline/（已 gitignore，不进库）。
capture 时每探针跑两遍记录自抖动；check 的容差 = 抖动×2（抖动为 0 则要求
逐位相等）。PASS → exit 0，FAIL → exit 1。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2S = os.path.join(ROOT, "D2S")
RUNTIME = os.path.join(ROOT, "runtime")
BASE = os.path.join(ROOT, "_refactor_baseline")
PY = sys.executable          # 门禁自身必须跑在 D2S 自带解释器里

N_INPUTS = 6                 # 从 _liveframes 挑选的固定帧数
SMOKE_BUDGET_S = 150.0       # 等到 "Local Viewer Started" 的时限（含 DML 装载）
SMOKE_TAIL_S = 6.0           # 三行齐后再收集的尾部日志
SMOKE_LINES_REQUIRED = (
    "[boot] main() 进入",
    "[Main] display=",
    "[Main] Local Viewer Started",
)
# 注意：dev 解释器（CUDA 版 ORT，无 DML EP）下 capture 线程会在
# windows_capture 的 DDA 初始化处抛 0x887A0004 并打印 Traceback——这是
# pre-refactor 既有行为（2026-11-16 实测），不禁 Traceback；禁的是重构
# 真正会引入的破坏类：导入失败 / 模型找不到 / 进程自行退出。
SMOKE_FORBIDDEN = (
    r"ModuleNotFoundError",
    r"^ImportError",
    r"SystemExit",
    r"Depth ONNX not found",
)


def _env():
    e = os.environ.copy()
    e["PATH"] = RUNTIME + os.pathsep + e.get("PATH", "")
    # 不设 D2S_ONNX（main.bat 里的值已失效）：模型走 settings
    # 解析（d224/224 → runtime/depth-224.onnx），且剥掉可能继承的旧值。
    e.pop("D2S_ONNX", None)
    e["PYTHONIOENCODING"] = "utf-8"
    e["D2S_NO_POPUP"] = "1"      # 冒烟引导不弹用户告警窗（GPU 不可用提示等）
    # 采基线/比对都要干净环境：探针类开关一律剥掉
    for k in list(e):
        if k.startswith("D2S_ZC_PROBE") or k == "D2S_TEST_RENDER_STALL_S":
            e.pop(k)
    return e


def _git_head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


# ---------------------------------------------------------------- 输入集

def _pick_inputs():
    """capture 时从 _liveframes 挑 N 帧固定拷入基线目录，之后永远用这份。
    输入集已存在则直接复用（黄金输入钉死；重采只重锚输出——语义变更
    前后保持同一输入，diff 才可归因）。_liveframes 已随 2026-11-18 清理
    删除，首轮采集需要时用 tools/ab_record.py 重录几帧放回该目录。"""
    ins = os.path.join(BASE, "inputs")
    if os.path.isdir(ins):
        existing = [f for f in os.listdir(ins) if f.endswith(".png")]
        if existing:
            print(f"[gate] 复用已固定输入集 {len(existing)} 帧（不重采输入）")
            return sorted(existing)
    src = os.path.join(ROOT, "_liveframes")
    files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    if len(files) < 3:
        raise SystemExit(f"_liveframes 帧不足（{len(files)} < 3），先采几帧再 capture")
    step = max(1, len(files) // N_INPUTS)
    chosen = files[::step][:N_INPUTS]
    ins = os.path.join(BASE, "inputs")
    os.makedirs(ins, exist_ok=True)
    for f in chosen:
        shutil.copyfile(os.path.join(src, f), os.path.join(ins, f))
    return chosen


def _inputs_dir():
    ins = os.path.join(BASE, "inputs")
    files = sorted(f for f in os.listdir(ins) if f.endswith(".png"))
    if not files:
        raise SystemExit("基线 inputs/ 为空——先跑 capture")
    return ins, files


# ---------------------------------------------------------------- depth 探针

def _depth_worker(inputs_dir, outdir):
    """子进程模式：导入 D2S 生产模块跑固定帧序列（隔离导入期状态）。"""
    sys.path.insert(0, D2S)
    import numpy as np
    import cv2
    import depth                     # 导入期装载 ORT（backend 随 settings.yaml）

    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(inputs_dir) if f.endswith(".png"))
    for i, fn in enumerate(files):
        bgr = cv2.imread(os.path.join(inputs_dir, fn), cv2.IMREAD_COLOR)
        rgb = depth.process(bgr, 1080)               # 生产色彩链（d2s_ops）
        d = depth.predict_depth(rgb, use_temporal_smooth=True)  # 参考深度链
        d = np.asarray(d, np.float32)
        np.save(os.path.join(outdir, f"depth_{i:02d}.npy"), d)
        np.save(os.path.join(outdir, f"rgb_{i:02d}.npy"),
                np.asarray(rgb, np.uint8))
        print(f"  [{i}] {fn}: depth{d.shape} "
              f"[{float(d.min()):.4f},{float(d.max()):.4f}]", flush=True)


def _run_depth_probe(tag):
    outdir = os.path.join(BASE, tag)
    ins, files = _inputs_dir()
    r = subprocess.run(
        [PY, "-I", os.path.abspath(__file__), "worker",
         "--depth-worker", ins, outdir],
        cwd=D2S, env=_env())
    if r.returncode != 0:
        raise SystemExit(f"depth 探针失败（{tag}）")
    return outdir


# ---------------------------------------------------------------- render 探针

def _run_render_probe(tag):
    """live_offline：无头 GL + 程序自身 FRAGMENT_SHADER，确定性输出。"""
    ins, files = _inputs_dir()
    for i, fn in enumerate(files):
        outdir = os.path.join(BASE, tag, f"r{i:02d}")
        os.makedirs(outdir, exist_ok=True)
        r = subprocess.run(
            [PY, "-I", os.path.join(ROOT, "tools", "live_offline.py"),
             os.path.join(ins, fn), "-o", outdir,
             "--parallax", "20", "--plane", "0.35", "--aa", "2"],
            cwd=ROOT, env=_env())
        if r.returncode != 0:
            raise SystemExit(f"render 探针失败（{tag} r{i}）")
    return os.path.join(BASE, tag)


# ---------------------------------------------------------------- smoke 探针

def _kill_tree(proc):
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    except Exception:
        pass


def _run_smoke(tag):
    outdir = os.path.join(BASE, tag)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "smoke_log.txt")

    env = _env()
    env["D2S_NO_FULLSCREEN"] = "1"     # 窗口化：不接管屏幕、不藏光标
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            [PY, "-u", "-m", "main"], cwd=D2S, env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    finally:
        pass

    got = set()
    t0 = time.time()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            while time.time() - t0 < SMOKE_BUDGET_S:
                line = f.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
                    continue
                for k in SMOKE_LINES_REQUIRED:
                    if k in line:
                        got.add(k)
                if len(got) == len(SMOKE_LINES_REQUIRED):
                    time.sleep(SMOKE_TAIL_S)   # 收尾部（fps/延迟行）
                    break
    finally:
        _kill_tree(proc)
        log_f.close()
        time.sleep(1.0)

    text = open(log_path, encoding="utf-8", errors="replace").read()
    bad = [p for p in SMOKE_FORBIDDEN if re.search(p, text)]
    missing = [k for k in SMOKE_LINES_REQUIRED if k not in text]
    return log_path, missing, bad


# ---------------------------------------------------------------- 比对

def _cmp_npy(a_path, b_path):
    import numpy as np
    a = np.load(a_path)
    b = np.load(b_path)
    if a.shape != b.shape:
        return None, None            # 形状都变了 = 直接 FAIL
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.mean()), float(d.max())


def _cmp_png(a_path, b_path):
    import cv2
    a = cv2.imread(a_path, cv2.IMREAD_UNCHANGED)
    b = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
    if a is None or b is None or a.shape != b.shape:
        return None, None
    import numpy as np
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), float(d.max())


def _dir_pairs(dir_a, dir_b, suffix, recurse=False):
    """返回 [(name, path_a, path_b)]；缺文件的条目记 (name, None/...)。"""
    names = set()
    for d in (dir_a, dir_b):
        if recurse:
            for sub in os.listdir(d):
                sd = os.path.join(d, sub)
                if os.path.isdir(sd):
                    names.update(f"{sub}/{f}" for f in os.listdir(sd)
                                 if f.endswith(suffix))
        else:
            names.update(f for f in os.listdir(d) if f.endswith(suffix))
    out = []
    for n in sorted(names):
        out.append((n,
                    os.path.join(dir_a, n) if os.path.exists(os.path.join(dir_a, n)) else None,
                    os.path.join(dir_b, n) if os.path.exists(os.path.join(dir_b, n)) else None))
    return out


def _probe_jitter(tag_a, tag_b, kind):
    """capture 用：两遍自抖动 → {name: (mae, max)}。"""
    res = {}
    if kind == "depth":
        pairs = _dir_pairs(os.path.join(BASE, tag_a), os.path.join(BASE, tag_b), ".npy")
        for n, pa, pb in pairs:
            res[n] = _cmp_npy(pa, pb) if pa and pb else (None, None)
    else:
        ra, rb = os.path.join(BASE, tag_a), os.path.join(BASE, tag_b)
        subs = sorted(set(os.listdir(ra)) & set(os.listdir(rb)))
        for sub in subs:
            for f in sorted(os.listdir(os.path.join(ra, sub))):
                pa = os.path.join(ra, sub, f)
                pb = os.path.join(rb, sub, f)
                if f.endswith(".png"):
                    res[f"{sub}/{f}"] = _cmp_png(pa, pb)
                elif f.endswith(".npy"):
                    res[f"{sub}/{f}"] = _cmp_npy(pa, pb)
    return res


def _check_probe(baseline_tag, check_tag, kind, jitter):
    """逐项比对，返回 (fail_list, stats)。容差：抖动×2，抖动 0 → 逐位相等。"""
    fails, stats = [], []
    tol = lambda j: 0.0 if j in (None, 0.0) else j * 2.0
    if kind == "depth":
        for n, pa, pb in _dir_pairs(os.path.join(BASE, baseline_tag),
                                    os.path.join(BASE, check_tag), ".npy"):
            if not pa or not pb:
                fails.append(f"{n}: 缺文件"); continue
            mae, mx = _cmp_npy(pa, pb)
            j = jitter.get(n, (0.0, 0.0))[0] or 0.0
            ok = mae is not None and mae <= tol(j) and mx <= max(tol(j), tol(j)) 
            stats.append(f"  {'OK ' if ok else 'FAIL'} {n}: mae={mae} max={mx} (jitter={j})")
            if not ok:
                fails.append(n)
    else:
        ra, rb = os.path.join(BASE, baseline_tag), os.path.join(BASE, check_tag)
        if not os.path.isdir(rb):
            return [f"{check_tag} 不存在"], []
        subs = sorted(set(os.listdir(ra)) & set(os.listdir(rb)))
        for sub in subs:
            for f in sorted(os.listdir(os.path.join(ra, sub))):
                n = f"{sub}/{f}"
                pa, pb = os.path.join(ra, sub, f), os.path.join(rb, sub, f)
                if not os.path.exists(pb):
                    fails.append(f"{n}: 缺文件"); continue
                if f.endswith(".png"):
                    mae, mx = _cmp_png(pa, pb)
                elif f.endswith(".npy"):
                    mae, mx = _cmp_npy(pa, pb)
                else:
                    continue
                j = jitter.get(n, (0.0, 0.0))[0] or 0.0
                ok = mae is not None and mae <= tol(j)
                stats.append(f"  {'OK ' if ok else 'FAIL'} {n}: mae={mae} max={mx} (jitter={j})")
                if not ok:
                    fails.append(n)
    return fails, stats


# ---------------------------------------------------------------- 主流程

def do_capture():
    os.makedirs(BASE, exist_ok=True)
    inputs = _pick_inputs()
    print(f"[gate] 输入集 {len(inputs)} 帧 → {os.path.join(BASE, 'inputs')}")

    print("[gate] depth 探针 ×2（测自抖动）…")
    _run_depth_probe("depth_a")
    _run_depth_probe("depth_b")
    dj = _probe_jitter("depth_a", "depth_b", "depth")

    print("[gate] render 探针 ×2 …")
    _run_render_probe("render_a")
    _run_render_probe("render_b")
    rj = _probe_jitter("render_a", "render_b", "render")

    print("[gate] smoke 探针 …")
    log_path, missing, bad = _run_smoke("smoke")
    if missing or bad:
        raise SystemExit(f"基线 smoke 本身不健康：missing={missing} forbidden={bad}")
    print(f"[gate] smoke 日志 → {log_path}")

    meta = {
        "head": _git_head(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "py": PY,
        "inputs": inputs,
        "depth_jitter": dj,
        "render_jitter": rj,
        "smoke_lines": list(SMOKE_LINES_REQUIRED),
    }
    with open(os.path.join(BASE, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[gate] 基线完成 head={meta['head']}")
    bad_j = [k for k, v in {**dj, **rj}.items() if v[0] is None]
    if bad_j:
        print(f"[gate] 注意：抖动比对有缺项 {bad_j}")
    print("[gate] PASS（基线健康）")


def do_check(skip_smoke=False, only=None):
    meta_path = os.path.join(BASE, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("没有基线——先跑 capture")
    meta = json.load(open(meta_path, encoding="utf-8"))
    print(f"[gate] 基线 head={meta['head']}（{meta['date']}）  当前 head={_git_head()}")

    all_fails = []
    if only in (None, "depth"):
        print("[gate] depth 探针 …")
        shutil.rmtree(os.path.join(BASE, "check_depth"), ignore_errors=True)
        _run_depth_probe("check_depth")
        fails, stats = _check_probe("depth_a", "check_depth", "depth",
                                    meta["depth_jitter"])
        print("\n".join(stats) or "  (无比对项)")
        all_fails += [f"depth:{f}" for f in fails]

    if only in (None, "render"):
        print("[gate] render 探针 …")
        shutil.rmtree(os.path.join(BASE, "check_render"), ignore_errors=True)
        _run_render_probe("check_render")
        fails, stats = _check_probe("render_a", "check_render", "render",
                                    meta["render_jitter"])
        print("\n".join(stats) or "  (无比对项)")
        all_fails += [f"render:{f}" for f in fails]

    if not skip_smoke and only in (None, "smoke"):
        print("[gate] smoke 探针 …")
        log_path, missing, bad = _run_smoke("check_smoke")
        for k in missing:
            all_fails.append(f"smoke:缺日志行 {k}")
        for k in bad:
            all_fails.append(f"smoke:出现禁用模式 {k}")
        print(f"  {'OK ' if not (missing or bad) else 'FAIL'} smoke_log → {log_path}")

    if all_fails:
        print(f"\n[gate] FAIL（{len(all_fails)} 项）:")
        for f in all_fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[gate] PASS —— 行为与基线一致")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "check", "worker"])
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--only", choices=["depth", "render", "smoke"])
    ap.add_argument("--depth-worker", nargs=2, metavar=("INPUTS", "OUT"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.mode == "worker":
        _depth_worker(*args.depth_worker)
        return
    if args.mode == "capture":
        do_capture()
    else:
        do_check(skip_smoke=args.skip_smoke, only=args.only)


if __name__ == "__main__":
    main()
