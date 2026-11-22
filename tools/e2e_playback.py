# -*- coding: utf-8 -*-
"""回放 E2E v2：真实发布包管线 + 受控输入（静态 + 平移运动）+ 输出比对。

v2 新增（2026-11-19，用户指出无运动内容）：
  - 输入含【平移滑动段】：静态 1.2s → 滑动 2.4s 交替；播放器记录相位时间线
  - A5 运动活性：滑动段内输出持续变化（>阈值）
  - A6 行为级 MC 验证：同一输入跑两遍（D2S_MC=1/0），滑动段内按时间戳
    对齐比对——两次输出必须有差（"MC 在跑却对输出无影响"式失效在此现形）

断言族（对应历史 bug 类别）：
  A1 无 [mc] 异常 / A2 [mc] 遥测在跑 / A3 静止段逐位稳定 / A4 静态期数符合
用法：D2S\\python3\\python.exe -I tools\\e2e_playback.py [--release Cubist-1.0.224]
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP = os.environ.get("TEMP", ".")
S1, SL, S2, CC = 1.2, 2.4, 1.6, 1.6
CYCLES = 3            # 三合一周期（静止→滑动→静止→硬切）×3
DUMP_HZ = "4"

PLAYER_SRC = r"""
import cv2, time, sys, numpy as np
imgs = [cv2.imread(p) for p in sys.argv[1:]]
sh, sw = 1080, 1920
ups = [cv2.resize(i, (sw, sh), interpolation=cv2.INTER_AREA) for i in imgs]
cv2.namedWindow("e2e", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("e2e", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
t0 = time.perf_counter()
log = open(sys.argv[0] + ".phases", "w")
ph = None
S1, SL, S2, C = 1.2, 2.4, 1.6, 1.6
cyc = S1 + SL + S2 + C
def emit(tag):
    global ph
    if ph != tag:
        ph = tag
        log.write("%.0f %s" % (time.time() * 1000, tag) + chr(10))
        log.flush()
while True:
    t = time.perf_counter() - t0
    n = int(t / cyc)
    k = (n * 2) % len(imgs)
    tt = t - n * cyc
    a = ups[k]
    b = ups[(k + 1) % len(ups)]
    c = ups[(k + 2) % len(ups)]
    if tt < S1:
        emit("s%d" % (n * 2))
        frame = a
    elif tt < S1 + SL:
        emit("m%d" % (n * 2))
        x = int((tt - S1) / SL * sw)
        frame = np.hstack([a, b])[:, x:x + sw]
    elif tt < S1 + SL + S2:
        emit("s%d" % (n * 2 + 1))
        frame = b
    else:
        emit("c%d" % (n * 2 + 1))
        frame = c
    cv2.imshow("e2e", frame)
    if cv2.waitKey(30) == 27:
        break
"""


def make_imgs():
    d = os.path.join(TEMP, "d2s-e2e-imgs")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    paths = []
    for kind in range(4):
        tex = rng.uniform(0.25, 0.85, (224, 224)).astype(np.float32)
        if kind == 0:
            m = ((xx - 112) ** 2 + (yy - 112) ** 2) <= 60 ** 2
            tex = np.where(m, np.clip(tex + 0.2, 0, 1), tex * 0.4)
        elif kind == 1:
            tex[((yy // 14) % 2 == 0)] *= 0.5
        elif kind == 2:
            tex[((xx // 14 + yy // 14) % 2 == 0)] = 0.85
        p = os.path.join(d, f"img{kind}.png")
        cv2.imwrite(p, (np.stack([tex] * 3, 2) * 255).astype(np.uint8))
        paths.append(p)
    return paths


def session(rel, mc_on, imgs, tag, render=None):
    app = os.path.join(rel, "app")
    pyexe = os.path.join(rel, "python", "python.exe")
    rt = os.path.join(rel, "runtime")
    dump = os.path.join(TEMP, f"d2s-e2e-{tag}")
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", dump], capture_output=True)
    os.makedirs(dump, exist_ok=True)
    player_py = os.path.join(dump, "player.py")
    with open(player_py, "w", encoding="utf-8") as f:
        f.write(PLAYER_SRC)
    env = os.environ.copy()
    env["PATH"] = rt + os.pathsep + env.get("PATH", "")
    player = subprocess.Popen([pyexe, "-u", player_py] + imgs, env=env,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    aenv = os.environ.copy()
    aenv["PATH"] = rt + os.pathsep + app + os.pathsep + aenv["PATH"]
    aenv["D2S_FRAME_DUMP"] = dump
    aenv["D2S_FRAME_DUMP_HZ"] = DUMP_HZ
    aenv["D2S_NO_POPUP"] = "1"
    aenv["D2S_MC"] = "1" if mc_on else "0"
    if render:
        aenv["D2S_RENDER"] = render
    alog = open(os.path.join(dump, "app_stdout.log"), "w",
                encoding="utf-8", errors="replace")
    appproc = subprocess.Popen([pyexe, "-u", "-m", "main"], cwd=app, env=aenv,
                               stdout=alog, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    time.sleep(10)
    with open(os.path.join(TEMP, "d2s-3d-mode"), "w") as f:
        f.write("3d")
    total = CYCLES * (S1 + SL + S2 + CC) + 6
    time.sleep(total)
    for p in (appproc, player):
        subprocess.run(["taskkill", "/F", "/PID", str(p.pid), "/T"],
                       capture_output=True)
    alog.close()
    time.sleep(1)
    return dump


def load_run(dump):
    frames = []
    for f in sorted(os.listdir(dump)):
        if f.startswith("frame_") and f.endswith(".png"):
            ts = int(f[6:-4])
            a = cv2.imread(os.path.join(dump, f))
            if a is not None:
                frames.append((ts, a))
    log = open(os.path.join(dump, "app_stdout.log"), encoding="utf-8",
               errors="replace").read()
    phases = []
    pl = os.path.join(dump, "player.py.phases")
    if os.path.exists(pl):
        for ln in open(pl):
            parts = ln.split()
            if len(parts) == 2:
                phases.append((int(parts[0]), parts[1]))
    return frames, log, phases


def main(release_name, d3d=False):
    rel = os.path.join(ROOT, "release", release_name)
    imgs = make_imgs()
    if d3d:
        print("[e2e] D3D 单会话（真实发布包，静态+滑动+硬切输入）…")
        dump1 = session(rel, False, imgs, "d3d", render="d3d")
        dump0 = None
    else:
        print("[e2e] 会话1/2：MC=ON（真实发布包，静态+滑动输入）…")
        dump1 = session(rel, True, imgs, "mc1")
        print("[e2e] 会话2/2：MC=OFF（同一输入，行为级比对基准）…")
        dump0 = session(rel, False, imgs, "mc0")

    f1, log1, ph1 = load_run(dump1)
    f0, _log0, _ph0 = load_run(dump0) if dump0 else ([], "", [])
    fails = []

    def A(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    if d3d:
        import re as _re
        _cap_errs = _re.findall(r"\[capture\] D3D 环初始化失败[^\n]*", log1)
        _tb = log1.count("Traceback")
        print(f"\n[e2e] D3D: {len(f1)} 帧 | 相位 {len(ph1)} 条 | "
              f"环失败 {len(_cap_errs)} | Traceback {_tb}\n")
        A("A1 无异常（Traceback/采集失败）", not _tb and not _cap_errs)
    else:
        print(f"\n[e2e] MC=1: {len(f1)} 帧 | MC=0: {len(f0)} 帧 | 相位 {len(ph1)} 条\n")
        A("A1 无 [mc] 异常", "[mc] 步失败" not in log1)
        # 注意别用 '"D2S_MC=0" not in log' 判 OFF——ON 会话的横幅文案里也含这串
        A("A2 MC 遥测在跑", "[mc] gpu" in log1)

    def phase_of(ts, phases):
        s0t = next((t for t, n in phases if n.startswith("s")), None)
        if s0t is None:
            return None
        PER = int((S1 + SL + S2 + CC) * 1000)
        rel = (ts - s0t) % PER
        E = 350   # 边缘带（漂移+延迟）
        if rel < E:
            return "s-edge"
        if rel < S1 * 1000:
            return "s"
        if rel < (S1 + SL) * 1000:
            return "m-edge" if rel < (S1 + E / 1000) * 1000 else "m"
        if rel < (S1 + SL + E / 1000) * 1000:
            return "s-edge"
        if rel < (S1 + SL + S2) * 1000:
            return "s"
        if rel < (S1 + SL + S2 + E / 1000) * 1000:
            return "c-edge"
        return "c"

    static_pairs, motion_pairs = [], []
    for i in range(len(f1) - 1):
        # 相邻两帧必须同相位才计入（段界跨越对会把滑动尾帧算进静态，
        # 伪差可达 50+，v1 首跑实锤）
        p0 = phase_of(f1[i][0], ph1)
        p1_ = phase_of(f1[i + 1][0], ph1)
        if not p0 or p0 != p1_:
            continue
        d = float(np.mean(cv2.absdiff(f1[i][1], f1[i + 1][1])))
        if p0 == "s":
            static_pairs.append(d)
        elif p0 == "m":
            motion_pairs.append(d)
    if static_pairs:
        A("A3 静止段稳定", max(static_pairs) < 1.5,
          f"(n={len(static_pairs)} 最大相邻差={max(static_pairs):.2f})")
    else:
        A("A3 静止段稳定", False, "(无静态相位样本)")
    if motion_pairs:
        A("A5 运动段活性", float(np.median(motion_pairs)) > 2.0,
          f"(n={len(motion_pairs)} 中位相邻差={np.median(motion_pairs):.2f})")
    else:
        A("A5 运动段活性", False, "(无运动相位样本)")
    toggles = sum(1 for _, p in ph1 if p.startswith("s"))
    A("A4 静态期数符合", CYCLES * 2 <= toggles <= 3 * CYCLES * 2,
      f"(s 相位={toggles} 三合一周期每轮2个s)")

    # A6：按【会话内相对时间】对齐（两会话先后运行，绝对时间戳天然不相交
    # ——v1 直接比绝对 ts 对齐 0 帧实锤）
    t0_1 = f1[0][0] if f1 else 0
    t0_0 = f0[0][0] if f0 else 0
    diffs, n_align = [], 0
    for ts, a in f1:
        p = phase_of(ts, ph1)
        if not (p and (p == "m" or p == "m-edge")):
            continue
        rel = ts - t0_1
        best = min(f0, key=lambda x: abs((x[0] - t0_0) - rel), default=None)
        if best and abs((best[0] - t0_0) - rel) < 400:
            diffs.append(float(np.mean(cv2.absdiff(a, best[1]))))
            n_align += 1
    if n_align >= 4 and not d3d:
        A("A6 MC 行为级差异", float(np.median(diffs)) > 0.5,
          f"(对齐 {n_align} 帧 中位差={np.median(diffs):.2f})")
    elif not d3d:
        A("A6 MC 行为级差异", False, f"(对齐仅 {n_align} 帧)")

    # A7 硬切换响应：c-edge 帧与前帧差应远大于静止基线
    cut_diffs = []
    for i in range(len(f1)):
        if phase_of(f1[i][0], ph1) == "c-edge":
            if i > 0:
                cut_diffs.append(float(np.mean(
                    cv2.absdiff(f1[i - 1][1], f1[i][1]))))
    if cut_diffs:
        A("A7 硬切换跳变", float(np.median(cut_diffs)) > 10.0,
          f"(n={len(cut_diffs)} 中位跳变={np.median(cut_diffs):.1f})")
    else:
        A("A7 硬切换跳变", False, "(无 c-edge 采样)")

    if fails:
        print(f"\n[e2e] FAIL: {fails}")
        return 1
    print("\n[e2e] ALL PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="Cubist-1.0.224")
    ap.add_argument("--d3d", action="store_true",
                    help="D3D 包：单会话 + A1/A3/A4/A5/A7（无 MC 断言）")
    a = ap.parse_args()
    sys.exit(main(a.release, d3d=a.d3d))
