# -*- coding: utf-8 -*-
"""一键冒烟：全包编译 + 黄金基线门禁（重构后日常改动快速回归）。

用法（仓库根，D2S 自带解释器）：
  D2S\\python3\\python.exe -I tools\\smoke.py [--skip-smoke]

包含：
  1) py_compile D2S 全部第一方 .py
  2) ruff 保守规则（E9/F821/F811/E722，ruff.toml）
  3) tools/refactor_gate.py check（depth/render 逐位 + 启动冒烟）
"""
import os
import py_compile
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2S = os.path.join(ROOT, "D2S")
PY = sys.executable


def main():
    skip = "--skip-smoke" in sys.argv
    fails = []

    # 1) 编译
    for f in sorted(os.listdir(D2S)):
        if f.endswith(".py"):
            try:
                py_compile.compile(os.path.join(D2S, f), doraise=True)
            except py_compile.PyCompileError as e:
                fails.append(f"compile: {f}: {e}")
    print(f"[smoke] compile: {'OK' if not fails else 'FAIL'} ({len(fails)})")

    # 2) ruff
    r = subprocess.run([PY, "-m", "ruff", "check", "D2S/",
                        "--config", "ruff.toml", "--output-format", "concise"],
                       cwd=ROOT, capture_output=True, text=True)
    print(f"[smoke] ruff: {'OK' if r.returncode == 0 else 'FAIL'}")
    if r.returncode != 0:
        fails.append("ruff:\n" + r.stdout)

    # 3) MC-EMA GL 对拍（渲染内部覆盖——2026-11-18 产品化起常驻）
    g2 = subprocess.run([PY, "-I", os.path.join(ROOT, "tools", "mc_gl_test.py")],
                        cwd=ROOT)
    print(f"[smoke] mc-gl: {'OK' if g2.returncode == 0 else 'FAIL'}")
    if g2.returncode != 0:
        fails.append("mc-gl 对拍（见上）")

    # 4) 行为门禁
    cmd = [PY, "-I", os.path.join(ROOT, "tools", "refactor_gate.py"), "check"]
    if skip:
        cmd.append("--skip-smoke")
    g = subprocess.run(cmd, cwd=ROOT)
    print(f"[smoke] gate: {'OK' if g.returncode == 0 else 'FAIL'}")
    if g.returncode != 0:
        fails.append("gate (见上)")

    if fails:
        print("\n[smoke] FAIL:")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("[smoke] ALL PASS")


if __name__ == "__main__":
    main()
