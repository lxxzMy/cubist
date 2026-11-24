# -*- coding: utf-8 -*-
"""构建便携发布包（当前默认 Cubist-1.5-d3d，纯 D3D12 主路径）。

流程：
  1. 准备包目录：默认**就地重建**（--dst 指定的包即基线）；
     也可 --src 从另一份包复制出新包（做新版本时才用）
  2. 剔除用户残留：settings.yaml.bak*、logs 内容
  3. 调 package_shim（全量重编 app/**/*.pyc + runtime DLL：shim v2 /
     d3drender.dll / dxcompiler+dxil），并清理孤儿 pyc
  4. 补资源：app/data/rootsig.blob（D3D12 模块的 blob 缓存）、app/settings.yaml.default
  5. 版本单源：NOTICE.txt 头
  6. csc 重编 立体派.exe
  7. 自检：pyc/DLL 清单、启动器、产品模型 + 署名/许可文件、settings 模板

归档前（不在本脚本里做，避免动到在用的包）：把 app/settings.yaml 换成
settings.yaml.default 的副本，免得把开发机调参带给用户。

用法：
    python tools/make_release_d3d.py            # 就地重建 1.5-d3d
    python tools/make_release_d3d.py --check    # 只自检现有包
    python tools/make_release_d3d.py --src release/Cubist-1.5-d3d \
        --dst release/Cubist-1.6-d3d --version "立体派 Cubist 1.6-d3d"
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = None                    # None = 就地（以 DST 为基线）
DST = os.path.join(ROOT, "release", "Cubist-1.5-d3d")
D2S = os.path.join(ROOT, "D2S")
VERSION_LINE = "立体派 Cubist 1.5-d3d"

_SKIP_BAK = ("settings.yaml.bak",)


def _log(msg):
    print(msg, flush=True)


def prepare_tree():
    if SRC is None or os.path.normcase(os.path.abspath(SRC)) == \
            os.path.normcase(os.path.abspath(DST)):
        if not os.path.isdir(DST):
            raise SystemExit(f"[x] 包不存在（就地重建需要先有一份基线）: {DST}")
        _log(f"[1] 就地重建 {os.path.relpath(DST, ROOT)}")
        return
    if not os.path.isdir(SRC):
        raise SystemExit(f"[x] 源包不存在: {SRC}")
    if os.path.isdir(DST):
        _log(f"[1] 清理旧包 {os.path.relpath(DST, ROOT)}")
        shutil.rmtree(DST)
    _log(f"[1] 复制 {os.path.relpath(SRC, ROOT)} → "
         f"{os.path.relpath(DST, ROOT)}")
    shutil.copytree(SRC, DST)


def clean_user_leftovers():
    """剔除用户残留（备份 settings/logs/临时文件），保留结构。"""
    app = os.path.join(DST, "app")
    n = 0
    for f in glob.glob(os.path.join(app, "settings.yaml.bak*")):
        os.remove(f)
        n += 1
    logs = os.path.join(app, "logs")
    if os.path.isdir(logs):
        for f in os.listdir(logs):
            p = os.path.join(logs, f)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)      # dxc-cache 等子目录
                else:
                    os.remove(p)
                n += 1
            except OSError:
                pass
    else:
        os.makedirs(logs, exist_ok=True)
    # 发布包里的 pid 残留会让 launcher 误判"已在运行"
    _log(f"[2] 清理用户残留 {n} 项")


def run_package_shim():
    _log("[3] package_shim：重编 pyc + 同步 runtime DLL")
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                      "package_shim.py"),
                         "--only", os.path.basename(DST)]).returncode
    if rc != 0:
        raise SystemExit(f"[x] package_shim 失败 rc={rc}")


def add_resources():
    app = os.path.join(DST, "app")
    pairs = [
        (os.path.join(D2S, "data", "rootsig.blob"),
         os.path.join(app, "data", "rootsig.blob")),
        # 设置模板跟着源码走（不碰 app/settings.yaml：那是包的使用者的配置）
        (os.path.join(D2S, "settings.yaml.default"),
         os.path.join(app, "settings.yaml.default")),
    ]
    for src, dst in pairs:
        if not os.path.isfile(src):
            raise SystemExit(f"[x] 缺资源: {src}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    _log("[4] 资源补齐 data/rootsig.blob + settings.yaml.default")


def stamp_version():
    notice = os.path.join(DST, "NOTICE.txt")
    if os.path.isfile(notice):
        with open(notice, encoding="utf-8") as f:
            txt = f.read()
        lines = txt.splitlines()
        if lines:
            lines[0] = (f"{VERSION_LINE} — 第三方组件声明" if "第三方" in
                        lines[0] else VERSION_LINE)
        with open(notice, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    _log(f"[5] 版本标记 → {VERSION_LINE}")


def build_launcher():
    """用 csc 直接重编 立体派.exe（不经 build-launcher.ps1：该脚本无 BOM，
    Windows PowerShell 5.1 按 ANSI 解析中文会炸）。"""
    csc = os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                       "Microsoft.NET", "Framework64", "v4.0.30319", "csc.exe")
    if not os.path.isfile(csc):
        raise SystemExit(f"[x] 缺 csc.exe: {csc}")
    src = os.path.join(ROOT, "tools", "launcher.cs")
    icon = os.path.join(DST, "app", "icon.ico")
    out = os.path.join(DST, "立体派.exe")
    if not os.path.isfile(src):
        raise SystemExit(f"[x] 缺 {src}")
    cmd = [csc, "/nologo", "/target:winexe", f"/out:{out}",
           "/r:System.Windows.Forms.dll"]
    if os.path.isfile(icon):
        cmd.append(f"/win32icon:{icon}")
    cmd.append(src)
    _log("[6] 重编启动器 立体派.exe")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"[x] csc 失败 rc={rc}")


def self_check():
    ok = True
    app = os.path.join(DST, "app")
    pycs = glob.glob(os.path.join(app, "*.pyc")) + \
        glob.glob(os.path.join(app, "gpu", "*.pyc"))
    src_py = glob.glob(os.path.join(D2S, "*.py")) + \
        glob.glob(os.path.join(D2S, "gpu", "*.py"))
    _log(f"[7] pyc {len(pycs)} / 源码 {len(src_py)}")
    if len(pycs) < len(src_py):
        _log("  [x] pyc 数量少于源码（有模块没编进去）")
        ok = False
    for rel in ("runtime/d2s_dml_shim.dll", "runtime/d3drender.dll",
                "runtime/dxcompiler.dll", "runtime/dxil.dll",
                "python/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
                "python/Lib/site-packages/onnxruntime/capi/DirectML.dll",
                "app/icon.ico", "app/data/rootsig.blob",
                "app/settings_dialog.pyc", "app/gpu/d3d_window.pyc",
                "app/gpu/d3d_stats.pyc", "app/gpu/hlsl_stats.pyc",
                # 发布包只带产品模型（model_catalog.PRODUCT_MODEL）及其署名/许可
                "models/zipdepth_wide_512x288.onnx", "models/README.md",
                "models/LICENSE-ZipDepth-MIT.txt", "立体派.exe"):
        p = os.path.join(DST, rel.replace("/", os.sep))
        e = os.path.exists(p)
        _log(f"  {'[v]' if e else '[x]'} {rel}")
        ok = ok and e
    if not ok:
        raise SystemExit("[x] 自检失败")
    _log("[7] 自检 OK")


def main():
    global SRC, DST, VERSION_LINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检现有包")
    ap.add_argument("--src", default=None,
                    help="源包目录（默认省略=就地重建 --dst）")
    ap.add_argument("--dst", default=None,
                    help="目标包目录（默认 1.5-d3d）")
    ap.add_argument("--version", default=None, help="版本行（NOTICE 首行）")
    a = ap.parse_args()
    if a.src:
        SRC = os.path.abspath(a.src)
    if a.dst:
        DST = os.path.abspath(a.dst)
    if a.version:
        VERSION_LINE = a.version
    if not a.check:
        prepare_tree()
        clean_user_leftovers()
        run_package_shim()
        add_resources()
        stamp_version()
        build_launcher()
    if not os.path.isdir(DST):
        raise SystemExit("[x] 包不存在")
    self_check()
    _log(f"\n[make-release] 完成: {os.path.relpath(DST, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
