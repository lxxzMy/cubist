# -*- coding: utf-8 -*-
"""把零拷贝 shim 的产物装进发布包（P4）。

做三件事：
  1. runtime/d2s_dml_shim.dll  → <release>/runtime/
  2. D2S/ 全部第一方 .py 重编到 <release>/app/（全量枚举，2026-11-18
     起不再手工列清单——清单曾漏 6 模块致发布包新旧混编 import 崩）
  3. 校验：DLL 在、pyc 比源码新、打印清单

用法：
    python tools/package_shim.py            # 装进所有 release/*/
    python tools/package_shim.py --dry-run  # 只报告
"""
import argparse
import glob
import hashlib
import os
import py_compile
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2S = os.path.join(ROOT, "D2S")
DLL = os.path.join(ROOT, "runtime", "d2s_dml_shim.dll")
# D3D12 生产后端 C 助手（1.1-d3d 起主路径必需）：构建产物在 _toolchain/src
_d3dr_cands = (os.path.join(ROOT, "runtime", "d3drender.dll"),
               os.path.join(ROOT, "_toolchain", "src", "d3drender.dll"))
D3DRENDER = next((p for p in _d3dr_cands if os.path.isfile(p)), _d3dr_cands[-1])
# DXC/SM6 着色器编译运行时（D2 迁移）：随包分发到 <release>/runtime/
EXTRA_DLLS = ("dxcompiler.dll", "dxil.dll")

# 2026-11-18 定案：手工清单曾漏 6 个模块（capture/depth/tray/cursor_hide/
# d2s_ops/notify + gpu/rtcom 等），造成"新 dxgi_gpu_grab 导入旧 rtcom 缺
# HRESULT 常量→发布包 import 即崩"。改为全量枚举 D2S 第一方 .py，
# 并清理 app/ 里没有对应源码的孤儿 pyc（如已删除的 wgc_winrt/dx_iids）。
def first_party_modules():
    """(源码相对D2S, 发布包内pyc路径, 标签) 全量枚举，不含手工清单。"""
    out = []
    for sub, prefix in (("", ""), ("gpu", "gpu/")):
        d = os.path.join(D2S, sub) if sub else D2S
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                rel = prefix + f[:-3] + ".pyc"
                out.append((f"{sub}/{f}" if sub else f, rel, rel))
    return out


MODULES = first_party_modules()


def _md5(p):
    try:
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def releases():
    out = []
    for d in glob.glob(os.path.join(ROOT, "release", "*")):
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "app")):
            out.append(d)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="只处理该名字的发布目录（精确匹配 basename）")
    args = ap.parse_args()

    if not os.path.isfile(DLL):
        print(f"[x] 找不到 {DLL}（先在 _toolchain/src 里跑 build_shim.sh）")
        return 2

    targets = [r for r in releases() if not args.only or args.only == os.path.basename(r)]
    if not targets:
        print("[x] 没找到 release/*/app 目录")
        return 2

    rc = 0
    for rel in targets:
        name = os.path.basename(rel)
        print(f"\n== {name} ==")
        dst_dll = os.path.join(rel, "runtime", "d2s_dml_shim.dll")
        if args.dry_run:
            print(f"  [dry] {DLL} -> {dst_dll}")
        elif os.path.isfile(dst_dll) and _md5(dst_dll) == _md5(DLL):
            print(f"  [=] DLL 已是最新（md5 相同）")
        else:
            try:
                os.makedirs(os.path.dirname(dst_dll), exist_ok=True)
                shutil.copy2(DLL, dst_dll)
                print(f"  [ok] DLL -> {os.path.relpath(dst_dll, rel)} "
                      f"({os.path.getsize(dst_dll)//1024}KB)")
            except PermissionError:
                print(f"  [x] DLL 被占用：{os.path.relpath(dst_dll, rel)}")
                print("      发布包里的 app 正在运行，先退出（或结束 pythonw.exe）再打包")
                rc = 1

        # D3D12 生产后端 C 助手（1.1-d3d 主路径）
        d3dr_dst = os.path.join(rel, "runtime", "d3drender.dll")
        if args.dry_run:
            print(f"  [dry] {D3DRENDER} -> {d3dr_dst}")
        elif not os.path.isfile(D3DRENDER):
            print(f"  [x] 缺 d3drender.dll（先在 _toolchain/src 跑 "
                  f"build_d3drender.cmd）")
            rc = 1
        elif os.path.isfile(d3dr_dst) and _md5(d3dr_dst) == _md5(D3DRENDER):
            print("  [=] d3drender.dll 已是最新")
        else:
            try:
                os.makedirs(os.path.dirname(d3dr_dst), exist_ok=True)
                shutil.copy2(D3DRENDER, d3dr_dst)
                print(f"  [ok] d3drender.dll -> "
                      f"{os.path.relpath(d3dr_dst, rel)} "
                      f"({os.path.getsize(d3dr_dst)//1024}KB)")
            except PermissionError:
                print("  [x] d3drender.dll 被占用（发布包 app 在跑？）")
                rc = 1

        # DXC/SM6 编译运行时（dxcompiler.dll+dxil.dll；缺 dxil.dll 时
        # 未签名 DXIL 会被 D3D12 拒收——必须随包）
        for dll_name in EXTRA_DLLS:
            src_dll = os.path.join(ROOT, "runtime", dll_name)
            dst = os.path.join(rel, "runtime", dll_name)
            if not os.path.isfile(src_dll):
                print(f"  [x] 缺 {dll_name}（先拷进 runtime/）")
                rc = 1
                continue
            if args.dry_run:
                print(f"  [dry] {src_dll} -> {dst}")
            elif os.path.isfile(dst) and _md5(dst) == _md5(src_dll):
                print(f"  [=] {dll_name} 已是最新")
            else:
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src_dll, dst)
                    print(f"  [ok] {dll_name} -> {os.path.relpath(dst, rel)} "
                          f"({os.path.getsize(dst)//1024}KB)")
                except PermissionError:
                    print(f"  [x] {dll_name} 被占用")
                    rc = 1

        for src_rel, rel_pyc, label in MODULES:
            src = os.path.join(D2S, src_rel.replace("/", os.sep))
            dst = os.path.join(rel, "app", rel_pyc.replace("/", os.sep))
            if not os.path.isfile(src):
                print(f"  [!] 源码缺失: {src}")
                rc = 1
                continue
            if args.dry_run:
                print(f"  [dry] {src} -> {dst}")
                continue
            try:
                py_compile.compile(src, cfile=dst, doraise=True)
                print(f"  [ok] {label}  ({time.strftime('%H:%M:%S')})")
            except Exception as e:
                print(f"  [x] {label} 编译失败: {e}")
                rc = 1

        # 校验 + 孤儿 pyc 清理（源码已删的模块残留 = 定时炸弹）
        checks = [("runtime/d2s_dml_shim.dll", dst_dll),
                  ("runtime/d3drender.dll", d3dr_dst)]
        for _s, rel_pyc, label in MODULES:
            checks.append((label, os.path.join(rel, "app", rel_pyc)))
        for label, p in checks:
            ok = os.path.isfile(p)
            print(f"  {'[v]' if ok else '[x]'} {label}")   # ASCII：GBK 控制台可打印
            if not ok:
                rc = 1
        wanted = {rel_pyc for _s, rel_pyc, _l in MODULES}   # 统一正斜杠比较
        for sub in ("", os.path.join("app", "gpu")):
            scan = os.path.join(rel, "app") if sub == "" else os.path.join(rel, sub)
            if not os.path.isdir(scan):
                continue
            for f in os.listdir(scan):
                base = os.path.basename(scan)
                relname = (f if base == "app" else "gpu/" + f)
                if f.endswith(".pyc") and relname not in wanted:
                    if args.dry_run:
                        print(f"  [dry] 删除孤儿 {relname}")
                    else:
                        os.remove(os.path.join(scan, f))
                        print(f"  [rm] 孤儿 {relname}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
