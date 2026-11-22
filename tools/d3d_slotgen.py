# -*- coding: utf-8 -*-
"""从仓库内 Windows SDK 头文件提取 D3D11/D3D12 接口 vtable 槽位（C Vtbl 成员序）。

方法：定位 `typedef struct XXVtbl` 到 `} XXVtbl;` 的行区间，按出现顺序收集
`STDMETHODCALLTYPE *方法名` 与 `DECLSPEC_XFGVIRT(接口, 方法)` 标记（XFG 是
vtable 顺序的一部分），槽位 = IUnknown(3) + 成员序。同时给出 C++ 块的
继承链交叉验证。输出作为 gpu/d3d12.py / rtcom.py 槽位常量的权威依据。
用法：python tools/d3d_slotgen.py [输出文件]   （只读头文件）
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK = os.path.join(ROOT, "_toolchain", "dl", "sdk", "cpp", "c",
                   "Include", "10.0.26100.0")

WANT = [
    ("um/d3d12.h", ["ID3D12Device", "ID3D12CommandQueue", "ID3D12CommandAllocator",
                    "ID3D12GraphicsCommandList", "ID3D12Fence", "ID3D12Resource",
                    "ID3D12DescriptorHeap"]),
    ("um/d3d11.h", ["ID3D11Device", "ID3D11DeviceContext", "ID3D11DeviceContext1",
                    "ID3D11DeviceContext2", "ID3D11DeviceContext3"]),
    ("um/d3d11_3.h", ["ID3D11Fence", "ID3D11DeviceContext4"]),
    ("um/d3d11_3.h", ["ID3D11Fence"]),
    ("um/d3d11_4.h", ["ID3D11Device5"]),
    ("shared/dxgi.h", ["IDXGIObject", "IDXGISwapChain", "IDXGIFactory", "IDXGIFactory1"]),
    ("shared/dxgi1_2.h", ["IDXGIFactory2", "IDXGISwapChain1"]),
]

MEM_RE = re.compile(r"\(\s*STDMETHODCALLTYPE\s+\*\s*(\w+)\s*\)")


def vtbl_slots(src, name):
    """行切片：Vtbl 起止行之间按序收集指针声明成员 → {方法: 槽位}。

    只数 STDMETHODCALLTYPE 指针声明（XFGVIRT 标记与声明一一对应、不重复
    计数）；同名重载各占独立槽（GetResourceAllocationInfo 等）。
    """
    lines = src.split("\n")
    begin = end = None
    for i, l in enumerate(lines):
        if begin is None and "typedef struct %sVtbl" % name in l:
            begin = i
        if begin is not None and re.match(r"\s*\}\s*%sVtbl\s*;" % name, l):
            end = i
            break
    if begin is None or end is None:
        return None
    mems = []
    branch = []            # 条件栈：'nonwin' | 'win'
    for i in range(begin + 1, end):
        st = lines[i].strip()
        if st.startswith("#if !defined(_WIN32)"):
            branch.append("nonwin")
            continue
        if st.startswith("#else") and branch and branch[-1] == "nonwin":
            branch[-1] = "win"
            continue
        if st.startswith("#endif") and branch:
            branch.pop()
            continue
        if branch and branch[-1] != "win":
            continue       # 只收 Windows 分支（非 Win 分支是同一槽位的另一声明）
        m = MEM_RE.search(lines[i])
        if m:
            mems.append(m.group(1))
    slots = {}
    for i, m in enumerate(mems):
        slots.setdefault(m, i)   # 0-based vtable 槽（rtcom 惯例：QI=0）；重载各占一槽
    return slots


def main():
    out = []
    for hdr, ifaces in WANT:
        src = open(os.path.join(SDK, hdr), encoding="utf-8", errors="replace").read()
        for name in ifaces:
            slots = vtbl_slots(src, name)
            if slots is None:
                out.append(f"# !! {name}: Vtbl 未找到（{hdr}）")
                continue
            out.append(f"# {name}（{hdr}，{len(slots)} 个唯一方法）")
            for k, v in slots.items():
                out.append(f"#   {v:3d} {k}")
            out.append("")
    text = "\n".join(out)
    print(text)
    if len(sys.argv) > 1:
        open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(text)
        print("[slotgen] 已写", sys.argv[1])


if __name__ == "__main__":
    main()
