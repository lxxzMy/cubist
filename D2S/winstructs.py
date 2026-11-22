# -*- coding: utf-8 -*-
"""Win32 ctypes 结构体单源（此前 CURSORINFO/RECT/MONITORINFO 在 4 处复制粘贴，
字段一致但类型对象各异）。

只放结构体定义，零导入副作用。特别地**不**设置任何 windll 函数的
argtypes/restype：windll 函数对象全进程共享，签名只允许在调用点附近
设置一次（见 capture.py / viewer.py 的 GetIconInfo 注释——两模块共用
同一个函数对象，重复设置会互相打架）。
"""
import ctypes
import ctypes.wintypes as wt


class CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("flags", wt.DWORD),
                ("hCursor", wt.HANDLE), ("ptScreenPos", wt.POINT)]


class ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wt.BOOL), ("xHotspot", wt.DWORD),
                ("yHotspot", wt.DWORD), ("hbmMask", wt.HANDLE),
                ("hbmColor", wt.HANDLE)]


class RECT(ctypes.Structure):
    """L/T/R/B 命名沿用量产代码（GetWindowRect/GetClientRect 系）。"""
    _fields_ = [("L", ctypes.c_long), ("T", ctypes.c_long),
                ("R", ctypes.c_long), ("B", ctypes.c_long)]


class POINTXY(ctypes.Structure):
    _fields_ = [("X", ctypes.c_long), ("Y", ctypes.c_long)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", wt.DWORD)]
