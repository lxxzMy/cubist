# -*- coding: utf-8 -*-
"""2D/3D 模式文件协议（%TEMP%\\d2s-3d-mode）单源。

历史：此协议曾在 8 处复制粘贴，各自带私有的 150ms 节流时间戳。150ms 是
当年的折中（逐拍 open+read 在动画场景吃掉渲染线程 ~55% CPU，py-spy 定案）。
2026-11-22 改为 50ms：现在只有 4 个调用点、单站点 20 次读/秒，开销可忽略；
外部模式变更（托盘/设置）响应 50ms。热键 Pause 是事件驱动，不受此影响。

零导入副作用（settings_dialog 在重定向 stdout 之前也允许导入本模块）。
"""
import os
import tempfile

MODE_FILE = os.path.join(tempfile.gettempdir(), "d2s-3d-mode")
MODE_READ_THROTTLE_S = 0.05

# 各调用点的私有节流窗（key → [last_read_t, cached]）。保持站点间独立，
# 与重构前逐点时间戳的语义逐位一致。
# 2026-11-22：150ms → 50ms（外部模式变更响应；实测单站点 20 次读/秒，
# 开销可忽略。150ms 是当年"逐帧读 × 8 站点"吃 55% CPU 的折中值）
_throttle = {}


def write_mode(want3d: bool):
    with open(MODE_FILE, "w") as f:
        f.write("3d" if want3d else "2d")


def read_mode(default: bool = False) -> bool:
    """直读（True=3d）。IO 异常按 default（各站点历史默认不同：
    proc/vis3d/cursor 按 2d；渲染循环两个深度门按 3d —— 保真传递）。"""
    try:
        with open(MODE_FILE) as f:
            return f.read().strip() == "3d"
    except OSError:
        return default


def read_mode_throttled(now: float, key: str, default: bool = False) -> bool:
    """150ms 节流读：每个调用点用自己的 key（窗口独立，语义同原私有时间戳）。
    now 用调用方已有的时钟（time.perf_counter），避免为读模式单开系统调用。"""
    st = _throttle.setdefault(key, [0.0, None])
    if st[1] is None or now - st[0] > MODE_READ_THROTTLE_S:
        st[0] = now
        st[1] = read_mode(default)
    return st[1]
