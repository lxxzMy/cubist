# -*- coding: utf-8 -*-
"""用户可见提示（pythonw 没有控制台，日志客户看不到）。

用法：
    import notify
    notify.warn("3D 加速不可用", "……", once_key="gpu-unsupported")

行为：
    1. 把提示写到 %TEMP%\\d2s-last-warning.txt（支持远程取证）；
    2. 后台线程弹一个置顶 MessageBox（非阻塞，不打断渲染）。
所有异常吞掉 —— 提示失败绝不能影响主流程。
"""
import os
import threading

_seen = set()


def warn(title, text, once_key=None):
    """显示一次性提示；once_key 相同则本次会话只弹一次。
    D2S_NO_POPUP=1 时只写取证文件不弹窗（冒烟/自动化引导用）。"""
    _no_box = os.environ.get("D2S_NO_POPUP") == "1"
    try:
        if once_key is not None:
            if once_key in _seen:
                return False
            _seen.add(once_key)
        p = os.path.join(os.environ.get("TEMP", "."), "d2s-last-warning.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{text}\n")
    except Exception:
        pass
    if _no_box:
        return True
    try:
        threading.Thread(target=_box, args=(title, text), daemon=True).start()
        return True
    except Exception:
        return False


def _box(title, text):
    try:
        import ctypes
        MB_ICONWARNING = 0x00000030
        MB_TOPMOST = 0x00040000
        MB_SETFOREGROUND = 0x00010000
        ctypes.windll.user32.MessageBoxW(
            None, text, title, MB_ICONWARNING | MB_TOPMOST | MB_SETFOREGROUND)
    except Exception:
        pass
