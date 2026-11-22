# -*- coding: utf-8 -*-
"""pythonw 启动时 stdout/stderr 为 None 的兜底（单源）。

print 到 None 会抛 AttributeError 直接崩，用户看不到任何报错（双击
启动"毫无反应"的真因）。utils 在导入期就会 print，所以 main /
settings_dialog 都必须在 `import utils` **之前**完成兜底。

零依赖（不 import 任何项目模块），导入无副作用。
"""
import os
import sys


class NullStream:
    def write(self, *_a):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


_STALE_S = 7 * 24 * 3600    # 超过 7 天没动过的日志视为陈旧，启动时清掉


def _cleanup_old_logs(ld):
    """启动清理：app.log / app.log.1 / tear_*.png 超过 7 天没更新就删。

    语义是"7 天都没启动过就从头开始"：日常使用每次启动都会刷新 mtime，
    不会误删。诊断转储（tear_*.png）同样按这个规则收。
    """
    import time
    now = time.time()
    try:
        names = os.listdir(ld)
    except OSError:
        return
    for n in names:
        if not (n in ("app.log", "app.log.1")
                or (n.startswith("tear_") and n.endswith(".png"))):
            continue
        p = os.path.join(ld, n)
        try:
            if now - os.path.getmtime(p) > _STALE_S:
                os.remove(p)
        except OSError:
            pass


def ensure(header=None):
    """stdout/stderr 任一为 None 时兜底到 <本目录>/logs/app.log（或 NullStream）。
    header 非空时写一行启动分隔（幂等：已接管过则只补 header 日志行）。
    启动时顺带做日志维护：>7 天的清理、>4MB 的滚动。"""
    _ld = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    try:
        os.makedirs(_ld, exist_ok=True)
        _cleanup_old_logs(_ld)
    except Exception:
        pass
    if sys.stdout is not None and sys.stderr is not None:
        if header:
            print(header, flush=True)
        return
    try:
        _lf_path = os.path.join(_ld, "app.log")
        # 发布版日志上限：启动时超过 4MB 就滚动一次（app.log → app.log.1，
        # 旧 .1 被覆盖）。长时间 3D 运行不会无限增长；最多留 ~8MB。
        try:
            if os.path.getsize(_lf_path) > 4 * 1024 * 1024:
                os.replace(_lf_path, _lf_path + ".1")
        except OSError:
            pass
        _lf = open(_lf_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        _lf = None
    _sink = _lf if _lf is not None else NullStream()
    sys.stdout = sys.stdout or _sink
    sys.stderr = sys.stderr or _sink
    if header:
        print(header, flush=True)
