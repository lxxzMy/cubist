# -*- coding: utf-8 -*-
"""显卡布局块（P4 收口，2026-11-20）：GL / 采集 / 推理各落在哪块卡的唯一入口。

adapter.py 仍是枚举/匹配/建设备的内部实现；本模块收口全部调用方（viewer
GL 登记、depth 的 DML device_id、dml_gl 的零拷贝 LUID、dxgi_* 的采集设备），
并提供一行式布局报告（boot 与推理 runner 就绪各打一次，后者补全采集/推理）。

跨卡 = wglDXOpenDeviceNV 必失败 = 有 SBS 但视差 0（退化 2D）——布局报告
的 mismatch 告警即为此守。选卡逻辑零改动（仍 adapter.pick 的
"GL 同卡 → 屏所在卡 → 显存最大"），本模块只做入口与可见性。
"""
from . import adapter as _ad

import ctypes as _ct

_layout = {"gl": None, "inference": None}


# 模块级（不能每调一次新建）：ctypes 的 Structure 子类/CFUNCTYPE 类型每次
# 定义都是一个新类型，类型↔StgDict↔字段描述符成环 → 循环垃圾。
class _RECT(_ct.Structure):
    _fields_ = [("left", _ct.c_long), ("top", _ct.c_long),
                ("right", _ct.c_long), ("bottom", _ct.c_long)]


_MONPROC = _ct.WINFUNCTYPE(_ct.c_int, _ct.c_void_p, _ct.c_void_p,
                           _ct.POINTER(_RECT), _ct.c_void_p)
_U32 = _ct.windll.user32


def monitor_rects():
    """活动显示器矩形 [(l, t, r, b)]，按 (left, top) 排序。

    这是 MONITOR_INDEX 的唯一口径：设置面板列表、覆盖层定位、DDA 输出
    选择全部按这个顺序（win32api/mss/DXGI 三套枚举顺序互不一致——本机
    双屏实测 win32api/mss 的第 1 个都是右侧屏，而主屏在另一块卡上；
    直接用各自序号会张冠李戴）。
    """
    out = []

    @_MONPROC
    def _cb(hmon, hdc, lprc, data):
        r = lprc.contents
        out.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return 1

    try:
        _U32.EnumDisplayMonitors(None, None, _cb, 0)
    except Exception:
        return []
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def monitor_rect(index):
    """1-based MONITOR_INDEX → 显示器矩形；越界回退第一块（无显示器→None）。"""
    ms = monitor_rects()
    if not ms:
        return None
    try:
        i = int(index or 1)
    except Exception:
        i = 1
    if 1 <= i <= len(ms):
        return ms[i - 1]
    return ms[0]


def monitor_adapter(index, refresh=False):
    """1-based 显示器序号 → 该屏所在 adapter（None=枚举失败/找不到）。

    与 pick_for_monitor 同一套包含判断（矩形中心落在哪个 adapter 的输出
    矩形里），但**不建 D3D 设备**——供"采集屏/输出屏是否同卡"这类纯判定
    使用。refresh=True 用于显示器插拔后刷新枚举缓存。
    """
    rect = monitor_rect(index)
    if rect is None:
        return None
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    try:
        ads = _ad.enum_adapters(refresh=refresh) if refresh \
            else _ad.enum_adapters()
    except TypeError:          # 旧签名无 refresh 参数
        ads = _ad.enum_adapters()
    except Exception:
        return None
    for a in ads:
        try:
            for rc in a.rects():
                if rc[0] <= cx < rc[2] and rc[1] <= cy < rc[3]:
                    return a
        except Exception:
            continue
    return None


def monitors_same_adapter(index, refresh=False):
    """与该屏同卡的所有显示器序号（1-based，含自己）。

    设置面板用它做"采集屏/输出屏"两个下拉的联动过滤；枚举失败或无
    显示器时回退 [index]（不过滤，宁可让用户选错也不把列表清空）。
    """
    a = monitor_adapter(index, refresh=refresh)
    try:
        i0 = int(index or 1)
    except Exception:
        i0 = 1
    if a is None:
        return [i0]
    out = []
    for k, rect in enumerate(monitor_rects(), 1):
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        try:
            for rc in a.rects():
                if rc[0] <= cx < rc[2] and rc[1] <= cy < rc[3]:
                    out.append(k)
                    break
        except Exception:
            break
    return out or [i0]


def output_index_for(index, adapter):
    """1-based 显示器序号 → 该 adapter 内输出序号（None=不在此卡）。

    DDA 采集要"该卡内的输出序号"；AB 分离时采集屏与输出屏可能不同，
    但必须同卡（零拷贝）——本函数即"采集屏是否在这块卡上"的判据。
    """
    rect = monitor_rect(index)
    if rect is None or adapter is None:
        return None
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    try:
        for k, rc in enumerate(adapter.rects()):
            if rc[0] <= cx < rc[2] and rc[1] <= cy < rc[3]:
                return k
    except Exception:
        return None
    return None


def pick_for_monitor(index, refresh=False):
    """(AdapterInfo, 该卡内输出序号, rect)——采集/渲染必须落在屏所在卡。

    双屏分属两块 GPU 时（本机实测：主屏 NVIDIA、副屏 AMD 核显），仅靠
    输出序号会在错误的卡上找输出 → DuplicateOutput 必失败。这里按矩形
    中心匹配 adapter。refresh=True 用于显示器插拔后刷新枚举缓存。
    """
    rect = monitor_rect(index)
    a = _ad.pick(monitor_rect=rect, refresh=refresh) if rect is not None \
        else _ad.pick(refresh=refresh)
    out_idx = 0
    if a is not None and rect is not None:
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        for k, rc in enumerate(a.rects()):
            if rc[0] <= cx < rc[2] and rc[1] <= cy < rc[3]:
                out_idx = k
                break
    return a, out_idx, rect


def gl_device(renderer):
    """viewer 建完 GL 上下文后登记（glGetString(GL_RENDERER)）。"""
    _ad.set_gl_renderer(renderer)
    _layout["gl"] = (renderer or "").strip() or None
    return renderer


def pick_adapter():
    """唯一选卡口径（GL 同卡 → 屏所在卡 → 显存最大；None=用默认）。"""
    return _ad.pick()


def dml_device_id():
    """推理侧（DirectML EP）的 device_id——多卡机上固定到与 GL 同卡。"""
    a = _ad.pick()
    return int(a.index) if a is not None else None


def interop_luid():
    """零拷贝 shim 的共享 LUID 锚（与 GL/D3D11 同卡的凭证）。"""
    a = _ad.pick()
    if a is not None and getattr(a, "luid", 0):
        return int(a.luid) & 0xFFFFFFFFFFFFFFFF
    return 0


def note_inference(desc):
    """推理 runner 就绪时登记（如 "DML @ adapter 1 NVIDIA ..."）。"""
    _layout["inference"] = str(desc)


def _override_index():
    """设置面板「推理显卡」的手动指定（"Auto" 或 adapter 序号字符串）。"""
    try:
        from utils import settings
        v = settings.get("Inference GPU", "Auto")
        if v in (None, "", "Auto", "auto"):
            return None
        return int(v)
    except Exception:
        return None


def inference_device_id():
    """推理（DML EP）device_id：手动指定优先，否则锚点（与 GL 同卡）。

    指定卡已消失（驱动更新/换机）时自动退回锚点——宁可跑对卡不肯跑没卡。
    """
    i = _override_index()
    if i is not None:
        try:
            for a in _ad.enum_adapters():
                if a.index == i:
                    return i
        except Exception:
            pass
    return dml_device_id()


def inference_note(is_dml=True):
    """runner 就绪日志的推理侧描述。

    手动指定仅 DML 后端生效——CUDA/TRT 的 GL 纹理注册（cudaGraphicsGL
    RegisterImage）要求 CUDA 上下文与 GL 同卡，选了别的卡互操作必失败，
    这不是可配置项是硬绑定。
    """
    if not is_dml:
        a = _ad.pick()
        return (f"CUDA/TRT @ {a.name}（隐式随 GL 卡）" if a is not None
                else "CUDA/TRT（未显式选卡）")
    i = _override_index()
    a = None
    try:
        ads = _ad.enum_adapters()
        a = next((x for x in ads if x.index == i), None) if i is not None else None
    except Exception:
        a = None
    if a is None:
        a = _ad.pick()
        if a is None:
            return "DML（默认设备）"
        return f"DML @ adapter {a.index} {a.name}（自动）"
    note = f"DML @ adapter {a.index} {a.name}（手动）"
    if _mismatch(_ad.gl_renderer(), a.name):
        note += " —— 与 GL 不同卡：零拷贝将退回读回路径"
    return note


def _mismatch(a, b):
    """两设备名互不含（同 adapter._norm_gl 口径）即视为跨卡。"""
    na, nb = _ad._norm_gl(a or ""), _ad._norm_gl(b or "")
    return bool(na and nb and na not in nb and nb not in na)


def report(detail=False):
    """GPU 布局日志行（返回 list[str]，调用方负责打印）。

    detail=True 附 adapter 枚举明细（boot 一次）；False 仅一行
    （runner 就绪时，采集/推理侧已知的完整图景）。
    """
    lines = []
    try:
        gl = _ad.gl_renderer()
        a = _ad.pick()
        cap = _ad.capture_device_name()
        inf = _layout["inference"]
        bits = []
        if gl:
            bits.append(f"GL={gl}")
        if a is not None:
            bits.append(f"锚点=adapter[{a.index}] {a.name}（{_ad.pick_reason(a)}）")
        if cap:
            bits.append(f"采集={cap}")
        if inf:
            bits.append(f"推理={inf}")
        lines.append("[GPU 布局] " + " | ".join(bits))
        if _mismatch(gl, cap):
            lines.append("[GPU 布局] 警告：GL 与采集设备不在同一块显卡 —— "
                         "D3D11↔GL 互操作会失败，SBS 退化成 2D。")
        if detail:
            try:
                _seen = {}
                for x in _ad.enum_adapters():
                    rects = ", ".join(f"{rc[0]},{rc[1]} {rc[2]-rc[0]}x{rc[3]-rc[1]}"
                                      for rc in x.rects()) or "-"
                    lines.append(f"[GPU 布局] adapter[{x.index}] = {x.name}  "
                                 f"VRAM {x.vram // (1 << 20)}MB  LUID {x.luid:#x}  "
                                 f"输出: {rects}")
                    k = (x.name, x.vram // (1 << 20))
                    if k in _seen:
                        lines.append(f"[GPU 布局] 注：adapter[{_seen[k]}] 与 "
                                     f"adapter[{x.index}] 同名同显存——远程/虚拟"
                                     f"显示驱动（IddCx）克隆形态，设置面板已去重，"
                                     f"手动选卡请以 Auto 为宜")
                    else:
                        _seen[k] = x.index
            except Exception as e:
                lines.append(f"[GPU 布局] adapter 枚举失败: {e}")
    except Exception as e:
        lines.append(f"[GPU 布局] 报告失败: {e}")
    return lines
