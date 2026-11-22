# -*- coding: utf-8 -*-
"""WGL_NV_DX_interop2 封装（GL 线程侧）：D3D11 纹理 -> GL 纹理零拷贝换装。

M0 闸 2 定案的坑：access 枚举是 WGL_ACCESS_*_NV（0x0/0x1）；
wglDXRegisterObjectNV（非 Resource）；wglGetProcAddress 必须走裸 opengl32
（PyOpenGL 返回 bytes 指针会被 WINFUNCTYPE 当垃圾地址）。
BGRA8 的 D3D 内存被 GL 按 RGBA 解释 —— 注册时设 swizzle 静默纠 channel。
"""
import ctypes

GL_TEXTURE_2D = 0x0DE1
WGL_ACCESS_READ_ONLY_NV = 0x0000
WGL_ACCESS_READ_WRITE_NV = 0x0001
# GL_TEXTURE_SWIZZLE_RGBA 参数：BGRA 内存 -> GL 返回 RGBA，且 alpha 恒 1。
# Chromium/Electron 窗口的 BGRA A 通道=0（预乘表面）——皮肤是 LAYERED 窗口，
# DWM 按逐像素 alpha 合成，透传 A=0 会把整个画面压黑（实测 3.5% 亮度）。
# GL_ONE=1（常量），GL_BLUE=0x1905/GL_GREEN=0x1904/GL_RED=0x1903
SWIZZLE_BGRA = (ctypes.c_int * 4)(0x1905, 0x1904, 0x1903, 1)

_op32 = ctypes.WinDLL("opengl32")
_op32.wglGetProcAddress.restype = ctypes.c_void_p
_op32.wglGetProcAddress.argtypes = [ctypes.c_char_p]


def _wgl(name, restype, *atypes):
    """取 WGL 扩展函数（普通 C 函数，无 COM this 位）。"""
    p = _op32.wglGetProcAddress(name.encode())
    if not p:
        raise RuntimeError(f"{name} 不可用（NV interop 缺失？）")
    return ctypes.WINFUNCTYPE(restype, *atypes)(p)


class DxInterop:
    """一个 D3D11 设备 ↔ 当前 GL context 的互操作会话（GL 线程独占）。"""

    def __init__(self, d3d_device_ptr):
        self.P = ctypes.c_void_p
        self._open = _wgl("wglDXOpenDeviceNV", self.P, self.P)
        self._reg = _wgl("wglDXRegisterObjectNV", self.P,
                         self.P, self.P, ctypes.c_uint, ctypes.c_uint,
                         ctypes.c_uint)
        self._unreg = _wgl("wglDXUnregisterObjectNV", ctypes.c_long,
                           self.P, self.P)
        self._lock = _wgl("wglDXLockObjectsNV", ctypes.c_long,
                          self.P, ctypes.c_int, self.P)
        self._unlock = _wgl("wglDXUnlockObjectsNV", ctypes.c_long,
                            self.P, ctypes.c_int, self.P)
        if isinstance(d3d_device_ptr, ctypes.c_void_p):
            d3d_device_ptr = d3d_device_ptr.value
        self.dxh = self._open(self.P(d3d_device_ptr))
        if not self.dxh:
            raise RuntimeError("wglDXOpenDeviceNV 失败")
        self._handles = {}   # gl_tex_glo -> interop HANDLE

    def register(self, gl_texture, d3d_tex_ptr):
        """moderngl 纹理对象挂接到 D3D 纹理存储。

        swizzle 不在此设：实测本机驱动在 GLFW 上下文对 interop 纹理采样
        无视 GL_TEXTURE_SWIZZLE_RGBA（状态回读正确、采样仍 R<->B 交换），
        通道交换统一走 shader 侧 .bgra（viewer 四 shader + 预处理）。
        """
        from OpenGL.GL import glBindTexture, glGetError
        self._drain_gl_errors(glGetError)
        if isinstance(d3d_tex_ptr, ctypes.c_void_p):
            d3d_tex_ptr = d3d_tex_ptr.value
        glBindTexture(GL_TEXTURE_2D, gl_texture.glo)
        h = self._reg(self.dxh, self.P(d3d_tex_ptr),
                      gl_texture.glo, GL_TEXTURE_2D, WGL_ACCESS_READ_WRITE_NV)
        if not h:
            raise RuntimeError("wglDXRegisterObjectNV 失败")
        self._handles[gl_texture.glo] = h
        return h

    def reregister(self, gl_texture, d3d_tex_ptr):
        """换代重挂：同一 GL 纹理对象换到新 D3D 纹理（GL 纹理永生，
        删除已注册纹理会让驱动 interop 记账坏掉——glBind 报 1282）。"""
        if gl_texture.glo in self._handles:
            try:
                self._unreg(self.dxh,
                            ctypes.c_void_p(self._handles[gl_texture.glo]))
            except Exception:
                pass
            self._handles.pop(gl_texture.glo, None)
        return self.register(gl_texture, d3d_tex_ptr)

    @staticmethod
    def _drain_gl_errors(glGetError):
        """排空残留 GL 错误位（错误是粘滞的，不清会把旧账记到下一个调用）。"""
        for _ in range(8):
            if glGetError() == 0:
                break

    def lock(self, gl_texture):
        h = self._handles[gl_texture.glo]
        ok = self._lock(self.dxh, 1, ctypes.byref(ctypes.c_void_p(h)))
        _retries = 0
        if not ok:
            # 采集线程可能尚在尾 DMA 上——短重试（规约允许等待语义）
            import time as _t
            for _ in range(4):
                _t.sleep(0.002)
                _retries += 1
                ok = self._lock(self.dxh, 1, ctypes.byref(ctypes.c_void_p(h)))
                if ok:
                    break
        # 诊断（D2S_ZC_PROBE）：记录本次 lock 结果与重试次数
        self.last_lock = {"ok": bool(ok), "retries": _retries,
                          "glo": gl_texture.glo}
        if not ok:
            raise RuntimeError("wglDXLockObjectsNV 失败")

    def unlock(self, gl_texture):
        h = self._handles.get(gl_texture.glo)
        if h:
            self._unlock(self.dxh, 1, ctypes.byref(ctypes.c_void_p(h)))

    def unregister(self, gl_texture):
        h = self._handles.pop(gl_texture.glo, None)
        if h:
            self._unreg(self.dxh, ctypes.c_void_p(h))
