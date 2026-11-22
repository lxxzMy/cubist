
## 运行 2026-11-21 03:23:00

| 探针 | 结果 | 备注 |
|---|---|---|
| A1 设备创建(debug layer) | 406.1ms | LUID 0x1730f |
| A2 适配器 | NVIDIA GeForce GTX 1650 | adapter[0] |
| A3 LUID 与 D3D11 侧一致 | True | d3d12 0x1730f vs d3d11 0x1730f |
| 烟测 A后 前 | 0x0 |  |
| 烟测 A后 建后 | 0x0 |  |
| 烟测 A后 释放后 | 0x0 |  |
| 烟测 A后 | ok |  |
| B 设备态 开始 | 0x0 |  |
| B dxc | FAIL | OSError: [Errno -2147024809] IDxcCompiler3.Compile |
| B 设备态 dxc 后 | 0x0 |  |
| B1 fxc 编译 | 4.4ms | fxc/ps_5_0 452B |
| B2 fxc 缓存命中 | 0.48ms | 磁盘 .cso |
| B 设备态 fxc 后 | 0x0 |  |
| B后 设备移除原因 | 0x0 |  |
| 烟测 B后 前 | 0x0 |  |
| 烟测 B后 建后 | 0x0 |  |
| 烟测 B后 释放后 | 0x0 |  |
| 烟测 B后 | ok |  |
| E 纹理共享句柄 | ok |  |
| E1 句柄 open（一次性） | fence 0.07ms tex 0.57ms | 缓存复用，不逐帧 open |
| E2 逐帧重 open 100 次均值 | 0.029ms/帧 | （印证缓存必要性） |
| E3 桥接单帧总时长 p50 | 0.31ms | p99 2.35ms / max 2.35ms |
| E4 D3D11 侧均摊（copy+signal） | 0.08ms |  |
| E5 D3D12 侧均摊（open wait+copy+submit） | 0.34ms |  |
| E6 host 同步均摊 | 0.00ms/帧均值已含 |  |
| E7 折算 fps 上限 | 3188.8 | （GL 时代同点 5-10ms 同步 → 对账见终报） |
| 烟测 E后 前 | 0x0 |  |
| 烟测 E后 建后 | 0x0 |  |
| 烟测 E后 释放后 | 0x0 |  |
| 烟测 E后 | ok |  |
| D framebuffer_size | 800x450 | GLFW NO_API |
| D GetClientRect | 800x450 | win32 客户区 |
| D 一致 | True | NO_API 下 fb=客户区像素 → 现有渲染尺寸代码可平移 |
| 烟测 D后 前 | 0x0 |  |
| 烟测 D后 建后 | 0x0 |  |
| 烟测 D后 释放后 | 0x0 |  |
| 烟测 D后 | ok |  |
| C plain×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C plain×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+alpha×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+alpha×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+noredir×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+noredir×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| 烟测 C后 前 | 0x0 |  |
| 烟测 C后 建后 | 0x0 |  |
| 烟测 C后 释放后 | 0x0 |  |
| 烟测 C后 | ok |  |
| F1 50 轮创建/释放 | alloc 150 free 50 | alloc==free=漏为零 |
| F2 存活对象差 | {'dev': 1, 'queue': 1, 'fence': 1, 'alloc': 1, 'cl': 1, 'heap0': 2, 'heap2': 1, 'cbres': 1, 'res': 101, 'factory': 6} | res 应不增长 |

## 运行 2026-11-21 03:23:24

| 探针 | 结果 | 备注 |
|---|---|---|
| A1 设备创建(debug layer) | 403.1ms | LUID 0x1730f |
| A2 适配器 | NVIDIA GeForce GTX 1650 | adapter[0] |
| A3 LUID 与 D3D11 侧一致 | True | d3d12 0x1730f vs d3d11 0x1730f |
| 烟测 A后 前 | 0x0 |  |
| 烟测 A后 建后 | 0x0 |  |
| 烟测 A后 释放后 | 0x0 |  |
| 烟测 A后 | ok |  |
| B 设备态 开始 | 0x0 |  |
| B dxc | FAIL | OSError: [Errno -2147024809] IDxcCompiler3.Compile |
| B 设备态 dxc 后 | 0x0 |  |
| B1 fxc 编译 | 4.1ms | fxc/ps_5_0 452B |
| B2 fxc 缓存命中 | 0.45ms | 磁盘 .cso |
| B 设备态 fxc 后 | 0x0 |  |
| B后 设备移除原因 | 0x0 |  |
| 烟测 B后 前 | 0x0 |  |
| 烟测 B后 建后 | 0x0 |  |
| 烟测 B后 释放后 | 0x0 |  |
| 烟测 B后 | ok |  |
| E 纹理共享句柄 | ok |  |
| E1 句柄 open（一次性） | fence 0.07ms tex 0.59ms | 缓存复用，不逐帧 open |
| E2 逐帧重 open 100 次均值 | 0.027ms/帧 | （印证缓存必要性） |
| E3 桥接单帧总时长 p50 | 0.31ms | p99 2.39ms / max 2.39ms |
| E4 D3D11 侧均摊（copy+signal） | 0.06ms |  |
| E5 D3D12 侧均摊（open wait+copy+submit） | 0.34ms |  |
| E6 host 同步均摊 | 0.00ms/帧均值已含 |  |
| E7 折算 fps 上限 | 3272.3 | （GL 时代同点 5-10ms 同步 → 对账见终报） |
| 烟测 E后 前 | 0x0 |  |
| 烟测 E后 建后 | 0x0 |  |
| 烟测 E后 释放后 | 0x0 |  |
| 烟测 E后 | ok |  |
| D framebuffer_size | 800x450 | GLFW NO_API |
| D GetClientRect | 800x450 | win32 客户区 |
| D 一致 | True | NO_API 下 fb=客户区像素 → 现有渲染尺寸代码可平移 |
| 烟测 D后 前 | 0x0 |  |
| 烟测 D后 建后 | 0x0 |  |
| 烟测 D后 释放后 | 0x0 |  |
| 烟测 D后 | ok |  |
| C plain×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C plain×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+alpha×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+alpha×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+noredir×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+noredir×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| 烟测 C后 前 | 0x0 |  |
| 烟测 C后 建后 | 0x0 |  |
| 烟测 C后 释放后 | 0x0 |  |
| 烟测 C后 | ok |  |
| F1 50 轮创建/释放 | alloc 150 free 150 | alloc==free=漏为零 |
| F2 存活对象差 | {'dev': 1, 'queue': 1, 'fence': 1, 'alloc': 1, 'cl': 1, 'heap0': 2, 'heap2': 1, 'cbres': 1, 'res': 1, 'factory': 6} | res 应不增长 |

## 运行 2026-11-21 03:24:04

| 探针 | 结果 | 备注 |
|---|---|---|
| A1 设备创建(debug layer) | 404.5ms | LUID 0x1730f |
| A2 适配器 | NVIDIA GeForce GTX 1650 | adapter[0] |
| A3 LUID 与 D3D11 侧一致 | True | d3d12 0x1730f vs d3d11 0x1730f |
| 烟测 A后 前 | 0x0 |  |
| 烟测 A后 建后 | 0x0 |  |
| 烟测 A后 释放后 | 0x0 |  |
| 烟测 A后 | ok |  |
| B 设备态 开始 | 0x0 |  |
| B dxc | FAIL | OSError: [Errno -2147024809] IDxcCompiler3.Compile |
| B 设备态 dxc 后 | 0x0 |  |
| B1 fxc 编译 | 4.4ms | fxc/ps_5_0 452B |
| B2 fxc 缓存命中 | 0.52ms | 磁盘 .cso |
| B 设备态 fxc 后 | 0x0 |  |
| B后 设备移除原因 | 0x0 |  |
| 烟测 B后 前 | 0x0 |  |
| 烟测 B后 建后 | 0x0 |  |
| 烟测 B后 释放后 | 0x0 |  |
| 烟测 B后 | ok |  |
| E 纹理共享句柄 | ok |  |
| E1 句柄 open（一次性） | fence 0.09ms tex 0.59ms | 缓存复用，不逐帧 open |
| E2 逐帧重 open 100 次均值 | 0.028ms/帧 | （印证缓存必要性） |
| E3 桥接单帧总时长 p50 | 0.33ms | p99 2.58ms / max 2.58ms |
| E4 D3D11 侧均摊（copy+signal） | 0.06ms |  |
| E5 D3D12 侧均摊（open wait+copy+submit） | 0.30ms |  |
| E6 host 同步均摊 | 0.00ms/帧均值已含 |  |
| E7 折算 fps 上限 | 3007.5 | （GL 时代同点 5-10ms 同步 → 对账见终报） |
| 烟测 E后 前 | 0x0 |  |
| 烟测 E后 建后 | 0x0 |  |
| 烟测 E后 释放后 | 0x0 |  |
| 烟测 E后 | ok |  |
| D framebuffer_size | 800x450 | GLFW NO_API |
| D GetClientRect | 800x450 | win32 客户区 |
| D 一致 | True | NO_API 下 fb=客户区像素 → 现有渲染尺寸代码可平移 |
| 烟测 D后 前 | 0x0 |  |
| 烟测 D后 建后 | 0x0 |  |
| 烟测 D后 释放后 | 0x0 |  |
| 烟测 D后 | ok |  |
| C plain×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C plain×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+alpha×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+alpha×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+noredir×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+noredir×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| 烟测 C后 前 | 0x0 |  |
| 烟测 C后 建后 | 0x0 |  |
| 烟测 C后 释放后 | 0x0 |  |
| 烟测 C后 | ok |  |
| F1 50 轮创建/释放 | alloc 150 free 150 | alloc==free=漏为零 |
| F2 存活对象差 | {'dev': 1, 'queue': 1, 'fence': 1, 'alloc': 1, 'cl': 1, 'heap0': 2, 'heap2': 1, 'cbres': 1, 'res': 1, 'factory': 6} | res 应不增长 |

## 运行 2026-11-21 16:08:18

| 探针 | 结果 | 备注 |
|---|---|---|
| A1 设备创建(debug layer) | 410.5ms | LUID 0x1730f |
| A2 适配器 | NVIDIA GeForce GTX 1650 | adapter[0] |
| A3 LUID 与 D3D11 侧一致 | True | d3d12 0x1730f vs d3d11 0x1730f |
| 烟测 A后 前 | 0x0 |  |
| 烟测 A后 建后 | 0x0 |  |
| 烟测 A后 释放后 | 0x0 |  |
| 烟测 A后 | ok |  |
| B 设备态 开始 | 0x0 |  |
| B1 dxc 编译 | 8.8ms | dxc/ps_6_0 2528B |
| B2 dxc 缓存命中 | 0.55ms | 磁盘 .cso |
| B 设备态 dxc 后 | 0x0 |  |
| B1 fxc 编译 | 4.7ms | fxc/ps_5_0 452B |
| B2 fxc 缓存命中 | 0.57ms | 磁盘 .cso |
| B 设备态 fxc 后 | 0x0 |  |
| B后 设备移除原因 | 0x0 |  |
| 烟测 B后 前 | 0x0 |  |
| 烟测 B后 建后 | 0x0 |  |
| 烟测 B后 释放后 | 0x0 |  |
| 烟测 B后 | ok |  |
| E 纹理共享句柄 | ok |  |
| E1 句柄 open（一次性） | fence 0.10ms tex 0.84ms | 缓存复用，不逐帧 open |
| E2 逐帧重 open 100 次均值 | 0.034ms/帧 | （印证缓存必要性） |
| E3 桥接单帧总时长 p50 | 0.35ms | p99 1.75ms / max 2.68ms |
| E4 D3D11 侧均摊（copy+signal） | 0.05ms |  |
| E5 D3D12 侧均摊（open wait+copy+submit） | 0.30ms |  |
| E6 host 同步均摊 | 0.00ms/帧均值已含 |  |
| E7 折算 fps 上限 | 2859.6 | （GL 时代同点 5-10ms 同步 → 对账见终报） |
| 烟测 E后 前 | 0x0 |  |
| 烟测 E后 建后 | 0x0 |  |
| 烟测 E后 释放后 | 0x0 |  |
| 烟测 E后 | ok |  |
| D framebuffer_size | 800x450 | GLFW NO_API |
| D GetClientRect | 800x450 | win32 客户区 |
| D 一致 | True | NO_API 下 fb=客户区像素 → 现有渲染尺寸代码可平移 |
| 烟测 D后 前 | 0x0 |  |
| 烟测 D后 建后 | 0x0 |  |
| 烟测 D后 释放后 | 0x0 |  |
| 烟测 D后 | ok |  |
| C plain×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C plain×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+alpha×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+alpha×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| C layered+noredir×flip | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=flip) |
| C layered+noredir×blt | 不支持 | OSError: winerror=None [Errno -2005270527] CreateSwapChainForHwnd(model=blt) |
| 烟测 C后 前 | 0x0 |  |
| 烟测 C后 建后 | 0x0 |  |
| 烟测 C后 释放后 | 0x0 |  |
| 烟测 C后 | ok |  |
| F1 50 轮创建/释放 | alloc 150 free 150 | alloc==free=漏为零 |
| F2 存活对象差 | {'dev': 1, 'queue': 1, 'fence': 1, 'alloc': 1, 'cl': 1, 'heap0': 2, 'heap2': 1, 'cbres': 1, 'res': 1, 'factory': 6} | res 应不增长 |
