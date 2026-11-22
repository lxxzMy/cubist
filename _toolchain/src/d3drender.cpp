// d3drender.dll — D3D12 图形渲染 C 助手（迁移 R1，2026-11-21）
// 封装全部 D3D12 ABI 敏感操作（PSO 创建/命令录制/屏障/描述符），
// Python 只调简单 C 函数（句柄 + 字节缓冲）。shim 同款模式。
// 编译：build_d3drender.cmd（与 build_shim.sh 同款 cl.exe 调用）
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <d3d11.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dcomp.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dcomp.lib")

// flip 链缓冲数：2 个缓冲时，合成器还攥着上一帧时我们就得复用另一块，
// 运动内容上会出现一条/两条滚动的撕裂缝（3D 全屏覆盖层实测）。3 个缓冲
// 让驱动总有空闲缓冲可排到下一个 VBlank，是 flip 模型的常规配置。
#define SWAP_N 3
// per-frame allocator 组数。流水线每拍要用 3 组（prep / 深度链 / 显示），
// 2~3 组时下一拍就落回还在飞的组 → begin_frame 阻塞等 GPU（实测 proc 30ms）；
// 6 组 = 两拍的量，留出流水余量。
#define ALLOC_N 6

// ---- 上下文 ------------------------------------------------------------
struct RenderCtx {
    ID3D12Device* dev = NULL;
    ID3D12CommandQueue* queue = NULL;
    // per-frame allocator 轮换（ALLOC_N 组）：消除 present 内无条件 fence
    // 等待——begin_frame 切组前只等"该组上一次提交"完成（流水线关键）
    ID3D12CommandAllocator* alloc[ALLOC_N] = {NULL, NULL, NULL, NULL, NULL, NULL};
    ID3D12GraphicsCommandList* cl[ALLOC_N] = {NULL, NULL, NULL, NULL, NULL, NULL};
    UINT frame_i = 0;                  // 当前帧组（begin_frame 切换）
    UINT64 alloc_done[ALLOC_N] = {0, 0, 0, 0, 0, 0};   // 各组最后一次提交的 fence 值
    ID3D12CommandAllocator* rb_alloc = NULL;   // readback 专用（旁路轮换）
    ID3D12GraphicsCommandList* rb_cl = NULL;
    ID3D12Fence* fence = NULL;
    // SRV 全部直接建在 arena（shader-visible）里，每纹理一固定槽 ——
    // shader-visible 堆 CPU 句柄 write-only，不可作 CopyDescriptors 源，
    // "每帧拷贝进 arena" 方案无效，固定槽即可
    ID3D12DescriptorHeap* arena = NULL;
    ID3D12DescriptorHeap* rtv_heap = NULL;     // CPU RTV
    ID3D12RootSignature* rs = NULL;
    HANDLE wait_event = NULL;
    UINT64 fence_val = 0;
    UINT srv_inc = 0, rtv_inc = 0;
    UINT arena_next = 0, rtv_next = 0;
    // cbuffer upload（root CBV b0）
    ID3D12Resource* cb_res = NULL;
    void* cb_map = NULL;
    UINT cb_next = 0;
    static const UINT CB_STRIDE = 64;   // floats/pass (256B)
    static const UINT CB_SLOTS = 512;
    UINT64 arena_base_cpu = 0;
    UINT64 arena_base_gpu = 0;
    // PSO 缓存
    struct PSOEntry { char name[64]; ID3D12PipelineState* pso; };
    PSOEntry pso_cache[32] = {};
    UINT pso_n = 0;
    // Swapchain
    IDXGISwapChain1* sc = NULL;
    IDXGISwapChain2* sc2 = NULL;            // waitable-object 接口
    IDXGISwapChain3* sc3 = NULL;            // GetCurrentBackBufferIndex
    // DirectComposition（D2S_DCOMP=1）：交换链走 composition + 逐像素 alpha
    IDCompositionDevice* dcomp = NULL;
    IDCompositionTarget* dcomp_target = NULL;
    IDCompositionVisual* dcomp_visual = NULL;
    HANDLE sc_waitable = NULL;              // GetFrameLatencyWaitableObject
    UINT sc_w = 0, sc_h = 0;
    ID3D12Resource* backbuf[SWAP_N] = {};
    D3D12_CPU_DESCRIPTOR_HANDLE bb_rtv[SWAP_N] = {};
    UINT sc_idx = 0;
    D3D12_RESOURCE_STATES bb_state[SWAP_N] = {};   // 0 == COMMON
    // 错误
    char err[512] = {};
    bool frame_open = false;   // cl 正在录制（防 open-list Reset 序列错）
};

static RenderCtx* g_ctx = NULL;

// 当前后台缓冲序号：优先问 DXGI（Present 被遮挡/丢弃时手动轮转必然错位，
// 错位后会画进正在扫描输出的前缓冲 → 撕裂）。sc3 不可用时退回自记序号。
static UINT cur_bb(RenderCtx* s) {
    if (!s) return 0;
    if (s->sc3) {
        UINT i = s->sc3->GetCurrentBackBufferIndex();
        if (i < SWAP_N) return i;
    }
    return s->sc_idx % SWAP_N;
}

static void set_err(RenderCtx* s, const char* msg) {
    if (s && s->err[0] == 0) strncpy_s(s->err, msg, _TRUNCATE);
}

// debug 设备上把最新 ERROR 级 InfoQueue 消息追加到 err（诊断 Close/ECL 失败）
static void drain_infoq(RenderCtx* s) {
    if (!s || !s->dev) return;
    ID3D12InfoQueue* iq = NULL;
    if (FAILED(s->dev->QueryInterface(__uuidof(ID3D12InfoQueue), (void**)&iq)))
        return;
    UINT64 n = iq->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T sz = 0;
        iq->GetMessage(i, NULL, &sz);
        if (!sz) continue;
        auto* m = (D3D12_MESSAGE*)malloc(sz);
        iq->GetMessage(i, m, &sz);
        if (m->Severity >= D3D12_MESSAGE_SEVERITY_ERROR) {
            strncat_s(s->err, " | ", _TRUNCATE);
            strncat_s(s->err, m->pDescription, _TRUNCATE);
        }
        free(m);
    }
    iq->Release();
}

// ---- 导出 --------------------------------------------------------------

extern "C" {

__declspec(dllexport)
RenderCtx* d3dr_create(uint64_t adapter_luid_match, int debug) {
    auto* s = new RenderCtx();
    if (debug) {
        ID3D12Debug* dbg = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(__uuidof(ID3D12Debug), (void**)&dbg))) {
            dbg->EnableDebugLayer(); dbg->Release();
        }
        ID3D12Debug1* dbg1 = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(__uuidof(ID3D12Debug1), (void**)&dbg1))) {
            dbg1->SetEnableGPUBasedValidation(TRUE); dbg1->Release();
        }
    }
    // 枚举 adapter（LUID 匹配或第一个硬件卡）
    IDXGIFactory1* fct = NULL;
    CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fct);
    IDXGIAdapter1* adapter = NULL;
    for (UINT i = 0; fct->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); adapter = NULL; continue; }
        if (adapter_luid_match == 0 ||
            (d.AdapterLuid.HighPart == (int32_t)(adapter_luid_match >> 32) &&
             d.AdapterLuid.LowPart == (uint32_t)adapter_luid_match)) {
            break;  // 匹配或第一个非软件卡
        }
        adapter->Release(); adapter = NULL;
    }
    if (!adapter) { set_err(s, "no adapter"); delete s; fct->Release(); return NULL; }
    // D3D12 设备
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                                   __uuidof(ID3D12Device), (void**)&s->dev);
    adapter->Release(); fct->Release();
    if (FAILED(hr)) { set_err(s, "D3D12CreateDevice failed"); delete s; return NULL; }
    // 队列
    // 高优先级：与游戏/其他 GPU 客户端抢调度窗口时优先跑我们这条队列。
    // 我们每帧 GPU 工作量很小（224² 链 + 1080p 显示，实测 <2% 利用率），但
    // DML Run / 统计读回都是同步等待——对方占满引擎时会被拖到 15~40ms
    // （2026-11-22 Hogwarts Legacy 场景实测：proc 7.7→46ms）。
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    if (FAILED(s->dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&s->queue))) {
        set_err(s, "CreateCommandQueue failed"); delete s; return NULL; }
    // fence + event
    if (FAILED(s->dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&s->fence))) {
        set_err(s, "CreateFence failed"); delete s; return NULL; }
    s->wait_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    // allocator + command list（per-frame 轮换 ALLOC_N 组 + readback 专用 1 组）
    for (int q = 0; q < ALLOC_N; ++q) {
        if (FAILED(s->dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                __uuidof(ID3D12CommandAllocator), (void**)&s->alloc[q]))) {
            set_err(s, "CreateCommandAllocator failed"); delete s; return NULL; }
        if (FAILED(s->dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                s->alloc[q], NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&s->cl[q]))) {
            set_err(s, "CreateCommandList failed"); delete s; return NULL; }
        s->cl[q]->Close();   // 新建列表先 Close 才能 Reset
    }
    if (FAILED(s->dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            __uuidof(ID3D12CommandAllocator), (void**)&s->rb_alloc))) {
        set_err(s, "CreateCommandAllocator(rb) failed"); delete s; return NULL; }
    if (FAILED(s->dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            s->rb_alloc, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&s->rb_cl))) {
        set_err(s, "CreateCommandList(rb) failed"); delete s; return NULL; }
    s->rb_cl->Close();
    // 描述符堆：SRV 直接建在 shader-visible arena 固定槽（每纹理一槽），
    // 不做每帧 CopyDescriptors（shader-visible 堆 CPU 句柄 write-only，
    // 不能作拷贝源）
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 4096;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(s->dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&s->arena))) {
        set_err(s, "CreateDescriptorHeap(arena) failed"); delete s; return NULL; }
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 4096;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(s->dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&s->rtv_heap))) {
        set_err(s, "CreateDescriptorHeap(RTV) failed"); delete s; return NULL; }
    s->srv_inc = s->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s->rtv_inc = s->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE ac = s->arena->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE ag = s->arena->GetGPUDescriptorHandleForHeapStart();
    s->arena_base_cpu = ac.ptr;
    s->arena_base_gpu = ag.ptr;
    // cbuffer upload
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = RenderCtx::CB_STRIDE * 4 * RenderCtx::CB_SLOTS;
    bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(s->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
            D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
            __uuidof(ID3D12Resource), (void**)&s->cb_res))) {
        set_err(s, "CreateCommittedResource(cbuffer) failed"); delete s; return NULL; }
    D3D12_RANGE nr = {0, 0};
    s->cb_res->Map(0, &nr, &s->cb_map);
    g_ctx = s;
    return s;
}

__declspec(dllexport)
void d3dr_destroy(RenderCtx* s) {
    if (!s) return;
    if (s->rb_cl) s->rb_cl->Release();
    if (s->rb_alloc) s->rb_alloc->Release();
    for (int q = 0; q < ALLOC_N; ++q) {
        if (s->cl[q]) s->cl[q]->Release();
        if (s->alloc[q]) s->alloc[q]->Release();
    }
    if (s->queue) s->queue->Release();
    if (s->fence) s->fence->Release();
    if (s->arena) s->arena->Release();
    if (s->rtv_heap) s->rtv_heap->Release();
    if (s->cb_res) s->cb_res->Release();
    if (s->rs) s->rs->Release();
    for (UINT i = 0; i < s->pso_n; ++i)
        if (s->pso_cache[i].pso) s->pso_cache[i].pso->Release();
    if (s->sc) s->sc->Release();
    if (s->dcomp_visual) s->dcomp_visual->Release();
    if (s->dcomp_target) s->dcomp_target->Release();
    if (s->dcomp) s->dcomp->Release();
    for (int q = 0; q < SWAP_N; ++q) if (s->backbuf[q]) s->backbuf[q]->Release();
    if (s->wait_event) CloseHandle(s->wait_event);
    delete s;
    if (g_ctx == s) g_ctx = NULL;
}

__declspec(dllexport)
const char* d3dr_get_err(RenderCtx* s) { return s ? s->err : "null ctx"; }

// ---- 根签名（首次调用时创建，之后缓存） --------------------------------
// 布局：root slot0..7 = 独立 SRV table（t0..t7 各一槽，GL location 语义），
// root slot8 = CBV(b0)，静态采样器 s0(linear)/s1(point)。
// 独立参数而非单一连续表：5-SRV 中值 pass 等场景的纹理 SRV 在 arena 里
// 分散分配，连续表要求连续描述符（不成立）；root 空间 8×2+2=18 DWORD < 64。
static ID3D12RootSignature* get_root_sig(RenderCtx* s) {
    if (s->rs) return s->rs;
    D3D12_DESCRIPTOR_RANGE ranges[8] = {};
    D3D12_ROOT_PARAMETER params[9] = {};
    for (UINT i = 0; i < 8; ++i) {
        ranges[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[i].NumDescriptors = 1;
        ranges[i].BaseShaderRegister = i;
        ranges[i].RegisterSpace = 0;
        ranges[i].OffsetInDescriptorsFromTableStart = 0;
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[i].DescriptorTable.NumDescriptorRanges = 1;
        params[i].DescriptorTable.pDescriptorRanges = &ranges[i];
    }
    params[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[8].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[8].Descriptor.ShaderRegister = 0;
    params[8].Descriptor.RegisterSpace = 0;
    D3D12_STATIC_SAMPLER_DESC samp[2] = {};
    samp[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samp[0].AddressU = samp[0].AddressV = samp[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    samp[0].MinLOD = 0.0f; samp[0].MaxLOD = 1e6f;
    samp[0].ShaderRegister = 0;
    samp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    samp[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samp[1].AddressU = samp[1].AddressV = samp[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp[1].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    samp[1].MinLOD = 0.0f; samp[1].MaxLOD = 1e6f;
    samp[1].ShaderRegister = 1;
    samp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {9, params, 2, samp, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob* blob = NULL; ID3DBlob* err = NULL;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        set_err(s, "D3D12SerializeRootSignature failed");
        if (err) err->Release();
        return NULL;
    }
    HRESULT hr = s->dev->CreateRootSignature(0, blob->GetBufferPointer(),
        blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&s->rs);
    blob->Release(); if (err) err->Release();
    if (FAILED(hr)) { set_err(s, "CreateRootSignature failed"); return NULL; }
    return s->rs;
}

// ---- PSO ---------------------------------------------------------------
__declspec(dllexport)
void* d3dr_create_pso_blend(RenderCtx* s, const char* name,
                            const uint8_t* vs_data, uint32_t vs_len,
                            const uint8_t* ps_data, uint32_t ps_len,
                            uint32_t rtv_fmt, int blend);

__declspec(dllexport)
void* d3dr_create_pso(RenderCtx* s, const char* name,
                      const uint8_t* vs_data, uint32_t vs_len,
                      const uint8_t* ps_data, uint32_t ps_len,
                      uint32_t rtv_fmt) {
    return d3dr_create_pso_blend(s, name, vs_data, vs_len, ps_data, ps_len,
                                 rtv_fmt, 0);
}

// blend=1：alpha 混合（SrcAlpha/InvSrcAlpha；overlay/cursor 用）
__declspec(dllexport)
void* d3dr_create_pso_blend(RenderCtx* s, const char* name,
                            const uint8_t* vs_data, uint32_t vs_len,
                            const uint8_t* ps_data, uint32_t ps_len,
                            uint32_t rtv_fmt, int blend) {
    if (!s || !s->dev) return NULL;
    // 缓存命中（缓存键含 name——需混合的调用方用不同 name）
    for (UINT i = 0; i < s->pso_n; ++i)
        if (strcmp(s->pso_cache[i].name, name) == 0) return s->pso_cache[i].pso;
    ID3D12RootSignature* rs = get_root_sig(s);
    if (!rs) return NULL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = rs;
    pd.VS.pShaderBytecode = vs_data; pd.VS.BytecodeLength = vs_len;
    pd.PS.pShaderBytecode = ps_data; pd.PS.BytecodeLength = ps_len;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    if (blend) {
        auto& rt = pd.BlendState.RenderTarget[0];
        rt.BlendEnable = TRUE;
        rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = (DXGI_FORMAT)rtv_fmt;
    pd.SampleDesc.Count = 1;
    pd.SampleMask = 0xFFFFFFFF;   // 全零 desc 的 0 = 不写任何样本
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = s->dev->CreateGraphicsPipelineState(&pd,
        __uuidof(ID3D12PipelineState), (void**)&pso);
    if (FAILED(hr)) {
        char buf[64]; sprintf_s(buf, sizeof(buf), "CreatePSO failed hr=%08x", (unsigned)hr);
        set_err(s, buf); return NULL;
    }
    if (s->pso_n < 32) {
        strncpy_s(s->pso_cache[s->pso_n].name, name, _TRUNCATE);
        s->pso_cache[s->pso_n].pso = pso;
        s->pso_n++;
    }
    return pso;
}

// ---- 纹理 --------------------------------------------------------------
__declspec(dllexport)
void* d3dr_create_tex(RenderCtx* s, uint32_t w, uint32_t h,
                      uint32_t fmt, int heap_type, int renderable) {
    if (!s || !s->dev) return NULL;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = (D3D12_HEAP_TYPE)heap_type;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = (DXGI_FORMAT)fmt;
    d.SampleDesc.Count = 1;
    d.Flags = renderable ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_STATES init = D3D12_RESOURCE_STATE_COMMON;
    if (heap_type == D3D12_HEAP_TYPE_UPLOAD) init = D3D12_RESOURCE_STATE_GENERIC_READ;
    if (heap_type == D3D12_HEAP_TYPE_READBACK) init = D3D12_RESOURCE_STATE_COPY_DEST;
    ID3D12Resource* res = NULL;
    if (FAILED(s->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
            init, NULL, __uuidof(ID3D12Resource), (void**)&res))) {
        set_err(s, "CreateCommittedResource failed"); return NULL; }
    return res;
}

__declspec(dllexport)
void d3dr_release_tex(void* tex) {
    if (tex) {
        // D3D12 spec: resources must be unmapped before Release (repeated
        // Map/Release cycles left mapped corrupted heap state in practice)
        ((ID3D12Resource*)tex)->Unmap(0, NULL);
        ((ID3D12Resource*)tex)->Release();
    }
}

__declspec(dllexport)
void* d3dr_alloc_rtv(RenderCtx* s, void* tex, uint32_t fmt) {
    if (!s || !s->rtv_heap || !tex) return NULL;
    auto* res = (ID3D12Resource*)tex;
    D3D12_CPU_DESCRIPTOR_HANDLE base = s->rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE h = {base.ptr + (SIZE_T)s->rtv_next * s->rtv_inc};
    D3D12_RENDER_TARGET_VIEW_DESC d = {};
    d.Format = (DXGI_FORMAT)fmt;
    d.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    s->dev->CreateRenderTargetView(res, &d, h);
    s->rtv_next++;
    return (void*)(uintptr_t)h.ptr;
}

// SRV 建 arena 固定槽；返回 **GPU 句柄**（root table 直接绑定用）
__declspec(dllexport)
uint64_t d3dr_alloc_srv(RenderCtx* s, void* tex, uint32_t fmt) {
    if (!s || !s->arena || !tex) return 0;
    auto* res = (ID3D12Resource*)tex;
    D3D12_CPU_DESCRIPTOR_HANDLE h = {s->arena_base_cpu +
                                     (UINT64)s->arena_next * s->srv_inc};
    D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
    d.Format = (DXGI_FORMAT)fmt;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 0xFFFFFFFF;
    s->dev->CreateShaderResourceView(res, &d, h);
    s->arena_next++;
    return s->arena_base_gpu + (UINT64)(s->arena_next - 1) * s->srv_inc;
}

// ---- 上传 / 读回 ---------------------------------------------------------
__declspec(dllexport)
void* d3dr_make_upload_buf(RenderCtx* s, uint32_t nbytes) {
    if (!s || !s->dev) return NULL;
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = nbytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* res = NULL;
    if (FAILED(s->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
            __uuidof(ID3D12Resource), (void**)&res))) {
        set_err(s, "CreateCommittedResource(upload buf) failed"); return NULL; }
    return res;
}

__declspec(dllexport)
void* d3dr_buf_map(void* buf) {
    if (!buf) return NULL;
    void* ptr = NULL;
    D3D12_RANGE nr = {0, 0};
    ((ID3D12Resource*)buf)->Map(0, &nr, &ptr);
    return ptr;
}

__declspec(dllexport)
void d3dr_copy_buf_to_tex(RenderCtx* s, void* buf, void* dst_tex,
                          uint32_t w, uint32_t h, uint32_t fmt, uint32_t row_pitch) {
    if (!s || !s->cl[s->frame_i]) return;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = (ID3D12Resource*)buf;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    // D3D12_PLACED_SUBRESOURCE_FOOTPRINT 的 Footprint 字段是嵌套
    // D3D12_SUBRESOURCE_FOOTPRINT（Format/Width/Height/Depth/RowPitch）
    src.PlacedFootprint.Footprint.Format = (DXGI_FORMAT)fmt;
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = row_pitch;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = (ID3D12Resource*)dst_tex;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    s->cl[s->frame_i]->CopyTextureRegion(&dst, 0, 0, 0, &src, NULL);
}

__declspec(dllexport)
void* d3dr_make_readback_buf(RenderCtx* s, uint32_t nbytes) {
    if (!s || !s->dev) return NULL;
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = nbytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* res = NULL;
    HRESULT hr = s->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_COPY_DEST, NULL,
            __uuidof(ID3D12Resource), (void**)&res);
    if (FAILED(hr)) {
        char buf[96];
        sprintf_s(buf, sizeof(buf), "CreateCommittedResource(readback buf)"
                  " failed hr=%08x removed=%08x", (unsigned)hr,
                  (unsigned)s->dev->GetDeviceRemovedReason());
        set_err(s, buf);
        return NULL;
    }
    return res;
}

// 设备健康探针（device removed reason；0 = OK）
__declspec(dllexport)
int32_t d3dr_dev_removed_reason(RenderCtx* s) {
    if (!s || !s->dev) return -1;
    return (int32_t)s->dev->GetDeviceRemovedReason();
}

__declspec(dllexport)
void* d3dr_buf_ptr(void* buf) {
    if (!buf) return NULL;
    void* ptr = NULL;
    // readRange=NULL：整个资源要读（驱动做 CPU 缓存失效；{0,0} 会跳过）
    ((ID3D12Resource*)buf)->Map(0, NULL, &ptr);
    return ptr;
}

// tex → readback buffer（RowPitch 由调用方给，须 256 对齐）
__declspec(dllexport)
int d3dr_copy_tex_to_buf(RenderCtx* s, void* src_tex, void* dst_buf,
                         uint32_t w, uint32_t h, uint32_t fmt, uint32_t row_pitch) {
    if (!s || !s->cl) return -1;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = (ID3D12Resource*)src_tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = (ID3D12Resource*)dst_buf;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = (DXGI_FORMAT)fmt;
    dst.PlacedFootprint.Footprint.Width = w;
    dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = row_pitch;
    s->cl[s->frame_i]->CopyTextureRegion(&dst, 0, 0, 0, &src, NULL);
    return 0;
}

// ---- 命令录制 / 提交 -----------------------------------------------------
__declspec(dllexport)
int d3dr_begin_frame(RenderCtx* s) {
    if (!s || !s->cl[0]) return -1;
    // per-frame 轮换：切到下一组 allocator；若该组上一次提交 GPU 未完成
    // 才等（流水线关键——只在真正的资源依赖处同步，非每帧硬等）
    s->frame_i = (s->frame_i + 1) % ALLOC_N;
    UINT64 cur = s->fence->GetCompletedValue();
    if (cur < s->alloc_done[s->frame_i]) {
        s->fence->SetEventOnCompletion(s->alloc_done[s->frame_i], s->wait_event);
        WaitForSingleObject(s->wait_event, 10000);
    }
    if (s->frame_open) s->cl[s->frame_i]->Close();   // 防御 open Reset
    // 关键：allocator 必须先 Reset。只 Reset list 不 Reset allocator 时，
    // 每次提交都会向 allocator 追加 command buffer 内存且永不回收
    // （实测 NV 200KB/次、AMD 66KB/次；几分钟耗尽内存 → Close 返回
    // E_OUTOFMEMORY / 驱动 TDR）。fence 等待保证上一轮 GPU 已用完该组。
    if (FAILED(s->alloc[s->frame_i]->Reset())) return -2;
    if (FAILED(s->cl[s->frame_i]->Reset(s->alloc[s->frame_i], NULL))) return -2;
    s->frame_open = true;
    return 0;
}

__declspec(dllexport)
int d3dr_draw_pass(RenderCtx* s, void* pso,
                   uint64_t* srv_gpu_handles, uint32_t n_srvs,
                   const float* cb_data, uint32_t cb_floats,
                   void* rtv_handle, uint32_t dst_w, uint32_t dst_h,
                   int32_t vx, int32_t vy, int32_t vw, int32_t vh,
                   int use_scissor, int32_t sx, int32_t sy,
                   int32_t sw, int32_t sh) {
    if (!s || !s->cl[s->frame_i] || !pso) return -1;
    auto* cl = s->cl[s->frame_i];
    // 本帧描述符堆（SRV 全在 arena）
    ID3D12DescriptorHeap* heaps[] = {s->arena};
    cl->SetDescriptorHeaps(1, heaps);
    // cbuffer 槽
    s->cb_next = (s->cb_next + 1) % RenderCtx::CB_SLOTS;
    UINT slot = s->cb_next;
    if (cb_data && cb_floats > 0) {
        UINT n = min(cb_floats, RenderCtx::CB_STRIDE);
        memcpy((uint8_t*)s->cb_map + slot * RenderCtx::CB_STRIDE * 4,
               cb_data, n * 4);
    }
    cl->SetPipelineState((ID3D12PipelineState*)pso);
    cl->SetGraphicsRootSignature(s->rs);
    // root 参数布局：slot0..7 = SRV tables，slot8 = CBV(b0)
    cl->SetGraphicsRootConstantBufferView(8,
        s->cb_res->GetGPUVirtualAddress() + slot * RenderCtx::CB_STRIDE * 4);
    // SRV：root slot i ← 第 i 个 SRV 的 arena GPU 句柄（GL location 语义）
    for (UINT i = 0; i < n_srvs && i < 8; ++i) {
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = {srv_gpu_handles[i]};
        cl->SetGraphicsRootDescriptorTable(i, gpu);
    }
    if (vw <= 0 || vh <= 0) return 0;
    D3D12_VIEWPORT vp = {(float)vx, (float)vy, (float)vw, (float)vh,
                         0.0f, 1.0f};
    cl->RSSetViewports(1, &vp);
    D3D12_RECT rct;
    if (use_scissor) {
        // 有符号 + 裁到目标范围：viewport/scissor 原为 uint32，负坐标
        // （光标移出屏幕上方/左侧时的 y<0）会被当成 42 亿 → 整只画到
        // 屏幕外（"鼠标到顶上就消失"的真因）。越界部分应被裁掉、剩余照画；
        // 只有完全在目标外才跳过（唯一合法的"整只不画"）。
        LONG l = sx < 0 ? 0 : (LONG)sx;
        LONG t = sy < 0 ? 0 : (LONG)sy;
        LONG r = (LONG)(sx + sw);
        LONG b = (LONG)(sy + sh);
        if (r > (LONG)dst_w) r = (LONG)dst_w;
        if (b > (LONG)dst_h) b = (LONG)dst_h;
        if (r <= l || b <= t) return 0;
        rct = {l, t, r, b};
        cl->RSSetScissorRects(1, &rct);
    } else {
        rct = {0, 0, (LONG)dst_w, (LONG)dst_h};
        cl->RSSetScissorRects(1, &rct);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = {(SIZE_T)rtv_handle};
    cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cl->DrawInstanced(3, 1, 0, 0);
    return 0;
}

// 清渲染目标（viewer 的 ctx.clear 对应物）。draw_pass 只覆盖自己的视口，
// TAB/深度图这类带黑边的模式如果不先清，视口外会残留 3 缓冲里的旧帧。
// 调用方保证 RT 已 transition 到 RENDER_TARGET。
__declspec(dllexport)
int d3dr_clear_rtv(RenderCtx* s, void* rtv_handle,
                   float r, float g, float b, float a) {
    if (!s || !s->cl[s->frame_i] || !rtv_handle) return -1;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = {(SIZE_T)rtv_handle};
    const float rgba[4] = {r, g, b, a};
    s->cl[s->frame_i]->ClearRenderTargetView(rtv, rgba, 0, NULL);
    return 0;
}

__declspec(dllexport)
int d3dr_submit(RenderCtx* s) {
    if (!s || !s->cl[s->frame_i]) return -1;
    HRESULT hr = s->cl[s->frame_i]->Close();
    s->frame_open = false;
    if (FAILED(hr)) {
        char buf[64];
        sprintf_s(buf, sizeof(buf), "cl->Close failed hr=%08x", (unsigned)hr);
        set_err(s, buf);
        drain_infoq(s);
        return (int)hr;
    }
    ID3D12CommandList* lists[] = {s->cl[s->frame_i]};
    s->queue->ExecuteCommandLists(1, lists);
    s->fence_val++;
    s->queue->Signal(s->fence, s->fence_val);
    s->alloc_done[s->frame_i] = s->fence_val;   // 组级完成对账（流水线）
    return 0;
}

__declspec(dllexport)
int d3dr_wait(RenderCtx* s, uint32_t timeout_ms) {
    if (!s || !s->fence) return -1;
    UINT64 cur = s->fence->GetCompletedValue();
    if (cur < s->fence_val) {
        s->fence->SetEventOnCompletion(s->fence_val, s->wait_event);
        DWORD r = WaitForSingleObject(s->wait_event, timeout_ms);
        if (r != WAIT_OBJECT_0) return (int)r;
    }
    return 0;
}

// ---- Swapchain -----------------------------------------------------------
__declspec(dllexport)
int d3dr_create_swapchain(RenderCtx* s, void* hwnd,
                          uint32_t w, uint32_t h, int use_flip) {
    if (!s || !s->queue) return -1;
    IDXGIFactory1* fct1 = NULL;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fct1)))
        return (int)0x887A0001;
    IDXGIFactory2* fct = NULL;
    fct1->QueryInterface(__uuidof(IDXGIFactory2), (void**)&fct);
    fct1->Release();
    if (!fct) return (int)0x80004002;
    DXGI_SWAP_CHAIN_DESC1 d1 = {};
    d1.Width = w; d1.Height = h;
    d1.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    d1.SampleDesc.Count = 1;
    d1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    d1.BufferCount = SWAP_N;
    d1.Scaling = (use_flip) ? DXGI_SCALING_NONE : DXGI_SCALING_STRETCH;
    d1.SwapEffect = (use_flip) ? DXGI_SWAP_EFFECT_FLIP_DISCARD : DXGI_SWAP_EFFECT_DISCARD;
    d1.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    // frame-latency waitable：flip 完成的确定性原语（destroy/resize 前
    // 排空 in-flight flip；Present 竞态 sleep 是不可靠的）
    d1.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    // ---- DirectComposition 分支（D2S_DCOMP=1）----------------------------
    // 目的：彻底去掉"layered alpha=254"这个补丁。
    // 交换链声明 PREMULTIPLIED + 全屏 alpha=1 → 视觉上完全不透明；但 DWM
    // 认为该平面"可能带透明"，只能走合成 → 不会被提升为 MPO 独立翻转 →
    // 运动时的滚动横缝（flip 缓冲被扫描输出直读）根因消失。
    // 同时"半透明平面"会迫使 DWM 合成下方桌面 → 采集不会全黑。
    char _dc[8] = {0};
    bool _use_dcomp = GetEnvironmentVariableA("D2S_DCOMP", _dc, 8) > 0
                      && _dc[0] == '1';
    HRESULT hr;
    if (_use_dcomp) {
        // ID3D12Device 不实现 IDXGIDevice（DComp 的老 API 要它）→ 在同一块
        // adapter 上建一个辅助 D3D11 设备，只为取 IDXGIDevice 喂给 DComp。
        // 辅助设备不参与任何渲染，创建一次、用完即放（显存/CPU 开销可忽略）。
        LUID lu = s->dev->GetAdapterLuid();
        IDXGIFactory1* af1 = NULL;
        IDXGIAdapter* ad = NULL;
        if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                        (void**)&af1)) && af1) {
            for (UINT i = 0;; ++i) {
                IDXGIAdapter* cand = NULL;
                if (af1->EnumAdapters(i, &cand) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC dd = {};
                cand->GetDesc(&dd);
                if (dd.AdapterLuid.LowPart == lu.LowPart &&
                    dd.AdapterLuid.HighPart == lu.HighPart) {
                    ad = cand;
                    break;
                }
                cand->Release();
            }
            af1->Release();
        }
        ID3D11Device* d11 = NULL;
        ID3D11DeviceContext* c11 = NULL;
        D3D_FEATURE_LEVEL fl11 = D3D_FEATURE_LEVEL_11_0;
        IDXGIDevice* dxdev = NULL;
        HRESULT h0 = ad ? D3D11CreateDevice(ad, D3D_DRIVER_TYPE_UNKNOWN, NULL,
                                            0, NULL, 0, D3D11_SDK_VERSION,
                                            &d11, &fl11, &c11)
                        : E_FAIL;
        if (ad) { ad->Release(); ad = NULL; }
        if (FAILED(h0) || !d11) {
            fct->Release();
            return (int)(FAILED(h0) ? h0 : 0x80004001);
        }
        d11->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxdev);
        IDCompositionDevice* dcd = NULL;
        HRESULT hc = dxdev ? DCompositionCreateDevice(
                                 dxdev, __uuidof(IDCompositionDevice),
                                 (void**)&dcd)
                           : E_NOINTERFACE;
        if (dxdev) dxdev->Release();
        // 辅助 D3D11 设备只为这一下；拿到 DComp 设备后即可释放
        c11->Release();
        d11->Release();
        if (FAILED(hc) || !dcd) {
            fct->Release();
            return (int)(FAILED(hc) ? hc : 0x80004001);
        }
        HRESULT h1 = dcd->CreateTargetForHwnd((HWND)hwnd, TRUE,
                                              &s->dcomp_target);
        if (FAILED(h1)) {
            dcd->Release(); fct->Release();
            return (int)h1;              // 真实 hr（便于定位）
        }
        HRESULT h2 = dcd->CreateVisual(&s->dcomp_visual);
        if (FAILED(h2)) {
            dcd->Release(); fct->Release();
            return (int)h2;
        }
        d1.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
        d1.Scaling = DXGI_SCALING_STRETCH;   // Composition 链只支持 STRETCH
        d1.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        hr = fct->CreateSwapChainForComposition(s->queue, &d1, NULL, &s->sc);
        if (SUCCEEDED(hr)) {
            hr = s->dcomp_visual->SetContent(s->sc);
            if (SUCCEEDED(hr)) {
                hr = s->dcomp_target->SetRoot(s->dcomp_visual);
            }
            if (SUCCEEDED(hr)) {
                hr = dcd->Commit();
            }
        }
        s->dcomp = dcd;      // 保留到 destroy
        fct->Release();
        if (FAILED(hr)) return (int)hr;
    } else {
        hr = fct->CreateSwapChainForHwnd(s->queue, (HWND)hwnd, &d1, NULL, NULL, &s->sc);
        fct->Release();
        if (FAILED(hr)) return (int)hr;
    }
    if (SUCCEEDED(s->sc->QueryInterface(__uuidof(IDXGISwapChain2), (void**)&s->sc2))) {
        // 合成路径（layered alpha<255）下 1 会让 Present 卡等合成，
        // 合成偶尔超时就把帧率压到 55；2 允许排一帧、Present 立即返回
        s->sc2->SetMaximumFrameLatency(2);
        s->sc_waitable = s->sc2->GetFrameLatencyWaitableObject();
    }
    s->sc->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&s->sc3);
    s->sc_w = w; s->sc_h = h;
    // backbuffer RTV
    for (UINT i = 0; i < SWAP_N; ++i) {
        ID3D12Resource* bb = NULL;
        if (FAILED(s->sc->GetBuffer(i, __uuidof(ID3D12Resource), (void**)&bb))) return -2;
        D3D12_RENDER_TARGET_VIEW_DESC rd = {};
        rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE base = s->rtv_heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE h = {base.ptr + (SIZE_T)(s->rtv_next + i) * s->rtv_inc};
        s->dev->CreateRenderTargetView(bb, &rd, h);
        s->backbuf[i] = bb;
        s->bb_rtv[i] = h;
    }
    s->rtv_next += SWAP_N;
    s->sc_idx = 0;
    return 0;
}

// 帧呈现一条龙：backbuffer 屏障（RT→COMMON）+ 命令列表提交 + Present。
// 约定：调用方在 draw_pass 后【不得】手动 submit——屏障必须随本帧列表
// 提交（先 submit 再录屏障会丢序，flip 早于屏障执行）。
// flags 透传给 IDXGISwapChain::Present：传 DXGI_PRESENT_DO_NOT_WAIT(0x400)
// 时队列满不再阻塞 CPU，而是返回 DXGI_ERROR_WAS_STILL_DRAWING(0x887A000A)
// —— 调用方按"本拍跳过"处理（内容已过时，不追）。
__declspec(dllexport)
int d3dr_present_flags(RenderCtx* s, int sync_interval, unsigned flags) {
    if (!s || !s->sc) return -1;
    UINT cur = cur_bb(s);
    if (s->bb_state[cur] != D3D12_RESOURCE_STATE_COMMON) {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = s->backbuf[cur];
        b.Transition.StateBefore = s->bb_state[cur];
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s->cl[s->frame_i]->ResourceBarrier(1, &b);
        s->bb_state[cur] = D3D12_RESOURCE_STATE_COMMON;
    }
    HRESULT hr = s->cl[s->frame_i]->Close();
    s->frame_open = false;
    if (FAILED(hr)) {
        char buf[64];
        sprintf_s(buf, sizeof(buf), "present cl->Close failed hr=%08x",
                  (unsigned)hr);
        set_err(s, buf);
        drain_infoq(s);
        return (int)hr;
    }
    ID3D12CommandList* lists[] = {s->cl[s->frame_i]};
    s->queue->ExecuteCommandLists(1, lists);
    s->fence_val++;
    s->queue->Signal(s->fence, s->fence_val);
    s->alloc_done[s->frame_i] = s->fence_val;
    // per-frame allocator 轮换：下一帧 begin_frame 切组并对账——present
    // 内不再无条件等 GPU（流水线关键；allocator 竞争由组级对账消除）
    // FLIP 模型 Present 后 backbuffer 隐式回 COMMON
    s->sc_idx = (s->sc_idx + 1) % SWAP_N;
    HRESULT pr = s->sc->Present((UINT)sync_interval, (UINT)flags);
    if (pr == DXGI_ERROR_WAS_STILL_DRAWING) {
        return (int)pr;          // 非阻塞呈现：本拍未上屏，调用方跳过即可
    }
    if (FAILED(pr)) {
        char buf[96];
        sprintf_s(buf, sizeof(buf), "Present failed hr=%08x", (unsigned)pr);
        set_err(s, buf);
        drain_infoq(s);
        if (pr == 0x887A0005 || pr == 0x887A0006 || pr == 0x887A0007) {
            HRESULT rr = s->dev->GetDeviceRemovedReason();
            char buf2[80];
            sprintf_s(buf2, sizeof(buf2), " removed_reason=%08x",
                      (unsigned)rr);
            strncat_s(s->err, buf2, _TRUNCATE);
            drain_infoq(s);
        }
    }
    return (int)pr;
}

__declspec(dllexport)
int d3dr_present(RenderCtx* s, int sync_interval) {
    return d3dr_present_flags(s, sync_interval, 0);
}

// 内部 fence 查询（非阻塞）：流水线里 Python 用它判断"某一组提交完成没有"。
// 只读 GetCompletedValue，不等待。
__declspec(dllexport)
unsigned long long d3dr_completed_fence(RenderCtx* s) {
    if (!s || !s->fence) return 0;
    return (unsigned long long)s->fence->GetCompletedValue();
}

// 最近一次提交分配到的 fence 值（与 d3dr_completed_fence 比较）
__declspec(dllexport)
unsigned long long d3dr_last_fence(RenderCtx* s) {
    if (!s) return 0;
    return (unsigned long long)s->fence_val;
}

__declspec(dllexport)
void* d3dr_get_backbuffer_rtv(RenderCtx* s) {
    if (!s || !s->sc) return NULL;
    UINT idx = cur_bb(s);
    return (void*)(uintptr_t)s->bb_rtv[idx].ptr;
}

// 反向 fence signal：消费者（D3D12 渲染队列）在帧末 signal 一张共享
// fence，采集侧（D3D11）写某环槽前 Wait 到该槽最后的值（槽回收）。
__declspec(dllexport)
int d3dr_signal_fence(RenderCtx* s, void* fence, uint64_t value) {
    if (!s || !s->queue || !fence) return -1;
    HRESULT hr = s->queue->Signal((ID3D12Fence*)fence, (UINT64)value);
    return (int)hr;
}

__declspec(dllexport)
void* d3dr_get_backbuffer_tex(RenderCtx* s) {
    if (!s || !s->sc) return NULL;
    return s->backbuf[cur_bb(s)];
}

// 当前后缓冲序号（探针用）：画面"回退"时对照日志里的 bb 值
__declspec(dllexport)
unsigned d3dr_cur_bb(RenderCtx* s) {
    return s ? cur_bb(s) : 0xFFFFFFFFu;
}

__declspec(dllexport)
// 与 d3dr_alloc_srv 等价（别名，保留兼容旧调用点）
uint64_t d3dr_get_srv_handle(RenderCtx* s, void* tex, uint32_t fmt) {
    return d3dr_alloc_srv(s, tex, fmt);
}

// tex → tex 整资源拷贝（CopyResource；两纹理格式/尺寸须一致；状态由调用方
// 显式 transition：src→COPY_SOURCE，dst→COPY_DEST）
__declspec(dllexport)
int d3dr_copy_tex_to_tex(RenderCtx* s, void* dst_tex, void* src_tex) {
    if (!s || !s->cl) return -1;
    s->cl[s->frame_i]->CopyResource((ID3D12Resource*)dst_tex, (ID3D12Resource*)src_tex);
    return 0;
}

// 显式资源屏障（RENDER_TARGET/COPY_DEST 等写状态不能从 COMMON 隐式提升，
// 由 Python 侧状态跟踪在录制序列中显式调用）
__declspec(dllexport)
int d3dr_transition(RenderCtx* s, void* tex, uint32_t from_state, uint32_t to_state) {
    if (!s || !s->cl || !tex) return -1;
    if (from_state == to_state) return 0;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = (ID3D12Resource*)tex;
    b.Transition.StateBefore = (D3D12_RESOURCE_STATES)from_state;
    b.Transition.StateAfter = (D3D12_RESOURCE_STATES)to_state;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s->cl[s->frame_i]->ResourceBarrier(1, &b);
    for (int q = 0; q < SWAP_N; ++q)
        if (s->backbuf[q] == (ID3D12Resource*)tex)
            s->bb_state[q] = (D3D12_RESOURCE_STATES)to_state;
    return 0;
}

// swapchain 重建式 resize：先销毁（GPU idle 由调用方保证——本导出内部
// 只清引用）再由 create_swapchain 重建。ResizeBuffers 在本环境的
// flip+隐藏/可见窗口上均 access violation（WinError 2173），绕开。
// debug probe: sc pointer sanity (remove after dangling-ptr case closes)
__declspec(dllexport)
uint64_t d3dr_get_sc_ptr(RenderCtx* s) {
    return s ? (uint64_t)(uintptr_t)s->sc : 0;
}

// 后缓冲索引探针已移除（2026-11-22：横线撕裂经确认为 2D 视频自带，
// 探针使命结束；渲染器自记 sc_idx 与 DXGI GetCurrentBackBufferIndex
// 实测 594 帧 0 次错位）。


// debug probe: descriptor slot counters (low 32 = rtv_next, high 32 = arena_next)
__declspec(dllexport)
uint64_t d3dr_get_desc_counts(RenderCtx* s) {
    if (!s) return 0;
    return ((uint64_t)s->arena_next << 32) | (uint64_t)s->rtv_next;
}

// ---- readback 专用列表（旁路 per-frame 轮换：不打乱帧级 allocator 对账）----
__declspec(dllexport)
int d3dr_rb_begin(RenderCtx* s) {
    if (!s || !s->rb_cl) return -1;
    if (FAILED(s->rb_alloc->Reset())) return -2;   // allocator 必须先 Reset
    if (FAILED(s->rb_cl->Reset(s->rb_alloc, NULL))) return -2;
    return 0;
}

__declspec(dllexport)
int d3dr_rb_barrier(RenderCtx* s, void* tex, uint32_t from_state,
                    uint32_t to_state) {
    if (!s || !s->rb_cl || !tex || from_state == to_state) return 0;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = (ID3D12Resource*)tex;
    b.Transition.StateBefore = (D3D12_RESOURCE_STATES)from_state;
    b.Transition.StateAfter = (D3D12_RESOURCE_STATES)to_state;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s->rb_cl->ResourceBarrier(1, &b);
    return 0;
}

__declspec(dllexport)
int d3dr_rb_copy_tex_to_buf(RenderCtx* s, void* src_tex, void* dst_buf,
                            uint32_t w, uint32_t h, uint32_t fmt,
                            uint32_t row_pitch) {
    if (!s || !s->rb_cl) return -1;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = (ID3D12Resource*)src_tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = (ID3D12Resource*)dst_buf;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = (DXGI_FORMAT)fmt;
    dst.PlacedFootprint.Footprint.Width = w;
    dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = row_pitch;
    s->rb_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, NULL);
    return 0;
}

__declspec(dllexport)
int d3dr_rb_exec(RenderCtx* s) {
    if (!s || !s->rb_cl) return -1;
    HRESULT hr = s->rb_cl->Close();
    if (FAILED(hr)) return (int)hr;
    ID3D12CommandList* lists[] = {s->rb_cl};
    s->queue->ExecuteCommandLists(1, lists);
    s->fence_val++;
    s->queue->Signal(s->fence, s->fence_val);
    // rb_alloc 单独：下次 Reset 前由调用方 d3dr_wait 全局对账（readback 语义）
    return 0;
}

// 裸设备/队列指针（shim v2 DML 注入用；COM 引用计数不增——调用方不 Release）
__declspec(dllexport)
uint64_t d3dr_get_dev(RenderCtx* s) {
    return s ? (uint64_t)(uintptr_t)s->dev : 0;
}

__declspec(dllexport)
uint64_t d3dr_get_queue(RenderCtx* s) {
    return s ? (uint64_t)(uintptr_t)s->queue : 0;
}

// ---- compute（V2 GPU 判据：直方图/分位/cut） -----------------------------
// 根签名：slot0..3 = UAV 表（u0..u3 各一槽），slot4..7 = SRV 表（t0..t3
// 各一槽），slot8 = CBV(b0)。
// 每槽独立表（与图形根签名同构）：描述符在 arena 里按需分配，**不保证
// 连续**（z 乒乓交换后 [cur,prev] 的槽位次序反转——4 宽连续表曾因此把 t1
// 绑到无关描述符，cut 比例恒错）；独立槽位只依赖句柄本身。
static ID3D12RootSignature* g_cs_rs = NULL;
static ID3D12RootSignature* get_cs_root_sig(RenderCtx* s) {
    if (g_cs_rs) return g_cs_rs;
    D3D12_DESCRIPTOR_RANGE ranges[8] = {};
    D3D12_ROOT_PARAMETER params[9] = {};
    for (UINT i = 0; i < 4; ++i) {          // UAV u0..u3 -> root slot i
        ranges[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[i].NumDescriptors = 1;
        ranges[i].BaseShaderRegister = i;
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[i].DescriptorTable.NumDescriptorRanges = 1;
        params[i].DescriptorTable.pDescriptorRanges = &ranges[i];
    }
    for (UINT i = 0; i < 4; ++i) {          // SRV t0..t3 -> root slot 4+i
        ranges[4 + i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[4 + i].NumDescriptors = 1;
        ranges[4 + i].BaseShaderRegister = i;
        params[4 + i].ParameterType =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[4 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[4 + i].DescriptorTable.NumDescriptorRanges = 1;
        params[4 + i].DescriptorTable.pDescriptorRanges = &ranges[4 + i];
    }
    params[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[8].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[8].Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsd = {9, params, 0, NULL,
                                     D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob* blob = NULL;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &blob, NULL))) {
        set_err(s, "cs rootsig serialize failed");
        return NULL;
    }
    HRESULT hr = s->dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                             blob->GetBufferSize(),
                                             __uuidof(ID3D12RootSignature),
                                             (void**)&g_cs_rs);
    blob->Release();
    if (FAILED(hr)) { set_err(s, "cs rootsig create failed"); g_cs_rs = NULL; }
    return g_cs_rs;
}

__declspec(dllexport)
void* d3dr_create_cs_pso(RenderCtx* s, const char* name,
                         const uint8_t* cs_data, uint32_t cs_len) {
    if (!s || !s->dev) return NULL;
    ID3D12RootSignature* rs = get_cs_root_sig(s);
    if (!rs) return NULL;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = rs;
    pd.CS.pShaderBytecode = cs_data;
    pd.CS.BytecodeLength = cs_len;
    ID3D12PipelineState* pso = NULL;
    HRESULT hr = s->dev->CreateComputePipelineState(
        &pd, __uuidof(ID3D12PipelineState), (void**)&pso);
    if (FAILED(hr)) {
        char buf[64];
        sprintf_s(buf, sizeof(buf), "CreateCSPSO failed hr=%08x",
                  (unsigned)hr);
        set_err(s, buf);
        return NULL;
    }
    (void)name;
    return pso;
}

// UAV 目标 buffer（DEFAULT heap + ALLOW_UNORDERED_ACCESS）
__declspec(dllexport)
void* d3dr_create_uav_buf(RenderCtx* s, uint32_t nbytes) {
    if (!s || !s->dev) return NULL;
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = nbytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* res = NULL;
    if (FAILED(s->dev->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, NULL,
            __uuidof(ID3D12Resource), (void**)&res))) {
        set_err(s, "CreateCommittedResource(uav buf) failed");
        return NULL;
    }
    return res;
}

// raw UAV 描述符（arena 固定槽；返回 GPU 句柄）。nbytes = 总字节数
__declspec(dllexport)
uint64_t d3dr_alloc_uav(RenderCtx* s, void* buf, uint32_t nbytes) {
    if (!s || !s->arena || !buf) return 0;
    D3D12_CPU_DESCRIPTOR_HANDLE h = {s->arena_base_cpu +
                                     (UINT64)s->arena_next * s->srv_inc};
    D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements = nbytes / 4;
    // structured (uint) 视图：此 runtime 拒收 raw 视图（stride=0 + UNKNOWN
    // → DXGI_ERROR_INVALID_CALL 打掉设备，selftest v0/v1/v4 实锤）；
    // stride=4 结构化视图 UAV/SRV 均健康（v2/v7）
    d.Buffer.StructureByteStride = 4;
    d.Buffer.CounterOffsetInBytes = 0;
    if (s->arena_next >= 4096) { set_err(s, "uav: arena full"); return 0; }
    // 注意：CreateUnorderedAccessView 返回 void（D3D12 视图创建无 hr）；
    // 非法参数会把设备打成 invalid-call 态，靠 d3dr_dev_removed_reason 探针
    s->dev->CreateUnorderedAccessView((ID3D12Resource*)buf, NULL, &d, h);
    s->arena_next++;
    return s->arena_base_gpu + (UINT64)(s->arena_next - 1) * s->srv_inc;
}

// raw SRV 描述符（ByteAddressBuffer，arena 固定槽；返回 GPU 句柄）
__declspec(dllexport)
uint64_t d3dr_alloc_srv_buf(RenderCtx* s, void* buf, uint32_t nbytes) {
    if (!s || !s->arena || !buf) return 0;
    D3D12_CPU_DESCRIPTOR_HANDLE h = {s->arena_base_cpu +
                                     (UINT64)s->arena_next * s->srv_inc};
    D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.FirstElement = 0;
    d.Buffer.NumElements = nbytes / 4;
    d.Buffer.StructureByteStride = 4;   // structured（raw 视图被 runtime 拒收）
    s->dev->CreateShaderResourceView((ID3D12Resource*)buf, &d, h);
    s->arena_next++;
    return s->arena_base_gpu + (UINT64)(s->arena_next - 1) * s->srv_inc;
}

// buffer→buffer 拷贝（CopyBufferRegion；readback 用，须录进 rb 列表）
__declspec(dllexport)
int d3dr_rb_copy_buf_to_buf(RenderCtx* s, void* src_buf, void* dst_buf,
                            uint32_t nbytes) {
    if (!s || !s->rb_cl) return -1;
    s->rb_cl->CopyBufferRegion((ID3D12Resource*)dst_buf, 0,
                               (ID3D12Resource*)src_buf, 0, nbytes);
    return 0;
}

// UAV 之间的执行屏障（同一资源前后两个 dispatch 读写依赖时必须——
// 仅仅是状态转换覆盖不到"UAV→UAV 读写同资源"这一档）
__declspec(dllexport)
int d3dr_uav_barrier(RenderCtx* s, void* res) {
    if (!s || !s->cl[s->frame_i] || !res) return -1;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.UAV.pResource = (ID3D12Resource*)res;
    s->cl[s->frame_i]->ResourceBarrier(1, &b);
    return 0;
}

// compute dispatch（录进当前帧列表；UAV 状态由调用方 transition 管理）
__declspec(dllexport)
int d3dr_dispatch(RenderCtx* s, void* cs_pso,
                  uint64_t* uav_handles, uint32_t n_uav,
                  uint64_t* srv_handles, uint32_t n_srv,
                  const float* cb_data, uint32_t cb_floats,
                  uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!s || !s->cl[s->frame_i] || !cs_pso) return -1;
    auto* cl = s->cl[s->frame_i];
    ID3D12DescriptorHeap* heaps[] = {s->arena};
    cl->SetDescriptorHeaps(1, heaps);
    // cbuffer 槽（复用同一 upload 环形）
    s->cb_next = (s->cb_next + 1) % RenderCtx::CB_SLOTS;
    UINT slot = s->cb_next;
    if (cb_data && cb_floats > 0) {
        UINT n = min(cb_floats, RenderCtx::CB_STRIDE);
        memcpy((uint8_t*)s->cb_map + slot * RenderCtx::CB_STRIDE * 4,
               cb_data, n * 4);
    }
    cl->SetPipelineState((ID3D12PipelineState*)cs_pso);
    cl->SetComputeRootSignature(get_cs_root_sig(s));
    cl->SetComputeRootConstantBufferView(
        8, s->cb_res->GetGPUVirtualAddress() +
               slot * RenderCtx::CB_STRIDE * 4);
    // 每槽独立表：srv[i] -> root slot 4+i，uav[i] -> root slot i
    for (UINT i = 0; i < n_srv && i < 4 && srv_handles; ++i) {
        D3D12_GPU_DESCRIPTOR_HANDLE g = {srv_handles[i]};
        cl->SetComputeRootDescriptorTable(4 + i, g);
    }
    for (UINT i = 0; i < n_uav && i < 4 && uav_handles; ++i) {
        D3D12_GPU_DESCRIPTOR_HANDLE g = {uav_handles[i]};
        cl->SetComputeRootDescriptorTable(i, g);
    }
    cl->Dispatch(gx, gy, gz);
    return 0;
}

// 等 flip 落地：waitable object（MaximumFrameLatency=1 下等一次即排空）
__declspec(dllexport)
int d3dr_wait_flip(RenderCtx* s, uint32_t timeout_ms) {
    if (!s || !s->sc_waitable) return -1;
    DWORD r = WaitForSingleObject(s->sc_waitable, timeout_ms);
    return (r == WAIT_OBJECT_0) ? 0 : (int)r;
}

__declspec(dllexport)
int d3dr_destroy_swapchain(RenderCtx* s) {
    if (!s || !s->sc) return -1;
    // 排空 in-flight flip（MaximumFrameLatency=1：waitable 触发即空闲）
    if (s->sc_waitable)
        WaitForSingleObject(s->sc_waitable, 2000);
    for (int q = 0; q < SWAP_N; ++q) {
        if (s->backbuf[q]) { s->backbuf[q]->Release(); s->backbuf[q] = NULL; }
        s->bb_state[q] = D3D12_RESOURCE_STATE_COMMON;
    }
    if (s->sc2) { s->sc2->Release(); s->sc2 = NULL; }
    if (s->sc3) { s->sc3->Release(); s->sc3 = NULL; }
    s->sc_waitable = NULL;
    __try {
        s->sc->Release();
        s->sc = NULL;
        s->sc_idx = 0;
        return 0;
    } __except (1) {
        return -200;   // SEH: sc->Release 访问违例（悬垂指针实锤标记）
    }
}

__declspec(dllexport)
int d3dr_resize_swapchain(RenderCtx* s, uint32_t w, uint32_t h) {
    if (!s || !s->sc) return -1;
    for (int q = 0; q < SWAP_N; ++q) {
        if (s->backbuf[q]) { s->backbuf[q]->Release(); s->backbuf[q] = NULL; }
    }
    HRESULT hr = s->sc->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) return (int)hr;
    s->sc_w = w; s->sc_h = h;
    for (UINT i = 0; i < SWAP_N; ++i) {
        ID3D12Resource* bb = NULL;
        if (FAILED(s->sc->GetBuffer(i, __uuidof(ID3D12Resource), (void**)&bb))) return -2;
        D3D12_RENDER_TARGET_VIEW_DESC rd = {};
        rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE base = s->rtv_heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE h = {base.ptr + (SIZE_T)(s->rtv_next + i) * s->rtv_inc};
        s->dev->CreateRenderTargetView(bb, &rd, h);
        s->backbuf[i] = bb;
        s->bb_rtv[i] = h;
        s->bb_state[i] = D3D12_RESOURCE_STATE_COMMON;   // 新 buffer 回 COMMON
    }
    s->rtv_next += SWAP_N;
    return 0;
}

} // extern "C"
