// NTHANDLE 创建最小复现（与 d2s_dml_shim.cpp 同参数）：裁决 C++ vs ctypes
// 构建：见 build_nthandle.cmd / 直接 cl 调用与 build_shim.sh 同款
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <initguid.h>
#include <dxgi1_2.h>
#include <d3d12.h>
#include <cstdio>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

int main() {
    // 枚举锚点 adapter（LUID 匹配由调用者保证；这里取第一个有输出的硬件卡）
    IDXGIFactory2* fct = NULL;
    CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fct);
    IDXGIAdapter1* adapter = NULL;
    for (UINT i = 0; fct->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.DedicatedVideoMemory > 0) break;
        adapter->Release(); adapter = NULL;
    }
    if (!adapter) { printf("no adapter\n"); return 1; }
    DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
    printf("adapter LUID: %08x%08x\n", d.AdapterLuid.HighPart, d.AdapterLuid.LowPart);

    ID3D11Device* dev = NULL;
    ID3D11DeviceContext* ctx = NULL;
    HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL,
                                   D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0,
                                   D3D11_SDK_VERSION, &dev, NULL, &ctx);
    if (FAILED(hr)) { printf("D3D11CreateDevice failed %08x\n", hr); return 1; }
    D3D11_FEATURE_DATA_D3D11_OPTIONS opt = {0};
    if (SUCCEEDED(dev->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &opt, sizeof(opt)))) {
        printf("d3d11 options ok\n");
    }

    D3D11_TEXTURE2D_DESC td = {0};
    td.Width = 1920; td.Height = 1080; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    ID3D11Texture2D* tex = NULL;
    { const unsigned char* pb = (const unsigned char*)&td;
      printf("td hex: "); for (size_t q = 0; q < sizeof(td); ++q) printf("%02x", pb[q]);
      printf("\n"); }
    hr = dev->CreateTexture2D(&td, NULL, &tex);
    printf("NTHANDLE CreateTexture2D: %08x %s\n", hr, SUCCEEDED(hr) ? "OK" : "(FAIL)");
    if (SUCCEEDED(hr)) {
        IDXGIResource1* r1 = NULL;
        if (SUCCEEDED(tex->QueryInterface(__uuidof(IDXGIResource1), (void**)&r1))) {
            const DWORD accs[] = {
                DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                GENERIC_ALL, 0};
            for (int q = 0; q < 3; ++q) {
                HANDLE h = NULL;
                HRESULT h2 = r1->CreateSharedHandle(NULL, accs[q], NULL, &h);
                printf("tex CreateSharedHandle(acc=%08x): %08x %s\n", accs[q], h2,
                       SUCCEEDED(h2) ? "OK" : "(FAIL)");
                if (SUCCEEDED(h2) && h) CloseHandle(h);
            }
            r1->Release();
        } else {
            printf("IDXGIResource1 QI: FAIL\n");
        }
        tex->Release();
    }
    // 对照：旧 SHARED
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    hr = dev->CreateTexture2D(&td, NULL, &tex);
    printf("legacy SHARED: %08x %s\n", hr, SUCCEEDED(hr) ? "OK" : "(FAIL)");
    if (tex) tex->Release();
    // 对照：None 适配器 + HARDWARE + NULL FL（= rtcom create_device 等价路径）
    {
        ID3D11Device* dev2 = NULL; ID3D11DeviceContext* ctx2 = NULL;
        HRESULT hr2 = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL,
                                        D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0,
                                        D3D11_SDK_VERSION, &dev2, NULL, &ctx2);
        printf("None+HARDWARE 设备: %08x\n", hr2);
        if (SUCCEEDED(hr2)) {
            D3D11_TEXTURE2D_DESC t2 = td;
            t2.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
            ID3D11Texture2D* tex2 = NULL;
            HRESULT hr3 = dev2->CreateTexture2D(&t2, NULL, &tex2);
            printf("  NTHANDLE on None-device: %08x %s\n", hr3,
                   SUCCEEDED(hr3) ? "OK" : "(FAIL)");
            if (tex2) tex2->Release();
        }
        if (dev2) dev2->Release();
        if (ctx2) ctx2->Release();
    }
    // ---- root signature 序列化（Python 引用 blob） ----
    {
        D3D12_DESCRIPTOR_RANGE ranges[1] = {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 8;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        D3D12_ROOT_PARAMETER rparams[2] = {};
        rparams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rparams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rparams[0].DescriptorTable.NumDescriptorRanges = 1;
        rparams[0].DescriptorTable.pDescriptorRanges = ranges;
        rparams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rparams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rparams[1].Descriptor.ShaderRegister = 0;
        rparams[1].Descriptor.RegisterSpace = 0;
        D3D12_STATIC_SAMPLER_DESC rsamp[2] = {};
        rsamp[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        rsamp[0].AddressU = rsamp[0].AddressV = rsamp[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        rsamp[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        rsamp[0].ShaderRegister = 0;
        rsamp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rsamp[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        rsamp[1].AddressU = rsamp[1].AddressV = rsamp[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        rsamp[1].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        rsamp[1].ShaderRegister = 1;
        rsamp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rsd = {2, rparams, 2, rsamp, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ID3DBlob* rblob = NULL; ID3DBlob* rerr = NULL;
        HRESULT h3 = D3D12SerializeRootSignature(&rsd, (D3D_ROOT_SIGNATURE_VERSION)1, &rblob, &rerr);
        if (SUCCEEDED(h3) && rblob) {
            printf("rootsig blob: ");
            const unsigned char* rp = (const unsigned char*)rblob->GetBufferPointer();
            for (SIZE_T q = 0; q < rblob->GetBufferSize(); ++q) printf("%02x", rp[q]);
            printf("\n(%zu bytes)\n", rblob->GetBufferSize());
            rblob->Release();
        } else {
            printf("rootsig serialize FAIL: %08x\n", h3);
        }
        if (rerr) rerr->Release();
    }
    // ---- swapchain 探针（flip/blt × plain 窗口，D3D12 队列设备） ----
    {
        WNDCLASSW wc = {0};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(NULL);
        wc.lpszClassName = L"nhsprobe";
        RegisterClassW(&wc);
        HWND hwnd = CreateWindowExW(0, L"nhsprobe", L"", WS_OVERLAPPEDWINDOW,
                                    0, 0, 640, 480, NULL, NULL, wc.hInstance, NULL);
        ShowWindow(hwnd, SW_SHOW);
        // D3D12 设备 + 直连队列（与 Python 侧同构）
        ID3D12Device* dev12 = NULL;
        if (FAILED(D3D12CreateDevice(adapter, (D3D_FEATURE_LEVEL)0xB000,
                                     __uuidof(ID3D12Device), (void**)&dev12))) {
            printf("D3D12CreateDevice FAIL\n");
        } else {
            D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
            ID3D12CommandQueue* q = NULL;
            dev12->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&q);
            const int effs[] = {4, 0};
            for (int ei = 0; ei < 2; ++ei) { int eff = effs[ei];   // 4=FLIP_DISCARD, 0=DISCARD(BLT)
                DXGI_SWAP_CHAIN_DESC1 d1 = {0};
                d1.Width = 640; d1.Height = 480;
                d1.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                d1.SampleDesc.Count = 1;
                d1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
                d1.BufferCount = 2;
                d1.Scaling = (eff == 4) ? DXGI_SCALING_NONE : DXGI_SCALING_STRETCH;
                d1.SwapEffect = (DXGI_SWAP_EFFECT)eff;
                d1.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
                IDXGISwapChain1* sc = NULL;
                HRESULT h2 = fct->CreateSwapChainForHwnd(q, hwnd, &d1, NULL, NULL, &sc);
                printf("swapchain eff=%d: %08x %s\n", eff, h2,
                       SUCCEEDED(h2) ? "OK" : "(FAIL)");
                if (sc) { sc->Release(); sc = NULL; }
            }
            if (q) q->Release();
            dev12->Release();
        }
        DestroyWindow(hwnd);
    }
    // ---- PSO 探针（D3D12 设备上 FXC SM5_0 编译 → PSO 创建） ----
    printf("[PSO probe start]\n");
    {
        ID3DBlob* vsb = NULL; ID3DBlob* psb = NULL; ID3DBlob* perr = NULL;
        D3DCompile(
            "struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
            "VOut VSMain(uint vid : SV_VertexID) { VOut o;\n"
            "float2 p = float2((vid==1)?3.0:-1.0, (vid==2)?-3.0:1.0);\n"
            "o.pos = float4(p, 0.0, 1.0); o.uv = float2((p.x+1)*0.5, (1-p.y)*0.5);\n"
            "return o; }\n",
            0, NULL, NULL, NULL, "VSMain", "vs_5_0", 0, 0, &vsb, &perr);
        D3DCompile(
            "float4 PSMain() : SV_Target { return float4(0,1,0,1); }\n",
            0, NULL, NULL, NULL, "PSMain", "ps_5_0", 0, 0, &psb, &perr);
        printf("FXC: vs=%s ps=%s\n",
               vsb ? "OK" : "FAIL", psb ? "OK" : "FAIL");
        // D3D12 设备（与 Python 同构）
        ID3D12Device* dv12 = NULL;
        if (FAILED(D3D12CreateDevice(adapter, (D3D_FEATURE_LEVEL)0xB000,
                                     __uuidof(ID3D12Device), (void**)&dv12))) {
            printf("D3D12 dev FAIL\n");
        } else {
            D3D12_ROOT_PARAMETER rp2[2] = {};
            rp2[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rp2[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            rp2[0].Descriptor.ShaderRegister = 0;
            D3D12_STATIC_SAMPLER_DESC ss[1] = {};
            ss[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            ss[0].AddressU = ss[0].AddressV = ss[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            ss[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            ss[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_SIGNATURE_DESC rsd2 = {1, rp2, 1, ss, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            ID3DBlob* rsb = NULL;
            if (FAILED(D3D12SerializeRootSignature(&rsd2, (D3D_ROOT_SIGNATURE_VERSION)1, &rsb, NULL))) {
                printf("RS serialize FAIL\n");
            } else {
                ID3D12RootSignature* rs12 = NULL;
                dv12->CreateRootSignature(0, rsb->GetBufferPointer(),
                                          rsb->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature), (void**)&rs12);
                printf("RS create: %s\n", rs12 ? "OK" : "FAIL");
                D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
                pd.pRootSignature = rs12;
                if (vsb) { pd.VS.pShaderBytecode = vsb->GetBufferPointer();
                           pd.VS.BytecodeLength = vsb->GetBufferSize(); }
                if (psb) { pd.PS.pShaderBytecode = psb->GetBufferPointer();
                           pd.PS.BytecodeLength = psb->GetBufferSize(); }
                pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
                pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pd.RasterizerState.DepthClipEnable = TRUE;
                pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                pd.NumRenderTargets = 1;
                pd.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
                pd.SampleDesc.Count = 1;
                ID3D12PipelineState* pso12 = NULL;
                HRESULT hp = dv12->CreateGraphicsPipelineState(&pd,
                            __uuidof(ID3D12PipelineState), (void**)&pso12);
                printf("PSO create: %08x %s\n", hp,
                       SUCCEEDED(hp) ? "OK" : "(FAIL)");
                if (pso12) pso12->Release();
                if (rs12) rs12->Release();
            }
            if (rsb) rsb->Release();
            dv12->Release();
        }
        if (vsb) vsb->Release();
        if (psb) psb->Release();
        if (perr) perr->Release();
    }
    dev->Release(); ctx->Release(); adapter->Release(); fct->Release();
    return 0;
}
