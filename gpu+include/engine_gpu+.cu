#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <SDL2/SDL.h>



static cudaGraphicsResource* g_cuda_tex_res = nullptr;
static GLuint g_gl_tex = 0;
static SDL_Window* g_win = nullptr;
static int g_img_w = 0, g_img_h = 0;

static float* g_d_quant = nullptr;
static int g_tile_sz = 8;

static cudaArray_t         g_atlas_array = nullptr;
static cudaTextureObject_t g_atlas_tex   = 0;
static int g_atlas_w = 0, g_atlas_h = 0;


__device__ __forceinline__ int d_sensitivity(int x, float gamma){
    float t = x / 10.0f;
    float num = powf(t, gamma);
    float den = num + powf(1.0f - t, gamma);
    if (den == 0.0f) return x;
    float val = 10.0f * num / den;
    int iv = (int)val;
    if (iv < 0) iv = 0; if (iv > 9) iv = 9;
    return iv;
}

__global__ void k_render_digits(cudaSurfaceObject_t surf,
                                const float* __restrict__ quant,
                                cudaTextureObject_t atlas,
                                int img_w, int img_h, int tile_sz, float gamma){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h) return;

    int w_d = img_w / tile_sz;
    int tx = x / tile_sz;
    int ty = y / tile_sz;
    int idx = ty * w_d + tx;

    float q10 = quant[idx];
    if (q10 < 0.f) q10 = 0.f;
    if (q10 > 10.f) q10 = 10.f;

    int digit = (int)q10;
    if (digit > 9) digit = 9;

    digit = d_sensitivity(digit, gamma);

    int lx = x % tile_sz;
    int ly = y % tile_sz;
    int srcX = digit * tile_sz + lx;
    int srcY = ly;

    uchar4 rgba = tex2D<uchar4>(atlas, (float)srcX + 0.5f, (float)srcY + 0.5f);

    surf2Dwrite(rgba, surf, x * sizeof(uchar4), y);
}


static void draw_fullscreen_quad(GLuint tex, SDL_Window* win){
    int dw, dh;
    SDL_GL_GetDrawableSize(win, &dw, &dh);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, dw, dh);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBegin(GL_TRIANGLE_STRIP);
    glTexCoord2f(0.f, 1.f); glVertex2f(-1.f, -1.f);
    glTexCoord2f(1.f, 1.f); glVertex2f( 1.f, -1.f);
    glTexCoord2f(0.f, 0.f); glVertex2f(-1.f,  1.f);
    glTexCoord2f(1.f, 0.f); glVertex2f( 1.f,  1.f);
    glEnd();
}


extern "C" int gpu_gl_init(SDL_Window* win, int img_w, int img_h){
    g_win = win;
    g_img_w = img_w;
    g_img_h = img_h;
    g_tile_sz = 8;

    glGenTextures(1, &g_gl_tex);
    glBindTexture(GL_TEXTURE_2D, g_gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, g_img_w, g_img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, g_gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    unsigned int count = 0;
    int devs[8] ={0};
    cudaError_t err = cudaGLGetDevices(&count, devs, 8, cudaGLDeviceListAll);
    err = cudaSetDevice(devs[0]);
    err = cudaGraphicsGLRegisterImage(  &g_cuda_tex_res, g_gl_tex, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsSurfaceLoadStore);
    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);

    int w_d = g_img_w / g_tile_sz;
    int h_d = g_img_h / g_tile_sz;
    size_t qbytes = (size_t)w_d * h_d * sizeof(float);
    cudaMalloc(&g_d_quant, qbytes);

    return 1;
}


extern "C" void gpu_gl_render_digits(   int img_w, 
                                        int img_h,
                                        int tile_sz,
                                        const float* d_quant,
                                        float gamma){
    if (!g_cuda_tex_res || !g_atlas_tex) return;


    cudaError_t err = cudaGraphicsMapResources(1, &g_cuda_tex_res, 0);

    cudaArray_t arr = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&arr, g_cuda_tex_res, 0, 0);
    if (err != cudaSuccess){
        cudaGraphicsUnmapResources(1, &g_cuda_tex_res, 0);
        return;
    }

    cudaResourceDesc rd{}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
    cudaSurfaceObject_t surf = 0;
    err = cudaCreateSurfaceObject(&surf, &rd);
    if (err != cudaSuccess){
        cudaGraphicsUnmapResources(1, &g_cuda_tex_res, 0);
        return;
    }

    dim3 block(16,16), grid((g_img_w+15)/16, (g_img_h+15)/16);
    k_render_digits<<<grid, block>>>(surf, d_quant, g_atlas_tex,
                                     g_img_w, g_img_h, g_tile_sz, gamma);


    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &g_cuda_tex_res, 0);

    draw_fullscreen_quad(g_gl_tex, g_win);
    SDL_GL_SwapWindow(g_win);
}



extern "C" void gpu_gl_shutdown(){
    if (g_d_quant){
        cudaFree(g_d_quant);
        g_d_quant = nullptr;
    }
    if (g_cuda_tex_res){
        cudaGraphicsUnregisterResource(g_cuda_tex_res);
        g_cuda_tex_res = nullptr;
    }
    if (g_gl_tex){
        glDeleteTextures(1, &g_gl_tex);
        g_gl_tex = 0;
    }
    g_win = nullptr;
}


__global__ void quantDownScaleKernel(   int w,
                                        int h, 
                                        const unsigned char* pixels,
                                        float* out,
                                        int outW){
    int tileX = blockIdx.x, tileY = blockIdx.y;
    int lx = threadIdx.x, ly = threadIdx.y;
    int x = tileX * 8 + lx, y = tileY * 8 + ly;

    __shared__ float sum;
    if (lx == 0 && ly == 0) sum = 0.f;
    __syncthreads();

    if (x < w && y < h) atomicAdd(&sum, (float)pixels[y * w + x]);
    __syncthreads();

    if (lx == 0 && ly == 0) out[tileY * (w/8) + tileX] = sum / 1632.0f;
}

extern "C" void quantDownScale_gpu( int w, 
                                    int h,
                                    unsigned char* pixels,
                                    float* out){
    int outW = w/8, outH = h/8;
    if (outW <= 0 || outH <= 0) return;

    unsigned char *d_in = nullptr;
    float *d_out = nullptr;

    size_t inBytes  = (size_t)w*h;
    size_t outBytes = (size_t)outW*outH*sizeof(float);

    cudaMalloc(&d_in,  inBytes);
    cudaMalloc(&d_out, outBytes);
    cudaMemcpy(d_in, pixels, inBytes, cudaMemcpyHostToDevice);

    dim3 block(8,8), grid(outW, outH);
    quantDownScaleKernel<<<grid, block>>>(w, h, d_in, d_out, outW);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, outBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

extern "C" int gpu_upload_digit_atlas(  const unsigned char* rgba,
                                        int width, 
                                        int height,
                                        int pitch_bytes){
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    if (g_atlas_array) cudaFreeArray(g_atlas_array);
    cudaMallocArray(&g_atlas_array, &ch, width, height, cudaArraySurfaceLoadStore);

    cudaMemcpy2DToArray(g_atlas_array, 0, 0,
                        rgba, pitch_bytes,
                        width * sizeof(uchar4), height,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = g_atlas_array;

    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;

    if (g_atlas_tex) cudaDestroyTextureObject(g_atlas_tex);
    cudaCreateTextureObject(&g_atlas_tex, &res, &td, nullptr);

    g_atlas_w = width;
    g_atlas_h = height;
    return 1;
}

extern "C" void gpu_atlas_shutdown(){
    if (g_atlas_tex)  { cudaDestroyTextureObject(g_atlas_tex); g_atlas_tex = 0; }
    if (g_atlas_array){ cudaFreeArray(g_atlas_array); g_atlas_array = nullptr; }
    g_atlas_w = g_atlas_h = 0;
}

__global__ void k_quant_from_luma(  int w, 
                                    int h,
                                    const unsigned char* __restrict__ y,
                                    int pitch,
                                    float* __restrict__ out,
                                    int outW){
    int tileX = blockIdx.x, tileY = blockIdx.y;
    int lx = threadIdx.x, ly = threadIdx.y;
    int x = tileX * 8 + lx;
    int yrow = tileY * 8 + ly;

    __shared__ float sum;
    if (lx == 0 && ly == 0) sum = 0.f;
    __syncthreads();

    if (x < w && yrow < h){
        unsigned char p = y[yrow * pitch + x];
        atomicAdd(&sum, (float)p);
    }
    __syncthreads();

    if (lx == 0 && ly == 0){
        out[tileY * outW + tileX] = sum / 1632.0f;
    }
}

extern "C" void quant_from_luma_device( int w, 
                                        int h,
                                        const void* d_luma,
                                        int luma_pitch,
                                        float* d_quant_out){
    int outW = w / 8, outH = h / 8;
    if (outW <= 0 || outH <= 0) return;
    dim3 block(8,8), grid(outW, outH);
    k_quant_from_luma<<<grid, block>>>(w, h, (const unsigned char*)d_luma, luma_pitch,
                                       d_quant_out, outW);
    cudaDeviceSynchronize();
}




__global__ void k_quant_from_luma_p010( int w, 
                                        int h,
                                        const uint16_t* __restrict__ y16,
                                        int pitch16_bytes,
                                        float* __restrict__ out, int outW){
    int tileX = blockIdx.x, tileY = blockIdx.y;
    int lx = threadIdx.x, ly = threadIdx.y;
    int x = tileX * 8 + lx;
    int yrow = tileY * 8 + ly;

    __shared__ float sum;
    if (lx == 0 && ly == 0) sum = 0.f;
    __syncthreads();

    if (x < w && yrow < h){
        const uint16_t* row = (const uint16_t*)((const char*)y16 + yrow * pitch16_bytes);
        float p = float(row[x] >> 8);
        atomicAdd(&sum, p);
    }
    __syncthreads();

    if (lx == 0 && ly == 0){
        out[tileY * outW + tileX] = sum / 1632.0f;
    }
}

extern "C" void quant_from_luma_device_p010(int w, 
                                            int h,
                                            const void* d_luma, 
                                            int luma_pitch,
                                            float* d_quant_out){
    int outW = w / 8, outH = h / 8;
    if (outW <= 0 || outH <= 0) return;
    dim3 block(8,8), grid(outW, outH);
    k_quant_from_luma_p010<<<grid, block>>>(w, h,
        (const uint16_t*)d_luma, luma_pitch, d_quant_out, outW);
    cudaDeviceSynchronize();
}
