#define STB_IMAGE_IMPLEMENTATION
#define STB_EASY_FONT_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <libavutil/pixfmt.h>

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "stb/stb_easy_font.h"

int  gpu_gl_init(SDL_Window* win, int img_w, int img_h);
int  gpu_upload_digit_atlas(const unsigned char* rgba, int width, int height, int pitch_bytes);
void gpu_gl_render_digits(int img_w, int img_h, int tile_sz, const float* d_quant, float gamma);
void gpu_atlas_shutdown(void);
void gpu_gl_shutdown(void);

void quant_from_luma_device(int w, int h, const void* d_luma, int pitch, float* d_quant_out);
void quant_from_luma_device_p010(int w, int h, const void* d_luma, int pitch, float* d_quant_out);

int  fc_open_cuda(const char *filename, int *out_w, int *out_h);
int  fc_read_cuda_luma(CUdeviceptr *d_luma, int *pitch, int *w, int *h);
int  fc_get_sw_format(void);
void fc_close(void);



int main(int argc, char** argv){
    if (argc < 2){
        fprintf(stderr, "Usage: %s input.mp4\n", argv[0]);
        return 1;
    }

    const int VSYNC = 0;
    const int resize = 0 ;

    setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    setenv("SDL_VIDEODRIVER", "x11", 1);
    SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, "1");

    SDL_Init(SDL_INIT_VIDEO);
    int img_flags = IMG_INIT_PNG | IMG_INIT_JPG;
    IMG_Init(img_flags);


    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

    int vid_w = 0, vid_h = 0;
    fc_open_cuda(argv[1], &vid_w, &vid_h);

    const int   tile_sz   = 8;
    const float gamma_val = 1.0f;
    const int   img_w     = (vid_w / tile_sz) * tile_sz;
    const int   img_h     = (vid_h / tile_sz) * tile_sz;

    const int sw_fmt   = fc_get_sw_format();
    const int luma_bpp = (sw_fmt == AV_PIX_FMT_P010) ? 2 : 1;

    size_t luma_copy_pitch = 0;
    unsigned char* d_luma_copy = NULL;
    cudaMallocPitch((void**)&d_luma_copy, &luma_copy_pitch,(size_t)img_w * 2 /* bytes */, img_h);

    int winW;
    int winH;
    if (resize){
        winW = (img_w > 1920 || img_h > 1080) ? img_w/2 : img_w;
        winH = (img_w > 1920 || img_h > 1080) ? img_h/2 : img_h;
    }
    else{
        winW = img_w;
        winH = img_h;
    }
    

    SDL_Window* win = SDL_CreateWindow( "Quantized Mosaic",
                                        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        winW, winH,
                                        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    SDL_GLContext gl = SDL_GL_CreateContext(win);
    SDL_GL_SetSwapInterval(VSYNC);
    gpu_gl_init(win, img_w, img_h);

    int dw, dh; SDL_GL_GetDrawableSize(win, &dw, &dh);
    glViewport(0, 0, dw, dh);

    unsigned char* atlas_rgba = (unsigned char*)malloc(tile_sz * 10 * tile_sz * 4);

    for (int d = 0; d < 10; ++d){
        char name[256]; snprintf(name, sizeof(name), "char/%d.png", d);
        SDL_Surface* surf = IMG_Load(name);
        SDL_Surface* rgba = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_ABGR8888, 0);
        SDL_FreeSurface(surf);
        for (int y = 0; y < tile_sz; ++y){
            unsigned char* dst = atlas_rgba + (y * (tile_sz*10) + d*tile_sz) * 4;
            unsigned char* src = (unsigned char*)rgba->pixels + y * rgba->pitch;
            memcpy(dst, src, tile_sz * 4);
        }
        SDL_FreeSurface(rgba);
    }
    gpu_upload_digit_atlas(atlas_rgba, tile_sz*10, tile_sz, tile_sz*10*4);
    free(atlas_rgba);

    const int w_d = img_w / tile_sz;
    const int h_d = img_h / tile_sz;
    float* d_quant = NULL;
    cudaMalloc((void**)&d_quant, (size_t)w_d * h_d * sizeof(*d_quant));

    int running = 1;
    Uint64 t0 = 0, t1 = 0;
    const double freq = (double)SDL_GetPerformanceFrequency();
    double total_ms = 0.0; int frames = 0; double ms = 0.0;

    int fps;
    static char fps_txt[64];
    double avg = 0.0;

    SDL_Event e;
    while (running){
        while (SDL_PollEvent(&e)){
            if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
                running = 0;
        }

        t0 = SDL_GetPerformanceCounter();

        CUdeviceptr d_luma = 0;
        int pitch = 0, w = 0, h = 0;
        if (!fc_read_cuda_luma(&d_luma, &pitch, &w, &h)) break;

        int luma_bpp_runtime = (pitch >= img_w * 2) ? 2 : 1;

        cudaMemcpy2D(   d_luma_copy,
                        luma_copy_pitch,
                        (const void*)(uintptr_t)d_luma,
                        pitch,
                        (size_t)img_w * luma_bpp_runtime,
                        img_h,
                        cudaMemcpyDeviceToDevice);


        if (luma_bpp_runtime == 2){
            quant_from_luma_device_p010(img_w, img_h,
                                        (const void*)d_luma_copy, (int)luma_copy_pitch,
                                        d_quant);
        } else{
            quant_from_luma_device(img_w, img_h,
                                   (const void*)d_luma_copy, (int)luma_copy_pitch,
                                   d_quant);
        }

        gpu_gl_render_digits(img_w, img_h, tile_sz, d_quant, gamma_val);

        

        cudaDeviceSynchronize();

        t1 = SDL_GetPerformanceCounter();
        ms = t1 - t0;
        avg = total_ms / frames;
        total_ms += ms * 1000.0 / freq;
        frames++;
    }


    
    

    if (frames > 0){
        avg = total_ms / frames;
        printf("%.1f FPS\n", 1000.0 / avg);
    }

    cudaFree(d_quant);
    cudaFree(d_luma_copy);
    fc_close();
    gpu_atlas_shutdown();
    gpu_gl_shutdown();
    SDL_GL_DeleteContext(gl);
    SDL_DestroyWindow(win);
    IMG_Quit();
    SDL_Quit();
    return 0;
}
