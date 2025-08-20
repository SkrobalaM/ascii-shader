// image_processing.c : Resize input image calculate luminance create quant matrix and build the image using SDL
//
// Compilation :
//   gcc -O2 image_processing.c -o image_processing -lm `sdl2-config --cflags --libs` -lSDL2_image
//
// Usage :
//   ./image_processing.c input.png


#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>


void quant_downScale (int w, int h, unsigned char* pixels, float* quant_pixels_downScaled){
	int w_d = (int)w/8;
	int h_d = (int)h/8;
	int sum = 0;
	float quant_pixel_down;
	for (int i = 0; i < h; ++i){
		for (int j = 0; j < w; ++j){
			if ((float)i/8 <= h_d || (float)j/8 <= w_d){
				quant_pixel_down = (float)pixels[i*w + j]/1632;
				quant_pixels_downScaled[(i/8) * (w/8) + (j/8)] = quant_pixels_downScaled[(i/8) * (w/8) + (j/8)] +quant_pixel_down;	

			}
		}
	}
}


static SDL_Texture* load_texture(SDL_Renderer* r, const char* path) {
    SDL_Texture* tex = IMG_LoadTexture(r, path);
    if (!tex) fprintf(stderr, "IMG_LoadTexture failed for %s: %s\n", path, IMG_GetError());
    return tex;
}

int main(int argc, char** argv) {
	if (argc < 2) {
        fprintf(stderr, "Usage: %s image.png\n", argv[0]);
        return 1;
    }

    const char* path = argv[1];
    char* path_textures="char";

    const int DISPLAY_SCALE = 2;
    int w = 0, h = 0, c = 0;
    int desired_channels = 1;

    unsigned char* pixels = stbi_load(path, &w, &h, &c, desired_channels);



    printf("%d\n",w);
    printf("%d\n",h);
	printf("%d\n\n",c);
	int w_d = (int)w/8;
	int h_d = (int)h/8;

	printf("%d\n",w_d);
	printf("%d\n",h_d);

	float* pixels_d = malloc(sizeof(float)*w_d*h_d);
	for (int i = 0; i < w_d; ++i)
	{
		for (int j = 0; j < h_d; ++j)
		{
			pixels_d[i+j] = 0.0;
		}
	}
	quant_downScale(w,h,pixels,pixels_d);

	
	stbi_image_free(pixels);
	int img_w = w_d * 8;
    int img_h = h_d * 8;
	
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
	SDL_Window* win = SDL_CreateWindow("Quantized Mosaic",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       img_w, img_h, SDL_WINDOW_SHOWN);
	SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

	SDL_Texture* digits[10] = {0};
    
    char patht[512];
    for (int d = 0; d < 10; ++d) {
        snprintf(patht, sizeof(patht), "%s/%d.png", path_textures, d);
        digits[d] = load_texture(ren, patht);
    }

    SDL_SetWindowSize(win, img_w * DISPLAY_SCALE, img_h * DISPLAY_SCALE);
    SDL_RenderSetLogicalSize(ren, img_w, img_h);

    SDL_Texture* mosaic = SDL_CreateTexture(
    ren,
    SDL_PIXELFORMAT_RGBA8888,
    SDL_TEXTUREACCESS_TARGET,
    img_w, img_h);

	// build once
	SDL_SetRenderTarget(ren, mosaic);
	SDL_SetRenderDrawColor(ren, 16,16,16,255);
	SDL_RenderClear(ren);
	float gamma = 1.5f;
	int max_val = 9;
	for (int y = 0; y < h_d; ++y) {
	    for (int x = 0; x < w_d; ++x) {
	        int v = pixels_d[y*w_d + x];
	        float norm = (float)v / max_val;
			norm = powf(norm, gamma);         
			v = (int)(norm * max_val + 0.5f);
	        if (v < 0) v = 0; if (v > 9) v = 9;
	        SDL_Rect dst = { x * 8, y * 8, 8, 8 };
	        SDL_RenderCopy(ren, digits[v], NULL, &dst);
	    }
	}

	SDL_SetRenderTarget(ren, NULL); 
	int running = 1;
	SDL_Event e;
	while (running) {
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
            running = 0;
    }

    SDL_RenderClear(ren);
    SDL_RenderCopy(ren, mosaic, NULL, NULL);
    SDL_RenderPresent(ren);
    SDL_Delay(16);
}

}

