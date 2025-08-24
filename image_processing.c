/*
image_processing.c :
	In a thread uses ffmpeg to exract frame froma video
	When at least n number of frames have been extracted
		frame by frame : resize input image calculate luminance create quant matrix and build the image using SDL

Compilation :
	gcc -O2 image_processing.c -o image_processing -lm `sdl2-config --cflags --libs` -lSDL2_image -lSDL2_ttf

Usage :
	./image_processing.c input.mp4

Performance :
	no optimisation
		FF7_vid.mp4 : 	Average FPS  = 39
						Average rendering delay = 5.961 ms
	-01
		FF7_vid.mp4 : 	Average FPS  = 92
						Average rendering delay = 1.804 ms
	-02
		FF7_vid.mp4 : 	Average FPS  = 95
						Average rendering delay = 1.574 ms
	-03
		FF7_vid.mp4 : 	Average FPS  = 102
						Average rendering delay = 1.302 ms
	-0fast
		FF7_vid.mp4 : 	Average FPS  = 102
						Average rendering delay = 1.300 ms
*/

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include "stb/stb_image.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include "extract.h"

void * extractT(void *video_name) {
    extract((char *)video_name);
    pthread_exit(NULL);
}

int fileExists(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

void quantDownScale (int w, int h, unsigned char* pixels, float* quant_pixels_downScaled){
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

void clearImage(int w, int h, float* pixels){
	memset(pixels, 0, sizeof(float) * (size_t)w * (size_t)h);
}

static SDL_Texture* loadTexture(SDL_Renderer* r, const char* path) {
    SDL_Texture* tex = IMG_LoadTexture(r, path);
    if (!tex) fprintf(stderr, "IMG_LoadTexture failed for %s: %s\n", path, IMG_GetError());
    return tex;
}

void loadTilesLopp(int number_tiles,const char* path_textures,SDL_Renderer* ren,SDL_Texture** digits){
	char name[512];
	for (int d = 0; d < number_tiles; ++d) {
        snprintf(name, sizeof(name), "%s/%d.png", path_textures, d);
        digits[d] = loadTexture(ren, name);
    }
}

int sensitivity(int x, float gamma) {
    float t = x / 10.0;

    float num = pow(t, gamma);
    float den = num + pow(1.0 - t, gamma);

    if (den == 0.0) return (int)x;

    return (int)10.0 * num / den;
}

void renderingEngine(int gamma, int w, int h, int number_tiles,float* pixels, SDL_Texture** digits, SDL_Renderer* ren, SDL_Texture* mosaic){
	SDL_SetRenderTarget(ren, mosaic);
	SDL_SetRenderDrawColor(ren, 16,16,16,255);
	SDL_RenderClear(ren);
	int max_val = number_tiles -1;
	for (int y = 0; y < h; ++y) {
	    for (int x = 0; x < w; ++x) {
	        int v = pixels[y*w + x];       
			v = sensitivity(v,gamma);
	        if (v < 0) v = 0; if (v > 9) v = 9;
	        SDL_Rect dst = { x * 8, y * 8, 8, 8 };
	        SDL_RenderCopy(ren, digits[v], NULL, &dst);
	    }
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
        fprintf(stderr, "Usage: %s image.png\n", argv[0]);
        return 1;
    }

    int status;
    status = system("rm -f video_data/*");

    void* vid = argv[1];
    pthread_t extract_t;
    pthread_create(&extract_t, NULL, extractT, vid);

    

    const char* frame = "img0001.jpg";
    const char* path_textures="char";
    const char* dir = "video_data/";
    const int DISPLAY_SCALE = 1;
    const int size_tiles = 8;
    const int number_tiles = 10;

    int check = 1;
    char* buffer_frame = "img0200.jpg";
    char path_buffer[256];

    char buf_fps[512];
    SDL_Color green = {51, 153, 51, 255};

    snprintf(path_buffer, sizeof(path_buffer), "%s%s", dir,buffer_frame);

    char path[256];
    snprintf(path, sizeof(path), "%s%s", dir,frame);

    float gamma = 1.5f;
    SDL_Texture** digits = malloc(sizeof(SDL_Texture*)*number_tiles);

    
    
    int w = 0, h = 0, c = 0;
    int desired_channels = 1;

    while(check){
    	if (fileExists(path_buffer)){
    		check = 0;
    	}
    } 

    unsigned char* pixels = stbi_load(path, &w, &h, &c, desired_channels);

	int w_d = (int)w/size_tiles;
	int h_d = (int)h/size_tiles;
	int img_w = w_d * size_tiles;
    int img_h = h_d * size_tiles;


	float* pixels_d = malloc(sizeof(float)*w_d*h_d);

	
	
	
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
	SDL_Window* win = SDL_CreateWindow("Quantized Mosaic",
                                       SDL_WINDOWPOS_CENTERED, 
                                       SDL_WINDOWPOS_CENTERED,
                                       img_w, img_h, 
                                       SDL_WINDOW_SHOWN);

	SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
	SDL_SetWindowSize(win, img_w * DISPLAY_SCALE, img_h * DISPLAY_SCALE);
    SDL_RenderSetLogicalSize(ren, img_w, img_h);
    SDL_Texture* mosaic = SDL_CreateTexture(ren,
										    SDL_PIXELFORMAT_RGBA8888,
										    SDL_TEXTUREACCESS_TARGET,
										    img_w, img_h);

	loadTilesLopp(number_tiles,path_textures,ren,digits);

	TTF_Init();
	TTF_Font* font = TTF_OpenFont("font/font.ttf", 32);
	
	
    
	
	
	

	int max_frame = countFiles(dir);
	int frame_nb=1;
	int running = 1;


	int avg_fps = 0;
	int nb_fps = 0;
	int fps_last_time = 0;
	int fps_frames = 0;
	int fps_current = 0;

	int ticks = 0;
	int ms = 0;
	double u_sec = 0.0;
	double total_rendering_delay = 0.0;
	double avg_rendering_delay;


	SDL_Event e;
	while (running && frame_nb<max_frame-1) {
	    while (SDL_PollEvent(&e)) {
	        if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
	            running = 0;
	    }

	    


	    clearImage(w_d, h_d, pixels_d);
	    ticks = SDL_GetPerformanceCounter();
		quantDownScale(w,h,pixels,pixels_d);
		ms = (SDL_GetPerformanceCounter() - ticks);
		u_sec = ms/1000000.0;
		total_rendering_delay += u_sec;
		
		stbi_image_free(pixels);

		
		

		renderingEngine(gamma, w_d, h_d, number_tiles,pixels_d, digits, ren, mosaic);

		SDL_SetRenderTarget(ren, NULL);
	    SDL_RenderClear(ren);
	    SDL_RenderCopy(ren, mosaic, NULL, NULL);
	    

	    fps_frames++;
		Uint32 now = SDL_GetTicks();
		if (now - fps_last_time >= 500) {
		    fps_current = 2*fps_frames;
		    fps_frames = 0;
		    fps_last_time = now;
		    avg_fps += fps_current;
		    nb_fps += 1;
		    max_frame = countFiles(dir);
		}


		snprintf(buf_fps, sizeof(buf_fps), "FPS: %d / Rendering delay : %.3f ms", fps_current,u_sec);

		SDL_Surface* surf = TTF_RenderText_Solid(font, buf_fps, green);
		SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
		SDL_Rect dst = { 10, 10, surf->w, surf->h };
		SDL_RenderCopy(ren, tex, NULL, &dst);
		SDL_FreeSurface(surf);
		SDL_DestroyTexture(tex);


		SDL_RenderPresent(ren);
	    //SDL_Delay(16);


	    frame_nb ++;
		snprintf(path, sizeof(path), "%simg%04d.jpg", dir, frame_nb);
		
		pixels = stbi_load(path, &w, &h, &c, desired_channels);

	    
	}

	pthread_join(extract_t, NULL);
	status = system("rm -f video_data/*");

	avg_fps = avg_fps/nb_fps;
	avg_rendering_delay = total_rendering_delay/frame_nb;

	snprintf(buf_fps, sizeof(buf_fps), "Average FPS : %d", avg_fps);
	SDL_SetRenderTarget(ren, NULL);
	SDL_RenderClear(ren);
	SDL_Surface* surf = TTF_RenderText_Solid(font, buf_fps, green);
	SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
	SDL_Rect dst = { 10, 10, surf->w, surf->h };
	SDL_RenderCopy(ren, tex, NULL, &dst);
	SDL_FreeSurface(surf);
	SDL_DestroyTexture(tex);

	snprintf(buf_fps, sizeof(buf_fps), "Average frame rendering delay : %.3f ms", avg_rendering_delay);
	surf = TTF_RenderText_Solid(font, buf_fps, green);
	tex = SDL_CreateTextureFromSurface(ren, surf);
	SDL_Rect dst2 = { 10, 50, surf->w, surf->h };
	SDL_RenderCopy(ren, tex, NULL, &dst2);
	SDL_FreeSurface(surf);
	SDL_DestroyTexture(tex);


	running = 1;
	while (running) {
	    while (SDL_PollEvent(&e)) {
	        if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
	            running = 0;
	    }
	    SDL_RenderPresent(ren);
		SDL_Delay(16);
	}
	

	return 0;
}

