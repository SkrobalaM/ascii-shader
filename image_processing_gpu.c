/*
image_processing.c :
	In a thread uses ffmpeg to exract frame froma video
	When at least n number of frames have been extracted
		frame by frame : resize input image calculate luminance create quant matrix and build the image using SDL

Compilation :
	make
	make clean

Usage :
	./image_processing_gpu.c input.mp4

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
#include "include/extract.h"
#include "include/engine_cpu.h"

void quantDownScale_gpu(int w, int h, unsigned char* pixels, float* out);

void * extractT(void *video_name) {
    extract((char *)video_name);
    pthread_exit(NULL);
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
	int fps_update = 10;
	Uint64 start;
	Uint64 end;
	double elapsedMS;
	int fps = 0;

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

	    
	    start = SDL_GetPerformanceCounter();
	    clearImage(w_d, h_d, pixels_d);
	    ticks = SDL_GetPerformanceCounter();
		quantDownScale_gpu(w,h,pixels,pixels_d);
		ms = (SDL_GetPerformanceCounter() - ticks);
		u_sec = ms/1000000.0;
		total_rendering_delay += u_sec;
		
		stbi_image_free(pixels);
		renderingEngine(gamma, w_d, h_d, number_tiles,pixels_d, digits, ren, mosaic);

		SDL_SetRenderTarget(ren, NULL);
	    SDL_RenderClear(ren);
	    SDL_RenderCopy(ren, mosaic, NULL, NULL);

		if (frame_nb%fps_update == 0) {
			Uint32 now = SDL_GetTicks();
		    end = SDL_GetPerformanceCounter();
			elapsedMS = (end - start) * 1000.0 / (double)SDL_GetPerformanceFrequency();
			fps = (int)1000.0 / elapsedMS;
			avg_fps += fps;
			nb_fps += 1;
		    max_frame = countFiles(dir);
		}

		snprintf(buf_fps, sizeof(buf_fps), "FPS: %d / Rendering delay : %.3f ms", fps,u_sec);
		SDL_Surface* surf = TTF_RenderText_Solid(font, buf_fps, green);
		SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
		SDL_Rect dst = { 10, 10, surf->w, surf->h };
		SDL_RenderCopy(ren, tex, NULL, &dst);
		SDL_FreeSurface(surf);
		SDL_DestroyTexture(tex);

		SDL_RenderPresent(ren);

	    frame_nb ++;
		snprintf(path, sizeof(path), "%simg%04d.jpg", dir, frame_nb);
		pixels = stbi_load(path, &w, &h, &c, desired_channels); 
	}

	pthread_join(extract_t, NULL);
	status = system("rm -f video_data/*");

	avg_fps = avg_fps/nb_fps;
	avg_rendering_delay = total_rendering_delay/frame_nb;

	SDL_SetRenderTarget(ren, NULL);
	SDL_RenderClear(ren);
	
	snprintf(buf_fps, sizeof(buf_fps), "Average FPS : %d", avg_fps);
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

