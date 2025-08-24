#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>



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