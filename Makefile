
APP_GPU   = image_processing_gpu
C_SRC_GPU = image_processing_gpu.c
CU_SRC    = include/engine_gpu.cu

APP_CPU   = image_processing
C_SRC_CPU = image_processing.c

OBJ_C_GPU = $(C_SRC_GPU:.c=.o)
OBJ_CU    = $(CU_SRC:.cu=.o)
OBJ_C_CPU = $(C_SRC_CPU:.c=.o)


CC   = gcc
CXX  = g++
NVCC = nvcc


ARCH ?= sm_89
HOST_NO_AMX = -mno-amx-tile -mno-amx-int8 -mno-amx-bf16

CFLAGS    += -Ofast
NVCCFLAGS  = -O3 -arch=$(ARCH) -Xcompiler "$(HOST_NO_AMX)"

SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS   := $(shell sdl2-config --libs) -lSDL2_image -lSDL2_ttf

CUDA_HOME ?= /usr/local/cuda
LDFLAGS_GPU := -L$(CUDA_HOME)/lib64
LDLIBS_GPU  := -lcudart -lm



all: gpu

gpu: $(APP_GPU)
cpu: $(APP_CPU)


$(APP_GPU): $(OBJ_C_GPU) $(OBJ_CU)
	$(CXX) -o $@ $^ $(LDFLAGS_GPU) $(SDL_LIBS) $(LDLIBS_GPU)

$(APP_CPU): $(OBJ_C_CPU)
	$(CC) -o $@ $^ $(CFLAGS) $(SDL_LIBS) -lm


%.o: %.c
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(SDL_CFLAGS) -c $< -o $@


clean:
	rm -f $(OBJ_C_GPU) $(OBJ_CU) $(OBJ_C_CPU) $(APP_GPU) $(APP_CPU)
