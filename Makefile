ARCH      ?= sm_89
CLANG_VER ?= 14
CUDA_HOME ?= /usr/local/cuda

CC        := clang-$(CLANG_VER)
CXX       := clang++-$(CLANG_VER)
GCC       := gcc
GXX       := g++
NVCC      := nvcc

HOST_NO_AMX := -mno-amx-tile -mno-amx-int8 -mno-amx-bf16

CFLAGS     := -O3 -std=c11 $(shell sdl2-config --cflags)
CFLAGS_CPU := -Ofast
NVCCFLAGS  := -O3 -std=c++14 -arch=$(ARCH) -ccbin $(CXX) \
              -allow-unsupported-compiler $(shell sdl2-config --cflags) \
              -diag-suppress 550 \
              -Xcompiler "$(HOST_NO_AMX)"

SDL_LIBS   := $(shell sdl2-config --libs) -lSDL2_image -lSDL2_ttf
FFMPEG_LIBS := -lavformat -lavcodec -lavutil
LDFLAGS    := -L$(CUDA_HOME)/lib64
LIBS       := -lGL $(SDL_LIBS) -lcudart -lm

APP1    := main_gpu+
APP_GPU := main_gpu
APP_CPU := main_cpu

C_SRC1   := main_gpu+.c gpu+include/ffmpeg_cuda.c
CU_SRC1  := gpu+include/engine_gpu+.cu
OBJ1     := $(C_SRC1:.c=.o) $(CU_SRC1:.cu=.o)

C_SRC_GPU := main_gpu.c
CU_SRC_GPU := include/engine_gpu.cu
C_SRC_CPU := main_cpu.c

OBJ_C_GPU := $(C_SRC_GPU:.c=.o)
OBJ_CU    := $(CU_SRC_GPU:.cu=.o)
OBJ_C_CPU := $(C_SRC_CPU:.c=.o)

.PHONY: all clean gpu cpu

all: $(APP1) $(APP_GPU) $(APP_CPU)

gpu: $(APP_GPU)
cpu: $(APP_CPU)
gpu+:$(APP1)

$(APP1): $(OBJ1)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS) $(FFMPEG_LIBS)

$(APP_GPU): $(OBJ_C_GPU) $(OBJ_CU)
	$(GXX) -o $@ $^ $(LDFLAGS) $(SDL_LIBS) -lcudart -lm

$(APP_CPU): $(OBJ_C_CPU)
	$(GCC) -o $@ $^ $(CFLAGS_CPU) $(SDL_LIBS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ1) $(OBJ_C_GPU) $(OBJ_CU) $(OBJ_C_CPU) \
	      $(APP1) $(APP_GPU) $(APP_CPU)
