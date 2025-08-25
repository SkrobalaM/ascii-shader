```
  ___           _ _                        _                                _            
 / _ \         (_|_)                      | |                              (_)           
/ /_\ \___  ___ _ _     _ __ ___ _ __   __| | ___ _ __      ___ _ __   __ _ _ _ __   ___ 
|  _  / __|/ __| | |   | '__/ _ \ '_ \ / _` |/ _ \ '__|    / _ \ '_ \ / _` | | '_ \ / _ \
| | | \__ \ (__| | |   | | |  __/ | | | (_| |  __/ |      |  __/ | | | (_| | | | | |  __/
\_| |_/___/\___|_|_|   |_|  \___|_| |_|\__,_|\___|_|       \___|_| |_|\__, |_|_| |_|\___|
                                                                       __/ |             
                                                                      |___/              
```

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![FFmpeg](https://shields.io/badge/FFmpeg-%23171717.svg?logo=ffmpeg&style=for-the-badge&labelColor=171717&logoColor=5cb85c)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff)](#)

This program extracts frames from a video file and processes them to display a **quantized mosaic visualization** in real time.

![screenshot](example/img_example.png)

https://github.com/user-attachments/assets/c4a85c45-899b-42e4-9996-00ba79a45079

# CPU

## How It Works

**Frame Extraction**:
   - Uses `ffmpeg` via an external thread to extract `.jpg` images into `video_data/`.

**Frame Processing Loop**:
   - Waits until a minimum number of frames are available.
   - Loads one frame at a time and:
     - Converts it to grayscale.
     - Downscales it into 8×8 pixel blocks.
     - Maps luminance to one of 10 tile images (0–9).
     - Renders the mosaic in a SDL2 window.

**Visualization**:
   - Each tile represents a quantized brightness level.
   - Rendering continues until all frames are processed or the user exits.

## Compilation

To compile the program, make sure you have `SDL2` `SDL2_image` `stb` and `ffmpeg` installed:

```bash
make cpu
```

## Usage

```bash
./image_processing input.mp4
```

# GPU
To improve performance the gpu is used. The gpu.c only quantize and samples frames on the gpu and copy back data to the cpu to display the frame.
This solution improves slightly the rendering speed (2-5%).
The gpu+.c is loading quantizing sampling and rendering frames on the gpu. It improves a lot rendering (350-450%).

## Compilation
```bash
make gpu
make gpu+
```

# Folder Structure

- `video_data/` – Temporary storage for extracted frames.
- `char/` – Folder containing 10 grayscale tile images (`0.png` to `9.png`).
- `video/` – RGB videos to use as input.
- `include/` – Include for CPU and GPU version.
- `gpu+include/` – Include for GPU+ version.
- `font/` – Font for CPU and GPU FPS display.
- `example/` – Ascii rendering example.

# Performances
![screenshot](example/performances.png)

# Dependencies

- [SDL2](https://www.libsdl.org/)
- [stb_image.h](https://github.com/nothings/stb) 
- [ffmpeg](https://ffmpeg.org/)
