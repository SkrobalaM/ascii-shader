

__global__ void quantDownScaleKernel(int w, int h, const unsigned char* pixels, float* out, int outW) {
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

extern "C" void quantDownScale_gpu(int w, int h, unsigned char* pixels, float* out) {
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
