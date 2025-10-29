#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

// Optimized convolution kernel with shared memory
__global__ void optimizedConvolutionKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const float* __restrict__ filter,
    int filterSize,
    int width,
    int height,
    int channels,
    int pitch
) {
    // Shared memory for filter coefficients (up to 11x11)
    extern __shared__ float s_filter[];
    
    // Shared memory for image tile (with halo)
    __shared__ unsigned char s_tile[TILE_SIZE + MAX_FILTER_SIZE][TILE_SIZE + MAX_FILTER_SIZE][4];
    
    // Global position
    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Local position in tile
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    int halfFilter = filterSize / 2;
    
    // Cooperatively load filter into shared memory
    int filterElements = filterSize * filterSize;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    
    for (int i = tid; i < filterElements; i += numThreads) {
        s_filter[i] = filter[i];
    }
    
    // Load tile with halo region
    int tileX = blockIdx.x * TILE_SIZE - halfFilter;
    int tileY = blockIdx.y * TILE_SIZE - halfFilter;
    
    // Load main region
    for (int dy = 0; dy < TILE_SIZE + filterSize - 1; dy += blockDim.y) {
        for (int dx = 0; dx < TILE_SIZE + filterSize - 1; dx += blockDim.x) {
            int tx = lx + dx;
            int ty = ly + dy;
            int ix = tileX + tx;
            int iy = tileY + ty;
            
            if (tx < TILE_SIZE + filterSize - 1 && ty < TILE_SIZE + filterSize - 1) {
                // Clamp coordinates
                ix = max(0, min(ix, width - 1));
                iy = max(0, min(iy, height - 1));
                
                // Load pixel
                for (int c = 0; c < channels; c++) {
                    int idx = iy * pitch + ix * channels + c;
                    s_tile[ty][tx][c] = input[idx];
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution if within bounds
    if (gx < width && gy < height) {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        // Apply filter
        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                int tx = lx + fx;
                int ty = ly + fy;
                float filterVal = s_filter[fy * filterSize + fx];
                
                for (int c = 0; c < channels; c++) {
                    sum[c] += static_cast<float>(s_tile[ty][tx][c]) * filterVal;
                }
            }
        }
        
        // Write output
        for (int c = 0; c < channels; c++) {
            float val = sum[c];
            val = fmaxf(0.0f, fminf(val, 255.0f));
            int outIdx = gy * pitch + gx * channels + c;
            output[outIdx] = static_cast<unsigned char>(val);
        }
    }
}

// Fast box blur kernel using separable convolution
__global__ void boxBlurKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    int radius
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= width || gy >= height) return;
    
    int filterSize = 2 * radius + 1;
    float scale = 1.0f / (filterSize * filterSize);
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int sx = max(0, min(gx + dx, width - 1));
                int sy = max(0, min(gy + dy, height - 1));
                int idx = (sy * width + sx) * channels + c;
                sum += static_cast<float>(input[idx]);
            }
        }
        
        int outIdx = (gy * width + gx) * channels + c;
        output[outIdx] = static_cast<unsigned char>(sum * scale);
    }
}

// Sobel edge detection kernel
__global__ void sobelKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= width - 1 || gy >= height - 1 || gx < 1 || gy < 1) {
        if (gx < width && gy < height) {
            for (int c = 0; c < channels; c++) {
                output[(gy * width + gx) * channels + c] = 0;
            }
        }
        return;
    }
    
    // Sobel operators
    const int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int c = 0; c < channels; c++) {
        float gradX = 0.0f;
        float gradY = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int sx = gx + dx;
                int sy = gy + dy;
                int idx = (sy * width + sx) * channels + c;
                float pixelVal = static_cast<float>(input[idx]);
                
                gradX += pixelVal * sobelX[dy + 1][dx + 1];
                gradY += pixelVal * sobelY[dy + 1][dx + 1];
            }
        }
        
        float magnitude = sqrtf(gradX * gradX + gradY * gradY);
        magnitude = fminf(magnitude, 255.0f);
        
        int outIdx = (gy * width + gx) * channels + c;
        output[outIdx] = static_cast<unsigned char>(magnitude);
    }
}

// Sharpen kernel
__global__ void sharpenKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    float amount
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= width - 1 || gy >= height - 1 || gx < 1 || gy < 1) {
        if (gx < width && gy < height) {
            for (int c = 0; c < channels; c++) {
                int idx = (gy * width + gx) * channels + c;
                output[idx] = input[idx];
            }
        }
        return;
    }
    
    // Laplacian kernel for sharpening
    const float laplacian[3][3] = {
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    };
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int sx = gx + dx;
                int sy = gy + dy;
                int idx = (sy * width + sx) * channels + c;
                sum += static_cast<float>(input[idx]) * laplacian[dy + 1][dx + 1];
            }
        }
        
        int centerIdx = (gy * width + gx) * channels + c;
        float original = static_cast<float>(input[centerIdx]);
        float sharpened = original + amount * sum;
        sharpened = fmaxf(0.0f, fminf(sharpened, 255.0f));
        
        output[centerIdx] = static_cast<unsigned char>(sharpened);
    }
}

// Emboss kernel
__global__ void embossKernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gx >= width - 1 || gy >= height - 1 || gx < 1 || gy < 1) {
        if (gx < width && gy < height) {
            for (int c = 0; c < channels; c++) {
                output[(gy * width + gx) * channels + c] = 128;
            }
        }
        return;
    }
    
    const float emboss[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int sx = gx + dx;
                int sy = gy + dy;
                int idx = (sy * width + sx) * channels + c;
                sum += static_cast<float>(input[idx]) * emboss[dy + 1][dx + 1];
            }
        }
        
        sum += 128.0f;  // Add gray level
        sum = fmaxf(0.0f, fminf(sum, 255.0f));
        
        int outIdx = (gy * width + gx) * channels + c;
        output[outIdx] = static_cast<unsigned char>(sum);
    }
}

// Host wrapper functions
extern "C" {

void generateGaussianFilter(float* filter, int size, float sigma) {
    int halfSize = size / 2;
    float sum = 0.0f;
    float twoSigmaSquare = 2.0f * sigma * sigma;
    
    for (int y = -halfSize; y <= halfSize; y++) {
        for (int x = -halfSize; x <= halfSize; x++) {
            int idx = (y + halfSize) * size + (x + halfSize);
            float exponent = -(x * x + y * y) / twoSigmaSquare;
            filter[idx] = expf(exponent);
            sum += filter[idx];
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

cudaError_t launchConvolutionKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    const float* d_filter,
    int filterSize,
    int width,
    int height,
    int channels,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + TILE_SIZE - 1) / TILE_SIZE,
        (height + TILE_SIZE - 1) / TILE_SIZE
    );
    
    int sharedMemSize = filterSize * filterSize * sizeof(float);
    
    optimizedConvolutionKernel<<<gridDim, blockDim, sharedMemSize, stream>>>(
        d_input, d_output, d_filter, filterSize, width, height, channels, width
    );
    
    return cudaGetLastError();
}

cudaError_t launchBoxBlurKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int radius,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    boxBlurKernel<<<gridDim, blockDim, 0, stream>>>(
        d_input, d_output, width, height, channels, radius
    );
    
    return cudaGetLastError();
}

cudaError_t launchSobelKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    sobelKernel<<<gridDim, blockDim, 0, stream>>>(
        d_input, d_output, width, height, channels
    );
    
    return cudaGetLastError();
}

cudaError_t launchSharpenKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    float amount,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    sharpenKernel<<<gridDim, blockDim, 0, stream>>>(
        d_input, d_output, width, height, channels, amount
    );
    
    return cudaGetLastError();
}

cudaError_t launchEmbossKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    embossKernel<<<gridDim, blockDim, 0, stream>>>(
        d_input, d_output, width, height, channels
    );
    
    return cudaGetLastError();
}

} // extern "C"