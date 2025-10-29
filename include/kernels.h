#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Configuration constants
#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define MAX_FILTER_SIZE 11

#ifdef __cplusplus
extern "C" {
#endif

// Filter generation functions
void generateGaussianFilter(float* filter, int size, float sigma);

// Kernel launch functions
cudaError_t launchConvolutionKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    const float* d_filter,
    int filterSize,
    int width,
    int height,
    int channels,
    cudaStream_t stream = 0
);

cudaError_t launchBoxBlurKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int radius,
    cudaStream_t stream = 0
);

cudaError_t launchSobelKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    cudaStream_t stream = 0
);

cudaError_t launchSharpenKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    float amount,
    cudaStream_t stream = 0
);

cudaError_t launchEmbossKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    cudaStream_t stream = 0
);

cudaError_t launchLaplacianKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    cudaStream_t stream = 0
);

cudaError_t launchMotionBlurKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int length,
    float angle,
    cudaStream_t stream = 0
);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H