#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include "kernels.h"
#include "image_io.h"
#include "stream_manager.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Config {
    std::string inputDir;
    std::string outputDir;
    std::string filterType;
    int batchSize = 0;  // 0 = auto-detect
    int numStreams = 4;
    bool enableProfiling = false;
    bool verbose = false;
    std::string metricsFile;
    int deviceId = 0;
};

struct PerformanceMetrics {
    double totalTime = 0.0;
    double loadTime = 0.0;
    double transferTime = 0.0;
    double kernelTime = 0.0;
    double saveTime = 0.0;
    int numImages = 0;
    size_t totalBytes = 0;
};

class ProgressBar {
private:
    int total;
    int barWidth = 50;
    
public:
    ProgressBar(int t) : total(t) {}
    
    void update(int current) {
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(barWidth * progress);
        
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "█";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% (" 
                  << current << "/" << total << ")\r";
        std::cout.flush();
    }
    
    void finish() {
        std::cout << std::endl;
    }
};

void printUsage(const char* progName) {
    std::cout << "CUDA Parallel Image Filter Pipeline\n\n";
    std::cout << "Usage: " << progName << " [OPTIONS]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -i, --input PATH        Input directory with images\n";
    std::cout << "  -o, --output PATH       Output directory for results\n";
    std::cout << "  -f, --filter TYPE       Filter type to apply\n\n";
    std::cout << "Optional:\n";
    std::cout << "  -b, --batch-size N      Images per batch (default: auto)\n";
    std::cout << "  -s, --streams N         CUDA streams (default: 4)\n";
    std::cout << "  -d, --device N          CUDA device ID (default: 0)\n";
    std::cout << "  -p, --profile           Enable profiling\n";
    std::cout << "  -m, --metrics FILE      Save metrics to JSON file\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  -h, --help              Show this help\n\n";
    std::cout << "Filter types:\n";
    std::cout << "  gaussian_blur           5x5 Gaussian blur\n";
    std::cout << "  box_blur                3x3 box blur\n";
    std::cout << "  sharpen                 Image sharpening\n";
    std::cout << "  sobel                   Sobel edge detection\n";
    std::cout << "  laplacian              Laplacian edge detection\n";
    std::cout << "  emboss                  3D emboss effect\n\n";
}

bool parseArgs(int argc, char** argv, Config& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            return false;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.inputDir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.outputDir = argv[++i];
        } else if ((arg == "-f" || arg == "--filter") && i + 1 < argc) {
            config.filterType = argv[++i];
        } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            config.batchSize = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--streams") && i + 1 < argc) {
            config.numStreams = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            config.deviceId = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--profile") {
            config.enableProfiling = true;
        } else if ((arg == "-m" || arg == "--metrics") && i + 1 < argc) {
            config.metricsFile = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    return !config.inputDir.empty() && !config.outputDir.empty() && 
           !config.filterType.empty();
}

void printDeviceInfo(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    std::cout << "\n=== CUDA Device Information ===\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "================================\n\n";
}

int determineOptimalBatchSize(const std::vector<ImageData>& images, int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    // Calculate average image size
    size_t avgSize = 0;
    for (const auto& img : images) {
        avgSize += img.width * img.height * img.channels;
    }
    avgSize /= images.size();
    
    // Use 60% of available memory for batch processing
    size_t availableMemory = static_cast<size_t>(prop.totalGlobalMem * 0.6);
    size_t batchSize = availableMemory / (avgSize * 2);  // *2 for input+output
    
    // Clamp to reasonable range
    batchSize = std::max(size_t(4), std::min(batchSize, size_t(64)));
    
    return static_cast<int>(batchSize);
}

void processBatch(
    const std::vector<ImageData>& batch,
    std::vector<ImageData>& outputBatch,
    const Config& config,
    StreamManager& streamMgr,
    PerformanceMetrics& metrics
) {
    auto batchStart = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory for batch
    std::vector<unsigned char*> d_inputs(batch.size());
    std::vector<unsigned char*> d_outputs(batch.size());
    std::vector<float*> d_filters(batch.size());
    
    for (size_t i = 0; i < batch.size(); i++) {
        size_t imageSize = batch[i].width * batch[i].height * batch[i].channels;
        
        CUDA_CHECK(cudaMalloc(&d_inputs[i], imageSize));
        CUDA_CHECK(cudaMalloc(&d_outputs[i], imageSize));
        
        // Async transfer to device
        cudaStream_t stream = streamMgr.getStream(i % config.numStreams);
        CUDA_CHECK(cudaMemcpyAsync(d_inputs[i], batch[i].data, imageSize,
                                   cudaMemcpyHostToDevice, stream));
    }
    
    auto transferEnd = std::chrono::high_resolution_clock::now();
    metrics.transferTime += std::chrono::duration<double, std::milli>(
        transferEnd - batchStart).count();
    
    // Process each image
    auto kernelStart = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < batch.size(); i++) {
        cudaStream_t stream = streamMgr.getStream(i % config.numStreams);
        
        if (config.filterType == "gaussian_blur") {
            // Generate and upload Gaussian filter
            float h_filter[25];
            generateGaussianFilter(h_filter, 5, 1.4f);
            CUDA_CHECK(cudaMalloc(&d_filters[i], 25 * sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(d_filters[i], h_filter, 25 * sizeof(float),
                                      cudaMemcpyHostToDevice, stream));
            
            launchConvolutionKernel(d_inputs[i], d_outputs[i], d_filters[i], 5,
                                   batch[i].width, batch[i].height, 
                                   batch[i].channels, stream);
        } else if (config.filterType == "box_blur") {
            launchBoxBlurKernel(d_inputs[i], d_outputs[i], batch[i].width,
                               batch[i].height, batch[i].channels, 1, stream);
        } else if (config.filterType == "sobel") {
            launchSobelKernel(d_inputs[i], d_outputs[i], batch[i].width,
                             batch[i].height, batch[i].channels, stream);
        } else if (config.filterType == "sharpen") {
            launchSharpenKernel(d_inputs[i], d_outputs[i], batch[i].width,
                               batch[i].height, batch[i].channels, 1.5f, stream);
        } else if (config.filterType == "emboss") {
            launchEmbossKernel(d_inputs[i], d_outputs[i], batch[i].width,
                              batch[i].height, batch[i].channels, stream);
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < config.numStreams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streamMgr.getStream(i)));
    }
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    metrics.kernelTime += std::chrono::duration<double, std::milli>(
        kernelEnd - kernelStart).count();
    
    // Transfer results back
    for (size_t i = 0; i < batch.size(); i++) {
        size_t imageSize = batch[i].width * batch[i].height * batch[i].channels;
        outputBatch[i].data = new unsigned char[imageSize];
        outputBatch[i].width = batch[i].width;
        outputBatch[i].height = batch[i].height;
        outputBatch[i].channels = batch[i].channels;
        
        cudaStream_t stream = streamMgr.getStream(i % config.numStreams);
        CUDA_CHECK(cudaMemcpyAsync(outputBatch[i].data, d_outputs[i], imageSize,
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    // Final sync
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    for (size_t i = 0; i < batch.size(); i++) {
        CUDA_CHECK(cudaFree(d_inputs[i]));
        CUDA_CHECK(cudaFree(d_outputs[i]));
        if (d_filters[i]) CUDA_CHECK(cudaFree(d_filters[i]));
    }
    
    auto batchEnd = std::chrono::high_resolution_clock::now();
    double batchTime = std::chrono::duration<double, std::milli>(
        batchEnd - batchStart).count();
    
    std::cout << "  Batch processed in " << std::fixed << std::setprecision(1) 
              << batchTime << " ms (" 
              << static_cast<int>(batch.size() * 1000.0 / batchTime) 
              << " img/s)\n";
}

void saveMetrics(const PerformanceMetrics& metrics, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"total_time_ms\": " << metrics.totalTime << ",\n";
    file << "  \"load_time_ms\": " << metrics.loadTime << ",\n";
    file << "  \"transfer_time_ms\": " << metrics.transferTime << ",\n";
    file << "  \"kernel_time_ms\": " << metrics.kernelTime << ",\n";
    file << "  \"save_time_ms\": " << metrics.saveTime << ",\n";
    file << "  \"num_images\": " << metrics.numImages << ",\n";
    file << "  \"total_bytes\": " << metrics.totalBytes << ",\n";
    file << "  \"avg_time_per_image_ms\": " << metrics.totalTime / metrics.numImages << ",\n";
    file << "  \"throughput_img_per_sec\": " << metrics.numImages * 1000.0 / metrics.totalTime << "\n";
    file << "}\n";
    
    file.close();
}

int main(int argc, char** argv) {
    Config config;
    
    if (!parseArgs(argc, argv, config)) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(config.deviceId));
    
    std::cout << "=== CUDA Parallel Image Filter Pipeline ===\n";
    printDeviceInfo(config.deviceId);
    
    PerformanceMetrics metrics;
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Load images
    std::cout << "Loading images from: " << config.inputDir << "\n";
    auto loadStart = std::chrono::high_resolution_clock::now();
    
    std::vector<ImageData> images = loadImagesFromDirectory(config.inputDir);
    
    if (images.empty()) {
        std::cerr << "Error: No images found in " << config.inputDir << "\n";
        return 1;
    }
    
    auto loadEnd = std::chrono::high_resolution_clock::now();
    metrics.loadTime = std::chrono::duration<double, std::milli>(
        loadEnd - loadStart).count();
    metrics.numImages = images.size();
    
    std::cout << "Loaded " << images.size() << " images\n";
    
    // Calculate total bytes
    for (const auto& img : images) {
        metrics.totalBytes += img.width * img.height * img.channels;
    }
    
    // Determine batch size
    if (config.batchSize == 0) {
        config.batchSize = determineOptimalBatchSize(images, config.deviceId);
    }
    
    std::cout << "\nConfiguration:\n";
    std::cout << "  Batch size: " << config.batchSize << " images\n";
    std::cout << "  CUDA streams: " << config.numStreams << "\n";
    std::cout << "  Filter: " << config.filterType << "\n";
    std::cout << "  Total batches: " << (images.size() + config.batchSize - 1) / config.batchSize << "\n\n";
    
    // Initialize stream manager
    StreamManager streamMgr(config.numStreams);
    
    // Process in batches
    std::vector<ImageData> allOutputs;
    int numBatches = (images.size() + config.batchSize - 1) / config.batchSize;
    
    std::cout << "Processing images...\n";
    
    for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        int batchStart = batchIdx * config.batchSize;
        int batchEnd = std::min(batchStart + config.batchSize, 
                                static_cast<int>(images.size()));
        
        std::vector<ImageData> batch(images.begin() + batchStart, 
                                     images.begin() + batchEnd);
        std::vector<ImageData> outputBatch(batch.size());
        
        std::cout << "Batch " << (batchIdx + 1) << "/" << numBatches << ": ";
        
        processBatch(batch, outputBatch, config, streamMgr, metrics);
        
        allOutputs.insert(allOutputs.end(), outputBatch.begin(), outputBatch.end());
    }
    
    // Save results
    std::cout << "\nSaving results to: " << config.outputDir << "\n";
    auto saveStart = std::chrono::high_resolution_clock::now();
    
    ProgressBar saveProgress(allOutputs.size());
    for (size_t i = 0; i < allOutputs.size(); i++) {
        std::string filename = config.outputDir + "/processed_" + 
                              config.filterType + "_" + std::to_string(i) + ".png";
        saveImage(filename, allOutputs[i]);
        saveProgress.update(i + 1);
    }
    saveProgress.finish();
    
    auto saveEnd = std::chrono::high_resolution_clock::now();
    metrics.saveTime = std::chrono::duration<double, std::milli>(
        saveEnd - saveStart).count();
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    metrics.totalTime = std::chrono::duration<double, std::milli>(
        totalEnd - totalStart).count();
    
    // Print summary
    std::cout << "\n=== Processing Complete ===\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) 
              << metrics.totalTime << " ms\n";
    std::cout << "  Load: " << metrics.loadTime << " ms\n";
    std::cout << "  Transfer: " << metrics.transferTime << " ms\n";
    std::cout << "  Kernel: " << metrics.kernelTime << " ms\n";
    std::cout << "  Save: " << metrics.saveTime << " ms\n";
    std::cout << "Average: " << metrics.totalTime / metrics.numImages << " ms/image\n";
    std::cout << "Throughput: " << static_cast<int>(metrics.numImages * 1000.0 / metrics.totalTime) 
              << " images/second\n";
    std::cout << "Total data processed: " << metrics.totalBytes / (1024 * 1024) << " MB\n";
    
    // Save metrics if requested
    if (!config.metricsFile.empty()) {
        saveMetrics(metrics, config.metricsFile);
        std::cout << "Metrics saved to: " << config.metricsFile << "\n";
    }
    
    // Cleanup
    for (auto& img : images) {
        delete[] img.data;
    }
    for (auto& img : allOutputs) {
        delete[] img.data;
    }
    
    return 0;
}