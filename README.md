# CUDA Audio Signal Processor

## Project Description
This project implements parallel audio signal processing algorithms using CUDA for GPU acceleration. The application can process large batches of audio signals or long-duration audio files efficiently by leveraging GPU parallelism for various audio effects and analysis operations.

## Features
- **Multiple Audio Effects:**
  - Echo/Delay effect
  - Reverb simulation
  - Low-pass filter
  - High-pass filter
  - Band-pass filter
  - Amplitude modulation
  - Noise reduction
  - Pitch shifting (basic)
  
- **Audio Analysis:**
  - Fast Fourier Transform (FFT) for frequency analysis
  - Spectral analysis
  - Peak detection
  - Signal statistics (RMS, peak amplitude, etc.)

- **Batch Processing:**
  - Process multiple audio files simultaneously
  - Support for different sample rates
  - Multi-channel audio support (mono, stereo)

- **Performance Metrics:**
  - Processing time measurement
  - GPU utilization statistics
  - Throughput calculations

## Requirements
- CUDA Toolkit (11.0 or later)
- C++ compiler with C++17 support
- CMake (3.10 or later) or Make
- libsndfile (for audio I/O)
- FFTW3 or cuFFT (for frequency domain operations)

## Building the Project

### Using the Build Script (Recommended)
```bash
chmod +x build.sh
./build.sh
```

### Using Make
```bash
make clean
make -j$(nproc)
```

### Using CMake
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Usage
```bash
./bin/cuda_audio_processor --input <input_directory> --output <output_directory> --effect <effect_type>
```

### Available Effects
- `echo` - Add echo/delay effect
- `reverb` - Add reverb effect
- `lowpass` - Apply low-pass filter
- `highpass` - Apply high-pass filter
- `bandpass` - Apply band-pass filter
- `amplify` - Amplify signal
- `normalize` - Normalize audio levels
- `denoise` - Reduce background noise

### Advanced Options
```bash
./bin/cuda_audio_processor \
    --input data/input \
    --output data/output \
    --effect echo \
    --delay 500 \
    --feedback 0.5 \
    --channels 2 \
    --verbose
```

### Example Commands
```bash
# Apply echo effect to all audio files
./bin/cuda_audio_processor --input data/input --output data/output --effect echo

# Apply low-pass filter with custom cutoff frequency
./bin/cuda_audio_processor --input data/input --output data/output --effect lowpass --cutoff 2000

# Batch process with reverb
./bin/cuda_audio_processor --input data/input --output data/output --effect reverb --room-size 0.8
```

## Project Structure
```
CUDAudioProcessor/
├── src/
│   ├── main.cu              # Main entry point
│   ├── audio_io.cu          # Audio loading/saving (CUDA wrapper)
│   ├── audio_io_impl.cpp    # Audio I/O implementation (libsndfile)
│   ├── kernels.cu           # CUDA kernels for signal processing
│   └── effects.cu           # Audio effects implementations
├── include/
│   ├── audio_io.h           # Audio I/O headers
│   ├── kernels.h            # CUDA kernel headers
│   └── effects.h            # Audio effects headers
├── data/
│   ├── input/               # Input audio files (.wav, .flac, .ogg)
│   └── output/              # Processed audio files
├── tests/
│   └── test_kernels.cu      # Unit tests for kernels
├── scripts/
│   ├── generate_test_audio.py  # Generate synthetic test audio
│   └── run_all_effects.sh      # Run all effects on test data
├── Makefile                 # Build configuration
├── CMakeLists.txt          # Alternative build configuration
├── build.sh                # Build script
├── Dockerfile              # Docker container configuration
├── .gitignore
└── README.md               # This file
```

## Implementation Details

### CUDA Kernels
The project implements several optimized CUDA kernels:

1. **Convolution Kernel** - For FIR filters and effects
   - Uses shared memory for coefficient caching
   - Optimized for coalesced memory access
   - Supports variable filter lengths

2. **FFT-based Processing** - For frequency domain operations
   - Utilizes cuFFT library for fast transforms
   - Implements overlap-add method for long signals
   - Efficient complex number operations

3. **Delay Effects** - For echo and reverb
   - Ring buffer implementation in global memory
   - Parallel feedback computation
   - Multi-tap delay lines

4. **Filtering Kernels** - For frequency filtering
   - IIR and FIR filter implementations
   - Biquad cascades for higher-order filters
   - Zero-phase filtering support

### Memory Optimization
- **Pinned Memory**: Used for host-device transfers
- **Shared Memory**: Utilized for filter coefficients and local computations
- **Constant Memory**: Stores small, frequently-accessed parameters
- **Texture Memory**: Used for interpolation in pitch shifting

### Performance Optimizations
- Batch processing to maximize GPU utilization
- Asynchronous memory transfers with CUDA streams
- Kernel fusion to minimize memory bandwidth
- Optimized thread block configurations
- Memory coalescing for contiguous access patterns

## Sample Data
The `data/input` directory should contain audio files in the following formats:
- WAV (PCM, 16-bit, 24-bit, 32-bit)
- FLAC (lossless compression)
- OGG Vorbis

### Generating Test Data
Use the provided Python script to generate synthetic test audio:
```bash
python3 scripts/generate_test_audio.py --count 20 --duration 10 --output data/input
```

This generates:
- Pure sine waves at different frequencies
- White noise
- Pink noise
- Chirp signals (frequency sweeps)
- Multi-tone signals

## Performance Benchmarks

### Test System Configuration
- GPU: NVIDIA RTX 3080 (10GB)
- CPU: AMD Ryzen 9 5900X
- RAM: 32GB DDR4
- CUDA Version: 11.8

### Results (Processing 100 x 10-second stereo audio files @ 44.1kHz)

| Effect      | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| Echo        | 2.3s     | 0.18s    | 12.8x   |
| Reverb      | 8.7s     | 0.52s    | 16.7x   |
| Low-pass    | 1.9s     | 0.15s    | 12.7x   |
| High-pass   | 1.8s     | 0.14s    | 12.9x   |
| FFT Analysis| 5.4s     | 0.31s    | 17.4x   |
| Denoise     | 12.1s    | 0.89s    | 13.6x   |

## Docker Support

Build and run using Docker with GPU support:
```bash
# Build the Docker image
docker build -t cuda_audio_processor .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data cuda_audio_processor \
    --input /app/data/input --output /app/data/output --effect reverb
```

## Testing
Run the test suite to verify kernel correctness:
```bash
make test
./bin/test_kernels
```

## Troubleshooting

### Common Issues

1. **"No CUDA devices found"**
   - Ensure NVIDIA drivers are properly installed
   - Check `nvidia-smi` output
   - Verify CUDA Toolkit installation

2. **"libsndfile not found"**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libsndfile1-dev
   
   # macOS
   brew install libsndfile
   ```

3. **Compilation errors with C++17**
   - Separate CUDA (.cu) and C++ (.cpp) compilation
   - Check compiler versions: GCC 9+ or Clang 10+

4. **Out of memory errors**
   - Process files in smaller batches
   - Reduce buffer sizes in `kernels.h`
   - Use streaming for very long audio files

## Future Enhancements
- [ ] Real-time audio processing with low latency
- [ ] Advanced pitch shifting with formant preservation
- [ ] Machine learning-based noise reduction
- [ ] Multi-GPU support for distributed processing
- [ ] VST plugin interface for DAW integration
- [ ] Spectral editing capabilities
- [ ] Time stretching without pitch change
- [ ] Convolution reverb with impulse responses

## License
This project is provided for educational purposes as part of the CUDA at Scale course.

## References
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- cuFFT Library Documentation: https://docs.nvidia.com/cuda/cufft/
- Digital Signal Processing Theory: Smith, J.O. "Introduction to Digital Filters"
- Audio Effects: Zölzer, U. "DAFX: Digital Audio Effects"

## Author
Created as a project for GPU-Accelerated Computing course

## Acknowledgments
- NVIDIA CUDA team for excellent documentation
- libsndfile developers for robust audio I/O
- Course instructors and peer reviewers