// Simple CUDA vs CPU Benchmark
// Tests convolution operation speed: CPU vs GPU

#include "../mojo-gpu/cuda_kernels.cuh"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

// CPU Convolution (Simple implementation)
void conv_cpu(const float *input, const float *kernel, float *output, int width,
              int height, int kernel_size) {
  int pad = kernel_size / 2;
  int out_size = width * height;

  for (int i = 0; i < out_size; i++) {
    output[i] = 0.0f;
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0.0f;

      for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
          int iy = y - pad + ky;
          int ix = x - pad + kx;

          if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
          }
        }
      }

      output[y * width + x] = sum;
    }
  }
}

int main() {
  cout << "==========================================================" << endl;
  cout << "       MOJO CNN - GPU (CUDA) Performance Test" << endl;
  cout << "==========================================================" << endl;
  cout << endl;

  // Test parameters
  const int width = 28;
  const int height = 28;
  const int kernel_size = 5;
  const int batch_size = 5000; // Large batch for accurate timing

  int input_size = width * height;
  int kernel_elem = kernel_size * kernel_size;

  cout << "Test Configuration:" << endl;
  cout << "  Image size: " << width << "x" << height << endl;
  cout << "  Kernel size: " << kernel_size << "x" << kernel_size << endl;
  cout << "  Batch size: " << batch_size << " images" << endl;
  cout << endl;

  // Allocate host memory
  float *h_input = new float[input_size * batch_size];
  float *h_kernel = new float[kernel_elem];
  float *h_output_cpu = new float[input_size * batch_size];
  float *h_output_gpu = new float[input_size * batch_size];

  // Fill with random data
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (int i = 0; i < input_size * batch_size; i++) {
    h_input[i] = dis(gen);
  }
  for (int i = 0; i < kernel_elem; i++) {
    h_kernel[i] = dis(gen);
  }

  cout << "==========================================================" << endl;
  cout << "Running CPU Convolution..." << endl;
  auto cpu_start = high_resolution_clock::now();

  for (int b = 0; b < batch_size; b++) {
    conv_cpu(h_input + b * input_size, h_kernel, h_output_cpu + b * input_size,
             width, height, kernel_size);
  }

  auto cpu_end = high_resolution_clock::now();
  auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();

  cout << "  CPU Time: " << cpu_duration << " ms" << endl;
  cout << "  Speed: " << (float)batch_size / (cpu_duration / 1000.0f)
       << " images/sec" << endl;
  cout << endl;

  cout << "==========================================================" << endl;
  cout << "Running GPU Convolution (CUDA)..." << endl;

  // Allocate GPU memory
  float *d_input = mojo::cuda::gpu_allocate<float>(input_size * batch_size);
  float *d_kernel = mojo::cuda::gpu_allocate<float>(kernel_elem);
  float *d_output = mojo::cuda::gpu_allocate<float>(input_size * batch_size);

  // Copy to GPU
  mojo::cuda::gpu_copy_host_to_device(d_input, h_input,
                                      input_size * batch_size);
  mojo::cuda::gpu_copy_host_to_device(d_kernel, h_kernel, kernel_elem);

  auto gpu_start = high_resolution_clock::now();

  // Run GPU convolution for batch
  mojo::cuda::conv2d_forward_gpu(d_input, d_kernel, nullptr, d_output,
                                 batch_size, 1, height, width, 1, kernel_size,
                                 kernel_size, 1, 2);

  mojo::cuda::gpu_synchronize();

  auto gpu_end = high_resolution_clock::now();
  auto gpu_duration = duration_cast<milliseconds>(gpu_end - gpu_start).count();

  // Copy back
  mojo::cuda::gpu_copy_device_to_host(h_output_gpu, d_output,
                                      input_size * batch_size);

  cout << "  GPU Time: " << gpu_duration << " ms" << endl;
  cout << "  Speed: " << (float)batch_size / (gpu_duration / 1000.0f)
       << " images/sec" << endl;
  cout << endl;

  // Calculate speedup
  float speedup =
      (float)cpu_duration / (float)(gpu_duration > 0 ? gpu_duration : 1);

  cout << "==========================================================" << endl;
  cout << "RESULTS:" << endl;
  cout << "  CPU: " << cpu_duration << " ms" << endl;
  cout << "  GPU: " << gpu_duration << " ms" << endl;
  cout << "  SPEEDUP: " << speedup << "x FASTER on GPU!" << endl;
  cout << "==========================================================" << endl;
  cout << endl;

  // Verify correctness (check first image)
  float max_diff = 0.0f;
  for (int i = 0; i < min(100, input_size); i++) {
    float diff = abs(h_output_cpu[i] - h_output_gpu[i]);
    max_diff = max(max_diff, diff);
  }

  cout << "Verification:" << endl;
  cout << "  Max difference between CPU and GPU: " << max_diff << endl;
  if (max_diff < 0.01f) {
    cout << "  ✓ PASSED - Results match!" << endl;
  } else {
    cout << "  ✗ FAILED - Results differ!" << endl;
  }
  cout << endl;

  // Cleanup
  delete[] h_input;
  delete[] h_kernel;
  delete[] h_output_cpu;
  delete[] h_output_gpu;

  mojo::cuda::gpu_free(d_input);
  mojo::cuda::gpu_free(d_kernel);
  mojo::cuda::gpu_free(d_output);

  return 0;
}
