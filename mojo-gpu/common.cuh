// == mojo-gpu ================================================================
//    common.cuh: Common includes, macros, and base types
// ============================================================================

#pragma once

// CUDA and cuDNN
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

// Standard libraries
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Error checking macros
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDNN_CHECK(call)                                                      \
  do {                                                                         \
    cudnnStatus_t status = call;                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << " at "    \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace mojo {

// ============================================================================
// Global Handles (Singleton)
// ============================================================================
class CudnnHandle {
private:
  cudnnHandle_t cudnn_handle = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  cudaStream_t stream = nullptr;
  bool initialized = false;

  CudnnHandle() {}

public:
  static CudnnHandle &instance() {
    static CudnnHandle inst;
    return inst;
  }

  void init() {
    if (initialized)
      return;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    initialized = true;
  }

  void destroy() {
    if (!initialized)
      return;
    cudnnDestroy(cudnn_handle);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    initialized = false;
  }

  cudnnHandle_t cudnn() {
    if (!initialized)
      init();
    return cudnn_handle;
  }
  cublasHandle_t cublas() {
    if (!initialized)
      init();
    return cublas_handle;
  }
  cudaStream_t getStream() {
    if (!initialized)
      init();
    return stream;
  }
  void sync() { CUDA_CHECK(cudaStreamSynchronize(stream)); }
  ~CudnnHandle() { destroy(); }
};

inline cudnnHandle_t cudnn() { return CudnnHandle::instance().cudnn(); }
inline cublasHandle_t cublas() { return CudnnHandle::instance().cublas(); }
inline cudaStream_t stream() { return CudnnHandle::instance().getStream(); }
inline void sync() { CudnnHandle::instance().sync(); }

// ============================================================================
// Tensor Class
// ============================================================================
class Tensor {
public:
  float *d_data = nullptr;
  float *h_data = nullptr;
  int n = 0, c = 0, h = 0, w = 0;
  cudnnTensorDescriptor_t desc = nullptr;
  bool desc_created = false;
  bool owns_data = true;

  Tensor() = default;
  Tensor(int n_, int c_, int h_, int w_) { allocate(n_, c_, h_, w_); }
  ~Tensor() { free(); }

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  Tensor(Tensor &&other) noexcept {
    d_data = other.d_data;
    h_data = other.h_data;
    n = other.n;
    c = other.c;
    h = other.h;
    w = other.w;
    desc = other.desc;
    desc_created = other.desc_created;
    owns_data = other.owns_data;
    other.d_data = nullptr;
    other.h_data = nullptr;
    other.desc = nullptr;
    other.desc_created = false;
    other.owns_data = false;
  }

  void allocate(int n_, int c_, int h_, int w_) {
    if (d_data && n == n_ && c == c_ && h == h_ && w == w_)
      return;
    free();
    n = n_;
    c = c_;
    this->h = h_;
    this->w = w_;
    size_t bytes = size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, n, c, h_, w_));
    desc_created = true;
    owns_data = true;
  }

  void free() {
    if (owns_data) {
      if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
      }
      if (h_data) {
        cudaFreeHost(h_data);
        h_data = nullptr;
      }
      if (desc_created) {
        cudnnDestroyTensorDescriptor(desc);
        desc_created = false;
      }
    }
    n = c = h = w = 0;
  }

  inline size_t size() const { return (size_t)n * c * h * w; }
  inline size_t bytes() const { return size() * sizeof(float); }

  void toDevice(cudaStream_t s = 0) {
    CUDA_CHECK(
        cudaMemcpyAsync(d_data, h_data, bytes(), cudaMemcpyHostToDevice, s));
  }
  void toHost(cudaStream_t s = 0) {
    CUDA_CHECK(
        cudaMemcpyAsync(h_data, d_data, bytes(), cudaMemcpyDeviceToHost, s));
  }
  void toDeviceSync() {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes(), cudaMemcpyHostToDevice));
  }
  void toHostSync() {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes(), cudaMemcpyDeviceToHost));
  }
  void zero() { CUDA_CHECK(cudaMemset(d_data, 0, bytes())); }
  void resizeBatch(int new_n) {
    if (new_n != n)
      allocate(new_n, c, h, w);
  }
};

// ============================================================================
// Filter Tensor
// ============================================================================
class FilterTensor {
public:
  float *d_data = nullptr;
  float *h_data = nullptr;
  int k = 0, c = 0, h = 0, w = 0;
  cudnnFilterDescriptor_t desc = nullptr;
  bool desc_created = false;

  FilterTensor() = default;
  FilterTensor(int k_, int c_, int h_, int w_) { allocate(k_, c_, h_, w_); }
  ~FilterTensor() { free(); }

  void allocate(int k_, int c_, int h_, int w_) {
    if (d_data && k == k_ && c == c_ && h == h_ && w == w_)
      return;
    free();
    k = k_;
    c = c_;
    this->h = h_;
    this->w = w_;
    size_t bytes_sz = size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data, bytes_sz));
    CUDA_CHECK(cudaMallocHost(&h_data, bytes_sz));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, k, c, h_, w_));
    desc_created = true;
  }

  void free() {
    if (d_data) {
      cudaFree(d_data);
      d_data = nullptr;
    }
    if (h_data) {
      cudaFreeHost(h_data);
      h_data = nullptr;
    }
    if (desc_created) {
      cudnnDestroyFilterDescriptor(desc);
      desc_created = false;
    }
    k = c = h = w = 0;
  }

  inline size_t size() const { return (size_t)k * c * h * w; }
  inline size_t bytes() const { return size() * sizeof(float); }

  void toDevice() {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes(), cudaMemcpyHostToDevice));
  }
  void toHost() {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes(), cudaMemcpyDeviceToHost));
  }

  void initXavier() {
    float scale = sqrtf(2.0f / (float)(c * h * w));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < size(); i++)
      h_data[i] = dist(gen);
    toDevice();
  }
};

// ============================================================================
// Activation Types
// ============================================================================
enum class ActivationType { NONE, RELU, ELU, SIGMOID, TANH, SOFTMAX };

inline ActivationType parseActivation(const std::string &name) {
  if (name == "relu")
    return ActivationType::RELU;
  if (name == "elu")
    return ActivationType::ELU;
  if (name == "sigmoid")
    return ActivationType::SIGMOID;
  if (name == "tanh")
    return ActivationType::TANH;
  if (name == "softmax")
    return ActivationType::SOFTMAX;
  return ActivationType::NONE;
}

inline std::string activationName(ActivationType type) {
  switch (type) {
  case ActivationType::RELU:
    return "relu";
  case ActivationType::ELU:
    return "elu";
  case ActivationType::SIGMOID:
    return "sigmoid";
  case ActivationType::TANH:
    return "tanh";
  case ActivationType::SOFTMAX:
    return "softmax";
  default:
    return "none";
  }
}

// ============================================================================
// Fused Bias + Activation CUDA Kernel (avoids cuDNN descriptor overhead)
// ============================================================================
__device__ __forceinline__ float apply_activation(float x, int act_type) {
  switch (act_type) {
  case 1:
    return fmaxf(x, 0.0f); // RELU
  case 2:
    return x >= 0.0f ? x : (expf(x) - 1.0f); // ELU
  case 3:
    return 1.0f / (1.0f + expf(-x)); // SIGMOID
  case 4:
    return tanhf(x); // TANH
  default:
    return x; // NONE
  }
}

__global__ void fusedBiasActivationKernel(float *__restrict__ output,
                                          const float *__restrict__ bias,
                                          int batch_size, int channels,
                                          int spatial, int act_type) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * channels * spatial;
  if (idx >= total)
    return;

  int c = (idx / spatial) % channels;
  output[idx] = apply_activation(output[idx] + bias[c], act_type);
}

inline void fusedBiasActivation(Tensor &output, const Tensor &bias,
                                ActivationType type) {
  int total = output.n * output.c * output.h * output.w;
  int spatial = output.h * output.w;
  int act_type = 0;
  if (type == ActivationType::RELU)
    act_type = 1;
  else if (type == ActivationType::ELU)
    act_type = 2;
  else if (type == ActivationType::SIGMOID)
    act_type = 3;
  else if (type == ActivationType::TANH)
    act_type = 4;

  int block = 256;
  int grid = (total + block - 1) / block;
  fusedBiasActivationKernel<<<grid, block, 0, stream()>>>(
      output.d_data, bias.d_data, output.n, output.c, spatial, act_type);
}

// Forward activation (in-place)
inline void activationForward(Tensor &x, ActivationType type) {
  if (type == ActivationType::NONE)
    return;
  const float alpha = 1.0f, beta = 0.0f;

  if (type == ActivationType::SOFTMAX) {
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn(), CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, x.desc,
                                    x.d_data, &beta, x.desc, x.d_data));
  } else {
    cudnnActivationDescriptor_t act_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (type == ActivationType::RELU)
      mode = CUDNN_ACTIVATION_RELU;
    else if (type == ActivationType::ELU)
      mode = CUDNN_ACTIVATION_ELU;
    else if (type == ActivationType::SIGMOID)
      mode = CUDNN_ACTIVATION_SIGMOID;
    else if (type == ActivationType::TANH)
      mode = CUDNN_ACTIVATION_TANH;
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, mode,
                                             CUDNN_NOT_PROPAGATE_NAN, 1.0));
    CUDNN_CHECK(cudnnActivationForward(cudnn(), act_desc, &alpha, x.desc,
                                       x.d_data, &beta, x.desc, x.d_data));
    cudnnDestroyActivationDescriptor(act_desc);
  }
}

inline void activationBackward(Tensor &dx, const Tensor &dy, const Tensor &y,
                               const Tensor &x, ActivationType type) {
  if (type == ActivationType::NONE) {
    CUDA_CHECK(
        cudaMemcpy(dx.d_data, dy.d_data, dy.bytes(), cudaMemcpyDeviceToDevice));
    return;
  }
  const float alpha = 1.0f, beta = 0.0f;
  if (type == ActivationType::SOFTMAX) {
    CUDNN_CHECK(cudnnSoftmaxBackward(
        cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
        y.desc, y.d_data, dy.desc, dy.d_data, &beta, dx.desc, dx.d_data));
  } else {
    cudnnActivationDescriptor_t act_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (type == ActivationType::RELU)
      mode = CUDNN_ACTIVATION_RELU;
    else if (type == ActivationType::ELU)
      mode = CUDNN_ACTIVATION_ELU;
    else if (type == ActivationType::SIGMOID)
      mode = CUDNN_ACTIVATION_SIGMOID;
    else if (type == ActivationType::TANH)
      mode = CUDNN_ACTIVATION_TANH;
    CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, mode,
                                             CUDNN_NOT_PROPAGATE_NAN, 1.0));
    CUDNN_CHECK(cudnnActivationBackward(cudnn(), act_desc, &alpha, y.desc,
                                        y.d_data, dy.desc, dy.d_data, x.desc,
                                        x.d_data, &beta, dx.desc, dx.d_data));
    cudnnDestroyActivationDescriptor(act_desc);
  }
}

// ============================================================================
// Convolution Descriptor
// ============================================================================
class ConvolutionDescriptor {
public:
  cudnnConvolutionDescriptor_t desc = nullptr;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  size_t workspace_size = 0;
  void *workspace = nullptr;

  void create(int pad_h, int pad_w, int stride_h, int stride_w) {
    if (desc)
      destroy();
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(
        desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  }

  void findBestAlgorithms(cudnnTensorDescriptor_t in, cudnnFilterDescriptor_t w,
                          cudnnTensorDescriptor_t out) {
    int cnt;
    cudnnConvolutionFwdAlgoPerf_t fp;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn(), in, w, desc,
                                                       out, 1, &cnt, &fp));
    fwd_algo = fp.algo;
    cudnnConvolutionBwdDataAlgoPerf_t bdp;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn(), w, out, desc, in, 1, &cnt, &bdp));
    bwd_data_algo = bdp.algo;
    cudnnConvolutionBwdFilterAlgoPerf_t bfp;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn(), in, out, desc, w, 1, &cnt, &bfp));
    bwd_filter_algo = bfp.algo;

    size_t s1, s2, s3;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn(), in, w, desc,
                                                        out, fwd_algo, &s1));
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn(), w, out, desc, in, bwd_data_algo, &s2));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn(), in, out, desc, w, bwd_filter_algo, &s3));
    workspace_size = std::max({s1, s2, s3});
    if (workspace_size > 0) {
      if (workspace)
        cudaFree(workspace);
      CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
  }

  void destroy() {
    if (desc) {
      cudnnDestroyConvolutionDescriptor(desc);
      desc = nullptr;
    }
    if (workspace) {
      cudaFree(workspace);
      workspace = nullptr;
    }
  }
  ~ConvolutionDescriptor() { destroy(); }
};

// ============================================================================
// Pooling Descriptor
// ============================================================================
class PoolingDescriptor {
public:
  cudnnPoolingDescriptor_t desc = nullptr;

  void create(int h, int w, int sh, int sw,
              cudnnPoolingMode_t mode = CUDNN_POOLING_MAX) {
    if (desc)
      destroy();
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc, mode, CUDNN_NOT_PROPAGATE_NAN,
                                            h, w, 0, 0, sh, sw));
  }
  void destroy() {
    if (desc) {
      cudnnDestroyPoolingDescriptor(desc);
      desc = nullptr;
    }
  }
  ~PoolingDescriptor() { destroy(); }
};

} // namespace mojo
