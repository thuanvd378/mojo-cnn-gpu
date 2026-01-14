// == mojo-gpu ================================================================
//    cudnn_utils.cuh: cuDNN and cuBLAS handle management
// ============================================================================

#pragma once

namespace mojo {

// Global handles for cuDNN and cuBLAS
// Singleton pattern for easy access
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

    cudnn_handle = nullptr;
    cublas_handle = nullptr;
    stream = nullptr;
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

// Convenience functions
inline cudnnHandle_t cudnn() { return CudnnHandle::instance().cudnn(); }
inline cublasHandle_t cublas() { return CudnnHandle::instance().cublas(); }
inline cudaStream_t stream() { return CudnnHandle::instance().getStream(); }
inline void sync() { CudnnHandle::instance().sync(); }

// Convolution descriptor wrapper
class ConvolutionDescriptor {
public:
  cudnnConvolutionDescriptor_t desc = nullptr;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  size_t fwd_workspace_size = 0;
  size_t bwd_data_workspace_size = 0;
  size_t bwd_filter_workspace_size = 0;

  void *workspace = nullptr;
  size_t workspace_size = 0;

  int pad_h = 0, pad_w = 0;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  ConvolutionDescriptor() = default;

  void create(int pad_h_, int pad_w_, int stride_h_, int stride_w_,
              int dilation_h_ = 1, int dilation_w_ = 1) {
    if (desc)
      destroy();

    pad_h = pad_h_;
    pad_w = pad_w_;
    stride_h = stride_h_;
    stride_w = stride_w_;
    dilation_h = dilation_h_;
    dilation_w = dilation_w_;

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Enable tensor cores if available
    CUDNN_CHECK(cudnnSetConvolutionMathType(
        desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  }

  void findBestAlgorithms(cudnnTensorDescriptor_t input_desc,
                          cudnnFilterDescriptor_t filter_desc,
                          cudnnTensorDescriptor_t output_desc) {
    // Forward algorithm
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn(), input_desc, filter_desc, desc, output_desc, 1,
        &returned_algo_count, &fwd_perf));
    fwd_algo = fwd_perf.algo;

    // Get workspace size for forward
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn(), input_desc, filter_desc, desc, output_desc, fwd_algo,
        &fwd_workspace_size));

    // Backward data algorithm
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn(), filter_desc, output_desc, desc, input_desc, 1,
        &returned_algo_count, &bwd_data_perf));
    bwd_data_algo = bwd_data_perf.algo;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn(), filter_desc, output_desc, desc, input_desc, bwd_data_algo,
        &bwd_data_workspace_size));

    // Backward filter algorithm
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn(), input_desc, output_desc, desc, filter_desc, 1,
        &returned_algo_count, &bwd_filter_perf));
    bwd_filter_algo = bwd_filter_perf.algo;

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn(), input_desc, output_desc, desc, filter_desc, bwd_filter_algo,
        &bwd_filter_workspace_size));

    // Allocate workspace (max of all)
    workspace_size = std::max({fwd_workspace_size, bwd_data_workspace_size,
                               bwd_filter_workspace_size});
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

// Pooling descriptor wrapper
class PoolingDescriptor {
public:
  cudnnPoolingDescriptor_t desc = nullptr;

  int window_h = 2, window_w = 2;
  int pad_h = 0, pad_w = 0;
  int stride_h = 2, stride_w = 2;

  void create(int window_h_, int window_w_, int stride_h_, int stride_w_,
              int pad_h_ = 0, int pad_w_ = 0,
              cudnnPoolingMode_t mode = CUDNN_POOLING_MAX) {
    if (desc)
      destroy();

    window_h = window_h_;
    window_w = window_w_;
    stride_h = stride_h_;
    stride_w = stride_w_;
    pad_h = pad_h_;
    pad_w = pad_w_;

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc, mode, CUDNN_NOT_PROPAGATE_NAN,
                                            window_h, window_w, pad_h, pad_w,
                                            stride_h, stride_w));
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
