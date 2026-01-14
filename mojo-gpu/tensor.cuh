// == mojo-gpu ================================================================
//    tensor.cuh: GPU-native Tensor class with cuDNN descriptors
// ============================================================================

#pragma once

namespace mojo {

// Tensor class - GPU-native data container
// Follows NCHW layout for cuDNN compatibility
class Tensor {
public:
  float *d_data = nullptr; // GPU data (primary storage)
  float *h_data = nullptr; // CPU data (pinned memory for fast transfer)

  int n = 0; // batch size
  int c = 0; // channels
  int h = 0; // height
  int w = 0; // width

  cudnnTensorDescriptor_t desc = nullptr;
  bool desc_created = false;
  bool owns_data = true;

  Tensor() = default;

  Tensor(int n_, int c_, int h_, int w_) { allocate(n_, c_, h_, w_); }

  ~Tensor() { free(); }

  // No copy (use move semantics)
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  // Move constructor
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

  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      free();
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
    return *this;
  }

  void allocate(int n_, int c_, int h_, int w_) {
    if (d_data && n == n_ && c == c_ && h == h_ && w == w_) {
      return; // Already allocated with same size
    }

    free();

    n = n_;
    c = c_;
    h = h_;
    w = w_;
    size_t bytes = size() * sizeof(float);

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    // Allocate pinned host memory for fast transfers
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));

    // Create cuDNN tensor descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, n, c, h, w));
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

  // Copy from host (pinned) to device - async
  void toDevice(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes(), cudaMemcpyHostToDevice,
                               stream));
  }

  // Copy from device to host (pinned) - async
  void toHost(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, bytes(), cudaMemcpyDeviceToHost,
                               stream));
  }

  // Synchronous versions
  void toDeviceSync() {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes(), cudaMemcpyHostToDevice));
  }

  void toHostSync() {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes(), cudaMemcpyDeviceToHost));
  }

  // Fill with zeros
  void zero() { CUDA_CHECK(cudaMemset(d_data, 0, bytes())); }

  // Fill with value
  void fill(float value);

  // Resize batch dimension (keep c, h, w)
  void resizeBatch(int new_n) {
    if (new_n == n)
      return;
    allocate(new_n, c, h, w);
  }

  // Debug print
  void print(const std::string &name = "Tensor", int max_elements = 10) const {
    std::vector<float> tmp(size());
    cudaMemcpy(tmp.data(), d_data, bytes(), cudaMemcpyDeviceToHost);

    std::cout << name << " [" << n << "," << c << "," << h << "," << w << "]: ";
    for (int i = 0; i < std::min((int)size(), max_elements); i++) {
      std::cout << tmp[i] << " ";
    }
    if ((int)size() > max_elements)
      std::cout << "...";
    std::cout << std::endl;
  }
};

// Filter tensor for convolution weights
class FilterTensor {
public:
  float *d_data = nullptr;
  float *h_data = nullptr;

  int k = 0; // num output channels (filters)
  int c = 0; // num input channels
  int h = 0; // filter height
  int w = 0; // filter width

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
    h = h_;
    w = w_;
    size_t bytes = size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, k, c, h, w));
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

  // Xavier initialization
  void initXavier() {
    float scale = sqrtf(2.0f / (float)(c * h * w));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (size_t i = 0; i < size(); i++) {
      h_data[i] = dist(gen);
    }
    toDevice();
  }
};

} // namespace mojo
