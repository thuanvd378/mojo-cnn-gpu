// == mojo-gpu ================================================================
//    solver.cuh: Optimizers (SGD + Adam) on GPU
// ============================================================================

#pragma once

#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>


// Forward declare cublas handle accessor
namespace mojo {
inline cublasHandle_t cublas();
}

// Error checking macros (if not already defined)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

namespace mojo {

// Base optimizer class
class Optimizer {
public:
  float learning_rate;

  Optimizer(float lr = 0.01f) : learning_rate(lr) {}
  virtual ~Optimizer() = default;

  virtual void update(float *weights, float *gradients, size_t size) = 0;
  virtual void reset() {}
};

// ============================================================================
// SGD with Momentum
// ============================================================================
class SGD : public Optimizer {
public:
  float momentum;
  std::map<float *, float *> velocity; // Velocity buffers per weight

  SGD(float lr = 0.01f, float momentum_ = 0.9f)
      : Optimizer(lr), momentum(momentum_) {}

  ~SGD() {
    for (auto &v : velocity) {
      if (v.second)
        cudaFree(v.second);
    }
  }

  void update(float *weights, float *gradients, size_t size) override {
    // Create velocity buffer if needed
    if (velocity.find(weights) == velocity.end()) {
      float *v;
      CUDA_CHECK(cudaMalloc(&v, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(v, 0, size * sizeof(float)));
      velocity[weights] = v;
    }

    float *v = velocity[weights];

    // v = momentum * v - lr * grad
    // w = w + v
    const float neg_lr = -learning_rate;
    const float mom = momentum;

    // v = momentum * v
    CUBLAS_CHECK(cublasSscal(cublas(), size, &mom, v, 1));

    // v = v - lr * grad
    CUBLAS_CHECK(cublasSaxpy(cublas(), size, &neg_lr, gradients, 1, v, 1));

    // w = w + v
    const float one = 1.0f;
    CUBLAS_CHECK(cublasSaxpy(cublas(), size, &one, v, 1, weights, 1));
  }
};

// ============================================================================
// Adam Optimizer
// ============================================================================
// CUDA kernel for Adam update
__global__ void adamUpdateKernel(float *weights, const float *gradients,
                                 float *m, float *v, float lr, float beta1,
                                 float beta2, float epsilon, float beta1_t,
                                 float beta2_t, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float g = gradients[idx];

  // Update biased first moment estimate
  m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;

  // Update biased second raw moment estimate
  v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

  // Bias-corrected estimates
  float m_hat = m[idx] / (1.0f - beta1_t);
  float v_hat = v[idx] / (1.0f - beta2_t);

  // Update weights
  weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

class Adam : public Optimizer {
public:
  float beta1, beta2, epsilon;
  int t = 0; // Time step

  std::map<float *, float *> m; // First moment
  std::map<float *, float *> v; // Second moment

  Adam(float lr = 0.001f, float beta1_ = 0.9f, float beta2_ = 0.999f,
       float eps = 1e-8f)
      : Optimizer(lr), beta1(beta1_), beta2(beta2_), epsilon(eps) {}

  ~Adam() {
    for (auto &pair : m) {
      if (pair.second)
        cudaFree(pair.second);
    }
    for (auto &pair : v) {
      if (pair.second)
        cudaFree(pair.second);
    }
  }

  void update(float *weights, float *gradients, size_t size) override {
    // Create momentum buffers if needed
    if (m.find(weights) == m.end()) {
      float *m_buf;
      float *v_buf;
      CUDA_CHECK(cudaMalloc(&m_buf, size * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&v_buf, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(m_buf, 0, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(v_buf, 0, size * sizeof(float)));
      m[weights] = m_buf;
      v[weights] = v_buf;
    }

    t++;
    float beta1_t = powf(beta1, t);
    float beta2_t = powf(beta2, t);

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    adamUpdateKernel<<<grid_size, block_size>>>(
        weights, gradients, m[weights], v[weights], learning_rate, beta1, beta2,
        epsilon, beta1_t, beta2_t, size);
    CUDA_CHECK(cudaGetLastError());
  }

  void reset() override {
    t = 0;
    for (auto &pair : m) {
      CUDA_CHECK(cudaMemset(pair.second, 0, sizeof(float)));
    }
    for (auto &pair : v) {
      CUDA_CHECK(cudaMemset(pair.second, 0, sizeof(float)));
    }
  }
};

// Factory function
inline Optimizer *createOptimizer(const std::string &name, float lr = 0.01f) {
  if (name == "adam") {
    return new Adam(lr);
  } else if (name == "sgd") {
    return new SGD(lr);
  }
  return new SGD(lr);
}

} // namespace mojo
