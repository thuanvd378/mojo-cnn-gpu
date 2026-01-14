// == mojo-gpu ================================================================
//    layers.cuh: Neural network layers using cuDNN
// ============================================================================

#pragma once

namespace mojo {

// Base layer class
class Layer {
public:
  std::string name;
  Tensor output; // Output of this layer (y = f(x))
  Tensor delta;  // Gradient w.r.t. output (dL/dy)
  ActivationType activation = ActivationType::NONE;

  int batch_size = 0;

  virtual ~Layer() = default;

  virtual void forward(const Tensor &input) = 0;
  virtual void backward(const Tensor &input, Tensor &input_delta) = 0;
  virtual void updateWeights(float lr) {}
  virtual void resizeBatch(int n) { batch_size = n; }

  virtual std::string getConfig() const = 0;
  virtual int outputSize() const { return output.c * output.h * output.w; }
};

// ============================================================================
// Input Layer
// ============================================================================
class InputLayer : public Layer {
public:
  InputLayer(const std::string &name_, int h, int w, int c = 1) {
    name = name_;
    output.allocate(1, c, h, w); // Will resize batch later
  }

  void resizeBatch(int n) override {
    batch_size = n;
    output.allocate(n, output.c, output.h, output.w);
    delta.allocate(n, output.c, output.h, output.w);
  }

  void forward(const Tensor &input) override {
    // Just copy input to output (input layer doesn't transform)
    CUDA_CHECK(cudaMemcpy(output.d_data, input.d_data, input.bytes(),
                          cudaMemcpyDeviceToDevice));
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    // Nothing to do for input layer
  }

  std::string getConfig() const override {
    return "input " + std::to_string(output.w) + " " +
           std::to_string(output.h) + " " + std::to_string(output.c);
  }
};

// ============================================================================
// Convolution Layer (cuDNN)
// ============================================================================
class Conv2DLayer : public Layer {
public:
  FilterTensor weights;
  Tensor bias;
  FilterTensor grad_weights;
  Tensor grad_bias;

  ConvolutionDescriptor conv_desc;
  cudnnTensorDescriptor_t bias_desc = nullptr;

  int kernel_size;
  int num_filters;
  int stride;
  int padding;

  Tensor pre_activation; // Store for backward pass

  Conv2DLayer(const std::string &name_, int kernel_size_, int num_filters_,
              int stride_ = 1, ActivationType act = ActivationType::RELU,
              int in_channels = 0, int in_h = 0, int in_w = 0) {
    name = name_;
    kernel_size = kernel_size_;
    num_filters = num_filters_;
    stride = stride_;
    activation = act;
    padding = kernel_size / 2; // Same padding

    // Weights will be initialized when we know input size
  }

  ~Conv2DLayer() {
    if (bias_desc)
      cudnnDestroyTensorDescriptor(bias_desc);
  }

  void init(int in_channels, int in_h, int in_w, int batch_size_) {
    batch_size = batch_size_;

    // Calculate output size
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate tensors
    weights.allocate(num_filters, in_channels, kernel_size, kernel_size);
    weights.initXavier();

    bias.allocate(1, num_filters, 1, 1);
    CUDA_CHECK(cudaMemset(bias.d_data, 0, bias.bytes()));

    grad_weights.allocate(num_filters, in_channels, kernel_size, kernel_size);
    grad_bias.allocate(1, num_filters, 1, 1);

    output.allocate(batch_size, num_filters, out_h, out_w);
    delta.allocate(batch_size, num_filters, out_h, out_w);
    pre_activation.allocate(batch_size, num_filters, out_h, out_w);

    // Setup convolution
    conv_desc.create(padding, padding, stride, stride);

    // Bias descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_filters, 1, 1));
  }

  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;

    int out_h = output.h;
    int out_w = output.w;

    output.allocate(n, num_filters, out_h, out_w);
    delta.allocate(n, num_filters, out_h, out_w);
    pre_activation.allocate(n, num_filters, out_h, out_w);
  }

  void forward(const Tensor &input) override {
    const float alpha = 1.0f, beta = 0.0f;

    // Find best algorithm if not done
    conv_desc.findBestAlgorithms(input.desc, weights.desc, output.desc);

    // Convolution
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn(), &alpha, input.desc, input.d_data, weights.desc, weights.d_data,
        conv_desc.desc, conv_desc.fwd_algo, conv_desc.workspace,
        conv_desc.workspace_size, &beta, output.desc, output.d_data));

    // Add bias
    CUDNN_CHECK(cudnnAddTensor(cudnn(), &alpha, bias_desc, bias.d_data, &alpha,
                               output.desc, output.d_data));

    // Store pre-activation for backward
    CUDA_CHECK(cudaMemcpy(pre_activation.d_data, output.d_data, output.bytes(),
                          cudaMemcpyDeviceToDevice));

    // Activation
    activationForward(output, activation);
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    const float alpha = 1.0f, beta = 0.0f;

    // Activation backward
    Tensor act_grad;
    act_grad.allocate(delta.n, delta.c, delta.h, delta.w);
    activationBackward(act_grad, delta, output, pre_activation, activation);

    // Bias gradient
    CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn(), &alpha, act_grad.desc,
                                             act_grad.d_data, &beta, bias_desc,
                                             grad_bias.d_data));

    // Weight gradient
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn(), &alpha, input.desc, input.d_data, act_grad.desc,
        act_grad.d_data, conv_desc.desc, conv_desc.bwd_filter_algo,
        conv_desc.workspace, conv_desc.workspace_size, &beta, grad_weights.desc,
        grad_weights.d_data));

    // Input gradient (if not first layer)
    if (input_delta.d_data) {
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          cudnn(), &alpha, weights.desc, weights.d_data, act_grad.desc,
          act_grad.d_data, conv_desc.desc, conv_desc.bwd_data_algo,
          conv_desc.workspace, conv_desc.workspace_size, &beta,
          input_delta.desc, input_delta.d_data));
    }
  }

  void updateWeights(float lr) override {
    // W = W - lr * grad_W
    const float neg_lr = -lr;
    CUBLAS_CHECK(cublasSaxpy(cublas(), weights.size(), &neg_lr,
                             grad_weights.d_data, 1, weights.d_data, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), bias.size(), &neg_lr, grad_bias.d_data,
                             1, bias.d_data, 1));
  }

  std::string getConfig() const override {
    return "convolution " + std::to_string(kernel_size) + " " +
           std::to_string(num_filters) + " " + std::to_string(stride) + " " +
           activationName(activation);
  }
};

// ============================================================================
// Max Pooling Layer (cuDNN)
// ============================================================================
class MaxPoolLayer : public Layer {
public:
  PoolingDescriptor pool_desc;
  int pool_size;
  int stride;

  Tensor indices; // For backward pass

  MaxPoolLayer(const std::string &name_, int pool_size_, int stride_ = 0) {
    name = name_;
    pool_size = pool_size_;
    stride = stride_ > 0 ? stride_ : pool_size_;
    activation = ActivationType::NONE;

    pool_desc.create(pool_size, pool_size, stride, stride);
  }

  void init(int in_channels, int in_h, int in_w, int batch_size_) {
    batch_size = batch_size_;

    int out_h = (in_h - pool_size) / stride + 1;
    int out_w = (in_w - pool_size) / stride + 1;

    output.allocate(batch_size, in_channels, out_h, out_w);
    delta.allocate(batch_size, in_channels, out_h, out_w);
  }

  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;
    output.allocate(n, output.c, output.h, output.w);
    delta.allocate(n, output.c, output.h, output.w);
  }

  void forward(const Tensor &input) override {
    const float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnPoolingForward(cudnn(), pool_desc.desc, &alpha, input.desc,
                                    input.d_data, &beta, output.desc,
                                    output.d_data));
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    const float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnPoolingBackward(
        cudnn(), pool_desc.desc, &alpha, output.desc, output.d_data, delta.desc,
        delta.d_data, input.desc, input.d_data, &beta, input_delta.desc,
        input_delta.d_data));
  }

  std::string getConfig() const override {
    return "max_pool " + std::to_string(pool_size) + " " +
           std::to_string(stride);
  }
};

// ============================================================================
// Fully Connected Layer (cuBLAS)
// ============================================================================
class FCLayer : public Layer {
public:
  Tensor weights; // [out_features, in_features]
  Tensor bias;    // [out_features]
  Tensor grad_weights;
  Tensor grad_bias;

  Tensor pre_activation;

  int in_features;
  int out_features;

  FCLayer(const std::string &name_, int out_features_,
          ActivationType act = ActivationType::NONE) {
    name = name_;
    out_features = out_features_;
    activation = act;
  }

  void init(int in_features_, int batch_size_) {
    in_features = in_features_;
    batch_size = batch_size_;

    // Weight matrix: [out_features, in_features]
    weights.allocate(1, 1, out_features, in_features);
    bias.allocate(1, out_features, 1, 1);
    grad_weights.allocate(1, 1, out_features, in_features);
    grad_bias.allocate(1, out_features, 1, 1);

    // Xavier initialization
    float scale = sqrtf(2.0f / in_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (size_t i = 0; i < weights.size(); i++) {
      weights.h_data[i] = dist(gen);
    }
    weights.toDeviceSync();
    CUDA_CHECK(cudaMemset(bias.d_data, 0, bias.bytes()));

    output.allocate(batch_size, out_features, 1, 1);
    delta.allocate(batch_size, out_features, 1, 1);
    pre_activation.allocate(batch_size, out_features, 1, 1);
  }

  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;
    output.allocate(n, out_features, 1, 1);
    delta.allocate(n, out_features, 1, 1);
    pre_activation.allocate(n, out_features, 1, 1);
  }

  void forward(const Tensor &input) override {
    // Y = X * W^T + b
    // X: [batch, in_features], W: [out_features, in_features]
    // Y: [batch, out_features]

    const float alpha = 1.0f, beta = 0.0f;

    // GEMM: C = alpha * A * B + beta * C
    // Y = X * W^T
    CUBLAS_CHECK(cublasSgemm(
        cublas(),
        CUBLAS_OP_T,                         // W transposed
        CUBLAS_OP_N,                         // X not transposed
        out_features,                        // M: rows of W^T = out_features
        batch_size,                          // N: cols of X = batch_size
        in_features,                         // K: cols of W^T = in_features
        &alpha, weights.d_data, in_features, // W: [out, in] stored row-major
        input.d_data, in_features,           // X: [batch, in]
        &beta, output.d_data, out_features   // Y: [batch, out]
        ));

    // Add bias (broadcast)
    // Each row of Y += bias
    for (int b = 0; b < batch_size; b++) {
      CUBLAS_CHECK(cublasSaxpy(cublas(), out_features, &alpha, bias.d_data, 1,
                               output.d_data + b * out_features, 1));
    }

    // Store pre-activation
    CUDA_CHECK(cudaMemcpy(pre_activation.d_data, output.d_data, output.bytes(),
                          cudaMemcpyDeviceToDevice));

    // Activation
    activationForward(output, activation);
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    const float alpha = 1.0f, beta = 0.0f;

    // Activation backward
    Tensor act_grad;
    act_grad.allocate(delta.n, delta.c, delta.h, delta.w);
    activationBackward(act_grad, delta, output, pre_activation, activation);

    // Bias gradient: sum over batch
    Tensor ones;
    ones.allocate(1, batch_size, 1, 1);
    std::fill(ones.h_data, ones.h_data + batch_size, 1.0f);
    ones.toDeviceSync();

    CUBLAS_CHECK(cublasSgemv(cublas(), CUBLAS_OP_N, out_features, batch_size,
                             &alpha, act_grad.d_data, out_features, ones.d_data,
                             1, &beta, grad_bias.d_data, 1));

    // Weight gradient: dW = dY^T * X
    CUBLAS_CHECK(cublasSgemm(cublas(),
                             CUBLAS_OP_N,  // dY not transposed
                             CUBLAS_OP_T,  // X transposed
                             out_features, // M
                             in_features,  // N
                             batch_size,   // K
                             &alpha, act_grad.d_data, out_features,
                             input.d_data, in_features, &beta,
                             grad_weights.d_data, out_features));

    // Input gradient: dX = dY * W
    if (input_delta.d_data) {
      CUBLAS_CHECK(cublasSgemm(cublas(),
                               CUBLAS_OP_N,  // W not transposed
                               CUBLAS_OP_N,  // dY not transposed
                               in_features,  // M
                               batch_size,   // N
                               out_features, // K
                               &alpha, weights.d_data, in_features,
                               act_grad.d_data, out_features, &beta,
                               input_delta.d_data, in_features));
    }
  }

  void updateWeights(float lr) override {
    const float neg_lr = -lr;
    CUBLAS_CHECK(cublasSaxpy(cublas(), weights.size(), &neg_lr,
                             grad_weights.d_data, 1, weights.d_data, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), bias.size(), &neg_lr, grad_bias.d_data,
                             1, bias.d_data, 1));
  }

  std::string getConfig() const override {
    return "fully_connected " + std::to_string(out_features) + " " +
           activationName(activation);
  }
};

// ============================================================================
// Dropout Layer
// ============================================================================
class DropoutLayer : public Layer {
public:
  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  void *states = nullptr;
  size_t states_size = 0;
  void *reserve_space = nullptr;
  size_t reserve_space_size = 0;

  float dropout_rate;
  bool is_training = true;

  DropoutLayer(const std::string &name_, float rate = 0.5f) {
    name = name_;
    dropout_rate = rate;
    activation = ActivationType::NONE;
  }

  void init(int channels, int height, int width, int batch_size_) {
    batch_size = batch_size_;

    output.allocate(batch_size, channels, height, width);
    delta.allocate(batch_size, channels, height, width);

    // Create dropout descriptor
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));

    CUDNN_CHECK(cudnnDropoutGetStatesSize(cudnn(), &states_size));
    CUDA_CHECK(cudaMalloc(&states, states_size));

    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc, cudnn(), dropout_rate,
                                          states, states_size, time(nullptr)));

    CUDNN_CHECK(
        cudnnDropoutGetReserveSpaceSize(output.desc, &reserve_space_size));
    CUDA_CHECK(cudaMalloc(&reserve_space, reserve_space_size));
  }

  ~DropoutLayer() {
    if (dropout_desc)
      cudnnDestroyDropoutDescriptor(dropout_desc);
    if (states)
      cudaFree(states);
    if (reserve_space)
      cudaFree(reserve_space);
  }

  void forward(const Tensor &input) override {
    if (is_training) {
      CUDNN_CHECK(cudnnDropoutForward(cudnn(), dropout_desc, input.desc,
                                      input.d_data, output.desc, output.d_data,
                                      reserve_space, reserve_space_size));
    } else {
      // In inference, just copy
      CUDA_CHECK(cudaMemcpy(output.d_data, input.d_data, input.bytes(),
                            cudaMemcpyDeviceToDevice));
    }
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    CUDNN_CHECK(cudnnDropoutBackward(
        cudnn(), dropout_desc, delta.desc, delta.d_data, input_delta.desc,
        input_delta.d_data, reserve_space, reserve_space_size));
  }

  std::string getConfig() const override {
    return "dropout " + std::to_string(dropout_rate);
  }
};

} // namespace mojo
