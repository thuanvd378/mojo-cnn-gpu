// == mojo-gpu ================================================================
//    mojo.h: Main include file for Mojo-CNN GPU
// ============================================================================

#pragma once

// All base types and utilities
#include "common.cuh"

namespace mojo {

// ============================================================================
// Optimizer Base Class
// ============================================================================
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
  std::map<float *, float *> velocity;

  SGD(float lr = 0.01f, float mom = 0.9f) : Optimizer(lr), momentum(mom) {}
  ~SGD() {
    for (auto &v : velocity)
      if (v.second)
        cudaFree(v.second);
  }

  void update(float *weights, float *gradients, size_t size) override {
    if (velocity.find(weights) == velocity.end()) {
      float *v;
      CUDA_CHECK(cudaMalloc(&v, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(v, 0, size * sizeof(float)));
      velocity[weights] = v;
    }
    float *v = velocity[weights];
    const float neg_lr = -learning_rate, one = 1.0f;
    CUBLAS_CHECK(cublasSscal(cublas(), size, &momentum, v, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), size, &neg_lr, gradients, 1, v, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), size, &one, v, 1, weights, 1));
  }
};

// ============================================================================
// Adam Optimizer
// ============================================================================
__global__ void adamKernel(float *w, const float *g, float *m, float *v,
                           float lr, float b1, float b2, float eps, float b1t,
                           float b2t, size_t sz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sz)
    return;
  m[i] = b1 * m[i] + (1.0f - b1) * g[i];
  v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
  w[i] -= lr * (m[i] / (1.0f - b1t)) / (sqrtf(v[i] / (1.0f - b2t)) + eps);
}

class Adam : public Optimizer {
public:
  float beta1, beta2, epsilon;
  int t = 0;
  std::map<float *, float *> m, v;

  Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
      : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {}
  ~Adam() {
    for (auto &p : m)
      if (p.second)
        cudaFree(p.second);
    for (auto &p : v)
      if (p.second)
        cudaFree(p.second);
  }

  void update(float *weights, float *gradients, size_t size) override {
    if (m.find(weights) == m.end()) {
      float *mb, *vb;
      CUDA_CHECK(cudaMalloc(&mb, size * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&vb, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(mb, 0, size * sizeof(float)));
      CUDA_CHECK(cudaMemset(vb, 0, size * sizeof(float)));
      m[weights] = mb;
      v[weights] = vb;
    }
    t++;
    int bs = 256, gs = (size + bs - 1) / bs;
    adamKernel<<<gs, bs>>>(weights, gradients, m[weights], v[weights],
                           learning_rate, beta1, beta2, epsilon, powf(beta1, t),
                           powf(beta2, t), size);
    CUDA_CHECK(cudaGetLastError());
  }
};

inline Optimizer *createOptimizer(const std::string &name, float lr = 0.01f) {
  if (name == "adam")
    return new Adam(lr);
  return new SGD(lr);
}

// ============================================================================
// Base Layer Class
// ============================================================================
class Layer {
public:
  std::string name;
  Tensor output, delta;
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
  InputLayer(const std::string &nm, int h, int w, int c = 1) {
    name = nm;
    output.allocate(1, c, h, w);
  }
  void resizeBatch(int n) override {
    batch_size = n;
    output.allocate(n, output.c, output.h, output.w);
    delta.allocate(n, output.c, output.h, output.w);
  }
  void forward(const Tensor &input) override {
    CUDA_CHECK(cudaMemcpy(output.d_data, input.d_data, input.bytes(),
                          cudaMemcpyDeviceToDevice));
  }
  void backward(const Tensor &input, Tensor &input_delta) override {}
  std::string getConfig() const override {
    return "input " + std::to_string(output.w) + " " +
           std::to_string(output.h) + " " + std::to_string(output.c);
  }
};

// ============================================================================
// Convolution Layer
// ============================================================================
class Conv2DLayer : public Layer {
public:
  FilterTensor weights, grad_weights;
  Tensor bias, grad_bias, pre_act, act_grad; // act_grad pre-allocated
  ConvolutionDescriptor conv_desc;
  cudnnTensorDescriptor_t bias_desc = nullptr;
  int kernel_size, num_filters, stride, padding;
  bool algo_cached = false; // Cache algorithm selection

  Conv2DLayer(const std::string &nm, int ks, int nf, int st = 1,
              ActivationType act = ActivationType::RELU)
      : kernel_size(ks), num_filters(nf), stride(st) {
    name = nm;
    activation = act;
    padding = ks / 2;
  }
  ~Conv2DLayer() {
    if (bias_desc)
      cudnnDestroyTensorDescriptor(bias_desc);
  }

  void init(int in_c, int in_h, int in_w, int bs) {
    batch_size = bs;
    int oh = (in_h + 2 * padding - kernel_size) / stride + 1;
    int ow = (in_w + 2 * padding - kernel_size) / stride + 1;
    weights.allocate(num_filters, in_c, kernel_size, kernel_size);
    weights.initXavier();
    bias.allocate(1, num_filters, 1, 1);
    CUDA_CHECK(cudaMemset(bias.d_data, 0, bias.bytes()));
    grad_weights.allocate(num_filters, in_c, kernel_size, kernel_size);
    grad_bias.allocate(1, num_filters, 1, 1);
    output.allocate(bs, num_filters, oh, ow);
    delta.allocate(bs, num_filters, oh, ow);
    pre_act.allocate(bs, num_filters, oh, ow);
    act_grad.allocate(bs, num_filters, oh, ow); // Pre-allocate
    conv_desc.create(padding, padding, stride, stride);
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_filters, 1, 1));
    // Algorithm will be cached on first forward call
    algo_cached = false;
  }

  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;
    output.allocate(n, num_filters, output.h, output.w);
    delta.allocate(n, num_filters, output.h, output.w);
    pre_act.allocate(n, num_filters, output.h, output.w);
    act_grad.allocate(n, num_filters, output.h, output.w);
    algo_cached = false; // Need to re-cache for new batch size
  }

  void forward(const Tensor &input) override {
    const float a = 1.0f, b = 0.0f;
    // Cache algorithm selection (only once per batch size)
    if (!algo_cached) {
      conv_desc.findBestAlgorithms(input.desc, weights.desc, output.desc);
      algo_cached = true;
    }
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn(), &a, input.desc, input.d_data, weights.desc, weights.d_data,
        conv_desc.desc, conv_desc.fwd_algo, conv_desc.workspace,
        conv_desc.workspace_size, &b, output.desc, output.d_data));
    CUDNN_CHECK(cudnnAddTensor(cudnn(), &a, bias_desc, bias.d_data, &a,
                               output.desc, output.d_data));
    // Only save pre-activation if needed for backward
    if (activation != ActivationType::NONE &&
        activation != ActivationType::RELU) {
      CUDA_CHECK(cudaMemcpyAsync(pre_act.d_data, output.d_data, output.bytes(),
                                 cudaMemcpyDeviceToDevice, stream()));
    }
    activationForward(output, activation);
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    const float a = 1.0f, b = 0.0f;
    // Use pre-allocated act_grad instead of creating new tensor
    activationBackward(act_grad, delta, output, pre_act, activation);
    CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn(), &a, act_grad.desc,
                                             act_grad.d_data, &b, bias_desc,
                                             grad_bias.d_data));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn(), &a, input.desc, input.d_data, act_grad.desc, act_grad.d_data,
        conv_desc.desc, conv_desc.bwd_filter_algo, conv_desc.workspace,
        conv_desc.workspace_size, &b, grad_weights.desc, grad_weights.d_data));
    if (input_delta.d_data)
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          cudnn(), &a, weights.desc, weights.d_data, act_grad.desc,
          act_grad.d_data, conv_desc.desc, conv_desc.bwd_data_algo,
          conv_desc.workspace, conv_desc.workspace_size, &b, input_delta.desc,
          input_delta.d_data));
  }

  void updateWeights(float lr) override {
    const float nlr = -lr;
    CUBLAS_CHECK(cublasSaxpy(cublas(), weights.size(), &nlr,
                             grad_weights.d_data, 1, weights.d_data, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), bias.size(), &nlr, grad_bias.d_data, 1,
                             bias.d_data, 1));
  }

  std::string getConfig() const override {
    return "convolution " + std::to_string(kernel_size) + " " +
           std::to_string(num_filters) + " " + std::to_string(stride) + " " +
           activationName(activation);
  }
};

// ============================================================================
// Max Pooling Layer
// ============================================================================
class MaxPoolLayer : public Layer {
public:
  PoolingDescriptor pool_desc;
  int pool_size, stride;

  MaxPoolLayer(const std::string &nm, int ps, int st = 0)
      : pool_size(ps), stride(st > 0 ? st : ps) {
    name = nm;
    pool_desc.create(ps, ps, stride, stride);
  }
  void init(int c, int h, int w, int bs) {
    batch_size = bs;
    int oh = (h - pool_size) / stride + 1;
    int ow = (w - pool_size) / stride + 1;
    output.allocate(bs, c, oh, ow);
    delta.allocate(bs, c, oh, ow);
  }
  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;
    output.allocate(n, output.c, output.h, output.w);
    delta.allocate(n, output.c, output.h, output.w);
  }
  void forward(const Tensor &input) override {
    const float a = 1.0f, b = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn(), pool_desc.desc, &a, input.desc,
                                    input.d_data, &b, output.desc,
                                    output.d_data));
  }
  void backward(const Tensor &input, Tensor &input_delta) override {
    const float a = 1.0f, b = 0.0f;
    CUDNN_CHECK(cudnnPoolingBackward(cudnn(), pool_desc.desc, &a, output.desc,
                                     output.d_data, delta.desc, delta.d_data,
                                     input.desc, input.d_data, &b,
                                     input_delta.desc, input_delta.d_data));
  }
  std::string getConfig() const override {
    return "max_pool " + std::to_string(pool_size) + " " +
           std::to_string(stride);
  }
};

// ============================================================================
// Fully Connected Layer
// ============================================================================
class FCLayer : public Layer {
public:
  Tensor weights, bias, grad_weights, grad_bias, pre_act;
  Tensor act_grad, ones; // Pre-allocated for backward
  int in_features, out_features;

  FCLayer(const std::string &nm, int of,
          ActivationType act = ActivationType::NONE)
      : out_features(of) {
    name = nm;
    activation = act;
  }

  void init(int inf, int bs) {
    in_features = inf;
    batch_size = bs;
    weights.allocate(1, 1, out_features, in_features);
    bias.allocate(1, out_features, 1, 1);
    grad_weights.allocate(1, 1, out_features, in_features);
    grad_bias.allocate(1, out_features, 1, 1);
    float sc = sqrtf(2.0f / in_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, sc);
    for (size_t i = 0; i < weights.size(); i++)
      weights.h_data[i] = d(gen);
    weights.toDeviceSync();
    CUDA_CHECK(cudaMemset(bias.d_data, 0, bias.bytes()));
    output.allocate(bs, out_features, 1, 1);
    delta.allocate(bs, out_features, 1, 1);
    pre_act.allocate(bs, out_features, 1, 1);
    act_grad.allocate(bs, out_features, 1, 1); // Pre-allocate
    ones.allocate(1, bs, 1, 1);                // Pre-allocate ones vector
    std::fill(ones.h_data, ones.h_data + bs, 1.0f);
    ones.toDeviceSync();
  }

  void resizeBatch(int n) override {
    if (n == batch_size)
      return;
    batch_size = n;
    output.allocate(n, out_features, 1, 1);
    delta.allocate(n, out_features, 1, 1);
    pre_act.allocate(n, out_features, 1, 1);
    act_grad.allocate(n, out_features, 1, 1);
    ones.allocate(1, n, 1, 1);
    std::fill(ones.h_data, ones.h_data + n, 1.0f);
    ones.toDeviceSync();
  }

  void forward(const Tensor &input) override {
    const float a = 1.0f, b = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas(), CUBLAS_OP_T, CUBLAS_OP_N, out_features,
                             batch_size, in_features, &a, weights.d_data,
                             in_features, input.d_data, in_features, &b,
                             output.d_data, out_features));
    // Use Sger for batched bias addition: Y += bias * ones^T
    CUBLAS_CHECK(cublasSger(cublas(), out_features, batch_size, &a, bias.d_data,
                            1, ones.d_data, 1, output.d_data, out_features));
    // Only save pre-activation if needed
    if (activation != ActivationType::NONE &&
        activation != ActivationType::RELU) {
      CUDA_CHECK(cudaMemcpyAsync(pre_act.d_data, output.d_data, output.bytes(),
                                 cudaMemcpyDeviceToDevice, stream()));
    }
    activationForward(output, activation);
  }

  void backward(const Tensor &input, Tensor &input_delta) override {
    const float a = 1.0f, b = 0.0f;
    // Use pre-allocated tensors
    activationBackward(act_grad, delta, output, pre_act, activation);
    CUBLAS_CHECK(cublasSgemv(cublas(), CUBLAS_OP_N, out_features, batch_size,
                             &a, act_grad.d_data, out_features, ones.d_data, 1,
                             &b, grad_bias.d_data, 1));
    CUBLAS_CHECK(cublasSgemm(cublas(), CUBLAS_OP_N, CUBLAS_OP_T, out_features,
                             in_features, batch_size, &a, act_grad.d_data,
                             out_features, input.d_data, in_features, &b,
                             grad_weights.d_data, out_features));
    if (input_delta.d_data)
      CUBLAS_CHECK(cublasSgemm(cublas(), CUBLAS_OP_N, CUBLAS_OP_N, in_features,
                               batch_size, out_features, &a, weights.d_data,
                               in_features, act_grad.d_data, out_features, &b,
                               input_delta.d_data, in_features));
  }

  void updateWeights(float lr) override {
    const float nlr = -lr;
    CUBLAS_CHECK(cublasSaxpy(cublas(), weights.size(), &nlr,
                             grad_weights.d_data, 1, weights.d_data, 1));
    CUBLAS_CHECK(cublasSaxpy(cublas(), bias.size(), &nlr, grad_bias.d_data, 1,
                             bias.d_data, 1));
  }

  std::string getConfig() const override {
    return "fully_connected " + std::to_string(out_features) + " " +
           activationName(activation);
  }
};

// ============================================================================
// Cross-entropy loss kernel
// ============================================================================
__global__ void crossEntropyKernel(const float *out, const float *tgt,
                                   float *loss, float *grad, int bs, int nc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bs * nc)
    return;
  int b = i / nc;
  float o = out[i], t = tgt[i];
  grad[i] = (o - t) / bs;
  if (t > 0.5f)
    atomicAdd(&loss[b], -logf(fmaxf(o, 1e-7f)));
}

// ============================================================================
// Loss Accumulation Kernel
// ============================================================================
__global__ void accumulateKernel(const float *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  atomicAdd(dst, src[i]);
}

// ============================================================================
// Network Class
// ============================================================================
class Network {
public:
  std::vector<Layer *> layers;
  Optimizer *optimizer = nullptr;
  int batch_size = 32;
  float learning_rate = 0.01f;
  int epoch = 0;
  int train_samples = 0, train_updates = 0, train_skipped = 0;
  float estimated_accuracy = 0.0f, total_loss = 0.0f;
  Tensor input_buffer, target_buffer, loss_buffer;
  Tensor epoch_loss_buffer; // Accumulates loss over the entire epoch

  Network(const std::string &opt = "adam") {
    CudnnHandle::instance().init();
    optimizer = createOptimizer(opt, learning_rate);
    epoch_loss_buffer.allocate(1, 1, 1, 1); // Single float on GPU
  }
  ~Network() {
    for (auto l : layers)
      delete l;
    delete optimizer;
  }

  void setLearningRate(float lr) {
    learning_rate = lr;
    if (optimizer)
      optimizer->learning_rate = lr;
  }
  void setBatchSize(int bs) { batch_size = bs; }

  void push_back(const std::string &name, const std::string &cfg) {
    std::istringstream iss(cfg);
    std::string type;
    iss >> type;
    if (type == "input") {
      int w, h, c;
      iss >> w >> h >> c;
      layers.push_back(new InputLayer(name, h, w, c));
    } else if (type == "convolution") {
      int k, f, s;
      std::string a;
      iss >> k >> f >> s >> a;
      layers.push_back(new Conv2DLayer(name, k, f, s, parseActivation(a)));
    } else if (type == "max_pool" || type == "semi_stochastic_pool") {
      int sz, st;
      iss >> sz >> st;
      layers.push_back(new MaxPoolLayer(name, sz, st));
    } else if (type == "fully_connected" || type == "softmax") {
      int sz;
      iss >> sz;
      ActivationType at =
          (type == "softmax") ? ActivationType::SOFTMAX : ActivationType::NONE;
      layers.push_back(new FCLayer(name, sz, at));
    }
  }

  void init() {
    if (layers.empty())
      return;
    InputLayer *inp = dynamic_cast<InputLayer *>(layers[0]);
    if (!inp)
      return;
    int c = inp->output.c, h = inp->output.h, w = inp->output.w;
    inp->resizeBatch(batch_size);
    for (size_t i = 1; i < layers.size(); i++) {
      Layer *l = layers[i];
      Layer *p = layers[i - 1];
      if (Conv2DLayer *cv = dynamic_cast<Conv2DLayer *>(l))
        cv->init(p->output.c, p->output.h, p->output.w, batch_size);
      else if (MaxPoolLayer *pl = dynamic_cast<MaxPoolLayer *>(l))
        pl->init(p->output.c, p->output.h, p->output.w, batch_size);
      else if (FCLayer *fc = dynamic_cast<FCLayer *>(l))
        fc->init(p->output.c * p->output.h * p->output.w, batch_size);
    }
    int os = layers.back()->outputSize();
    input_buffer.allocate(batch_size, c, h, w);
    target_buffer.allocate(batch_size, os, 1, 1);
    loss_buffer.allocate(batch_size, 1, 1, 1);
  }

  void forward() {
    for (size_t i = 0; i < layers.size(); i++)
      layers[i]->forward(i == 0 ? input_buffer : layers[i - 1]->output);
  }

  void backward() {
    Layer *ol = layers.back();
    int os = ol->outputSize();
    CUDA_CHECK(cudaMemset(loss_buffer.d_data, 0, loss_buffer.bytes()));
    int bs = 256, gs = (batch_size * os + bs - 1) / bs;
    crossEntropyKernel<<<gs, bs>>>(ol->output.d_data, target_buffer.d_data,
                                   loss_buffer.d_data, ol->delta.d_data,
                                   batch_size, os);

    // Accumulate total loss for the epoch on GPU
    // 1 block, 256 threads is enough for batch_size elements
    int acc_bs = 256;
    int acc_gs = (batch_size + acc_bs - 1) / acc_bs;
    accumulateKernel<<<acc_gs, acc_bs>>>(loss_buffer.d_data,
                                         epoch_loss_buffer.d_data, batch_size);

    for (int i = layers.size() - 1; i >= 1; i--)
      layers[i]->backward(layers[i - 1]->output, layers[i - 1]->delta);
    for (auto l : layers)
      l->updateWeights(learning_rate);
  }

  float trainBatch(const float *in, const int *labels, int bs) {
    if (bs != batch_size) {
      batch_size = bs;
      for (auto l : layers)
        l->resizeBatch(bs);
      input_buffer.resizeBatch(bs);
      target_buffer.resizeBatch(bs);
      loss_buffer.resizeBatch(bs);
    }
    memcpy(input_buffer.h_data, in, input_buffer.bytes());
    input_buffer.toDevice(stream());
    int os = layers.back()->outputSize();
    std::fill(target_buffer.h_data, target_buffer.h_data + bs * os, 0.0f);
    for (int b = 0; b < bs; b++)
      target_buffer.h_data[b * os + labels[b]] = 1.0f;
    target_buffer.toDevice(stream());

    target_buffer.toDevice(stream());

    forward();
    backward();

    train_samples += bs;
    train_updates++;
    return 0.0f; // Return 0 because we don't wait for the result
  }

  int predictClass(const float *in) {
    if (batch_size != 1) {
      batch_size = 1;
      for (auto l : layers)
        l->resizeBatch(1);
      input_buffer.resizeBatch(1);
    }
    memcpy(input_buffer.h_data, in, input_buffer.bytes());
    input_buffer.toDeviceSync();
    forward();
    sync();
    Layer *ol = layers.back();
    ol->output.toHostSync();
    int best = 0;
    float bv = ol->output.h_data[0];
    for (int i = 1; i < ol->outputSize(); i++)
      if (ol->output.h_data[i] > bv) {
        bv = ol->output.h_data[i];
        best = i;
      }
    return best;
  }

  // Batch prediction - returns predictions for all samples
  void predictBatch(const float *in, int *predictions, int bs) {
    if (bs != batch_size) {
      batch_size = bs;
      for (auto l : layers)
        l->resizeBatch(bs);
      input_buffer.resizeBatch(bs);
    }
    memcpy(input_buffer.h_data, in, input_buffer.bytes());
    input_buffer.toDevice(stream());
    sync();
    forward();
    sync();
    Layer *ol = layers.back();
    ol->output.toHostSync();
    int os = ol->outputSize();
    for (int b = 0; b < bs; b++) {
      int best = 0;
      float bv = ol->output.h_data[b * os];
      for (int i = 1; i < os; i++) {
        float v = ol->output.h_data[b * os + i];
        if (v > bv) {
          bv = v;
          best = i;
        }
      }
      predictions[b] = best;
    }
  }

  void startEpoch() {
    epoch++;
    train_samples = train_updates = train_skipped = 0;
    total_loss = 0.0f;
    // Reset GPU loss accumulator
    CUDA_CHECK(
        cudaMemset(epoch_loss_buffer.d_data, 0, epoch_loss_buffer.bytes()));
  }

  void endEpoch() {
    // Sync once at the end of epoch to get total loss
    sync();

    epoch_loss_buffer.toHostSync();
    total_loss = epoch_loss_buffer.h_data[0];

    if (train_samples > 0)
      estimated_accuracy = 100.0f * (1.0f - total_loss / train_samples);
  }
  int getEpoch() const { return epoch; }
  int outSize() const {
    return layers.empty() ? 0 : layers.back()->outputSize();
  }
  std::string getConfiguration() const {
    std::string s;
    for (size_t i = 0; i < layers.size(); i++)
      s += "  " + std::to_string(i) + " : " + layers[i]->name + " : " +
           layers[i]->getConfig() + "\n";
    return s;
  }
};

} // namespace mojo

// Utilities
#include "util.h"
