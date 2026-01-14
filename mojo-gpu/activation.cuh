// == mojo-gpu ================================================================
//    activation.cuh: GPU activation functions using cuDNN
// ============================================================================

#pragma once

namespace mojo {

enum class ActivationType { NONE, RELU, ELU, SIGMOID, TANH, SOFTMAX };

// Convert string to activation type
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
  if (name == "identity" || name == "none")
    return ActivationType::NONE;
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

// Activation descriptor wrapper
class ActivationDescriptor {
public:
  cudnnActivationDescriptor_t desc = nullptr;
  ActivationType type = ActivationType::NONE;

  void create(ActivationType type_, double coef = 1.0) {
    if (desc)
      destroy();
    type = type_;

    if (type == ActivationType::NONE || type == ActivationType::SOFTMAX) {
      return; // No cuDNN descriptor needed
    }

    CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc));

    cudnnActivationMode_t mode;
    switch (type) {
    case ActivationType::RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case ActivationType::ELU:
      mode = CUDNN_ACTIVATION_ELU;
      break;
    case ActivationType::SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case ActivationType::TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      mode = CUDNN_ACTIVATION_IDENTITY;
    }

    CUDNN_CHECK(cudnnSetActivationDescriptor(desc, mode,
                                             CUDNN_NOT_PROPAGATE_NAN, coef));
  }

  void destroy() {
    if (desc) {
      cudnnDestroyActivationDescriptor(desc);
      desc = nullptr;
    }
  }

  ~ActivationDescriptor() { destroy(); }
};

// Forward activation (in-place)
inline void activationForward(Tensor &x, ActivationType type) {
  if (type == ActivationType::NONE)
    return;

  const float alpha = 1.0f, beta = 0.0f;

  if (type == ActivationType::SOFTMAX) {
    // Softmax per sample (channel dimension)
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn(), CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, x.desc,
                                    x.d_data, &beta, x.desc, x.d_data));
  } else {
    ActivationDescriptor act;
    act.create(type);

    CUDNN_CHECK(cudnnActivationForward(cudnn(), act.desc, &alpha, x.desc,
                                       x.d_data, &beta, x.desc, x.d_data));
  }
}

// Backward activation
// dy: incoming gradient, y: output from forward, x: input to forward
inline void activationBackward(Tensor &dx, const Tensor &dy, const Tensor &y,
                               const Tensor &x, ActivationType type) {
  if (type == ActivationType::NONE) {
    // Just copy dy to dx
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
    ActivationDescriptor act;
    act.create(type);

    CUDNN_CHECK(cudnnActivationBackward(cudnn(), act.desc, &alpha, y.desc,
                                        y.d_data, dy.desc, dy.d_data, x.desc,
                                        x.d_data, &beta, dx.desc, dx.d_data));
  }
}

} // namespace mojo
