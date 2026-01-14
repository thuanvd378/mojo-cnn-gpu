// == mojo-gpu ================================================================
//    network.cuh: Main network class
// ============================================================================

#pragma once

namespace mojo {

// Cross-entropy loss kernel
__global__ void crossEntropyLossKernel(const float *output, const float *target,
                                       float *loss, float *grad, int batch_size,
                                       int num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * num_classes)
    return;

  int b = idx / num_classes;
  int c = idx % num_classes;

  float o = output[idx];
  float t = target[idx];

  // Gradient: output - target (for softmax + cross-entropy)
  grad[idx] = (o - t) / batch_size;

  // Loss: -target * log(output)
  if (t > 0.5f) {
    atomicAdd(&loss[b], -logf(fmaxf(o, 1e-7f)));
  }
}

class Network {
public:
  std::vector<Layer *> layers;
  Optimizer *optimizer = nullptr;

  int batch_size = 32;
  float learning_rate = 0.01f;

  // Training statistics
  int epoch = 0;
  int train_samples = 0;
  int train_updates = 0;
  int train_skipped = 0;
  float estimated_accuracy = 0.0f;
  float total_loss = 0.0f;

  // Pinned memory for input
  Tensor input_buffer;
  Tensor target_buffer;
  Tensor loss_buffer;

  Network(const std::string &optimizer_name = "adam") {
    CudnnHandle::instance().init();
    optimizer = createOptimizer(optimizer_name, learning_rate);
  }

  ~Network() {
    for (auto layer : layers)
      delete layer;
    delete optimizer;
  }

  void setLearningRate(float lr) {
    learning_rate = lr;
    if (optimizer)
      optimizer->learning_rate = lr;
  }

  void setBatchSize(int bs) { batch_size = bs; }

  // Add layer by config string (like mojo original)
  void push_back(const std::string &name, const std::string &config) {
    std::istringstream iss(config);
    std::string type;
    iss >> type;

    if (type == "input") {
      int w, h, c;
      iss >> w >> h >> c;
      layers.push_back(new InputLayer(name, h, w, c));
    } else if (type == "convolution") {
      int kernel, filters, stride;
      std::string act;
      iss >> kernel >> filters >> stride >> act;
      layers.push_back(
          new Conv2DLayer(name, kernel, filters, stride, parseActivation(act)));
    } else if (type == "max_pool" || type == "semi_stochastic_pool") {
      int size, stride;
      iss >> size >> stride;
      layers.push_back(new MaxPoolLayer(name, size, stride));
    } else if (type == "fully_connected" || type == "softmax") {
      int size;
      iss >> size;
      ActivationType act =
          (type == "softmax") ? ActivationType::SOFTMAX : ActivationType::NONE;
      layers.push_back(new FCLayer(name, size, act));
    } else if (type == "dropout") {
      float rate;
      iss >> rate;
      layers.push_back(new DropoutLayer(name, rate));
    }
  }

  // Initialize all layers (call after adding all layers)
  void init() {
    if (layers.empty())
      return;

    // Get input layer dimensions
    InputLayer *input = dynamic_cast<InputLayer *>(layers[0]);
    if (!input) {
      std::cerr << "First layer must be input layer!" << std::endl;
      return;
    }

    int c = input->output.c;
    int h = input->output.h;
    int w = input->output.w;

    input->resizeBatch(batch_size);

    // Initialize each layer based on previous layer's output
    for (size_t i = 1; i < layers.size(); i++) {
      Layer *layer = layers[i];
      Layer *prev = layers[i - 1];

      if (Conv2DLayer *conv = dynamic_cast<Conv2DLayer *>(layer)) {
        conv->init(prev->output.c, prev->output.h, prev->output.w, batch_size);
      } else if (MaxPoolLayer *pool = dynamic_cast<MaxPoolLayer *>(layer)) {
        pool->init(prev->output.c, prev->output.h, prev->output.w, batch_size);
      } else if (FCLayer *fc = dynamic_cast<FCLayer *>(layer)) {
        int in_features = prev->output.c * prev->output.h * prev->output.w;
        fc->init(in_features, batch_size);
      } else if (DropoutLayer *drop = dynamic_cast<DropoutLayer *>(layer)) {
        drop->init(prev->output.c, prev->output.h, prev->output.w, batch_size);
      }
    }

    // Allocate buffers
    int out_size = layers.back()->outputSize();
    input_buffer.allocate(batch_size, c, h, w);
    target_buffer.allocate(batch_size, out_size, 1, 1);
    loss_buffer.allocate(batch_size, 1, 1, 1);
  }

  // Forward pass
  void forward() {
    for (size_t i = 0; i < layers.size(); i++) {
      if (i == 0) {
        layers[i]->forward(input_buffer);
      } else {
        layers[i]->forward(layers[i - 1]->output);
      }
    }
  }

  // Backward pass
  void backward() {
    // Calculate output gradient (softmax + cross-entropy)
    Layer *output_layer = layers.back();
    int out_size = output_layer->outputSize();

    int block_size = 256;
    int total = batch_size * out_size;
    int grid_size = (total + block_size - 1) / block_size;

    CUDA_CHECK(cudaMemset(loss_buffer.d_data, 0, loss_buffer.bytes()));

    crossEntropyLossKernel<<<grid_size, block_size>>>(
        output_layer->output.d_data, target_buffer.d_data, loss_buffer.d_data,
        output_layer->delta.d_data, batch_size, out_size);

    // Backward through all layers
    for (int i = layers.size() - 1; i >= 1; i--) {
      layers[i]->backward(layers[i - 1]->output, layers[i - 1]->delta);
    }

    // Update weights
    for (auto layer : layers) {
      layer->updateWeights(learning_rate);
    }
  }

  // Train on a batch
  float trainBatch(const float *input_data, const int *labels,
                   int actual_batch_size) {
    // Resize if needed
    if (actual_batch_size != batch_size) {
      batch_size = actual_batch_size;
      for (auto layer : layers) {
        layer->resizeBatch(batch_size);
      }
      input_buffer.resizeBatch(batch_size);
      target_buffer.resizeBatch(batch_size);
      loss_buffer.resizeBatch(batch_size);
    }

    // Copy input to pinned memory then to GPU
    int input_size = input_buffer.size();
    memcpy(input_buffer.h_data, input_data, input_size * sizeof(float));
    input_buffer.toDevice(stream());

    // Create one-hot targets
    int out_size = layers.back()->outputSize();
    std::fill(target_buffer.h_data,
              target_buffer.h_data + batch_size * out_size, 0.0f);
    for (int b = 0; b < batch_size; b++) {
      target_buffer.h_data[b * out_size + labels[b]] = 1.0f;
    }
    target_buffer.toDevice(stream());

    sync();

    // Forward
    forward();

    // Backward + update
    backward();

    sync();

    // Get loss
    loss_buffer.toHostSync();
    float batch_loss = 0;
    for (int b = 0; b < batch_size; b++) {
      batch_loss += loss_buffer.h_data[b];
    }

    train_samples += batch_size;
    train_updates++;
    total_loss += batch_loss;

    return batch_loss / batch_size;
  }

  // Predict class
  int predictClass(const float *input_data) {
    // Single sample prediction
    int old_batch = batch_size;
    if (batch_size != 1) {
      batch_size = 1;
      for (auto layer : layers) {
        layer->resizeBatch(1);
      }
      input_buffer.resizeBatch(1);
    }

    int input_size = input_buffer.size();
    memcpy(input_buffer.h_data, input_data, input_size * sizeof(float));
    input_buffer.toDeviceSync();

    forward();
    sync();

    // Get output
    Layer *out = layers.back();
    out->output.toHostSync();

    // Find max
    int best = 0;
    float best_val = out->output.h_data[0];
    for (int i = 1; i < out->outputSize(); i++) {
      if (out->output.h_data[i] > best_val) {
        best_val = out->output.h_data[i];
        best = i;
      }
    }

    return best;
  }

  // Start new epoch
  void startEpoch() {
    epoch++;
    train_samples = 0;
    train_updates = 0;
    train_skipped = 0;
    total_loss = 0.0f;
  }

  // End epoch
  void endEpoch() {
    if (train_samples > 0) {
      estimated_accuracy = 100.0f * (1.0f - total_loss / train_samples);
    }
  }

  int getEpoch() const { return epoch; }
  int outSize() const {
    return layers.empty() ? 0 : layers.back()->outputSize();
  }

  // Get configuration string
  std::string getConfiguration() const {
    std::string config;
    for (size_t i = 0; i < layers.size(); i++) {
      config += "  " + std::to_string(i) + " : " + layers[i]->name + " : " +
                layers[i]->getConfig() + "\n";
    }
    return config;
  }

  // Read model from file (simplified)
  bool read(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Cannot open model file: " << filename << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      // Skip comments and empty lines
      if (line.empty() || line[0] == '#' || line[0] == ';')
        continue;

      // Parse "name : config"
      size_t colon = line.find(':');
      if (colon != std::string::npos) {
        std::string name = line.substr(0, colon);
        std::string config = line.substr(colon + 1);

        // Trim whitespace
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        config.erase(0, config.find_first_not_of(" \t"));
        config.erase(config.find_last_not_of(" \t") + 1);

        push_back(name, config);
      }
    }

    init();
    return true;
  }
};

} // namespace mojo
