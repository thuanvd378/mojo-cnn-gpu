// == mojo-gpu ================================================================
//    train_mnist_gpu.cpp: MNIST training example using cuDNN
// ============================================================================

#include <chrono> // Added for timing
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mnist_parser.h"
#include <mojo.h>

using namespace mnist;

// Test accuracy on test set using BATCH inference
float test(mojo::Network &cnn,
           const std::vector<std::vector<float>> &test_images,
           const std::vector<int> &test_labels, int batch_size = 256) {

  mojo::progress progress((int)test_images.size(), "  testing:\t\t");

  int correct = 0;
  int total = (int)test_images.size();
  int input_dim = 28 * 28;

  // Prepare batch buffers
  std::vector<float> batch_input(batch_size * input_dim);
  std::vector<int> batch_predictions(batch_size);

  for (int k = 0; k < total; k += batch_size) {
    int current_batch = std::min(batch_size, total - k);

    // Fill batch
    for (int b = 0; b < current_batch; b++) {
      std::copy(test_images[k + b].begin(), test_images[k + b].end(),
                batch_input.data() + b * input_dim);
    }

    // Batch predict
    cnn.predictBatch(batch_input.data(), batch_predictions.data(),
                     current_batch);

    // Count correct
    for (int b = 0; b < current_batch; b++) {
      if (batch_predictions[b] == test_labels[k + b])
        correct++;
    }

    if ((k / batch_size) % 10 == 0)
      progress.draw_progress(k);
  }

  return (float)correct / total * 100.0f;
}

int main(int argc, char *argv[]) {
  std::cout << "=============================================================="
            << std::endl;
  std::cout << "  MOJO CNN - GPU (cuDNN) Version" << std::endl;
  std::cout << "=============================================================="
            << std::endl;

  // Print GPU info
  mojo::printGPUInfo();
  std::cout << std::endl;

  // Configuration - Default batch size 256 for better GPU utilization
  // == parse command line arguments
  // ============================================

  int batch_size = 256;
  float learning_rate = 0.005f;
  int num_epochs = 50;
  std::string solver = "adam";
  std::string data_path = "../data/mnist/";

  if (argc >= 2)
    batch_size = std::stoi(argv[1]);
  if (argc >= 3)
    learning_rate = std::stof(argv[2]);
  if (argc >= 4)
    num_epochs = std::stoi(argv[3]);

  std::cout << "[Config] Batch size: " << batch_size << std::endl;
  std::cout << "[Config] Learning rate: " << learning_rate << std::endl;
  std::cout << "[Config] Epochs: " << num_epochs << std::endl;
  std::cout << "[Config] Optimizer: " << solver << std::endl;
  std::cout << std::endl;

  // Parse MNIST data
  std::vector<std::vector<float>> train_images, test_images;
  std::vector<int> train_labels, test_labels;

  if (!parse_train_data(data_path, train_images, train_labels)) {
    std::cerr << "Error: Could not parse training data!" << std::endl;
    return 1;
  }
  if (!parse_test_data(data_path, test_images, test_labels)) {
    std::cerr << "Error: Could not parse test data!" << std::endl;
    return 1;
  }

  std::cout << "[Data] Training samples: " << train_images.size() << std::endl;
  std::cout << "[Data] Test samples: " << test_images.size() << std::endl;
  std::cout << std::endl;

  // Create network
  mojo::Network cnn(solver);
  cnn.setBatchSize(batch_size);
  cnn.setLearningRate(learning_rate);

  // Build network (LeNet-5 style for MNIST)
  cnn.push_back("I1", "input 28 28 1");
  cnn.push_back(
      "C1", "convolution 5 8 1 elu");  // 28->24 (no padding) or 28->28 (same)
  cnn.push_back("P1", "max_pool 2 2"); // 28->14 or 24->12
  cnn.push_back("C2", "convolution 5 16 1 elu");
  cnn.push_back("P2", "max_pool 2 2");
  cnn.push_back("C3", "convolution 3 32 1 elu");
  cnn.push_back("FC1", "fully_connected 64 relu");
  cnn.push_back("FC2", "softmax 10");

  cnn.init();

  std::cout << "==  Network Configuration  =================================="
            << std::endl;
  std::cout << cnn.getConfiguration() << std::endl;

  // Prepare batch buffers
  int train_samples = (int)train_images.size();
  int input_dim = 28 * 28;

  std::vector<float> input_batch(batch_size * input_dim);
  std::vector<int> label_batch(batch_size);

  // Training loop
  mojo::progress overall_progress(-1, "  overall:\t\t");

  // Start total timer
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    overall_progress.draw_header("Epoch " + std::to_string(epoch + 1), true);

    mojo::progress progress(train_samples, "  [GPU] training:\t");
    cnn.startEpoch();

    // Shuffle training data
    std::vector<int> indices = mojo::get_shuffled_indices(train_samples);

    auto epoch_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < train_samples; i += batch_size) {
      int current_batch = std::min(batch_size, train_samples - i);

      // Prepare batch
      for (int b = 0; b < current_batch; b++) {
        int idx = indices[i + b];
        const float *img = train_images[idx].data();
        std::copy(img, img + input_dim, input_batch.data() + b * input_dim);
        label_batch[b] = train_labels[idx];
      }

      // Train batch
      cnn.trainBatch(input_batch.data(), label_batch.data(), current_batch);

      if ((i / batch_size) % 50 == 0) {
        progress.draw_progress(i);
      }
    }

    cnn.endEpoch();

    auto epoch_end = std::chrono::high_resolution_clock::now();
    float epoch_time =
        std::chrono::duration<float>(epoch_end - epoch_start).count();

    std::cout << std::endl;
    std::cout << "  mini batch:\t\t" << batch_size << std::endl;
    std::cout << "  training time:\t" << epoch_time << " seconds on GPU (cuDNN)"
              << std::endl;
    std::cout << "  model updates:\t" << cnn.train_updates << std::endl;
    std::cout << "  estimated accuracy:\t" << cnn.estimated_accuracy << "%"
              << std::endl;

    // Test accuracy with batch inference
    float accuracy = test(cnn, test_images, test_labels, batch_size);
    std::cout << "  test accuracy:\t" << accuracy << "% ("
              << (100.0f - accuracy) << "% error)" << std::endl;
    std::cout << std::endl;

    // Early stopping if very high accuracy
    if (accuracy > 99.0f) {
      std::cout << "Reached 99% accuracy, stopping." << std::endl;
      break;
    }
  }

  // Calculate total duration
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_seconds = end_time - start_time;

  std::cout << std::endl;
  std::cout << "Training complete!" << std::endl;
  std::cout << "Total Training Time: " << total_seconds.count() << " seconds"
            << std::endl;

  return 0;
}
