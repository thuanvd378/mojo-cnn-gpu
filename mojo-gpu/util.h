// == mojo-gpu ================================================================
//    util.h: Utility classes and functions
// ============================================================================

#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>


namespace mojo {

// ============================================================================
// Progress Bar and Timer
// ============================================================================
class progress {
private:
  int total;
  std::string title;
  std::chrono::high_resolution_clock::time_point start_time;

public:
  progress(int total_ = 100, const std::string &title_ = "")
      : total(total_), title(title_) {
    start_time = std::chrono::high_resolution_clock::now();
  }

  void reset(int total_ = -1, const std::string &title_ = "") {
    if (total_ > 0)
      total = total_;
    if (!title_.empty())
      title = title_;
    start_time = std::chrono::high_resolution_clock::now();
  }

  void draw_progress(int current) {
    if (total <= 0)
      return;

    float elapsed = elapsed_seconds();
    int percent = (current * 100) / total;
    float remaining =
        (current > 0) ? (elapsed / current) * (total - current) : 0;

    std::cout << "\r" << title << percent << "% (" << (int)remaining
              << "sec remaining)              " << std::flush;
  }

  void draw_header(const std::string &header, bool newline = true) {
    std::cout << "==  " << header << "  ";
    int len = 60 - header.length();
    for (int i = 0; i < len; i++)
      std::cout << "=";

    // Time elapsed
    float elapsed = elapsed_seconds();
    int hours = (int)elapsed / 3600;
    int mins = ((int)elapsed % 3600) / 60;
    int secs = (int)elapsed % 60;

    std::cout << " " << hours << ":" << std::setfill('0') << std::setw(2)
              << mins << ":" << std::setfill('0') << std::setw(2) << secs;

    if (newline)
      std::cout << std::endl;
  }

  float elapsed_seconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>(now - start_time).count();
  }
};

// ============================================================================
// Shuffle indices for training
// ============================================================================
inline std::vector<int> get_shuffled_indices(int size) {
  std::vector<int> indices(size);
  for (int i = 0; i < size; i++)
    indices[i] = i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  return indices;
}

// ============================================================================
// Float to string helper
// ============================================================================
inline std::string float2str(float val, int precision = 2) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << val;
  return oss.str();
}

// ============================================================================
// GPU Info
// ============================================================================
inline void printGPUInfo() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "[GPU] " << prop.name << std::endl;
  std::cout << "[GPU] Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "[GPU] Memory: " << (prop.totalGlobalMem / (1024 * 1024))
            << " MB" << std::endl;
  std::cout << "[GPU] CUDA Cores: " << prop.multiProcessorCount * 128
            << " (approx)" << std::endl;
}

} // namespace mojo
