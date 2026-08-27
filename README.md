```
 ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗         ██╗   ██╗██████╗ 
██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║         ██║   ██║╚════██╗
██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║         ██║   ██║ █████╔╝
██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║         ╚██╗ ██╔╝██╔═══╝ 
╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗     ╚████╔╝ ███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝      ╚═══╝  ╚══════╝
```

# Glacier.ML (v2)

[![C++20](https://img.shields.io/badge/C%2B%2B20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![Eigen3](https://img.shields.io/badge/Eigen3.0-1F1232?style=for-the-badge&logo=matrix&logoColor=white)](https://eigen.tuxfamily.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-26667F?style=for-the-badge&logo=openmp&logoColor=white)](https://www.openmp.org/)
[![OpenBLAS](https://img.shields.io/badge/OpenBLAS-E00?style=for-the-badge&logo=openblas&logoColor=white)](https://www.openblas.net/)
[![CMake](https://img.shields.io/badge/CMake-06466B?style=for-the-badge&logo=cmake&logoColor=white)](https://cmake.org/)
[![GTest](https://img.shields.io/badge/GTest-E03C31?style=for-the-badge&logo=gtest&logoColor=white)](https://github.com/google/googletest)
[![Glacier.ML CI](https://github.com/skandanyal/Glacier.ML/actions/workflows/main.yml/badge.svg)](https://github.com/skandanyal/Glacier.ML/actions/workflows/main.yml)

**Glacier.ML** is a performance-oriented C++20 numerical algorithms library for classical machine learning on modern multicore CPU architectures.

With **v2**, Glacier.ML establishes a design architecture centered on a **strict separation between high-level orchestration layers and pure computational numerical cores**.

---

## Architecture Paradigm: Decoupled Core vs Orchestration Layer

Glacier.ML v2 structures algorithm design around a two-tiered system architecture. This ensures that dataset management, format conversions, and I/O validation never interfere with performance-critical mathematical loops.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Orchestration Layer (High-Level API)                  │
│  - Dataset ingestion & input validation                                     │
│  - Parallel feature standardization (Z-score scaling via OpenMP)            │
│  - Label encoding & domain string-to-numeric translation                    │
│  - Thread allocation & execution policies                                   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼  (Normalized Dense Eigen Matrices)
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Numerical Core Layer (Computation Kernel)                │
│  - Pure matrix/vector operations (Eigen3 + SIMD / OpenBLAS)                 │
│  - Pre-allocated workspace buffers (zero heap allocations in hot loops)    │
│  - Mathematical stability safeguards & explicit activation clamping         │
│  - Agnostic to dynamic strings, logging, or dataset file structures         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Pillars of v2

### 1. Orchestrator vs. Numerical Kernel Separation
* **Orchestration Layer**: Manages user-facing data types (`std::vector`), dataset validation, lexicographical label mapping, OpenMP parallel feature scaling, and thread count configurations.
* **Numerical Core Layer**: Executes core mathematical optimization kernels operating purely on continuous memory buffers (`Eigen::MatrixXf`, `Eigen::VectorXf`). The core layer contains zero heap allocations during training iterations, no dynamic string processing, and no disk/logging overhead.

### 2. Zero Hot-Loop Dynamic Allocations
To eliminate cache churn and memory fragmentation, model workspace buffers are pre-allocated **once** during kernel initialization or before entering optimization iterations. Hot loops execute strictly in-place.

### 3. Numerical Stability Safeguards
Kernels incorporate defensive mathematical clamping (e.g., bounding exponential activation inputs within safe floating-point limits $[-50.0, +50.0]$) to guard against numerical overflow, underflow, and NaN propagation during training.

### 4. Header-only Library Architecture -> Compiled Library Architecture (v2)
Glacier.ML v2 transitions from legacy header-only templates to a **compiled static library architecture**. This yields faster compilation cycles, clean ABI boundaries, and enables downstream Link-Time Optimization (LTO).

---

## Project Structure

```
Glacier.ML/
├── .github/workflows/       # GitHub Actions CI workflow (main.yml)
├── include/Glacier/         # Public API headers (Orchestration Layer)
│   ├── Models/              # User-facing model interfaces
│   └── Utils/               # Utilities & logging
├── src/                     # Source implementations
│   ├── Models/              # Model implementations & decoupled Core kernels
│   └── Utils/               # Helper routines
├── tests/                   # Automated GoogleTest verification suites
├── benchmark_drivers/       # Performance driver benchmarks & profiling scripts
├── Model_README/            # Individual algorithm specifications
├── Datasets/                # Benchmark datasets
└── CMakeLists.txt           # Modern CMake C++20 build configuration
```

---

## Quickstart Usage

```cpp
#include <iostream>
#include <vector>
#include <string>
#include "Glacier/Models/LogisticRegression.hpp"

int main() {
    // 1. Ingest training data via Orchestration Layer
    std::vector<std::vector<float>> X_train = {
        {1.0f, 2.0f},
        {1.5f, 1.8f},
        {8.0f, 9.0f},
        {9.5f, 8.5f}
    };
    std::vector<std::string> y_train = {"Class_A", "Class_A", "Class_B", "Class_B"};

    // 2. Instantiate model with 2 thread workers
    Glacier::Models::Logistic_Regression model(X_train, y_train, /*no_threads=*/2);

    // 3. Train model (preprocessing & label mapping run before passing to Numerical Core)
    model.train(/*learning_rate=*/0.1f, /*iterations=*/500);

    // 4. Perform prediction
    std::vector<float> sample = {8.5f, 8.8f};
    std::string prediction = model.predict(sample, /*decision_boundary=*/0.5f);

    std::cout << "Prediction: " << prediction << std::endl;
    return 0;
}
```

---

## Building and Testing

### Prerequisites
* **C++ Compiler**: GCC $\ge 11$ or Clang $\ge 13$ (C++20 enabled)
* **Build Tools**: CMake $\ge 3.20$ & Ninja or Make
* **Libraries**: `Eigen3`, `OpenMP`, `OpenBLAS`

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/skandanyal/Glacier.ML.git
cd Glacier.ML

# Configure and compile
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running Test Suite

```bash
# Execute unit tests via CTest
ctest --test-dir build --output-on-failure
```

---

## Benchmarking & Performance

Wall-clock training and prediction execution compared against Python `scikit-learn` on a $10,000 \times 10$ dataset (Hardware: AMD Ryzen 6600H, 6C/12T):

| Algorithm | Comparison vs `scikit-learn` | Architectural Focus |
| :--- | :--- | :--- |
| **Logistic Regression** | Parity on small datasets; up to $2\times$ scale delta | Decoupled core kernel, clamped activation, zero-allocation hot loop |
| **k-Nearest Neighbors** | $4\times$ to $30\times$ baseline | Distance metric SIMD vectorization & cache locality |
| **Support Vector Machine (PEGASOS)** | $4\times$ to $10\times$ **faster** | Stochastic sub-gradient descent on primal formulations |

---

## Benchmarking Environment

Benchmarks have been conducted on:

```
AMD Ryzen 6600H (6 cores / 12 threads)
```

Benchmark results are exploratory and used primarily for relative comparison and profiling, not for leaderboard claims.

---

## Testing Environment

Models are tested on both local (as mentioned above) and cloud runtimes with the following configurations:

- OS: Linux Ubuntu-24.04
- Cloud instance: Through automated GitHub workflows

---

## Non-Goals

Glacier.ML intentionally excludes:
* Deep learning autograd engines or graph compilers.
* GPU runtime hardware targets (focused purely on multicore x86_64 CPU efficiency).
* Dynamic framework overhead or heavy external dependencies.

---

## License

Distributed under the terms specified in [`LICENSE.txt`](file:///home/skandan-c-y/CLionProjects/Glacier.ML/LICENSE.txt).
