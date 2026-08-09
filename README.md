```
 ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
```

# Glacier.ML

Glacier.ML is a **performance-oriented C++20 numerical algorithms** library for implementing and studying classical machine learning algorithms on modern multicore CPUs.

## The project prioritizes:

- explicit cost models
- predictable memory access
- parallel execution
- numerical correctness verification
  It is designed as a learning and experimentation platform, not as a production ML framework.

## Scope

Glacier.ML focuses on:

- classical supervised ML algorithms
- performance-critical numerical kernels
- empirical benchmarking and profiling
- reproducible experiments
  The library is intentionally kept minimal.

## Non-goals

Glacier.ML is not:

- a deep learning framework
- a PyTorch / TensorFlow alternative
- a GPU-first system (for now)
- an auto-differentiation engine
- a deployment or inference runtime
  These exclusions are deliberate.

## Language and Core Stack

**Language:**  
![C++20](https://img.shields.io/badge/C++20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

**Numerical & Parallelism**  
![Eigen](https://img.shields.io/badge/Eigen3.0-1F1232?style=for-the-badge&logo=matrix&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-26667F?style=for-the-badge&logo=openmp&logoColor=white)
![OpenBLAS](https://img.shields.io/badge/OpenBLAS-E00?style=for-the-badge&logo=openblas&logoColor=white)

**Build and Profiling:**  
![CMake](https://img.shields.io/badge/CMake-06466B?style=for-the-badge&logo=cmake&logoColor=white)
![perf](https://img.shields.io/badge/Perf-E03C31?style=for-the-badge&logo=linux&logoColor=white)

**Testing**  
![GTest](https://img.shields.io/badge/GTest-E03C31?style=for-the-badge&logo=gtest&logoColor=white)
[![Glacier.ML CI](https://github.com/skandanyal/Glacier.ML/actions/workflows/main.yml/badge.svg)](https://github.com/skandanyal/Glacier.ML/actions/workflows/main.yml)

## Architecture

Glacier.ML is migrating from a **header-only** library architecture, to a **compiled library (linked-object architecture)**.
This allows for:

- modular compilation, shorter compile times
- easier refactoring and integration enabling faster development
- explicit boundaries
- extended binary stability
  Link-Time Optimization (LTO) will be evaluated after architectural stabilization.

## Benchmarked Algorithms

|                    Models                     |             Comparision against `Scikit-learn`              |
| :-------------------------------------------: | :---------------------------------------------------------: |
|              Logistic Regression              | parity in smaller datasets (1000x10), upto 2x slower beyond |
|       k-Nearest Neighbors (Classifier)        |                      4x to 30x slower                       |
| Support Vector Machine - PEGASOS (Classifier) |                      4x to 10x faster                       |

- Results derived considering Wall clock time taken to initialize and train a model and to predict on a dataset of size 10000x10.
- Hot spots identified using `perf`.

## Benchmarking Environment

Benchmarks have been conducted on:

```
AMD Ryzen 6600H (6 cores / 12 threads)
```

Benchmark results are exploratory and used primarily for relative comparison and profiling, not for leaderboard claims.

## Testing Environment

Models are tested on both local (as mentioned above) and cloud runtimes with the following configurations:

- OS: Linux Ubuntu-24.04
- Cloud instance: Through automated GitHub workflows

## These implementations are used as vehicles for studying:

- memory layouts
- vectorization opportunities
- threading strategies
- algorithmic trade-offs

## Example Usage

```C++
#include "Glacier/Models/MLmodel.hpp"
#include "Glacier/Utils/utilities.hpp"

int main() {
    std::vector<std::vector<float>> X, X_t;
    std::vector<std::string> y, y_t;

    Glacier::Utils::read_csv_c("../Datasets/training_dataset.csv", X, y, true);
    Glacier::Utils::read_csv_c("../Datasets/testing_dataset.csv", X_t, y_t, true);

    Glacier::Models::MLmodel model(X, y);

    float regularization = 1.0f;
    model.train(regularization);

    model.analyze_2_targets(X_t, y_t);
    return 0;
}
```

This represents a high-level convenience path. Lower-level components are exposed for experimentation and profiling.

## Windows Terminal Note

To enable ANSI color output on Windows terminals:

```
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif
```

## Status

`ITERATION 2 WORK IN PROGRESS`

- Models now tested for stability and mathematical correctness using `GTest`.
- GitHub workflow based `Continuous Integration` incorporated.
- This repository reflects an evolving understanding of performance-oriented system design.

## License

See `LICENSE.txt.`
