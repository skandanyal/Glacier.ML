```
 ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗     
██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║     
██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║     
██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║     
╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
```
```Licensed for viewing only. Not open-source. See LICENSE.txt for details.```


Glacier.ML is a **performance-focused, header-only C++20 machine learning library**, designed to build and experiment with machine learning algorithms while emphasizing efficiency on modern multicore CPU architectures.

It provides:

* Core supervised machine learning algorithms implemented in modern C++
* Performance-oriented implementations of common ML computations (e.g., distance metrics, reductions, matrix operations)
* Plots, datasets, and reproducible benchmarks for empirical evaluation
* Test scaffolding to verify numerical correctness and algorithmic behavior


![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow?style=for-the-badge&logo=github&logoColor=black)
[![License](https://img.shields.io/badge/license-Glacier%20Custom%20License-blue?style=for-the-badge)](./LICENSE.txt)


## Languages and Frameworks used:
**Language:**
C++ 20

**Core stack:**     
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge\&logo=c%2B%2B\&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen-1F1232?style=for-the-badge\&logo=matrix\&logoColor=white)
![Boost](https://img.shields.io/badge/Boost-1F1232?style=for-the-badge\&logo=code\&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-26667F?style=for-the-badge\&logo=openmp\&logoColor=white)
![OpenBLAS](https://img.shields.io/badge/OpenBLAS-E00?style=for-the-badge\&logo=openblas\&logoColor=white)

**Development and Profiling:**       
![CMake](https://img.shields.io/badge/CMake-06466B?style=for-the-badge\&logo=cmake\&logoColor=white)
![perf](https://img.shields.io/badge/Perf-E03C31?style=for-the-badge\&logo=linux\&logoColor=white)

**Integration underway:**        
![GTest](https://img.shields.io/badge/GTest-00BF63?style=for-the-badge\&logo=googletest\&logoColor=white)


## Resources used:
[![Statistical Learning (Stanford)](https://img.shields.io/badge/Statistical%20Learning-Stanford%20Online-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/playlist?list=PLoROMvodv4rPP6braWoRt5UCXYZ71GZIQ&si=0pQvuCwQpy7xMw9u)
[![Applied Multivariate Statistical Analysis](https://img.shields.io/badge/Multivariate%20Analysis-NPTEL-orange?style=for-the-badge&logo=nptel&logoColor=white)](https://youtube.com/playlist?list=PLbMVogVj5nJRt-ZxRG1KRjxNoy7J_IaW2&si=CrIS5DlyWzsbJeCj)
![Core ML Concepts](https://img.shields.io/badge/Core%20ML%20Concepts-4th%20Sem%20Lab-blueviolet?style=for-the-badge&logo=code&logoColor=white)


## If using Windows:
Use this block of code in the beginning of working file to enable ANSI colour logging in Windows terminal:
```
#ifdef _WIN32           
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD dwMode = 0;
	GetConsoleMode(hOut, &dwMode);
	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	SetConsoleMode(hOut, dwMode);
#endif
```


## Demo
```
#include "Glacier/Models/MLmodel.hpp
#include "Glacier/Utils/utilities.hpp"

int main() {
    std::vector<std::vector<float>> X, X_t;
    std::vector<std::string> y, y_t;

    Glacier::Utils::read_csv_c("../Datasets/training_dataset.csv", X, y, true);
    Glacier::Utils::read_csv_c("../Datasets/testing_dataset.csv", X_t, y_t, true);

    std::vector<std::vector<float>> X_p = {
        {0.002020161, 43, 0, 0.228941685, 12500.0, 9, 0, 2, 0, 1.0}
    };
    std::vector<std::string> y_p = {"was_in_trouble"};

    // Initialize models
    Glacier::Models::MLmodel md(X, y);

    // Hyperparameters
    float hp1=1;

    // Train
    md.train(hp1);

    // Predict sample
    auto md_pred = md.predict(X_p);

    // Analysis on validation set
    md.analyze_2_targets(X_t, y_t);

    return 0;
}
```


## Models benchmarked so far:

|     Classifiers     | Regressors |
|:-------------------:|:----------:|
| Logistic Regression |     -      |
|   KNN Classifier    |     -      |
|   SVM Classifier    |     -      |


## Models benchmarked on:
* **AMD Ryzen 6600H processor** with 6 cores and 12 threads
