```
 ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗     
██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║     
██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║     
██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║     
╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
```


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

[//]: # (Deprecated the mention of Boost from the README)
**Core stack:**     
![C++](https://img.shields.io/badge/C++20-00599C?style=for-the-badge\&logo=c%2B%2B\&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen3.0-1F1232?style=for-the-badge\&logo=matrix\&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-26667F?style=for-the-badge\&logo=openmp\&logoColor=white)
![OpenBLAS](https://img.shields.io/badge/OpenBLAS-E00?style=for-the-badge\&logo=openblas\&logoColor=white)

**Development and Profiling:**       
![CMake](https://img.shields.io/badge/CMake-06466B?style=for-the-badge\&logo=cmake\&logoColor=white)
![perf](https://img.shields.io/badge/Perf-E03C31?style=for-the-badge\&logo=linux\&logoColor=white)

**Integration underway:**        
![GTest](https://img.shields.io/badge/GTest-00BF63?style=for-the-badge\&logo=googletest\&logoColor=white)


## Migrating the library architecture (As of Jan 1, 2026) 
Glacier.ML is migrating from a header-only architecture to a linker-object library architecture. This decision was made 
deliberately to accommodate the growing complexity in developing the library's modules. This can be greatly solved by 
moving from a monolithic header-only architecture, to a modular architecture as that of a linked-object 
architecture, improving binary stability.

A header-only library involves declaring all the functions and their definitions in a singular `.hpp` file. While the 
functions are all declared at the beginning of the file, they are also defined in the same file, using the `inline` keyword
before the function definition. The project began with this architecture to explore the scope while maintaining 
architectural simplicity, allowing the author to focus more on the logic and minimally on their arrangement. 

Whereas using a linked-object library architecture allows the same header-file to be broken down into multiple smaller 
chunks of logic with files of their own. This in turn allows for faster refactoring and lesser compile time after the initial 
build by the end user. The consumer will now have to use a complex build command, which can always be overcome by
a correctly written CMakeLists file. But it is the developer who benefits the most, where clarity and the ability to refactor
the code faster, beat the other trade-offs. 

Once the migration is complete, the author plans on exploring *Link Time Optimizations* options to compensate for the performance 
drop derived from the employed architecture. 


## Models benchmarked so far:
* Logistic Regression
* KNN Classifier
* SVM Classifier


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


## Models benchmarked on:
* **AMD Ryzen 6600H processor** with 6 cores and 12 threads
