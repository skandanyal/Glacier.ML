## High Level Design:

|                    Architecture overview                    |                        Workflow overview                        |
|:-----------------------------------------------------------:|:---------------------------------------------------------------:|
|                 Modular framework structure                 |                      Input data ingestion                       |
| Core ML algorithms (KNN, SVM, extensible for future models) |     Preprocessing (cleaning, normalization, transformation)     |
|    Optimization layer (OpenMP, SIMD, memory management)     |      Algorithm selection (KNN, Logistic Regression, etc.)       |
| Data handling layer (loading, preprocessing, normalization) | Model execution (distance computation, kernel evaluation, etc.) |
|       Abstraction boundaries for clean extensibility        |    Parallelization & vectorization applied at critical steps    |
|    Separation of algorithm logic vs. optimization logic     |               Results aggregated into predictions               |
|               Designed for future scalability               |       Output delivered (predictions, performance metrics)       |

## Low Level Design:

|                 Data input &<br>Pre-processing<br>`constructor()`                  |                   Core Model Logic<br>`.train()`                   |
|:----------------------------------------------------------------------------------:|:------------------------------------------------------------------:|
|                               Constructor loads data                               |                      Accepts hyper-parameters                      |
| Checks for anomalies in the dataset,<br>Normalizes the data for easier computation |         Conducts model specific mathematical computations          |
|               Builds the Eigen matrices for mathematical computation               | Stores parameters and hyprer-parameters for predicting the results | 

|              Optimization layer<br>`constructor()` & `.train()`               |      Results and Evaluation<br>`.predict()` & `.analyze()`       |
|:-----------------------------------------------------------------------------:|:----------------------------------------------------------------:|
|           Uses cache-friendly flat matrices for memory optimization           | Produces the result using stored parameters and hyper-parameters |
| Uses OpenMP to implement multithreading and SIMD instructions (vectorization) |            Generates task specific evaluation metrics            |

## Operating System 
|           OS            |     Purpose     |
|:-----------------------:|:---------------:|
|   Linux (Ubuntu 24.0)   | For development |
| Linux / Windows / MacOS |   For useage    |

## Programming languages and libraries
|        Purpose         |                  Language and libraries                  |
|:----------------------:|:--------------------------------------------------------:|
|       Core logic       |          C++ 20, Eigen, Boost, OpenMP, OpenBLAS          |
| Website infrastructure |       Golang, HTML and CSS, Javascript, Javascript       |
|        Database        |                        PostgreSQL                        |
|      Benchmarking      | Python, Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn |

## Tools
|      Tools      |              Purpose              |
|:---------------:|:---------------------------------:|
| Version control |          Git and GitHub           |
|  Build System   |               Cmake               |
|       IDE       |         CLion and PyCharm         |
|     Server      | Golang with Gin or Echo framework |
|    Profiling    |               Perf                |
|     Testing     |               GTest               |

## Hardware requirements
|     Hardware     |                      requirements                      |
|:----------------:|:------------------------------------------------------:|
|    Processor     | Any multi-core CPU<br>Intel i5 / AMD Ryzen 5 or better |
|      Memory      |           Minimum 4 GB RAM, 8 GB recommended           |
| Any other device |                  Standard PC / Laptop                  | 

## Literature survey - i
| Sl no. |                                         Title                                          |           Authors           | Year of publication |
|:------:|:--------------------------------------------------------------------------------------:|:---------------------------:|:-------------------:|
|   1.   |     A Survey on Machine Learning Accelerators and Evolutionary Hardware Platforms      | Sathwika Bavikadi<br>et al. |        2022         |  
|   2.   | Accelerating Learning to Rank via SVM with OpenCL and OpenMP on Heterogenous Platforms |    Huming  Zhu<br>et al.    |        2016         |  
|   3.   |         A Decision support tool for Predicting patients at risk of readmission         |    Eren Demir<br>et al.     |        2014         |  
|   4.   |       Importance of Explicit Vectorization for CPU and GPU Software Performance        |  Neil G Dickson<br>et al.   |        2010         |  
|   5.   |                                   MLPACK: A Scalable                                   |                             |                     |

## Literature survey - ii
| Sl no. |                                                      Title                                                      |            Authors            | Year of publication |
|:------:|:---------------------------------------------------------------------------------------------------------------:|:-----------------------------:|:-------------------:|
|   1.   |      Optimization Of KNN, SVM, And SVM Kernel in Water Potability Prediction with Hyperparameter Approach       | Roy Hendro Siburian<br>et al. |        2025         |
|   2.   |               Ensemble-based Machine Learning and Deep Learning Approaches for Autism Prediction                |   Dr.K.Suthendran<br>et al.   |        2025         |
|   3.   |    Simulation of Hierarchical Parallel Computing Model for Fluid Machinery Based on Support Vector Machines     |     Dechen Wei<br>et al.      |        2023         |
|   4.   | Design and Implementation of Parallel Processing Algorithm for Big Data in High Performance Computing Framework |     Qihao Dong<br>et al.      |        2023         |
|   5.   |                       Comparative Analysis Of Optimization Techniques On Multi-class SVM                        |  Neha J Deshpande<br>et al.   |        2022         |
|   6.   |                          Performance portable Vlasov code with C++ parallel algorithm                           |    Yuuichi Asahi<br>et al.    |        2022         |

### Observations
1. Modern HPC applications have been abstracted by Python.
2. Could not find papers on optimizing Supervised ML algorithms in the last 5 years on IEEE Xtreme.
3. Algorithmic optimization trends are appearing more in applications in edge devices

### Try to learn about:
1. OpenMPI
2. MapReduce method

## Next steps
time and memory benchmarks, algorithmic improvements
add newer research papers 
