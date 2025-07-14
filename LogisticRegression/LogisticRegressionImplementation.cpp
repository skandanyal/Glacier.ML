//
// Created by skandan-c-y on 7/2/25.
//

#include <iostream>
#include <chrono>
#include "LogisticRegressionFlow.h"

#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);		    // errors and exits
#define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"							// deeper info to be used during development
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";												// high level info while users are using it
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " nanooseconds. \n";					// time taken

#if DEBUG_MODE
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"						// deeper info to be used during development
#else
    #define LOG_DEBUG(x, x_val)
#endif


// constructor
Logistic_Regression::Logistic_Regression(std::vector<std::vector<double> > &X_i, std::vector<std::vector<double> > &Y_i) : X(), Y(), Beta(), F_x() {
    if (X_i.empty() || Y_i.empty()) {                                 // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    for (auto &row : X_i)                                                                                  // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
        }

    for (size_t i = 0; i < X_i.size(); i++) {
        for (size_t j = 0; j < X_i[i].size(); j++) {
            if (!std::isfinite(X_i[i][j]))
                std::cout << "[BAD VALUE] X_i[" << i << "][" << j << "] = " << X_i[i][j] << "\n";
        }
        if (!std::isfinite(Y_i[i]))
            std::cout << "[BAD VALUE] Y_i[" << i << "] = " << Y_i[i] << "\n";
    }

    Eigen::Index nrows = X_i.size();
    Eigen::Index ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in X before adding 1 column", nrows);
    LOG_DEBUG("Number of columns in X before adding 1 column", ncols);
    std::cout << "\n";

    X = Eigen::MatrixXf(nrows, ncols + 1);                                                                          // use this method to resize an eigen matrix;
    Y = Eigen::VectorXf(nrows);                                                                                         // use (nrows, 1) to ensure column vector

    X.col(0) = Eigen::VectorXf::Ones((Eigen::Index) nrows);

    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X(row, col + 1) = X_i[row][col];                                                                            // X matrix, with 0th column as 1
    LOG_DEBUG("Number of rows in x_train", X.rows());
    LOG_DEBUG("Number of cols in x_train", X.cols());
    std::cout << "\n";

    size_t Y_i_size = Y_i.size();
    for (size_t i = 0; i < Y_i_size; i++)
        Y(i) = Y_i[i];
    LOG_DEBUG("Number of rows in y_train", Y.rows());
    LOG_DEBUG("Number of cols in y_train", Y.cols());
    std::cout << "\n";
}

void Logistic_Regression::train() {
    auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    Beta = (X.transpose() * X).completeOrthogonalDecomposition().solve(X.transpose() * Y);

    LOG_DEBUG("Number of rows in Beta", Beta.rows());
    LOG_DEBUG("Number of cols in Beta", Beta.cols());
    std::cout << "\n";

    F_x = X * Beta; // (n * 1)                              logit function
    P_x = 1/(1 + std::exp((-1 * F_x).value()));        // sogmoid function, float

    ////////////////////// Training ends here /////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

