#include "MultipleLinearRegressionFlow.h"
#include <iostream>
#include <chrono>

#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";                                                // high level info while users are using it
#define LOG_ERROR(x) std::cout << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);            // errors and exits
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " milliseconds. \n";         // time taken

#if DEBUG_MODE
    #define LOG_DEBUG(x, xval) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << xval << "\n";                     // deeper into to be used during development
#else
    #define LOG_DEBUG(x)
#endif


Multiple_Linear_Regression::Multiple_Linear_Regression(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i) : X(), Y(), Beta(), E() {
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
};

void Multiple_Linear_Regression::train() {
    auto train_start = std::chrono::high_resolution_clock::now();

    // Beta is (n x m) as X is (m * n)
    // Beta = (X.transpose() * X).inverse() * X.transpose() * Y;                                                        // (m x n) * (n * m) * (m * n) * (n x 1)
    // Beta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    Beta = (X.transpose() * X).completeOrthogonalDecomposition().solve(X.transpose() * Y);

    LOG_DEBUG("Number of rows in Beta", Beta.rows());
    LOG_DEBUG("Number of cols in Beta", Beta.cols());
    std::cout << "\n";

    E = Y - X * Beta; // (n * 1)

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

void Multiple_Linear_Regression::print_Rcoeff_values() {
    LOG_INFO("Regression coefficients: ")
    for (int i = 0; i < Beta.rows(); i++)
        std::cout << "B" << i << ": " << Beta(i) << "\n";
    std::cout << "\n";
}

float Multiple_Linear_Regression::predict(std::vector<float> &x_pred) {
    auto beta_size = Beta.size();
    if (x_pred.size() + 1 != beta_size) {
        LOG_ERROR(("Incompatible size of vector passed. Expected size: " + std::to_string(beta_size)));
        return -1;
    }

    Eigen::VectorXf x(beta_size);

    x(0) = 1.0f;
    for (size_t i = 1; i < beta_size; i++) {
        x(i) = x_pred[i - 1];
    }

    return x.dot(Beta);
}

std::vector<float> Multiple_Linear_Regression::predict(std::vector<std::vector<float>> &X) {
    if (Beta.size() == 0) {
        LOG_ERROR("Train the data using train() before using predict().");
    }

    /*
     * size_t is unsigned long long, whereas Eigen uses long long for indexing and
     * sizing hence using this explicit conversion over here - Eigen::Index
     * instead of auto
     */

    auto nrows = static_cast<Eigen::Index>(X.size());
    auto ncols = static_cast<Eigen::Index>(X[0].size());

    Eigen::MatrixXf X_pred(nrows, ncols + 1);
    X_pred.col(0) = Eigen::VectorXf::Ones(nrows);

    for (Eigen::Index row = 0; row < nrows; row++)
        for (Eigen::Index col = 0; col < ncols; col++)
            X_pred(row, col + 1) = X[row][col];                                                                         // X matrix, with 0th column as 1

    Eigen::VectorXf Y_pred = X_pred * Beta;

    std::vector<float> result(nrows);
    for (Eigen::Index i = 0; i < nrows; i++)
    result[i] = Y_pred[i];

    return result;
}

float Multiple_Linear_Regression::R_squared() {
    float sst = (Y.array() - Y.mean()).square().sum();
    float ssr = E.squaredNorm();
    return 1 - ssr / sst;
}

void Multiple_Linear_Regression::analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) {
    if (x_test.size() != y_test.size()) {
        LOG_ERROR("Test size does not match.");
    }

    float n_test = x_test.size();
    float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;

    std::vector<float> y_pred = predict(x_test);
    for (size_t i = 0; i < y_pred.size(); i++) {
        float error = y_test[i] - y_pred[i];
        mse += error * error;
        mae += abs(error);
        if (y_test[i] != 0) {
            mape += std::abs(error / y_test[i]);
        }
    }

    mse = mse / n_test;
    rmse = std::sqrt(mse);
    mae = mae / n_test;
    mape = mape / n_test * 100;

    LOG_INFO("Evaluation metrics: ");
    std::cout << "R_squared: " << R_squared() << "\n";
    std::cout << "MSE: " << mse << "\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "MAE: " << mae << "\n";
    std::cout << "MAPE: " << mape << "\n";
}
