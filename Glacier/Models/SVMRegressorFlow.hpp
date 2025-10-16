//
// Created by skandan-c-y on 9/14/25.
//

#ifndef SVMREGRESSORFLOW_HPP
#define SVMREGRESSORFLOW_HPP

#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <cblas.h>
#include <Eigen/Dense>

#include "Utils/logs.hpp"
#include "Utils/utilities.hpp"

class SVMRegressor {
    // the {} braces are for the constructor to initialize these variables outside the constructor
    Eigen::MatrixXf X;                      // (n x p)
    Eigen::VectorXf Y;                      // (n x 1)
    Eigen::VectorXf mean;                   // (p x 1)
    Eigen::VectorXf std_dev;                // (p x 1)
    Eigen::Index nrows{}, ncols{};
    Eigen::VectorXf w;
    std::vector<size_t> indices;

public:
    SVMRegressor(std::vector<std::vector<float>> &x_i, std::vector<float> &y_i);
    void train(float lambda, float epsilon, int epochs);
    float predict(std::vector<float> &x_pred);
    std::vector<float> predict(std::vector<std::vector<float>>& x_test);
    void print_predict(std::vector<std::vector<float>>& x_val, std::vector<float> &y_val);
    // mimic predict_proba from sklearn
    // void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);
    // void show_final_weights();
    // void show_support_vectors(); // no. of features = no. of SVs

private:
    static float R2_score(std::vector<float> &actual, std::vector<float> &predicted);
};


// constructor
inline SVMRegressor::SVMRegressor(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i) {

    // Check if the inputs are valid or not
    // check for empty dataset
    if (X_i.empty() || Y_i.empty()) {
        LOG_ERROR("Input data cannot be empty.");
    }

    // check for inconsistency in the dataset
    for (auto &row : X_i)                                                                                  // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
        }

    // check for infinite values in the dataset
    for (size_t i = 0; i < X_i.size(); i++) {
        for (size_t j = 0; j < X_i[i].size(); j++) {
            if (!std::isfinite(X_i[i][j]))
                std::cout << "[BAD VALUE] X_i[" << i << "][" << j << "] = " << X_i[i][j] << "\n";
        }
        if (!std::isfinite(Y_i[i]))
            std::cout << "[BAD VALUE] Y_i[" << i << "] = " << Y_i[i] << "\n";
    }

    nrows = X_i.size();
    ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in X before adding 1 column", nrows);
    LOG_DEBUG("Number of columns in X before adding 1 column", ncols);
    std::cout << "\n";

    // Inititalizing Eigen X and Y matrix
    X = Eigen::MatrixXf(nrows, ncols + 1);                                                                          // use this method to resize an eigen matrix;
    Y = Eigen::VectorXf(nrows);                                                                                         // use (nrows, 1) to ensure column vector

    X.col(0) = Eigen::VectorXf::Ones((Eigen::Index) nrows);

    // populating the Eigen X matrix with the data
    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X(row, col + 1) = X_i[row][col];                                                                            // X matrix, with 0th column as 1
    LOG_DEBUG("Number of rows in x_train", X.rows());
    LOG_DEBUG("Number of cols in x_train", X.cols());
    std::cout << "\n";

    // populating the Eigen Y matrix with data
    size_t Y_i_size = Y_i.size();
    for (size_t i = 0; i < Y_i_size; i++)
        Y(i) = Y_i[i];
    LOG_DEBUG("Number of rows in y_train", Y.rows());
    LOG_DEBUG("Number of cols in y_train", Y.cols());
    std::cout << "\n";

    // normalize X
    for (int colm = 0; colm < ncols; colm++) {
        // calculating mean and standard deviation for normalization
        mean(colm) = X.col(colm).sum() / (float) nrows;
        std_dev(colm) = std::sqrt((X.col(colm).array() - mean(colm)).square().sum() / (float) nrows);

        // normalizing X
        X.col(colm).array() = (X.col(colm).array() - mean(colm)) / std_dev(colm);
    }
    LOG_UPDATE("Mean and Std_dev vectors are formed.");
    LOG_UPDATE("Dataset normalized.");
}

inline void SVMRegressor::train(float lambda, float epsilon, int epochs) {
      auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    /*
     * eta = 1 / (lambda * step count)
     *
     * Inside the epsilon tube:
     *      w(t+1) = (1 - eta(t) * lambda) * w(t)
     *      b(t+1) = b
     *      no loss, regularization only on w
     *
     * Outside the epsilon tube:
     *      Under predicted - y(t) - P(t) > Epsilon
     *             w(t+1) = (1 - n(t)*lambda)*w(t) + n(t)*X(t)
     *             b(t+1) = b(t) + n(t)
     *
     *      Over Predicted - P(t) - y(t) > Epsilon
     *             w(t+1) = (1-n(t)*lambda)*w(t) - n(t)*X(t)
     *             b(t+q) = b(t) - n(t)
     */

    // w has just as  many cols as the training dataset, plus the bias
    w = Eigen::VectorXf::Zero((Eigen::Index)ncols + 1);

    // random engine to shuffle the dataset after every epoch
    std::random_device rd;
    std::mt19937 gen(rd());

    // initialize a vector containing all the unique values of indices in the X matrix
    indices.resize(nrows);
    std::iota(indices.begin(), indices.end(), 0);

    const int dim_full = (int)w.size();
    const int dim_w = (int)ncols;
    const int bias_idx = ncols;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t row = 0; row < nrows; row++) {
            size_t step = row+1;
            // eta = 1 / (lambda * step count)
            float eta = 1.0f / (float)(lambda * step);

            auto dim = w.size();
            int curr_idx = (int)indices[row];

            // prediction
            float score = cblas_sdot(dim_full, w.data(), 1, X.row(curr_idx).data(), 1);

            // residual / error
            float residual = score - Y(curr_idx);

            // Regularization step -> w = (1 - eta * lambda) * w
            // scaling it regardless of prediction
            cblas_sscal(dim, 1.0f - eta*lambda, w.data(), 1);

            /*
            * Outside the epsilon tube:
            *      Under predicted - y(t) - P(t) > Epsilon
            *             w(t+1) = (1 - n(t)*lambda)*w(t) + n(t)*X(t)
            *             b(t+1) = b(t) + n(t)
            *
            *      Over Predicted - P(t) - y(t) > Epsilon
            *             w(t+1) = (1-n(t)*lambda)*w(t) - n(t)*X(t)
            *             b(t+q) = b(t) - n(t)
             */
            // loss gradient
            float alpha = 0.0f;

            if (residual < epsilon) { // P_i - y_i > epsilon - over prediction
                alpha = -eta;
            } else if (residual > epsilon) {// P_i - y_i < epsilon - under prediction
                alpha = eta;
            }

            if (std::abs(alpha) > 1e-8) {
                // loss update on the feature weights
                cblas_saxpy(dim_w, alpha, X.row(curr_idx).data(), 1, w.data(), 1);

                // update bias term
                w(bias_idx) += alpha;
            }
        }
    }

    ////////////////////// Training ends here ////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

inline float SVMRegressor::predict(std::vector<float> &x_pred) {
    LOG_INFO("Singular prediction initiated...");

    if (x_pred.size() != nrows) {
      LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////


    ////////////////////// Prediction ends here //////////////////////
}

inline std::vector<float> SVMRegressor::predict(std::vector<std::vector<float>> &x_pred) {
    if (x_pred[0].size() != ncols) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    size_t Nrows = x_pred.size();

    LOG_DEBUG("Number of rows in X_test", Nrows);
    LOG_DEBUG("Number of columns in X_test", x_pred[0].size());
    std::cout << "\n";

    std::vector<float> result;
    for (size_t row = 0; row < Nrows; row++) {
        result.push_back(predict(x_pred[row]));
    }

    return result;

    ////////////////////// Prediction ends here //////////////////////
}

inline void SVMRegressor::print_predict(std::vector<std::vector<float>> &x_pred, std::vector<float> &y_pred) {
    std::vector<float> y_test = predict(x_pred);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_pred[i] << "\n";
    }
    std::cout << "\n";
}

inline float SVMRegressor::R2_score(std::vector<float>& actual, std::vector<float>& predicted) {
    float mean_y = 0;
#pragma omp parallel for simd reduction(+:mean_y)
    for (int i=0; i<actual.size(); i++) {
        mean_y += actual[i];
    }

    float ss_res = 0.0, ss_tot = 0.0;
#pragma omp parallel for reduction(+:ss_res) reduction(+:ss_tot)
    for (size_t i = 0; i < actual.size(); ++i) {
        ss_res += (float) std::pow(actual[i] - predicted[i], 2);
        ss_tot += (float) std::pow(actual[i] - mean_y, 2);
    }

    return 1.0 - ss_res / ss_tot;
}


inline void SVMRegressor::analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) {
    if (x_test[0].size() != y_test.size()) {
        LOG_ERROR("Test size does not match.");
    }

    float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;
    std::vector<float> y_pred = predict(x_test);
#pragma omp parallel for reduction(+:mse) reduction(+:mae) reduction(+:mape)
    for (size_t i = 0; i < y_pred.size(); i++) {
        float error = y_test[i] - y_pred[i];
        mse += error * error;
        mae += std::abs(error);

        if (y_test[i] != 0) {
            mape += std::abs(error / y_test[i]);
        }
    }

    mse = mse / (float) x_test.size();
    rmse = std::sqrt(mse);
    mae = mae / (float) x_test.size();
    mape = mape / (float) x_test.size() * 100;

    LOG_INFO("Evaluation metrics: ");
    std::cout << "R2 score: " << R2_score(y_test, y_pred) << "\n";
    std::cout << "MSE: " << mse << "\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "MAE: " << mae << "\n";
    std::cout << "MAPE: " << mape << "\n";
}

#endif //SVMREGRESSORFLOW_HPP
