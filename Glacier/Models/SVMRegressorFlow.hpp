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

namespace Glacier::Models {
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
        // check for the number of threads, cut it by two, and use those many
        int threads = omp_get_max_threads() / 2;
        omp_set_num_threads(threads);
        LOG_DEBUG("Number of threads", threads);

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

        // Initializing Eigen X and Y matrix
        // X contains features + bias in the last colm
        X = Eigen::MatrixXf(nrows, ncols + 1);                                                                          // use this method to resize an eigen matrix;
        Y = Eigen::VectorXf(nrows);                                                                                         // use (nrows, 1) to ensure column vector

        // for now, the last column in the default X container is the bias term, initialized to 0
        X.col(ncols) = Eigen::VectorXf::Zero((Eigen::Index) nrows);

        // populating the Eigen X matrix with the data
        for (size_t row = 0; row < nrows; row++)
            for (size_t col = 0; col < ncols; col++)
                X(row, col) = X_i[row][col];
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

        // noting the dimensions of the weight vector, whose last element represents the bias term
        const int dim_w = (int)ncols;
        const int dim_b = (int)ncols+1;

        // initializing the w vector
        w - Eigen::VectorXf::Zero((Eigen::Index)dim_b);

        // random engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // initialize the indices vector, shuffle them later
        indices.resize(nrows);
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; epoch++) {
            // shuffle the dataset in each epoch
            std::ranges::shuffle(indices, gen);

            // step counter
            int step = 0;

            // for every row in the dataset
            for (int row = 0; row < nrows; row++) {
                step++;

                // current row is stored in the shuffled indices vector
                const int current_idx = indices[row];

                // eta = 1 / (lambda * t)
                const float eta = 1.0f / (lambda * step);

                // P_i - prediction = w.T * X
                const float P_i = cblas_sdot(dim_b, w.data(), 1, X.row(current_idx).data(), 1);

                // Residual P_i - Y_i
                const float residual = P_i - Y[current_idx];;

                // Regularization step - w = (1 - eta * lambda) * w
                cblas_sscal(dim_w, 1.0f - eta * lambda, w.data(), 1);

                // loss gradient update as the predicted value lies outside the epsilon tubed
                float alpha = 0.0f;
                if (residual > epsilon) {
                    // w = w - eta*X
                    alpha = -eta;
                } else if (residual < -epsilon) {
                    // w = w + eta*X
                    alpha = eta;
                }
                // Updating w
                cblas_saxpy(dim_w, alpha, X.row(current_idx).data(), 1, w.data(), 1);

                // update the bias
                w(dim_b) += alpha;
            }

            // predicted value lies inside the epsilon tube, hence no need to update the w further
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

        if (x_pred.size() != ncols) {
            LOG_ERROR("Train and test dataset have different number of features.");
        }

        ////////////////////// Prediction begins here //////////////////////

        // loading the input vector into an eigen matrix
        Eigen::VectorXf x_p(ncols + 1);

        // transfer the normalized data into the eigen vector
        for (int i=0; i< ncols; i++) {
            x_p(i) = (x_pred[i] - mean[i]) / std_dev[i];
        }
        x_p(ncols) = 1.0f;

        // predict the output
        float answer = w.transpose().dot(x_p);

        ////////////////////// Prediction ends here //////////////////////

        return answer;
    }

    inline std::vector<float> SVMRegressor::predict(std::vector<std::vector<float>> &x_pred) {
        if (x_pred[0].size() != ncols) {
            LOG_ERROR("Train and test dataset have different number of features.");
        }

        ////////////////////// Prediction begins here //////////////////////

        size_t Nrows = x_pred.size();
        size_t Ncols = x_pred[0].size();

        LOG_DEBUG("Number of rows in X_test", Nrows);
        LOG_DEBUG("Number of columns in X_test", Ncols);
        std::cout << "\n";

        Eigen::MatrixXf X_pred(Nrows, Ncols + 1);
        std::vector<float> result;

        // normalize X
        for (int row = 0; row < Nrows; row++) {
            for (int colm = 0; colm < Ncols; colm++) {
                X_pred(row, colm) = (x_pred[row][colm] - mean[colm]) / std_dev[colm];
            }
        }
        // bias colm
        X_pred.col(Ncols) = Eigen::VectorXf::Ones((Eigen::Index) nrows);

        // calculating the prediction
        Eigen::VectorXf answer = X_pred * w;

        // Mapping the vector back to std::vector<float>
        Eigen::Map<Eigen::VectorXf> (result.data(), Nrows) = answer;

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
}

#endif //SVMREGRESSORFLOW_HPP
