//
// Created by skandan-c-y on 7/22/25.
//

#include <algorithm>

#include "KNNRegressorFlow.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include "omp.h"
#include "logs.h"



// constructor
KNNRegressor::KNNRegressor(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i) {
    // std::cout << R"(
    //        ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
    //       ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
    //       ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
    //       ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
    //       ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
    //        ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
    //   )" << "\n";

    // check for the number of threads, cut it by two, and use those many
    int threads = omp_get_max_threads() / 2;
    omp_set_num_threads(threads);

    // check for empty dataset
    if (X_i.empty() || Y_i.empty()) { // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    // check for inconsistency in the dataset
    for (auto &row : X_i) // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
      }

    nrows = X_i.size();
    ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in x_train", nrows);
    LOG_DEBUG("Number of cols in x_train", ncols);
    std::cout << "\n";

    X.resize(nrows * ncols, 0.0);
    Y.resize(nrows, 0.0);

    // Populating the Y matrix with the data
    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X[row * ncols + col] = X_i[row][col];

    // populating the Y matrix with the data
    size_t Y_i_size = Y_i.size();
    for (size_t i = 0; i < Y_i_size; i++)
        Y[i] = Y_i[i];

    // Regression specific block ahead -
    // check for infinite values - parallelized, SIMD
    int bad_values_X = 0, bad_values_Y = 0;
#pragma omp parallel for default(none) shared(bad_values_X, bad_values_Y, nrows, ncols, X, Y) simd reduction(+:bad_values)
    for (int i = 0; i < nrows * ncols; i++) {
        if (!std::isfinite(X[i]))
            bad_values_X++;
        if (!std::isfinite(Y[i]))
            bad_values_Y++;
    } if (bad_values_X > 0) {
        LOG_ERROR("Infinite values exist in the X dataset.")
    } if (bad_values_Y > 0) {
        LOG_ERROR("Infinite values exist in the X dataset.")
    }

    // for normalizing X
    mean.resize(ncols, 0.0f);
    std_dev.resize(ncols, 0.0f);

    // calculating mean and standard deviation for normalization
#pragma omp parallel for default(none) shared(mean, std_dev, nrows, ncols, X)
    for (size_t colm = 0; colm < ncols; colm++) {
        float col_sum = 0.0f;
#pragma omp simd reduction(+:col_sum)
        for (size_t row = 0; row < nrows; row++) {
            col_sum += X[row * ncols + colm];
        } mean[colm] = col_sum / (float) nrows;

        float col_sq_diff = 0.0f;
#pragma omp simd reduction(+:col_sq_diff)
        for (size_t row = 0; row < nrows; row++) {
            col_sq_diff += (X[row * ncols + colm] - mean[colm]) * (X[row * ncols + colm] - mean[colm]);
        } std_dev[colm] = std::sqrt(col_sq_diff / (float) nrows);

        if (std_dev[colm] == 0.0f)
            std_dev[colm] = 1e-8;
    }

    // normalizing the dataset
#pragma omp parallel for default(none) shared(mean, std_dev, nrows, ncols, X)
    for (size_t row = 0; row < nrows; row++) {
        for (size_t colm = 0; colm < ncols; colm++) {
            X[row * ncols + colm] = (X[row * ncols + colm] - mean[colm]) / std_dev[colm];
        }
    }

    LOG_DEBUG("Number of rows in y_train", Y.size());
    LOG_DEBUG("Since y_train is a std::vector, number of columns in y_train", 1);
    std::cout << "\n";
}

void KNNRegressor::train(int k_i, const std::string &distance_metric_i, int p_i) {
      auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    k = k_i;

    if (distance_metric_i == "Manhattan")
        distance_metric = 1;
    else if (distance_metric_i == "Euclidean")
        distance_metric = 2;
    else if (distance_metric_i == "Minkowski") {
        distance_metric = 3;
        p = p_i;
    } else {
        LOG_ERROR("Unknown distance metric.");
    }

    ////////////////////// Training ends here ////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

float KNNRegressor::predict(std::vector<float> &x_pred) {
    LOG_INFO("Singular prediction initiated...");

    if (x_pred.size() != nrows) {
      LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    // normalizing the prediction vector
#pragma omp parallel for default(none) \
shared(x_pred, mean, std_dev) \
private(col)
    for (size_t col = 0; col < x_pred.size(); col++) {
        x_pred[col] = (x_pred[col] - mean[col]) / std_dev[col];
    }

    std::vector<std::pair<double, float>> least_distance(nrows);

#pragma omp parallel for default(none) \
private(row, col, distance) \
shared(distance_metric, nrows, ncols, X, Y, x_pred, least_distance, p)
    for (size_t row = 0; row < nrows; row++) {
        double distance = 0.0;

        switch (distance_metric) {
            case 1: // Manhattan distance
#pragma omp simd reduction(+:distance)
                for (size_t col = 0; col < ncols; col++) {
                    distance += std::abs(X[row * ncols + col] - x_pred[col]);
                }
                break;
            case 2: // Euclidean distance
#pragma omp simd reduction(+:distance)
                for (size_t col = 0; col < ncols; col++) {
                    float diff = X[row * ncols + col] - x_pred[col];
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
                break;
            case 3: // Minkowski distance
#pragma omp simd reduction(+:distance)
                for (size_t col = 0; col < ncols; col++) {
                    distance += static_cast<float>(std::pow(std::abs(X[row * ncols + col] - x_pred[col]), p));
                }
                distance = std::pow(distance, 1.0/p);
                break;
            default:
                LOG_ERROR("Unknown distance metric.");
        }
        least_distance[row] = {distance, Y[row]};
    }

    // finding the k-th element
    std::ranges::nth_element(least_distance, least_distance.begin() + k);

    // averaging
    float answer = 0.0f;
#pragma omp parallel simd reduction(+:answer) default(none)\
    shared(answer)
    for (size_t row = 0; row < k; row++) {
        answer += least_distance[row].second;
    }

    return answer / (float) k;

    ////////////////////// Prediction ends here //////////////////////
}

std::vector<float> KNNRegressor::predict(std::vector<std::vector<float>> &x_pred) {
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

void KNNRegressor::print_predict(std::vector<std::vector<float>> &x_pred, std::vector<float> &y_pred) {
    std::vector<float> y_test = predict(x_pred);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_pred[i] << "\n";
    }
    std::cout << "\n";
}

float KNNRegressor::R2_score(std::vector<float>& actual, std::vector<float>& predicted) {
    float mean_y = 0;
#pragma omp parallel for simd reduction(+=mean_y)
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


void KNNRegressor::analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) {
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
