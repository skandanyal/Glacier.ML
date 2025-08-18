//
// Created by skandan-c-y on 7/22/25.
//

#include "KNNRegressorFlow.h"
#include <chrono>
#include <iostream>
#include <numeric>

#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);		    // errors and exits
#define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"							// deeper info to be used during development
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";												// high level info while users are using it
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " seconds. \n";	// time taken

#if DEBUG_MODE
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"						// deeper info to be used during development
#else
    #define LOG_DEBUG(x, x_val)
#endif

// constructor
KNNRegressor::KNNRegressor(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i) {
    std::cout << R"(
           ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
          ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
          ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
          ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
          ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
           ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
      )" << "\n";

    // check for empty dataset
    if (X_i.empty() || Y_i.empty()) { // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    // check for inconsistency in the dataset
    for (auto &row : X_i) // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
      }

    // check for infinite values in the dataset
    for (size_t i = 0; i < X_i.size(); i++) {
        for (size_t j = 0; j < X_i[i].size(); j++) {
            if (!std::isfinite(X_i[i][j]))
                std::cout << "[BAD VALUE] X_i[" << i << "][" << j << "] = " << X_i[i][j]
                    << "\n";
        }
        if (!std::isfinite(Y_i[i]))
            std::cout << "[BAD VALUE] Y_i[" << i << "] = " << Y_i[i] << "\n";
    }

    size_t nrows = X_i.size();
    size_t ncols = X_i[0].size();

    X.resize(nrows, std::vector<float>(ncols, 0.0));
    Y.resize(nrows, 0.0);

    // populating Eigen X matrix with the data
    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X[row][col] = X_i[row][col];
    LOG_DEBUG("Number of rows in x_train", X.size());
    LOG_DEBUG("Number of cols in x_train", X[0].size());
    std::cout << "\n";

    // for normalizing X
    mean.resize(ncols, 0.0f);
    std_dev.resize(ncols, 0.0f);

    // calculating mean and standard deviation for normalization
    for (size_t colm = 0; colm < ncols; colm++) {
        float col_sum = 0.0f;
        for (size_t row = 0; row < nrows; row++) {
            col_sum += X[row][colm];
        } mean[colm] = col_sum / (float) nrows;

        float col_sq_diff = 0.0f;
        for (size_t row = 0; row < nrows; row++) {
            col_sq_diff += std::pow(X[row][colm] - mean[colm], 2);
        }
        std_dev[colm] = std::sqrt(col_sq_diff / (float) nrows);

        if (std_dev[colm] == 0.0f)
            std_dev[colm] = 1e-8;
    }

    // normalizing the dataset
    for (size_t row = 0; row < nrows; row++) {
        for (size_t colm = 0; colm < ncols; colm++) {
            X[row][colm] = (X[row][colm] - mean[colm]) / std_dev[colm];
        }
    }

    // populating the Eigen Y matrix with the data
    size_t Y_i_size = Y_i.size();
    for (size_t i = 0; i < Y_i_size; i++)
      Y[i] = Y_i[i];

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

    if (x_pred.size() != X[0].size()) {
      LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    // normalize the vector first
    for (size_t col = 0; col < x_pred.size(); col++) {
        x_pred[col] = (x_pred[col] - mean[col]) / std_dev[col];
    }

    // initialize the min_heap
    MinHeap heap;

    switch (distance_metric) {
        case 1:
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::abs(X[row][col] - x_pred[col]);
                }
                LOG_DEBUG("Manhattan distance", std::to_string(distance));
                std::pair<double, float> dist = {distance, Y[row]};

                // push into a min heap
                heap.push(dist);
            }
            break;

            ///////////////////////////////////////////////////////////

        case 2:
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::pow(X[row][col] - x_pred[col], 2);
                }
                LOG_DEBUG("Euclidean distance", std::to_string(distance));
                std::pair<double, float> to_push = {std::sqrt(distance), Y[row]};

                // push into a min heap
                heap.push(to_push);
            }
            break;

            //////////////////////////////////////////////////////////

        case 3:
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::pow(std::abs(X[row][col] - x_pred[col]), p);
                }
                LOG_DEBUG("Minkowski distance", std::to_string(distance));
                std::pair<double, float> to_push = {std::pow(distance, 1.0f/p), Y[row]};

                // push into a min heap
                heap.push(to_push);
            }
            break;

            ////////////////////////////////////////////////////////

        default:
            LOG_ERROR("Unknown distance metric.");
    }

    // averaging the top k values to return the prediction value
    float answer = 0.0f;
    for (int row = 0; row < k; row++) {
        float ans = heap.top();
        answer += ans;
        heap.pop();
    }

    return answer / (float) k;

    ////////////////////// Prediction ends here //////////////////////
}

std::vector<float> KNNRegressor::predict(std::vector<std::vector<float>> &x_pred) {
    LOG_INFO("Block prediction initiated...");

    if (x_pred[0].size() != (int)X[0].size()) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    size_t nrows = x_pred.size();
    size_t ncols = x_pred[0].size();

    LOG_DEBUG("Number of rows in X_test", nrows);
    LOG_DEBUG("Number of columns in X_test", ncols);
    std::cout << "\n";

    std::vector<float> result;
    for (size_t row = 0; row < nrows; row++) {
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
    if (actual.size() != predicted.size() || actual.empty()) return 0.0;

    float mean_y = std::accumulate(actual.begin(), actual.end(), 0.0) / (float) actual.size();

    float ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        ss_res += std::pow(actual[i] - predicted[i], 2);
        ss_tot += std::pow(actual[i] - mean_y, 2);
    }

    return 1.0 - (ss_res / ss_tot);
}


void KNNRegressor::analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) {
    if (x_test[0].size() != y_test.size()) {
        LOG_ERROR("Test size does not match.");
    }

    float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;

    std::vector<float> y_pred = predict(x_test);
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
