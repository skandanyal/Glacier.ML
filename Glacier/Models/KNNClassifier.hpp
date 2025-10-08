//
// Created by skandan-c-y on 7/14/25.
//

#ifndef KNNCLASSIFIER_HPP
#define KNNCLASSIFIER_HPP

#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include "omp.h"
#include "../Utils/logs.hpp"

class KNNClassifier {
private:
    std::vector<float> X;      // (n x p)
    std::vector<int> Y;                     // (n x 1)
    std::vector<std::string> labels;        // (p x 1)
    // std::pair<double, float> Dist;
    std::vector<float> mean;                // (p x 1)
    std::vector<float> std_dev;             // (p x 1)
    int distance_metric{};
    int k{};
    int p{};
    size_t nrows{}, ncols{};
    // the {} braces are for the constructor to initialize these variables outside the constructor

public:
    // constructor
    KNNClassifier(std::vector<std::vector<float>> &X_i, std::vector<std::string> &Y_i);
    void train(int k_i, std::string& distance_metric_i, int p_i=2);
    std::string predict(std::vector<float> &x_pred);
    std::vector<std::string> predict(std::vector<std::vector<float>>& x_pred);
    void print_predict(std::vector<std::vector<float>>& x_pred, std::vector<std::string> &y_pred);
    void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
};

inline KNNClassifier::KNNClassifier(std::vector<std::vector<float>> &X_i, std::vector<std::string> &Y_i) {
    // std::cout << R"(
    //      ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
    //     ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
    //     ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
    //     ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
    //     ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
    //hb      ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
    // )" << std::endl;

    // check for the number of threads, cut it by two, and use those many
    int threads = omp_get_max_threads() / 2;
    omp_set_num_threads(threads);
    LOG_DEBUG("Number of threads", threads);

    // check if input matrices are empty or not
    if (X_i.empty() || Y_i.empty()) {                                                                                   // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    // check if every row has a target
    if (X_i.size() != Y_i.size()) {
        LOG_ERROR("Input and output data must have the same size.");
    }

    // check for empty dataset
    for (auto &row : X_i)                                                                                  // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
        }
    LOG_UPDATE("Dataset alright");

    nrows = X_i.size();
    ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in x_train", X_i.size());
    LOG_DEBUG("Number of cols in x_train", X_i[0].size());
    std::cout << "\n";

    X.resize(nrows * ncols, 0.0);
    Y.resize(nrows);                                                                                      // use (nrows, 1) to ensure column vector

    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X[row * ncols + col] = X_i[row][col];
    LOG_UPDATE("Converted to 1D vector");

    // Classification specific block ahead -
    // check for infinite values - parallelized, SIMD
    int bad_values_X = 0;
    #pragma omp parallel for default(none) \
    shared(bad_values_X, nrows, ncols, X)
    for (int i = 0; i < nrows * ncols; i++) {
        if (!std::isfinite(X[i]))
            bad_values_X++;
    } if (bad_values_X > 0) {
        LOG_ERROR("Infinite values exist in the X dataset.");
    }

    // normalize X
    mean.resize(ncols, 0.0f);
    std_dev.resize(ncols, 0.0f);

    // calculating mean and standard deviation for normalization
    #pragma omp parallel for \
    default(none) \
    shared(mean, std_dev, nrows, ncols, X)
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
    LOG_UPDATE("Mean and Std_dev vectors are formed.");

    // normalizing the dataset
    #pragma omp parallel for default(none) \
    shared(mean, std_dev, nrows, ncols, X)
    for (size_t row = 0; row < nrows; row++) {
        for (size_t colm = 0; colm < ncols; colm++) {
            X[row * ncols + colm] = (X[row * ncols + colm] - mean[colm]) / std_dev[colm];
        }
    }
    LOG_UPDATE("Dataset normalized.");

    // initialize Y
    std::map<std::string, bool> seen;
    for (const std::string& target : Y_i)
        seen[target] = true;
    for (auto &[key, _] : seen)
        labels.push_back(key);

    if (labels.size() < 2) {
        LOG_ERROR("Less than two classification classes detected. Classification requires the dataset to have at least two target classes.");
    }

    // multiclass classification by default
    std::ranges::sort(labels);

    std::map<std::string, int> indexing;
    for (int i = 0; i < labels.size(); i++)
        indexing[labels[i]] = i;
    for (size_t j = 0; j < nrows; j++) {
        Y[j] = indexing[Y_i[j]];
    }
    LOG_UPDATE("Labels vector formed and indexed.");

    LOG_DEBUG("Size of labels", labels.size());
    LOG_DEBUG("Number of rows in y_train", Y.size());
    std::cout << "\n";
}

inline void KNNClassifier::train(int k_i, std::string& distance_metric_i, int p_i) {
    auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    if (k_i % 2 == 0) k = k_i - 1; // always store an odd value of k
    else k = k_i;

    if (distance_metric_i == "manhattan")
        distance_metric = 1;
    else if (distance_metric_i == "euclidean")
        distance_metric = 2;
    else if (distance_metric_i == "minkowski") {
        distance_metric = 3;
        p = p_i;
    } else {
        LOG_ERROR("Unknown distance metric.");
    }
    LOG_DEBUG("Distance metric", distance_metric_i);
    LOG_DEBUG("K", k);
    LOG_UPDATE("Hyperparameters set. ");

    ////////////////////// Training ends here ////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);

    LOG_TIME("Training", duration.count() / 1000);
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

inline std::string KNNClassifier::predict(std::vector<float> &x_pred) {
    if (x_pred.size() != ncols) {
    LOG_ERROR("Train and test dataset have different number of features.");
}

////////////////////// Prediction begins here //////////////////////

// normalizing the prediction vector
#pragma omp parallel for default(none) \
shared(x_pred, mean, std_dev)
for (size_t col = 0; col < x_pred.size(); col++) {
    x_pred[col] = (x_pred[col] - mean[col]) / std_dev[col];
}

std::vector<std::pair<double, int>> least_distance(nrows);

if (distance_metric > 3 || distance_metric < 1) {
    LOG_ERROR("Distance metric is undefined.");
}

#pragma omp parallel for default(none) \
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
            continue;
    }
    least_distance[row] = {distance, Y[row]};
}

// finding the k-th element
std::ranges::nth_element(least_distance, least_distance.begin() + k);

// voting
std::unordered_map<std::string, int> voting;
for (size_t i=0; i<k; i++) {
    voting[labels[least_distance[i].second]]++;
}
std::string answer;
int highest_vote = 0;
for (const auto &thing : voting) {
    if (thing.second > highest_vote) {
        highest_vote = thing.second;
        answer = thing.first;
    }
}

////////////////////// Prediction ends here //////////////////////

return answer;
}

inline std::vector<std::string> KNNClassifier::predict(std::vector<std::vector<float>>& x_pred) {
    if (x_pred[0].size() != ncols) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    size_t Nrows = x_pred.size();
    size_t Ncols = x_pred[0].size();

    LOG_DEBUG("Number of rows in x_pred", Nrows);
    LOG_DEBUG("Number of columns in x_pred", Ncols);
    std::cout << "\n";

    /*
     * HOW TO PREDICT USING KNN:
     * 1. take a prediction vector and normalize it
     * 2. subtract that distance from the training dataset
     * 3. conduct voting / averaging
     * 4. add the distance back to thr training dataset
     * 5. take the next vector and repeat from step 2
     */

    std::vector<std::string> answer(Nrows);
    for (size_t row = 0; row < Nrows; row++) {
        answer[row] = predict(x_pred[row]);
    }

    return answer;
}

inline void KNNClassifier::print_predict(std::vector<std::vector<float>>& x_pred, std::vector<std::string> &y_pred) {
    std::vector<std::string> y_test = predict(x_pred);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_pred[i] << "\n";
    }
    std::cout << "\n";
}
// mimic predict_proba from sklearn
void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);



// std::vector<std::string> KNNClassifier::predict(std::vector<std::vector<float>> &x_pred) {


    // ==============================================================================

//     // convert into a 1d matrix, normalize in the same block, calculate the nearest distance
//     std::vector<float> x_here(Nrows * Ncols ,0.0f);                                    // x_pred in 1d
//     std::vector<std::pair<double, int>> least_distance(nrows * Nrows, {0.0f, 0});   // least distances
//
//     for (int i = 0; i < Nrows; i++) {
//         for (int j = 0; j < Ncols; j++) {
//             x_here[i * Ncols + j] = X[i * Ncols + j] - (x_pred[i][j] - mean[j]) / std_dev[j];
//         }
//     }
//
//     if (distance_metric > 3 || distance_metric < 1) {
//         LOG_ERROR("Distance metric is undefined.");
//     }
//
// #pragma omp parallel for default(none) \
//         shared(distance_metric, nrows, ncols, X, Y, x_pred, least_distance, p)
//     for (size_t row = 0; row < nrows; row++) {
//         double distance = 0.0;
//
//         switch (distance_metric) {
//             case 1: // Manhattan distance
// #pragma omp simd reduction(+:distance)
//                 for (size_t col = 0; col < ncols; col++) {
//                     distance += std::abs(X[row * ncols + col] - x_pred[col]);
//                 }
//                 break;
//             case 2: // Euclidean distance
// #pragma omp simd reduction(+:distance)
//                 for (size_t col = 0; col < ncols; col++) {
//                     float diff = X[row * ncols + col] - x_pred[col];
//                     distance += diff * diff;
//                 }
//                 distance = std::sqrt(distance);
//                 break;
//             case 3: // Minkowski distance
// #pragma omp simd reduction(+:distance)
//                 for (size_t col = 0; col < ncols; col++) {
//                     distance += static_cast<float>(std::pow(std::abs(X[row * ncols + col] - x_pred[col]), p));
//                 }
//                 distance = std::pow(distance, 1.0/p);
//                 break;
//             default:
//                 continue;
//         }
//         least_distance[row] = {distance, Y[row]};
//     }
//
//     // finding the k-th element
//     std::ranges::nth_element(least_distance, least_distance.begin() + k);
//
//     // voting
//     std::unordered_map<std::string, int> voting;
//     for (size_t i=0; i<k; i++) {
//         voting[labels[least_distance[i].second]]++;
//     }
//     std::string answer;
//     int highest_vote = 0;
//     for (const auto &thing : voting) {
//         if (thing.second > highest_vote) {
//             highest_vote = thing.second;
//             answer = thing.first;
//         }
//     }
//
//     return result;
// }

#endif //KNNCLASSIFIER_HPP
