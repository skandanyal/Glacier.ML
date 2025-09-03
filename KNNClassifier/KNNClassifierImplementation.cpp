//
// Created by skandan-c-y on 7/14/25.
//

#include "KNNClassifierFlow.h"
#include "min_heap.hpp"
#include <iostream>
#include <cmath>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include "omp.h"
#include "logs.h"

// constructor
KNNClassifier::KNNClassifier(std::vector<std::vector<float> > &X_i, std::vector<std::string> &Y_i) {
    // std::cout << R"(
    //      ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
    //     ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
    //     ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
    //     ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
    //     ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
    //      ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
    // )" << std::endl;

    // check for the number of threads, cut it by two, and use those many
    int threads = omp_get_max_threads() / 2;
    omp_set_num_threads(threads);

    // check if input matrices are empty or not
    if (X_i.empty() || Y_i.empty()) {                                                                                   // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    // check for empty dataset
    for (auto &row : X_i)                                                                                  // Check if all the rows are of the same size
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
        }

    nrows = X_i.size();
    ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in x_train", X.size());
    LOG_DEBUG("Number of cols in x_train", X[0].size());
    std::cout << "\n";

    X.resize(nrows * ncols, 0.0);
    Y.resize(nrows);                                                                                      // use (nrows, 1) to ensure column vector

    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X[row * ncols + col] = X_i[row][col];

    // Classification specific block ahead -
    // check for infinite values - parallelized, SIMD
    int bad_values_X = 0;
#pragma omp parallel for default(none) \
    shared(bad_values_X, nrows, ncols, X)
    for (int i = 0; i < nrows * ncols; i++) {
        if (!std::isfinite(X[i]))
            bad_values_X++;
    } if (bad_values_X > 0) {
        LOG_ERROR("Infinite values exist in the X dataset.")
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

    // normalizing the dataset
#pragma omp parallel for default(none) \
    shared(mean, std_dev, nrows, ncols, X)
    for (size_t row = 0; row < nrows; row++) {
        for (size_t colm = 0; colm < ncols; colm++) {
            X[row * ncols + colm] = (X[row * ncols + colm] - mean[colm]) / std_dev[colm];
        }
    }

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

    LOG_DEBUG("Size of labels", labels.size());
    LOG_DEBUG("Number of rows in y_train", Y.size());
    std::cout << "\n";
}

void KNNClassifier::train(int k_i, std::string &distance_metric_i, int p_i) {
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

    ////////////////////// Training ends here ////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start);

    LOG_TIME("Training", duration.count() / 1000);
    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

std::string KNNClassifier::predict(std::vector<float> &x_pred) {
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

std::vector<std::string> KNNClassifier::predict(std::vector<std::vector<float>> &x_pred) {
    if (x_pred[0].size() != ncols) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    size_t Nrows = x_pred.size();

    LOG_DEBUG("Number of rows in X_test", Nrows);
    LOG_DEBUG("Number of columns in X_test", Ncols);
    std::cout << "\n";

    std::vector<std::string> result;
    for (size_t row = 0; row < Nrows; row++) {
        result.push_back(predict(x_pred[row]));
    }

    return result;
}

void KNNClassifier::print_predict(std::vector<std::vector<float>> &x_pred, std::vector<std::string> &y_pred) {
    std::vector<std::string> y_test = predict(x_pred);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_pred[i] << "\n";
    }
    std::cout << "\n";
}

void KNNClassifier::analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test) {
    LOG_INFO("Analysis initiated...");
    std::vector<std::string> y_pred = predict(x_test);

    float tp = 0, fn = 0, fp = 0, tn = 0;

    for (size_t i=0; i<y_pred.size(); i++) {
        if (y_test[i] == labels[0] && y_pred[i] == labels[0])   // tp
            tp++;

        else if (y_test[i] == labels[0] && y_pred[i] == labels[1])    // fn
            fn++;

        else if (y_test[i] == labels[1] && y_pred[i] == labels[0])    // fp
            fp++;

        else if (y_test[i] == labels[1] && y_pred[i] == labels[1])    // tn
            tn++;
    }

    LOG_INFO("Confusion matrix: ");
    std::cout << "Actually " << labels[0] << ", Predicted " << labels[0] << ": " << tp << "\n";
    std::cout << "Actually " << labels[0] << ", Predicted " << labels[1] << ": " << fn << "\n";
    std::cout << "Actually " << labels[1] << ", Predicted " << labels[0] << ": " << fp << "\n";
    std::cout << "Actually " << labels[1] << ", Predicted " << labels[1] << ": " << tn << "\n";
    std::cout << "Total number of rows: " << x_test.size() << "\n\n";

    LOG_INFO("Evaluation Metrics: (Out of 1)");

    // correct classifications / total classifications
    float accuracy = (tp + tn) / (tp + tn + fp + fn);
    std::cout << "Accuracy: " << accuracy << "\n";

    // true positives / true positives + false negatives
    float recall = tp / (tp + fn);
    std::cout << "Recall: " << recall << "\n";

    // false positives / false positives + true negatives
    float false_positive_rate = fp / (fp + tn);
    std::cout << "False positive rate: " << false_positive_rate << "\n";

    // true positives / true positives + false positives
    float precision = tp / (tp + fp);
    std::cout << "Precision: " << precision << "\n\n";
}