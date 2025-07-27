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

// errors and exits
#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);

// deeper info to be used during development
#define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"

// high level info while users are using it
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";

// time taken
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " seconds. \n";

// deeper info to be used during development
#if DEBUG_MODE
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"
#else
    #define LOG_DEBUG(x, x_val)
#endif


// constructor
KNNClassifier::KNNClassifier(std::vector<std::vector<float> > &X_i, std::vector<std::string> &Y_i) {
    std::cout << R"(
         ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
        ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
        ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
        ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
        ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
         ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
    )" << std::endl;

    if (X_i.empty() || Y_i.empty()) {                                                                                   // Check if the inputs are valid or not
        LOG_ERROR("Input data cannot be empty.");
    }

    for (auto &row : X_i)                                                                                  // Check if all the rows are of the same size
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
    }

    size_t nrows = X_i.size();
    size_t ncols = X_i[0].size();

    X.resize(nrows, std::vector<float>(ncols, 0.0));
    Y.resize(nrows);                                                                                      // use (nrows, 1) to ensure column vector

    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X[row][col] = X_i[row][col];
    LOG_DEBUG("Number of rows in x_train", X.size());
    LOG_DEBUG("Number of cols in x_train", X[0].size());
    std::cout << "\n";

    // normalize X
    mean.resize(ncols, 0.0f);
    std_dev.resize(ncols, 0.0f);;

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

    // initialize Y
    std::map<std::string, bool> seen;
    for (std::string target : Y_i)
        seen[target] = true;
    for (auto &[key, _] : seen)
        labels.push_back(key);

    if (labels.size() < 2) {
        LOG_ERROR("Less than two classification classes detected. Classification requires the dataset to have at least two target classes.");
    }

    // multiclass classification by default
    std::ranges::sort(labels);

    // for (size_t i = 0; i < Y_i.size(); i++)
    //     for (int j = 0; j < labels.size(); j++)
    //         if (Y_i[i] == labels[j])
    //             Y[i] = j;

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

std::string KNNClassifier::predict(std::vector<float> &x_pred) {
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
        case 1: // Manhattan distance
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::abs(X[row][col] - x_pred[col]);
                }
                LOG_DEBUG("Manhattan distance", std::to_string(distance));
                std::pair<double, int> dist = {distance, Y[row]};

                // push into a min heap
                heap.push(dist);
            }
            break;

            ///////////////////////////////////////////////////////////

        case 2: // Euclidean distance
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::pow(X[row][col] - x_pred[col], 2);
                }
                LOG_DEBUG("Euclidean distance", std::to_string(distance));
                std::pair<double, int> to_push = {std::sqrt(distance), Y[row]};

                // push into a min heap
                heap.push(to_push);
            }
            break;

            //////////////////////////////////////////////////////////

        case 3: // Minkowski distance
            for (size_t row = 0; row < X.size(); row++) {
                double distance = 0;

                for (size_t col = 0; col < X[row].size(); col++) {
                    distance += std::pow(std::abs(X[row][col] - x_pred[col]), p);
                }
                LOG_DEBUG("Minkowski distance", std::to_string(distance));
                std::pair<double, int> to_push = {std::pow(distance, 1.0f/p), Y[row]};

                // push into a min heap
                heap.push(to_push);
            }
            break;

            ////////////////////////////////////////////////////////

        default:
            LOG_ERROR("Unknown distance metric.");
    }

    // voting to find the most frequently occurring target among the nearest k neighbours
    std::unordered_map<std::string, size_t> voting;
    for (int i = 0; i < labels.size(); i++)
        voting[labels[i]] = 0;

    for (int row = 0; row < k; row++) {
        voting[labels[heap.top()]]++;
        heap.pop();
    }

    std::string answer;
    size_t highest = 0;
    for (auto &[label, vote] : voting) {
        if (vote > highest) {
            highest = vote;
            answer = label;
        }
    }

    ////////////////////// Prediction ends here //////////////////////

    return answer;
}

std::vector<std::string> KNNClassifier::predict(std::vector<std::vector<float>> &x_pred) {
    LOG_INFO("Block prediction initiated...");

    if (x_pred[0].size() != (int)X[0].size()) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    size_t nrows = x_pred.size();
    size_t ncols = x_pred[0].size();

    LOG_DEBUG("Number of rows in X_test", nrows);
    LOG_DEBUG("Number of columns in X_test", ncols);
    std::cout << "\n";

    std::vector<std::string> result;
    for (size_t row = 0; row < nrows; row++) {
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