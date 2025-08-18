//
// Created by skandan-c-y on 7/2/25.
//

#include <iostream>
#include <chrono>
#include <map>

#include "LogisticRegressionFlow.h"

#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);		    // errors and exits
#define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"							// deeper info to be used during development
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";												// high level info while users are using it
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " microseconds. \n";					// time taken

#if DEBUG_MODE
    #define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"						// deeper info to be used during development
#else
    #define LOG_DEBUG(x, x_val)
#endif


// constructor
Logistic_Regression::Logistic_Regression(std::vector<std::vector<float>> &X_i, std::vector<std::string> &Y_i) : X(), Y(), Beta(), F_x() {
    std::cout << R"(
         ██████╗ ██╗      █████╗  ██████╗██╗███████╗██████╗    ███╗   ███╗██╗
        ██╔════╝ ██║     ██╔══██╗██╔════╝██║██╔════╝██╔══██╗   ████╗ ████║██║
        ██║  ███╗██║     ███████║██║     ██║█████╗  ██████╔╝   ██╔████╔██║██║
        ██║   ██║██║     ██╔══██║██║     ██║██╔══╝  ██╔══██╗   ██║╚██╔╝██║██║
        ╚██████╔╝███████╗██║  ██║╚██████╗██║███████╗██║  ██║██╗██║ ╚═╝ ██║███████╗
         ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
    )" << std::endl;

    // check for empty dataset
    if (X_i.empty() || Y_i.empty()) {                                                                                   // Check if the inputs are valid or not
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
        // if (!std::isfinite(Y_i[i]))
        //     std::cout << "[BAD VALUE] Y_i[" << i << "] = " << Y_i[i] << "\n";   // not applicable here in classification
    }

    Eigen::Index nrows = X_i.size();
    Eigen::Index ncols = X_i[0].size();
    LOG_DEBUG("Number of rows in X before adding 1 column", nrows);
    LOG_DEBUG("Number of columns in X before adding 1 column", ncols);
    std::cout << "\n";

    X = Eigen::MatrixXf(nrows, ncols + 1);                                                                          // use this method to resize an eigen matrix;
    Y = Eigen::VectorXf(nrows);                                                                                         // use (nrows, 1) to ensure column vector

    // initialize X
    X.col(0) = Eigen::VectorXf::Ones(nrows);

    // populating Eigen X matrix with the data
    for (Eigen::Index row = 0; row < nrows; row++)
        for (Eigen::Index col = 0; col < ncols; col++)
            X(row, col + 1) = X_i[row][col];                                                                            // X matrix, with 0th column as 1
    LOG_DEBUG("Number of rows in x_train", X.rows());
    LOG_DEBUG("Number of cols in x_train", X.cols());
    std::cout << "\n";

    // calculate mean and standard deviation for the given dataset
    mean = Eigen::VectorXf::Zero(ncols);
    std_dev = Eigen::VectorXf::Zero(ncols);

    // normalize X
    for (Eigen::Index colm = 0; colm < ncols; colm++) {
        mean(colm) = X.col(colm + 1).mean();
        std_dev(colm) = std::sqrt((X.col(colm + 1).array() - mean(colm)).square().sum() / X.rows());

        if (std_dev(colm) == 0)
            std_dev(colm) = 1e-8;

        // normalization
        X.col(colm + 1) = (X.col(colm + 1).array() - mean(colm)) / std_dev(colm);
    }

    // initialize Y
    std::map<std::string, bool> seen;
    for (std::string target : Y_i)
        seen[target] = true;
    for (auto &[key, _] : seen)
        labels.push_back(key);

    // check for the number of target classes
    if (labels.size() < 2) {
        LOG_ERROR("Less than two classification classes detected. Binary classification requires the dataset to have two target classes.");
    } else if (labels.size() > 2) {
        LOG_ERROR("More than two classification classes detected. Binary classification requires the dataset to have two target classes.");
    }

    if (labels[0] > labels[1]) std::swap(labels[0], labels[1]);

    // populate the Eigen Y matrix with the data
    for (size_t i = 0; i < Y_i.size(); i++) {
        if (Y_i[i] == labels[0]) Y(i) = 0;
        else if (Y_i[i] == labels[1]) Y(i) = 1;
    }
    LOG_DEBUG("Size of labels", labels.size());
    LOG_DEBUG("Number of rows in y_train", Y.rows());
    LOG_DEBUG("Number of cols in y_train", Y.cols());
    std::cout << "\n";

    // initialize Beta
    Beta = Eigen::VectorXf::Zero(X.cols());
}

void Logistic_Regression::train(float alpha, int iterations) {
    LOG_INFO("Training initiated...");
    auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    LOG_DEBUG("Number of rows in Beta", Beta.rows());
    LOG_DEBUG("Number of cols in Beta", Beta.cols());
    std::cout << "\n";

    // training loop
    LOG_INFO("Loss measures");
    double loss = 0, prev_loss = 0;
    for (int i = 1; i <= iterations; i++) {
        loss = 0.0f;  // reset this for every loop

        //please, this isn't chatgpt's work. it gives a far more concise code to do the same...
        // step 1 - calculate F_x and P_x
        P_x.resize(X.rows());
        F_x = X * Beta;
        for(int j=0; j<F_x.size(); j++){
            P_x(j) = std::clamp(sigmoid(F_x(j)), 1e-8f, 1.0f - 1e-8f);
        }

        // step 2 - calculate error and gradient(delta)
        Delta = X.transpose() * (P_x - Y);

        // step 3 - update the beta matrix
        Beta -= alpha * Delta;

        // logging the loss
        prev_loss = loss;
        for (Eigen::Index row=0; row < Y.size(); row++) {
            float prob_val = std::clamp(P_x(row), 1e-6f, 1.0f - 1e-6f);
            loss += -1 * (Y[row] * std::log(prob_val) + (1 - Y[row]) * std::log(1 - prob_val));
        }
        loss /= Y.size();

        if (i % 500 == 0) {
            // loss = std::clamp(loss, 0.0001, 0.9999);
            LOG_DEBUG("Loss at iteration " + std::to_string(i), loss);
        }

        if (i > 10)
            if (loss < 0.002f && prev_loss - loss < 1e-6) {
                LOG_INFO("Early stopping engaged at step: " + std::to_string(i));
                break;
            }
    }
    LOG_DEBUG("Final loss at the end ", loss);
    std::cout << "\n";

    ////////////////////// Training ends here /////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";

    /*
     * FURTHER ENHANCEMENTS:
     * 1. (done) log the loss at every iteration, or after series of iterations
     * 2. (done) reduce or maximise alpha based on the loss
     * 3. (somehow does not work as intended) add an early stopping mechanism (hard code this, no need to let the user decide the threshold)
     */
}

std::string Logistic_Regression::predict(std::vector<float> &x_pred) {
    LOG_INFO("Singular prediction initiated...");

    if (x_pred.size() + 1 != Beta.cols()) {
        LOG_ERROR(("Incompatible size of vector passed. Expected size: " + std::to_string(Beta.cols())));
    }

    ////////////////////// Prediction begins here //////////////////////

    Eigen::VectorXf x = Eigen::VectorXf::Zero(Beta.cols());
    x(0) = 1.0f;

    for (Eigen::Index i = 0; i < Beta.cols() - 1; i++)
        x(i+1) = x_pred[i];

    // normalizing the x matrix
    for (Eigen::Index i = 0; i < Beta.cols() - 1; i++)
        x(i+1) = (x(i+1) - mean(i)) / std_dev(i);

    float ans = x.dot(Beta);
    if (ans < 0.5f)
        return labels[0];
    return labels[1];

    ////////////////////// Prediction ends here //////////////////////
}

std::vector<std::string> Logistic_Regression::predict(std::vector<std::vector<float>>& X_test) {
    LOG_INFO("Block prediction initiated...");
    if (Beta.size() == 0) {
        LOG_ERROR("Train the data using train() before using predict().");
    }

    if (X_test[0].size() != (int) X.cols() - 1) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    Eigen::Index nrows = X_test.size();
    Eigen::Index ncols = X_test[0].size();
    LOG_DEBUG("Number of rows in X_test", nrows);
    LOG_DEBUG("Number of columns in X_test", ncols);
    std::cout << "\n";

    if (mean.size() != ncols) {
        LOG_ERROR("Mismatch in mean/std_dev size. Possible unnormalized feature set.");
    }

    Eigen::MatrixXf X_pred(nrows, ncols + 1);
    X_pred.col(0) = Eigen::VectorXf::Ones(nrows);

    for (Eigen::Index row = 0; row < nrows; row++)
        for (Eigen::Index col = 0; col < ncols; col++)
            X_pred(row, col + 1) = X_test[row][col];                                                                    // X matrix, with 0th column as 1

    // normalizing the X_pred matrix
    for (int colm = 0; colm < ncols; colm++)
        X_pred.col(colm + 1) = (X_pred.col(colm + 1).array() - mean(colm)) / std_dev(colm);

    Eigen::VectorXf F_x_pred = X_pred * Beta;
    P_x_pred.resize(nrows);
    for(int i=0; i < nrows; i++){
        P_x_pred(i) = std::clamp(sigmoid(F_x_pred(i)), 1e-8f, 1.0f - 1e-8f);
    }

    std::vector<std::string> result(nrows);
    for (Eigen::Index i = 0; i < nrows; i++) {
        if (P_x_pred(i) < 0.5f) result[i] = labels[0];
        else result[i] = labels[1];
    }

    return result;

    ////////////////////// Prediction ends here //////////////////////
}

void Logistic_Regression::print_predict(std::vector<std::vector<float> > &x_test, std::vector<std::string> &y_val) {
    std::vector<std::string> y_test = predict(x_test);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_val[i] << "\n";
    }
    std::cout << "\n";
}

void Logistic_Regression::analyze(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test) {
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
    float accuracy = (tp + tn) / (tp + tn + fp + fn);
    std::cout << "Accuracy: " << accuracy << "\n";                                                                      // correct classifications / total classifications

    float recall = tp / (tp + fn);
    std::cout << "Recall: " << recall << "\n";                                                                          // true positives / true positives + false negatives

    float false_positive_rate = fp / (fp + tn);
    std::cout << "False positive rate: " << false_positive_rate << "\n";                                                // false positives / false positives + true negatives

    float precision = tp / (tp + fp);
    std::cout << "Precision: " << precision << "\n\n";                                                                  // true positives / true positives + false positives
}

void Logistic_Regression::print_Beta_values() {
    LOG_INFO("Regression coefficients: ");
    for (int i = 0; i < Beta.rows(); i++)
        std::cout << "B" << i << ": " << Beta(i) << "\n";
    std::cout << "\n";
}

float Logistic_Regression::sigmoid(float x) {
    float y = std::clamp(x, -100.0f, 100.0f);
    return 1 / (1 + std::exp(-1 * y));
}


