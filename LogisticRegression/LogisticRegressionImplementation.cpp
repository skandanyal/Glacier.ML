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
Logistic_Regression::Logistic_Regression(std::vector<std::vector<double>> &X_i, std::vector<std::string> &Y_i) : X(), Y(), Beta(), F_x() {
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
    X.col(0) = Eigen::VectorXf::Ones((Eigen::Index) nrows);

    for (size_t row = 0; row < nrows; row++)
        for (size_t col = 0; col < ncols; col++)
            X(row, col + 1) = X_i[row][col];                                                                            // X matrix, with 0th column as 1
    LOG_DEBUG("Number of rows in x_train", X.rows());
    LOG_DEBUG("Number of cols in x_train", X.cols());
    std::cout << "\n";

    // normalize X
    for (size_t colm = 0; colm < ncols; colm++) {
        mean(colm) = X.col(colm).mean();
        std_dev(colm) = std::sqrt((X.col(colm).array() - mean(colm)).square().sum() / (X.rows() - 1));

        if (std_dev(colm) == 0)
            std_dev(colm) = 1e-8;

        // normalization
        X.col(colm) = (X.col(colm).array() - mean(colm)) / std_dev(colm);
    }

    // initialize Y
    size_t Y_i_size = Y_i.size();
    for (size_t i = 0; i < Y_i_size; i++) {
        if (labels.size() == 0) labels.push_back(Y_i[i]);
        else if (labels[0] != Y_i[i]) labels.push_back(Y_i[i]);

        if (labels.size() > 2)
            LOG_ERROR("More than two classification classes detected. Binary classification required the dataset to have only two target classes.")
    }
    if (labels.size() < 2)
        LOG_ERROR("Less than two classification classes detected. Binary classification required the dataset to have two target classes.")

    if (labels[0] > labels[1]) std::swap(labels[0], labels[1]);

    for (size_t i = 0; i < Y_i.size(); i++) {
        if (Y_i[i] == labels[0]) Y(i) = 0;
        else if (Y_i[i] == labels[1]) Y(i) = 1;
    }

    LOG_DEBUG("Number of rows in y_train", Y.rows());
    LOG_DEBUG("Number of cols in y_train", Y.cols());
    std::cout << "\n";

    // initialize Beta
    Beta = Eigen::VectorXf::Zero(X.cols());
}

void Logistic_Regression::train(float alpha, int iterations) {
    auto train_start = std::chrono::high_resolution_clock::now();

    ////////////////////// Training begins here //////////////////////

    LOG_DEBUG("Number of rows in Beta", Beta.rows());
    LOG_DEBUG("Number of cols in Beta", Beta.cols());
    std::cout << "\n";

    // training loop
    float loss = 0.0f;
    for (int i = 1; i <= iterations; i++) {
        loss = 0.0f;  // reset this for every loop

        //please, this isn't chatgpt's work. it gives a far more concise code to do the same...
        // step 1 - calculate F_x and P_x
        P_x.resize(X.rows());
        F_x = X * Beta;
        for(int i=0; i<F_x.size(); i++){
            P_x(i) = std::clamp(sigmoid(F_x(i)), 1e-8f, 1.0f - 1e-8f);
        }

        // step 2 - calculate error and gradient(delta)
        Delta = X.transpose() * (P_x - Y);

        // step 3 - update the beta matrix
        Beta -= alpha * Delta;

        // logging the loss
        for (size_t row=0; row < Y.size(); row++)
            loss += -1 * (Y(row) * std::log(P_x(row)) + (1 - Y(row)) * std::log(1 - P_x(row)));
        loss = loss / Y.size();

        LOG_DEBUG("Loss at iteration " + std::to_string(i), loss);
    }
    LOG_DEBUG("Final loss at the end ", loss);

    ////////////////////// Training ends here /////////////////////////

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start);

    LOG_TIME("Training", duration.count());
    LOG_INFO("Model training is complete.");
    std::cout << "\n";

    /*
     * FURTHER ENHANCEMENTS:
     * 1. (done) log the loss at every iteration, or after series of iterations
     * 2. reduce or maximise alpha based on the loss
     * 3. add an early stopping mechanism (hard code this, no need to let the user decide the threshold)
     */
}

std::string Logistic_Regression::predict(std::vector<double> &x_pred) {
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

    for (int colm = 0; colm < mean.size(); colm++)     // normalizing the x matrix
        x.col(colm) = (x.col(colm).array() - mean(colm)) / std_dev(colm);

    float ans = x.dot(Beta);
    if (ans < 0.5f)
        return labels[0];
    return labels[1];
}

std::vector<std::string> Logistic_Regression::predict(std::vector<std::vector<float>>& X_test) {
    if (Beta.size() == 0) {
        LOG_ERROR("Train the data using train() before using predict().");
    }

    if (X_test[0].size() != (int) X.cols()) {
        LOG_ERROR("Train and test dataset have different numebr of features.");
    }

    auto nrows = static_cast<Eigen::Index>(X_test.size());
    auto ncols = static_cast<Eigen::Index>(X_test[0].size());

    if (mean.size() != ncols)
        LOG_ERROR("Mismatch in mean/std_dev size. Possible unnormalized feature set.");

    Eigen::MatrixXf X_pred(nrows, ncols + 1);
    X_pred.col(0) = Eigen::VectorXf::Ones(nrows);

    for (Eigen::Index row = 0; row < nrows; row++)
        for (Eigen::Index col = 0; col < ncols; col++)
            X_pred(row, col + 1) = X_test[row][col];                                                                         // X matrix, with 0th column as 1

    for (int colm = 0; colm < mean.size(); colm++)     // normalizing the X_pred matrix
        X_pred.col(colm) = (X_pred.col(colm).array() - mean(colm)) / std_dev(colm);

    Eigen::VectorXf F_x_pred = X_pred * Beta;
    P_x_pred.resize(nrows);
    for(int i=0; i<F_x_pred.size(); i++){
        P_x_pred(i) = std::clamp(sigmoid(F_x_pred(i)), 1e-8f, 1.0f - 1e-8f);
    }

    std::vector<std::string> result(nrows);
    for (Eigen::Index i = 0; i < nrows; i++) {
        if (P_x_pred(i) < 0.5f) result[i] = labels[0];
        else result[i] = labels[1];
    }

    return result;
}

float Logistic_Regression::sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}


