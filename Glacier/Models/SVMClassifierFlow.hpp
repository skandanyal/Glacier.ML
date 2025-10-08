//
// Created by skandan-c-y on 9/14/25.
//

#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <chrono>
#include <cblas.h>
#include <Eigen/Dense>
#include "Utils/utilities.hpp"
#include "Utils/logs.hpp"

#ifndef SVMCLASSIFIERFLOW_HPP
#define SVMCLASSIFIERFLOW_HPP


namespace Glacier::Models {
    class SVMClassifier {
    private:
        Eigen::MatrixXf X;          // (n x p+1), 1 for bias
        Eigen::VectorXf w;          // (1 x p)
        Eigen::VectorXf Y;          // (N X 1)
        Eigen::Index nrows{};
        Eigen::Index ncols{};
        Eigen::VectorXf mean;
        Eigen::VectorXf std_dev;
        std::vector<std::string> labels;    // 2


    public:
        SVMClassifier(std::vector<std::vector<float>> &x, std::vector<std::string> &y);
        void train(float lambda, int epochs);
        std::string predict(std::vector<float> &x_pred);
        std::vector<std::string> predict(std::vector<std::vector<float>>& x_pred);
        void print_predict(std::vector<std::vector<float>>& x_pred, std::vector<std::string> &y_pred);
        // mimic predict_proba from sklearn
        void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
        void show_final_weights();
        void show_support_vectors(); // no. of features = no. of SVs
    };

    // constructor
    inline SVMClassifier::SVMClassifier(std::vector<std::vector<float> > &X_i, std::vector<std::string> &Y_i) {
        // check for the number of threads, cut it by two, and use those many
        // int threads = omp_get_max_threads() / 2;
        // omp_set_num_threads(threads);
        // LOG_DEBUG("Number of threads", threads);

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
        LOG_DEBUG("Number of rows in input X dataset", X_i.size());
        LOG_DEBUG("Number of cols in input X dataset", X_i[0].size());
        std::cout << "\n";

        // Resizing all the DSs to the required size
        X.resize(nrows, ncols + 1);           // no. of rows + (no. of cols + 1 bias)
        Y.resize(nrows);                                   // no of rows
        mean.resize(ncols);                                 // no of cols
        std_dev.resize(ncols);                              // no of cols


        // populating the Eigen X matrix with the data while rejecting NaN or infinite values
        X.col(ncols) = Eigen::VectorXf::Ones((Eigen::Index) nrows);
        for (Eigen::Index row = 0; row < nrows; row++) {
            for (Eigen::Index col = 0; col < ncols; col++) {
                if (std::isfinite(X_i[row][col])) {
                    X(row, col) = X_i[row][col];            // X matrix, with 0th column as 1
                } else {
                    LOG_ERROR("Bad value found at  X_i[" + std::to_string(row) + "][" + std::to_string(col) + "]. Program stalling...");
                }
            }
        }
        LOG_DEBUG("Number of rows in Eigen x_train", X.rows());
        LOG_DEBUG("Number of cols in Eigen x_train", X.cols());
        std::cout << "\n";

        // normalize X
        for (int colm = 0; colm < ncols; colm++) {
            // calculating mean and standard deviation for normalization
            mean(colm) = X.col(colm).sum() / nrows;
            std_dev(colm) = std::sqrt((X.col(colm).array() - mean(colm)).square().sum() / nrows);

            // normalizing X
            X.col(colm).array() = (X.col(colm).array() - mean(colm)) / std_dev(colm);
        }
        LOG_UPDATE("Mean and Std_dev vectors are formed.");
        LOG_UPDATE("Dataset normalized.");

        // initialize Y
        // store the labels present in the dataset
        std::map<std::string, bool> seen;
        for (const std::string& target : Y_i)
            seen[target] = true;
        for (auto &[key, _] : seen)
            labels.push_back(key);

        if (labels.size() < 2) {
            LOG_ERROR("Less than two classification classes detected. Classification requires the dataset to have at least two target classes.");
        } if (labels.size() > 2) {
            LOG_ERROR("More than two classification classes detected. Classification requires the dataset to have at least two target classes.");
        }

        // multiclass classification by default
        std::ranges::sort(labels);

        // assign -1 and 1 to segregate btw the two classes
        std::map<std::string, int> indexing;
        indexing[labels[0]] = -1;
        indexing[labels[1]] = 1;
        for (long j = 0; j < nrows; j++) {
            Y[j] = (float) indexing[Y_i[j]];
        }
        LOG_UPDATE("Labels vector formed and indexed.");

        LOG_DEBUG("Size of labels", labels.size());
        LOG_DEBUG("Number of rows in y_train", Y.size());
        std::cout << "\n";
    }

    inline void SVMClassifier::train(float lambda, int epochs) {

        ////////////////////// Training begins here //////////////////////

        // w has just as  many cols as the training dataset, plus the bias
        w = Eigen::VectorXf::Zero((Eigen::Index)ncols + 1);

        int dim = w.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int row = 0; row < nrows; row++) {
                float eta = 1.0f / (lambda * (float) (row + 1));

                // BLAS dot product: margin = Y(row) * (w.dot(X.row(row)))
                float margin = Y(row) * cblas_sdot(dim, w.data(), 1, X.row(row).data(), 1);

                if (margin >= 1.0f) {
                    // scale w -> w = (1 - eta*lambda)*w
                    cblas_sscal(dim, 1 - eta*lambda, w.data(), 1);
                } else {
                    // scale w first
                    cblas_sscal(dim, 1 - eta*lambda, w.data(), 1);
                    // w += eta * Y(row) * X.row(row).transpose()
                    float alpha = eta * Y(row);
                    cblas_saxpy(dim, alpha, X.row(row).data(), 1, w.data(), 1);
                }
            }

            // if (epoch % 10 == 0) {
            //     LOG_UPDATE("Weights at Epoch " + std::to_string(epoch) + " : " +
            //                w.transpose().format(Eigen::IOFormat(Eigen::StreamPrecision,
            //                                                    Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]")));
            // }
        }

        ////////////////////// Training ends here ////////////////////////

        LOG_INFO("Model training is complete.");
        std::cout << "\n";
    }

    inline void SVMClassifier::show_final_weights() {
        std::cout << "Final weights: " << w.transpose() << "\n";
    }

    inline void SVMClassifier::show_support_vectors() {
        std::cout << "Support vectors are: \n";
        // for (int row = 0; row < nrows; row++) {
        //     int quant =  Y(row) * (w.transpose().dot(X.row(row)));
        //     if (quant < 1) {
        //         std::string vector = "";
        //         for (int i=0; i<ncols; i++) {
        //             vector += std::to_string()
        //         }
        //     }
        // }
    }

    inline std::string SVMClassifier::predict(std::vector<float> &x_pred) {
        if (x_pred.size() != ncols) {
            LOG_ERROR("Train and test dataset have different number of features.");
        }

        ////////////////////// Prediction begins here //////////////////////

        /*
         * Steps:
         * -----
         * Predict by computing sign(w.dot(x_aug))
         */

        Eigen::VectorXf x_p(ncols + 1, 0);
        for (int i=0; i<ncols; i++)
            x_p(i) = (x_pred[i] - mean(i)) / std_dev(i);
        x_p(ncols) = 1;

        float qty = w.transpose().dot(x_p);

        ////////////////////// Prediction ends here //////////////////////

        return qty >= 0 ? labels[1] : labels[0];
    }

    inline std::vector<std::string> SVMClassifier::predict(std::vector<std::vector<float>> &x_pred) {
        if (x_pred[0].size() != ncols) {
            LOG_ERROR("Train and test dataset have different number of features.");
        }

        Eigen::Index Nrows = x_pred.size();
        Eigen::Index Ncols = x_pred[0].size();

        LOG_DEBUG("Number of rows in x_pred", Nrows);
        LOG_DEBUG("Number of columns in x_pred", Ncols);
        std::cout << "\n";

        // check if input matrices are empty or not
        if (x_pred.empty()) {                                                                                   // Check if the inputs are valid or not
            LOG_ERROR("Input data cannot be empty.");
        }

        // load the matrix into an Eigen matrix
        Eigen::MatrixXf x_p(Nrows, Ncols+1);
        x_p.col(ncols) = Eigen::VectorXf::Ones((Eigen::Index) Nrows);
        for (int row = 0; row < Nrows; row++) {
            for (int col = 0; col < Ncols; col++) {
                if (std::isfinite(x_pred[row][col])){
                    x_p(row, col) = x_pred[row][col];
                }
                else {
                    LOG_ERROR("Bad value found at  X_i[" + std::to_string(row) + "][" + std::to_string(col) + "]. Program stalling...");
                }
            }
        }
        for (int colm = 0; colm < Ncols; colm++) {
            X.col(colm).array() = (X.col(colm).array() - mean(colm)) / std_dev(colm);
        }

        // predict using the formula sign(X_Pred * w)
        Eigen::VectorXf score = x_p * w;
        std::vector<std::string> answer(Nrows);
        for (int i=0; i < Nrows; i++)
            answer[i] = score(i) >= 0 ? labels[1] : labels[0];

        return answer;
    }

    inline void SVMClassifier::print_predict(std::vector<std::vector<float>> &x_pred, std::vector<std::string> &y_pred) {
        std::vector<std::string> y_test = predict(x_pred);

        std::cout << "Predicted\t|\tActual\n";
        for (int i = 0; i < y_test.size(); i++) {
            std::cout << y_test[i] << "\t|\t" << y_pred[i] << "\n";
        }

        std::cout << "\n";
    }

    inline void SVMClassifier::analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test) {
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
}

#endif //SVMCLASSIFIERFLOW_HPP
