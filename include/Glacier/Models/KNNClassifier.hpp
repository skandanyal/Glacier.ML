#ifndef KNNCLASSIFIER_HPP
#define KNNCLASSIFIER_HPP

#pragma once
#include <vector>
#include <string>


namespace Glacier::Models {
    class KNNClassifier {

    private:
        std::vector<float> X;      // (n x p)
        std::vector<int> Y;                     // (n x 1)
        std::vector<std::string> labels;        // (p x 1)
        std::vector<float> mean;                // (p x 1)
        std::vector<float> std_dev;             // (p x 1)
        int distance_metric{};
        int k{};
        int p{};
        int no_threads{};
        size_t nrows{}, ncols{};
        // the {} braces are for the constructor to initialize these variables outside the constructor

    public:
        // constructor
        KNNClassifier(std::vector<std::vector<float>> &X_i, std::vector<std::string> &Y_i, int no_threads=0);
        void train(int k_i, std::string& distance_metric_i, int p_i=2);
        std::string predict(std::vector<float> &x_pred);
        std::vector<std::string> predict(std::vector<std::vector<float>>& x_pred);
        void print_predict(std::vector<std::vector<float>>& x_pred, std::vector<std::string> &y_pred);
        void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
    };
}
#endif //KNNCLASSIFIER_HPP
