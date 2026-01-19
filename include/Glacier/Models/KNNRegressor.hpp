#ifndef KNNREGRESSOR_HPP
#define KNNREGRESSOR_HPP

#pragma once
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "omp.h"
#include "../Utils/logs.hpp"
#include "../Utils/utilities.hpp"


namespace Glacier::Models {
    class KNNRegressor {
    private:
        std::vector<float> X;      // (n x p)
        std::vector<float> Y;                     // (n x 1)
        std::vector<std::string> labels;        // (p x 1)
        std::vector<float> mean;                // (p x 1)
        std::vector<float> std_dev;             // (p x 1)
        int distance_metric{};
        int k{};
        int p{};
        size_t nrows{}, ncols{};
        // the {} braces are for the constructor to initialize these variables outside the constructor

    public:
        KNNRegressor(std::vector<std::vector<float>> &X, std::vector<float> &Y);
        void train(int k, const std::string& distance_metric, int p=2);
        std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
        float predict(std::vector<float> &X_pred);
        void print_predict(std::vector<std::vector<float>>& x_val, std::vector<float> &y_val);
        void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);

    private:
        static float R2_score(std::vector<float> &actual, std::vector<float> &predicted);
    };
}
#endif //KNNREGRESSOR_HPP
