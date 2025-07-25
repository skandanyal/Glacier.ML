//
// Created by skandan-c-y on 7/22/25.
//

#ifndef KNNREGRESSORFLOW_H
#define KNNREGRESSORFLOW_H

#include <string>
#include <vector>
#include <cmath>
#include "min_heap.hpp"

class KNNRegressor {
private:
    std::vector<std::vector<float>> X;
    std::vector<float> Y;
    // std::pair<double, float> Dist;
    std::vector<float> mean;
    std::vector<float> std_dev;
    int distance_metric{};
    int k{};
    int p{};


public:
    KNNRegressor(std::vector<std::vector<float>> &X, std::vector<float> &Y);
    void train(int k, const std::string& distance_metric, int p=2);
    std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
    float predict(std::vector<float> &X_pred);
    void print_predict(std::vector<std::vector<float>>& x_val, std::vector<float> &y_val);
    void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);

private:
    float R2_score(std::vector<float> &actual, std::vector<float> &predicted);
};

#endif //KNNREGRESSORFLOW_H
