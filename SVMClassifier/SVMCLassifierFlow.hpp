//
// Created by skandan-c-y on 9/14/25.
//
#pragma once

#include <vector>
#include <string>

#ifndef SVMCLASSIFIERFLOW_HPP
#define SVMCLASSIFIERFLOW_HPP

class SVMClassifier {
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
    size_t nrows, ncols;
    // the {} braces are for the constructor to initialize these variables outside the constructor

public:
    SVMClassifier(std::vector<std::vector<float>> &x, std::vector<std::string> &y);
    void train(int k, std::string& distance_metric, int p=2);
    std::string predict(std::vector<float> &x_pred);
    std::vector<std::string> predict(std::vector<std::vector<float>>& x_test);
    void print_predict(std::vector<std::vector<float>>& x_val, std::vector<std::string> &y_val);
    // mimic predict_proba from sklearn
    void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
};

#endif //SVMCLASSIFIERFLOW_HPP
