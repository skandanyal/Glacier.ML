//
// Created by skandan-c-y on 9/14/25.
//

#include <chrono>
#include <iostream>
#include "Models/SVMClassifierFlow.hpp"

int main() {
    std::vector<std::vector<float>> x_train, x_pred;
    std::vector<std::string> y_train, y_pred;
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-10000.csv", x_train, y_train, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs_val_2.csv", x_pred, y_pred, true);

    std::vector<float> something = {0.0,57,0,5.0,5400.0,4,0,0,0,0.0};
    float lambda = 0.3f;
    int epochs = 30;

    auto start_time_500 = std::chrono::high_resolution_clock::now();

    Glacier::Models::SVMClassifier model(x_train, y_train);
    model.train(lambda, epochs);
    std::cout << model.predict(something);
    // model.analyze_2_targets(x_pred, y_train);

    auto end_time_500 = std::chrono::high_resolution_clock::now();

    std::cout << "140000 rows training + 1000 rows prediction: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_500 - start_time_500) << " milli seconds\n";

    return 0;
}
