//
// Created by skandan-c-y on 7/2/25.
//

#include <iostream>
#include <vector>
#include "Glacier/Models/LogisticRegression.hpp"
#include "Glacier/Utils/utilities.hpp"
#include <chrono>
#include <numeric>

#define RUNS 5

int main() {
    std::vector<std::vector<float>> x_train_500, x_train_1000, x_train_5000, x_train_10000, x_train_50000,
                                    x_train_100000, x_train_140000, x_test;
    std::vector<std::string> y_train_500, y_train_1000, y_train_5000, y_train_10000, y_train_50000,
                             y_train_100000, y_train_140000, y_test;

    // creating the dataset containers
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-500.csv", x_train_500, y_train_500, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-1000.csv", x_train_1000, y_train_1000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-5000.csv", x_train_5000, y_train_5000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-10000.csv", x_train_10000, y_train_10000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-50000.csv", x_train_50000, y_train_50000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-100000.csv", x_train_100000, y_train_100000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs-140000.csv", x_train_140000, y_train_140000, true);
    Glacier::Utils::read_csv_c("Datasets/cs_datasets/cs_val_2.csv", x_test, y_test, true);

    // hyperparameters - float alpha, int iterations
    float alpha = 0.001;
    int iterations = 2000;

    // ---------------------- warm-up ----------------------
    for (int i = 0; i < 5; i++) {
        Glacier::Models::Logistic_Regression iceberg_500(x_train_500, y_train_500, 0);
        iceberg_500.train(alpha, iterations);
        iceberg_500.predict(x_test, 0.5);
    }

    std::cout << "DATASET SIZE:\t\t TIME TAKEN\n";

    // 500 rows
    std::vector<double> timing_500;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_500, y_train_500, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_500.push_back(elapsed.count());
    }
    double avg_500 = std::accumulate(timing_500.begin(), timing_500.end(), 0.0) / RUNS;
    std::cout << "500\t\t " << avg_500 << " milli seconds \n";

    // 1000 rows
    std::vector<double> timing_1000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_1000, y_train_1000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_1000.push_back(elapsed.count());
    }
    double avg_1000 = std::accumulate(timing_1000.begin(), timing_1000.end(), 0.0) / RUNS;
    std::cout << "1000\t\t " << avg_1000 << " milli seconds \n";

    // 5000 rows
    std::vector<double> timing_5000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_5000, y_train_5000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_5000.push_back(elapsed.count());
    }
    double avg_5000 = std::accumulate(timing_5000.begin(), timing_5000.end(), 0.0) / RUNS;
    std::cout << "5000\t\t " << avg_5000 << " milli seconds \n";

    // 10000 rows
    std::vector<double> timing_10000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_10000, y_train_10000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_10000.push_back(elapsed.count());
    }
    double avg_10000 = std::accumulate(timing_10000.begin(), timing_10000.end(), 0.0) / RUNS;
    std::cout << "10000\t\t " << avg_10000 << " milli seconds \n";

    // 50000 rows
    std::vector<double> timing_50000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_50000, y_train_50000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_50000.push_back(elapsed.count());
    }
    double avg_50000 = std::accumulate(timing_50000.begin(), timing_50000.end(), 0.0) / RUNS;
    std::cout << "50000\t\t " << avg_50000 << " milli seconds \n";

    // 100000 rows
    std::vector<double> timing_100000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_100000, y_train_100000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_100000.push_back(elapsed.count());
    }
    double avg_100000 = std::accumulate(timing_100000.begin(), timing_100000.end(), 0.0) / RUNS;
    std::cout << "100000\t\t " << avg_100000 << " milli seconds \n";

    // 140000 rows
    std::vector<double> timing_140000;
    for (int i = 0; i < RUNS; i++) {
        auto start = std::chrono::system_clock::now();
        Glacier::Models::Logistic_Regression model(x_train_140000, y_train_140000, 0);
        model.train(alpha, iterations);
        std::vector<std::string> pred =
            model.predict(x_test, 0.5);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timing_140000.push_back(elapsed.count());
    }
    double avg_140000 = std::accumulate(timing_140000.begin(), timing_140000.end(), 0.0) / RUNS;
    std::cout << "140000\t\t " << avg_140000 << " milli seconds \n";

    return 0;
}