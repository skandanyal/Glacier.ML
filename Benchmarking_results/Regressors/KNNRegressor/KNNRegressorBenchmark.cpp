//
// Created by skandan-c-y on 7/14/25.
//
#include "../../../Glacier/Models/KNNRegressor.hpp"
#include "../../../Glacier/Utils/utilities.hpp"
#include <chrono>
#include <iostream>

int main() {
    // initializing the container vectors
    std::vector<std::vector<float>> x_test, x_train_100, x_train_500, x_train_1000, x_train_5000, x_train_10000, x_train_50000, x_train_95000;
    std::vector<float> y_test, y_train_100, y_train_500, y_train_1000, y_train_5000, y_train_10000, y_train_50000, y_train_95000;

    // loading the data onto the vectors
    Glacier::Utils::read_csv_r("../Datasets/mi_datasets/mi_100.csv", x_train_100, y_train_100, true);
    Glacier::Utils::read_csv_r("../Datasets/mi_datasets/mi_100.csv", x_train_500, y_train_500, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs-1000.csv", x_train_1000, y_train_1000, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs-5000.csv", x_train_5000, y_train_5000, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs-10000.csv", x_train_10000, y_train_10000, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs-50000.csv", x_train_50000, y_train_50000, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs-100000.csv", x_train_95000, y_train_95000, true);
    Glacier::Utils::read_csv_r("../Datasets/cs_datasets/cs_val_2.csv", x_test, y_test, true);

    // initializing the models
    Glacier::Models::KNNRegressor iceberg_100(x_train_100, y_train_100);
    Glacier::Models::KNNRegressor iceberg_500(x_train_500, y_train_500);
    Glacier::Models::KNNRegressor iceberg_1000(x_train_1000, y_train_1000);
    Glacier::Models::KNNRegressor iceberg_5000(x_train_5000, y_train_5000);
    Glacier::Models::KNNRegressor iceberg_10000(x_train_10000, y_train_10000);
    Glacier::Models::KNNRegressor iceberg_50000(x_train_50000, y_train_50000);
    Glacier::Models::KNNRegressor iceberg_95000(x_train_95000, y_train_95000);


    // hyperparameters
    // int k = 0;
    std::vector<std::string> distance_metric_str = {"manhattan", "euclidean", "minkowski"};   // manhattan, euclidean, minkowski
    int p = 3;

    // function signature = k , distance_metric, p

    // Benchmarking begins here

    std::cout << "Time taken: \n";
    for (int metric = 1; metric < 3; metric++) {
        for (int i=0; i<1; i++) {
            auto start_time_500 = std::chrono::high_resolution_clock::now();

            iceberg_500.train(23,  distance_metric_str[metric], p);
            iceberg_500.predict(x_val);

            auto end_time_500 = std::chrono::high_resolution_clock::now();

            std::cout << "500 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_500 - start_time_500).count() << " micro seconds\n";
        }
        std::cout << "\n";

        // for (int i=0; i<1; i++) {
        //     auto start_time_1000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_1000.train(32,  distance_metric_str[metric], p);
        //     iceberg_1000.predict(x_val);
        //
        //     auto end_time_1000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "1000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_1000 - start_time_1000) << " micro seconds\n";
        // }
        // std::cout << "\n";
        //
        // for (int i=0; i<1; i++) {
        //     auto start_time_5000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_5000.train(71,  distance_metric_str[metric], p);
        //     iceberg_5000.predict(x_val);
        //
        //     auto end_time_5000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "5000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_5000 - start_time_5000) << " micro seconds\n";
        // }
        // std::cout << "\n";
        //
        // for (int i=0; i<1; i++) {
        //     auto start_time_10000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_10000.train(100,  distance_metric_str[metric], p);
        //     iceberg_10000.predict(x_val);
        //
        //     auto end_time_10000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "10000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_10000 - start_time_10000) << " micro seconds\n";
        // }
        // std::cout << "\n";
        //
        // for (int i=0; i<1; i++) {
        //     auto start_time_50000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_50000.train(224,  distance_metric_str[metric], p);
        //     iceberg_50000.predict(x_val);
        //
        //     auto end_time_50000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "50000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_50000 - start_time_50000) << " micro seconds\n";
        // }
        // std::cout << "\n";
        //
        // for (int i=0; i<1; i++) {
        //     auto start_time_100000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_100000.train(316,  distance_metric_str[metric], p);
        //     iceberg_100000.predict(x_val);
        //
        //     auto end_time_100000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "100000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_100000 - start_time_100000) << " micro seconds\n";
        // }
        // std::cout << "\n";
        //
        // for (int i=0; i<1; i++) {
        //     auto start_time_140000 = std::chrono::high_resolution_clock::now();
        //
        //     iceberg_140000.train(374,  distance_metric_str[metric], p);
        //     iceberg_140000.predict(x_val);
        //
        //     auto end_time_140000 = std::chrono::high_resolution_clock::now();
        //
        //     std::cout << "140000 rows: " << std::chrono::duration_cast<std::chrono::microseconds>( end_time_140000 - start_time_140000) << " micro seconds\n";
        // }
        std::cout << "\n";
    }


    iceberg_500.analyze_2_targets(x_val, y_val);

    return 0;
}

