//
// Created by skandan-c-y on 9/14/25.
//

#include <chrono>
#include <iostream>
#include "Models/SVMClassifierFlow.hpp"

int main() {
    std::vector<std::vector<float>> x_train_500, x_train_1000, x_train_5000, x_train_10000, x_train_50000,
    x_train_100000, x_train_140000, x_test;
    std::vector<std::string> y_train_500, y_train_1000, y_train_5000, y_train_10000, y_train_50000,
    y_train_100000, y_train_140000, y_test;

    // creating the dataset containers
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-500.csv", x_train_500, y_train_500, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-1000.csv", x_train_1000, y_train_1000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-5000.csv", x_train_5000, y_train_5000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-10000.csv", x_train_10000, y_train_10000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-50000.csv", x_train_50000, y_train_50000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-100000.csv", x_train_100000, y_train_100000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs-140000.csv", x_train_140000, y_train_140000, true);
    Glacier::Utils::read_csv("../Datasets/credit_scores/cs_val_2.csv", x_test, y_test, true);

    // hyperparameters - float lambda, int epochs
    float lambda = 0.5f;
    int epochs = 200;

    // training each model by measuring its time
    // 500 rows
    auto start1 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_500(x_train_500, y_train_500);
    iceberg_500.train(lambda, epochs);
    iceberg_500.predict(x_test);

    auto end1 = std::chrono::system_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "Time taken: " << time1.count() << " milli seconds \n";
    // 4ms

    // 1000 rows
    auto start2 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_1000(x_train_1000, y_train_1000);
    iceberg_1000.train(lambda, epochs);
    iceberg_1000.predict(x_test);

    auto end2 = std::chrono::system_clock::now();
    auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    std::cout << "Time taken: " << time2.count() << " milli seconds \n";
    // 6ms

    // 5000 rows
    auto start3 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_5000(x_train_5000, y_train_5000);
    iceberg_5000.train(lambda, epochs);
    iceberg_5000.predict(x_test);

    auto end3 = std::chrono::system_clock::now();
    auto time3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);

    std::cout << "Time taken: " << time3.count() << " milli seconds \n";
    // 25ms

    // 10000 rows
    auto start4 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_10000(x_train_10000, y_train_10000);
    iceberg_10000.train(lambda, epochs);
    iceberg_10000.predict(x_test);

    auto end4 = std::chrono::system_clock::now();
    auto time4 = std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4);

    std::cout << "Time taken: " << time4.count() << " milli seconds \n";
    // 50ms

    // 50000 rows
    auto start5 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_50000(x_train_50000, y_train_50000);
    iceberg_50000.train(lambda, epochs);
    iceberg_50000.predict(x_test);

    auto end5 = std::chrono::system_clock::now();
    auto time5 = std::chrono::duration_cast<std::chrono::milliseconds>(end5 - start5);

    std::cout << "Time taken: " << time5.count() << " milli seconds \n";
    // 248ms

    // 100000 rows
    auto start6 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_100000(x_train_100000, y_train_100000);
    iceberg_100000.train(lambda, epochs);
    iceberg_100000.predict(x_test);

    auto end6 = std::chrono::system_clock::now();
    auto time6 = std::chrono::duration_cast<std::chrono::milliseconds>(end6 - start6);

    std::cout << "Time taken: " << time6.count() << " milli seconds \n";
    // 491ms

    // 140000 rows
    auto start7 = std::chrono::system_clock::now();

    Glacier::Models::SVMClassifier iceberg_140000(x_train_140000, y_train_140000);
    iceberg_140000.train(lambda, epochs);
    iceberg_140000.predict(x_test);

    auto end7 = std::chrono::system_clock::now();
    auto time7 = std::chrono::duration_cast<std::chrono::milliseconds>(end7 - start7);

    std::cout << "Time taken: " << time7.count() << " milli seconds \n";
    // 695ms


    // analyzing the output
    iceberg_140000.analyze_2_targets(x_test, y_test);

    /*
    *[INFO]  Confusion matrix:
    Actually was_in_trouble, Predicted was_in_trouble: 522
    Actually was_in_trouble, Predicted was_not_in_trouble: 200
    Actually was_not_in_trouble, Predicted was_in_trouble: 6362
    Actually was_not_in_trouble, Predicted was_not_in_trouble: 2916
    Total number of rows: 10000

    [INFO]  Evaluation Metrics: (Out of 1)
    Accuracy: 0.3438
    Recall: 0.722992
    False positive rate: 0.685708
    Precision: 0.075828
    */

    return 0;
}
