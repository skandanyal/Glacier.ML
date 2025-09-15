//
// Created by skandan-c-y on 7/22/25.
//

#include "../Models/KNNRegressor.hpp"
#include "utilities.hpp"

int main() {

    // std::vector<std::vector<float>> x_train;
    // std::vector<float> y_train;

    // Utils::read_csv("../Datasets/Regression_datasets/train_sample_1.csv", x_train, y_train, true);
    // Utils::read_csv("../Datasets/Regression_datasets/test_sample_1.csv", x_test, y_test, true);

    std::vector<std::vector<float>> x_train = {
        {5.1, 3.5, 1.4, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {6.2, 2.8, 4.8, 1.8},
        {5.9, 3.0, 5.1, 1.8},
        {5.5, 2.3, 4.0, 1.3},
        {6.5, 3.0, 5.8, 2.2},
        {5.0, 3.6, 1.4, 0.2},
        {6.7, 3.1, 4.4, 1.4},
        {5.6, 2.9, 3.6, 1.3},
        {6.3, 2.5, 5.0, 1.9}
    };

    std::vector<float> y_train = {
        0, 0, 1, 1, 1, 2, 0, 1, 1, 2
    };

    std::vector<std::vector<float>> x_test = {
        {5.1, 3.3, 1.7, 0.5},
        {6.4, 2.9, 4.3, 1.3},
        {6.0, 3.0, 5.1, 1.8}
    };

    // std::vector<std::vector<float>> x_train = {
    //     {1,2,3,4}, {5,6,7,8}
    // };


    // std::vector<float> y_train = {
    //     1, 2
    // };

    // std::vector<float> x_test = {9, 10, 11, 12};

//   {6.4, 2.9, 4.3, 1.3},
//   {6.0, 3.0, 5.1, 1.8}



    // Multiple Linear Regression Workflow
    KNNRegressor iceberg(x_train, y_train);							                                            // initialize a model object

    int k = 4;
    std::string distance_metric = "Minkowski";
    int p = 3;
    iceberg.train(k, distance_metric, p);											                                    // train the model

    // std::vector<float> x_test = {6.4, 2.9, 4.3, 1.3};
    // std::cout << iceberg.predict(x_test) << std::endl;

    std::vector<float> answer = iceberg.predict(x_test);

    for ( auto ans : answer) {
        std::cout << ans << std::endl;
    }
    // iceberg.print_predict(x_test, y_test);
    // iceberg.analyze(x_test, y_test);						                                                        // test the model to find its accuracy using test data

    return 0;
}