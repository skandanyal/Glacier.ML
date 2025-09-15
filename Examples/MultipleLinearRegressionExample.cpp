#include <iostream>
#include "../Models/MultipleLinearRegression.hpp"
#include "utilities.hpp"

// void print_dataset_details() {
//     std::string description = "X1: Square feet\nX2: Number of bedrooms\n3:Distance to City Centre\nX4:Age of house (years)\n";
// }

int main() {

    std::vector<std::vector<float>> x_train, x_test;
    std::vector<float> y_train, y_test;

    Utils::read_csv("../Datasets/Regression_datasets/train_sample_1.csv", x_train, y_train, true);
    Utils::read_csv("../Datasets/Regression_datasets/test_sample_1.csv", x_test, y_test, true);


    // Multiple Linear Regression Workflow
    Multiple_Linear_Regression iceberg(x_train, y_train);							                                // initialize a model object
    iceberg.train();											                                                        // train the model

    iceberg.print_Rcoeff_values();									                                                    // print the coefficient values

    std::vector<float> Y_pred = iceberg.predict(x_test);
    iceberg.analyze(x_test, y_test);						                                                        // test the model to find its accuracy using test data

    std::vector<float> x_test2 = {-1, 5, 7, -1, 10};
    float y_test_2 = iceberg.predict(x_test2);
    std::cout << "\n ans " << y_test_2 << std::endl;

    return 0;
}