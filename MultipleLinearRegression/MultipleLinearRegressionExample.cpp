#include <iostream>
#include "MultipleLinearRegression.h"

void print_dataset_details() {
    std::string description = "X1: Square feet\nX2: Number of bedrooms\n3:Distance to City Centre\nX4:Age of house (years)\n";
}

int main() {
    // Preparing model dataset

    std::vector<std::vector<float>> X_train = {
        {1500, 3 , 5, 10},
        {1800, 4, 3, 5},
        {2100, 4, 8, 15},
        {2400, 5, 2, 2},
        {3000, 5, 6, 8}
    };
    std::vector<float> Y_train = {75, 95, 85, 120, 130};

    std::vector<std::vector<float>> X_test = {
        {2000, 3, 4, 8},
        {2700, 5, 2, 3},
        {3200, 4, 7, 10},
    };
    std::vector<float> Y_test = {80, 115, 125};


    // Multiple Linear Regression Workflow

    MultipleLinearRegression iceberg;							                                 // initialize a model object

    iceberg.fit(X_train, Y_train);						                                 // fit the training data
    iceberg.train();											                                 // train the model

    iceberg.print_values();									                                     // print the coefficient values

    std::vector<std::vector<float>> X_pred = {{2500, 4, 5, 7}};
    std::vector<float> Y_pred = iceberg.predict(X_pred);
    std::cout << "\nPredicted price of the house is rs." << Y_pred[0] << " Lakh.\n\n";

    iceberg.analyze(X_test, Y_test);						                                 // test the model to find its accuracy using test data

    return 0;
}