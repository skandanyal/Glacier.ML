#include <iostream>
#include "../Glacier/Models/MultipleLinearRegression.hpp"
#include "../Glacier/Utils/utilities.hpp"



int main() {

    std::vector<std::vector<float>> x_train, x_test;
    std::vector<float> y_train, y_test;

    Glacier::Utils::read_csv(__path_to_training_dataset__, x_train, y_train, true);
    Glacier::Utils::read_csv(__path_to_test_dataset__, x_test, y_test, true);


    // Multiple Linear Regression Workflow
    Glacier::Multiple_Linear_Regression iceberg(x_train, y_train);							                                // initialize a model object
    iceberg.train();											                                                        // train the model

    iceberg.print_Rcoeff_values();									                                                    // print the coefficient values

    std::vector<float> Y_pred = iceberg.predict(x_test);
    iceberg.analyze(x_test, y_test);						                                                        // test the model to find its accuracy using test data

    std::vector<float> x_test2 = {-1, 5, 7, -1, 10};
    float y_test_2 = iceberg.predict(x_test2);
    std::cout << "\n ans " << y_test_2 << std::endl;

    return 0;
}