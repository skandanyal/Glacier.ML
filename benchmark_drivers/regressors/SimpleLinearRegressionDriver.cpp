#include <iostream>
#include "Models/SimpleLinearRegression.hpp"
#include "Utils/utilities.hpp"

int main() {
	std::vector<float> x_train, y_train;
	std::vector<float> x_test, y_test;

	Glacier::Utils::read_csv_1d(__path_to_training_dataset__, x_train, y_train, true);
	Glacier::Utils::read_csv_1d(__path_to_test_dataset__, x_test, y_test, true);


	Glacier::Simple_Linear_Regression iceberg(x_train, y_train);
	iceberg.train();
	iceberg.print_values();

	iceberg.predict_print(x_test, y_test);
	iceberg.analyze(x_test, y_test);

	return 0;
}


