//
// Created by Dell on 22-03-2025.
//

#include <iostream>
#include "testing_slr.h"

int main() {
	std::vector<float> x_train = {1, 2, 3, 4, 5};
	std::vector<float> y_train = {2, 3, 6, 8, 10};

	std::vector<float> x_test = {6, 7, 8};
	std::vector<float> y_test = {12, 15, 16};

	Simple_Linear_Regression model(x_test, y_test);
	model.train();
	model.print_values();

	model.test(x_test, y_test);
	model.predict_print(x_test);

	return 0;
}


