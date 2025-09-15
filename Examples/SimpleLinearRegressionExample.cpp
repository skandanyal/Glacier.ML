#include <iostream>
#include "../Models/SimpleLinearRegression.hpp"
#include "utilities.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
	// Enable ANSI color in Windows terminal
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD dwMode = 0;
	GetConsoleMode(hOut, &dwMode);
	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	SetConsoleMode(hOut, dwMode);
#endif


int main() {
	std::vector<float> x_train, y_train;
	std::vector<float> x_test, y_test;

	Utils::read_csv_1d(__path_to_training_dataset__, x_train, y_train, true);
	Utils::read_csv_1d(__path_to_test_dataset__, x_test, y_test, true);


	Simple_Linear_Regression iceberg(x_train, y_train);
	iceberg.train();
	iceberg.print_values();

	iceberg.predict_print(x_test, y_test);
	iceberg.analyze(x_test, y_test);

	return 0;
}


