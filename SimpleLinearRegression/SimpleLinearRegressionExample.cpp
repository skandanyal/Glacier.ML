#include <iostream>
#include "SimpleLinearRegressionFlow.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main() {

#ifdef _WIN32
	// Enable ANSI color in Windows terminal
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD dwMode = 0;
	GetConsoleMode(hOut, &dwMode);
	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	SetConsoleMode(hOut, dwMode);
#endif

	std::vector<float> x_train = {1, 2, 3, 4, 5};
	std::vector<float> y_train = {2, 3, 6, 8, 10};

	std::vector<float> x_test = {6, 7, 8};
	std::vector<float> y_test = {12, 15, 16};

	Simple_Linear_Regression iceberg(x_train, y_train);
	iceberg.train();
	iceberg.print_values();

	iceberg.predict_print(x_test);
	iceberg.analyze(x_test, y_test);

	return 0;
}


