#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include "testing_slr.h"

Simple_Linear_Regression::Simple_Linear_Regression(std::vector<float> &x, std::vector<float> &y) {
	if(y.size() != x.size()) {
		throw std::runtime_error("Size of Y vector and X vector are not the same.");
	}
	this->y = y;
	this->x = x;
	this->n = y.size();

	m = 0.0, c = 0.0;

	sum_y = std::reduce(y.begin(), y.end());
	sum_x = std::reduce(x.begin(), x.end());
}

void Simple_Linear_Regression::train() {
		m = calculate_m();
		c = calculate_c();
		std::cout << "Model is trained.\n";
	}

float Simple_Linear_Regression::calculate_m() {
	float ssx = 0.0, ssxy = 0.0;

	for(int i=0; i<n; i++) {
		ssx += x[i] * x[i];
		ssxy += x[i] * y[i];
	}

	float numer = n * ssxy - sum_x * sum_y;
	float denr = n * ssx - sum_x * sum_x;
	return numer / denr;
}

float Simple_Linear_Regression::calculate_c() {
	return (sum_y - m * sum_x) / n;
}

void Simple_Linear_Regression::print_values() const {
	std::cout << "m: " << m << "\nc: " << c << "\n";
}

float Simple_Linear_Regression::predict(float a) {
	return m * a + c;
}

void Simple_Linear_Regression::test(std::vector<float> &x_test, std::vector<float> &y_test) {
	if (x_test.size() != y_test.size()) {
		throw std::runtime_error("Test size does not match.\n");
	}

	/*
	 * Using RMSE, MAE, MAPE
	 * MSE - Mean Squared Error:
	 *		(1/n) * sum(y_gen - y_pred)^2
	 * RMSE - Root Mean Squared Error:
	 *		MSE^0.5
	 * MAE - Mean Absolute Error:
	 *		(1/n) * sum(abs(y_gen - y_pred))
	 * MAPE - Mean Absolute Percentage Error:
	 *		(100/n) * abs((y_gen - y_pred) / y_pred) %
	 */

	size_t n_test = x_test.size();
	float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;

	for(size_t i=0; i<n_test; i++) {
		float y_pred = predict(x_test[i]);
		float error = y_test[i] - y_pred;

		mse += error * error;
		mae += abs(error);
		if (y_test[i] != 0) {
			mape += std::abs(error / y_test[i]);
		}
	}
	mse = mse / n_test;
	rmse = std::sqrt(mse);
	mae = mae / n_test;
	mape = mape / n_test * 100;

	std::cout << "RMSE: " << rmse << "\n";
	std::cout << "MAE: " << mae << "\n";
	std::cout << "MAPE: " << mape << "%\n";
}

void Simple_Linear_Regression::predict_print(std::vector<float> const x_pred) {
	std::cout << "X values:\t\tY values:\n";
	for(float i : x_pred) {
		std::cout << i << "\t\t" << predict(i) << "\n";
	}
}

std::vector<float> Simple_Linear_Regression::predict(std::vector<float> x_pred) {
	std::vector<float> result(x_pred.size(), 0.0);

	std::cout << "X values:\t\tY values:\n";
	for(size_t i=0; i<x_pred.size(); i++) {
		result[i] = predict(x_pred[i]);
	}

	return result;
}

