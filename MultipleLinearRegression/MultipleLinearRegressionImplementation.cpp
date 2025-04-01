#include <iostream>
#include "MultipleLinearRegression.h"

MultipleLinearRegression::MultipleLinearRegression() : X(), Y(), Beta(), E() {};

void MultipleLinearRegression::fit(std::vector<std::vector<float>> &X_ip, std::vector<float> &Y_ip) {
	if (X_ip.empty() || Y_ip.empty()) {					  // Check if the inputs are valid or not
		std::cerr << "Input data cannot be empty.\n";
		std::exit(EXIT_FAILURE);
	}

	for(auto &row : X_ip)					                  // Check if all the rows are of the same size
		if(row.size() != X_ip[0].size()) {
			std::cerr << "Row sizes not consistent.\n";
			std::exit(EXIT_FAILURE);
		}

	size_t nrows = X_ip.size();
	size_t ncols = X_ip[0].size();

	X = Eigen::MatrixXf(nrows, ncols+1);                                      // use this method to resize an eigen matrix;
	Y = Eigen::VectorXf(nrows);                                               // use (nrows, 1) to ensure column vector

	X.col(0) = Eigen::VectorXf::Ones(nrows);

	for(size_t row=0; row<nrows; row++)
		for(size_t col=0; col<ncols; col++)
			X(row, col+1) = X_ip[row][col];				  // X matrix, with 0th column as 1

	for(size_t i=0; i<Y_ip.size(); i++)
		Y(i) = Y_ip[i];						          // Y matrix containing target column
}

void MultipleLinearRegression::train() {
	// Beta is (n x m) as X is (m * n)
	Beta = (X.transpose() * X).inverse() * X.transpose() * Y;                 // (m x n) * (n * m) * (m * n) * (n x 1)
	E = Y - X * Beta;                                                         // (n * 1)
	std::cout << "Model training is complete.\n";
}

void MultipleLinearRegression::print_values() {

	for(int i=0; i<Beta.rows(); i++)
		std::cout << "B" << i << ": " << Beta(i) << "\n";
}

std::vector<float> MultipleLinearRegression::predict(std::vector<std::vector<float>> &X) {
	if(Beta.size() == 0) {
		std::cerr << "Train the data using train() before using predict().\n";
		std::exit(EXIT_FAILURE);
	}

	/*
	 * size_t is unsigned long long, whereas Eigen uses long long for indexing and sizing
	 * hence using this explicit conversion over here - Eigen::Index instead of auto
	 */

	auto nrows = static_cast<Eigen::Index>(X.size());
	auto ncols = static_cast<Eigen::Index>(X[0].size());

	Eigen::MatrixXf X_pred(nrows, ncols+1);
	X_pred.col(0) = Eigen::VectorXf::Ones(nrows);

	for(Eigen::Index row=0; row<nrows; row++)
		for(Eigen::Index col=0; col<ncols; col++)
			X_pred(row, col+1) = X[row][col];			   // X matrix, with 0th column as 1

	Eigen::VectorXf Y_pred = X_pred * Beta;

	std::vector<float> result(nrows);
	for(Eigen::Index i=0; i<nrows; i++)
		result[i] = Y_pred[i];

	return result;
}

float MultipleLinearRegression::R_squared() {
	float sst = (Y.array() - Y.mean()).square().sum();
	float ssr = E.squaredNorm();
	return 1 - ssr/sst;
}

void MultipleLinearRegression::test(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) {
	if (x_test.size() != y_test.size()) {
		std::cerr << "Test size does not match.\n";
		std::exit(EXIT_FAILURE);
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

	float n_test = x_test.size();
	float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;

	std::vector<float> y_pred = predict(x_test);
	for(size_t i=0; i<y_pred.size(); i++) {
		float error = y_test[i] - y_pred[i];
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



