#include <iostream>
#include "MultipleLinearRegression.h"

MultipleLinearRegression::MultipleLinearRegression() {}

void MultipleLinearRegression::fit(std::vector<std::vector<float>> &X_ip, std::vector<float> &Y_ip) {
	size_t nrows = X_ip.size();
	size_t ncols = X_ip[0].size();

	X = Eigen::MatrixXf(nrows, ncols+1);              // use this method to resize an eigen matrix;
	Y = Eigen::VectorXf(nrows);                      // use (nrows, 1) to ensure column vector

	X.col(0) = Eigen::VectorXf::Ones(nrows);

	for(size_t row=0; row<nrows; row++)
		for(size_t col=0; col<ncols; col++)
			X(row, col+1) = X_ip[row][col];				  // X matrix, with 0th column as 1

	for(size_t i=0; i<Y_ip.size(); i++)
		Y(i) = Y_ip[i];								  // Y matrix containing target column
}

void MultipleLinearRegression::train() {
	// Beta is (n x m) as X is (m * n)
	Beta = (X.transpose() * X).inverse() * X.transpose() * Y;      // (m x n) * (n * m) * (m * n) * (n x 1)
	E = Y - X * Beta;                                              // (n * 1)
}

void MultipleLinearRegression::print_values() {
	for(int i=0; i<Beta.rows(); i++)
		std::cout << "B" << i << ": " << Beta(i) << "\n";
}

std::vector<float> MultipleLinearRegression::predict(std::vector<std::vector<float>> &X) {
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
			X_pred(row, col+1) = X[row][col];				  // X matrix, with 0th column as 1

	Eigen::VectorXf Y_pred = X_pred * Beta;

	std::vector<float> result(nrows);
	for(Eigen::Index i=0; i<nrows; i++)
		result[i] = Y_pred[i];

	return result;
}


