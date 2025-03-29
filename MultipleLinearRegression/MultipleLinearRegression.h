#ifndef MULTIPLELINEARREGRESSION_H
#define MULTIPLELINEARREGRESSION_H

#include <vector>
#include <deque>
#include <Eigen/Dense>

class MultipleLinearRegression {
private:
	Eigen::MatrixXf<Eigen::Dynamic, Eigen::Dynamic> X;
	Eigen::MatrixXf<Eigen::Dynamic, Eigen::Dynamic> Beta;
	Eigen::MatrixXf<Eigen::Dynamic, 1> Y;
	Eigen::MatrixXf<Eigen::Dynamic, 1> E;

public:
	MultipleLinearRegression();
	void fit(std::vector<std::vector<float>> &X, std::vector<float> &Y);
	void train();
	void print_values();
	std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
};

#endif //MULTIPLELINEARREGRESSION_H
