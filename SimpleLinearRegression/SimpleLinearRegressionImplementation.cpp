#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <boost/math/distributions/students_t.hpp>
#include "SimpleLinearRegressionFlow.h"
#include "utilities.h"


#define LOG_DEBUG(x, x_val) std::cout << "\033[35m[DEBUG] \033[0m" << x << ": " << x_val<< "\n"							// deeper info to be used during development
#define LOG_INFO(x) std::cout << "\033[36m[INFO]  \033[0m" << x << "\n";												// high level info while users are using it
#define LOG_TIME(task, duration) std::cout << "\033[32m[TIME]  \033[0m" << task << " took " << duration << " nanooseconds. \n";					// time taken

#if DEBUG_MODE
	#define LOG_ERROR(x) std::cerr << "[ERROR] " << x << " Exiting program here... \n"; std::exit(EXIT_FAILURE);		// errors and exits
#else
	#define LOG_ERROR(x)
#endif

// constructor
Simple_Linear_Regression::Simple_Linear_Regression(std::vector<float> &x, std::vector<float> &y) {
	if(y.size() != x.size()) {
		LOG_ERROR("Size of Y vector and X vector are not the same.\n");
	}
	this->y = y;
	this->x = x;
	this->n = y.size();

	m = 0.0, c = 0.0, x_test_mean = 0.0, x_test_size = 0.0;

	sum_y = std::reduce(y.begin(), y.end());
	sum_x = std::reduce(x.begin(), x.end());
}

/* train()
 * |_ calculate_m()
 * |_ calculate_c()
 */
void Simple_Linear_Regression::train() {
	auto train_start = std::chrono::high_resolution_clock::now();

	m = calculate_m();
	c = calculate_c();

	auto train_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(train_end - train_start);

	LOG_TIME("Training", duration.count());
	LOG_INFO("Model training is complete.\n");
}


float Simple_Linear_Regression::calculate_m() {
	// float ssx = 0.0, ssxy = 0.0;
	//
	// for(int i=0; i<n; i++) {
	// 	ssx += x[i] * x[i];
	// 	ssxy += x[i] * y[i];
	// }
	//
	// float numer = n * ssxy - sum_x * sum_y;
	// float denr = n * ssx - sum_x * sum_x;
	// return numer / denr;

	float x_mean = sum_x / n;
	float y_mean = sum_y / n;

	float num = 0.0, den = 0.0;
	for (int i = 0; i < n; i++) {
		num += (x[i] - x_mean) * (y[i] - y_mean);
		den += (x[i] - x_mean) * (x[i] - x_mean);
	}
	return num / den;
}

float Simple_Linear_Regression::calculate_c() {
	return (sum_y - m * sum_x) / n;
}

void Simple_Linear_Regression::print_values() const {
	LOG_INFO("Regression coefficients: ");
	std::cout << "m: " << m << "\nc: " << c << "\n\n";
}

// predict() for float x, returns float result
float Simple_Linear_Regression::predict(float a) {
	return m * a + c;
}

/* analyze(x_test, y_test)
 * |_ mean()
 * |_ predict(float x)
 * |_ hypothesis()
 * |_ print_confidence_intervals()
 */
void Simple_Linear_Regression::analyze(std::vector<float> &x_test, std::vector<float> &y_test) {
	if (x_test.size() <= 2) {
		LOG_ERROR("Too few test points. Degrees of freedom <= 0");
	}

	if (x_test.size() != y_test.size()) {
		LOG_ERROR("Test size does not match.");
	}

	x_test_size = x_test.size(), x_test_mean = Utils::mean(x_test);
	float mse = 0.0, rmse = 0.0, mae = 0.0, mape = 0.0;
	float rss = 0.0, sse_x = 0.0;																						// rss - residual sum of squares
																														// sse_x - sum of squares of errors in x

	for(size_t i=0; i<x_test_size; i++) {
		float y_pred = predict(x_test[i]);
		float y_error = y_test[i] - y_pred;

		rss += y_error * y_error;
		mae += std::abs(y_error);
		if (y_test[i] != 0) {
			mape += std::abs(y_error / y_test[i]);
		}
		sse_x += std::pow((x_test[i] - x_test_mean), 2);
	}
	mse = rss / x_test_size;
	rmse = std::sqrt(mse);
	mae = mae / x_test_size;
	mape = mape / x_test_size * 100;

	float res_var = rss / (x_test_size - 2);

	float SEm = std::sqrt(res_var / sse_x);
	float SEc = std::sqrt(res_var * ((1.0 / x_test_size) + (std::pow(x_test_mean, 2) / sse_x)));

	float t_statistic_m = m / SEm; LOG_DEBUG("t_static_m", t_statistic_m);
	float t_statistic_c = c / SEc; LOG_DEBUG("t_static_c", t_statistic_c); std::cout << "\n";

	LOG_INFO("Evaluation metrics: ");
	std::cout << "RMSE: " << rmse << "\n";
	std::cout << "MAE: " << mae << "\n";
	std::cout << "MAPE: " << mape << "%\n";
	std::cout << "\n";

	hypothesis("Slope", t_statistic_m, (int) x_test_size-2);
	print_confidence_intervals("Slope", t_statistic_m, SEm);

	hypothesis("Intercept", t_statistic_c, (int) x_test_size-2);
	print_confidence_intervals("Intercept", t_statistic_c, SEc);
}

void Simple_Linear_Regression::hypothesis(std::string param, float t_statistic, int dof) {
	LOG_INFO("Hypothesis testing: " + param);
	double p_value = Utils::get_p_value(t_statistic, dof);
	LOG_DEBUG("p_value of " + param, p_value);

	if (p_value < 0.05) {
		std::cout << "[RESULT] "<< param << " is statistically significant in 95% confidence level. \n";
	} else {
		std::cout << "[RESULT] "<< param << " is NOT statistically significant in 95% confidence level. \n";
	}

	if (p_value < 0.01) {
		std::cout << "[RESULT] "<< param << " is statistically significant in 99% confidence level. \n";
	} else {
		std::cout << "[RESULT] "<< param << " is NOT statistically significant in 99% confidence level. \n";
	}

	std::cout << "\n";
}

void Simple_Linear_Regression::print_confidence_intervals(const std::string& param, float measure, float SE) {
	LOG_INFO("Confidence intervals of " + param);

	std::vector<std::pair<std::string, float>> confidence_levels = {
		{"95%", 0.05},
		{"99%", 0.01}
	};
	boost::math::students_t dist(x_test_size - 2);  // global x_test_size assumed

	for (auto &[label, alpha] : confidence_levels) {
		double t_critical = boost::math::quantile(boost::math::complement(dist, alpha / 2));
		float margin = t_critical * SE;
		float lower = measure - margin;
		float upper = measure + margin;

		std::cout << "Confidence interval for " << label << " confidence level: [" << lower << ", " << upper << "]\n";
	}
	std::cout << "\n";
}

void Simple_Linear_Regression::predict_print(const std::vector<float>& x_pred) {
	LOG_INFO("Predicted values: ");
	std::cout << "X values:\tY values:\n";
	for(float i : x_pred) {
		std::cout << i << "\t\t" << predict(i) << "\n";
	}
	std::cout << "\n";
}

// predict() for std::vector<float> x, returns a std::vector<float> result
std::vector<float> Simple_Linear_Regression::predict(const std::vector<float> &x_pred) {
	std::vector<float> result(x_pred.size(), 0.0);
	size_t x_pred_size = x_pred.size();
	for(size_t i=0; i<x_pred_size; i++) {
		result[i] = predict(x_pred[i]);
	}

	return result;
}

