#ifndef TESTING_SLR_H
#define TESTING_SLR_H

#include <vector>

class Simple_Linear_Regression {
private:
	float m, c;
	std::vector<float> y;
	std::vector<float> x;
	int n;
	float sum_x, sum_y;
	float x_test_size, x_test_mean;

public:
	Simple_Linear_Regression(std::vector<float> &x, std::vector<float> &y);
	// void read_csv(const std::string& filename, std::vector<float>& x, std::vector<float>& y, bool has_header = true);
	void train();
	void print_values() const;
	float predict(float a);
	void predict_print(const std::vector<float>& x_predict);
	std::vector<float> predict(const std::vector<float> &x_predict);
	void analyze(std::vector<float> &x_test, std::vector<float> &y_test);

private:
	float calculate_m();
	float calculate_c();
	void hypothesis(std::string param, float t_statistic, int dof);
	void print_confidence_intervals(const std::string& param, float measure, float SE);
	// float mean(std::vector<float> x);
	// double get_p_value(double t_stat, int dof);
};



#endif //TESTING_SLR_H
