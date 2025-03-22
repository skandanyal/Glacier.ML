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

public:
	Simple_Linear_Regression(std::vector<float> &x, std::vector<float> &y);
	void train();
	void test(std::vector<float> &x_test, std::vector<float> &y_test);
	void print_values() const;
	float predict(float a);
	void predict_print(std::vector<float> x_predict);
	std::vector<float> predict(std::vector<float> x_predict);

private:
	float calculate_m();
	float calculate_c();
};



#endif //TESTING_SLR_H
