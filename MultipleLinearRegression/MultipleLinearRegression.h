#ifndef MULTIPLELINEARREGRESSION_H
#define MULTIPLELINEARREGRESSION_H

#include <Eigen/Dense>
#include <vector>

class MultipleLinearRegression {
private:
    Eigen::MatrixXf X;
    Eigen::MatrixXf Beta;
    Eigen::MatrixXf Y;
    Eigen::MatrixXf E;

public:
    MultipleLinearRegression();
    void fit(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i);
    void train();
    void print_values();
    std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
    float R_squared();
    void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);
};

#endif // MULTIPLELINEARREGRESSION_H
