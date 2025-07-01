#ifndef MULTIPLELINEARREGRESSION_H
#define MULTIPLELINEARREGRESSION_H

#include <Eigen/Dense>
#include <vector>

class Multiple_Linear_Regression {
private:
    Eigen::MatrixXf X;
    Eigen::VectorXf Beta;
    Eigen::VectorXf Y;
    Eigen::MatrixXf E;

public:
    Multiple_Linear_Regression(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i);
    void fit(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i);
    void train();
    void print_Rcoeff_values();
    std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
    float predict(std::vector<float> &X_pred);
    void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);

private:
    float R_squared();
};

#endif // MULTIPLELINEARREGRESSION_H

/*
 * TEMPLATE:
 * ---------
 * constructor()
 * train()
 * print_values()
 * predict(single value) return a singular answer
 * predict(multiple values) return multiple answers
 * analyze()
 * /

/*
 * data gets fit using constructor. so remive fit()
 *
 */

