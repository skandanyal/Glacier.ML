//
// Created by skandan-c-y on 7/2/25.
//

#ifndef LOGISTICREGRESSIONFLOW_H
#define LOGISTICREGRESSIONFLOW_H

#include <vector>
#include <Eigen/Dense>

class Logistic_Regression {
private:
    Eigen::MatrixXf X;
    Eigen::VectorXf Beta;
    Eigen::VectorXf Y;
    Eigen::MatrixXf F_x;
    float P_x;

public:
    Logistic_Regression(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y);
    void train();
    float predict(std::vector<double> x);
    std::vector<float> predict(std::vector<std::vector<float>>& x_test);
    void analyze(std::vector<std::vector<double>>& x_test, std::vector<float>& y_test);
    void print_values();

private:
    float sigmoid();

};
#endif //LOGISTICREGRESSIONFLOW_H
