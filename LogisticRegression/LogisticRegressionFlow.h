//
// Created by skandan-c-y on 7/2/25.
//

#ifndef LOGISTICREGRESSIONFLOW_H
#define LOGISTICREGRESSIONFLOW_H

#include <vector>
#include <Eigen/Dense>

class Logistic_Regression {
private:
    Eigen::MatrixXf X;              // (n x p)
    Eigen::VectorXf mean;           // (p x 1)
    Eigen::VectorXf std_dev;        // (p x 1)
    Eigen::VectorXf Beta;           // (p x 1)
    Eigen::VectorXf Y;              // (n x 1)
    std::vector<std::string> labels; // 2
    Eigen::VectorXf F_x;            // (n x 1)
    Eigen::VectorXf P_x;            // (n x 1)
    Eigen::VectorXf F_x_pred;       // (n x 1)
    Eigen::VectorXf P_x_pred;       // (n x 1)
    Eigen::MatrixXf Delta;          // (p x 1)

public:
    Logistic_Regression(std::vector<std::vector<float>> &x, std::vector<std::string> &y);
    void train(float alpha, int iterations);
    std::string predict(std::vector<float> &x_pred);
    std::vector<std::string> predict(std::vector<std::vector<float>>& x_test);
    void print_predict(std::vector<std::vector<float>>& x_val, std::vector<std::string> &y_val);;
    // mimic predict_proba from sklearn
    void analyze(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
    void print_Beta_values();

private:
    float sigmoid(float x);

};
#endif //LOGISTICREGRESSIONFLOW_H
