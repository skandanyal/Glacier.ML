#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#pragma once
#include <vector>
#include <Eigen/Dense>

namespace Glacier::Models {
    class Logistic_Regression {
    private:
        Eigen::MatrixXf X;                  // (n x p)
        Eigen::VectorXf mean;               // (p x 1)
        Eigen::VectorXf std_dev;            // (p x 1)
        Eigen::VectorXf Beta;               // (p x 1)
        Eigen::VectorXf Y;                  // (n x 1)
        std::vector<std::string> labels;    // 2
        Eigen::VectorXf F_x;                // (n x 1)
        Eigen::VectorXf P_x;                // (n x 1)
        Eigen::VectorXf F_x_pred;           // (n x 1)
        Eigen::VectorXf P_x_pred;           // (n x 1)
        Eigen::MatrixXf Delta;              // (p x 1)
        int no_threads{};

    public:
        Logistic_Regression(std::vector<std::vector<float>> &x, std::vector<std::string> &y, int no_threads=0);
        void train(float alpha, int iterations);
        std::string predict(std::vector<float> &x_pred);
        std::vector<std::string> predict(std::vector<std::vector<float>>& x_test);
        void print_predict(std::vector<std::vector<float> > &x_test, std::vector<std::string> &y_val);
        // mimic predict_proba from sklearn
        void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
        void print_Beta_values();

    private:
        static float sigmoid(float x);
    };
}

#endif //LOGISTICREGRESSION_HPP
