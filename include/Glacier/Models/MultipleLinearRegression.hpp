#ifndef MULTIPLELINEARREGRESSION_HPP
#define MULTIPLELINEARREGRESSION_HPP

#pragma once
#include <Eigen/Dense>
#include <vector>


namespace Glacier {
    class Multiple_Linear_Regression {
    private:
        Eigen::MatrixXf X;
        Eigen::VectorXf Beta;
        Eigen::VectorXf Y;
        Eigen::MatrixXf E;
        int no_threads{};

    public:
        Multiple_Linear_Regression(std::vector<std::vector<float>> &X_i, std::vector<float> &Y_i, int no_threads=0);
        void train();
        std::vector<float> predict(std::vector<std::vector<float>> &X_pred);
        float predict(std::vector<float> &X_pred);
        void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);
        void print_Rcoeff_values();

    private:
        float R_squared();
    };
};

// t-test is for testing individual params, f-test is for testing the entire model

#endif //MULTIPLELINEARREGRESSION_HPP

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

