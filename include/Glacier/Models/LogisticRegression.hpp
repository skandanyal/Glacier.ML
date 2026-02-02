#ifndef LOGISTICREGRESSION_HPP
#define LOGISTICREGRESSION_HPP

#pragma once
#include "Models/LogR/core/LogRCore.hpp"
#include <vector>
#include <Eigen/Dense>


namespace Glacier::Models {
    class Logistic_Regression {
    private:

        // inputs
        Eigen::MatrixXf X_;                  // (n x p)
        Eigen::VectorXf Y_;                  // (n x 1)
        std::vector<std::string> labels_;    // 2

        // size of X_
        long nrows_{}, ncols_{};

        // hyperparameters
        float lr_{};
        int iterations_{};
        float decision_boundary_{};
        int no_threads_{};

        // for normalization
        Eigen::VectorXf mean_;               // (p x 1)
        Eigen::VectorXf std_dev_;            // (p x 1)


    public:
        Logistic_Regression(std::vector<std::vector<float>> &x,
            std::vector<std::string> &y,
            int no_threads=0
            );

        void train(float alpha, int iteration);

        std::string predict(std::vector<float> &x_pred,
            float decision_boundary
            );

        std::vector<std::string> predict(std::vector<std::vector<float>>& x_pred,
            float decision_boundary
            );

    private:
        Glacier::Core::LogRCore core_;
    };
}

#endif //LOGISTICREGRESSION_HPP
