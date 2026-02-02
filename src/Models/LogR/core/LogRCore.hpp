#ifndef GLACIER_ML_LOGRCORE_HPP
#define GLACIER_ML_LOGRCORE_HPP

#pragma once
#include <Eigen/Dense>

namespace Glacier::Core {
    class LogRCore {

    private:
        // model parameters
        Eigen::VectorXf beta_;             // (p x 1)
        Eigen::VectorXf z_;                // (n x 1)
        Eigen::VectorXf p_;                // (n x 1)
        Eigen::VectorXf F_x_pred_;           // (n x 1)
        Eigen::VectorXf P_x_pred_;           // (n x 1)
        Eigen::VectorXf delta_;              // (p x 1)

        // toggle
        bool is_trained_;


    public:
        LogRCore(long n_features);

        void train(const Eigen::MatrixXf &X,
            const Eigen::VectorXf &Y,
            float lr,
            int iterations);

        Eigen::VectorXi predict(const Eigen::MatrixXf &X,
                                float decision_boundary);

    private:
        Eigen::MatrixXf predict_proba(const Eigen::MatrixXf &X);
    };
}

#endif // GLACIER_ML_LOGRCORE_HPP

// INPUT:
// - X: (n × d) matrix, normalized, includes bias column
// - y: (n) vector, values ∈ {0,1}
//
// ASSUMES:
// - X.rows() == y.size()
// - X.col(0) == 1
// - No NaNs or Infs
//
// PROVIDES:
// - fit() / here, train()
// - predict_proba()
// - predict()
//
// DOES NOT:
// - normalize data
// - log output
// - manage threads
// - read files
