#include "LogRCore.hpp"

using namespace Glacier::Core;
LogRCore::LogRCore(long n_features) :
    beta_(n_features) {
    /*
     * Inputs:
     * - X: (n × d+1) matrix, normalized, includes bias column
     * - y: (n) vector, values ∈ {0,1}
     */
    beta_ = Eigen::VectorXf::Zero(n_features);
}

void LogRCore::train(const Eigen::MatrixXf &X,
        const Eigen::VectorXf &Y,
        const float lr,
        const int iterations)
{
    /*
    * future enhancements:
    * 1. L2 regularization
    * 2. Early stopping
    * 3. Eigen uses 1 thread, manual threading using OpenMP, GEMM using OpenBLAS
    */

    // sanity checks
    assert(X.rows() == Y.size());             // number of rows to be equal
    assert(z_.size() == X.rows());            // z_ to contain as many rows as X and Y
    assert(p_.size() == Y.size());            // p_ to contain as many rows as X and Y

    Eigen::Index nr = X.rows();

    for (int i=0; i<iterations; i++) {

        // x = X * beta -> logit function
        z_ = X * beta_;

        // p = sigmoid(z) = 1 / (1 + e ^ (-z))
        // Eigen style coding - still need to learn this properly :}
        z_ = z_.cwiseMin(50.0f).cwiseMax(50.0f);
        p_ = 1.0f / (1.0f + (-z_.array()).exp());

        // compute the gradient
        delta_ = X.transpose() * (p_ - Y) / nr;

        // updated beta -> beta = beta - lr * grad
        beta_ -= lr * delta_;
    }
}

Eigen::MatrixXf LogRCore::predict_proba(const Eigen::MatrixXf &X, const float decision_boundary) {}

Eigen::MatrixXi LogRCore::predict(const Eigen::MatrixXi &X, const float decision_boundary) {}
