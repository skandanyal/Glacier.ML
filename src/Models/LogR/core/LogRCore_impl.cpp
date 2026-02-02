#include "LogRCore.hpp"

using namespace Glacier::Core;
LogRCore::LogRCore(long n_features) :
    beta_(n_features), is_trained_(false) {
    /*
     * Inputs:
     * - X: (n × d+1) matrix, normalized, includes bias column
     * - y: (n) vector, values ∈ {0,1}
     */
    beta_ = Eigen::VectorXf::Zero(n_features);
    Eigen::initParallel();
}

/*
    * extension ideas:
    * 1. L2 regularization
    * 2. Early stopping
    * 3. Eigen uses 1 thread, manual threading using OpenMP, GEMM using OpenBLAS
    */

void LogRCore::train(const Eigen::MatrixXf &X,
        const Eigen::VectorXf &Y,
        const float lr,
        const int iterations)
{

    // sanity checks
    assert (X.rows() == Y.size());             // number of rows to be equal
    assert (z_.size() == X.rows());            // z_ to contain as many rows as X and Y
    assert (p_.size() == Y.size());            // p_ to contain as many rows as X and Y
    assert (X.rows() >= 2);                    // at least two rows

    const Eigen::Index n = X.rows();

    z_.resize(n);
    p_.resize(n);
    delta_.resize(X.cols());

    for (int i=0; i<iterations; i++) {

        // x = X * beta -> logit function
        z_ = X * beta_;

        // p = sigmoid(z) = 1 / (1 + e ^ (-z))
        // Eigen style coding - still need to learn this properly :}
        z_ = z_.cwiseMin(50.0f).cwiseMax(-50.0f);
        p_ = 1.0f / (1.0f + (-z_.array()).exp());

        // compute the gradient
        delta_ = X.transpose() * (p_ - Y) / n;

        // updated beta -> beta = beta - lr * grad
        beta_ -= lr * delta_;
    }

    is_trained_ = true;
}

Eigen::MatrixXf LogRCore::predict_proba(const Eigen::MatrixXf &X)
{
    assert (X.cols() == beta_.size());
    assert (is_trained_ == true);

    Eigen::VectorXf z = X * beta_;
    z = z.cwiseMin(50.0f).cwiseMax(-50.0f);

    Eigen::MatrixXf p = 1.0f / (1.0f + (-z.array()).exp());
    return p;
}

Eigen::VectorXi LogRCore::predict(const Eigen::MatrixXf &X,
    const float decision_boundary)
            {

    assert (is_trained_ == true);
    assert (X.cols() == beta_.size());

    Eigen::VectorXf p = predict_proba(X);

    Eigen::VectorXi y_hat(X.rows());
    for (Eigen::Index i = 0; i < p.size(); ++i) {
        y_hat(i, 0) = (p[i] >= decision_boundary) ? 1 : 0;
    }
    return y_hat;
}
