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
        const int iterations) {

    float nr = X.rows();
    Eigen::Index nc = X.cols();

    for (int i=0; i<iterations; i++) {

        // x = X * beta -> logit function
        z_ = X * beta_;

        // p = sigmoid(z) = 1 / (1 + e ^ (-z))
        for (Eigen::Index col = 0; col < nc-1; col++) {
            float z = std::clamp(z_(col), -50.0f, 50.0f);
            p_(col) = 1 / (1 + exp(-1 * z));
        }

        // compute the gradient
        delta_ = X.transpose() * (p_ - Y) / nr;

        // updated beta -> beta = beta - lr * grad
        beta_ -= lr * delta_;
    }
}

Eigen::MatrixXf LogRCore::predict_proba(const Eigen::MatrixXf &X, const float decision_boundary) {}

Eigen::MatrixXi LogRCore::predict(const Eigen::MatrixXi &X, const float decision_boundary) {}
