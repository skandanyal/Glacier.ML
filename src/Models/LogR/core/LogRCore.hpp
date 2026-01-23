//
// Created by skandan-c-y on 1/23/26.
//

#ifndef GLACIER_ML_LOGRCORE_HPP
#define GLACIER_ML_LOGRCORE_HPP

#pragma once
#include <Eigen/Dense>

class LogRCore {
public:
    LogRCore (int n_features);

    void fit (
        Eigen::MatrixXf &X,
        Eigen::MatrixXf &Y,
        float lr,
        int iterations
    );

    Eigen::MatrixXf predict_proba (const Eigen::MatrixXf &X) const;
    Eigen::MatrixXi predict (const Eigen::MatrixXi &X) const;

private:
    Eigen::VectorXf beta;
};

#endif //GLACIER_ML_LOGRCORE_HPP


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
// - fit()
// - predict_proba()
// - predict()
//
// DOES NOT:
// - normalize data
// - log output
// - manage threads
// - read files
