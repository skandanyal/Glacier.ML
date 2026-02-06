#include "Glacier/Utils/utilities.hpp"
#include "Glacier/Utils/logs.hpp"
#include "Glacier/Models/LogisticRegression.hpp"
#include "Models/LogR/core/LogRCore.hpp"
#include <map>

using namespace Glacier;

// constructor
Models::Logistic_Regression::Logistic_Regression
        (std::vector<std::vector<float>> &X_i,
        std::vector<std::string> &Y_i,
        int no_threads) :
    nrows_(static_cast<long>(X_i.size())),
    ncols_(static_cast<long>(X_i[0].size())),        // ncols = no. of cols before adding the bias colm
    X_(X_i.size(),  X_i[0].size() + 1),                    // +1 for the bias colm.
    Y_(X_i.size() + 1),
    mean_(X_i[0].size()),
    std_dev_(X_i[0].size()),
    core_(X_i[0].size() + 1)
{

    // the only job here is to prepare the X and Y matrices for the train fn to work upon

    // set number of threads as given by the user. else, use half as many available
    if (no_threads == 0) {
        omp_set_num_threads(omp_get_max_threads()/2);
        Eigen::setNbThreads(omp_get_max_threads()/2);
    } else {
        no_threads_ = no_threads;
        omp_set_num_threads(no_threads_);
        Eigen::setNbThreads(no_threads_);
    }
    LOG_DEBUG("Number of threads", no_threads_);

    // nrows_, ncols_
    LOG_DEBUG("Number of rows in X before adding the bias column", nrows_);
    LOG_DEBUG("Number of columns in X before adding the bias column", ncols_);

    // check for empty dataset
    if (X_i.empty() || Y_i.empty()) {
        LOG_ERROR("Input data cannot be empty.");
    }

    // check for inconsistency in the dataset
    for (auto &row : X_i) {
        if (row.size() != X_i[0].size()) {
            LOG_ERROR("Row sizes not consistent.");
        }
    }

    // check for infinite values in the dataset
    for (size_t i = 0; i < X_i.size(); i++) {
        for (size_t j = 0; j < X_i[i].size(); j++) {
            if (!std::isfinite(X_i[i][j]))
                std::cout << "[BAD VALUE] X_i[" << i << "][" << j << "] = " << X_i[i][j] << "\n";
        }
    }

    // bias column in X_
    X_.col(0) = Eigen::VectorXf::Ones(nrows_);

    // populating Eigen X matrix with the data
#pragma omp parallel for shared (X_, X_i)
    for (Eigen::Index col = 0; col < ncols_; col++)
        for (Eigen::Index row = 0; row < nrows_; row++)
            // X matrix, with 0th column as 1
            X_(row, col + 1) = X_i[row][col];
    LOG_DEBUG("Number of rows in x_train", X_.rows());
    LOG_DEBUG("Number of cols in x_train", X_.cols());
    std::cout << "\n";

    // normalize X
    // mean_ and std_dev_ have been resized already
#pragma omp parallel for shared (mean_, std_dev_, X_)
    for (long colm = 0; colm < ncols_; colm++) {
        mean_(colm) = X_.col(colm + 1).mean();
        std_dev_(colm) =
            std::sqrt((X_.col(colm + 1).array() - mean_(colm)).square().sum() / static_cast<float>(X_.rows()));

        if (std_dev_(colm) == 0)
            std_dev_(colm) = 1e-8;

        // normalization
        X_.col(colm + 1) = (X_.col(colm + 1).array() - mean_(colm)) / std_dev_(colm);
    }
    // X_ is now ready to be used:
    // X_1[0] is the bias column, rest are all normalized featured

    // initialize Y
    std::map<std::string, bool> seen;
    for (const std::string& target : Y_i)
        seen[target] = true;
    for (auto &[key, _] : seen)
        labels_.push_back(key);

    // check for the number of target classes
    if (labels_.size() < 2) {
        LOG_ERROR("Less than two classification classes detected. Binary classification requires the dataset to have two"
                  " target classes.");
    } else if (labels_.size() > 2) {
        LOG_ERROR("More than two classification classes detected. Binary classification requires the dataset to have two"
                  " target classes.");
    }

    // the labels are organized in the ascending order
    if (labels_[0] > labels_[1]) std::swap(labels_[0], labels_[1]);

    // populate the Eigen Y matrix with the data
    // LABELS LOGIC DEFINED HERE!!!!!!!!!!!!!
    for (auto i = 0; i < Y_i.size(); i++) {
        if (Y_i[i] == labels_[0]) Y_(i) = 0;
        else if (Y_i[i] == labels_[1]) Y_(i) = 1;
    }
    // Y_ is ready to be used
    LOG_DEBUG("Size of labels", labels_.size());
    LOG_DEBUG("Number of rows in y_train", Y_.rows());
    LOG_DEBUG("Number of cols in y_train", Y_.cols());
    std::cout << "\n";
}

void Models::Logistic_Regression::train(
    const float lr,
    const int iteration)
{

    lr_ = lr;
    iterations_ = iteration;

    /////////////////////// training loop ////////////////////////////

    core_.train(X_, Y_, lr_, iterations_);

    //////////////////////////////////////////////////////////////////

    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

std::string Models::Logistic_Regression::predict
    (std::vector<float>& x_pred,
        float decision_boundary)
{

    if (x_pred.size() != ncols_) {
        LOG_ERROR("Dataset size does not match number of columns");
    }

    if (x_pred.empty()) return "";

    Eigen::MatrixXf xp = Eigen::MatrixXf::Ones(1, ncols_+1);

    for (Eigen::Index col = 0; col < ncols_; col++) {
        xp(0, col+1) = (x_pred[col] - mean_(col)) / std_dev_(col);
    }

    Eigen::VectorXi result = core_.predict(xp, decision_boundary);
    return labels_[result[0]];
}

std::vector<std::string> Models::Logistic_Regression::predict
    (std::vector<std::vector<float>>& x_pred,
    float decision_boundary)
{

    if (x_pred[0].size() != ncols_) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    if (x_pred.empty()) return {};

    int pred_rows = static_cast<int>(x_pred.size());

    Eigen::MatrixXf xp;
    xp.resize(pred_rows, ncols_+1);
    xp.col(0) = Eigen::VectorXf::Ones(pred_rows);

    for (Eigen::Index row = 0; row < pred_rows; row++) {
        for (Eigen::Index col = 0; col < static_cast<int>(x_pred[0].size()); col++) {

            // first column of X matrix is the bias set to 1
            float val = x_pred[row][col];
            xp(row, col+1) = (val - mean_(col)) / std_dev_(col);
        }
    }

    //////////////////////////// prediction ///////////////////////////

    Eigen::VectorXi result = core_.predict(xp, decision_boundary);

    ///////////////////////////////////////////////////////////////////

    std::vector<std::string> ans (pred_rows);
    for (int i=0; i< pred_rows; i++) {
        if (result[i] == 0) ans[i] = labels_[0];
        else ans[i] = labels_[1];
    }

    return ans;
}