#include "Glacier/Utils/utilities.hpp"
#include "Glacier/Utils/logs.hpp"
#include "Glacier/Models/LogisticRegression.hpp"
#include "Models/LogR/core/LogRCore.hpp"
#include <map>

// initialize Beta
// for iter:
//     z = X * Beta
//     p = sigmoid(z)
//     loss = cross_entropy(p, Y)
//     grad = Xᵀ (p − Y) / n
//     Beta -= lr * grad
// until converged

// Golden core
// z = X * beta
// p = sigmoid(z)
// loss = cross_entropy(p, y)
// grad = Xᵀ (p − y) / n

using namespace Glacier;

// constructor
Models::Logistic_Regression::Logistic_Regression
        (std::vector<std::vector<float>> &X_i,
        std::vector<std::string> &Y_i,
        int no_threads) :
    nrows_(X_i.size()),
    ncols_(X_i[0].size()),
    X_(nrows_, ncols_ + 1), // +1 for the bias colm.
    Y_(nrows_),
    mean_(ncols_),
    std_dev_(ncols_),
    core_(ncols_ + 1)
{

    // the only job here is to prepare the X and Y matrices for the train fn to work upon

    // set number of threads as given by the user. else, use half as many available
    if (no_threads == 0) {
        omp_set_num_threads(omp_get_max_threads()/2);
    } else {
        omp_set_num_threads(no_threads);
    }
    LOG_DEBUG("Number of threads", threads);

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
    for (Eigen::Index row = 0; row < nrows_; row++)
        for (Eigen::Index col = 0; col < ncols_; col++)
            // X matrix, with 0th column as 1
            X_(row, col + 1) = X_i[row][col];
    LOG_DEBUG("Number of rows in x_train", X.rows());
    LOG_DEBUG("Number of cols in x_train", X.cols());
    std::cout << "\n";

    // normalize X
    // mean_ and std_dev_ have been resized already
    for (long colm = 0; colm < ncols_; colm++) {
        mean_(colm) = X_.col(colm + 1).mean();
        std_dev_(colm) = std::sqrt((X_.col(colm + 1).array() - mean_(colm)).square().sum() / X_.rows());

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
    for (auto i = 0; i < Y_i.size(); i++) {
        if (Y_i[i] == labels_[0]) Y_(i) = 0;
        else if (Y_i[i] == labels_[1]) Y_(i) = 1;
    }
    // Y_ is ready to be used
    LOG_DEBUG("Size of labels", labels.size());
    LOG_DEBUG("Number of rows in y_train", Y.rows());
    LOG_DEBUG("Number of cols in y_train", Y.cols());
    std::cout << "\n";
}

void Models::Logistic_Regression::train(
    const float lr,
    const int iterations)
{

    lr_ = lr;
    iterations_ = iterations;

    // training loop
    core_.train(X_, Y_, lr_, iterations_);
    LOG_DEBUG("Final loss at the end ", loss);
    std::cout << "\n";

    LOG_INFO("Model training is complete.");
    std::cout << "\n";
}

std::string Models::Logistic_Regression::predict(
    std::vector<float> &x_pred)
{
    LOG_INFO("Singular prediction initiated...");

    if (x_pred.size() + 1 != Beta.cols()) {
        LOG_ERROR(("Incompatible size of vector passed. Expected size: " + std::to_string(Beta.cols())));
    }

    ////////////////////// Prediction begins here //////////////////////

    Eigen::VectorXf x = Eigen::VectorXf::Zero(Beta.cols());
    x(0) = 1.0f;

    for (Eigen::Index i = 0; i < Beta.cols() - 1; i++)
        x(i+1) = x_pred[i];

    // normalizing the x matrix
    for (Eigen::Index i = 0; i < Beta.cols() - 1; i++)
        x(i+1) = (x(i+1) - mean(i)) / std_dev(i);

    float ans = x.dot(Beta);
    if (ans < 0.0f)
        return labels[0];
    return labels[1];

    ////////////////////// Prediction ends here //////////////////////
}

std::vector<std::string> Logistic_Regression::predict(std::vector<std::vector<float>>& X_test) {
    LOG_INFO("Block prediction initiated...");
    if (Beta.size() == 0) {
        LOG_ERROR("Train the data using train() before using predict().");
    }

    if (X_test[0].size() != (int) X.cols() - 1) {
        LOG_ERROR("Train and test dataset have different number of features.");
    }

    ////////////////////// Prediction begins here //////////////////////

    Eigen::Index nrows_ = X_test.size();
    Eigen::Index ncols_ = X_test[0].size();
    LOG_DEBUG("Number of rows in X_test", nrows_);
    LOG_DEBUG("Number of columns in X_test", ncols_);
    std::cout << "\n";

    if (mean.size() != ncols_) {
        LOG_ERROR("Mismatch in mean/std_dev size. Possible unnormalized feature set.");
    }

    Eigen::MatrixXf X_pred(nrows_, ncols_ + 1);
    X_pred.col(0) = Eigen::VectorXf::Ones(nrows_);

    for (Eigen::Index row = 0; row < nrows_; row++)
        for (Eigen::Index col = 0; col < ncols_; col++)
            X_pred(row, col + 1) = X_test[row][col];                                                                    // X matrix, with 0th column as 1

    // normalizing the X_pred matrix
    for (int colm = 0; colm < ncols_; colm++)
        X_pred.col(colm + 1) = (X_pred.col(colm + 1).array() - mean(colm)) / std_dev(colm);

    Eigen::VectorXf F_x_pred = X_pred * Beta;
    P_x_pred.resize(nrows_);
    for(int i=0; i < nrows_; i++){
        P_x_pred(i) = std::clamp(sigmoid(F_x_pred(i)), 1e-8f, 1.0f - 1e-8f);
    }

    std::vector<std::string> result(nrows_);
    for (Eigen::Index i = 0; i < nrows_; i++) {
        if (P_x_pred(i) < 0.0f) result[i] = labels[0];
        else result[i] = labels[1];
    }

    return result;

    ////////////////////// Prediction ends here //////////////////////
}

void Logistic_Regression::print_predict(std::vector<std::vector<float> > &x_test, std::vector<std::string> &y_val) {
    std::vector<std::string> y_test = predict(x_test);

    std::cout << "Predicted\t|\tActual\n";
    for (int i = 0; i < y_test.size(); i++) {
        std::cout << y_test[i] << "\t|\t" << y_val[i] << "\n";
    }
    std::cout << "\n";
}

void Logistic_Regression::analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test) {
    LOG_INFO("Analysis initiated...");
    std::vector<std::string> y_pred = predict(x_test);

    float tp = 0, fn = 0, fp = 0, tn = 0;

    for (size_t i=0; i<y_pred.size(); i++) {
        if (y_test[i] == labels[0] && y_pred[i] == labels[0])   // tp
            tp++;

        else if (y_test[i] == labels[0] && y_pred[i] == labels[1])    // fn
            fn++;

        else if (y_test[i] == labels[1] && y_pred[i] == labels[0])    // fp
            fp++;

        else if (y_test[i] == labels[1] && y_pred[i] == labels[1])    // tn
            tn++;
    }

    LOG_INFO("Confusion matrix: ");
    std::cout << "Actually " << labels[0] << ", Predicted " << labels[0] << ": " << tp << "\n";
    std::cout << "Actually " << labels[0] << ", Predicted " << labels[1] << ": " << fn << "\n";
    std::cout << "Actually " << labels[1] << ", Predicted " << labels[0] << ": " << fp << "\n";
    std::cout << "Actually " << labels[1] << ", Predicted " << labels[1] << ": " << tn << "\n";
    std::cout << "Total number of rows: " << x_test.size() << "\n\n";

    LOG_INFO("Evaluation Metrics: (Out of 1)");
    float accuracy = (tp + tn) / (tp + tn + fp + fn);
    std::cout << "Accuracy: " << accuracy << "\n";                                                                      // correct classifications / total classifications

    float recall = tp / (tp + fn);
    std::cout << "Recall: " << recall << "\n";                                                                          // true positives / true positives + false negatives

    float false_positive_rate = fp / (fp + tn);
    std::cout << "False positive rate: " << false_positive_rate << "\n";                                                // false positives / false positives + true negatives

    float precision = tp / (tp + fp);
    std::cout << "Precision: " << precision << "\n\n";                                                                  // true positives / true positives + false positives
}

void Logistic_Regression::print_Beta_values() {
    LOG_INFO("Regression coefficients: ");
    for (int i = 0; i < Beta.rows(); i++)
        std::cout << "B" << i << ": " << Beta(i) << "\n";
    std::cout << "\n";
}

inline float Logistic_Regression::sigmoid (float x) {
    float y = std::clamp(x, -100.0f, 100.0f);
    return 1 / (1 + std::exp(-1 * y));
}