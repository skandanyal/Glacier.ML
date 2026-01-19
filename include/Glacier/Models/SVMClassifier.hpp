#ifndef SVMCLASSIFIERFLOW_HPP
#define SVMCLASSIFIERFLOW_HPP

#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace Glacier::Models {
    class SVMClassifier {
    private:
        Eigen::MatrixXf X;          // (n x p+1), 1 for bias
        Eigen::VectorXf w;          // (1 x p)
        Eigen::VectorXf Y;          // (N X 1)
        Eigen::Index nrows{};
        Eigen::Index ncols{};
        Eigen::VectorXf mean;
        Eigen::VectorXf std_dev;
        std::vector<std::string> labels;    // 2
        float lambda{};
        int epochs{};
        int no_threads{};


    public:
        SVMClassifier(std::vector<std::vector<float>> &x, std::vector<std::string> &y, int no_threads=0);
        void train(float lambda, int epochs);
        std::string predict(std::vector<float> &x_pred);
        std::vector<std::string> predict(std::vector<std::vector<float>>& x_pred);
        void print_predict(std::vector<std::vector<float>>& x_pred, std::vector<std::string> &y_pred);
        // mimic predict_proba from sklearn
        void analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);
        void show_final_weights();
        void show_support_vectors(); // no. of features = no. of SVs
    };
}

#endif //SVMCLASSIFIERFLOW_HPP
