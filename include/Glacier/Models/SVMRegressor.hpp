#ifndef SVMREGRESSORFLOW_HPP
#define SVMREGRESSORFLOW_HPP

#include <vector>
#include <Eigen/Dense>

namespace Glacier::Models {
    class SVMRegressor {
        // the {} braces are for the constructor to initialize these variables outside the constructor
        Eigen::MatrixXf X;                      // (n x p)
        Eigen::VectorXf Y;                      // (n x 1)
        Eigen::VectorXf mean;                   // (p x 1)
        Eigen::VectorXf std_dev;                // (p x 1)
        Eigen::Index nrows{}, ncols{};
        Eigen::VectorXf w;
        std::vector<int> indices;
        int no_threads{};

    public:
        SVMRegressor(std::vector<std::vector<float>> &x_i, std::vector<float> &y_i, int no_threads=0);
        void train(float lambda, float epsilon, int epochs);
        float predict(std::vector<float> &x_pred);
        std::vector<float> predict(std::vector<std::vector<float>> &x_pred);
        void print_predict(std::vector<std::vector<float>> &x_pred, std::vector<float> &y_pred);
        void analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test);
        // void show_final_weights();
        // void show_support_vectors(); // no. of features = no. of SVs

    private:
        static float R2_score(std::vector<float> &actual, std::vector<float> &predicted);
    };
}

#endif //SVMREGRESSORFLOW_HPP
