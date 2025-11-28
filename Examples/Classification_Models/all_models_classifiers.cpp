#include "Models/KNNClassifier.hpp"
#include "Models/LogisticRegression.hpp"
#include "Models/SVMClassifierFlow.hpp"
#include "Utils/utilities.hpp"

int main() {
    std::vector<std::vector<float>> X, X_t;
    std::vector<std::string> y, y_t;

    Glacier::Utils::read_csv_c("../Datasets/cs_datasets/cs-10000.csv", X, y, true);
    Glacier::Utils::read_csv_c("../Datasets/cs_datasets/cs_val_2.csv", X_t, y_t, true);

    std::vector<std::vector<float>> X_p = {
        {0.002020161, 43, 0, 0.228941685, 12500.0, 9, 0, 2, 0, 1.0}
    };
    std::vector<std::string> y_p = {"was_in_trouble"};

    // Initialize models
    Glacier::Models::KNNClassifier knnc(X, y);
    Glacier::Models::SVMClassifier svmc(X, y);
    Glacier::Models::Logistic_Regression logR(X, y);

    // Hyperparameters
    float alpha = 0.0001f;
    int iterations = 2000;

    float lambda = 0.5f;
    int epochs = 200;

    std::vector<std::string> distance_metric_str =
    {"manhattan", "euclidean", "minkowski"};
    int p = 3;

    // Train
    knnc.train(100, distance_metric_str[2], p);
    svmc.train(lambda, epochs);
    logR.train(alpha, iterations);

    // Predict sample
    auto knn_pred = knnc.predict(X_p);
    auto svm_pred = svmc.predict(X_p);
    auto log_pred = logR.predict(X_p);

    // Analysis on validation set
    knnc.analyze_2_targets(X_t, y_t);
    svmc.analyze_2_targets(X_t, y_t);
    logR.analyze_2_targets(X_t, y_t);

    return 0;
}
