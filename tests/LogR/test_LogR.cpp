#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

#include "Glacier/Models/LogisticRegression.hpp"
#include "Models/LogR/core/LogRCore.hpp"
#include "Glacier/Utils/logs.hpp"

// --- Core Math Tests (LogRCore) ---

TEST(LogRCoreTest, ClampingStability) {
    // Tests if logit clamping prevents overflow in the sigmoid function
    long n_features = 2;
    Glacier::Core::LogRCore core(n_features);

    // Extreme values that would cause exp(-z) to overflow/underflow
    Eigen::MatrixXf X_extreme(2, 2);
    X_extreme << 1.0f, 1000.0f,
                 1.0f, -1000.0f;
    Eigen::VectorXf Y = Eigen::VectorXf::Zero(2);

    // train() internally calls the sigmoid which uses clamping
    EXPECT_NO_THROW(core.train(X_extreme, Y, 0.1f, 1));
}

TEST(LogRCoreTest, PredictRange) {
    // Verifies binary output constraints (0 or 1)
    Glacier::Core::LogRCore core(2);
    Eigen::MatrixXf X = Eigen::MatrixXf::Random(10, 2);
    Eigen::VectorXf Y = Eigen::VectorXf::Zero(10);

    core.train(X, Y, 0.01f, 1);

    // Note: predict_proba is private. We verify logic via the public predict() interface.
    Eigen::VectorXi preds = core.predict(X, 0.5f);

    for(int i = 0; i < preds.size(); ++i) {
        EXPECT_TRUE(preds(i) == 0 || preds(i) == 1);
    }
}

// --- Wrapper Logic Tests (Logistic_Regression) ---

class LogisticRegressionTest : public ::testing::Test {
protected:
    // Linearly separable dataset: 1.0/2.0 -> "A", 10.0/11.0 -> "B"
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {10.0f}, {11.0f}};
    std::vector<std::string> Y = {"A", "A", "B", "B"};
};

TEST_F(LogisticRegressionTest, LabelOrderingAndMapping) {
    // Verifies alphabetical label sorting (A=0, B=1) and prediction mapping
    Glacier::Models::Logistic_Regression model(X, Y, 1);
    model.train(0.1f, 100);

    // A high value (15.0) should yield "B"
    std::vector<float> query = {15.0f};
    std::string result = model.predict(query, 0.5f);

    EXPECT_EQ(result, "B");
}

TEST_F(LogisticRegressionTest, NormalizationConsistency) {
    // Verifies prediction path handles normalization using training statistics
    Glacier::Models::Logistic_Regression model(X, Y, 1);
    model.train(0.01f, 1);

    // Large value should not cause a crash or invalid memory access
    std::vector<float> query = {1000.0f};
    EXPECT_NO_THROW(model.predict(query, 0.5f));
}

TEST_F(LogisticRegressionTest, BatchInferenceSize) {
    Glacier::Models::Logistic_Regression model(X, Y, 1);
    std::vector<std::vector<float>> query_batch = {{1.5f}, {10.5f}};
    model.train(0.01f, 1);

    auto results = model.predict(query_batch, 0.5f);
    EXPECT_EQ(results.size(), 2);
}

// --- Systems & Error Handling Tests ---

// TEST(GlacierSystemsTest, EmptyDataHandling) {
//     // Tests program termination on empty input using GTest Death Tests
//     std::vector<std::vector<float>> X_empty;
//     std::vector<std::string> Y_empty;
//
//     // Matches the error string defined in LogR_impl.cpp
//     ASSERT_DEATH({
//         Glacier::Models::Logistic_Regression model(X_empty, Y_empty);
//     }, ".*Datasets cannot be left empty.*");
// }

TEST(GlacierSystemsTest, ThreadInitialization) {
    // Ensures thread pool scales without throwing exceptions
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}};
    std::vector<std::string> Y = {"A", "B"};

    EXPECT_NO_THROW({
        Glacier::Models::Logistic_Regression model(X, Y, 2);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}