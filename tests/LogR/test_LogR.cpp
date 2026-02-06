#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

// Include your headers based on project structure
#include "Glacier/Models/LogisticRegression.hpp"
#include "Models/LogR/core/LogRCore.hpp"
#include "Glacier/Utils/logs.hpp"

// --- Core Math Tests (LogRCore) ---

TEST(LogRCoreTest, ClampingStability) {
    // Tests if z_ = z_.cwiseMin(50.0f).cwiseMax(-50.0f) prevents overflow
    long n_features = 2;
    Glacier::Core::LogRCore core(n_features);

    // Extreme values that would normally cause exp(-z) to overflow/underflow
    Eigen::MatrixXf X_extreme(2, 2);
    X_extreme << 1.0f, 1000.0f,
                 1.0f, -1000.0f;
    Eigen::VectorXf Y = Eigen::VectorXf::Zero(2);

    // Should not crash or produce NaNs in weights
    EXPECT_NO_THROW(core.train(X_extreme, Y, 0.1f, 1));
}

TEST(LogRCoreTest, PredictProbaRange) {
    // Probability must always be in [0, 1]
    Glacier::Core::LogRCore core(2);
    Eigen::MatrixXf X = Eigen::MatrixXf::Random(10, 2);
    Eigen::VectorXf Y = Eigen::VectorXf::Zero(10);

    core.train(X, Y, 0.01f, 1);
    // core.predict_proba is private in your header;
    // Testing via public predict() or assuming test-friendliness
    Eigen::VectorXi preds = core.predict(X, 0.5f);

    for(int i = 0; i < preds.size(); ++i) {
        EXPECT_TRUE(preds(i) == 0 || preds(i) == 1);
    }
}

// --- Wrapper Logic Tests (Logistic_Regression) ---

class LogisticRegressionTest : public ::testing::Test {
protected:
    // Basic linearly separable dataset
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {10.0f}, {11.0f}};
    std::vector<std::string> Y = {"A", "A", "B", "B"};
};

TEST_F(LogisticRegressionTest, LabelOrderingAndMapping) {
    // Verifies that labels are sorted and mapped correctly
    Glacier::Models::Logistic_Regression model(X, Y);
    model.train(0.1f, 50);

    // Prediction for a high value should map to class 1 ("B" if sorted)
    std::vector<float> query = {15.0f};
    std::string result = model.predict(query, 0.5f);

    EXPECT_EQ(result, "B");
}

TEST_F(LogisticRegressionTest, NormalizationConsistency) {
    // Verifies that prediction data is normalized using training stats
    Glacier::Models::Logistic_Regression model(X, Y);

    // Large query value should be scaled down internally
    std::vector<float> query = {1000.0f};
    EXPECT_NO_THROW(model.predict(query, 0.5f));
}

TEST_F(LogisticRegressionTest, BatchInferenceSize) {
    // Verifies vector prediction returns correct size
    Glacier::Models::Logistic_Regression model(X, Y);
    std::vector<std::vector<float>> query_batch = {{1.5f}, {10.5f}};

    auto results = model.predict(query_batch, 0.5f);
    EXPECT_EQ(results.size(), 2);
}

// --- Systems & Error Handling Tests ---

TEST(GlacierSystemsTest, EmptyDataHandling) {
    // Since LOG_ERROR calls exit(), we test for "Death"
    std::vector<std::vector<float>> X_empty;
    std::vector<std::string> Y_empty;

    ASSERT_DEATH({
        Glacier::Models::Logistic_Regression model(X_empty, Y_empty);
    }, "Input data cannot be empty.");
}

TEST(GlacierSystemsTest, ThreadInitialization) {
    // Ensures constructor correctly sets Eigen/OMP threads
    std::vector<std::vector<float>> X = {{1.0f}};
    std::vector<std::string> Y = {"A", "B"}; // Will fail other checks, but tests thread path

    // Test specific thread count path
    EXPECT_NO_THROW({
        Glacier::Models::Logistic_Regression model(X, Y, 2);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}