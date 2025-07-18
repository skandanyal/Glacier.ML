//
// Created by skandan-c-y on 7/2/25.
//

#include "LogisticRegressionFlow.h"
#include "utilities.h"
#include <iostream>

int main() {

  std::vector<std::vector<float>> x_train, x_test, x_val;
  std::vector<std::string> y_train, y_test, y_val;

  Utils::read_csv("__path_to_training_dataset__", x_train, y_train, true);
  Utils::read_csv("__path_to_testing_dataset__", x_test, y_test, true);
  Utils::read_csv("__path_to_validation_dataset__", x_val, y_val, true);

  Logistic_Regression iceberg(x_train, y_train);

  // hyperparameters (example values)
  float alpha = 0.015;
  int iterations = 2000;

  iceberg.train(alpha, iterations);
  iceberg.print_predict(x_test, y_test);
  iceberg.analyze(x_val, y_val);
  iceberg.print_Beta_values();

  return 0;
}
