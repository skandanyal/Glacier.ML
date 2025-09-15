//
// Created by skandan-c-y on 7/2/25.
//

#include "../Models/LogisticRegression.hpp"
#include "utilities.hpp"
#include <iostream>

int main() {

  std::vector<std::vector<float>> x_train_500, x_train_1000, x_train_5000,
      x_train_10000, x_train_50000, x_train_100000, x_train_140000, x_train,
      x_test, x_val_cs, x_val;
  std::vector<std::string> y_train_500, y_train_1000, y_train_5000,
      y_train_10000, y_train_50000, y_train_100000, y_train_140000, y_train,
      y_test, y_cal_cs, y_val;

  Utils::read_csv("../Datasets/credit_scores/cs-140000.csv",
                  x_train_140000, y_train_140000);
  Logistic_Regression iceberg_140000(x_train_140000, y_train_140000);


  // hyperparameters
  float alpha = 0.0001;
  int iterations = 2000;


  iceberg_140000.train(alpha, iterations);
  // iceberg.print_Beta_values();
  // Logistic_Regression iceberg(x_train, y_train);
  // iceberg_test.print_predict(x_test, y_test);
  //
  // iceberg.train(alpha, iterations);
  // iceberg.analyze(x_val, y_val);

  return 0;
}
