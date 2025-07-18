# Logistic Regression

## Variables used
| Variable name used | Variable description | Size of matrix | Remarks |
|:------------------:|:--------------------:|:--------------:| :------:|
| `Y` | Target matrix | `n x 1` | Before adding column of 1's |
| `X` | Feature matrix | `n x p` | before adding column of 1's |
| `Beta` | Regression coefficients | `p x 1` | - |
| `F_x` | Logit values | `n x 1` | Size never gets modified |
| `P_x` | Sigmoid values | `n x 1` | Size never gets modified |
| `Delta` | Gradient update direction | `p x 1` | - |
| `mean` | Mean vector | `n x 1` | Size never gets modified |
| `std_dev` | Standard deviation vector | `n x 1` | Size never gets modified |
| `labels` | an `std::vector` containing <br>the labels of targets | 2 | - |
| `F_x_pred` | Logit values of prediction <br>dataset | `n' x 1` | Size never gets modified |
| `P_x_pred` | Sigmoid values of prediction <br> dataset | `p x 1` | Size ever gets modified |

## Functions available to be used
| Function call | Return type | Description |
|:-------------:|:-----------:|:-----------:|
| Logistic_Regression (std::vector<std::vector<float>> &x, std::vector<<std::string>> &y) | `void` | Constructor for the class Logistic_Regression |
| train (float alpha, int iterations) | `void` | Begins training the model |
| predict (std::vector<float> &x_pred) | `std::string` | Returns the most probable outcome for the given vector |
| predict (std::vector<std::vector<float>> &x_pred) |`std::vector<std::string>>`| Returns a std::vector of the most probable outcomes for the given matrix |
| print_predict (std::vector<std::vector<float>> &x_val, std::vector<<std::string>> &y_test) | `void` | Prints the most probable outcomes along with the actual outcomes |
| analyze (std::vector<std::vector<float>> &x_test, std::vector<<std::string>> &y_test) | `void` | Prints the various performace metrics based on the given dataset |
| print_Beta_values () | `void` | Prints all the regression coefficients of the model |
| sigmoid (float x) <br>(Private Function) | `float` | Applies sigmoid function to the given input |

## Theory
* Update it someday later.


<date, time - 18 July, 12:56 pm>
