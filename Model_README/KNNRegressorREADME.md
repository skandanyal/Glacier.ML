# KNN Regressor

## Variables used
| Variable name used |                 Variable description                  | Size of container |                Remarks                 |
|:------------------:|:-----------------------------------------------------:|:-----------------:|:--------------------------------------:|
|        `Y`         |                     Target matrix                     |      `n x 1`      |      Before adding column of 1's       |
|        `X`         |                    Feature matrix                     |      `n x p`      |      before adding column of 1's       |
|       `mean`       |                      Mean vector                      |      `n x 1`      |        Size never gets modified        |
|     `std_dev`      |               Standard deviation vector               |      `n x 1`      |        Size never gets modified        |
| `distance_metric`  |    Metric denoting the distance metric being used     |         1         | Represents a metric using an int value |
|        `k`         |         Numebr of neighbours to be considered         |         1         |                   -                    |
|        `p`         | Order parameter in calculating the Minkowski distance |         1         |                   -                    |
|      `nrows`       |      Number of rows in training feature dataset       |         1         |                   -                    | 
|      `ncols`       |   Number of columns in the training feature dataset   |         1         |                   -                    |

## Functions available to be used
|                                    Function call                                    |     Return type      |                            Description                            |
|:-----------------------------------------------------------------------------------:|:--------------------:|:-----------------------------------------------------------------:|
|      KNNRegressor(std::vector<std::vector<float>> &X, std::vector<float> &Y);       |        `void`        |           Constructor for the class Logistic_Regression           |
|             train (int k_i, std::string& distance_metric_i, int p_i=2)              |        `void`        |                     Begins training the model                     |
|                        predict (std::vector<float> &x_pred)                         |       `float`        |             Returns the outcome for the given vector              |
|                  predict (std::vector<std::vector<float>> &x_pred)                  | `std::vector<float>` |      Returns a std::vector of outcomes for the given matrix       |
| print_predict (std::vector<std::vector<float>> &x_test, std::vector<float> &y_test) |        `void`        |      Prints the outcomes along with the actual target values      |
|    analyze(std::vector<std::vector<float>> &x_test, std::vector<float> &y_test)     |        `void`        | Prints the various performance metrics based on the given dataset |

## Theory

### Normalization formulae:

* Mean ($\mu_j$) for the $j$-th feature (column):
$$  \mu_j = \frac{1}{N} \sum_{i=1}^{N} X_{i,j}$$
(N is the total number of samples)

* Standard Deviation ($\sigma_j$) for the $j$-th feature:
  $$  \sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (X_{i,j} - \mu_j)^2}$$

* Normalized Value ($X'_{i,j}$):
$$  X'_{i,j} = \frac{X_{i,j} - \mu_j}{\sigma_j}$$


### Distance Metrics

* Manhattan Distance (L1-norm):
  $$  d(\mathbf{x}, \mathbf{x}_{\text{train}})^{(1)} = \sum_{j=1}^{D} |x'_j - x'_{\text{train},j}|$$

* Euclidean Distance (L2-norm):
  $$  d(\mathbf{x}, \mathbf{x}_{\text{train}})^{(2)} = \sqrt{\sum_{j=1}^{D} (x'_j - x'_{\text{train},j})^2}$$

* Minkowski Distance (Lp-norm):
  $$  d(\mathbf{x}, \mathbf{x}_{\text{train}})^{(p)} = \left(\sum_{j=1}^{D} |x'_j - x'_{\text{train},j}|^p\right)^{1/p}$$


### Prediction and Averaging

* KNN Regression Prediction:
  $$  \hat{y} = \frac{1}{K} \sum_{i \in \text{KNN}(K)} y_i$$


### Evaluation Metrics

* Mean Squared Error (MSE):
  $$  \text{MSE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} (y_i - \hat{y}_i)^2$$

* Root Mean Squared Error (RMSE):
  $$  \text{RMSE} = \sqrt{\text{MSE}}$$

* Mean Absolute Error (MAE):
  $$  \text{MAE} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} |y_i - \hat{y}_i|$$

* R-squared ($R^2$ Score):
  $$  R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{N_{\text{test}}} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N_{\text{test}}} (y_i - \bar{y})^2}$$

($\text{SS}_{\text{res}}$ is the Sum of Squares of Residuals, $\text{SS}_{\text{tot}}$ is the Total Sum of Squares, and $\bar{y}$ is the mean of the actual values).