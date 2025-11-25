# KNN Classifier

## Variables used
| Variable name used |                 Variable description                  | Size of container |                Remarks                 |
|:------------------:|:-----------------------------------------------------:|:-----------------:|:--------------------------------------:|
|        `Y`         |                     Target matrix                     |      `n x 1`      |      Before adding column of 1's       |
|        `X`         |                    Feature matrix                     |      `n x p`      |      before adding column of 1's       |
|      `labels`      | an `std::vector` containing <br>the labels of targets |         2         |                   -                    |
|       `mean`       |                      Mean vector                      |      `n x 1`      |        Size never gets modified        |
|     `std_dev`      |               Standard deviation vector               |      `n x 1`      |        Size never gets modified        |
| `distance_metric`  |    Metric denoting the distance metric being used     |         1         | Represents a metric using an int value |
|        `k`         |         Numebr of neighbours to be considered         |         1         |                   -                    |
|        `p`         | Order parameter in calculating the Minkowski distance |         1         |                   -                    |
|      `nrows`       |      Number of rows in training feature dataset       |         1         |                   -                    | 
|      `ncols`       |   Number of columns in the training feature dataset   |         1         |                   -                    |

## Functions available to be used
|                                                  Function call                                                   |         Return type         |                               Description                                |
|:----------------------------------------------------------------------------------------------------------------:|:---------------------------:|:------------------------------------------------------------------------:|
|               KNNClassifier(std::vector<std::vector<float>> &X_i, std::vector<std::string> &Y_i);                |           `void`            |              Constructor for the class Logistic_Regression               |
|                            train (int k_i, std::string& distance_metric_i, int p_i=2)                            |           `void`            |                        Begins training the model                         |
|                                       predict (std::vector<float> &x_pred)                                       |        `std::string`        |          Returns the most probable outcome for the given vector          |
|                                predict (std::vector<std::vector<float>> &x_pred)                                 | `std::vector<std::string>>` | Returns a std::vector of the most probable outcomes for the given matrix |
|            print_predict (std::vector<std::vector<float>> &x_val, std::vector<<std::string>> &y_test)            |           `void`            |     Prints the most probable outcomes along with the actual outcomes     |
|          analyze_2_targets(std::vector<std::vector<float>> &x_test, std::vector<std::string> &y_test);           |           `void`            |    Prints the various performance metrics based on the given dataset     |

## Theory
* Update it someday later.


<date, time - Nov 25, 11.34 am>
