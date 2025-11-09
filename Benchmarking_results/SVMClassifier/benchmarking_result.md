*  The SVM Classifier (Pegasos) code is behaving stably. The results are published below.
* The `Give Me Some Credit` dataset with various predetermined number of rows are used. Number of columns in the dataset = 10

**Hyperparameters considered:**
* 

| Number of rows | Time taken by Glacier<br>`SVMClassifier` (PEGASOS) | Time taken by Scikit-learn<br>`SGDClassifier` | Speed-up |
|:--------------:|:--------------------------------------------------:|:----------------------------------------------|:--------:|
|      500       |                         4                          | 35                                            |  8.75x   |
|      1000      |                         6                          | 37                                            |  6.16x   |
|      5000      |                         25                         | 112                                           |  4.48x   |
|     10000      |                         50                         | 496                                           |  9.92x   |
|     50000      |                        248                         | 2074                                          |  8.36x   |
|     100000     |                        491                         | 4058                                          |  8.26x   |
|     140000     |                        695                         | 5423                                          |  7.80x   |

