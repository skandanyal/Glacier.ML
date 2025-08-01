* The `Give Me Some Credit` dataset with various predetermined number of rows are used. Number of columns in the dataset = 10
* Hyperparameters:

| Number of rows | Alpha (learning rate)<br>(Not configurable in `sklearn`) | Number of iterations | Time taken by `Glacier`<br>(in seconds) | Time taken by `sklearn`<br>(in seconds) |
|:--------------:|:-----:|:--------------------:|:--------------------------------------------:|:--------------------------------------------:|
| 500 | 0.01 | 2000 | 0.2966 | 0.2813|
| 1000 | 0.005 | 2000 | 0.5618 | 0.4432 |
| 5000 | - | - | - | 0.6379 |
| 10000 | 0.001 | 2000 | 5.8355 | 1.0718 |
| 50000 | 0.0001 | 2000 | 27.8964 | 46.1069 |
| 100000 | 0.0001 | 2000 | 59.6352 | 60.2744 |
| 140000 | 0.0001 | 2000 | 80.9204 | 63.5351 |
