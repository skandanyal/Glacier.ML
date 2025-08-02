* The KNN Classifier code is behaving stably. The results are published below.
* The code has not been optimized, parallelized or customised to fit a particular size of dataset.
* The `Give Me Some Credit` dataset with various predetermined number of rows are used. Number of columns in the dataset = 10

Distance metric considered - `Manhattan distance`

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | Time taken by `Glacier`<br>(in seconds) | Time taken by `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:---------------------------------------:|:---------------------------------------:|
| 500 |                                     23                                      |                   16                    |                 0.0606                  |
| 1000 |                                     32                                      |                   32                    |                 0.0804                  |
| 5000 |                                     71                                      |                   156                   |                 0.1428                  |
| 10000 |                                     100                                     |                   313                   |                 0.2254                  |
| 50000 |                                     224                                     |                  1549                   |                 0.7280                  |
| 100000 |                                     316                                     |                    🥲                     |                 1.2999                  |
| 140000 |                                     374                                     |                 🤡                |                 1.9920                  |

Distance metric considered - `Euclidean distance`

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | Time taken by `Glacier`<br>(in seconds) | Time taken by `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:---------------------------------------:|:---------------------------------------:|
| 500 |                                     23                                      |                   30                    |                 0.0381                  |
| 1000 |                                     32                                      |                   57                    |                 0.0518                  |
| 5000 |                                     71                                      |                   327                   |                 0.1261                  |
| 10000 |                                     100                                     |                   656                   |                 0.2104                  |
| 50000 |                                     224                                     |                        😭                 |                 0.6887                  |
| 100000 |                                     316                                     |                   🥲                    |                 1.2352                  |
| 140000 |                                     374                                     |                   🤡                    |                 1.7194                 |

Distance metric considered - `Minkowski distance` (p = 3)

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | Time taken by `Glacier`<br>(in seconds) | Time taken by `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:---------------------------------------:|:---------------------------------------:|
| 500 |                                     23                                      |                   32                    |                 0.1534                  |
| 1000 |                                     32                                      |                   72                    |                 0.1947                  |
| 5000 |                                     71                                      |                   365                   |                 0.4356                  |
| 10000 |                                     100                                     |                    🙏                     |                 0.7090                  |
| 50000 |                                     224                                     |                   😭                    |                 2.1708                  |
| 100000 |                                     316                                     |                   🥲                    |                 3.2155                  |
| 140000 |                                     374                                     |                   🤡                    |                 3.9870                  |

