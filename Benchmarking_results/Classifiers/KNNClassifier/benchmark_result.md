* The KNN Classifier code is behaving stably. The results are published below.
* The code has not been optimized, parallelized or customised to fit a particular size of dataset.
* The `Give Me Some Credit` dataset with various predetermined number of rows are used. Number of columns in the dataset = 10

Distance metric considered - `Manhattan distance`

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | `Glacier`<br>(in seconds)<br>(Unoptimized) | `Glacier`<br>(in seconds)<br>(Parallelized, Vectorized) | `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:------------------------------------------:|:-------------------------------------------------------:|:-------------------------:|
|      500       |                                     23                                      |                     16                     |                         0.1431                          |          0.0606           |
|      1000      |                                     32                                      |                     32                     |                         0.2627                          |          0.0804           |
|      5000      |                                     71                                      |                    156                     |                         1.2118                          |          0.1428           |
|     10000      |                                     100                                     |                    313                     |                         2.4269                          |          0.2254           |
|     50000      |                                     224                                     |                    1549                    |                         10.9363                         |          0.7280           |
|     100000     |                                     316                                     |                   > 1500                   |                         21.2234                         |          1.2999           |
|     140000     |                                     374                                     |                   > 1500                   |                         30.4121                         |          1.9920           |

Distance metric considered - `Euclidean distance`

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | `Glacier`<br>(in seconds)<br>(Unoptimized) | `Glacier`<br>(in seconds)<br>(Parallelized, Vectorized) | `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:------------------------------------------:|:-------------------------------------------------------:|:-------------------------:|
|      500       |                                     23                                      |                     30                     |                         0.1504                          |          0.0381           |
|      1000      |                                     32                                      |                     57                     |                         0.2801                          |          0.0518           |
|      5000      |                                     71                                      |                    327                     |                         1.4645                          |          0.1261           |
|     10000      |                                     100                                     |                    656                     |                         2.4758                          |          0.2104           |
|     50000      |                                     224                                     |                   > 650                    |                         13.0319                         |          0.6887           |
|     100000     |                                     316                                     |                   > 650                    |                         25.3686                         |          1.2352           |
|     140000     |                                     374                                     |                   > 650                    |                         37.9172                         |          1.7194           |

Distance metric considered - `Minkowski distance` (p = 3)

| Number of rows | Number of nearest neighbours considered<br>`k`<br>(sqrt(no. of neighbours)) | `Glacier`<br>(in seconds)<br>(Unoptimized) | `Glacier`<vr>(in seconds)<br>(Parallelized, Vectorized) | `sklearn`<br>(in seconds) |
|:--------------:|:---------------------------------------------------------------------------:|:------------------------------------------:|:-------------------------------------------------------:|:-------------------------:|
|      500       |                                     23                                      |                     32                     |                         0.5248                          |          0.1534           |
|      1000      |                                     32                                      |                     72                     |                         1.0524                          |          0.1947           |
|      5000      |                                     71                                      |                    365                     |                         5.3134                          |          0.4356           |
|     10000      |                                     100                                     |                   > 360                    |                         10.7089                         |          0.7090           |
|     50000      |                                     224                                     |                   > 360                    |                         53.7785                         |          2.1708           |
|     100000     |                                     316                                     |                   > 360                    |                        106.4602                         |          3.2155           |
|     140000     |                                     374                                     |                   > 360                    |                        150.5534                         |          3.9870           |

