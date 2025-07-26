### 16-07-2025, 02:12 am 
* Wrote the training and predict functions of logistic regression.
* My arms, upper back and thighs are still aching after yesterday's gym session (brdr made me improve on yesterday's arm day).
* SGPA is 7.26. This project better be worth these freaking compromises.

### 18-07-2025, 01:18 pm
* Polished logistic regression files to the best of my efforts.
* Will now move onto KNN.
* Came to Bangalore on 16-07-2025 night after travelling from Majestic bus stand till HSR layout for 2 hours. 
* Trust in yourself. You're not on a popular path, but you're on a legit path.

### 20-07-2025 09:38 am
* The logistic regression code is finally behaving stably. The results are published below.
* The code has not been optimized, parallelized or customised to fit a particular size of dataset.
* The `Give Me Some Credit` dataset with various predetermined number of rows are used. Number of columns in the dataset = 10

| Number of rows | Alpha (learning rate)<br>(Not configurable in `sklearn`) | Number of iterations | Time taken by `Glacier`<br>(in seconds) | Time taken by `sklearn`<br>(in seconds) |
|:--------------:|:-----:|:--------------------:|:--------------------------------------------:|:--------------------------------------------:|
| 500 | 0.01 | 2000 | 0.2966 | 0.2813|
| 1000 | 0.005 | 2000 | 0.5618 | 0.4432 |
| 5000 | - | - | - | 0.6379 |
| 10000 | 0.001 | 2000 | 5.8355 | 1.0718 |
| 50000 | 0.0001 | 2000 | 27.8964 | 46.1069 |
| 100000 | 0.0001 | 2000 | 59.6352 | 60.2744 |
| 140000 | 0.0001 | 2000 | 80.9204 | 63.5351 |

### 26-07-2025 10:16 pm
* KNN Regression model is fixed and uploaded, needs to be benchmarked.
* Datasets have been moved to a separate folder outside all the models.
* Man, working alone feels scary. You know what is even more scary? Testing out your models alone while it goes through bugs and unnoticed logical errors.
* Perplexity was dumb back in my 2nd sem. Turns out that it is stil equally dumb even today. Mathe ke upar 'Pro' chipkaane se koi bhi agent professional grade agent nahi ban jaata... :[ ChatGPT still rocks when it comes to answering basic questions.
