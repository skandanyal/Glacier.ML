# LinkedIn post 1 - July 20, 2025

Ladies and Gentlemen, Introducing Glacier.ML

Glacier is a header-only machine learning library, built from scratch entirely using C++. It internally uses Eigen for performing matrix operations and Boost for statistical computations. 

Last semester, I finished an NPTEL course 'Applied Multivariate Statistical Modeling' where I explored the mathematics of Linear Regression and the necessary statistical evaluation metrics. As a natural follow up to that course, I began building Glacier by referring to the course called 'Statistical Learning in Python' by Stanford Online on YouTube. 

What started as a 'lemme give it a shot' project soon turned into a serious affair once I showed a very primitive version to my AI professor Balaji Vijaykumar, who motivated me to push forward. Ultimately, it was his motivating words which gave me the initial push to keep building the project further.

Glacier is still a work in progress. Currently the library hosts three stable machine learning models (Simple and Multiple Linear Regression and Binary Logistic Regression) with more models on the roadmap. Hence I'll not share the link to the hosted repo, or reply actively to most comments for now. 

But here is a small teaser. The logistic regression model was trained, tested and validated against two real-life datasets - 
1. Pima Indians Diabetes Database, with 9 columns and 768 rows (link: https://lnkd.in/guAM_yGd)
2. Wisconsin Cancer Diagnostic Dataset, with 32 columns and 569 rows (link: https://lnkd.in/gkAkHayw)
The evaluation metrics are showcased in the pictures attached below.

The logistic regression model was also tested against the same model from Scikit-learn. And quite honestly, I'm surprised at how my barely optimized, non-parallelized model (which, now runs without throwing German sounding errors at me) fared against a popular battle tested model from Scikit-learn. Time stamps are also mentioned below. 

Truth be told, working upon this project taught me how to venture on a path which once seemed to be beyond my league. It gave me a chance to finally work on a math-centric project, while facing problems that a safe and popular looking project might never throw at me (I mean, who would even teach me what is float underflow? that too by giving me a hands-on example?). I just hope that it evolves into something beautiful someday.

More to share when the time's right :)

Edit - the time stamps show the time taken to train the models on the respective datasets.


# LinkedIn post 2 - (scheduled for) Aug 3, 2025

* update on glacier - 
  * KNN regressor and classifier
  * refactoring code to make it look more presentable
  * updating the documentation to make it look more readable


