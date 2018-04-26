# Machine-Learning
This repository contains implementation of Machine Learning techniques. The following approaches have been used - 

• **k-nearest neighbour classifier and regressor** on iris dataset from the UCI directory - https://archive.ics.uci.edu/ml/datasets/iris. The dataset contains 4 features and 150 instances with no missing values. The data was shuffled for the implementation of the technique and split 80% for training and 20% for testing.. The accuracy achieved for both the classification and regression was above 95%. The code implements both .csv(for classification) and .data(for regression) files

• Perceptron implementation for **logistic regression**. For the training dataset, 3000 training instances were generated n two    sets of random data points (1500 in each) from multi-variate normal distribution with
```
μ1 = [1 0]
μ2 = [0 1.5]
cov1 = [1 0.75]
       [0.75 1]
cov2 = [1 0.75]
       [0.75 1]
```
and label them 0 and 1. Testing data was generated in the same manner as training data, but sample 500 instances for each class, i.e., 1000 in total. Sigmoid function for activation function and cross entropy for objective function, and perform batch training. Maximum number of iterations is 3000. ROC and AUC curves were plotted.

• **Naive Bayes classifier** on the mushroom dataset from the UCI directory - https://archive.ics.uci.edu/ml/datasets/mushroom. The dataset contains 22 features and classifier was used to implement if the mushroom selected is edible or poisonous. The dataset contains total 8124 instances. 4000 instances have been used for training and the rest for testing. The accuracy achieved was 79.85% by removing the missing instances in the dataset.
