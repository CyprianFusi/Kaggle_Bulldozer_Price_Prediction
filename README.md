# Predicting the Sale Price of Bulldozers using Machine Learning
## The Goal 
* "Predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers".
The goal is to try and obtain a **better RMSLE** compared to the **best** score obtained at in the **Kaggle Competition**.

## Results Obtained
* Validation RMSLE obtained: **`0.21163`**, Validation R^2': `0.90759`
* Best Kaggle RMSLE: **`0.22909`**

## Dataset
For Dataset Description see **Kaggle's** [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data). The data for this competition is split into three parts:

* **Train.csv** is the training set, which contains data through the end of 2011.
* **Valid.csv** is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
* **Test.csv** is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition

## Evaluation
"The evaluation metric for this competition is the **RMSLE (root mean squared log error)** between the actual and predicted auction prices." See [Evaluation](https://www.kaggle.com/c/bluebook-for-bulldozers/overview) on Kaggle.
