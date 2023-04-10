# Supervised_learning-models-comparision

This code performs machine learning classification on the Titanic dataset using various classifiers and hyperparameter tuning.

## Required Libraries and Modules
The following libraries and modules are required to run this code:

- numpy
- pandas
- scikit-learn

## Dataset
The Titanic train and test datasets are loaded from the following URLs:

- Train dataset: "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
- Test dataset: "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

## Data Preprocessing
A function called preprocess() is defined to preprocess the data. The function fills missing values, drops unnecessary columns, and converts categorical variables into dummy variables. The train and test datasets are then preprocessed using this function.

## Feature Scaling
The features are scaled using the StandardScaler from scikit-learn.

## Classifiers
The following classifiers are initialized:

+ Logistic Regression
+ SVM
+ Decision Tree
+ Random Forest
+ KNN
## Hyperparameter Tuning
GridSearchCV from scikit-learn is used to perform hyperparameter tuning for each classifier. The best models are selected based on the accuracy score.

## Evaluation
The best models are used to predict the survival of passengers in the test dataset. The accuracy scores are calculated for each classifier and printed.

## Conclusion
This code can be used as a starting point for performing machine learning classification on the Titanic dataset. Different classifiers and hyperparameters can be tried to obtain the best possible accuracy score.
