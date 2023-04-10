# Import required libraries and modules
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Titanic train and test datasets from the provided URLs
train_data_url = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
test_data_url = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_data = pd.read_csv(train_data_url)
test_data = pd.read_csv(test_data_url)

# Define a function to preprocess the data
def preprocess(data):
    data['age'].fillna(data['age'].mean(), inplace=True)
    data['embark_town'].fillna(data['embark_town'].mode()[0], inplace=True)
    data.drop(['class', 'deck', 'alone'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['sex', 'embark_town'], drop_first=True)
    return data

# Preprocess the train and test data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Separate the features and target variable
X = train_data.drop('survived', axis=1)
y = train_data['survived']

# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize different classifiers
lr_model = LogisticRegression(max_iter=1000)
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
dt_model = DecisionTreeClassifier(random_state=1)
rf_model = RandomForestClassifier(random_state=0)
knn_model = KNeighborsClassifier()

# Hyperparameter tuning using GridSearchCV for each classifier
# SVM
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 1, 10],
              'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X, y)

# Decision Tree
param_grid = {'max_depth': [3, 4, 5, 6, 7],
              'criterion': ['gini', 'entropy']}
dt_grid = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
dt_grid.fit(X, y)

# Random Forest
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [3, 4, 5, 6, 7]}
rf_grid = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X, y)

# K Nearest Neighbors
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
knn_grid = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X, y)

# Update classifiers with best parameters from GridSearchCV
svm_model = svm_grid.best_estimator_
dt_model = dt_grid.best_estimator_
rf_model = rf_grid.best_estimator_
knn_model = knn_grid.best_estimator_

# Final evaluation on the test set
X_test = test_data.drop('survived', axis=1)
y_test = test_data['survived']
X_test = scaler.transform(X_test)

# Predict on test set
lr_model.fit(X, y)
lr_y_pred_test = lr_model.predict(X_test)
svm_y_pred_test = svm_model.predict(X_test)
dt_y_pred_test = dt_model.predict(X_test)
rf_y_pred_test = rf_model.predict(X_test)
knn_y_pred_test = knn_model.predict(X_test)

# Calculate accuracies
lr_test_accuracy = accuracy_score(y_test, lr_y_pred_test)
svm_test_accuracy = accuracy_score(y_test, svm_y_pred_test)
dt_test_accuracy = accuracy_score(y_test, dt_y_pred_test)
rf_test_accuracy = accuracy_score(y_test, rf_y_pred_test)
knn_test_accuracy = accuracy_score(y_test, knn_y_pred_test)

# Print accuracies
print("Test Accuracy with logistic regression: ", lr_test_accuracy)
print("Test Accuracy with support vector machine: ", svm_test_accuracy)
print("Test Accuracy with decision tree: ", dt_test_accuracy)
print("Test Accuracy with random forest: ", rf_test_accuracy)
print("Test Accuracy with k nearest neighbours: ", knn_test_accuracy)
