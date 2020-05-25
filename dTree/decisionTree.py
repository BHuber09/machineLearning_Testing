import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/bill_authentication.csv")

# X is the data, y is the labels..
X = df.drop('Class', axis=1)
y = df['Class']

# test_size: specifies the ratio of the test set, 20% for the test set and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# creating a fitting the model.
dTree = DecisionTreeClassifier()
dTree.fit(X_train, y_train)

y_pred = dTree.predict(X_test)

# getting the results from the model..
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))