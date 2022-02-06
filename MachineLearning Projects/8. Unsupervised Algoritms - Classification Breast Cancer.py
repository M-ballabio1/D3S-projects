#Classification of cancer dignosis
#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

#importing the dataset
dataset = load_breast_cancer()
X = dataset['data']
Y = dataset['target']


############ modify variables and data featuring
#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


######################## Model Unsupervised Learning Comparison

#Fitting K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, Y_train)
Y_pred_test2 = classifier2.predict(X_test)

#summary accuracy
print(accuracy_score(Y_test, Y_pred_test2))                  #95.1 Acuracy
print(classification_report(Y_test, Y_pred_test2))


print('-'*100)

#Fitting SVM
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, Y_train)
Y_pred_test3 = classifier3.predict(X_test)

#summary accuracy
print(accuracy_score(Y_test, Y_pred_test3))                  #97.2 Acuracy
print(classification_report(Y_test, Y_pred_test3))


print('-'*100)

#Fitting Naive_Bayes
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, Y_train)
Y_pred_test5 = classifier5.predict(X_test)

#summary accuracy
print(accuracy_score(Y_test, Y_pred_test5))                  #91.6 Acuracy
print(classification_report(Y_test, Y_pred_test5))
