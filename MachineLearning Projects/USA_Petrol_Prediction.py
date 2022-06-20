# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:19:51 2022

@author: matte
"""

#%% Introduction and import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#%% Path

path = r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_ML&DL'   #change this to run in local
file_excel = path+r'\Dataset_regressione_multipla.csv'

#%% Exploring dataset and some Variables

df = pd.read_csv(file_excel, sep=",")

# basic informations of dataset
print("Righe dataset iniziale:",df.shape[0])
print("Colonne dataset iniziale:",df.shape[1])
print(df.info())                                             
print(df.nunique())                                          
print(df.head())
print(50*('-'))                                             

#%% Data exploration

informations = pd.DataFrame(df.describe())

sns.heatmap(df.corr(), annot=True, cmap='YlOrBr', fmt='.0%')
fig = plt.gcf()
fig.set_size_inches(10,8)
plt.show()

sns.pairplot(df, diag_kind="kde")
plt.show()

#plt.bar(df_kit_revenue['Year'],df_kit_revenue['Revenue bln USD'])
#plt.title("Market's Revenue of kitchenware")
#plt.show()


#%% Spltting

#normalization data to make model training less sensitive to the scale of features.
df1 = preprocessing.normalize(df)
#Creating the X and y variables
X = df.drop('Petrol_Consumption', 1).values
y = df['Petrol_Consumption'].values


#HOLDOUT SPLITTING
# Spliting X and y into train and test version 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

#model1 - Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)

#model2 - DecisionTree Regression
Dt_10 = DecisionTreeRegressor()
Dt_10.fit(X_train, y_train)

#model2 - DecisionTree Regression
Dt_100 = DecisionTreeRegressor()
Dt_100.fit(X_train, y_train)

# predict prices of X_test
y_pred1 = lm.predict(X_test)
# predict prices of X_test
y_pred2 = Dt_10.predict(X_test)
# predict prices of X_test
y_pred3 = Dt_100.predict(X_test)


# evaluate the model on test set
print("R2 score Linear regression : %.2f" % r2_score(y_test, y_pred1))
print("R2 score DT(10) : %.2f" % r2_score(y_test, y_pred2))
print("R2 score DT(100) : %.2f" % r2_score(y_test, y_pred3))

print(50*('-'))

#%% Comparison of different models - Cross Validation
# to feed the random state

seed = 7

# comparison of 11 models
models = []
models.append(('LR', LinearRegression()))
models.append(('DTR - 5', DecisionTreeRegressor(max_depth=5)))
models.append(('DTR - 25', DecisionTreeRegressor(max_depth=25)))
models.append(('DTR - 100', DecisionTreeRegressor(max_depth=100)))


# evaluate each model in turn
results = []
names = []
scoring = 'rmse'
for name, model in models:
	kfold = KFold(n_splits=10)
	cv_results = cross_val_score(model, X, y, cv=kfold)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()