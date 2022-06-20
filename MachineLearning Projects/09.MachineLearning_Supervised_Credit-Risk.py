# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:12:26 2022

@author: matte
"""

#%% Introduction

'''

These are yhe differents variables that composed German_credit_dataset:
- Age (numeric)
- Sex (text: male, female)
- Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
- Housing (text: own, rent, or free)
- Saving accounts (text - little, moderate, quite rich, rich)
- Checking account (numeric, in DM - Deutsch Mark)
- Credit amount (numeric, in DM)
- Duration (numeric, in month)
- Purpose(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others
- Risk (Value target - Good or Bad Risk)

'''

#%% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score
from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#%% Path

path = r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_ML&DL\german_credit_risk'
file_excel = path+r'\german_credit_data_with_target.csv'

#%% Exploring dataset and some Variables

df = pd.read_csv(file_excel, sep=";")

# basic informations of dataset
print("Righe dataset iniziale:",df.shape[0])
print("Colonne dataset iniziale:",df.shape[1])
print(df.info())                                             #structure and datatypes of data
print(df.nunique())                                          #Looking unique values for each variables
print(df.head())                                             #Looking the data

df_good = df[df["Risk"] == 'good']
df_bad = df[df["Risk"] == 'bad']

#Let's look the Credit Amount column
interval = (18, 25, 35, 60, 120)
cats = ['Student', 'Young', 'Adult', 'Senior']
df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)

'''
#Example of PLOTLY usage
### FIRST page

trace0 = go.Bar(
            x = df[df["Risk"]== 'good']["Risk"].value_counts().index.values,
            y = df[df["Risk"]== 'good']["Risk"].value_counts().values,
            name='Good credit'
    )

trace1 = go.Bar(
            x = df[df["Risk"]== 'bad']["Risk"].value_counts().index.values,
            y = df[df["Risk"]== 'bad']["Risk"].value_counts().values,
            name='Bad credit'
    )

data = [trace0, trace1]
layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Risk Variable'
    ),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename='grouped-bar')
fig.show()


df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()
df_age = df['Age'].values.tolist()

### SECOND page
#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
#Second plot
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')

####

#Let's look the Credit Amount column
interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Young', 'Adult', 'Senior']
df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)

df_good = df[df["Risk"] == 'good']
df_bad = df[df["Risk"] == 'bad']


### THIRD page
trace0 = go.Box(
    y=df_good["Credit amount"],
    x=df_good["Age_cat"],
    name='Good credit',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_bad['Credit amount'],
    x=df_bad['Age_cat'],
    name='Bad credit',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount (US Dollar)',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')
'''

#%% Seaborn plot
def Visualization_data():
    #figure1
    fig, ax = plt.subplots(nrows=2, figsize=(12,8))
    plt.subplots_adjust(hspace = 0.4, top = 0.8)
    plt.legend(loc='center right', title='Team')

    g1 = sns.distplot(df_good["Age"], ax=ax[0], 
             color="b")
    g1 = sns.distplot(df_bad["Age"], ax=ax[0], 
             color='r')
    g1.set_title("Age Distribuition", fontsize=15)
    g1.set_xlabel("Age")
    g1.set_xlabel("Frequency")
    

    g2 = sns.countplot(x="Age",data=df, 
              palette="rocket", ax=ax[1], 
              hue = "Risk")
    g2.set_title("Age Counting by Risk", fontsize=15)
    g2.set_xlabel("Age")
    g2.set_xlabel("Count")
    plt.show()

    #figure2
    fig, ax = plt.subplots(figsize=(12,12), nrows=2)

    g3 = sns.boxplot(x="Job", y="Credit amount", data=df, 
            palette="rocket", ax=ax[0], hue="Risk")
    g3.set_title("Credit Amount by Job", fontsize=15)
    g3.set_xlabel("Job Reference", fontsize=12)
    g3.set_ylabel("Credit Amount", fontsize=12)

    g4 = sns.violinplot(x="Job", y="Age", data=df, ax=ax[1],  
               hue="Risk", split=True, palette="rocket")
    g4.set_title("Job Type reference x Age", fontsize=15)
    g4.set_xlabel("Job Reference", fontsize=12)
    g4.set_ylabel("Age", fontsize=12)

    plt.subplots_adjust(hspace = 0.4,top = 0.9)
    plt.show()
    
    
    #figure3
    plt.figure(figsize = (8,5))

    g5 = sns.distplot(df_good['Credit amount'], color='b')
    g5 = sns.distplot(df_bad["Credit amount"], color='r')
    g5.set_title("Credit Amount Frequency distribuition", fontsize=15)
    plt.show()
    
    #figure4
    print("Description of Distribuition Saving accounts by Risk:  ")
    print(pd.crosstab(df["Saving accounts"],df.Risk))

    fig, ax = plt.subplots(3,1, figsize=(12,12))
    g = sns.countplot(x="Saving accounts", data=df, palette="rocket", 
              ax=ax[0],hue="Risk")
    g.set_title("Saving Accounts Count", fontsize=15)
    g.set_xlabel("Saving Accounts type", fontsize=12)
    g.set_ylabel("Count", fontsize=12)

    g1 = sns.violinplot(x="Saving accounts", y="Job", data=df, palette="rocket", 
               hue = "Risk", ax=ax[1],split=True)
    g1.set_title("Saving Accounts by Job", fontsize=15)
    g1.set_xlabel("Savings Accounts type", fontsize=12)
    g1.set_ylabel("Job", fontsize=12)

    g = sns.boxplot(x="Saving accounts", y="Credit amount", data=df, ax=ax[2],
            hue = "Risk",palette="rocket")
    g2.set_title("Saving Accounts by Credit Amount", fontsize=15)
    g2.set_xlabel("Savings Accounts type", fontsize=12)
    g2.set_ylabel("Credit Amount(US)", fontsize=12)

    plt.subplots_adjust(hspace = 0.4,top = 0.9)
    plt.show()
    
    
    #figure5
    print("Values describe: ")
    print(pd.crosstab(df.Purpose, df.Risk))

    plt.figure(figsize = (14,12))

    plt.subplot(221)
    g = sns.countplot(x="Purpose", data=df, 
              palette="rocket", hue = "Risk")
    g.set_xticklabels(g.get_xticklabels(),rotation=45)
    g.set_xlabel("", fontsize=12)
    g.set_ylabel("Count", fontsize=12)
    g.set_title("Purposes Count", fontsize=20)

    plt.subplot(222)
    g1 = sns.violinplot(x="Purpose", y="Age", data=df, 
                    palette="rocket", hue = "Risk",split=True)
    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
    g1.set_xlabel("", fontsize=12)
    g1.set_ylabel("Count", fontsize=12)
    g1.set_title("Purposes by Age", fontsize=20)

    plt.subplot(212)
    g2 = sns.boxplot(x="Purpose", y="Credit amount", data=df, 
               palette="rocket", hue = "Risk")
    g2.set_xlabel("Purposes", fontsize=12)
    g2.set_ylabel("Credit Amount", fontsize=12)
    g2.set_title("Credit Amount distribuition by Purposes", fontsize=20)

    plt.subplots_adjust(hspace = 0.6, top = 0.8)
    plt.show()
    
    
    #figure6
    plt.figure(figsize = (12,14))

    g= plt.subplot(311)
    g= sns.countplot(x="Duration", data=df, 
              palette="rocket",  hue = "Risk")
    g.set_xlabel("Duration Distribuition", fontsize=12)
    g.set_ylabel("Count", fontsize=12)
    g.set_title("Duration Count", fontsize=20)

    g1 = plt.subplot(312)
    g1 = sns.pointplot(x="Duration", y ="Credit amount",data=df,
                   hue="Risk", palette="rocket")
    g1.set_xlabel("Duration", fontsize=12)
    g1.set_ylabel("Credit Amount(US)", fontsize=12)
    g1.set_title("Credit Amount distribuition by Duration", fontsize=20)

    g2 = plt.subplot(313)
    g2 = sns.distplot(df_good["Duration"], color='b')
    g2 = sns.distplot(df_bad["Duration"], color='r')
    g2.set_xlabel("Duration", fontsize=12)
    g2.set_ylabel("Frequency", fontsize=12)
    g2.set_title("Duration Frequency x good and bad Credit", fontsize=20)

    plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)
    plt.show()
    
#print(Visualization_data())

#%% Features engineering and Ploting some informations

print("Purpose : ",df.Purpose.unique())
print("Sex : ",df.Sex.unique())
print("Housing : ",df.Housing.unique())
print("Saving accounts : ",df['Saving accounts'].unique())
print("Risk : ",df['Risk'].unique())
print("Checking account : ",df['Checking account'].unique())
print("Aget_cat : ",df['Age_cat'].unique())

#trasform true variables to dummy variables
'''
def one_hot_encoder(df, nan_as_category = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
'''
df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
df['Checking account'] = df['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable

df = df.merge(pd.get_dummies(df.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Sex feature in dummies
df = df.merge(pd.get_dummies(df.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
df = df.merge(pd.get_dummies(df.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
df = df.merge(pd.get_dummies(df["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
# Housing get Age categorical
df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

#because ML model prediction improve with low values
df['Credit amount'] = np.log(df['Credit amount'])

#Excluding the missing columns
del df["Saving accounts"]
del df["Checking account"]
del df["Purpose"]
del df["Sex"]
del df["Housing"]
del df["Age_cat"]
del df["Risk"]
del df['Risk_good']

#correlation with new data
plt.figure(figsize=(14,12))
sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True,  linecolor='white', annot=True)
plt.show()

#%% Splitting dataset

#Creating the X and y variables
X = df.drop('Risk_bad', 1).values
y = df["Risk_bad"].values

# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)


#%% Comparison of different models
# to feed the random state

seed = 7

# comparison of 11 models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', XGBClassifier()))
models.append(('NeuralNet', MLPClassifier(alpha=1, max_iter=1000)))
models.append(('AdaBoost',AdaBoostClassifier()))
models.append(('QDA',QuadraticDiscriminantAnalysis()))


# evaluate each model in turn
results = []
names = []
scoring = 'recall'

for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(11,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#%% INFERENCE ML models

#First model
## Suppose to use Random Forest Classfier and do a GridSearch with fine tuning hyperparameters

#Setting the Hyper Parameters
param_grid = {"max_depth": [2,3,5, 7, 10,None],
              "n_estimators":[2,3,5,7,9,10,25,50,150],
              "max_features": [4,7,15,20]}

#Creating the classifier and Optimize the hyperparameters
model = RandomForestClassifier(random_state=2)

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)

k = grid_search.best_score_
k1 = grid_search.best_params_


#Apply the best parameters

rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)
#trainning with the best params
rf.fit(X_train, y_train)


###
### save the model to disk  --> accuracy score = 75,9%
filename1 = 'finalized_model_RandomForest.pkl'
pickle.dump(grid_search, open(filename1, 'wb'))
###


#Testing the model 
#Predicting using our model
y_pred = rf.predict(X_test)

# Verificaar os resultados obtidos
print(accuracy_score(y_test,y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))


'''
## the second model
GNB = GaussianNB()
model = GNB.fit(X_train, y_train)

# Printing the Training Score
print("Training score data: ")
print(model.score(X_train, y_train))

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

#Evaluation of ROC CURVE
#Predicting proba
y_pred_prob = model.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Create Pipeline with PCA and GaussianNB

features = []
features.append(('pca', PCA(n_components=2)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', GaussianNB()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))


## Another model with XGBClassifier Fine-tuning parameters
#Seting the Hyper Parameters
param_test1 = {
 'max_depth':[3,5,6,10],
 'min_child_weight':[3,5,10],
 'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10],
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}


#Creating the classifier
model_xg = XGBClassifier(random_state=2, use_label_encoder =False)

grid_search = GridSearchCV(model_xg, param_grid=param_test1, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model_XGBoost.pkl'
pickle.dump(grid_search, open(filename, 'wb'))

#save model as json file
#grid_search.save_model('model_file_name.json')

#best params optimize
g = grid_search.best_score_
g1 = grid_search.best_params_

y_pred = grid_search.predict(X_test)

# Verificaar os resultados obtidos
print(accuracy_score(y_test,y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

'''
