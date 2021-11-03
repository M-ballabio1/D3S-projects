#import module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')



#IMPORT csv file from directory
df = pd.read_csv(r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_Kaggle\cardio_train.csv', sep=";")
print("Righe dataset iniziale:",df.shape[0])
print("Colonne dataset iniziale:",df.shape[1])


##########################################
#########-------------------##############
#########DATA PREPROCESSING:##############
#########-------------------##############
##########################################


#column age with trasformation of age in years
df['age'] = round(df['age']/365.25,2)

#convert gender in binary value and replacing the gender column with another two-columns, one for male and the other is for female.
df.insert(3, 'female', (df['gender']==1).astype(int))
df.insert(4, 'male', (df['gender']==2).astype(int))


####################################
##### HEMODYNAMICAL PARAMETERS #####
####################################

#convert two columns Systolic pressure and Diastolic pressure in only column in Mean Arterial Pressure.
SP = df.iloc[:,7]
DP = df.iloc[:,8]
MAP = round((SP+2*DP)/3,2)
df.insert(8,'MAP',MAP)

#delete the other some columns
df.drop(['gender', 'id','ap_hi','ap_lo'], axis=1, inplace=True)

##### Heart Rate and Systolic Volume function depend on the ACTIVE, WEIGHT, MAP #####
#add Heart Rate
def Heart_Rate(x, y, z):
    if x == 0 and y > 80 and z > 105:
        return 90                                 #caso estremo
    elif x == 1 and y > 80 and z > 105:
        return 85                                 #caso estremo ma fa sport
    elif x == 0 or (y > 60 and y < 81) and z>105:
        return 75
    elif x == 1 or (y > 60 and y < 81) and z<106:
        return 60
    elif x == 0 or y < 61 and z>105:
        return 75
    elif x == 1 or y < 61 and z<106:
        return 60
    else:
        return None

df.insert(6, "HR", df.apply(lambda row: Heart_Rate(row['active'], row['weight'], row['MAP']), axis=1))
df['HR'].value_counts()

#add Systolic Volume
def Sys_Vol(x, y, z):
    if x == 0 or y > 80 and z > 105:
        return 100
    elif x == 1 or y > 80 and z < 106:
        return 70
    elif x == 0 or (y > 60 and y < 81) and z>105:
        return 90
    elif x == 1 or (y > 60 and y < 81) and z<106:
        return 60
    elif x == 0 or y < 61 and z>105:
        return 80
    elif x == 1 or y < 61 and z<106:
        return 60
    else:
        return None

df.insert(7, "SY_VOL", df.apply(lambda row: Sys_Vol(row['active'], row['weight'], row['MAP']), axis=1))
df['SY_VOL'].value_counts()

#insert new columns Cardiac Output (l/min)
CO = (df['SY_VOL']*df['HR'])/1000
df.insert(8,'CO',CO)

#add column Body Mass Index
df.insert(5, 'bmi', round((df['weight']/(df['height']/100)**2), 2))

'''
taking the bmi range AND DELETE height and weight
Below 18.5: Underweight --> 1
18.5 - 24.9: Normal --> 2
25.0 - 29.9: Overweight --> 3
30 and above: Obese --> 4
'''

# binning the bmi feature
df.loc[(df['bmi'] < 18.5), 'bmi_cat'] = 1
df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 25), 'bmi_cat'] = 2
df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_cat'] = 3
df.loc[(df['bmi'] >= 30), 'bmi_cat'] = 4

df['bmi_cat'] = df['bmi_cat'].astype('int')
df['bmi_cat'].value_counts()

#delete other columns
df.drop(['height', 'weight'], axis=1, inplace=True)
#delete outliers value in body mass index
df.drop(df.query('bmi >60 or bmi <15').index, axis=0, inplace=True)

#elimina alcuni righe duplicate (ne ha tolte in questo caso 97)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

print("Righe dataset finale:",df.shape[0])
print("Colonne dataset finale:",df.shape[1])


###########################################
#########-------------------------#########
#########-----DATA ANALYSIS-------#########
#########-------------------------#########
###########################################

sns.heatmap(df.corr(), annot=True, cmap='YlOrBr', fmt='.0%')
fig = plt.gcf()
fig.set_size_inches(10,8)
plt.show()

fig, ax = plt.subplots(ncols=3, figsize=(20,10))
plt.tight_layout(pad=18)
sns.boxplot(data=df, x='cardio', y='age', ax=ax[0])
sns.boxplot(data=df, x='cardio', y='bmi', showfliers=False, ax=ax[1])
sns.boxplot(data=df, x='cardio', y='MAP', showfliers=False, ax=ax[2])
ax[0].title.set_text('Age')
ax[0].set_xticklabels(['No-cardio', 'Cardio'])
ax[1].title.set_text('body mass index')
ax[1].set_xticklabels(['No-cardio', 'Cardio'])
ax[2].title.set_text('Mean Arterial Pressure')
ax[2].set_xticklabels(['No-cardio', 'Cardio'])
ax[0].set_xlabel("")
ax[1].set_xlabel("")
ax[2].set_xlabel("")
plt.show()

###################################################
#########---------------------------------#########
#########-------Predicting using ML-------#########
#########---------------------------------#########
###################################################

print(df.head())

#Divisione in label e target
X = df.drop(['cardio'], axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


##### LOGISTIC REGRESSION #####
#Random Forest Classifier

random_model = RandomForestClassifier(n_estimators=51,
                                      max_depth=10,
                                      random_state=0)

random_model.fit(X_train, y_train)
print(f"Testing accuracy: {round(accuracy_score(random_model.predict(X_test), y_test),4)*100}%")
print(f"Average testing accuracy: {round(cross_validate(random_model, X, y, cv=5)['test_score'].mean()*100,2)}%")

print('-'*50)


#K Neighbors Model
#L'algoritmo calcola la distanza tra il nuovo valore e i valori esistenti, quindi trova i k-vicini più vicini, quindi vota le previsioni.

k_model = KNeighborsClassifier(weights = 'uniform',
                               n_neighbors = 300,
                               leaf_size = 1,
                               algorithm = 'ball_tree')
k_model.fit(X_train, y_train)

cross_validate(k_model, X, y, cv=5)['test_score'].mean()
k_pred = k_model.predict(X_test)
print(f"score: {round((accuracy_score(k_pred, y_test)*100),2)}%")


print('-'*50)

#LogisticRegression

# fitting df2 data to the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print(logreg.intercept_)
print(logreg.coef_)

y_pred = logreg.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
