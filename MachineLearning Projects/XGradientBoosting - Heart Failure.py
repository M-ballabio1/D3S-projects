import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import xgboost as xgb
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from matplotlib import gridspec
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
import pickle
from matplotlib import pyplot
import mplcyberpunk
plt.style.use('cyberpunk')



#####################################################
######### IMPORT csv file from directory ############
#####################################################

df = pd.read_csv(r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_ML&DL\Kaggle dataset\heart_failure_clinical_records_dataset.csv', sep=";")
print("Righe dataset iniziale:",df.shape[0])
print("Colonne dataset iniziale:",df.shape[1])


##############################################################
######### VIEW DISTRIBUTION FEATURES ON THE DATASET ##########
##############################################################

numerical_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
categorical_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

##plotting with numerical features
plt.figure(figsize=(15, 20))

for i, col in enumerate(numerical_features):
    plt.subplot(6, 4, i * 2 + 1)
    plt.subplots_adjust(hspace=.25, wspace=.3)

    plt.grid(True)
    plt.title(col)
    sns.kdeplot(df.loc[df["DEATH_EVENT"] == 0, col], label="alive", color="#990303", shade=True, cut=0)
    sns.kdeplot(df.loc[df["DEATH_EVENT"] == 1, col], label="dead", color="#292323", shade=True, cut=0)
    plt.subplot(6, 4, i * 2 + 2)
    sns.boxplot(y=col, data=df, x="DEATH_EVENT", palette=["#990303", "#9C9999"])
plt.show()

##plotting with categorical features
plt.figure(figsize=(12, 8))

for i, col in enumerate(categorical_features):
    plt.subplot(2, 3, i+1)
    plt.title(col)
    plt.subplots_adjust(hspace =.5, wspace=.3)
    sns.countplot(data=df, x=col, hue="DEATH_EVENT", palette = ["#990303", "#9C9999"], alpha=0.8, edgecolor="k", linewidth=1)
plt.show()

##standardization values
df_norm = df.copy()

for i, col in enumerate(numerical_features):
    df_norm[[col]] = StandardScaler(with_mean=True, with_std=True).fit_transform(df_norm[[col]])

plt.figure(figsize=(16, 4))
gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.1, 1, 1])
plt.subplot(gs[0])
plt.grid(True)
plt.title("ejection fraction")
sns.kdeplot(df.loc[df["DEATH_EVENT"] == 0, "ejection_fraction"], label="alive", color="#990303", shade=True,
            kernel='gau', cut=0)
sns.kdeplot(df.loc[df["DEATH_EVENT"] == 1, "ejection_fraction"], label="dead", color="#292323", shade=True,
            kernel='gau', cut=0)
plt.subplot(gs[1])
sns.boxplot(y="ejection_fraction", data=df, x="DEATH_EVENT", palette=["#990303", "#9C9999"])
plt.subplot(gs[2])
plt.axis('off')
plt.subplot(gs[3])
plt.grid(True)
plt.title("ejection fraction")
sns.kdeplot(df_norm.loc[df["DEATH_EVENT"] == 0, "ejection_fraction"], label="alive", color="#990303", shade=True,
             cut=0)
sns.kdeplot(df_norm.loc[df["DEATH_EVENT"] == 1, "ejection_fraction"], label="dead", color="#292323", shade=True,
             cut=0)
plt.subplot(gs[4])
sns.boxplot(y="ejection_fraction", data=df_norm, x="DEATH_EVENT", palette=["#990303", "#9C9999"]);
plt.tight_layout()

## STATISTICAL TEST
# sw_df = pd.DataFrame(columns=["DEATH_EVENT=0", "DEATH_EVENT=1", "Both"])

index = [(feat, "statistic") for feat in numerical_features]
index.extend([(feat, "p-value") for feat in numerical_features])

index = pd.MultiIndex.from_tuples(index)

sw_df = pd.DataFrame(index=index, columns=["Both Classes", "DEATH_EVENT=0", "DEATH_EVENT=1"])

for feat in numerical_features:
    x = df_norm[feat]
    stat, p = shapiro(x)

    sw_df["Both Classes"].loc[(feat, "statistic")] = stat
    sw_df["Both Classes"].loc[(feat, "p-value")] = p

    x = df_norm.loc[df["DEATH_EVENT"] == 0, feat]
    stat, p = shapiro(x)
    sw_df["DEATH_EVENT=0"].loc[(feat, "statistic")] = stat
    sw_df["DEATH_EVENT=0"].loc[(feat, "p-value")] = p

    x = df_norm.loc[df["DEATH_EVENT"] == 1, feat]
    stat, p = shapiro(x)
    sw_df["DEATH_EVENT=1"].loc[(feat, "statistic")] = stat
    sw_df["DEATH_EVENT=1"].loc[(feat, "p-value")] = p

sw_df = sw_df.unstack()

pd.set_option('display.float_format', '{:.3g}'.format)
print(sw_df)

## valuation weight of different features in dataset

train_ratio = 0.75
val_ratio = 0.25

ho_train_df, ho_val_df = train_test_split(df_norm, train_size = train_ratio, random_state=42)
unnorm_ho_train_df, unnorm_ho_val_df = train_test_split(df, train_size = train_ratio, random_state=42)

print("Holdout split:")
print(f"Train samples: {len(ho_train_df)}")
print(f"Validation/Test samples: {len(ho_val_df)}")

all_features = categorical_features.copy()
all_features.extend(numerical_features)

MI = (mutual_info_classif(ho_train_df[all_features],
                             ho_train_df["DEATH_EVENT"], n_neighbors=20,
                             discrete_features=[True, True, True, True, True, False, False, False, False, False, False],
                             random_state=42))

plt.figure(figsize=(5.4, 4))
plt.barh(y=all_features, width=MI, color="#990303")
plt.title("Mutual information w.r.t. DEATH_EVENT (whole training set)");
plt.xlabel("Mutual information")
plt.gca().xaxis.grid(True, linestyle=':');
plt.tight_layout();

#count DEATH EVENT in dataset
type_counts = df['DEATH_EVENT'].value_counts()
target=pd.DataFrame(type_counts)
target.head()
plt.figure(figsize=(10,5))
number=target.head()
ax = sns.barplot(y='DEATH_EVENT',x=number.index, data=number.head())
plt.show()


######################################################
### PREPROCESSING DATA FOR DISCRETIZATION OF VALUE ###
######################################################
#controllo se ci sono valori nulli
'''
df_null = round(100*(df.isnull().sum())/len(df), 2)
print(df_null)
'''

def Age(x):
    if x > 70:
        return 0
    else:
        return 1

def Crea_Fosf(x,y):
    if x == 0 and (y > 26 and y < 193):
        return 0                                               #normal range CPK for female
    elif x == 1 and (y > 39 and y < 310):
        return 0
    elif x == 0 and y > 193:
        return 1
    elif x == 0 and y < 26:
        return 1
    elif x == 1 and y > 310:
        return 1
    elif x == 1 and y < 39:
        return 1
    else:
        return None

def Platelet(x):
    if x > 150000 and x < 450001:
        return 0                                               #normal range piastrine for female
    else:
        return 1                                               #out of the range

def Sodium(x):
    if x > 135 and x < 146:
        return 0                                               #normal range piastrine for female
    else:
        return 1                                               #out of the range

def Creatine(x,y):
    if x == 0 and (y > 0.59 and y < 1.04):
        return 0  # normal range CPK for female
    elif x == 1 and (y > 0.74 and y < 1.35):
        return 0
    else:
        return 1                                          #out of the range

def Ejection(x):                                              #https://www.pennmedicine.org/updates/blogs/heart-and-vascular-blog/2014/october/ejection-fraction-what-the-numbers-mean
    if x > 55 and x < 76:                                     ##normal range ejection fraction
        return 0
    else:
        return 1

def Angina(x,y,z):
    if x == 0 or y == 1 or z == 1:
        return 1
    else:
        return 0

def Electrolity(x,y,z):
    if x==1 or y==1 or z==1:
        return 1
    else:
        return 0

df.insert(0, "AGE segmentation", df.apply(lambda row: Age(row['age']), axis=1))
df['AGE segmentation'].value_counts()
df.insert(4, "CPK", df.apply(lambda row: Crea_Fosf(row['sex'], row['creatinine_phosphokinase']), axis=1))
df['CPK'].value_counts()
df.insert(1, "PLATELET", df.apply(lambda row: Platelet(row['platelets']), axis=1))
df['PLATELET'].value_counts()
df.insert(2, "SODIUM", df.apply(lambda row: Sodium(row['serum_sodium']), axis=1))
df['SODIUM'].value_counts()
df.insert(2, "CREATINE", df.apply(lambda row: Creatine(row['sex'],row['serum_creatinine']), axis=1))
df['CREATINE'].value_counts()
df.insert(5, "GRAVITY EJECTION", df.apply(lambda row: Ejection(row['ejection_fraction']), axis=1))
df['GRAVITY EJECTION'].value_counts()
df.insert(8, "ANGINA", df.apply(lambda row: Angina(row['AGE segmentation'],row['GRAVITY EJECTION'],row['PLATELET']), axis=1))
df['ANGINA'].value_counts()
df.insert(9,"ELECTROLITY", df.apply(lambda row: Electrolity(row['SODIUM'],row['CREATINE'],row['CPK']), axis=1))
df['ELECTROLITY'].value_counts()

df.drop(['AGE segmentation','CPK','PLATELET','age','creatinine_phosphokinase','platelets','serum_sodium','serum_creatinine','ejection_fraction','time','diabetes','sex','smoking'], axis=1, inplace=True)
print(df.columns)

##################################################
######### DATA ANALYSIS AND FORECASTING ##########
##################################################

#Correlation matrix

sns.heatmap(df.corr(), annot=True, cmap='YlOrBr', fmt='.0%')
fig = plt.gcf()
fig.set_size_inches(10,8)
plt.show()

#Divisione in label e target
X = df.drop(['DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.15, random_state=42)

#Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Gradient boosting
#è una tecnica di machine learning di regressione e problemi di Classificazione statistica che producono un modello predittivo
# nella forma di un insieme di predittivi deboli, tipicamente alberi di decisione. Costruisce un modello in maniera
# simile ai metodi di boosting, e li generalizza permettendo l'ottimizzazione di una funzione di perdita differenziabile arbitraria.

xgb_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=2000, eta=0.05, subsample=0.5, random_state=42, eval_metric=["error","logloss"], use_label_encoder=False)  #regularization parameter optimized
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, early_stopping_rounds=15, eval_metric=["auc", "logloss"], verbose=True, eval_set=eval_set)
y_pred = xgb_model.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# retrieve performance metrics
results = xgb_model.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
