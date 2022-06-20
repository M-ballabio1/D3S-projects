##libraries

import math
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

#%% New section

path = (r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_ML&DL\dataset statistica2')

data = pd.read_excel(path+"/USAEnergy_Dataset_Annual.xlsx")
df = pd.DataFrame(data)
print (df)
trasp = df.iloc[:,10]
co_china = df.iloc[:,15]
co_russia = df.iloc[:,16]
print(df.set_axis(['Years','Coal_PRO','Gas_PRO','Crude_oil_PRO','Gas_plant_iquids PRO','Tot_Fossil_Fuels PRO','Tot_Renewable_PRO',
                   'Tot_Energy_PRO','Petroleum_Imports_OPEC','Total_Energy Imports','Tot_Fossil_Fuels_CONS_Transport',
                   'Tot_Renewable_CONS','Tot_Fossil_Fuels_CONS','Tot_Energy_CONS','Tot_CO2_USA','Tot_CO2_China',
                   'Tot_CO2_Russia','Days_over_threshold_USA','aha','sakk'], axis='columns'))

sns.heatmap(df.corr(), annot=True, cmap='YlOrBr', fmt='.0%')
fig = plt.gcf()
fig.set_size_inches(10,8)
plt.show()
