# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:34:12 2022

@author: matte
"""

#%% Introduction

'''
Common Vulnerabilities and Exposures (CVE) is a list of computer security threats provided by 
the U.S. Department of Homeland Security and maintained by the MITRE corporation.

Per MITRE's terminology documentation, CVE distinguishes between vulnerabilities where:

A "vulnerability" is a weakness in the computational logic (e.g., code) found in software 
and some hardware components (e.g., firmware) that, when exploited, results in a negative 
impact to confidentiality, integrity, OR availability.

An "exposure" is a system configuration issue or a mistake in software that allows access 
to information or capabilities that can be used by a hacker as a stepping-stone
into a system or network.
'''

#%% Libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
 
import scipy.optimize as opt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import scipy.stats as sps
import plotly.io as pio

#%% Path

path = r'C:\Users\matte\OneDrive\Desktop\documenti\matteo\password e progetti\progetti\Dataset_ML&DL\cybersecurity\Common Vulnerabilities and Exposure - Cybersecurity'
file_excel0 = path+r'/cve.csv'
file_excel1 = path+r'/products.csv'
file_excel2 = path+r'/vendors.csv'

#%% Exploring dataset and some Variables

#import
cve = pd.read_csv(file_excel0)
products = pd.read_csv(file_excel1)
vendors = pd.read_csv(file_excel2)

#cve description
nRow, nCol = cve.shape
print(f'There are {nRow} rows and {nCol} columns')
cve.dropna(axis=0)
cve.describe()

#products description
nRow, nCol = products.shape
print(f'There are {nRow} rows and {nCol} columns')
products.dropna(axis=0)
products.describe()

#vendors description
nRow, nCol = vendors.shape
print(f'There are {nRow} rows and {nCol} columns')
vendors.dropna(axis=0)
vendors.describe()


#%% Visualization

#first
# Common Vulnerability Scoring System (CVSS) score, a measure of the severity of a vulnerability (we see that 35/40% vulnearbilties have score 5)
plt.figure(figsize=(10,6))
sns.kdeplot(data=cve['cvss'], shade=True)
plt.ylabel('Density')
plt.xlabel('CVSS score')
plt.title('')

#second
# Common Weakness Enumeration (CWE) code, identifying the type of weakness. We see that the most vulnerabilites has a code 90-100
plt.figure(figsize=(10,8))
sns.distplot(a=cve['cwe_code'], kde=False)
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
plt.xlabel('Common Weakness Enumeration Code')
plt.ylabel('Total number of times CWE code was identified')


#third
### EVALUATE SEVERITY OF VULNEARABILITY
# The Common Vulnerability Scoring System (CVSS) is an open framework for describing 
# the characteristics and severity of computer security exploits developed and maintained 
# by FIRST. These scores consider exploitability and impact alongside temporal and environmental 
# factors. Scores range from 0 to 10.

fig = go.Figure()
X = cve.cvss.sort_values().astype('int').value_counts().sort_index()[1:]

# Three traces
fig.add_trace(
    go.Bar(
        x=X.index.map(lambda x: "{}-{}".format(x-1,x)),
        y=X.values/np.sum(X.values)*100,
        marker_color=['#bbbbbb', '#bbbbbb', '#bbbbbb', '#dc322f', '#dc322f', '#dc322f', '#dc322f', '#bbbbbb', '#bbbbbb', '#bbbbbb'],
        text=np.vectorize(lambda x: str(x) + "%")(np.round((X.values/np.sum(X.values) * 100),1)),
        textposition='outside'
))

fig.update_layout(
    title=dict(
        text="Threat Severity Distribution",
        xref="paper",
        x=0., y=1.
    ),
    font=dict(
        family="Arial",
        size=14,
        color="#586e75"
    ),
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False
    ),
    annotations=[
        dict(
            xref='paper',
            yref='paper',
            x=0., y=1.2,
            showarrow=False,
            text ="CVSS scores reflect a threat's severity. Over 75 percent of scores fall in FIRSTs Medium (4.0-6.9) threat category<br>" +
            "with a thicker tail toward the higher end of the spectrum.",
            valign='top',
            align='left'
        ),
    ],
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    bargap=0
)

pio.renderers.default='browser'
fig.show()


#four
### EVALUATION MOST 25 PRODUCTS THAT WE AFFECTED BY VULNEARBILITIES (ONLY OPERTING SYSTEMS AND WEB BROWSER)
X = products.vulnerable_product.value_counts()[25:0:-1]

fig = go.Figure()
fig.add_trace(go.Bar(
    y=np.vectorize(lambda x: " ".join(map(lambda x: x.title() if len(x) > 2 else x.upper(), x.split("_"))))(X.index),
    x=X.values,
    orientation='h',
    marker_color= ["red"] * 3 + ["#bbbbbb"] + ["red"] * 5 + ["#859900"] + ["red"] + ["#859900"] + ["red"] * 2 + ["#bbbbbb"] * 2 + ["red"] * 2 + ["#859900"] * 2 + ["red"] * 5
))

fig.update_layout(
    height=800,
    title=dict(
        xref='paper',
        text="Affected Products",
        x=0, y=.965
    ),
    font=dict(
        family="Arial",
        size=14,
        color="#586e75"
    ),
    xaxis=dict(
        showgrid=False
    ),
    yaxis=dict(
        showgrid=False,
        tickmode="linear"
    ),
    annotations=[
        dict(
            xref='paper',
            yref='paper',
            x=0., y=1.075,
            showarrow=False,
            text="Most of the top 25 affected products are operating systems (blue) or web browsers (green)",
            valign='top',
            align='left'
        ),
    ],
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    bargap=.2
)

fig.show()


#%% Features engineering and Ploting some informations


#trasform true variables to dummy variables
'''
def one_hot_encoder(cve, nan_as_category = False):
    original_columns = list(cve.columns)
    categorical_columns = [col for col in cve.columns if cve[col].dtype == 'object']
    cve1 = pd.get_dummies(cve, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in cve.columns if c not in original_columns]
    return cve, new_columns
'''
cve['access authentication'] = cve['access authentication'].fillna('no_inf')
cve['access complexity'] = cve['access complexity'].fillna('no_inf')

