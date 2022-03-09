# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:17:31 2022

@author: matte
"""

### The strategy based on the Moving Average at 5 and 15 days. The goal is create a signal
### that it allows to buy or sell stock.

#%% Import module

from pathlib import Path               #combine path elements with /
import os
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt

#%% Define paths

path = r'C:\Users\matte\PycharmProjects\pythonProject_MachineLearning\Algorithmic trading'
immagini_path = path+r'\immagini\1.immagini'

#%% Main

if __name__ == '__main__':
        
        gld = pdr.get_data_yahoo('PFE')             #Pfizer
        day = np.arange(1, len(gld) + 1)            #adding new columns as index
        gld['day'] = day
        gld.drop(columns=['Adj Close', 'Volume'], inplace = True)    #remove two columns
        gld = gld[['day', 'Open', 'High', 'Low', 'Close']]
        gld.head()
        
        #adding moving average to dataframe to understand the delayed trend
        #at 1 weeks and 3 weeks (actual market opening)
        gld['5-day'] = gld['Close'].rolling(5).mean()
        gld['15-day'] = gld['Close'].rolling(15).mean()
        gld[19:25]
        
        #adding signal
        gld['signal'] = np.where(gld['5-day'] > gld['15-day'], 1, 0)
        gld['signal'] = np.where(gld['5-day'] < gld['15-day'], -1, gld['signal'])
        gld.dropna(inplace=True)
        gld.head()
        
        #calculate istantaneous return/systems return
        gld['return'] = np.log(gld['Close']).diff()
        gld['system_return'] = gld['signal'] * gld['return']
        gld['entry'] = gld.signal.diff()
        gld.head()
         
        print(np.exp(gld['return']).cumprod()[-1] -1)                          #cumulative return strategy 1
        print(np.exp(gld['system_return']).cumprod()[-1] -1)                   #cumulative return strategy 2

#%% Define functions

def Visualization():
    #plot - bot trading strategy (buy or sell)
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.grid(True, alpha = .3)
    plt.plot(gld.iloc[-252:]['Close'], label = 'GLD')
    plt.plot(gld.iloc[-252:]['5-day'], label = '5-day')
    plt.plot(gld.iloc[-252:]['15-day'], label = '15-day')
    plt.plot(gld[-252:].loc[gld.entry == 2].index, gld[-252:]['5-day'][gld.entry == 2], '^',
         color = 'g', markersize = 12)
    plt.plot(gld[-252:].loc[gld.entry == -2].index, gld[-252:]['15-day'][gld.entry == -2], 'v',
         color = 'r', markersize = 12)
    plt.legend(loc=2);
    plt.show()
    fig.savefig(immagini_path+r'\1.TimeSeries_Trading_strategy.png')
    
    #plot - comparison beetween 
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(np.exp(gld['return']).cumprod(), label='Buy/Hold')           #plot cumulative product of elements using system of Buy and Hold 
    plt.plot(np.exp(gld['system_return']).cumprod(), label='System')      #plot cumulative product of elements using System of signals
    plt.legend(loc=2)
    plt.grid(True, alpha=.3)
    fig.savefig(immagini_path+r'\2.Benchmark_system.png')
    
print(Visualization())


