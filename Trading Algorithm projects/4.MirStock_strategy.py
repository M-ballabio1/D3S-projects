# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:31:43 2022

@author: matte
"""


#%% Import module

from datetime import datetime
import os
import time
import sys
import math
import pandas as pd
import pandas_datareader as pdr
import seaborn as sn
import matplotlib.pyplot as plt
import yfinance as yf
import backtrader as bt

#%% Define paths

#path = sys.argv[1]                                     #usa come file path quella corrente
#path = r'C:\Users\matte\PycharmProjects\pythonProject_MachineLearning\Algorithmic trading'
#immagini_path = path+(r"\immagini\2.immagini")

#%% Main


# Create a subclass of SignaStrategy to define the indicators and signals
class SmaCross_MirStock(bt.SignalStrategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal
        self.signal_add(bt.SIGNAL_LONG, crossover)  # use it as LONG signal


cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

# Create a data feed
data = bt.feeds.PandasData(dataname=yf.download('PFE', '2018-01-01', '2021-09-01'))

cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(SmaCross_MirStock)  # Add the trading strategy
cerebro.run()  # run it all
cerebro.plot(iplot=False, style='candlestick')  # and plot it with a single command