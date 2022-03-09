# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:29:38 2022

@author: matte
"""

#https://www.youtube.com/watch?v=NSyli1E53Fk&list=PLqpCwow11-OpOadBABXeLCfTLLrFDHpAu

#%% Execute from Anaconda prompt

#python Kushiba_strategy.py   --> min. 23

#%% Import module

import datetime
import os
import time
import sys
import math
import pandas as pd
import pandas_datareader as pdr
import seaborn as sn
import matplotlib.pyplot as plt
import backtrader as bt

#%% Define paths

#path = sys.argv[1]                                     #usa come file path quella corrente
#path = r'C:\Users\matte\PycharmProjects\pythonProject_MachineLearning\Algorithmic trading'
#immagini_path = path+(r"\immagini\2.immagini")

#%% Main

# import dataset
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    return stockData

stockList = ['PFE']
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta(days=1000)
stockData = get_data(stockList[0], startDate, endDate)
actualStart = stockData.index[0]
data = bt.feeds.PandasData(dataname=stockData)


#%% First Strategy
#strategy Buy and Hold
class Kushiba_Strategy(bt.Strategy):
    def start(self):
        self.val_start = self.broker.get_cash()         #it's the function for init cash to invest
        
    
    def nextstart(self):
        size = math.floor((self.broker.get_cash() -15) / self.data[0])   #how buy stocks. This formula consider 15€ of costs of commission.
        self.buy(size=size)
    
    def stop(self):
        #calculate of actual return
        self.roi = (self.broker.get_value() / self.val_start) -1 
        print('-'*70)
        print('KUSHIBA STRATEGY - Buy and Hold')
        print('Starting Value:  ${:,.2f}'.format(self.val_start))
        print('ROI:              {:,.2f}%'.format(self.roi * 100.0))
        print('Annualised:       {:,.2f}%'.format(100*(1+self.roi)**(365/(endDate-actualStart).days) -1))
        print('Gross return:    ${:,.2f}'.format(self.broker.get_value() - self.val_start))




#%% Second Strategy
#strategy Buy and Hold but, investing more money monthly or annualy
class Kushiba_Strategy_Plus(bt.Strategy):
    params = dict(
        monthly_cash = 150,     # how invest 
        monthly_range = [5, 20] # when during the month investing
        )
    
    #initialize each parameters
    def __init__(self):
        self.order = None           #order initially
        self.totalcost = 0          #total money invest during time
        self.cost_wo_bro = 0        
        self.units = 0
        self.times = 0
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def start(self):
        self.broker.set_fundmode(fundmode=True, fundstartval=100.0)
        
        self.cash_start = self.broker.get_cash()         #it's the function for init cash to invest
        self.val_start = 100.0
        
        #add timer
        self.add_timer(
            when=bt.timer.SESSION_START ,
            monthdays=[i for i in self.p.monthly_range] ,
            monthcarry=True,                                #if 5 or 20 of that month is holiday or weekned. The order will send the next days valid.
            #timername='buytimer'
        )
        
    def notify_timer(self, timer, when, *args):
        self.broker.add_cash(self.p.monthly_cash)              #how buy stocks. This formula consider 15€ of costs of commission.

        target_value = self.broker.get_value() + self.p.monthly_cash - 10
        self.order_target_value(target=target_value)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price %.2f, Cost %.2f, Comm %.2f, Size %.0f' %
                    (order.executed.price,
                    order.executed.value,
                    order.executed.comm,
                    order.executed.size)
                )

                self.units += order.executed.size
                self.totalcost += order.executed.value + order.executed.comm
                self.cost_wo_bro += order.executed.value
                self.times += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            print(order.status, [order.Canceled, order.Margin, order.Rejected])

        self.order = None
    
    def stop(self):
        # calculate actual returns
        self.roi = (self.broker.get_value() / self.cash_start) - 1
        self.froi = (self.broker.get_fundvalue() - self.val_start)
        value = self.datas[0].close * self.units + self.broker.get_cash()
        print('-'*70)
        print('KUSHIBA STRATEGY PLUS - Buy and Hold and more ')
        print('Time in Market: {:.1f} years'.format((endDate - actualStart).days/365))
        print('#Times:         {:.0f}'.format(self.times))
        print('Value:         ${:,.2f}'.format(value))
        print('Cost:          ${:,.2f}'.format(self.totalcost))
        print('Gross Return:  ${:,.2f}'.format(value - self.totalcost))
        print('Gross %:        {:.2f}%'.format((value/self.totalcost - 1) * 100))
        print('ROI:            {:.2f}%'.format(100.0 * self.roi))
        print('Fund Value:     {:.2f}%'.format(self.froi))
        print('Annualised:     {:.2f}%'.format(100*((1+self.froi/100)**(365/(endDate - actualStart).days) - 1)))
        print('-'*70)
        
        
class FixedCommissionScheme(bt.CommInfoBase):
    paras = (
        ('commission',15),
        ('stocklike',True),
        ('commtype',bt.CommInfoBase().COMM_FIXED))
    
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission


def run(data):
    # KUSHIBA STRATEGY
    cerebro = bt.Cerebro(stdstats=False,optreturn=True,optdatas=True)
    cerebro.adddata(data)
    cerebro.addstrategy(Kushiba_Strategy)
    
    
    #broker information
    broker_args = dict(coc=True)
    cerebro.broker = bt.brokers.BackBroker(**broker_args)
    comminfo = FixedCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    
    cerebro.broker.set_cash(5000)
    
    cerebro.run()
    
    cerebro.plot(iplot=False, style='candlestick')
    
    
    #KUSHIBA STRATEGY PLUS
    cerebro1 = bt.Cerebro(stdstats=False,optreturn=True,optdatas=True)
    cerebro1.adddata(data)
    cerebro1.addstrategy(Kushiba_Strategy_Plus)

    # Broker Information
    broker_args = dict(coc=True)
    cerebro1.broker = bt.brokers.BackBroker(**broker_args)
    comminfo = FixedCommissionScheme()
    cerebro1.broker.addcommissioninfo(comminfo)

    cerebro1.broker.set_cash(1000)

    cerebro1.run()
    cerebro1.plot(iplot=False, style='candlestick')

    

if __name__ == '__main__':
    run(data)












