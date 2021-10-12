##OPTIMIZATION PORTFOLIO:


#Import libraries:
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from termcolor import colored

import mplcyberpunk
plt.style.use('cyberpunk')


#Stock present in my portfolio FROM NASDAQ
equity_asset = ['FB', 'INTC', 'PEP','NVDA','GILD','AMZN','AAPL', 'SBUX','GOOG','MSFT']
#percentual of weights of stocks in my portfolio
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#start date
Start ='2015-01-01'
today = datetime.today().strftime('%Y-%m-%d')

#create dataframe
df = pd.DataFrame()
#adjusted close price
for stock in equity_asset:
  df[stock] = web.DataReader(stock, data_source='yahoo', start = Start, end = today)['Adj Close']
print(df)


#visualize stocks
title = 'Portfolio Adj. Close Price History'
#store the stocks
my_stocks = df
#create plot and graph
for i in my_stocks.columns.values:
  plt.plot(my_stocks[i], label=i)
plt.title(title)
plt.xlabel(['Date'], fontsize=15)
plt.ylabel('Adj. Price USD ($)', fontsize=20)
plt.legend(my_stocks.columns.values, loc = 'upper left')
plt.show()

#Variazione percentuale tra l'elemento corrente e uno precedente.
#Calcola la variazione percentuale dalla riga immediatamente precedente per impostazione predefinita. Ciò è utile per confrontare la percentuale di variazione in una serie temporale di elementi.
returns = df.pct_change()
print(returns)

#covariance matrix annualizzata (per determinare come covarianza tra gli assets)
'SOTTO MATRICE DI COVARIANZA annualizzata:'
cov_matrix_annual = returns.cov()*252
print(cov_matrix_annual)

#calculate the portfolio variance --> Dot product of two arrays (WEIGHT.TRASPOSE and cov_matrix)
'VARIANZA DEL PORTAGLIO considerando distribuzione pesi uniforme ossia 20% ciascuno:'
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
print(portfolio_variance)

#calculate volatility
portfolio_volatility = np.sqrt(portfolio_variance)
print(portfolio_volatility)

#annual portfolio return

portfolioReturn = np.sum(returns.mean()*weights)*252
print(portfolioReturn)

#show expected annual return, volatility (risk) and variance

percent_var = str(round(portfolio_variance,2)*100)+'%'
percent_vol = str(round(portfolio_volatility,2)*100)+'%'
percent_ret = str(round(portfolioReturn,2)*100)+'%'

print(50*'-')
print('Performance portfolio con distribuzione UNIFORME')
print('expected annual return:',percent_ret)
print('expected risk:',percent_vol)
print('expected variance:',percent_var)



##########################################################################################################

print(50*'-')
##
##Portfolio Optimization
##

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#optimize for max sharpe ratio

ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

print(colored('Performance portfolio con distribuzione OTTIMIZZATA massimizzando il rischio e minimizzando il financial risk', attrs=['bold']))
ef.portfolio_performance(verbose = True)

#get discreate allocation of each stock
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 15000)

allocation, leftover = da.lp_portfolio()

print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
