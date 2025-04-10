# import numpy as np
# import pandas as pd
# import scipy
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import os

# data = pd.read_csv('Case2.csv', index_col = 0)
# TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

# class Allocator():
#     def __init__(self, train_data):
#         '''
#         Anything data you want to store between days must be stored in a class field
#         '''
        
#         self.running_price_paths = train_data.copy()
        
#         self.train_data = train_data.copy()
        
#         # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        
        
#     def allocate_portfolio(self, asset_prices):
#         '''
#         asset_prices: np array of length 6, prices of the 6 assets on a particular day
#         weights: np array of length 6, portfolio allocation for the next day
#         '''
#         self.running_price_paths.append(asset_prices, ignore_index = True)
    
#         ### TODO Implement your code here
#         weights = np.array([0,1,-1,0.5,0.1,-0.2])
        
#         return weights

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
# from sklearn.covariance import LedoitWolf

data = pd.read_csv('Case2.csv', index_col = 0)
TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

class Allocator():
    def __init__(self, train_data):
        '''
        Store price history and initialize parameters
        '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.lookback_window = 30000  # Number of ticks for rolling returns
        self.rebalance_freq = 1000    # Rebalance every N ticks
        self.tick_counter = 0
        
        # Parameters for mean-variance optimization
        self.BIANN_FACTOR_MEAN = 12
        self.FACTOR_VOLATILITY = np.sqrt(self.BIANN_FACTOR_MEAN)
        
    def calculate_rolling_returns(self, prices, window):
        '''
        Calculate compounded rolling returns over specified window
        '''
        returns = prices.pct_change()
        rolling_rets = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
        return rolling_rets.dropna()
    
    def tangency_portfolio(self, mean_returns, cov_matrix):
        '''
        Compute tangency portfolio weights
        '''
        inv_cov = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for stability
        ones = np.ones(mean_returns.shape)
        weights = inv_cov @ mean_returns
        weights /= (ones @ inv_cov @ mean_returns)  # Normalize to sum to 1
        return weights
    
    def allocate_portfolio(self, asset_prices):
        '''
        Main allocation function called daily
        '''
        # Update price history
        self.running_price_paths.loc[len(self.running_price_paths)] = asset_prices
        self.tick_counter += 1
        
        # Only rebalance periodically (every rebalance_freq ticks)
        if self.tick_counter % self.rebalance_freq != 0:
            return self.current_weights
        
        # Calculate rolling returns
        rolling_rets = self.calculate_rolling_returns(
            self.running_price_paths, 
            min(self.lookback_window, len(self.running_price_paths)))
        
        if len(rolling_rets) < 10:  # Insufficient data
            return np.array([1/6]*6)  # Equal weights
        
        # Calculate excess returns
        excess_rets = rolling_rets.sub(rolling_rets.mean(axis=1), axis=0)
        
        # Compute covariance matrix with shrinkage
        cov_matrix = excess_rets.cov()
        
        # Get tangency portfolio weights
        try:
            weights = self.tangency_portfolio(excess_rets.mean(), cov_matrix)
            
            # Rescale to fit -1 to 1 constraint
            max_leverage = 1.0 / np.abs(weights).max()
            weights = weights * max_leverage
            
            self.current_weights = weights
        except:
            weights = np.array([1/6]*6)  # Fallback to equal weights
        
        return weights


def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()