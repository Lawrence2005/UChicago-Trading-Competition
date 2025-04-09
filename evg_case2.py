import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('Case2.csv')

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)


class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data

        self.ticks_per_day = 30
        self.last_weights = np.full(6, 1/6)

        # Set up storage for full-day logic
        self.end_of_day_prices = self.running_price_paths.iloc[self.ticks_per_day - 1::self.ticks_per_day].copy()

        # Compute asset variance once on training set
        self.daily_log_returns = np.log(self.end_of_day_prices / self.end_of_day_prices.shift(1)).dropna()
        self.asset_variance = self.daily_log_returns.var() # avoid div by zero
        
        
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        ### TODO Implement your code here

        # GIVEN weight assigment
        # weights = np.array([0,1,-1,0.5,0.1,-0.2])

        # equal weight strategy
        # weights = np.full(6, 1/6)

        # group B performs better than group A
        # weights = np.array([-1, -1, -1, 1, 1, 1])

        # return weights

        # TEST STRATEGY

        tick_index = len(self.running_price_paths)

        # Append new tick to running data
        self.running_price_paths = pd.concat(
            [self.running_price_paths, pd.DataFrame([asset_prices])],
            ignore_index=True
        )

        # Only update weights at end of day
        if tick_index % self.ticks_per_day != self.ticks_per_day - 1:
            return self.last_weights

        # Step 1: extract end-of-day prices
        eod_prices = self.running_price_paths.iloc[self.ticks_per_day - 1::self.ticks_per_day]

        # Step 2: compute today's log return (from yesterday to today)
        if len(eod_prices) < 2:
            return self.last_weights  # not enough history yet

        today_return = np.log(eod_prices.iloc[-1].values / eod_prices.iloc[-2].values)

        # Step 3: compute r_i = return / variance for each asset
        
        r_i = np.abs(today_return / self.asset_variance.values)

        w1 = r_i[0] / (r_i[0] + r_i[1] + r_i[2])
        w2 = r_i[1] / (r_i[0] + r_i[1] + r_i[2])
        w3 = r_i[2] / (r_i[0] + r_i[1] + r_i[2])

        w4 = r_i[3] / (r_i[3] + r_i[4] + r_i[5])
        w5 = r_i[4] / (r_i[3] + r_i[4] + r_i[5])
        w6 = r_i[5] / (r_i[3] + r_i[4] + r_i[5])

        weights = np.array([-w1, -w2, -w3, w4, w5, w6])

        print(weights)

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