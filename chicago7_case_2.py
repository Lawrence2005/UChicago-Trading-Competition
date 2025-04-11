import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class Allocator():
    def __init__(self, train_data):
        '''
        Initialize with training data. Store historical prices and train per-asset Ridge models
        '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()

        # Preprocess training data: compute log returns
        # Using logs here because logs are additive over time due to log properties.
        self.returns = np.log(self.train_data / self.train_data.shift(1)).dropna()

        # Feature engineering: create lag features (lag1, lag2) for later Ridge regression
        self.features = pd.concat([
            self.returns.shift(1),
            self.returns.shift(2)
        ], axis=1).dropna()

        self.features.columns = [f"{col}_lag1" for col in train_data.columns] + [f"{col}_lag2" for col in train_data.columns]

        # Align targets with features to stop Ridge regression from throwing an error.
        self.targets = self.returns.loc[self.features.index]

        # Train Ridge regression models per asset (4, 5, 6)
        # We tested Assets 1, 2, 3 externally. They are fundamentally worse annd more
        # volatile than 4, 5, 6, so we're essentially ignoring them.
        self.models = {}
        for i in [3, 4, 5]:  # Asset indices 4, 5, 6
            model = Ridge(alpha=1.0)
            model.fit(self.features, self.targets.iloc[:, i])
            self.models[i] = model

    def allocate_portfolio(self, asset_prices):
        '''
        Given latest asset prices, return weights for next timestep.
        '''
        
        # Adding data to existing contained dataframe for prices
        new_row = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)

        if len(self.running_price_paths) < 3:
            return np.zeros(6)  # not enough data yet

        # Compute latest features and returns from running path
        returns = np.log(self.running_price_paths / self.running_price_paths.shift(1)).dropna()
        latest = returns.iloc[-2:]  # get last two returns

        if len(latest) < 2:
            return np.zeros(6)

        # Computing features using the same formula as before.
        latest_features = pd.concat([latest.iloc[-1], latest.iloc[-2]])
        latest_features.index = [f"{col}_lag1" for col in self.train_data.columns] + [f"{col}_lag2" for col in self.train_data.columns]
        latest_features = latest_features.to_frame().T

        # Predict next return for Assets 4, 5, 6 with the model.
        preds = {}
        for i in [3, 4, 5]:
            preds[i] = self.models[i].predict(latest_features)[0]

        # Convert predictions into weights based on expected return/vol
        weights = np.zeros(6)
        pred_vec = np.array([preds.get(i, 0) for i in range(6)])

        # Normalize nonzero predictions
        if np.linalg.norm(pred_vec) > 0:
            weights = pred_vec / np.linalg.norm(pred_vec)

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
