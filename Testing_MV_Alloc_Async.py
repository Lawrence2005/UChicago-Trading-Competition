import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import asyncio
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd

class MV_Alloc():
    def __init__(self, train_data, rebalance_freq, window):
        '''
        Store price history and initialize parameters
        '''
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.lookback_window = window  # Number of ticks for rolling returns
        self.rebalance_freq = rebalance_freq    # Rebalance every N ticks
        self.tick_counter = 0
        
        # Initialize weights to equal weights
        self.current_weights = np.array([1/6] * 6)
        
        # Parameters for mean-variance optimization

    def calculate_rolling_er(self, prices, window):
        '''
        Calculate compounded rolling returns over specified window
        '''
        returns = prices.pct_change()
        rolling_rets = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
        excess_returns = rolling_rets.sub(rolling_rets.mean(axis=1), axis=0).dropna()
        
        return excess_returns
    
    def tangency_portfolio(self, mean_returns, cov_matrix):
        '''
        Compute tangency portfolio weights
        '''
        inverted_cov = np.linalg.pinv(cov_matrix) if np.isclose(np.linalg.det(cov_matrix), 0) else np.linalg.inv(cov_matrix)
        one_vector = np.ones(mean_returns.shape)
        return (inverted_cov @ mean_returns) / (one_vector @ inverted_cov @ mean_returns)
    
    def allocate_portfolio(self, asset_prices):
        self.tick_counter += 1

        # Add new prices directly to NumPy array instead of DataFrame for speed
        if isinstance(self.running_price_paths, pd.DataFrame):
            new_row = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
            self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        else:
            self.running_price_paths = np.vstack([self.running_price_paths, asset_prices])

        # Only rebalance every N ticks
        if self.tick_counter % self.rebalance_freq != 0:
            return self.current_weights

        # Use only most recent window of data
        window = min(self.lookback_window, len(self.running_price_paths))
        prices = self.running_price_paths[-window:]

        # Compute returns (NumPy for speed)
        returns = np.diff(prices, axis=0) / prices[:-1]
        if returns.shape[0] < 2:
            return self.current_weights

        # Expected returns & covariance
        mu = np.mean(returns, axis=0)
        cov = np.cov(returns.T)

        # Tangency portfolio
        try:
            allocation = self.tangency_portfolio(mu, cov)

            # Target return scaling (optional)
            expected_return = mu @ allocation
            allocation *= 0.0075 / expected_return

            # Constraint to max leverage of 1
            max_leverage = 1.0 / np.max(np.abs(allocation))
            allocation *= max_leverage

            self.current_weights = allocation
        except Exception as e:
            print("Tangency portfolio failed:", e)
            self.current_weights = np.ones(6) / 6

        return self.current_weights


        

data = pd.read_csv('Case2.csv')
TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

def grading_1(train_data, test_data, rebalance_freq, window): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = MV_Alloc(train_data, rebalance_freq, window)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    print("finished")

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


# Dummy grading function for illustration — replace with your real one
def grading_wrapper(rebalance_freq, window):
    # Replace this with actual logic that depends on rebalance_freq and window
    # For now, it just returns dummy results
    import numpy as np
    capital = list(np.random.uniform(0.95, 1.05, size=100))
    weights = np.random.dirichlet(np.ones(6), size=100)
    sharpe = np.mean(np.diff(capital)) / np.std(np.diff(capital)) if np.std(np.diff(capital)) != 0 else 0
    return rebalance_freq, window, sharpe, capital, weights.tolist()

# Async driver that handles async task execution
async def run_all_tasks():
    combos = [(rf, w) for rf in range(30, 930, 30) for w in range(90, 1260, 30)]
    results = []

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, grading_1, TRAIN, TEST, rf, w)
            for rf, w in combos
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)

    return results

# Process results into separate CSVs
def save_results_to_csv(results):
    sharpe_data = []
    capital_data = []
    weights_data = []

    for rebalance_freq, window, sharpe, capital, weights in results:
        sharpe_data.append({'rebalance_freq': rebalance_freq, 'window': window, 'sharpe': sharpe})
        for i, val in enumerate(capital):
            capital_data.append({'rebalance_freq': rebalance_freq, 'window': window, 'step': i, 'capital': val})
        for i, weight_vector in enumerate(weights):
            weights_data.append({'rebalance_freq': rebalance_freq, 'window': window, 'step': i, **{f'asset_{j + 1}': w for j, w in enumerate(weight_vector)}})

    pd.DataFrame(sharpe_data).to_csv("Case_2_Results/sharpe_results.csv", index=False)
    pd.DataFrame(capital_data).to_csv("Case_2_Results/capital_results.csv", index=False)
    pd.DataFrame(weights_data).to_csv("Case_2_Results/weights_results.csv", index=False)

# Main function to run everything
def main():
    results = asyncio.run(run_all_tasks())
    save_results_to_csv(results)
    print("Finished writing CSV files!")

if __name__ == "__main__":
    main()