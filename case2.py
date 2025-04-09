import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Case2.csv')
returns = data.pct_change()

# excess over portfolio average
excess_returns = returns.sub(returns.mean(axis=1), axis=0).dropna()
excess_returns.to_csv('excess_returns.csv')

BIANN_FACTOR_MEAN = 6
FACTOR_VOLATILITY = np.sqrt(BIANN_FACTOR_MEAN)
FACTOR_SHARPE = np.sqrt(BIANN_FACTOR_MEAN)

stats = {
    "Mean": excess_returns.mean() * BIANN_FACTOR_MEAN,
    "Vol": excess_returns.std() * FACTOR_VOLATILITY
}
stats["Sharpe Ratio"] = stats["Mean"] / stats["Vol"]
stats = pd.DataFrame(stats)
stats.sort_values("Sharpe Ratio", ascending=True)
print(stats, '\n')

correlations = excess_returns.corr()
plt.figure(figsize=(10, 8))  # Optional: Adjust figure size
heatmap = sns.heatmap(
    correlations,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt="0.2f",
    cmap="coolwarm",  # Optional: Use a color gradient
    linewidths=0.5,   # Optional: Add grid lines
)
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

def tangency_portf(mean_returns, cov_matrix):
    inverted_cov = np.linalg.pinv(cov_matrix) if np.isclose(np.linalg.det(cov_matrix), 0) else np.linalg.inv(cov_matrix)
    one_vector = np.ones(mean_returns.shape)
    return (inverted_cov @ mean_returns) / (one_vector @ inverted_cov @ mean_returns)

w_tan = tangency_portf(excess_returns.mean(), excess_returns.cov())
w_tan_df = pd.DataFrame(w_tan, index=excess_returns.columns, columns=["Tangency Portfolio"])
print(w_tan_df.head(), '\n')

w_tan_rets = pd.DataFrame(excess_returns @ w_tan_df)
tan_stats = {
    "Mean": w_tan_rets.mean(),
    "Vol": w_tan_rets.std()
}
tan_stats["Sharpe Ratio"] = tan_stats["Mean"] / tan_stats["Vol"]    
tan_stats = pd.DataFrame(tan_stats)
print(tan_stats, '\n')

allocation = tangency_portf(excess_returns.mean(), excess_returns.cov())
allocation *= (0.0075 / (excess_returns.mean() @ allocation))
allocation_rets = pd.DataFrame(excess_returns @ allocation)
allocation_stats = {
    "Mean": allocation_rets.mean(),
    "Vol": allocation_rets.std()
}
allocation_stats["Sharpe Ratio"] = allocation_stats["Mean"] / allocation_stats["Vol"]
allocation_stats = pd.DataFrame(allocation_stats)
print(allocation_stats)