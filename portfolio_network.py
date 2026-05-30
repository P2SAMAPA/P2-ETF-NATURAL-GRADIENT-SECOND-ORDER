import torch
import torch.nn as nn
import numpy as np

class PortfolioNetwork(nn.Module):
    """
    Neural network that maps ETF features (lagged returns, volatility, etc.) to a score.
    The score is used to rank ETFs. The network is trained with K‑FAC/Shampoo.
    """
    def __init__(self, input_size, hidden_sizes, output_size=1, activation='relu'):
        super().__init__()
        layers = []
        prev = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev, hs))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev = hs
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def prepare_features(returns_df, window_days):
    """
    Convert returns DataFrame into features for the network:
    - For each ETF, use the last `window_days` returns + volatility + momentum.
    Returns:
        X: (n_etfs, window_days + 2) array
        y: next day return (to be predicted)
    """
    # Ensure at least window_days+1 rows
    if len(returns_df) < window_days + 1:
        return None, None
    last_window = returns_df.iloc[-window_days:]          # (window_days, n_etfs)
    next_day = returns_df.iloc[-1]                        # (n_etfs,)
    features = []
    for ticker in returns_df.columns:
        ret_series = last_window[ticker].values
        vol = ret_series.std()
        mom = ret_series[-1] - ret_series[0]   # simple momentum
        feat = np.concatenate([ret_series, [vol, mom]])
        features.append(feat)
    X = np.array(features)
    y = next_day.values
    return X, y
