import torch
import torch.nn as nn
import numpy as np

class PortfolioNetwork(nn.Module):
    """
    Neural network that maps ETF features to a score.
    Uses Xavier initialization for stability.
    """
    def __init__(self, input_size, hidden_sizes, output_size=1, activation='relu'):
        super().__init__()
        layers = []
        prev = input_size
        for hs in hidden_sizes:
            linear = nn.Linear(prev, hs)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev = hs
        final = nn.Linear(prev, output_size)
        nn.init.xavier_uniform_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        # Clamp output to avoid extreme values (optional)
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out.squeeze(-1)

def prepare_features(returns_df, window_days):
    """
    Convert returns DataFrame into features for the network:
    - For each ETF, use the last `window_days` returns + volatility + momentum.
    Returns:
        X: (n_etfs, window_days + 2) array
        y: next day return (to be predicted)
    """
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
    X = np.array(features, dtype=np.float32)
    y = next_day.values.astype(np.float32)
    # Replace any NaN or inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y
