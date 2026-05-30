import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import HfApi
import config
import data_manager as dm
from portfolio_network import PortfolioNetwork, prepare_features
from kfac_optimizer import KFACOptimizer, Shampoo

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def run_for_window(returns, window_days):
    if len(returns) < window_days + 1:
        return None
    X, y = prepare_features(returns, window_days)
    if X is None:
        return None
    input_size = X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PortfolioNetwork(input_size, config.HIDDEN_SIZES, config.OUTPUT_SIZE, config.ACTIVATION).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    if config.OPTIMIZER == 'kfac':
        optimizer = KFACOptimizer(model, lr=config.LEARNING_RATE, damping=config.KFAC_DAMPING, update_freq=config.KFAC_UPDATE_FREQ)
    else:  # shampoo
        optimizer = Shampoo(model.parameters(), lr=config.LEARNING_RATE, damping=config.KFAC_DAMPING, update_freq=config.SHAMPOO_PRECONDITIONER_UPDATE)
    for epoch in range(config.EPOCHS):
        train_one_epoch(model, dataloader, optimizer, criterion, device)
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
    # Convert numpy float32 to Python float for JSON serialization
    raw_scores = {ticker: float(pred[i]) for i, ticker in enumerate(returns.columns)}
    return raw_scores, None

def rolling_walkforward_backtest(returns_df, window_days, top_n=3):
    # Placeholder – not used in this version to keep runtime short
    return {}

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "optimizer": config.OPTIMIZER,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            try:
                raw_scores, _ = run_for_window(returns, w)
                if raw_scores is None:
                    continue
                norm_scores = normalize_scores(raw_scores)
                sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
                top_etfs = [{"ticker": t, "ng_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
                all_window_results.append({
                    "window": w,
                    "top_etfs": top_etfs,
                    "all_scores_raw": raw_scores,
                    "all_scores_norm": norm_scores
                })
            except Exception as e:
                print(f"    Failed for window {w}: {e}")
        # Use the last window as best (for simplicity)
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/ng_second_order_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
