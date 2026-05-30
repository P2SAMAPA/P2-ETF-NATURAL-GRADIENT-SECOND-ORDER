# Natural Gradient Second-Order Engine for ETFs

Implements second‑order natural gradient methods – **K‑FAC** (Martens & Grosse, 2015) and **Shampoo** (Gupta et al., 2018) – for deep portfolio networks. These optimizers precondition the gradient using block‑diagonal or full‑matrix approximations of the Fisher information, achieving much faster convergence than first‑order natural gradient.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Deep network with configurable hidden layers
- Features: lagged returns, volatility, momentum
- Training with K‑FAC or Shampoo (set in config)
- Score = predicted next‑day return
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-natural-gradient-second-order-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The network learns a ranking function from historical features.
- Second‑order natural gradient optimizers accelerate convergence, leading to more stable and accurate predictions.
- Higher predicted return → stronger long signal.

## Requirements

See `requirements.txt`.
