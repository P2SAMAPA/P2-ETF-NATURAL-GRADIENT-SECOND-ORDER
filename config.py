import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-natural-gradient-second-order-results"

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Network architecture
HIDDEN_SIZES = [64, 32]        # two hidden layers
ACTIVATION = 'relu'
OUTPUT_SIZE = 1                # predict next day return (regression)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
KFAC_DAMPING = 1e-4            # Tikhonov damping for K‑FAC
KFAC_UPDATE_FREQ = 10          # update Kronecker factors every N steps
SHAMPOO_PRECONDITIONER_UPDATE = 100  # update Shampoo preconditioners every N steps
OPTIMIZER = 'kfac'             # 'kfac' or 'shampoo'

TOP_N = 3
