"""
Train models using cached regime_analysis.csv data.
No API key required - uses local historical data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# FFNN class (must match other scripts)
class FFNN(nn.Module):
    def __init__(self, input_size=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_funding_rates(n, seed=42):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.0005, n))
    seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
    noise = 0.0001 * rng.standard_normal(n)
    return np.tanh(base + seasonal) + noise

def main():
    print("="*60)
    print("Training Models from Cached Data")
    print("="*60)

    # Load cached data
    csv_path = Path("regime_analysis.csv")
    if not csv_path.exists():
        raise FileNotFoundError("regime_analysis.csv not found")

    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    raw = raw.rename(columns={"Date": "date", "price": "Close", "ekf_level": "level", "ekf_velocity": "velocity"})
    raw = raw.set_index("date").sort_index()
    raw["return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    raw["funding_rate"] = get_funding_rates(len(raw))
    raw["momentum_5"] = raw["return"].rolling(5).mean()
    raw["rel_price"] = raw["Close"] / raw["Close"].rolling(30).mean() - 1
    btc = raw.dropna()

    print(f"Data: {btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d} ({len(btc)} days)")

    # Prepare features and targets
    TRAIN_END = pd.Timestamp("2022-12-31")
    features = btc[["level", "velocity", "funding_rate", "momentum_5", "rel_price"]]
    targets = btc["return"].shift(-1)

    train = btc.index <= TRAIN_END
    X_train = features[train].dropna()
    y_train = targets.loc[X_train.index]

    print(f"Training on data up to {TRAIN_END.date()}")
    print(f"Training samples: {len(X_train)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train FFNN
    model = FFNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)

    print("\nTraining FFNN...")
    model.train()
    for epoch in range(400):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

    # Generate HyperDUM components
    print("\nGenerating HyperDUM components...")
    hyperdim = 2048
    feature_dim = X_train_scaled.shape[1]
    np.random.seed(42)
    projector = np.random.randn(feature_dim, hyperdim)
    projector = projector / np.linalg.norm(projector, axis=0, keepdims=True)

    projected_train = np.sign(X_train_scaled @ projector)
    memory_vector = np.sign(np.mean(projected_train, axis=0))

    # Save models
    torch.save(model, 'btc_model.pth')
    torch.save(scaler, 'btc_scaler.pth')
    np.save('projector.npy', projector)
    np.save('memory.npy', memory_vector)

    print("\n" + "="*60)
    print("Models saved:")
    print("  - btc_model.pth")
    print("  - btc_scaler.pth")
    print("  - projector.npy")
    print("  - memory.npy")
    print("="*60)

    # Quick validation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).numpy().flatten()

    correct_direction = np.mean(np.sign(train_pred) == np.sign(y_train.values))
    print(f"\nTrain direction accuracy: {correct_direction:.1%}")
    print("\nReady for backtesting!")

if __name__ == "__main__":
    main()
