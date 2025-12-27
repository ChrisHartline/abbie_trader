"""
Threshold sweep for HyperDUM and regime stability cutoffs (2023-2024).

This script trains a lightweight FFNN on pre-2023 BTC-USD data, then
backtests 2023-01-01 to 2024-12-31 while sweeping:
- HyperDUM uncertainty thresholds
- Regime stability cutoffs (stable vs. warning)

Outputs:
- threshold_sweep_results.csv with per-combination metrics
- docs/threshold_sweep_lines.svg (line plots for hit rate/PnL/drawdown)
- docs/threshold_sweep_heatmap.svg (PnL heatmap)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-01-01")  # inclusive of 2024

VOL_TARGET = 0.20
MAX_GROSS_EXPOSURE = 0.50
KELLY_FRACTION = 0.25
INITIAL_USD = 10_000.0

HYPER_THRESHOLDS = np.round(np.arange(0.30, 0.56, 0.05), 3)
STABILITY_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
WARNING_OFFSET = 0.15  # warning threshold = stable - offset


# ------------------------------------------------------
# Models and utilities
# ------------------------------------------------------
class FFNN(nn.Module):
    def __init__(self, input_size: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def run_ekf(prices: pd.Series, dt: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    values = prices.values.astype(float)
    n = len(values)
    x = np.zeros((n, 3))
    P = np.zeros((n, 3, 3))

    x[0] = np.array([values[0], 0.0, -5.0])
    P[0] = np.eye(3)

    Q = np.diag([0.01, 1e-4, 1e-4])
    R = 0.5

    smoothed = np.zeros(n)

    for t in range(1, n):
        F = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])
        x_pred = F @ x[t - 1]
        P_pred = F @ P[t - 1] @ F.T + Q

        y = values[t] - x_pred[0]
        S = P_pred[0, 0] + R
        K = P_pred[:, 0] / S

        x[t] = x_pred + K * y
        P[t] = (np.eye(3) - np.outer(K, np.array([1, 0, 0]))) @ P_pred
        smoothed[t] = x[t][0]

    level = pd.Series(smoothed, index=prices.index)
    velocity = pd.Series(x[:, 1], index=prices.index)
    return level, velocity


def detect_trend_regime(prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    returns = prices.pct_change()
    up_move = returns.clip(lower=0)
    down_move = -returns.clip(upper=0)

    up_smooth = up_move.rolling(window).mean()
    down_smooth = down_move.rolling(window).mean()

    total_move = up_smooth + down_smooth
    dx = np.abs(up_smooth - down_smooth) / (total_move + 1e-10)
    adx = dx.rolling(window).mean()

    regime = (adx > 0.4).astype(int)
    return regime, adx


def detect_volatility_regime(prices: pd.Series, window: int = 20, threshold: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(365)
    median_vol = vol.rolling(window * 5).median()
    regime = (vol > threshold * median_vol).astype(int)
    return regime, vol


def detect_regime_stability(trend_regime: pd.Series, vol_regime: pd.Series, window: int = 20) -> pd.Series:
    trend_changes = (trend_regime != trend_regime.shift(1)).astype(int)
    vol_changes = (vol_regime != vol_regime.shift(1)).astype(int)

    trend_stability = 1 - (trend_changes.rolling(window).sum() / window)
    vol_stability = 1 - (vol_changes.rolling(window).sum() / window)

    stability = (trend_stability + vol_stability) / 2
    return stability


def get_funding_rates(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.0005, n))
    seasonal = 0.0002 * np.sin(np.arange(n) * 2 * np.pi / 180)
    noise = 0.0001 * rng.standard_normal(n)
    return np.tanh(base + seasonal) + noise


# ------------------------------------------------------
# Data prep
# ------------------------------------------------------
def load_btc_prices() -> pd.DataFrame:
    csv_path = Path("regime_analysis.csv")
    if not csv_path.exists():
        raise FileNotFoundError("regime_analysis.csv not found. Expected local historical data.")

    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    raw = raw.rename(columns={"Date": "date", "price": "Close", "ekf_level": "level", "ekf_velocity": "velocity"})
    raw = raw.set_index("date").sort_index()
    raw["return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    raw["funding_rate"] = get_funding_rates(len(raw))
    raw["momentum_5"] = raw["return"].rolling(5).mean()
    raw["rel_price"] = raw["Close"] / raw["Close"].rolling(30).mean() - 1
    return raw.dropna()


def prepare_datasets(btc: pd.DataFrame):
    features = btc[["level", "velocity", "funding_rate", "momentum_5", "rel_price"]]
    targets = btc["return"].shift(-1)

    train = btc.index <= TRAIN_END
    X_train = features[train].dropna()
    y_train = targets.loc[X_train.index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = FFNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)

    model.train()
    for _ in range(250):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # HyperDUM components based on training distribution
    hyperdim = 2048
    feature_dim = X_train_scaled.shape[1]
    rng = np.random.default_rng(42)
    projector = rng.standard_normal((feature_dim, hyperdim))
    projector = projector / np.linalg.norm(projector, axis=0, keepdims=True)
    projected_train = np.sign(X_train_scaled @ projector)
    memory_vector = np.sign(np.mean(projected_train, axis=0))

    # Prepare full dataset with predictions and HyperDUM distances
    X_all_scaled = scaler.transform(features.loc[X_train.index.union(btc.index[btc.index > TRAIN_END])])
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_all_scaled)).squeeze().numpy()

    feature_frame = features.loc[X_train.index.union(btc.index[btc.index > TRAIN_END])].copy()
    feature_frame["pred_return"] = preds

    projected_all = np.sign(X_all_scaled @ projector)
    memory = memory_vector.reshape(1, -1)
    hamming = np.mean(projected_all != memory, axis=1)
    feature_frame["hyperdum"] = hamming

    merged = btc.loc[feature_frame.index].copy()
    merged[["pred_return", "hyperdum"]] = feature_frame[["pred_return", "hyperdum"]]

    return merged, scaler, projector, memory_vector


# ------------------------------------------------------
# Backtest
# ------------------------------------------------------
@dataclass
class BacktestResult:
    hyper_threshold: float
    stability_threshold: float
    hit_rate: float
    total_return: float
    max_drawdown: float
    sharpe: float


def compute_regime_arrays(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    trend_regime, _ = detect_trend_regime(prices)
    vol_regime, _ = detect_volatility_regime(prices)
    stability = detect_regime_stability(trend_regime, vol_regime)
    return trend_regime, vol_regime, stability


def get_regime_multiplier(trend_regime: int, stability: float, stable_cutoff: float, warning_cutoff: float) -> Tuple[float, str]:
    if stability < warning_cutoff:
        return 0.0, "WARNING"
    if stability >= stable_cutoff and trend_regime == 0:
        return 1.0, "FAVORABLE"
    if stability >= stable_cutoff and trend_regime == 1:
        return 0.5, "CAUTION"
    return 0.75, "NEUTRAL"


def run_backtest(btc: pd.DataFrame, hyper_threshold: float, stable_cutoff: float, warning_cutoff: float) -> BacktestResult:
    start_mask = (btc.index >= TEST_START) & (btc.index < TEST_END)
    data = btc.loc[start_mask].copy()

    trend_regime, vol_regime, stability = compute_regime_arrays(btc["Close"])

    cash = INITIAL_USD
    position = 0.0
    equity_curve: List[float] = []
    returns: List[float] = []

    for idx in data.index:
        price = data.loc[idx, "Close"]
        actual_return = data.loc[idx, "return"]

        t_idx = trend_regime.index.get_loc(idx)
        regime_mult, _ = get_regime_multiplier(
            trend_regime.iloc[t_idx], stability.iloc[t_idx], stable_cutoff, warning_cutoff
        )

        hamming = data.loc[idx, "hyperdum"]
        if hamming > hyper_threshold or regime_mult == 0.0:
            target = 0.0
        else:
            recent_vol = data["return"].loc[:idx].tail(60).std() * np.sqrt(252)
            risk = min(MAX_GROSS_EXPOSURE, VOL_TARGET / max(recent_vol, 0.01))
            target = np.sign(data.loc[idx, "pred_return"]) * risk * KELLY_FRACTION * regime_mult

        equity = cash + position * price
        target_value = equity * target
        target_position = target_value / price
        delta = target_position - position

        if abs(delta) > 0.0001:
            position += delta
            cash -= delta * price

        equity = cash + position * price
        if not np.isnan(actual_return):
            cash += position * price * actual_return
            equity = cash + position * price
            equity_curve.append(equity)
            returns.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0.0)

    equity_series = pd.Series(equity_curve, index=data.index[: len(equity_curve)])
    eq_returns = pd.Series(returns, index=equity_series.index)

    cumulative = (1 + eq_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    hit_rate = (eq_returns > 0).mean() if len(eq_returns) > 0 else 0.0
    total_return = (equity_series.iloc[-1] / INITIAL_USD - 1) if len(equity_series) else 0.0
    sharpe = eq_returns.mean() / eq_returns.std() * np.sqrt(252) if eq_returns.std() > 0 else 0.0

    return BacktestResult(
        hyper_threshold=hyper_threshold,
        stability_threshold=stable_cutoff,
        hit_rate=hit_rate,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


# ------------------------------------------------------
# Plotting
# ------------------------------------------------------
def plot_lines(results: pd.DataFrame, output: Path):
    metrics = [
        ("hit_rate", "Hit Rate", lambda x: x * 100, "%"),
        ("total_return", "PnL (Total Return)", lambda x: x * 100, "%"),
        ("max_drawdown", "Max Drawdown", lambda x: x * 100, "%"),
    ]

    width, height = 1600, 400
    sections = len(metrics)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}</style>',
    ]

    x_padding = 60
    y_padding = 40
    section_width = width // sections

    for idx, (col, title, transform, suffix) in enumerate(metrics):
        x0 = idx * section_width
        x1 = x0 + section_width
        svg_parts.append(f'<rect x="{x0}" y="0" width="{section_width}" height="{height}" fill="#fafafa" stroke="#ddd"/>')
        svg_parts.append(f'<text x="{x0 + section_width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>')

        stab_values = sorted(results["stability_threshold"].unique())
        x_vals = sorted(results["hyper_threshold"].unique())
        x_scale = (section_width - 2 * x_padding) / (max(x_vals) - min(x_vals))

        # Determine y-range
        y_values = []
        for stab in stab_values:
            subset = results[results["stability_threshold"] == stab]
            y_values.extend(transform(subset[col].values))
        y_min, y_max = min(y_values), max(y_values)
        if y_max == y_min:
            y_max = y_min + 1.0
        y_scale = (height - 2 * y_padding) / (y_max - y_min)

        # Axes
        svg_parts.append(f'<line x1="{x0 + x_padding}" y1="{height - y_padding}" x2="{x1 - x_padding}" y2="{height - y_padding}" stroke="#000"/>')
        svg_parts.append(f'<line x1="{x0 + x_padding}" y1="{y_padding}" x2="{x0 + x_padding}" y2="{height - y_padding}" stroke="#000"/>')
        svg_parts.append(f'<text x="{x0 + section_width/2}" y="{height - 8}" text-anchor="middle">HyperDUM Threshold</text>')
        svg_parts.append(f'<text x="{x0 + 12}" y="{y_padding - 10}" text-anchor="start">{suffix}</text>')

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for s_idx, stab in enumerate(stab_values):
            subset = results[results["stability_threshold"] == stab].sort_values("hyper_threshold")
            path_cmds = []
            for _, row in subset.iterrows():
                x_pos = x0 + x_padding + (row["hyper_threshold"] - min(x_vals)) * x_scale
                y_val = transform(row[col])
                y_pos = height - y_padding - (y_val - y_min) * y_scale
                path_cmds.append(f"{x_pos},{y_pos}")
                svg_parts.append(f'<circle cx="{x_pos}" cy="{y_pos}" r="3" fill="{colors[s_idx % len(colors)]}"/>')
            if len(path_cmds) >= 2:
                svg_parts.append(f'<polyline fill="none" stroke="{colors[s_idx % len(colors)]}" stroke-width="2" points="{" ".join(path_cmds)}"/>')
            svg_parts.append(
                f'<text x="{x1 - x_padding + 10}" y="{y_padding + 15 + 15 * s_idx}" '
                f'text-anchor="start" fill="{colors[s_idx % len(colors)]}">Stability ≥ {stab:.2f}</text>'
            )

    svg_parts.append("</svg>")
    output.write_text("\n".join(svg_parts))


def plot_heatmap(results: pd.DataFrame, output: Path):
    pivot = results.pivot(index="stability_threshold", columns="hyper_threshold", values="total_return")
    rows, cols = pivot.shape
    cell_w, cell_h = 80, 40
    width = cell_w * cols + 140
    height = cell_h * rows + 120

    min_val, max_val = pivot.values.min(), pivot.values.max()
    if max_val == min_val:
        max_val = min_val + 1e-6

    def color_for(value: float) -> str:
        norm = (value - min_val) / (max_val - min_val)
        r = int(68 + norm * (253 - 68))
        g = int(1 + norm * (231 - 1))
        b = int(84 + norm * (37 - 84))
        return f"rgb({r},{g},{b})"

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}</style>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="16" font-weight="bold">Total Return Heatmap (%)</text>',
    ]

    for i, stab in enumerate(pivot.index):
        for j, hyper in enumerate(pivot.columns):
            val = pivot.loc[stab, hyper] * 100
            x = 100 + j * cell_w
            y = 40 + i * cell_h
            svg_parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color_for(pivot.loc[stab, hyper])}" stroke="#fff"/>')
            svg_parts.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 4}" text-anchor="middle" fill="#000">{val:.1f}</text>')

    # Axes labels
    for j, hyper in enumerate(pivot.columns):
        x = 100 + j * cell_w + cell_w / 2
        svg_parts.append(f'<text x="{x}" y="{40 + rows * cell_h + 20}" text-anchor="middle" transform="rotate(45 {x},{40 + rows * cell_h + 20})">{hyper:.2f}</text>')

    for i, stab in enumerate(pivot.index):
        y = 40 + i * cell_h + cell_h / 2
        svg_parts.append(f'<text x="80" y="{y + 4}" text-anchor="end">{stab:.2f}</text>')

    svg_parts.append(f'<text x="{width/2}" y="{height - 10}" text-anchor="middle">HyperDUM Threshold</text>')
    svg_parts.append(f'<text x="30" y="{height/2}" text-anchor="middle" transform="rotate(-90 30,{height/2})">Stability Cutoff</text>')
    svg_parts.append("</svg>")
    output.write_text("\n".join(svg_parts))


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    set_seeds(42)
    btc = load_btc_prices()
    btc, _, _, _ = prepare_datasets(btc)

    results: List[BacktestResult] = []
    for hyper in HYPER_THRESHOLDS:
        for stable in STABILITY_THRESHOLDS:
            warning = max(0.0, stable - WARNING_OFFSET)
            res = run_backtest(btc, hyper, stable, warning)
            results.append(res)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df = results_df.sort_values(["hyper_threshold", "stability_threshold"])
    results_df.to_csv("threshold_sweep_results.csv", index=False)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    plot_lines(results_df, docs_dir / "threshold_sweep_lines.svg")
    plot_heatmap(results_df, docs_dir / "threshold_sweep_heatmap.svg")

    best = results_df.sort_values("sharpe", ascending=False).iloc[0]
    print("Best by Sharpe (robustness-focused):")
    print(best)


if __name__ == "__main__":
    main()
