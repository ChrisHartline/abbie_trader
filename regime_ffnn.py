"""
Regime-Specific FFNN Training System

Architecture:
1. Regime Detector classifies market state
2. Each regime (BULL/BEAR/SIDEWAYS) has its own:
   - FFNN model (trained only on that regime's data)
   - HyperDUM memory (encodes "normal" patterns for that regime)
3. CRISIS regime = sit out (no model)

This creates specialist models that are experts in their specific market conditions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

from regime_classifier import RegimeSystem, Regime
from feature_engineering import FeatureEngineer


@dataclass
class FFNNConfig:
    """Configuration for FFNN training"""
    input_size: int = 14          # Number of features
    hidden_sizes: list = None     # Hidden layer sizes
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 400
    batch_size: int = 32          # For future batched training

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]


@dataclass
class HyperDUMConfig:
    """Configuration for HyperDUM"""
    hyperdim: int = 2048
    threshold: float = 0.35


class FFNN(nn.Module):
    """Feedforward Neural Network with configurable architecture"""

    def __init__(self, input_size: int, hidden_sizes: list, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HyperDUM:
    """Hyperdimensional Uncertainty Module for out-of-distribution detection"""

    def __init__(self, config: Optional[HyperDUMConfig] = None):
        self.config = config or HyperDUMConfig()
        self.projector = None
        self.memory_vector = None

    def fit(self, X: np.ndarray, seed: int = 42):
        """Build projector and memory from training data"""
        rng = np.random.default_rng(seed)

        feature_dim = X.shape[1]
        self.projector = rng.standard_normal((feature_dim, self.config.hyperdim))
        self.projector = self.projector / np.linalg.norm(self.projector, axis=0, keepdims=True)

        projected = np.sign(X @ self.projector)
        self.memory_vector = np.sign(np.mean(projected, axis=0))

    def compute_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute Hamming distance from memory"""
        if self.projector is None or self.memory_vector is None:
            raise ValueError("HyperDUM not fitted. Call fit() first.")

        projected = np.sign(X @ self.projector)
        distances = np.mean(projected != self.memory_vector, axis=1)
        return distances

    def is_in_distribution(self, X: np.ndarray) -> np.ndarray:
        """Check if samples are in-distribution (below threshold)"""
        distances = self.compute_distance(X)
        return distances < self.config.threshold


class RegimeModel:
    """A single regime's model (FFNN + Scaler + HyperDUM)"""

    def __init__(
        self,
        regime: Regime,
        ffnn_config: Optional[FFNNConfig] = None,
        hyperdum_config: Optional[HyperDUMConfig] = None,
    ):
        self.regime = regime
        self.ffnn_config = ffnn_config or FFNNConfig()
        self.hyperdum_config = hyperdum_config or HyperDUMConfig()

        self.model = None
        self.scaler = StandardScaler()
        self.hyperdum = HyperDUM(self.hyperdum_config)
        self.is_trained = False
        self.training_samples = 0

    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train the model on regime-specific data"""
        if len(X) < 50:
            if verbose:
                print(f"  [WARNING] Only {len(X)} samples for {self.regime.value} - may underfit")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build HyperDUM memory
        self.hyperdum.fit(X_scaled)

        # Create and train FFNN
        self.model = FFNN(
            input_size=self.ffnn_config.input_size,
            hidden_sizes=self.ffnn_config.hidden_sizes,
            dropout=self.ffnn_config.dropout,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.ffnn_config.learning_rate,
            weight_decay=self.ffnn_config.weight_decay,
        )
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        self.model.train()
        for epoch in range(self.ffnn_config.epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

            if verbose and epoch % 100 == 0:
                print(f"    Epoch {epoch}: Loss = {loss.item():.6f}")

        self.is_trained = True
        self.training_samples = len(X)

        if verbose:
            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                final_pred = self.model(X_tensor).numpy().flatten()
            direction_acc = np.mean(np.sign(final_pred) == np.sign(y))
            print(f"    Final direction accuracy: {direction_acc:.1%}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict returns and compute HyperDUM distances.

        Returns:
            (predictions, hamming_distances)
        """
        if not self.is_trained:
            raise ValueError(f"{self.regime.value} model not trained")

        X_scaled = self.scaler.transform(X)
        distances = self.hyperdum.compute_distance(X_scaled)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.model(X_tensor).numpy().flatten()

        return predictions, distances


class RegimeFFNNSystem:
    """
    Complete regime-specific FFNN system.

    Manages multiple RegimeModels and routes predictions based on detected regime.
    """

    def __init__(
        self,
        ffnn_config: Optional[FFNNConfig] = None,
        hyperdum_config: Optional[HyperDUMConfig] = None,
    ):
        self.ffnn_config = ffnn_config or FFNNConfig()
        self.hyperdum_config = hyperdum_config or HyperDUMConfig()

        self.regime_system = RegimeSystem()
        self.feature_engineer = FeatureEngineer()

        # One model per tradeable regime
        self.models: Dict[Regime, RegimeModel] = {
            Regime.BULL: RegimeModel(Regime.BULL, ffnn_config, hyperdum_config),
            Regime.BEAR: RegimeModel(Regime.BEAR, ffnn_config, hyperdum_config),
            Regime.SIDEWAYS: RegimeModel(Regime.SIDEWAYS, ffnn_config, hyperdum_config),
        }

        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and classify regimes for all data"""
        # Compute features
        features = self.feature_engineer.compute_features(df)
        features = features.dropna()

        # Classify regimes
        regime_history = self.regime_system.classify_history(df, min_lookback=100)
        regime_history['date'] = pd.to_datetime(regime_history['date'])
        regime_history = regime_history.set_index('date')

        # Align
        common_idx = features.index.intersection(regime_history.index)
        features = features.loc[common_idx]
        regime_history = regime_history.loc[common_idx]

        # Add regime column
        features['regime'] = regime_history['regime']

        # Target: next day return
        features['target'] = features['return'].shift(-1)
        features = features.dropna()

        return features

    def train(self, df: pd.DataFrame, train_end: str = "2022-12-31", verbose: bool = True):
        """
        Train regime-specific models.

        Args:
            df: DataFrame with OHLC data
            train_end: End date for training data
            verbose: Print training progress
        """
        if verbose:
            print("="*70)
            print("REGIME-SPECIFIC FFNN TRAINING")
            print("="*70)

        # Prepare data
        if verbose:
            print("\nPreparing features and classifying regimes...")

        features = self.prepare_data(df)

        # Split train/test
        train_mask = features.index <= train_end
        train_data = features[train_mask]

        if verbose:
            print(f"Training data: {train_data.index[0]:%Y-%m-%d} to {train_data.index[-1]:%Y-%m-%d}")
            print(f"Training samples: {len(train_data)}")

        # Get feature columns
        self.feature_names = self.feature_engineer.get_feature_names()

        # Train each regime model
        if verbose:
            print("\n" + "-"*70)
            print("TRAINING REGIME-SPECIFIC MODELS")
            print("-"*70)

        for regime in [Regime.BULL, Regime.BEAR, Regime.SIDEWAYS]:
            regime_data = train_data[train_data['regime'] == regime.value]

            if verbose:
                print(f"\n{regime.value} ({len(regime_data)} samples):")

            if len(regime_data) < 10:
                if verbose:
                    print(f"  [SKIPPED] Not enough samples")
                continue

            X = regime_data[self.feature_names].values
            y = regime_data['target'].values

            self.models[regime].train(X, y, verbose=verbose)

        self.is_trained = True

        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
            for regime, model in self.models.items():
                status = f"{model.training_samples} samples" if model.is_trained else "NOT TRAINED"
                print(f"  {regime.value:<12} {status}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all data.

        Returns DataFrame with predictions, regimes, and HyperDUM distances.
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")

        features = self.prepare_data(df)

        results = []
        for idx in features.index:
            row = features.loc[idx]
            regime_str = row['regime']

            # Map string to Regime enum
            regime = Regime(regime_str)

            result = {
                'date': idx,
                'close': row['close'],
                'actual_return': row['target'],
                'regime': regime_str,
            }

            if regime == Regime.CRISIS:
                # No prediction for crisis
                result['prediction'] = 0.0
                result['hamming_distance'] = 1.0
                result['trade'] = False
            else:
                # Get model for this regime
                model = self.models[regime]

                if model.is_trained:
                    X = row[self.feature_names].values.reshape(1, -1)
                    pred, dist = model.predict(X)
                    result['prediction'] = pred[0]
                    result['hamming_distance'] = dist[0]
                    result['trade'] = dist[0] < self.hyperdum_config.threshold
                else:
                    result['prediction'] = 0.0
                    result['hamming_distance'] = 1.0
                    result['trade'] = False

            results.append(result)

        return pd.DataFrame(results)

    def save(self, directory: str = "models"):
        """Save all models to directory"""
        path = Path(directory)
        path.mkdir(exist_ok=True)

        for regime, model in self.models.items():
            if model.is_trained:
                regime_path = path / regime.value.lower()
                regime_path.mkdir(exist_ok=True)

                torch.save(model.model, regime_path / "ffnn.pth")
                torch.save(model.scaler, regime_path / "scaler.pth")
                np.save(regime_path / "projector.npy", model.hyperdum.projector)
                np.save(regime_path / "memory.npy", model.hyperdum.memory_vector)

        # Save config
        config = {
            'ffnn': {
                'input_size': self.ffnn_config.input_size,
                'hidden_sizes': self.ffnn_config.hidden_sizes,
                'dropout': self.ffnn_config.dropout,
            },
            'hyperdum': {
                'hyperdim': self.hyperdum_config.hyperdim,
                'threshold': self.hyperdum_config.threshold,
            },
            'feature_names': self.feature_names,
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Models saved to {path}/")


def test_regime_ffnn():
    """Test the regime-specific FFNN system"""
    print("="*70)
    print("REGIME-SPECIFIC FFNN SYSTEM TEST")
    print("="*70)

    # Load data
    csv_path = Path("regime_analysis.csv")
    if not csv_path.exists():
        print("Error: regime_analysis.csv not found")
        return

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.rename(columns={'Date': 'date', 'price': 'close'})
    df = df.set_index('date').sort_index()
    df['high'] = df['close'] * 1.02
    df['low'] = df['close'] * 0.98

    print(f"Data: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")

    # Configure with expanded network
    ffnn_config = FFNNConfig(
        input_size=14,  # 14 features
        hidden_sizes=[128, 64, 32],
        dropout=0.3,
        epochs=300,
    )

    hyperdum_config = HyperDUMConfig(
        hyperdim=2048,
        threshold=0.35,
    )

    # Create and train system
    system = RegimeFFNNSystem(ffnn_config, hyperdum_config)
    system.train(df, train_end="2022-12-31", verbose=True)

    # Test predictions
    print("\n" + "-"*70)
    print("TEST PREDICTIONS (2023-2024)")
    print("-"*70)

    predictions = system.predict(df)
    test_preds = predictions[predictions['date'] >= '2023-01-01']

    print(f"\nTest samples: {len(test_preds)}")
    print(f"Trades allowed: {test_preds['trade'].sum()} ({test_preds['trade'].mean()*100:.1f}%)")

    # By regime
    print("\nBy regime:")
    for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'CRISIS']:
        regime_preds = test_preds[test_preds['regime'] == regime]
        if len(regime_preds) > 0:
            trades = regime_preds['trade'].sum()
            direction_correct = (np.sign(regime_preds['prediction']) == np.sign(regime_preds['actual_return'])).mean()
            print(f"  {regime:<12} {len(regime_preds):>4} days, {trades:>4} trades, direction acc: {direction_correct:.1%}")

    # Simple backtest
    print("\n" + "-"*70)
    print("SIMPLE BACKTEST")
    print("-"*70)

    initial = 10000
    equity = initial
    for _, row in test_preds.iterrows():
        if row['trade'] and not np.isnan(row['actual_return']):
            direction = np.sign(row['prediction'])
            ret = direction * row['actual_return'] * 0.25  # 25% position
            equity *= (1 + ret)

    total_return = (equity / initial - 1) * 100
    print(f"\nInitial: ${initial:,.2f}")
    print(f"Final:   ${equity:,.2f}")
    print(f"Return:  {total_return:+.2f}%")

    # Compare to buy and hold
    bh_start = test_preds['close'].iloc[0]
    bh_end = test_preds['close'].iloc[-1]
    bh_return = (bh_end / bh_start - 1) * 100
    print(f"Buy&Hold: {bh_return:+.2f}%")

    print("\n" + "="*70)

    return system, predictions


if __name__ == "__main__":
    system, predictions = test_regime_ffnn()
