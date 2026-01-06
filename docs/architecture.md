# Trading Pipeline Architecture

## Gate and data-flow order
1. **EKF** → estimate level and velocity from the latest close series.
2. **Feature prep** → `[EKF level, EKF velocity, funding rate, 5-bar momentum, 30-bar relative price]`, scaled with the stored `scaler`.
3. **FFNN** → predict 1d mean-reversion direction.
4. **HyperDUM veto** → Hamming distance vs. training memory (`UNCERTAINTY_THRESHOLD`). Any out-of-distribution set **blocks** trades.
5. **Regime multiplier** (main_with_regime only) → computed after HyperDUM using EKF states, realized vol, funding, momentum, and HyperDUM distance as a stability proxy.
6. **Risk sizing** → volatility target + fractional Kelly, then capped by gross exposure limits before orders.

This order is enforced in both live loops to prevent regressions.

## Regime classifier features
- **EKF level & velocity**: capture distance from equilibrium and trend speed.
- **Realized volatility**: 60-day annualized, feeds stability and sizing.
- **Funding rate**: primary exogenous driver for BTC returns.
- **Momentum**: 5-bar mean log return.
- **HyperDUM distance**: dampens stability when feature mixes are unfamiliar.

The classifier returns a multiplier (`0%`, `50%`, `75%`, `100%`) applied **after** HyperDUM passes and **before** risk sizing.

## Placement of the regime multiplier
- `main.py`: no regime filter; HyperDUM veto → risk sizing.
- `main_with_regime.py`: HyperDUM veto → regime multiplier → risk sizing (vol target + Kelly) → exposure cap.

Keep this ordering when adding gates or new signals to avoid unintended size amplification.
