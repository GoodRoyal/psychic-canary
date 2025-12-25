import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import entropy

class PsychicCanaryEngine:
    def __init__(self, risk_aversion=1.0, vol_window=30, entropy_window=60):
        self.risk_aversion = risk_aversion
        self.vol_window = vol_window
        self.entropy_window = entropy_window

    def compute_cov_matrix(self, returns: pd.DataFrame):
        return returns.cov() * 252  # Annualized

    def fisher_metric(self, weights: np.ndarray):
        # Fisher information metric on simplex
        return np.diag(1 / (weights + 1e-8))

    def full_metric(self, weights: np.ndarray, cov: pd.DataFrame):
        # Hybrid: Fisher + financial covariance
        n = len(weights)
        fisher = self.fisher_metric(weights)
        return 0.5 * (fisher + np.eye(n)) + cov.values

    def portfolio_entropy(self, weights: np.ndarray):
        return -np.sum(weights * np.log(weights + 1e-10))

    def portfolio_energy(self, mu: np.ndarray, weights: np.ndarray, cov: pd.DataFrame):
        return np.dot(mu, weights) - self.risk_aversion * np.dot(weights.T, np.dot(cov, weights))

    def free_energy(self, weights: np.ndarray, mu: np.ndarray, cov: pd.DataFrame, T: float):
        S = self.portfolio_entropy(weights)
        E = self.portfolio_energy(mu, weights, cov)
        return E - T * S

    def optimal_weights(self, mu: np.ndarray, cov: pd.DataFrame, T: float):
        n = len(mu)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]
        init_w = np.ones(n) / n
        result = minimize(
            lambda w: self.free_energy(w, mu, cov, T),
            init_w,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else init_w

    def compute_temperature(self, market_vol: float):
        # Volatility-scaled exploration (from your provisional)
        return market_vol * 2.0  # Tune based on your sims

    def infer_phase(self, df: pd.DataFrame, tickers: list = None):
        if tickers is None:
            tickers = list(df.columns)
        returns = df.pct_change().dropna()

        mu = returns.mean().values * 252
        cov = self.compute_cov_matrix(returns)

        # Current volatility (last 20 days vs full period)
        recent_vol = returns.tail(20).std().mean() * np.sqrt(252)
        historical_vol = returns.std().mean() * np.sqrt(252)
        vol_ratio = recent_vol / (historical_vol + 1e-8)

        T = self.compute_temperature(recent_vol)
        weights = self.optimal_weights(mu, cov, T)

        F = self.free_energy(weights, mu, cov, T)
        S = self.portfolio_entropy(weights)

        # Enhanced early warning: curvature from d²F/dT² approximation
        # When vol spikes relative to history, curvature increases
        curvature = vol_ratio * (1 + abs(F))

        # Multi-factor warning system:
        # 1. High temperature (volatility regime)
        # 2. Volatility spike (recent >> historical)
        # 3. Free energy becoming more negative (stress)
        temp_warning = T > 0.30  # High volatility regime
        vol_spike = vol_ratio > 1.5  # 50% vol increase
        stress = F < -0.5  # Negative free energy = stress

        warning = temp_warning and (vol_spike or stress)

        # Phase determination
        if warning:
            phase = "crisis_imminent"
        elif T > 0.25:
            phase = "exploration"  # High uncertainty
        else:
            phase = "exploitation"  # Stable regime

        # Action based on phase
        if phase == "crisis_imminent":
            action = "REDUCE RISK: Move to bonds/cash immediately"
        elif phase == "exploration":
            action = "CAUTION: Increase diversification, reduce leverage"
        else:
            action = "STABLE: Hold current allocation"

        return {
            "tickers": tickers,
            "phase": phase,
            "warning": bool(warning),
            "temperature": float(T),
            "entropy": float(S),
            "free_energy": float(F),
            "curvature_proxy": float(curvature),
            "vol_ratio": float(vol_ratio),
            "optimal_weights": {tickers[i]: float(w) for i, w in enumerate(weights)},
            "suggested_action": action,
            "holonomic_note": "Thermodynamic regime detection active",
            "data_points": int(len(returns))
        }
