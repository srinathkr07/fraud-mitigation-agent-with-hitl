"""
anomaly_detector.py
===================
Traditional ML layer — IsolationForest-based anomaly detector.

Workflow:
  1. Call `train(transactions_df)` once at startup.
  2. Call `get_risk_assessment(transaction_id)` to score any transaction.

Features used:
  log_amount, hour_of_day, day_of_week, is_weekend, is_nighttime,
  is_international, merchant_risk_score, velocity_1h, velocity_24h,
  amount_zscore (relative to account baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_MERCHANT_RISK_MAP = {"low": 0, "medium": 1, "high": 2}

_FLAG_RULES = [
    # (feature_index, threshold, label)
    (0, np.log1p(500), "HIGH_AMOUNT"),
    (4, 0.5,           "NIGHTTIME_TRANSACTION"),
    (5, 0.5,           "INTERNATIONAL_TRANSACTION"),
    (6, 1.5,           "HIGH_RISK_MERCHANT"),
    (7, 3,             "HIGH_VELOCITY_1H"),
    (8, 15,            "HIGH_VELOCITY_24H"),
    (9, 2.5,           "UNUSUAL_AMOUNT_FOR_ACCOUNT"),
]


class FraudDetector:
    """IsolationForest-based transaction anomaly detector."""

    def __init__(self) -> None:
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.10,
            random_state=42,
            max_samples="auto",
        )
        self.scaler = StandardScaler()
        self._risk_scores: dict[str, int] = {}
        self._flags: dict[str, list[str]] = {}
        self.is_trained = False

    # ── Feature extraction ────────────────────────────────────────────────────
    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return feature matrix aligned with df row order (vectorised per-account)."""
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        n = len(df)

        # Pre-compute scalar columns
        timestamps = pd.to_datetime(df["timestamp"])
        hours = timestamps.dt.hour.values
        dayofweek = timestamps.dt.dayofweek.values

        log_amount = np.log1p(df["amount"].values).astype(np.float32)       # 0
        hour_f = hours.astype(np.float32)                                    # 1
        dow_f = dayofweek.astype(np.float32)                                 # 2
        is_weekend = (dayofweek >= 5).astype(np.float32)                     # 3
        is_night = ((hours < 6) | (hours >= 22)).astype(np.float32)          # 4
        is_intl = df["is_international"].astype(np.float32).values           # 5
        merchant_risk = df["merchant_risk_level"].map(_MERCHANT_RISK_MAP).fillna(0).astype(np.float32).values  # 6

        # Per-account velocity and z-score (vectorised per group)
        vel_1h = np.zeros(n, dtype=np.float32)     # 7
        vel_24h = np.zeros(n, dtype=np.float32)     # 8
        amount_zscore = np.zeros(n, dtype=np.float32)  # 9

        ts_values = timestamps.values  # numpy datetime64 for fast arithmetic
        amounts = df["amount"].values

        for _acc_id, grp in df.groupby("account_id"):
            idx = grp.index.values  # positions in the sorted df
            grp_ts = ts_values[idx]
            grp_amt = amounts[idx]

            # Running expanding mean / std for z-score
            cum_sum = np.cumsum(grp_amt)
            cum_sq = np.cumsum(grp_amt ** 2)

            for pos_in_grp in range(len(idx)):
                i = idx[pos_in_grp]  # index into df
                ts_i = grp_ts[pos_in_grp]

                # Velocity: count of prior group transactions within 1h / 24h
                if pos_in_grp > 0:
                    prior_ts = grp_ts[:pos_in_grp]
                    td = (ts_i - prior_ts).astype("timedelta64[s]").astype(np.float64)
                    vel_1h[i] = float(np.sum(td <= 3600))
                    vel_24h[i] = float(np.sum(td <= 86400))

                    # Z-score from expanding mean/std of prior amounts
                    n_prior = pos_in_grp
                    avg_amt = cum_sum[pos_in_grp - 1] / n_prior
                    if n_prior > 1:
                        var = cum_sq[pos_in_grp - 1] / n_prior - avg_amt ** 2
                        std_amt = max(np.sqrt(max(var, 0.0)), 1e-8)
                    else:
                        std_amt = max(abs(avg_amt), 1.0)
                    zscore = (grp_amt[pos_in_grp] - avg_amt) / std_amt
                    amount_zscore[i] = min(zscore, 10.0)

        features = np.column_stack([
            log_amount, hour_f, dow_f, is_weekend, is_night,
            is_intl, merchant_risk, vel_1h, vel_24h, amount_zscore,
        ])
        return features.astype(np.float32)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> None:
        """Fit the model on transaction data and cache risk scores."""
        features = self._compute_features(df)
        features_scaled = self.scaler.fit_transform(features)

        # Fit on normal transactions for a clean decision boundary
        normal_mask = ~df.sort_values("timestamp")["is_fraud"].values
        self.model.fit(features_scaled[normal_mask])

        # Score all transactions
        raw_scores = self.model.decision_function(features_scaled)

        # Map decision_function output → risk score 0–100
        # IF scores: higher = more normal, lower = more anomalous
        # We invert: risk = clip((−raw + offset) * scale, 0, 100)
        max_s, min_s = raw_scores.max(), raw_scores.min()
        denom = max(max_s - min_s, 1e-8)

        sorted_df = df.sort_values("timestamp").reset_index(drop=True)
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            normalized = (max_s - raw_scores[i]) / denom
            risk_score = int(np.clip(normalized * 100, 0, 100))

            f = features[i]
            anomaly_flags = [
                label
                for idx, threshold, label in _FLAG_RULES
                if f[idx] > threshold
            ]

            tid = row["transaction_id"]
            self._risk_scores[tid] = risk_score
            self._flags[tid] = anomaly_flags

        self.is_trained = True

    # ── Inference ─────────────────────────────────────────────────────────────
    def get_risk_assessment(self, transaction_id: str) -> dict:
        """
        Return a risk assessment dict for a transaction.

        Keys:
            risk_score    int 0–100 (higher = more suspicious)
            risk_level    str  "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            anomaly_flags list[str]
            explanation   str  human-readable summary
        """
        if not self.is_trained:
            return {
                "risk_score": -1,
                "risk_level": "UNKNOWN",
                "anomaly_flags": [],
                "explanation": "Model not trained yet.",
            }

        score = self._risk_scores.get(transaction_id)
        if score is None:
            return {
                "risk_score": -1,
                "risk_level": "UNKNOWN",
                "anomaly_flags": [],
                "explanation": f"Transaction {transaction_id} not found in training data.",
            }

        flags = self._flags.get(transaction_id, [])

        if score >= 80:
            level = "CRITICAL"
        elif score >= 70:
            level = "HIGH"
        elif score >= 35:
            level = "MEDIUM"
        else:
            level = "LOW"

        explanation_parts = [f"ML risk score: {score}/100 ({level})."]
        if flags:
            explanation_parts.append(f"Anomaly flags triggered: {', '.join(flags)}.")
        else:
            explanation_parts.append("No individual anomaly flags triggered.")

        return {
            "risk_score": score,
            "risk_level": level,
            "anomaly_flags": flags,
            "explanation": " ".join(explanation_parts),
        }
