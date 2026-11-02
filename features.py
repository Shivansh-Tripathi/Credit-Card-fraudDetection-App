"""Behavioural feature engineering: velocity, amount deviation, frequency."""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["card_id", "timestamp"]).copy()
    g = df.groupby("card_id", sort=False)

    # Velocity: time since previous txn and txn speed
    df["secs_since_last"] = g["timestamp"].diff().fillna(1e6).clip(lower=0)
    df["velocity"] = 1.0 / (df["secs_since_last"] + 1.0)

    # Amount deviation vs the card's own history (shifted expanding stats -> no leakage)
    exp_mean = g["amount"].transform(lambda s: s.shift().expanding(min_periods=1).mean())
    exp_std = g["amount"].transform(lambda s: s.shift().expanding(min_periods=2).std())
    df["amount_dev"] = (df["amount"] - exp_mean.fillna(df["amount"].mean())) / (exp_std.fillna(df["amount"].std()) + 1e-6)
    df["amount_ratio"] = df["amount"] / (exp_mean.fillna(df["amount"].mean()) + 1e-6)

    # Frequency: number of previous txns by this card within the last 24h
    freq = np.zeros(len(df))
    ts_all = df["timestamp"].to_numpy()
    pos = 0
    for _, idx in g.indices.items():
        t = ts_all[idx]
        left = np.searchsorted(t, t - 86400, side="left")
        freq[idx] = np.arange(len(t)) - left
    df["freq_24h"] = freq

    df["hour"] = (df["timestamp"] // 3600) % 24
    df["is_night"] = df["hour"].isin(range(0, 6)).astype(int)
    df["log_amount"] = np.log1p(df["amount"])

    return df.sort_values("timestamp").reset_index(drop=True)


FEATURES = ["log_amount", "amount_dev", "amount_ratio", "secs_since_last", "velocity",
            "freq_24h", "hour", "is_night", "merchant_category"]
