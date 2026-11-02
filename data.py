"""Data loading for fraud detection.

Uses a real CSV if provided (Kaggle 'creditcardfraud' format: Time, V1..V28, Amount, Class)
or a real transaction log with columns: card_id, timestamp, amount, merchant_category, is_fraud.
Otherwise generates a synthetic, highly imbalanced transaction log so the pipeline runs end to end.
"""
import numpy as np
import pandas as pd


def generate_synthetic(n_cards=3000, n_tx=200_000, fraud_rate=0.0035, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    card_id = rng.integers(0, n_cards, n_tx)
    ts = np.sort(rng.integers(0, 90 * 24 * 3600, n_tx))  # 90 days, seconds
    card_mean = rng.lognormal(3.5, 0.5, n_cards)
    amount = rng.lognormal(np.log(card_mean[card_id]), 0.6)
    category = rng.integers(0, 12, n_tx)
    is_fraud = (rng.random(n_tx) < fraud_rate).astype(int)

    # Fraud behaves differently: larger amounts, odd hours, risky categories
    f = is_fraud == 1
    amount[f] *= rng.uniform(2.5, 9.0, f.sum())
    category[f] = rng.choice([3, 7, 9, 11], f.sum())
    hour_shift = rng.integers(0, 6, f.sum()) * 3600  # push to night
    ts[f] = (ts[f] // 86400) * 86400 + hour_shift
    df = pd.DataFrame({"card_id": card_id, "timestamp": ts, "amount": amount.round(2),
                       "merchant_category": category, "is_fraud": is_fraud})
    return df.sort_values("timestamp").reset_index(drop=True)


def load_transactions(path: str | None = None) -> pd.DataFrame:
    if path is None:
        return generate_synthetic()
    df = pd.read_csv(path)
    if {"Time", "Amount", "Class"}.issubset(df.columns):  # Kaggle format
        rng = np.random.default_rng(0)
        df = df.rename(columns={"Time": "timestamp", "Amount": "amount", "Class": "is_fraud"})
        if "card_id" not in df:  # Kaggle data has no card id: assign pseudo ids
            df["card_id"] = rng.integers(0, 2000, len(df))
        if "merchant_category" not in df:
            df["merchant_category"] = 0
    return df.sort_values("timestamp").reset_index(drop=True)
