import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


NIFTY_50_TICKERS = {
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
}


REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}
FEATURE_COLUMNS = [
    "rsi_14",
    "vwap",
    "returns_1",
    "returns_3",
    "intraday_range",
    "price_vs_vwap",
    "volume_zscore_20",
]


@dataclass
class TrainConfig:
    csv_path: Path
    model_out: Path
    metadata_out: Path
    random_state: int = 42


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_features(df: pd.DataFrame, target_threshold: float = 0.06) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])
    df = df[df["ticker"].isin(NIFTY_50_TICKERS)]

    grouped = df.groupby("ticker", group_keys=False)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cum_tpv"] = (typical_price * df["volume"]).groupby(df["ticker"]).cumsum()
    df["cum_vol"] = df.groupby("ticker")["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = df["cum_tpv"] / df["cum_vol"]

    df["rsi_14"] = grouped["close"].apply(_compute_rsi)
    df["returns_1"] = grouped["close"].pct_change(1)
    df["returns_3"] = grouped["close"].pct_change(3)
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    vol_mean = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    vol_std = grouped["volume"].rolling(20).std().reset_index(level=0, drop=True)
    df["volume_zscore_20"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["target_gt_6pct"] = (
        (df["high"] - df["open"]) / df["open"].replace(0, np.nan) >= target_threshold
    ).astype(int)
    return df


def build_training_frame(df: pd.DataFrame, target_threshold: float = 0.06) -> Tuple[pd.DataFrame, pd.Series]:
    feats = _compute_features(df, target_threshold=target_threshold)
    trainable = feats.dropna(subset=FEATURE_COLUMNS + ["target_gt_6pct"]).copy()
    x = trainable[FEATURE_COLUMNS]
    y = trainable["target_gt_6pct"].astype(int)
    return x, y


def train_model(config: TrainConfig) -> Dict[str, float]:
    if not config.csv_path.exists():
        raise FileNotFoundError(
            "Training CSV not found. Pass a valid path in TrainConfig(csv_path=...) "
            f"or place the file at '{config.csv_path}'."
        )
    raw_df = pd.read_csv(config.csv_path)
    _validate_columns(raw_df)
    requested_threshold = float(os.getenv("TARGET_INTRADAY_UPSIDE_THRESHOLD", "0.06"))
    threshold_candidates = [requested_threshold, 0.05, 0.04, 0.03, 0.02]
    threshold_candidates = list(dict.fromkeys(threshold_candidates))

    used_threshold = requested_threshold
    x, y = build_training_frame(raw_df, target_threshold=used_threshold)
    for th in threshold_candidates:
        x, y = build_training_frame(raw_df, target_threshold=th)
        if y.nunique() >= 2:
            used_threshold = th
            break
    else:
        raise ValueError(
            "Target has <2 classes even after fallback thresholds. "
            "Use a larger dataset window or a different data source."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=9,
        random_state=config.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "positive_rate_test": float(y_test.mean()),
        "used_target_threshold": float(used_threshold),
    }
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Training target threshold used: {used_threshold:.2%}")

    config.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "nifty_50": sorted(NIFTY_50_TICKERS),
        },
        config.model_out,
    )

    metadata = {
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "nifty_50_count": len(NIFTY_50_TICKERS),
    }
    config.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    cfg = TrainConfig(
        csv_path=Path("historical.csv"),
        model_out=Path("artifacts/trade_model.joblib"),
        metadata_out=Path("artifacts/training_metadata.json"),
    )
    results = train_model(cfg)
    print("Training complete:", results)
