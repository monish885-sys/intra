from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from model_trainer import NIFTY_50_TICKERS


# Map training tickers to Yahoo NSE format. Fallback uses "{ticker}.NS".
YF_TICKER_MAP: Dict[str, str] = {
    "M&M": "M&M.NS",
    "LT": "LT.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
}


def _to_yf_symbol(ticker: str) -> str:
    return YF_TICKER_MAP.get(ticker, f"{ticker}.NS")


def download_intraday_ohlcv(
    interval: str = "5m",
    period: str = "60d",
    output_path: Path = Path("historical.csv"),
) -> Path:
    rows: List[pd.DataFrame] = []
    for ticker in sorted(NIFTY_50_TICKERS):
        yf_symbol = _to_yf_symbol(ticker)
        try:
            df = yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                continue

            # yfinance can return MultiIndex columns depending on version/settings.
            if isinstance(df.columns, pd.MultiIndex):
                if yf_symbol in df.columns.get_level_values(-1):
                    df = df.xs(yf_symbol, axis=1, level=-1)
                elif yf_symbol in df.columns.get_level_values(0):
                    df = df.xs(yf_symbol, axis=1, level=0)
                else:
                    # Fallback to first symbol block.
                    df.columns = df.columns.get_level_values(0)

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df = df[["open", "high", "low", "close", "volume"]].reset_index()
            # Works for both "Datetime" and "Date" index names.
            dt_col = "Datetime" if "Datetime" in df.columns else "Date"
            df["date"] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
            df["ticker"] = ticker
            rows.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])
        except Exception as exc:
            print(f"Skipping {ticker} ({yf_symbol}) due to fetch error: {exc}")

    if not rows:
        raise RuntimeError("No intraday data downloaded. Check internet connectivity and try again.")

    out = pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    csv_path = download_intraday_ohlcv()
    print(f"Saved intraday dataset to: {csv_path}")
