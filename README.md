# Intraday Sentiment Trader

India equities helper that blends **news/Reddit sentiment** with a trained intraday model and optional email alerts.

## Core idea

Price alone is late. This stack fetches short-horizon context, scores sentiment with Gemini, and feeds a joblib trade model trained on historical bars.

## Components

| File | Role |
|------|------|
| `fetch_intraday_data.py` | Pull / prepare intraday bars |
| `sentiment_engine.py` | Scrape + Gemini sentiment label |
| `model_trainer.py` | Train / refresh the trade model |
| `main.py` | Run loop + alert path |
| `artifacts/` | Saved model + training metadata |
| `charges_config.yaml` | Brokerage / charge assumptions |

## Stack

Python · scikit-learn / joblib · BeautifulSoup · Google Gemini · SMTP

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # EMAIL_* and GEMINI keys — never commit secrets
python main.py
```

## Caution

Educational / research prototype. Not investment advice. Paper-trade before anything live.
