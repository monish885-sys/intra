import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from string import Template
from typing import Dict, Optional

import joblib
import pandas as pd
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from model_trainer import FEATURE_COLUMNS, NIFTY_50_TICKERS, _compute_features
from sentiment_engine import get_ticker_sentiment


IST = ZoneInfo("Asia/Kolkata")


@dataclass
class TradeDecision:
    approved: bool
    ticker: str
    confidence: float
    expected_upside_pct: float
    capital_required: float
    quantity: int
    estimated_charges: float
    expected_profit_after_charges: float
    sentiment_label: str
    reason: str


class BrokerageCalculator:
    def __init__(self, yaml_path: Path):
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.brokerage_percent = float(config["brokerage"]["intraday_percent"])
        self.brokerage_cap = float(config["brokerage"]["intraday_cap_per_order_inr"])
        self.stt_sell_percent = float(config["stt"]["intraday_sell_percent"])
        self.exchange_percent = float(config["exchange_txn"]["nse_equity_percent"])
        self.sebi_per_crore = float(config["sebi"]["per_crore_inr"])
        self.gst_percent = float(config["gst"]["percent_on_brokerage_and_txn"])
        self.stamp_buy_percent = float(config["stamp_duty"]["buy_side_percent"])

    def estimate_round_trip_charges(self, buy_value: float, sell_value: float) -> float:
        buy_brokerage = min(self.brokerage_cap, buy_value * self.brokerage_percent / 100.0)
        sell_brokerage = min(self.brokerage_cap, sell_value * self.brokerage_percent / 100.0)
        brokerage_total = buy_brokerage + sell_brokerage

        stt = sell_value * self.stt_sell_percent / 100.0
        exchange_txn = (buy_value + sell_value) * self.exchange_percent / 100.0
        sebi = ((buy_value + sell_value) / 10_000_000.0) * self.sebi_per_crore
        stamp = buy_value * self.stamp_buy_percent / 100.0
        gst = (brokerage_total + exchange_txn) * self.gst_percent / 100.0

        return brokerage_total + stt + exchange_txn + sebi + stamp + gst

    def evaluate_trade(
        self,
        last_price: float,
        expected_upside_pct: float,
        capital_min: int = 1000,
        capital_max: int = 8000,
        target_profit: int = 500,
    ) -> Dict[str, float]:
        for capital in range(capital_min, capital_max + 1, 100):
            qty = int(capital // last_price)
            if qty <= 0:
                continue
            buy_value = qty * last_price
            sell_price = last_price * (1 + expected_upside_pct / 100.0)
            sell_value = qty * sell_price
            gross_profit = sell_value - buy_value
            charges = self.estimate_round_trip_charges(buy_value, sell_value)
            net_profit = gross_profit - charges
            if expected_upside_pct >= 6.5 and net_profit >= target_profit:
                return {
                    "capital_required": buy_value,
                    "quantity": qty,
                    "estimated_charges": charges,
                    "expected_profit_after_charges": net_profit,
                    "sell_price": sell_price,
                }
        return {}


def load_latest_features(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            "Historical CSV not found. Set HISTORICAL_CSV_PATH or place file at "
            f"'{csv_path}'."
        )
    df = pd.read_csv(csv_path)
    features_df = _compute_features(df)
    features_df = features_df[features_df["ticker"].isin(NIFTY_50_TICKERS)].copy()
    latest = (
        features_df.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
    )
    latest = latest.dropna(subset=FEATURE_COLUMNS + ["close"])
    return latest


def score_candidates(model_path: Path, latest_df: pd.DataFrame) -> pd.DataFrame:
    blob = joblib.load(model_path)
    model = blob["model"]
    x = latest_df[blob["feature_columns"]]
    latest_df = latest_df.copy()
    latest_df["prob_gt_6pct"] = model.predict_proba(x)[:, 1]
    latest_df["expected_upside_pct"] = 6.0 + (latest_df["prob_gt_6pct"] * 3.0)
    return latest_df.sort_values("prob_gt_6pct", ascending=False)


def choose_trade(
    ranked_df: pd.DataFrame, charges: BrokerageCalculator
) -> Optional[TradeDecision]:
    for _, row in ranked_df.iterrows():
        ticker = row["ticker"]
        sentiment = get_ticker_sentiment(ticker)
        if sentiment["label"] in {"Bearish", "Controversial"}:
            continue

        feasibility = charges.evaluate_trade(
            last_price=float(row["close"]),
            expected_upside_pct=float(row["expected_upside_pct"]),
        )
        if not feasibility:
            continue

        return TradeDecision(
            approved=True,
            ticker=ticker,
            confidence=float(row["prob_gt_6pct"]),
            expected_upside_pct=float(row["expected_upside_pct"]),
            capital_required=float(feasibility["capital_required"]),
            quantity=int(feasibility["quantity"]),
            estimated_charges=float(feasibility["estimated_charges"]),
            expected_profit_after_charges=float(feasibility["expected_profit_after_charges"]),
            sentiment_label=sentiment["label"],
            reason=sentiment["rationale"],
        )
    return None


def render_email_html(template_path: Path, decision: Optional[TradeDecision]) -> str:
    template = Template(template_path.read_text(encoding="utf-8"))
    now_ist = datetime.now(IST).strftime("%d %b %Y %I:%M %p IST")
    if not decision:
        return template.substitute(
            report_time=now_ist,
            status="NO APPROVED TRADE",
            ticker="N/A",
            confidence="N/A",
            expected_upside="N/A",
            capital_required="N/A",
            quantity="N/A",
            charges="N/A",
            net_profit="N/A",
            sentiment="N/A",
            reason="No ticker met combined technical, sentiment, and risk filters today.",
        )
    return template.substitute(
        report_time=now_ist,
        status="APPROVED TRADE SIGNAL",
        ticker=decision.ticker,
        confidence=f"{decision.confidence:.2%}",
        expected_upside=f"{decision.expected_upside_pct:.2f}%",
        capital_required=f"Rs {decision.capital_required:,.2f}",
        quantity=str(decision.quantity),
        charges=f"Rs {decision.estimated_charges:,.2f}",
        net_profit=f"Rs {decision.expected_profit_after_charges:,.2f}",
        sentiment=decision.sentiment_label,
        reason=decision.reason,
    )


def send_email(subject: str, html_body: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("EMAIL_FROM", smtp_user or "")
    recipient = os.getenv("EMAIL_TO")
    if not all([smtp_host, smtp_user, smtp_password, recipient]):
        raise RuntimeError("SMTP env vars missing. Set SMTP_HOST/SMTP_USER/SMTP_PASSWORD/EMAIL_TO.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, [recipient], msg.as_string())


def run_pipeline() -> None:
    csv_path = Path(os.getenv("HISTORICAL_CSV_PATH", "historical.csv"))
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/trade_model.joblib"))
    template_path = Path("email_template.html")
    charges_path = Path("charges_config.yaml")

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model file not found. Train first via "
            "`python model_trainer.py` or set MODEL_PATH."
        )

    ranked = score_candidates(model_path, load_latest_features(csv_path))
    calculator = BrokerageCalculator(charges_path)
    decision = choose_trade(ranked, calculator)

    html = render_email_html(template_path, decision)
    subject = f"[Intraday Signal] {datetime.now(IST).strftime('%d-%m-%Y')} | {decision.ticker if decision else 'No Trade'}"
    delivery_mode = os.getenv("DELIVERY_MODE", "email").lower()
    if delivery_mode == "cli":
        print(subject)
        if decision:
            print(
                "Approved trade:",
                {
                    "ticker": decision.ticker,
                    "confidence": round(decision.confidence, 4),
                    "expected_upside_pct": round(decision.expected_upside_pct, 2),
                    "capital_required": round(decision.capital_required, 2),
                    "quantity": decision.quantity,
                    "estimated_charges": round(decision.estimated_charges, 2),
                    "expected_profit_after_charges": round(decision.expected_profit_after_charges, 2),
                    "sentiment_label": decision.sentiment_label,
                },
            )
        else:
            print("No approved trade for today.")
        print("Pipeline completed in CLI mode (no email sent).")
        return

    send_email(subject=subject, html_body=html)
    print("Pipeline completed. Email sent.")


def run_scheduler() -> None:
    scheduler = BlockingScheduler(timezone=IST)
    scheduler.add_job(
        run_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=11, minute=0, second=0, timezone=IST),
        id="daily_intraday_signal",
        replace_existing=True,
    )
    print("Scheduler started. Daily run at 11:00 AM IST (Mon-Fri).")
    scheduler.start()


if __name__ == "__main__":
    run_mode = os.getenv("RUN_MODE", "once").lower()
    if run_mode == "scheduler":
        run_scheduler()
    else:
        run_pipeline()
