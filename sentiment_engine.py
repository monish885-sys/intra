import os
import re
from dataclasses import dataclass
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass
class SentimentResult:
    label: str
    rationale: str
    sources_used: int


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _fetch_reddit_posts(ticker: str, limit: int = 10) -> List[str]:
    query = f"{ticker} stock OR {ticker} results OR {ticker} India"
    url = "https://www.reddit.com/r/IndiaInvestments/search.json"
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query, "restrict_sr": "on", "sort": "new", "limit": limit}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        posts = []
        for child in payload.get("data", {}).get("children", []):
            data = child.get("data", {})
            title = _clean_text(data.get("title", ""))
            body = _clean_text(data.get("selftext", ""))
            if title:
                posts.append(f"Reddit title: {title}. Body: {body}")
        return posts
    except Exception:
        return []


def _fetch_news_headlines(ticker: str, limit: int = 10) -> List[str]:
    # Google News RSS gives robust anonymous access for headline collection.
    url = (
        "https://news.google.com/rss/search?"
        f"q={ticker}%20stock%20India&hl=en-IN&gl=IN&ceid=IN:en"
    )
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")[:limit]
        return [f"News: {_clean_text(item.title.text)}" for item in items if item.title]
    except Exception:
        return []


def _fetch_x_snippets(ticker: str, limit: int = 10) -> List[str]:
    """
    Optional: uses X recent search API if BEARER token is available.
    If unavailable or rate-limited, returns an empty list gracefully.
    """
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        return []
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": f"({ticker} OR #{ticker}) lang:en -is:retweet",
        "max_results": min(max(limit, 10), 100),
        "tweet.fields": "created_at,text",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [f"X post: {_clean_text(item.get('text', ''))}" for item in data]
    except Exception:
        return []


def collect_market_text(ticker: str) -> List[str]:
    snippets: List[str] = []
    snippets.extend(_fetch_reddit_posts(ticker))
    snippets.extend(_fetch_x_snippets(ticker))
    snippets.extend(_fetch_news_headlines(ticker))
    return [s for s in snippets if s]


def analyze_sentiment_with_gemini(ticker: str, snippets: List[str]) -> SentimentResult:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return SentimentResult(
            label="Neutral",
            rationale="Gemini API key missing; defaulted to neutral.",
            sources_used=len(snippets),
        )

    if not snippets:
        return SentimentResult(
            label="Neutral",
            rationale="No market text snippets fetched; defaulted to neutral.",
            sources_used=0,
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a strict Indian equity sentiment classifier.
Ticker: {ticker}
Task:
1) Read the snippets.
2) Classify sentiment as exactly one of: Bullish, Neutral, Bearish, Controversial.
3) Provide a concise reason.
Output JSON only:
{{"label":"Bullish|Neutral|Bearish|Controversial","rationale":"..."}}

Snippets:
{chr(10).join(snippets[:30])}
"""
    try:
        response = model.generate_content(prompt)
        raw = (response.text or "").strip()
        label = "Neutral"
        rationale = "Model response parsing fallback."
        if "Bearish" in raw:
            label = "Bearish"
        elif "Controversial" in raw:
            label = "Controversial"
        elif "Bullish" in raw:
            label = "Bullish"
        rationale = raw[:600]
        return SentimentResult(label=label, rationale=rationale, sources_used=len(snippets))
    except Exception as exc:
        return SentimentResult(
            label="Neutral",
            rationale=f"Gemini inference failure; default neutral. Error: {exc}",
            sources_used=len(snippets),
        )


def get_ticker_sentiment(ticker: str) -> Dict[str, str]:
    snippets = collect_market_text(ticker)
    result = analyze_sentiment_with_gemini(ticker, snippets)
    return {
        "label": result.label,
        "rationale": result.rationale,
        "sources_used": str(result.sources_used),
    }
