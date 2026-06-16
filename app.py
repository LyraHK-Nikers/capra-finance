"""
CAPRA Finance — global market intelligence dashboard.

Live stocks (USA, Hong Kong, India, Europe, Asia) with comparison, CAGR, risk,
forecasts, financials, sector & market-cap analytics, plus Polymarket prediction
markets — all in one app.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from tickers import MARKET_PRESETS, NASDAQ_TRADER_SOURCES, WIKIPEDIA_INDEX_SOURCES

STORAGE_PATH = Path(__file__).parent / "user_state.json"


def load_storage() -> dict:
    if not STORAGE_PATH.exists():
        return {"portfolio": [], "alerts": []}
    try:
        return json.loads(STORAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"portfolio": [], "alerts": []}


def save_storage(data: dict) -> None:
    STORAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Optional: statsmodels for a slightly nicer forecast — fall back gracefully.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# Market presets are imported from tickers.py (USA / HK / India / Europe / Asia
# with ~100 large-caps each). Live index-constituent loading via Wikipedia is
# offered as an opt-in toggle in the sidebar — see fetch_index_constituents().

# Benchmark used for beta — S&P 500 by default, swappable in the sidebar.
BENCHMARKS = {
    "S&P 500 (USA)": "^GSPC",
    "Hang Seng (HK)": "^HSI",
    "NIFTY 50 (India)": "^NSEI",
    "STOXX 600 (Europe)": "^STOXX",
    "Nikkei 225 (Japan)": "^N225",
}

PERIOD_TO_YEARS = {"1y": 1, "3y": 3, "5y": 5, "10y": 10}


# ---------------------------------------------------------------------------
# Data fetching (cached). yfinance is rate-limited so we cache aggressively.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(tickers: tuple[str, ...], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Return adjusted close prices as a wide DataFrame indexed by date."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        list(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex when multiple tickers are passed.
    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            try:
                closes[t] = data[t]["Close"]
            except (KeyError, ValueError):
                continue
        df = pd.DataFrame(closes)
    else:
        df = data[["Close"]].rename(columns={"Close": tickers[0]})

    return df.dropna(how="all")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_volume(tickers: tuple[str, ...], period: str = "1mo") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        vols = {}
        for t in tickers:
            try:
                vols[t] = data[t]["Volume"]
            except (KeyError, ValueError):
                continue
        return pd.DataFrame(vols).dropna(how="all")
    return data[["Volume"]].rename(columns={"Volume": tickers[0]})


@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Most recent 1-day, 5-minute bars — used for the live-ish quote tiles."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        list(tickers),
        period="1d",
        interval="5m",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            try:
                out[t] = data[t]["Close"]
            except (KeyError, ValueError):
                continue
        return pd.DataFrame(out)
    return data[["Close"]].rename(columns={"Close": tickers[0]})


@st.cache_data(ttl=600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    """Fundamentals + analyst targets. Cached longer since this rarely changes intraday."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker: str, limit: int = 5) -> list[dict]:
    """Latest news items for a ticker via yfinance. Returns title/publisher/link/time."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out: list[dict] = []
    for item in raw[:limit]:
        # yfinance has shipped a few different shapes for `.news` over time — handle both.
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title:
            continue
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else content.get("publisher") or item.get("publisher")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link") or item.get("link")
        )
        pub_time = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub_time, (int, float)):
            pub_time = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
        out.append({"title": title, "publisher": publisher or "—", "link": link or "", "time": pub_time or ""})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector(ticker: str) -> tuple[str, str]:
    """Return (sector, industry) for a ticker. Cached 1h since sectors don't change."""
    info = fetch_info(ticker)
    return info.get("sector") or "Unknown", info.get("industry") or "Unknown"


POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"


def _http_get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockTracker/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@st.cache_data(ttl=300, show_spinner=False)
def fetch_symbol_search(query: str, limit: int = 8) -> list[dict]:
    """Yahoo Finance symbol search — match by company name OR ticker.

    Returns [{symbol, name, exchange, type}] for equities/ETFs worldwide.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return []
    try:
        import urllib.parse
        url = "https://query2.finance.yahoo.com/v1/finance/search?q=" + urllib.parse.quote(q)
        data = _http_get_json(url, timeout=10)
    except Exception:
        return []
    out: list[dict] = []
    for it in (data.get("quotes", []) if isinstance(data, dict) else []):
        if it.get("quoteType") not in ("EQUITY", "ETF"):
            continue
        sym = it.get("symbol")
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "name": it.get("shortname") or it.get("longname") or sym,
            "exchange": it.get("exchDisp") or "",
            "type": it.get("quoteType"),
        })
        if len(out) >= limit:
            break
    return out


@st.cache_data(ttl=60, show_spinner=False)
def fetch_polymarket_events(limit: int = 150) -> list[dict]:
    """Top active Polymarket events sorted by 24h volume."""
    try:
        return _http_get_json(
            f"{POLYMARKET_GAMMA}/events"
            f"?limit={limit}&active=true&closed=false&order=volume24hr&ascending=false"
        )
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_polymarket_markets(limit: int = 300) -> list[dict]:
    """Individual Polymarket markets — used for Top Movers (24h price change)."""
    try:
        return _http_get_json(
            f"{POLYMARKET_GAMMA}/markets"
            f"?limit={limit}&active=true&closed=false&order=volume24hr&ascending=false"
        )
    except Exception:
        return []


def _parse_outcomes_prices(market: dict) -> tuple[list[str], list[float]]:
    outs = market.get("outcomes")
    prices = market.get("outcomePrices")
    if isinstance(outs, str):
        try:
            outs = json.loads(outs)
        except Exception:
            outs = []
    if isinstance(prices, str):
        try:
            prices = [float(p) for p in json.loads(prices)]
        except Exception:
            prices = []
    return outs or [], prices or []


def _yes_price(market: dict) -> float | None:
    """The implied YES probability — first outcome that's labeled 'Yes' or, failing that, the first price."""
    outs, prices = _parse_outcomes_prices(market)
    if not outs or not prices:
        return None
    for o, p in zip(outs, prices):
        if str(o).strip().lower() == "yes":
            return p
    return prices[0] if prices else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_index_constituents(index_name: str) -> dict[str, str]:
    """Scrape current constituents of a major index from Wikipedia. Cached 24h.

    Returns {yahoo_symbol: company_name}. Empty dict on failure — caller should
    fall back to curated presets.
    """
    cfg = WIKIPEDIA_INDEX_SOURCES.get(index_name)
    if not cfg:
        return {}
    try:
        # Wikipedia blocks pandas' default UA — fetch via urllib with a real one.
        req = urllib.request.Request(
            cfg["url"],
            headers={"User-Agent": "Mozilla/5.0 (StockTracker/1.0; research)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        tables = pd.read_html(StringIO(html))
    except Exception:
        return {}

    # Wikipedia table indices drift; scan tables for one with the expected columns.
    candidates = [tables[cfg["table_index"]]] if cfg["table_index"] < len(tables) else []
    candidates += [t for t in tables if t is not (candidates[0] if candidates else None)]

    suffix = cfg["suffix"]
    sym_col_target = cfg["symbol_col"].lower()
    name_col_target = cfg["name_col"].lower()

    for tbl in candidates:
        # Flatten MultiIndex columns (some Wikipedia tables use them) and stringify.
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl = tbl.copy()
            tbl.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip() for c in tbl.columns]
        cols = {str(c).lower(): c for c in tbl.columns}
        sym_col = next((cols[c] for c in cols if sym_col_target in c), None)
        name_col = next((cols[c] for c in cols if name_col_target in c), None)
        if not sym_col or not name_col or sym_col == name_col:
            continue
        out: dict[str, str] = {}
        for _, row in tbl.iterrows():
            sym = str(row[sym_col]).strip()
            name = str(row[name_col]).strip()
            if sym.endswith(".0"):
                sym = sym[:-2]
            # Strip exchange prefixes like "SEHK:", "KRX:", "NYSE:", "NSE:"; also strip Unicode noise.
            sym = re.sub(r"^[A-Za-z]+\s*:\s*", "", sym)
            sym = re.sub(r"[^\w.\-]", "", sym)  # drop non-alphanumeric except . - _
            if not sym or sym.lower() in ("nan", "none"):
                continue

            if suffix == ".HK":
                digits = re.findall(r"\d+", sym)
                sym = (digits[0].zfill(4) + ".HK") if digits else None
            elif suffix == ".KS":
                digits = re.findall(r"\d+", sym)
                sym = (digits[0].zfill(6) + ".KS") if digits else None
            elif suffix == "":  # US: BRK.B -> BRK-B, otherwise leave alone
                if "." in sym and len(sym) <= 6:
                    sym = sym.replace(".", "-")
            elif suffix and "." in sym:
                # Symbol already has a Yahoo-style exchange code (e.g., AIR.PA on the DAX page).
                pass
            elif suffix:
                sym = sym + suffix

            if sym:
                out[sym] = name
        if out:
            return out
    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq_trader_listing(source_name: str) -> dict[str, str]:
    """Pipe-delimited symbol-directory files from NASDAQ Trader (nightly refresh).

    Gives full coverage of every US-listed common stock — far more than the
    NASDAQ-100 or S&P 500. Skips ETFs and test issues. Normalises class-share
    symbols to Yahoo Finance format (BRK.B → BRK-B).
    """
    cfg = NASDAQ_TRADER_SOURCES.get(source_name)
    if not cfg:
        return {}
    try:
        req = urllib.request.Request(
            cfg["url"],
            headers={"User-Agent": "Mozilla/5.0 (CAPRA-Finance research bot)"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return {}

    lines = text.strip().splitlines()
    if not lines:
        return {}

    header = [h.strip() for h in lines[0].split("|")]
    cols = {h.lower(): i for i, h in enumerate(header)}

    def _find_col(target: str) -> int | None:
        target_lc = target.lower()
        for h_lc, idx in cols.items():
            if target_lc == h_lc:
                return idx
        for h_lc, idx in cols.items():
            if target_lc in h_lc:
                return idx
        return None

    sym_idx = _find_col(cfg["symbol_col"])
    name_idx = _find_col(cfg["name_col"])
    etf_idx = _find_col("ETF")
    test_idx = _find_col("Test Issue")
    if sym_idx is None or name_idx is None:
        return {}

    out: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) <= max(sym_idx, name_idx):
            continue
        sym = parts[sym_idx].strip()
        if not sym:
            continue
        if cfg.get("exclude_etf") and etf_idx is not None and len(parts) > etf_idx and parts[etf_idx].strip().upper() == "Y":
            continue
        if test_idx is not None and len(parts) > test_idx and parts[test_idx].strip().upper() == "Y":
            continue
        # Yahoo uses BRK-B not BRK.B; preferred / class-share dots become dashes.
        if "." in sym and len(sym) <= 6:
            sym = sym.replace(".", "-")
        out[sym] = parts[name_idx].strip()
    return out


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@dataclass
class StockMetrics:
    ticker: str
    name: str
    price: float
    pct_change_1d: float
    pct_change_1m: float
    pct_change_3m: float
    pct_change_ytd: float
    cagr_1y: float
    cagr_3y: float
    cagr_5y: float
    volatility: float          # annualized stdev of daily log-returns
    sharpe: float              # annualized, rf=0
    max_drawdown: float        # worst peak-to-trough
    beta: float                # vs chosen benchmark
    avg_dollar_volume: float   # last 20 trading days
    rsi_14: float
    forecast_30d: float        # projected price in 30 trading days
    forecast_upside: float     # (forecast / price) - 1
    analyst_target: float | None
    analyst_upside: float | None
    pe_ratio: float | None
    risk_level: str            # Low / Medium / High / Very High
    best_buy_score: float      # 0-100 composite
    recommendation: str        # Strong Buy / Buy / Hold / Reduce / Sell
    annual_revenue: float | None       # TTM total revenue
    revenue_yoy: float | None          # YoY revenue growth (decimal)
    net_income: float | None           # TTM net income
    earnings_yoy: float | None         # YoY earnings growth (decimal)
    sales_growth_qoq: float | None     # Most-recent quarter revenue growth (decimal)
    market_cap: float | None           # USD market capitalization
    cap_category: str                  # Mega / Large / Mid / Small / Micro / Penny / Unknown
    period_returns: dict               # {"3M":x,"6M":x,"1Y":x,"3Y":x,"5Y":x} cumulative returns


# Market-cap buckets (USD). Yahoo `marketCap` is reported in the listing currency,
# but for the major exchanges we cover (US/HK/India/Europe/Japan/Korea/AU) the
# field is already converted to USD by yfinance.
CAP_BUCKETS = [
    ("Mega Cap",   200e9, float("inf"), "#a78bfa"),
    ("Large Cap",   10e9,        200e9, "#6366f1"),
    ("Mid Cap",      2e9,         10e9, "#3b82f6"),
    ("Small Cap",  500e6,          2e9, "#06b6d4"),
    ("Micro Cap",   50e6,        500e6, "#f59e0b"),
    ("Penny/Nano",     0,         50e6, "#ef4444"),
]
CAP_COLORS = {name: color for name, _, _, color in CAP_BUCKETS}


def _cap_category(market_cap: float | None) -> str:
    if market_cap is None or (isinstance(market_cap, float) and math.isnan(market_cap)) or market_cap <= 0:
        return "Unknown"
    for name, lo, hi, _ in CAP_BUCKETS:
        if lo <= market_cap < hi:
            return name
    return "Unknown"


def _cagr(series: pd.Series, years: float) -> float:
    """Compound annual growth rate over the trailing `years` window."""
    if series.empty or len(series) < 2:
        return float("nan")
    cutoff = series.index[-1] - pd.Timedelta(days=int(years * 365.25))
    window = series.loc[series.index >= cutoff].dropna()
    if len(window) < 2 or window.iloc[0] <= 0:
        return float("nan")
    total_return = window.iloc[-1] / window.iloc[0]
    actual_years = (window.index[-1] - window.index[0]).days / 365.25
    if actual_years <= 0:
        return float("nan")
    return total_return ** (1 / actual_years) - 1


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    cummax = series.cummax()
    drawdown = series / cummax - 1
    return float(drawdown.min())


def _beta(stock_returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.concat([stock_returns, bench_returns], axis=1, sort=False).dropna()
    if len(df) < 30:
        return float("nan")
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    return float(cov / var) if var > 0 else float("nan")


def _rsi(prices: pd.Series, window: int = 14) -> float:
    if len(prices) < window + 1:
        return float("nan")
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def _forecast(prices: pd.Series, horizon: int = 30) -> tuple[float, np.ndarray]:
    """Forecast `horizon` trading days ahead. Returns (point_estimate, full_path)."""
    series = prices.dropna()
    if len(series) < 60:
        return float("nan"), np.array([])

    if HAS_STATSMODELS and len(series) >= 120:
        try:
            model = ExponentialSmoothing(
                series.values, trend="add", seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)
            forecast_path = model.forecast(horizon)
            return float(forecast_path[-1]), np.asarray(forecast_path)
        except Exception:
            pass

    # Fallback: linear regression on log-prices (geometric trend).
    log_prices = np.log(series.values)
    x = np.arange(len(log_prices)).reshape(-1, 1)
    model = LinearRegression().fit(x, log_prices)
    future_x = np.arange(len(log_prices), len(log_prices) + horizon).reshape(-1, 1)
    forecast_path = np.exp(model.predict(future_x))
    return float(forecast_path[-1]), forecast_path


def _risk_level(volatility: float, max_dd: float) -> str:
    if math.isnan(volatility):
        return "Unknown"
    vol_score = volatility  # already annualized stdev
    dd_score = abs(max_dd) if not math.isnan(max_dd) else 0
    combined = vol_score * 0.6 + dd_score * 0.4
    if combined < 0.20:
        return "Low"
    if combined < 0.35:
        return "Medium"
    if combined < 0.55:
        return "High"
    return "Very High"


def _best_buy_score(m: dict) -> tuple[float, str]:
    """Composite 0-100 score. Bigger = better buy candidate.

    Blends: momentum, forecast upside, analyst upside, valuation, risk-adjusted return.
    """
    score = 50.0  # neutral baseline

    # Momentum (3M % change), capped at +/- 30%
    if not math.isnan(m["pct_change_3m"]):
        score += np.clip(m["pct_change_3m"] * 100, -30, 30) * 0.4

    # Forecast upside, capped at +/- 25%
    if not math.isnan(m["forecast_upside"]):
        score += np.clip(m["forecast_upside"] * 100, -25, 25) * 0.5

    # Analyst upside if available
    if m["analyst_upside"] is not None and not math.isnan(m["analyst_upside"]):
        score += np.clip(m["analyst_upside"] * 100, -30, 30) * 0.4

    # Sharpe ratio bonus (positive = good)
    if not math.isnan(m["sharpe"]):
        score += np.clip(m["sharpe"] * 10, -15, 15)

    # Penalty for very-high volatility
    if not math.isnan(m["volatility"]) and m["volatility"] > 0.45:
        score -= (m["volatility"] - 0.45) * 50

    # P/E sanity (very high P/E -> small penalty)
    if m["pe_ratio"] is not None and m["pe_ratio"] > 0:
        if m["pe_ratio"] > 50:
            score -= min((m["pe_ratio"] - 50) * 0.3, 10)

    # RSI sanity — extremes are warnings
    if not math.isnan(m["rsi_14"]):
        if m["rsi_14"] > 75:
            score -= 5  # overbought
        elif m["rsi_14"] < 25:
            score += 3  # potentially oversold = opportunity

    score = float(np.clip(score, 0, 100))

    if score >= 75:
        rec = "Strong Buy"
    elif score >= 60:
        rec = "Buy"
    elif score >= 45:
        rec = "Hold"
    elif score >= 30:
        rec = "Reduce"
    else:
        rec = "Sell"
    return score, rec


def compute_metrics(
    ticker: str,
    name: str,
    prices: pd.Series,
    benchmark_returns: pd.Series,
    volumes: pd.Series | None,
) -> StockMetrics:
    prices = prices.dropna()
    if prices.empty:
        # Return a row of NaNs so the dashboard still renders gracefully.
        return StockMetrics(
            ticker=ticker, name=name, price=float("nan"),
            pct_change_1d=float("nan"), pct_change_1m=float("nan"),
            pct_change_3m=float("nan"), pct_change_ytd=float("nan"),
            cagr_1y=float("nan"), cagr_3y=float("nan"), cagr_5y=float("nan"),
            volatility=float("nan"), sharpe=float("nan"),
            max_drawdown=float("nan"), beta=float("nan"),
            avg_dollar_volume=float("nan"), rsi_14=float("nan"),
            forecast_30d=float("nan"), forecast_upside=float("nan"),
            analyst_target=None, analyst_upside=None, pe_ratio=None,
            risk_level="Unknown", best_buy_score=float("nan"),
            recommendation="N/A",
            annual_revenue=None, revenue_yoy=None, net_income=None,
            earnings_yoy=None, sales_growth_qoq=None,
            market_cap=None, cap_category="Unknown",
            period_returns={},
        )

    price = float(prices.iloc[-1])
    daily_returns = prices.pct_change().dropna()

    def _pct(window_days: int) -> float:
        if len(prices) < 2:
            return float("nan")
        cutoff = prices.index[-1] - pd.Timedelta(days=window_days)
        past = prices.loc[prices.index <= cutoff]
        if past.empty:
            return float("nan")
        return float(price / past.iloc[-1] - 1)

    pct_1d = float(daily_returns.iloc[-1]) if not daily_returns.empty else float("nan")
    pct_1m = _pct(30)
    pct_3m = _pct(90)
    # YTD — match year on the index without constructing a tz-aware Timestamp
    last_year = prices.index[-1].year
    ytd_window = prices[prices.index.year == last_year]
    pct_ytd = float(price / ytd_window.iloc[0] - 1) if not ytd_window.empty else float("nan")

    # Cumulative price-change % over standard look-back windows (for the card toggle).
    # Measures from the FIRST price inside the window (robust when the fetched
    # history is exactly the window length, e.g. 5Y data for a 5Y look-back).
    def _period_ret(days: int) -> float:
        if len(prices) < 2:
            return float("nan")
        cutoff = prices.index[-1] - pd.Timedelta(days=days)
        window = prices.loc[prices.index >= cutoff]
        if len(window) < 2 or window.iloc[0] <= 0:
            return float("nan")
        # Require the window to actually span most of the requested period;
        # otherwise (e.g. a 1Y data fetch asked for a 5Y return) report "—".
        span_days = (prices.index[-1] - window.index[0]).days
        if span_days < days * 0.6:
            return float("nan")
        return float(price / window.iloc[0] - 1)

    period_returns = {
        "3M": _period_ret(91),
        "6M": _period_ret(182),
        "1Y": _period_ret(365),
        "3Y": _period_ret(365 * 3),
        "5Y": _period_ret(365 * 5),
    }

    vol = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else float("nan")
    mean_ret = float(daily_returns.mean() * 252) if not daily_returns.empty else float("nan")
    sharpe = mean_ret / vol if vol and vol > 0 else float("nan")
    mdd = _max_drawdown(prices)
    beta = _beta(daily_returns, benchmark_returns)

    if volumes is not None and not volumes.empty:
        recent_vol = volumes.tail(20).dropna()
        avg_dollar_vol = float((recent_vol * prices.reindex(recent_vol.index)).mean())
    else:
        avg_dollar_vol = float("nan")

    rsi = _rsi(prices)
    forecast_price, _ = _forecast(prices, horizon=30)
    forecast_upside = forecast_price / price - 1 if not math.isnan(forecast_price) and price > 0 else float("nan")

    info = fetch_info(ticker)
    analyst_target = info.get("targetMeanPrice")
    analyst_upside = (analyst_target / price - 1) if analyst_target and price > 0 else None
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")

    # For custom-typed tickers the display name defaults to the symbol — upgrade
    # it to the real company name from Yahoo when we have it.
    if (not name) or name == ticker:
        name = info.get("shortName") or info.get("longName") or ticker

    metrics_dict = dict(
        pct_change_3m=pct_3m,
        forecast_upside=forecast_upside,
        analyst_upside=analyst_upside,
        sharpe=sharpe,
        volatility=vol,
        pe_ratio=pe_ratio,
        rsi_14=rsi,
    )
    score, rec = _best_buy_score(metrics_dict)

    # Financial KPIs from the info dict — already fetched, so no extra HTTP call.
    annual_rev = info.get("totalRevenue")
    rev_yoy = info.get("revenueGrowth")  # TTM YoY (decimal, e.g. 0.12 = +12%)
    net_inc = info.get("netIncomeToCommon") or info.get("netIncome")
    # earningsGrowth is often null for loss-making firms; fall back to the
    # quarterly YoY earnings growth when available.
    earn_yoy = info.get("earningsGrowth")
    if earn_yoy is None:
        earn_yoy = info.get("earningsQuarterlyGrowth")
    qoq_sales = (
        info.get("quarterlyRevenueGrowth")
        or info.get("revenueQuarterlyGrowth")
        or info.get("earningsQuarterlyGrowth")
    )
    market_cap_val = info.get("marketCap")
    cap_cat = _cap_category(market_cap_val)

    return StockMetrics(
        ticker=ticker,
        name=name,
        price=price,
        pct_change_1d=pct_1d,
        pct_change_1m=pct_1m,
        pct_change_3m=pct_3m,
        pct_change_ytd=pct_ytd,
        cagr_1y=_cagr(prices, 1),
        cagr_3y=_cagr(prices, 3),
        cagr_5y=_cagr(prices, 5),
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        beta=beta,
        avg_dollar_volume=avg_dollar_vol,
        rsi_14=rsi,
        forecast_30d=forecast_price,
        forecast_upside=forecast_upside,
        analyst_target=analyst_target,
        analyst_upside=analyst_upside,
        pe_ratio=pe_ratio,
        risk_level=_risk_level(vol, mdd),
        best_buy_score=score,
        recommendation=rec,
        annual_revenue=annual_rev,
        revenue_yoy=rev_yoy,
        net_income=net_inc,
        earnings_yoy=earn_yoy,
        sales_growth_qoq=qoq_sales,
        market_cap=market_cap_val,
        cap_category=cap_cat,
        period_returns=period_returns,
    )


# ---------------------------------------------------------------------------
# Company-detail modal (opens when a stock card's Details button is clicked)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_statements(ticker: str) -> dict:
    """Income statement / balance sheet / cash flow for the detail dialog."""
    try:
        t = yf.Ticker(ticker)
        return {
            "income": t.income_stmt,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
        }
    except Exception:
        return {}


def _money(value) -> str:
    """Compact monetary formatter: $1.23B, $456.7M, etc."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return f"{sign}${v/1e12:.2f}T"
    if v >= 1e9:
        return f"{sign}${v/1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v/1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}${v/1e3:.1f}K"
    return f"{sign}${v:.0f}"


@st.dialog("📊 Company Details", width="large")
def show_company_detail(ticker: str, display_name: str = "") -> None:
    info = fetch_info(ticker)
    display_name = display_name or info.get("longName") or info.get("shortName") or ticker

    st.markdown(
        f"<h3 style='margin:0 0 4px 0;color:#f3f4f6;'>{ticker} · "
        f"<span style='color:#a78bfa;'>{display_name}</span></h3>",
        unsafe_allow_html=True,
    )
    sector = info.get("sector", "—")
    industry = info.get("industry", "—")
    exchange = info.get("fullExchangeName") or info.get("exchange", "—")
    st.caption(f"{sector} · {industry} · {exchange}")

    # ---- Top KPI strip --------------------------------------------------
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    change_pct = info.get("regularMarketChangePercent")
    if change_pct is None and info.get("previousClose"):
        change_pct = (price / info["previousClose"] - 1) * 100 if info["previousClose"] else None

    k = st.columns(4)
    k[0].metric("Price", f"{price:,.2f}", f"{change_pct:+.2f}%" if change_pct else None)
    k[1].metric("Market Cap", _money(info.get("marketCap")))
    k[2].metric("P/E (TTM)", f"{info.get('trailingPE', 0):.1f}" if info.get("trailingPE") else "—")
    lo = info.get("fiftyTwoWeekLow") or 0
    hi = info.get("fiftyTwoWeekHigh") or 0
    k[3].metric("52W Range", f"{lo:,.0f} – {hi:,.0f}")

    tab_chart, tab_fin, tab_ratios, tab_news, tab_about = st.tabs(
        ["📈 Price", "💰 Financials", "📐 Ratios", "📰 News", "ℹ️ About"]
    )

    with tab_chart:
        prices = fetch_history((ticker,), period="2y")
        if not prices.empty and ticker in prices.columns:
            series = prices[ticker].dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode="lines",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
                name=ticker,
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(t=10, l=10, r=10, b=10),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Price history unavailable.")

    with tab_fin:
        st.markdown("**Income Statement** (annual, last 4 years)")
        fin = fetch_financial_statements(ticker)
        income = fin.get("income")
        if income is not None and not income.empty:
            preferred = [
                "Total Revenue", "Cost Of Revenue", "Gross Profit",
                "Operating Income", "Operating Expense",
                "Net Income", "Net Income Common Stockholders", "Diluted EPS", "EBITDA",
            ]
            rows = [r for r in preferred if r in income.index]
            if rows:
                disp = income.loc[rows].copy()
                disp.columns = [c.strftime("%Y") if hasattr(c, "strftime") else str(c) for c in disp.columns]
                # Show in millions, except EPS which stays as-is
                def _scale(row):
                    if "EPS" in row.name:
                        return row.round(2)
                    return (row / 1e6).round(1)
                disp = disp.apply(_scale, axis=1)
                st.dataframe(disp, use_container_width=True)
                st.caption("In millions of reporting currency · EPS as reported")
            else:
                st.caption("No standard line items found.")
        else:
            st.caption("Income statement unavailable for this ticker.")

        st.markdown("**Balance Sheet Highlights**")
        balance = fin.get("balance")
        if balance is not None and not balance.empty:
            preferred_b = ["Total Assets", "Total Liab", "Total Liabilities Net Minority Interest",
                           "Total Equity Gross Minority Interest", "Cash And Cash Equivalents",
                           "Total Debt", "Working Capital"]
            rows_b = [r for r in preferred_b if r in balance.index]
            if rows_b:
                disp_b = balance.loc[rows_b].copy()
                disp_b.columns = [c.strftime("%Y") if hasattr(c, "strftime") else str(c) for c in disp_b.columns]
                disp_b = (disp_b / 1e6).round(1)
                st.dataframe(disp_b, use_container_width=True)
            else:
                st.caption("No standard line items found.")
        else:
            st.caption("Balance sheet unavailable.")

    with tab_ratios:
        rc1 = st.columns(3)
        rc1[0].metric("Gross Margin", f"{(info.get('grossMargins') or 0)*100:.1f}%")
        rc1[1].metric("Operating Margin", f"{(info.get('operatingMargins') or 0)*100:.1f}%")
        rc1[2].metric("Profit Margin", f"{(info.get('profitMargins') or 0)*100:.1f}%")
        rc2 = st.columns(3)
        rc2[0].metric("ROE", f"{(info.get('returnOnEquity') or 0)*100:.1f}%")
        rc2[1].metric("ROA", f"{(info.get('returnOnAssets') or 0)*100:.1f}%")
        rc2[2].metric("Debt/Equity", f"{(info.get('debtToEquity') or 0):.1f}")
        rc3 = st.columns(3)
        rc3[0].metric("Revenue Growth (YoY)", f"{(info.get('revenueGrowth') or 0)*100:+.1f}%")
        rc3[1].metric("Earnings Growth (YoY)", f"{(info.get('earningsGrowth') or 0)*100:+.1f}%")
        rc3[2].metric("Beta", f"{info.get('beta') or 0:.2f}")
        rc4 = st.columns(3)
        rc4[0].metric("P/B", f"{info.get('priceToBook') or 0:.1f}")
        rc4[1].metric("EV/EBITDA", f"{info.get('enterpriseToEbitda') or 0:.1f}")
        rc4[2].metric("Dividend Yield", f"{(info.get('dividendYield') or 0)*100:.2f}%")

        target = info.get("targetMeanPrice")
        if target and price:
            upside = (target / price - 1) * 100
            st.markdown(
                f"<div style='margin-top:14px;padding:12px;border-radius:8px;"
                f"background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);'>"
                f"<b>Analyst consensus target:</b> {target:,.2f} "
                f"<span style='color:{'#10b981' if upside >= 0 else '#ef4444'};'>"
                f"({upside:+.1f}% upside)</span> · "
                f"{info.get('numberOfAnalystOpinions', '?')} analysts"
                f"</div>",
                unsafe_allow_html=True,
            )

    with tab_news:
        items = fetch_news(ticker, limit=10)
        if not items:
            st.caption("No recent news for this ticker.")
        for n in items:
            title_link = f"[{n['title']}]({n['link']})" if n["link"] else f"**{n['title']}**"
            st.markdown(
                f"<div style='padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"{title_link}<br>"
                f"<span style='color:#9ca3af;font-size:0.75rem;'>{n['publisher']} · {n['time']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with tab_about:
        st.markdown(info.get("longBusinessSummary", "_No description available._"))
        meta = st.columns(2)
        meta[0].markdown(f"**Website:** {info.get('website', '—')}")
        meta[0].markdown(f"**Employees:** {info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "**Employees:** —")
        meta[1].markdown(f"**HQ:** {info.get('city', '')}, {info.get('country', '—')}")
        meta[1].markdown(f"**Currency:** {info.get('currency', '—')}")


# ---------------------------------------------------------------------------
# Prediction Markets page (Polymarket-style, real data via Gamma API)
# ---------------------------------------------------------------------------

def _render_event_card(event: dict) -> None:
    image = event.get("image") or event.get("icon")
    title = event.get("title") or "Untitled"
    end_date = event.get("endDate")
    slug = event.get("slug") or ""
    vol24h = float(event.get("volume24hr") or 0)
    vol_total = float(event.get("volume") or 0)
    markets = event.get("markets") or []

    days_to_end = ""
    if end_date:
        try:
            d = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            delta = (d - datetime.now(timezone.utc)).days
            if delta > 365:
                days_to_end = f"Ends in {delta // 30}mo"
            elif delta > 1:
                days_to_end = f"Ends in {delta}d"
            elif delta >= 0:
                days_to_end = "Ends today"
            else:
                days_to_end = "Ended"
        except Exception:
            pass

    poly_url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"
    img_html = (
        f'<img src="{image}" style="width:42px;height:42px;border-radius:6px;object-fit:cover;flex-shrink:0;">'
        if image else ""
    )

    # Build the inner markets HTML first so we can render the whole card as a single block.
    # IMPORTANT: every HTML literal lives at column 0 — Streamlit's markdown turns any
    # line indented ≥4 spaces into a <pre><code> block even with unsafe_allow_html=True.
    markets_sorted = sorted(markets, key=lambda m: -float(m.get("volume24hr") or 0))
    inner_markets_html = ""
    for m in markets_sorted[:3]:
        yes_pct = _yes_price(m)
        if yes_pct is None:
            continue
        pct = yes_pct * 100
        bar_color = "#10b981" if pct >= 50 else "#ef4444"
        q = m.get("question") or ""
        if event.get("title") and q.lower().startswith(event["title"].lower()):
            q = q[len(event["title"]):].strip(" ?:-")
        q = q[:70] or m.get("groupItemTitle") or "Market"
        inner_markets_html += (
            '<div style="margin:8px 0;">'
            '<div style="font-size:0.78rem;color:#cbd5e1;display:flex;justify-content:space-between;gap:8px;">'
            f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{q}</span>'
            f'<span style="font-weight:700;color:{bar_color};flex-shrink:0;">{pct:.0f}%</span>'
            '</div>'
            '<div style="background:rgba(255,255,255,0.06);border-radius:3px;height:6px;overflow:hidden;margin-top:4px;">'
            f'<div style="background:linear-gradient(90deg,{bar_color}88,{bar_color});height:100%;width:{pct:.0f}%;border-radius:3px;"></div>'
            '</div>'
            '</div>'
        )

    extra = max(0, len(markets) - 3)
    extra_txt = f" · +{extra} more markets" if extra > 0 else ""

    card_html = (
        '<div class="gt-card" style="min-height:220px;display:flex;flex-direction:column;">'
        '<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:10px;">'
        f'{img_html}'
        '<div style="flex:1;min-width:0;">'
        f'<div style="font-weight:600;font-size:0.95rem;color:#f3f4f6;line-height:1.3;letter-spacing:-0.01em;">{title}</div>'
        '<div style="color:#9ca3af;font-size:0.7rem;margin-top:6px;">'
        f'{days_to_end} · <span style="color:#a78bfa;">${vol24h/1000:,.0f}k</span> 24h · ${vol_total/1e6:.1f}M total'
        '</div></div></div>'
        f'{inner_markets_html}'
        '<div style="margin-top:auto;padding-top:10px;font-size:0.72rem;color:#9ca3af;">'
        f'<a href="{poly_url}" target="_blank" style="color:#a78bfa;text-decoration:none;font-weight:500;">Open on Polymarket →</a>{extra_txt}'
        '</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# Market configs for the Top Movers screener. USA uses Yahoo's curated predefined
# screeners; other markets use region-filtered EquityQuery screens with a market-cap
# floor to keep out illiquid micro-cap noise.
MOVER_MARKETS: dict[str, dict] = {
    "🇺🇸 USA": {"mode": "predefined"},
    "🇭🇰 Hong Kong": {"mode": "region", "regions": ["hk"], "min_mcap": 1e9},
    "🇮🇳 India": {"mode": "region", "regions": ["in"], "min_mcap": 1e9},
    "🇪🇺 Europe": {"mode": "region", "regions": ["gb", "de", "fr", "it", "es", "nl", "ch", "se"], "min_mcap": 1e9},
    "🌏 Asia": {"mode": "region", "regions": ["jp", "kr", "tw", "sg", "hk"], "min_mcap": 1e9},
}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_screener(key: str, count: int = 50) -> list[dict]:
    """Yahoo predefined screener (day_gainers / day_losers / most_actives)."""
    try:
        res = yf.screen(key, count=count)
        return res.get("quotes", []) if isinstance(res, dict) else []
    except Exception:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def fetch_region_movers(regions: tuple, direction: str, min_mcap: float, count: int = 50) -> list[dict]:
    """Region-filtered movers. direction: 'gainers' | 'losers' | 'active'."""
    try:
        from yfinance import EquityQuery
        base = [
            EquityQuery("is-in", ["region", *regions]),
            EquityQuery("gt", ["dayvolume", 100000]),
            EquityQuery("gt", ["intradaymarketcap", min_mcap]),
        ]
        q = EquityQuery("and", base)
        if direction == "active":
            sort_field, asc = "dayvolume", False
        else:
            sort_field, asc = "percentchange", (direction == "losers")
        res = yf.screen(q, sortField=sort_field, sortAsc=asc, count=count * 2)
        quotes = res.get("quotes", []) if isinstance(res, dict) else []
    except Exception:
        return []
    # Post-filter on the returned marketCap to drop micro-cap stragglers
    # (some listings bypass the intradaymarketcap query filter).
    clean = [q for q in quotes if (q.get("marketCap") or 0) >= min_mcap * 0.5]
    return clean[:count]


def _row_click_symbol(event, df: pd.DataFrame, state_key: str) -> str | None:
    """Return the newly-clicked Symbol from a selectable dataframe, or None.

    Tracks the previous selection per-table so a persisted selection doesn't
    reopen the dialog on every rerun (e.g. after the modal is closed)."""
    try:
        rows = event.selection.rows
    except Exception:
        rows = []
    cur = df.iloc[rows[0]]["Symbol"] if rows and rows[0] < len(df) else None
    prev = st.session_state.get(state_key)
    st.session_state[state_key] = cur
    return cur if (cur and cur != prev) else None


def _clean_analyst_rating(raw) -> str:
    """Yahoo's averageAnalystRating looks like '1.8 - Buy'. Return a tidy label."""
    if not raw or not isinstance(raw, str):
        return "—"
    # Keep the word part, prettied; fall back to the whole string.
    parts = raw.split(" - ")
    label = parts[-1].strip() if len(parts) > 1 else raw.strip()
    num = parts[0].strip() if len(parts) > 1 else ""
    return f"{label} ({num})" if num else label


def _movers_dataframe(quotes: list[dict]) -> pd.DataFrame:
    rows = []
    for q in quotes:
        chg = q.get("regularMarketChangePercent")
        if chg is None:
            continue
        rows.append({
            "Symbol": q.get("symbol", ""),
            "Company": q.get("shortName") or q.get("longName") or q.get("symbol", ""),
            "Analyst Rating": _clean_analyst_rating(q.get("averageAnalystRating")),
            "Price": q.get("regularMarketPrice"),
            "Change %": chg,
            "Change": q.get("regularMarketChange"),
            "Market Cap": q.get("marketCap"),
            "Volume": q.get("regularMarketVolume") or q.get("averageDailyVolume3Month"),
            "52W %": q.get("fiftyTwoWeekChangePercent"),
        })
    return pd.DataFrame(rows)


# Yahoo symbol suffixes per market — used to scope search results across pages.
MARKET_SUFFIXES = {
    "USA": set(),  # US tickers carry no exchange suffix (handled specially)
    "Hong Kong": {".HK"},
    "India": {".NS", ".BO"},
    "Europe": {".L", ".DE", ".PA", ".AS", ".MI", ".MC", ".SW", ".CO",
               ".ST", ".BR", ".VI", ".LS", ".HE", ".OL", ".IR", ".F", ".DE"},
    "Asia (ex-HK/India)": {".T", ".KS", ".KQ", ".TW", ".TWO", ".SI", ".AX", ".HK"},
}


def _symbol_in_markets(symbol: str, markets: list[str]) -> bool:
    """True if the ticker belongs to one of the selected markets (by suffix)."""
    sym = (symbol or "").upper()
    has_dot = "." in sym
    for m in markets:
        if m == "USA":
            if not has_dot:  # US tickers have no exchange suffix
                return True
        else:
            for suf in MARKET_SUFFIXES.get(m, set()):
                if sym.endswith(suf):
                    return True
    return False


# ---------------------------------------------------------------------------
# Stock Valuation page — 5-year DCF + comps + sensitivity (yfinance data).
# Faithful port of the live-stock-valuation model.
# ---------------------------------------------------------------------------

# Default DCF assumptions (same as the source model).
DCF_DEFAULTS = {"tax": 0.18, "wacc": 0.09, "termg": 0.035,
                "da": 0.07, "capex": 0.12, "nwc": 0.02, "taper": 0.02}


def _dcf_project(base: dict, a: dict) -> list[dict]:
    """5-year revenue → FCF projection. Growth tapers toward terminal growth."""
    rows, prev = [], base["rev"]
    for t in range(1, 6):
        g = max(a["termg"], base["g1"] - (t - 1) * a["taper"])
        rev = prev * (1 + g)
        ebit = rev * base["margin"]
        nopat = ebit * (1 - a["tax"])
        da, capex = rev * a["da"], rev * a["capex"]
        nwc = (rev - prev) * a["nwc"]
        rows.append({"t": t, "g": g, "rev": rev, "ebit": ebit,
                     "da": da, "capex": capex, "nwc": nwc,
                     "fcf": nopat + da - capex - nwc})
        prev = rev
    return rows


def _dcf(base: dict, a: dict) -> dict:
    pr = _dcf_project(base, a)
    sumpv = sum(r["fcf"] / (1 + a["wacc"]) ** r["t"] for r in pr)
    last = pr[-1]["fcf"]
    if a["wacc"] <= a["termg"]:
        return {"pr": pr, "sumpv": sumpv, "pvtv": float("nan"), "ev": float("nan"),
                "netDebt": base["debt"] - base["cash"], "eq": float("nan"),
                "ps": float("nan"), "upside": float("nan"), "tvShare": float("nan")}
    tv = last * (1 + a["termg"]) / (a["wacc"] - a["termg"])
    pvtv = tv / (1 + a["wacc"]) ** 5
    ev = sumpv + pvtv
    nd = base["debt"] - base["cash"]
    eq = ev - nd
    ps = eq / base["shares"] if base["shares"] else float("nan")
    return {"pr": pr, "sumpv": sumpv, "pvtv": pvtv, "ev": ev, "netDebt": nd,
            "eq": eq, "ps": ps, "upside": (ps / base["price"] - 1) if base["price"] else float("nan"),
            "tvShare": pvtv / ev if ev else float("nan")}


def _dcf_price_gm(base: dict, a: dict, g: float, m: float) -> float:
    """DCF per-share holding growth g and margin m constant (sensitivity)."""
    s, prev, last = 0.0, base["rev"], 0.0
    for t in range(1, 6):
        rev = prev * (1 + g)
        fcf = rev * m * (1 - a["tax"]) + rev * a["da"] - rev * a["capex"] - (rev - prev) * a["nwc"]
        s += fcf / (1 + a["wacc"]) ** t
        last, prev = fcf, rev
    if a["wacc"] <= a["termg"] or not base["shares"]:
        return float("nan")
    tv = last * (1 + a["termg"]) / (a["wacc"] - a["termg"])
    return (s + tv / (1 + a["wacc"]) ** 5 - (base["debt"] - base["cash"])) / base["shares"]


def _dcf_price_w(base: dict, a: dict, w: float, tg: float) -> float:
    """DCF per-share varying WACC w and terminal growth tg (sensitivity)."""
    pr = _dcf_project(base, a)
    if w <= tg or not base["shares"]:
        return float("nan")
    s = sum(r["fcf"] / (1 + w) ** r["t"] for r in pr)
    last = pr[-1]["fcf"]
    return (s + (last * (1 + tg) / (w - tg)) / (1 + w) ** 5 - (base["debt"] - base["cash"])) / base["shares"]


def _capm_wacc(base: dict, rf: float, erp: float, tax: float) -> float:
    """Estimate WACC from CAPM: blend cost of equity (rf + β·ERP) with after-tax cost of debt."""
    beta = base.get("beta") or 1.0
    ke = rf + beta * erp
    e = base["price"] * base["shares"]          # market value of equity
    d = max(0.0, base["debt"])                   # book value of debt (proxy)
    kd_after_tax = (rf + 0.015) * (1 - tax)      # cost of debt ≈ rf + credit spread
    tot = e + d
    if tot <= 0:
        return max(0.05, min(0.20, ke))
    wacc = (e / tot) * ke + (d / tot) * kd_after_tax
    return max(0.05, min(0.20, wacc))


def _implied_growth(base: dict, a: dict, lo: float = -0.5, hi: float = 1.0) -> tuple[float, str]:
    """Reverse DCF: solve for the constant 5-yr growth the CURRENT PRICE implies.

    DCF price is monotonically increasing in growth, so bisection converges.
    Returns (growth, edge) where edge is 'below' / 'above' / 'exact'.
    """
    def f(g):  # DCF price at growth g (current margin) minus the market price
        return _dcf_price_gm(base, a, g, base["margin"]) - base["price"]
    flo, fhi = f(lo), f(hi)
    if flo != flo or fhi != fhi:
        return float("nan"), "exact"
    if flo > 0:   # even at -50% growth the model exceeds price → market implies < -50%
        return lo, "below"
    if fhi < 0:   # even at +100% growth the model is below price → market implies > +100%
        return hi, "above"
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, "exact"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_valuation_base(ticker: str) -> dict | None:
    """Pull the fundamentals the DCF needs from Yahoo. Returns None if unusable."""
    info = fetch_info(ticker)
    if not info:
        return None
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    rev = info.get("totalRevenue")
    if not (price and shares and rev):
        return None
    margin = info.get("operatingMargins")
    if margin is None:
        margin = info.get("profitMargins") or 0.10
    g1 = info.get("revenueGrowth")
    if g1 is None:
        g1 = info.get("earningsGrowth") or 0.08
    return {
        "name": info.get("shortName") or info.get("longName") or ticker,
        "price": float(price),
        "shares": float(shares),
        "rev": float(rev),
        "margin": float(margin),
        "ni": float(info.get("netIncomeToCommon") or info.get("netIncome") or rev * margin * 0.8),
        "debt": float(info.get("totalDebt") or 0),
        "cash": float(info.get("totalCash") or 0),
        "g1": float(g1),
        "currency": info.get("currency", "USD"),
        "pe": info.get("trailingPE"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "sector": info.get("sector", "—"),
        "beta": info.get("beta"),
    }


def _sensitivity_heatmap(base, a, mode: str):
    """Plotly heatmap of DCF price across a 2-D assumption grid."""
    if mode == "gm":
        rows = [base["margin"] + d for d in (-0.04, -0.02, 0, 0.02, 0.04)]
        cols = [max(0.0, base["g1"] + d) for d in (-0.06, -0.03, 0, 0.03, 0.06)]
        z = [[_dcf_price_gm(base, a, g, m) for g in cols] for m in rows]
        x_title, y_title = "Revenue growth →", "Op margin ↓"
        x_labels = [f"{g*100:.0f}%" for g in cols]
        y_labels = [f"{m*100:.0f}%" for m in rows]
        title = "Sensitivity · Growth × Margin"
    else:
        rows = [a["termg"] + d for d in (-0.015, -0.0075, 0, 0.0075, 0.015)]
        cols = [a["wacc"] + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
        z = [[_dcf_price_w(base, a, w, tg) for w in cols] for tg in rows]
        x_title, y_title = "WACC →", "Terminal growth ↓"
        x_labels = [f"{w*100:.1f}%" for w in cols]
        y_labels = [f"{tg*100:.1f}%" for tg in rows]
        title = "Sensitivity · WACC × Terminal growth"

    text = [[("—" if (v != v) else f"{v:,.0f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, text=text, texttemplate="%{text}",
        textfont={"size": 12, "color": "#0a0a0f"},
        colorscale=[[0, "#ef4444"], [0.5, "#fcd34d"], [1, "#10b981"]],
        showscale=False, hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>Price %{{z:,.0f}}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=320, margin=dict(t=40, l=10, r=10, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        xaxis=dict(title=x_title, side="top"), yaxis=dict(title=y_title, autorange="reversed"),
    )
    return fig


def render_valuation() -> None:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        '<h2 style="margin:0;">💰 Stock Valuation</h2>'
        '<span style="color:#9ca3af;font-size:0.8rem;">5-year DCF · comps · sensitivity · Yahoo data</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Future cash flows → today's dollars → minus net debt → per share. "
               "Adjust the assumptions to stress-test the fair value. Educational, not investment advice.")

    # --- How to read this page (educational guide) -----------------------
    with st.expander("📖 How to read this page — a guide to valuing a stock", expanded=False):
        st.markdown(
            """
**The one question valuation answers:** *is this stock cheap, fair, or expensive relative to the cash it
will generate over time?* No single number is gospel — the goal is to **triangulate several lenses** and
check whether the expectations baked into today's price are realistic.

---

#### 🎯 The headline numbers (top of page)
| What you see | Plain meaning | How to use it |
|---|---|---|
| **DCF Fair Value** | What a share is worth based on projected future cash flows | Compare to the current price |
| **Upside / (Downside)** | Gap between fair value and price | Roughly: > **+25%** looks cheap, < **−25%** looks expensive — *if the assumptions hold* |
| **DCF Signal** | Mechanical Buy → Sell read from the upside | A starting point, **not** a verdict |
| **Price-vs-Fair-Value gauge** | Where the price sits in the under/overvalued zones | Needle in the green zone = trading below fair value |

#### 🔄 Reverse DCF — the most important sanity check
Instead of *you* guessing growth, it shows the **growth rate the current price already assumes**. Then ask:
*can the company realistically deliver that?*
- **Implied growth ≫ recent growth** → market is optimistic; a lot has to go right (risky).
- **Implied growth ≪ recent growth** → expectations are low; easier to beat (often safer).

#### 📐 Discount rate (WACC) & beta
WACC is the annual return investors demand for the risk taken. **Higher risk (higher beta) → higher WACC → lower fair value.**
β > 1 = more volatile than the market; β < 1 = defensive. Small WACC changes swing the fair value a lot —
that's what the sensitivity table shows.

#### 💵 Cash flow projection & the value bridge
**Free Cash Flow (FCF)** is the real fuel of value — the cash left after running *and* growing the business
(`FCF = NOPAT + D&A − Capex − ΔNWC`). A company can report "profit" yet burn cash; FCF cuts through that.
The **Value Bridge** shows how each year's discounted cash + the terminal value stack up to enterprise value, then to equity.

#### ⚠️ Terminal value %
How much of the valuation comes from year-6-and-beyond assumptions. **If it's > ~75%, the DCF rests heavily
on unprovable long-run guesses** — treat the fair value with extra caution.

#### 🌡️ Sensitivity tables
Fair value is a **range, not a point**. These show how it moves as growth/margin and WACC/terminal-growth change.
A narrow range = robust thesis; a wild range = fragile.

---

#### 📊 Key KPIs to check on *any* stock (beyond this page)
| KPI | What good looks like | Red flag |
|---|---|---|
| **Revenue growth** | Steady or accelerating | Decelerating / negative |
| **Operating & net margin** | Stable or expanding | Consistently shrinking |
| **ROE / ROIC** | > 15% / above its WACC | Low, or below cost of capital |
| **Free cash flow** | Positive & growing | Negative or erratic |
| **Net debt / EBITDA** | < 3× | > 4–5× (leverage risk) |
| **Interest coverage** | > 4× | < 2× (can't service debt) |
| **P/E & EV/EBITDA** | Reasonable vs peers *and* growth | Extreme vs history/peers |
| **Dividend payout** | Sustainable (< ~70% of earnings) | > 100% (paying more than it earns) |

#### ✅ A simple workflow
1. Check **fair value & upside** (DCF).
2. Run the **reverse DCF** — are the implied expectations realistic?
3. Confirm **FCF is positive and growing**.
4. Make sure **terminal value isn't dominating** the valuation.
5. **Compare to peers** (P/E, EV/EBITDA).
6. Check the **balance sheet** (debt) and **quality** (margins, ROE).
7. Read the **sensitivity range**, not just the single number.

> **Remember:** garbage in, garbage out — a DCF is only as good as its assumptions. This is an educational
> tool, **not investment advice**. Always do your own research and consider a margin of safety.
            """
        )

    # --- Market scope + ticker search -----------------------------------
    VAL_MARKETS = {
        "🌐 All": None,
        "🇺🇸 USA": ["USA"],
        "🇭🇰 Hong Kong": ["Hong Kong"],
        "🇮🇳 India": ["India"],
        "🇪🇺 Europe": ["Europe"],
        "🌏 Asia": ["Asia (ex-HK/India)"],
    }
    val_market = st.radio(
        "Market", options=list(VAL_MARKETS.keys()), horizontal=True,
        key="val_market", label_visibility="collapsed",
    )
    market_keys = VAL_MARKETS[val_market]
    placeholders = {
        "🇺🇸 USA": "e.g. Apple, AAPL, Microsoft, NVDA…",
        "🇭🇰 Hong Kong": "e.g. Tencent, 0700.HK, Alibaba, HSBC…",
        "🇮🇳 India": "e.g. Reliance, TCS, INFY, HDFC Bank…",
        "🇪🇺 Europe": "e.g. ASML, SAP, LVMH, Nestle…",
        "🌏 Asia": "e.g. Toyota, 7203.T, Samsung, TSMC…",
        "🌐 All": "e.g. Apple, Tencent, Reliance, Toyota…",
    }

    c1, c2 = st.columns([3, 2])
    with c1:
        vq = st.text_input("🔎 Company name or ticker", key="val_search",
                           placeholder=placeholders.get(val_market, ""))
    chosen = None
    if vq and len(vq.strip()) >= 2:
        matches = fetch_symbol_search(vq, limit=25)
        scoped_note = ""
        if market_keys:
            filt = [m for m in matches if _symbol_in_markets(m["symbol"], market_keys)]
            if filt:
                matches = filt
            elif matches:
                scoped_note = f" · no {val_market} match — showing all markets"
        matches = matches[:10]
        if matches:
            opts = {f"{m['symbol']} — {m['name']}"
                    + (f" · {m['exchange']}" if m.get("exchange") else ""): m["symbol"]
                    for m in matches}
            with c2:
                pick = st.selectbox(f"Matches{scoped_note}", list(opts.keys()), key="val_pick")
            chosen = opts.get(pick)
        else:
            chosen = vq.strip().upper()
    if not chosen:
        st.info(f"Pick a **{val_market}** market above, then search a company or ticker to value it.")
        return

    base = fetch_valuation_base(chosen)
    if not base:
        st.error(f"Couldn't load enough fundamentals for **{chosen}** to run a DCF "
                 "(needs price, shares outstanding, and revenue). Try a larger, well-covered company.")
        return

    cur = base["currency"]

    # --- Assumptions (editable) -----------------------------------------
    with st.expander("⚙️ Assumptions — tune the model", expanded=False):
        st.caption("WACC ≈ the yearly return investors demand (~7–10%). "
                   "Terminal growth = assumed growth forever after year 5.")
        a = dict(DCF_DEFAULTS)
        ac1, ac2, ac3 = st.columns(3)
        base["g1"] = ac1.slider("Year-1 revenue growth", -0.20, 0.60, float(round(base["g1"], 3)), 0.01)
        base["margin"] = ac1.slider("Operating margin", -0.10, 0.70, float(round(base["margin"], 3)), 0.01)
        a["taper"] = ac1.slider("Growth taper / yr", 0.0, 0.10, a["taper"], 0.005)
        a["tax"] = ac2.slider("Tax rate", 0.0, 0.40, a["tax"], 0.01)
        rf = ac2.slider("Risk-free rate", 0.0, 0.07, 0.04, 0.005, help="~10-yr govt bond yield")
        erp = ac2.slider("Equity risk premium", 0.03, 0.08, 0.05, 0.005)
        a["termg"] = ac3.slider("Terminal growth", 0.0, 0.05, a["termg"], 0.005)
        a["da"] = ac3.slider("D&A % of revenue", 0.0, 0.20, a["da"], 0.01)
        a["capex"] = ac3.slider("Capex % of revenue", 0.0, 0.30, a["capex"], 0.01)
        a["nwc"] = ac3.slider("ΔNWC % of revenue change", 0.0, 0.15, a["nwc"], 0.01)

        # --- WACC: CAPM-estimated by default, with manual override ----------
        capm = _capm_wacc(base, rf, erp, a["tax"])
        beta_disp = base.get("beta")
        beta_txt = f"β={beta_disp:.2f}" if beta_disp else "β≈1.0 (default)"
        use_capm = st.checkbox(
            f"📐 Use CAPM-estimated WACC: **{capm*100:.1f}%** ({beta_txt})",
            value=True,
            help="Cost of equity = risk-free + β × equity-risk-premium, blended with after-tax cost of debt.",
        )
        if use_capm:
            a["wacc"] = capm
        else:
            a["wacc"] = st.slider("WACC (manual discount rate)", 0.05, 0.15, round(capm, 3), 0.005)

    D = _dcf(base, a)

    # --- Verdict cards ---------------------------------------------------
    upside = D["upside"]
    sig = ("Strong Buy" if upside >= 0.25 else "Buy" if upside >= 0.10 else
           "Hold" if upside >= -0.10 else "Reduce" if upside >= -0.25 else "Sell") if upside == upside else "—"
    sig_color = _rec_color(sig)
    up_color = _color(upside)

    v = st.columns(4)
    v[0].metric(f"Current Price · {chosen}", f"{base['price']:,.2f}", help=base["name"])
    v[1].metric("DCF Fair Value", f"{D['ps']:,.2f}" if D["ps"] == D["ps"] else "—")
    v[2].metric("Upside / (Downside)", _fmt_pct(upside))
    v[3].markdown(
        f'<div class="gt-card" style="text-align:center;border:1px solid {sig_color}66;'
        f'background:linear-gradient(135deg,{sig_color}22,{sig_color}0a);">'
        f'<div style="font-size:0.7rem;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.05em;">DCF Signal</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{sig_color};margin-top:6px;text-shadow:0 0 12px {sig_color}55;">{sig}</div>'
        f'<div style="color:#94a3b8;font-size:0.66rem;margin-top:4px;">{base["sector"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Price vs Fair-Value gauge --------------------------------------
    if D["ps"] == D["ps"] and D["ps"] > 0:
        fair = D["ps"]
        axis_max = max(base["price"], fair) * 1.4
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=base["price"],
            number={"prefix": "", "font": {"size": 30, "color": "#f3f4f6"}},
            delta={"reference": fair, "increasing": {"color": "#ef4444"},
                   "decreasing": {"color": "#10b981"},
                   "suffix": " vs fair", "font": {"size": 14}},
            title={"text": "Current Price vs DCF Fair Value", "font": {"size": 14, "color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, axis_max], "tickcolor": "#64748b",
                         "tickfont": {"color": "#94a3b8", "size": 10}},
                "bar": {"color": "#a78bfa", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, fair], "color": "rgba(16,185,129,0.22)"},
                    {"range": [fair, axis_max], "color": "rgba(239,68,68,0.18)"},
                ],
                "threshold": {"line": {"color": "#f3f4f6", "width": 3}, "thickness": 0.85, "value": fair},
            },
        ))
        gauge.update_layout(
            height=240, margin=dict(t=50, l=30, r=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        )
        gc1, gc2 = st.columns([2, 1])
        gc1.plotly_chart(gauge, use_container_width=True)
        zone = "undervalued 🟢" if base["price"] < fair else "overvalued 🔴"
        gc2.markdown(
            f'<div class="gt-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">'
            f'<div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Verdict</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:#f3f4f6;margin:6px 0;">DCF says <span style="color:{up_color};">{zone}</span></div>'
            f'<div style="font-size:0.8rem;color:#cbd5e1;">Fair value <b>{fair:,.2f}</b> vs price <b>{base["price"]:,.2f}</b></div>'
            f'<div style="font-size:0.8rem;color:{up_color};font-weight:600;margin-top:4px;">{_fmt_pct(upside)} {"upside" if upside>=0 else "downside"}</div>'
            f'<div style="font-size:0.66rem;color:#64748b;margin-top:8px;">White line = fair value · green zone = undervalued · red zone = overvalued</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Reverse DCF — what growth the market is pricing in ---------------
    st.markdown("##### 🔄 Reverse DCF — what the market is pricing in")
    st.caption("Holds today's price as fact and solves for the constant 5-year revenue growth "
               "that justifies it. Compare it to the company's actual recent growth to judge if "
               "the market's expectation is realistic.")
    implied_g, edge = _implied_growth(base, a)
    assumed_g = base["g1"]
    if edge == "above":
        implied_str, implied_note = "≥ +100%/yr", "off the chart — extreme optimism priced in"
    elif edge == "below":
        implied_str, implied_note = "≤ −50%/yr", "deeply pessimistic expectations priced in"
    elif implied_g != implied_g:
        implied_str, implied_note = "—", "couldn't solve (check WACC vs terminal growth)"
    else:
        implied_str, implied_note = f"{implied_g*100:+.1f}%/yr", "implied by current price"

    gap = (implied_g - assumed_g) if (implied_g == implied_g and edge == "exact") else None
    if gap is None:
        verdict_txt, verdict_color = "—", "#9ca3af"
    elif gap > 0.02:
        verdict_txt = "Market expects MORE growth than recent fundamentals — priced for optimism"
        verdict_color = "#f59e0b"
    elif gap < -0.02:
        verdict_txt = "Market expects LESS growth than recent fundamentals — priced conservatively"
        verdict_color = "#10b981"
    else:
        verdict_txt = "Market expectations roughly match recent growth — fairly priced"
        verdict_color = "#a78bfa"

    rc = st.columns(3)
    rc[0].metric("Market-implied 5-yr growth", implied_str, help=implied_note)
    rc[1].metric("Recent revenue growth (Yr-1)", f"{assumed_g*100:+.1f}%")
    rc[2].metric("Expectation gap", f"{gap*100:+.1f} pts" if gap is not None else "—",
                 help="Implied growth minus recent growth. Large positive = priced for optimism.")
    st.markdown(
        f'<div class="gt-card" style="border-left:3px solid {verdict_color};">'
        f'<span style="color:{verdict_color};font-weight:600;">⚖️ {verdict_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- DCF waterfall + projection -------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("##### Discounted Cash Flow")
        waterfall = pd.DataFrame({
            "Item": ["Sum PV of 5-yr cash", "+ PV of terminal value", "= Enterprise value",
                     "− Net debt", "= Equity value", "÷ Shares", "Fair value / share",
                     "Current price", "Upside", "% from terminal value"],
            "Value": [
                _money(D["sumpv"]), _money(D["pvtv"]), _money(D["ev"]),
                _money(-D["netDebt"]), _money(D["eq"]), f"{base['shares']/1e6:,.0f}M",
                f"{D['ps']:,.2f}" if D["ps"] == D["ps"] else "—",
                f"{base['price']:,.2f}", _fmt_pct(upside),
                _fmt_pct(D["tvShare"]) if D["tvShare"] == D["tvShare"] else "—",
            ],
        })
        st.dataframe(waterfall, use_container_width=True, hide_index=True)

    with right:
        st.markdown("##### Free Cash Flow vs Present Value")
        pr = D["pr"]
        years = [f"Y{r['t']}" for r in pr]
        fcf_vals = [r["fcf"] for r in pr]
        pv_vals = [r["fcf"] / (1 + a["wacc"]) ** r["t"] for r in pr]
        fcf_fig = go.Figure()
        fcf_fig.add_trace(go.Bar(x=years, y=fcf_vals, name="Free cash flow",
                                 marker_color="#6366f1"))
        fcf_fig.add_trace(go.Bar(x=years, y=pv_vals, name="PV of FCF",
                                 marker_color="#a78bfa"))
        fcf_fig.update_layout(
            barmode="group", height=300, margin=dict(t=10, l=10, r=10, b=10),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fcf_fig, use_container_width=True)
        st.caption("Discounting shrinks far-out cash flows — Y5's PV is worth less than its FCF.")

    # --- Detailed cash flow projection (full FCF bridge) -----------------
    st.markdown("##### 💵 Cash Flow Projection (5-year FCF build)")
    pr = D["pr"]
    cf = pd.DataFrame({"Line item": [
        "Revenue", "Revenue growth", "Operating income (EBIT)", "NOPAT (EBIT after tax)",
        "(+) D&A", "(−) Capex", "(−) Δ Net working capital",
        "Free cash flow", "Discount factor", "PV of free cash flow",
    ]})
    for r in pr:
        t = r["t"]
        df_ = 1 / (1 + a["wacc"]) ** t
        nopat = r["ebit"] * (1 - a["tax"])
        cf[f"Year {t}"] = [
            _money(r["rev"]),
            f"{r['g']*100:+.1f}%",
            _money(r["ebit"]),
            _money(nopat),
            _money(r["da"]),
            f"({_money(r['capex'])})",
            f"({_money(r['nwc'])})",
            _money(r["fcf"]),
            f"{df_:.3f}",
            _money(r["fcf"] * df_),
        ]
    st.dataframe(cf, use_container_width=True, hide_index=True)
    tv_note = (f"**Terminal value** (year-5 FCF growing at {a['termg']*100:.1f}% forever, discounted): "
               f"{_money(D['pvtv'])} present value — that's {_fmt_pct(D['tvShare'])} of enterprise value."
               ) if D["pvtv"] == D["pvtv"] else ""
    if tv_note:
        st.caption(tv_note)
    st.caption("FCF = NOPAT + D&A − Capex − ΔNWC. Each year's cash is discounted to today at the WACC; "
               "the sum of these PVs plus the terminal value's PV is the enterprise value.")

    # --- Cumulative PV waterfall: cash flows → enterprise → equity --------
    if D["pvtv"] == D["pvtv"] and D["eq"] == D["eq"]:
        st.markdown("##### 🧱 Value Bridge — how cash flows stack into equity value")
        pv_years = [r["fcf"] / (1 + a["wacc"]) ** r["t"] for r in pr]
        wf_x = [f"PV Y{r['t']}" for r in pr] + ["PV Terminal", "Enterprise value", "Net debt", "Equity value"]
        wf_measure = ["relative"] * 5 + ["relative", "total", "relative", "total"]
        wf_y = pv_years + [D["pvtv"], 0, -D["netDebt"], 0]
        wf_text = ([_money(v) for v in pv_years] +
                   [_money(D["pvtv"]), _money(D["ev"]), _money(-D["netDebt"]), _money(D["eq"])])
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=wf_measure, x=wf_x, y=wf_y,
            text=wf_text, textposition="outside", textfont=dict(size=11, color="#cbd5e1"),
            connector={"line": {"color": "rgba(255,255,255,0.18)"}},
            increasing={"marker": {"color": "#6366f1"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#a78bfa"}},
        ))
        wf.update_layout(
            height=380, margin=dict(t=30, l=10, r=10, b=10),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Value"),
            xaxis=dict(tickangle=-30),
            showlegend=False,
        )
        st.plotly_chart(wf, use_container_width=True)
        ps_txt = f"{D['ps']:,.2f}" if D["ps"] == D["ps"] else "—"
        st.caption(
            f"5 years of discounted cash + terminal value = **{_money(D['ev'])}** enterprise value. "
            f"Subtract net debt of **{_money(D['netDebt'])}** → **{_money(D['eq'])}** equity value, "
            f"÷ {base['shares']/1e6:,.0f}M shares = **{ps_txt}** per share."
        )

    # --- Sensitivity heatmaps -------------------------------------------
    st.markdown("##### Sensitivity")
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(_sensitivity_heatmap(base, a, "gm"), use_container_width=True)
    with s2:
        st.plotly_chart(_sensitivity_heatmap(base, a, "wacc"), use_container_width=True)

    # --- Comps (optional peer multiples) --------------------------------
    st.markdown("##### Peer Comparison (optional)")
    peers_raw = st.text_input(
        "Peer tickers (comma-separated) — implied price from median P/E & EV/EBITDA",
        key="val_peers", placeholder="e.g. MSFT, GOOGL, META",
    )
    peer_syms = [p.strip().upper() for p in peers_raw.split(",") if p.strip()]
    if peer_syms:
        all_syms = [chosen] + [p for p in peer_syms if p != chosen]
        rows, pes, evs = [], [], []
        for sym in all_syms:
            b = fetch_valuation_base(sym)
            if not b:
                continue
            eps = b["ni"] / b["shares"] if b["shares"] else float("nan")
            ebitda = b["rev"] * (b["margin"] + DCF_DEFAULTS["da"])
            mc = b["price"] * b["shares"]
            ev = mc + b["debt"] - b["cash"]
            pe = (mc / b["ni"]) if b["ni"] else float("nan")
            evb = (ev / ebitda) if ebitda else float("nan")
            rows.append({"Ticker": sym, "Name": b["name"], "Price": b["price"],
                         "P/E": pe, "EV/EBITDA": evb, "Yr-1 growth": b["g1"] * 100})
            if pe == pe and pe > 0:
                pes.append(pe)
            if evb == evb and evb > 0:
                evs.append(evb)
        if rows:
            import statistics
            med_pe = statistics.median(pes) if pes else float("nan")
            med_ev = statistics.median(evs) if evs else float("nan")
            eps0 = base["ni"] / base["shares"] if base["shares"] else float("nan")
            ebitda0 = base["rev"] * (base["margin"] + DCF_DEFAULTS["da"])
            pe_price = med_pe * eps0 if (med_pe == med_pe and eps0 == eps0) else float("nan")
            ev_price = ((med_ev * ebitda0) - (base["debt"] - base["cash"])) / base["shares"] \
                if (med_ev == med_ev and base["shares"]) else float("nan")
            cc = st.columns(2)
            cc[0].metric("Implied price · median P/E", f"{pe_price:,.2f}" if pe_price == pe_price else "—",
                         _fmt_pct(pe_price / base["price"] - 1) if pe_price == pe_price else None)
            cc[1].metric("Implied price · median EV/EBITDA", f"{ev_price:,.2f}" if ev_price == ev_price else "—",
                         _fmt_pct(ev_price / base["price"] - 1) if ev_price == ev_price else None)
            comps_df = pd.DataFrame(rows)
            st.dataframe(
                comps_df, use_container_width=True, hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="%.2f"),
                    "P/E": st.column_config.NumberColumn(format="%.1f"),
                    "EV/EBITDA": st.column_config.NumberColumn(format="%.1f"),
                    "Yr-1 growth": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )

    st.divider()
    st.caption(
        "DCF assumes 5 years of explicit cash flows plus a terminal value. Fair value is highly "
        "sensitive to WACC and terminal growth — use the sensitivity tables to see the range. "
        "Fundamentals from Yahoo Finance. Educational tool, not investment advice."
    )


def render_top_movers() -> None:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        '<h2 style="margin:0;">🚀 Top Movers</h2>'
        '<span style="color:#9ca3af;font-size:0.8rem;">Live market screener · Yahoo Finance · 2-min cache</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📖 How to read this page", expanded=False):
        st.markdown(
            """
**Top Movers** shows the biggest daily winners and losers across a chosen market — a quick pulse of where
money is flowing *today*.

| Column | What it means |
|---|---|
| **Change %** | Today's price move — the ranking driver |
| **Analyst Rating** | Wall-Street consensus (lower = more bullish): **1–1.5 Strong Buy · 1.5–2.5 Buy · 2.5–3.5 Hold · 3.5+ Sell**) |
| **Market Cap** | Company size — bigger = generally more stable |
| **Volume** | Shares traded today — high volume confirms a move is meaningful |
| **52W %** | Price change over the past year — context vs the daily pop |

**How to use it**
- **Big % move + high volume + improving fundamentals** = a move worth investigating; a big move on thin volume often fades.
- A spike on a stock with a **Sell/Hold** consensus may be a short-term pop, not a trend.
- Switch the **market selector** (USA / Hong Kong / India / Europe / Asia) to scan each region.
- **Click any row** to open full analysis (chart, financials, ratios, news).

> Daily movers are momentum, not value — pair this with the **Valuation** page before acting. Educational, not advice.
            """
        )

    market = st.radio(
        "Market",
        options=list(MOVER_MARKETS.keys()),
        horizontal=True,
        key="mover_market",
        label_visibility="collapsed",
    )
    cfg = MOVER_MARKETS[market]

    with st.spinner(f"Loading {market} movers…"):
        if cfg["mode"] == "predefined":
            gainers = _movers_dataframe(fetch_screener("day_gainers", 50))
            losers = _movers_dataframe(fetch_screener("day_losers", 50))
            actives = _movers_dataframe(fetch_screener("most_actives", 50))
        else:
            regions = tuple(cfg["regions"])
            mcap = cfg["min_mcap"]
            gainers = _movers_dataframe(fetch_region_movers(regions, "gainers", mcap, 50))
            losers = _movers_dataframe(fetch_region_movers(regions, "losers", mcap, 50))
            actives = _movers_dataframe(fetch_region_movers(regions, "active", mcap, 50))

    if gainers.empty and losers.empty:
        st.error(f"Couldn't load {market} movers right now — Yahoo may be rate-limiting. Try again shortly.")
        return

    # KPI strip — company name as the headline, symbol + rating as context.
    k = st.columns(3)
    if not gainers.empty:
        g0 = gainers.iloc[0]
        k[0].metric(f"Biggest Gainer · {g0['Symbol']}", g0["Company"], f"{g0['Change %']:+.1f}%")
        k[0].caption(f"📊 Analyst: **{g0['Analyst Rating']}**")
    if not losers.empty:
        l0 = losers.iloc[0]
        k[1].metric(f"Biggest Loser · {l0['Symbol']}", l0["Company"], f"{l0['Change %']:+.1f}%")
        k[1].caption(f"📊 Analyst: **{l0['Analyst Rating']}**")
    if not actives.empty:
        a0 = actives.iloc[0]
        k[2].metric(f"Most Active · {a0['Symbol']}", a0["Company"], f"{a0['Change %']:+.1f}%")
        k[2].caption(f"📊 Analyst: **{a0['Analyst Rating']}**")

    _mover_col_config = {
        "Company": st.column_config.TextColumn(width="medium"),
        "Analyst Rating": st.column_config.TextColumn("Analyst Rating", help="Yahoo consensus analyst recommendation"),
        "Price": st.column_config.NumberColumn(format="%.2f"),
        "Change %": st.column_config.NumberColumn(format="%+.2f"),
        "Change": st.column_config.NumberColumn(format="%+.2f"),
        "Market Cap": st.column_config.NumberColumn(format="%.0f"),
        "Volume": st.column_config.NumberColumn(format="%.0f"),
        "52W %": st.column_config.NumberColumn(format="%+.1f"),
    }

    st.caption("👆 **Click any row** to open that company's full analysis (chart, financials, ratios, news).")

    g_col, l_col = st.columns(2)
    with g_col:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            '<span class="gt-pill" style="background:#10b98122;color:#10b981;border:1px solid #10b98155;">▲ TOP 50 GAINERS</span>'
            f'<span style="color:#9ca3af;font-size:0.8rem;">{len(gainers)} stocks</span></div>',
            unsafe_allow_html=True,
        )
        g_ev = st.dataframe(
            gainers, use_container_width=True, hide_index=True, height=560,
            column_config=_mover_col_config, on_select="rerun",
            selection_mode="single-row", key="tbl_gainers",
        )
    with l_col:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            '<span class="gt-pill" style="background:#ef444422;color:#ef4444;border:1px solid #ef444455;">▼ TOP 50 LOSERS</span>'
            f'<span style="color:#9ca3af;font-size:0.8rem;">{len(losers)} stocks</span></div>',
            unsafe_allow_html=True,
        )
        l_ev = st.dataframe(
            losers, use_container_width=True, hide_index=True, height=560,
            column_config=_mover_col_config, on_select="rerun",
            selection_mode="single-row", key="tbl_losers",
        )

    with st.expander(f"🔥 Most Active (by volume) — {len(actives)} stocks", expanded=False):
        if not actives.empty:
            a_ev = st.dataframe(
                actives, use_container_width=True, hide_index=True,
                column_config=_mover_col_config, on_select="rerun",
                selection_mode="single-row", key="tbl_actives",
            )
        else:
            a_ev = None
            st.caption("No data.")

    # Open the detail dialog for whichever table's row was just clicked.
    _sym_g = _row_click_symbol(g_ev, gainers, "_sel_gainers")
    _sym_l = _row_click_symbol(l_ev, losers, "_sel_losers")
    _sym_a = _row_click_symbol(a_ev, actives, "_sel_actives") if a_ev is not None else None
    _clicked = _sym_g or _sym_l or _sym_a
    if _clicked:
        show_company_detail(_clicked)

    st.divider()
    st.caption(
        "USA uses Yahoo's curated screener; other markets are region-filtered with a "
        "~$1B market-cap floor to keep out illiquid micro-caps. "
        "Click a column header to sort, or click any row to open full analysis."
    )


def render_prediction_markets() -> None:
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
          <h2 style="margin:0;">🎯 Prediction Markets</h2>
          <span style="color:#9ca3af;font-size:0.8rem;">
            Live odds from <a href="https://polymarket.com" target="_blank" style="color:#a78bfa;text-decoration:none;">polymarket.com</a>
            · Gamma API · 60s cache
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading Polymarket events…"):
        events = fetch_polymarket_events(150)
        all_markets = fetch_polymarket_markets(300)

    if not events:
        st.error("Couldn't reach the Polymarket API. It might be temporarily down — try Refresh in a moment.")
        return

    # --- KPIs ---------------------------------------------------------------
    total_24h = sum(float(e.get("volume24hr") or 0) for e in events)
    total_markets = sum(len(e.get("markets") or []) for e in events)
    total_liq = sum(float(e.get("liquidity") or 0) for e in events)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active events", f"{len(events):,}")
    k2.metric("Active markets", f"{total_markets:,}")
    k3.metric("24h volume", f"${total_24h/1e6:.1f}M")
    k4.metric("Total liquidity", f"${total_liq/1e6:.1f}M")

    # --- Top Movers (markets with biggest 24h price change) ---------------
    movers = []
    for m in all_markets:
        ch = m.get("oneDayPriceChange")
        try:
            ch = float(ch) if ch is not None else None
        except Exception:
            ch = None
        if ch is None or abs(ch) < 0.01:
            continue
        movers.append((m, ch))
    movers.sort(key=lambda x: -abs(x[1]))

    if movers:
        with st.expander(f"📊 Top Movers — biggest 24h price swings ({len(movers)} markets)", expanded=True):
            mc1, mc2 = st.columns(2)
            for i, (m, ch) in enumerate(movers[:10]):
                col = mc1 if i % 2 == 0 else mc2
                yes_pct = (_yes_price(m) or 0) * 100
                arrow = "🟢" if ch > 0 else "🔴"
                slug = m.get("slug") or ""
                question = (m.get("question") or "Untitled")[:90]
                url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"
                col.markdown(
                    f"{arrow} [{question}]({url})  \n"
                    f"&nbsp;&nbsp;&nbsp;**{yes_pct:.0f}%** · **{ch*100:+.1f}pt** 24h"
                )

    # --- Filters ------------------------------------------------------------
    # Aggregate tag popularity from the loaded events
    tag_counts: dict[str, int] = {}
    for e in events:
        for tag in (e.get("tags") or []):
            label = tag.get("label")
            if label and not label.lower().startswith(("rewards", "hide", "earn", "10-point")):
                tag_counts[label] = tag_counts.get(label, 0) + 1
    popular_tags = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])[:25]]

    f1, f2, f3 = st.columns([3, 2, 1])
    selected_tags = f1.multiselect(
        "Filter by category", options=popular_tags, default=[],
        help="Tags are pulled from currently-loaded events.",
    )
    search = f2.text_input("Search events", placeholder="Bitcoin, Election, World Cup…")
    sort_by = f3.selectbox(
        "Sort by",
        ["24h Volume", "Total Volume", "Liquidity", "Ending Soonest", "Newest"],
    )

    # --- Apply filters ------------------------------------------------------
    filtered = events
    if selected_tags:
        filtered = [
            e for e in filtered
            if any(t.get("label") in selected_tags for t in (e.get("tags") or []))
        ]
    if search.strip():
        s = search.strip().lower()
        filtered = [
            e for e in filtered
            if s in (e.get("title") or "").lower() or s in (e.get("description") or "").lower()
        ]

    sort_keys = {
        "24h Volume": lambda e: -float(e.get("volume24hr") or 0),
        "Total Volume": lambda e: -float(e.get("volume") or 0),
        "Liquidity": lambda e: -float(e.get("liquidity") or 0),
        "Ending Soonest": lambda e: e.get("endDate") or "9999-12-31",
        "Newest": lambda e: -(pd.Timestamp(e.get("createdAt")).value if e.get("createdAt") else 0),
    }
    filtered = sorted(filtered, key=sort_keys[sort_by])

    st.markdown(
        f"<div style='color:#6b7280;font-size:0.85rem;margin:8px 0;'>"
        f"Showing <b>{len(filtered)}</b> of {len(events)} events</div>",
        unsafe_allow_html=True,
    )

    # --- Cards grid ---------------------------------------------------------
    PAGE_SIZE = 30
    n_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input(
        f"Page (of {n_pages})", min_value=1, max_value=n_pages, value=1, step=1,
        key="market_page",
    )
    start = (int(page) - 1) * PAGE_SIZE
    page_events = filtered[start:start + PAGE_SIZE]

    cols_per_row = 3
    for r in range((len(page_events) + cols_per_row - 1) // cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= len(page_events):
                break
            with cols[c]:
                _render_event_card(page_events[idx])

    st.divider()
    st.caption(
        "**CAPRA Finance** aggregates markets and odds from Polymarket via their public Gamma API. "
        "Polymarket is a real-money decentralized prediction market — clicking through opens the live event."
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.{decimals}f}%"


def _fmt_num(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if abs(x) >= 1e12:
        return f"{x / 1e12:.{decimals}f}T"
    if abs(x) >= 1e9:
        return f"{x / 1e9:.{decimals}f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.{decimals}f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.{decimals}f}K"
    return f"{x:.{decimals}f}"


def _color(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#9ca3af"
    return "#16a34a" if value >= 0 else "#dc2626"


def _risk_color(level: str) -> str:
    return {
        "Low": "#16a34a",
        "Medium": "#eab308",
        "High": "#f97316",
        "Very High": "#dc2626",
        "Unknown": "#9ca3af",
    }.get(level, "#9ca3af")


def _rec_color(rec: str) -> str:
    return {
        "Strong Buy": "#15803d",
        "Buy": "#16a34a",
        "Hold": "#eab308",
        "Reduce": "#f97316",
        "Sell": "#dc2626",
    }.get(rec, "#9ca3af")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# Logo: prefer the SVG, fall back to PNG. Build a base64 data URI for inline use.
_LOGO_DIR = Path(__file__).parent
_LOGO_PATH = next((_LOGO_DIR / n for n in ("logo.svg", "logo.png") if (_LOGO_DIR / n).exists()), None)
_LOGO_DATA_URI = ""
if _LOGO_PATH:
    try:
        _mime = "image/svg+xml" if _LOGO_PATH.suffix == ".svg" else "image/png"
        _LOGO_DATA_URI = f"data:{_mime};base64," + base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    except Exception:
        _LOGO_DATA_URI = ""

st.set_page_config(
    page_title="CAPRA Finance",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar/top-bar branding (Streamlit's first-class logo slot).
if _LOGO_PATH:
    try:
        st.logo(str(_LOGO_PATH), size="large")
    except Exception:
        pass  # older Streamlit versions don't support st.logo

# ---------------------------------------------------------------------------
# Global CSS — typography, dark surfaces, refined inputs, custom scrollbars.
# Applied once per page; complements .streamlit/config.toml.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-feature-settings: 'cv02','cv03','cv04','cv11';
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a1428 0%, #0a0a0f 50%) fixed;
    }

    /* Main content padding */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

    /* Gradient title */
    h1 {
        background: linear-gradient(135deg, #c4b5fd 0%, #8b5cf6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    h2, h3 { color: #f3f4f6 !important; font-weight: 600 !important; letter-spacing: -0.015em; }

    /* Top page-selector radio — pill style */
    div[role="radiogroup"] { gap: 0.5rem; }
    div[role="radiogroup"] label {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 999px;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(139, 92, 246, 0.18);
        border-color: rgba(139, 92, 246, 0.4);
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
        border-color: #8b5cf6;
        color: white !important;
    }

    /* Metric cards — subtle glow */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(139, 92, 246, 0.3);
        background: rgba(139, 92, 246, 0.04);
    }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 0.75rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #f3f4f6 !important; font-weight: 700 !important; font-size: 1.6rem !important; }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #f3f4f6 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.1rem;
        font-weight: 500;
        color: white;
        transition: all 0.15s ease;
        box-shadow: 0 1px 3px rgba(139, 92, 246, 0.3);
    }
    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 23, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f3f4f6 !important;
        background: none !important;
        -webkit-text-fill-color: #f3f4f6 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600 !important;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Dividers */
    hr { border-color: rgba(255, 255, 255, 0.06) !important; margin: 1.5rem 0 !important; }

    /* Expanders */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.3); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.5); }

    /* Alerts / info / warning boxes */
    [data-testid="stAlertContainer"] {
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }

    /* Custom card class used by our HTML tiles */
    .gt-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(139,92,246,0.04) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
        height: 100%;
    }
    .gt-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.15);
    }

    .gt-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cinematic intro — Nolan-style logo reveal, plays once per browser session.
# A CSS-driven full-screen overlay renders, then the server sleeps while it
# animates, then reruns into the app. (Audio can't autoplay in browsers, so the
# drama is purely visual: slow fades, monumental type, a single violet accent.)
# ---------------------------------------------------------------------------
if not st.session_state.get("_intro_done"):
    _intro_logo = (
        f'<img class="logo" src="{_LOGO_DATA_URI}"/>' if _LOGO_DATA_URI
        else '<div class="logo" style="display:flex;align-items:center;justify-content:center;'
             'font-weight:800;font-size:3.5rem;color:#0a0a0f;">C</div>'
    )
    st.markdown(
        "<style>"
        '[data-testid="stHeader"]{display:none!important;}'
        ".stApp{background:#000!important;}"
        "#capra-intro{position:fixed;inset:0;z-index:2147483000;background:#000;display:flex;"
        "flex-direction:column;align-items:center;justify-content:center;overflow:hidden;"
        "font-family:'Inter',-apple-system,sans-serif;animation:introOut .7s ease 4.8s forwards;}"
        "#capra-intro .vignette{position:absolute;inset:0;background:"
        "radial-gradient(circle at 50% 44%,rgba(99,102,241,.12),rgba(0,0,0,0) 55%),"
        "radial-gradient(circle at 50% 50%,rgba(0,0,0,0) 38%,#000 100%);pointer-events:none;}"
        "#capra-intro .logo{width:138px;height:138px;border-radius:26px;background:#fff;opacity:0;"
        "transform:scale(.7);animation:logoIn 1.6s cubic-bezier(.2,.7,.2,1) .3s forwards,"
        "logoGlow 2.4s ease 1.4s forwards;}"
        "#capra-intro .line{width:0;height:1px;margin-top:32px;"
        "background:linear-gradient(90deg,transparent,#8b5cf6,transparent);"
        "animation:lineGrow 1.4s ease 1.2s forwards;}"
        "#capra-intro .word{margin-top:28px;font-weight:800;font-size:clamp(2.4rem,7vw,4.8rem);"
        "color:#fff;letter-spacing:.06em;text-align:center;line-height:1;opacity:0;"
        "animation:wordIn 1.4s ease 1.9s forwards;}"
        "#capra-intro .word .sub{display:block;font-size:.24em;font-weight:600;letter-spacing:.55em;"
        "color:#a78bfa;margin-top:14px;opacity:0;animation:wordIn 1.2s ease 2.6s forwards;}"
        "#capra-intro .tag{margin-top:26px;font-size:.78rem;font-weight:500;letter-spacing:.5em;"
        "color:#64748b;text-transform:uppercase;opacity:0;animation:wordIn 1.2s ease 3.2s forwards;}"
        "#capra-intro .timebar{position:absolute;bottom:0;left:0;height:2px;width:0;"
        "background:linear-gradient(90deg,#6366f1,#a78bfa);box-shadow:0 0 14px #8b5cf6;"
        "animation:timeFill 4.8s linear .2s forwards;}"
        "@keyframes logoIn{to{opacity:1;transform:scale(1);}}"
        "@keyframes logoGlow{0%{box-shadow:0 0 0 rgba(139,92,246,0);}"
        "50%{box-shadow:0 0 72px rgba(139,92,246,.55);}100%{box-shadow:0 0 38px rgba(139,92,246,.30);}}"
        "@keyframes lineGrow{to{width:min(440px,72vw);}}"
        "@keyframes wordIn{to{opacity:1;}}"
        "@keyframes timeFill{to{width:100%;}}"
        "@keyframes introOut{to{opacity:0;visibility:hidden;}}"
        "</style>"
        '<div id="capra-intro"><div class="vignette"></div>'
        f"{_intro_logo}"
        '<div class="line"></div>'
        '<div class="word">CAPRA<span class="sub">FINANCE</span></div>'
        '<div class="tag">Global Market Intelligence</div>'
        '<div class="timebar"></div></div>',
        unsafe_allow_html=True,
    )
    time.sleep(5.6)
    st.session_state["_intro_done"] = True
    st.rerun()

# Persistent user state (portfolio + alerts). Loaded from JSON on disk.
if "user_state" not in st.session_state:
    st.session_state.user_state = load_storage()

# Top header — CAPRA logo badge + wordmark + tagline.
_logo_inline = ""
if _LOGO_DATA_URI:
    _logo_inline = (
        f'<img src="{_LOGO_DATA_URI}" '
        f'style="height:52px;width:52px;border-radius:12px;background:#fff;'
        f'object-fit:cover;box-shadow:0 4px 18px rgba(139,92,246,0.25);">'
    )

st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0 14px 0;">'
    f'<div style="display:flex;align-items:center;gap:14px;">'
    f'{_logo_inline}'
    f'<div>'
    f'<h1 style="margin:0;font-size:1.85rem;line-height:1;">CAPRA Finance</h1>'
    f'<div style="color:#9ca3af;font-size:0.78rem;margin-top:4px;letter-spacing:0.02em;">'
    f'Global market intelligence · Stocks · Top movers'
    f'</div></div></div>'
    f'<div style="color:#6b7280;font-size:0.78rem;text-align:right;line-height:1.4;max-width:340px;">'
    f'Stocks &amp; movers via Yahoo Finance<br>'
    f'For research only, not investment advice.'
    f'</div></div>',
    unsafe_allow_html=True,
)

# Page selector — tab-style. Markets page renders + st.stop()s so the existing
# stock pipeline below only executes when on the Stocks view.
_active_view = st.radio(
    "View",
    ["📈 Global Stocks", "🚀 Top Movers", "💰 Stock Valuation"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)
if _active_view == "🚀 Top Movers":
    render_top_movers()
    st.stop()
if _active_view == "💰 Stock Valuation":
    render_valuation()
    st.stop()

# ---- Sidebar -------------------------------------------------------------
WATCHLIST_KEY = "_watchlist_selection"
CUSTOM_KEY = "_custom_tickers"      # tickers the user typed (sticky across reruns)
PRESET_SYMS_KEY = "_preset_symbols"  # symbols available from current presets


def _normalize_watchlist() -> None:
    """on_change for the watchlist box: uppercase, de-dupe, and remember any
    user-typed ticker that isn't part of the preset universe so it sticks."""
    raw = st.session_state.get(WATCHLIST_KEY, [])
    seen, out = set(), []
    for v in raw:
        vu = str(v).strip().upper()
        if vu and vu not in seen:
            seen.add(vu)
            out.append(vu)
    st.session_state[WATCHLIST_KEY] = out
    presets = st.session_state.get(PRESET_SYMS_KEY, set())
    custom = st.session_state.setdefault(CUSTOM_KEY, [])
    for vu in out:
        if vu not in presets and vu not in custom:
            custom.append(vu)


CUSTOM_NAMES_KEY = "_custom_names"  # {symbol: company name} for searched tickers


def _set_watchlist(tickers) -> None:
    """on_click for bulk buttons. Safe to set the widget key from a callback."""
    st.session_state[WATCHLIST_KEY] = list(tickers)


def _add_ticker(sym: str, name: str | None = None) -> None:
    """on_click for search results — add a ticker to the watchlist and remember it."""
    sym = (sym or "").strip().upper()
    if not sym:
        return
    wl = list(st.session_state.get(WATCHLIST_KEY, []))
    if sym not in wl:
        wl.append(sym)
    st.session_state[WATCHLIST_KEY] = wl
    custom = st.session_state.setdefault(CUSTOM_KEY, [])
    if sym not in custom:
        custom.append(sym)
    if name:
        st.session_state.setdefault(CUSTOM_NAMES_KEY, {})[sym] = name
    st.session_state["symbol_search"] = ""  # clear the search box after adding


with st.sidebar:
    st.header("Watchlist")

    if CUSTOM_KEY not in st.session_state:
        st.session_state[CUSTOM_KEY] = []

    # ---- Optional market presets (tucked away to keep things simple) --------
    with st.expander("➕ Add from market presets", expanded=False):
        selected_markets = st.multiselect(
            "Markets",
            options=list(MARKET_PRESETS.keys()),
            default=["USA", "Hong Kong", "India", "Europe", "Asia (ex-HK/India)"],
        )

    # Build the universe from selected presets.
    universe: dict[str, str] = {}
    for m in selected_markets:
        universe.update(MARKET_PRESETS[m])

    # Remember which symbols come from presets (used by the on_change callback).
    st.session_state[PRESET_SYMS_KEY] = set(universe.keys())

    # Sticky custom tickers (searched or typed) are part of the universe too,
    # carrying their proper company name when we have it.
    _custom_names = st.session_state.get(CUSTOM_NAMES_KEY, {})
    for t in st.session_state[CUSTOM_KEY]:
        universe.setdefault(t, _custom_names.get(t, t))

    if WATCHLIST_KEY not in st.session_state:
        st.session_state[WATCHLIST_KEY] = list(universe.keys())[: min(8, len(universe))]

    # Keep only tickers that still exist in the universe (preset OR custom).
    # (Runs BEFORE the widget is instantiated, so assigning the key is allowed.)
    st.session_state[WATCHLIST_KEY] = [
        t for t in st.session_state[WATCHLIST_KEY] if t in universe
    ]

    # ---- Primary control: search by COMPANY NAME or TICKER ------------------
    search_q = st.text_input(
        "🔎 Search & add — company name or ticker",
        key="symbol_search",
        placeholder="e.g. Royal, Apple, Tencent, Reliance, NVAX…",
    )
    scope_to_markets = False
    if selected_markets:
        scope_to_markets = st.checkbox(
            f"Only show results from selected market(s): {', '.join(selected_markets)}",
            value=True,
            key="search_scope",
        )
    if search_q and len(search_q.strip()) >= 2:
        results = fetch_symbol_search(search_q, limit=25)
        scoped_note = ""
        if scope_to_markets and selected_markets:
            filtered = [r for r in results if _symbol_in_markets(r["symbol"], selected_markets)]
            if filtered:
                results = filtered
            elif results:
                scoped_note = f" (no matches in {', '.join(selected_markets)} — showing all markets)"
        results = results[:10]
        if results:
            st.caption(f"Click to add ({len(results)} matches){scoped_note}:")
            for r in results:
                exch = f" · {r['exchange']}" if r["exchange"] else ""
                st.button(
                    f"➕ {r['symbol']} — {r['name']}{exch}",
                    key=f"addsym_{r['symbol']}",
                    use_container_width=True,
                    on_click=_add_ticker,
                    args=(r["symbol"], r["name"]),
                )
        else:
            st.caption("No matches — try a different spelling or paste the exact ticker below.")

    # ---- Current watchlist (also accepts a pasted symbol + Enter) -----------
    universe_keys = list(universe.keys())
    options_for_widget = list(dict.fromkeys(universe_keys + st.session_state[WATCHLIST_KEY]))

    chosen_tickers = st.multiselect(
        "⭐ Your watchlist",
        options=options_for_widget,
        key=WATCHLIST_KEY,
        accept_new_options=True,
        on_change=_normalize_watchlist,
        format_func=lambda t: f"{t} · {universe[t]}" if universe.get(t) and universe[t] != t else t,
        help="Search above by name, or paste a ticker here (e.g. 0700.HK, RELIANCE.NS) and press Enter.",
    )
    # Ensure freshly-typed tickers are usable downstream THIS run too.
    for t in chosen_tickers:
        universe.setdefault(t, t)

    # ---- Quick bulk-select from the loaded preset universe ------------------
    btn_cols = st.columns(4)
    btn_cols[0].button("Top 20", use_container_width=True, on_click=_set_watchlist, args=(universe_keys[:20],), help="From presets")
    btn_cols[1].button("Top 50", use_container_width=True, on_click=_set_watchlist, args=(universe_keys[:50],))
    btn_cols[2].button("Top 100", use_container_width=True, on_click=_set_watchlist, args=(universe_keys[:100],), help="~10–15s load")
    btn_cols[3].button("Clear", use_container_width=True, on_click=_set_watchlist, args=([],), key="clear_watchlist")

    if len(chosen_tickers) > 100:
        st.warning(
            f"Tracking **{len(chosen_tickers)}** tickers. Initial load may take "
            f"{len(chosen_tickers)//12 + 5}s and could hit Yahoo Finance rate limits."
        )

    st.divider()
    st.header("Analysis settings")
    period = st.selectbox("History period", ["1y", "3y", "5y", "10y"], index=2)
    benchmark_label = st.selectbox("Benchmark (for Beta)", list(BENCHMARKS.keys()), index=0)
    benchmark_symbol = BENCHMARKS[benchmark_label]

    auto_refresh = st.checkbox("Auto-refresh live quotes (every 30s)", value=True)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Data is cached for 30–600s to respect Yahoo's rate limits. "
        "If a ticker shows '—', Yahoo returned no data — check the symbol."
    )

    st.divider()
    with st.expander("💼 My Portfolio", expanded=False):
        st.caption("Holdings persist between sessions in `user_state.json`.")
        with st.form("add_holding", clear_on_submit=True):
            h_ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
            h_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
            h_cost = st.number_input("Avg cost / share", min_value=0.0, value=0.0, step=0.01, format="%.4f")
            if st.form_submit_button("Add holding") and h_ticker and h_qty > 0:
                st.session_state.user_state["portfolio"].append(
                    {"ticker": h_ticker, "qty": h_qty, "cost": h_cost}
                )
                save_storage(st.session_state.user_state)
                st.rerun()

        if st.session_state.user_state["portfolio"]:
            for i, h in enumerate(st.session_state.user_state["portfolio"]):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"`{h['ticker']}` · {h['qty']:g} @ {h['cost']:.2f}")
                if c2.button("✖", key=f"del_h_{i}"):
                    st.session_state.user_state["portfolio"].pop(i)
                    save_storage(st.session_state.user_state)
                    st.rerun()

    with st.expander("🔔 Price Alerts", expanded=False):
        with st.form("add_alert", clear_on_submit=True):
            a_ticker = st.text_input("Ticker for alert", placeholder="AAPL").strip().upper()
            a_op = st.selectbox("Condition", ["above", "below"])
            a_price = st.number_input("Price threshold", min_value=0.0, value=0.0, step=0.01, format="%.4f")
            if st.form_submit_button("Add alert") and a_ticker and a_price > 0:
                st.session_state.user_state["alerts"].append(
                    {"ticker": a_ticker, "op": a_op, "price": a_price}
                )
                save_storage(st.session_state.user_state)
                st.rerun()

        if st.session_state.user_state["alerts"]:
            for i, a in enumerate(st.session_state.user_state["alerts"]):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"`{a['ticker']}` {a['op']} **{a['price']:.2f}**")
                if c2.button("✖", key=f"del_a_{i}"):
                    st.session_state.user_state["alerts"].pop(i)
                    save_storage(st.session_state.user_state)
                    st.rerun()

# ---- Guard: nothing selected --------------------------------------------
if not chosen_tickers:
    st.info("Pick at least one ticker from the sidebar to get started.")
    st.stop()

tickers_tuple = tuple(chosen_tickers)

# ---- Load data ----------------------------------------------------------
with st.spinner(f"Loading {len(chosen_tickers)} tickers + benchmark…"):
    prices_df = fetch_history(tickers_tuple, period=period)
    # Scale volume window with period so Order Flow reflects the analysis horizon.
    vol_period = {"1y": "6mo", "3y": "1y", "5y": "2y", "10y": "2y"}.get(period, "3mo")
    volumes_df = fetch_volume(tickers_tuple, period=vol_period)
    bench_df = fetch_history((benchmark_symbol,), period=period)

if prices_df.empty:
    st.error("Could not load any price data. Yahoo Finance may be rate-limiting — wait a moment and click Refresh.")
    st.stop()

bench_series = bench_df.iloc[:, 0] if not bench_df.empty else pd.Series(dtype=float)
bench_returns = bench_series.pct_change().dropna() if not bench_series.empty else pd.Series(dtype=float)

# Visible status bar so users can SEE the settings actually applied.
data_start = prices_df.index.min().strftime("%Y-%m-%d") if not prices_df.empty else "—"
data_end = prices_df.index.max().strftime("%Y-%m-%d") if not prices_df.empty else "—"
st.info(
    f"📊 **Settings active:** Period = `{period}` ({data_start} → {data_end}) · "
    f"Benchmark = `{benchmark_label}` · "
    f"Tickers = `{len(chosen_tickers)}` · "
    f"Loaded at {datetime.now().strftime('%H:%M:%S')}"
)

with st.expander("📖 How to read this page", expanded=False):
    st.markdown(
        """
This dashboard tracks your watchlist live and scores each stock across performance, risk, and growth.
Here's what the key numbers mean.

#### 🟢 Live Quote cards
| On the card | Meaning |
|---|---|
| **% today** | Price change so far today |
| **Period change** (toggle 3M–5Y) | Total price return over the chosen window |
| **Risk pill** | Low → Very High, from volatility + worst drawdown |
| **Signal pill** | Best-Buy read: Strong Buy → Sell |
| **Revenue / Net Income + YoY** | Company scale and whether it's growing |
| **3Y CAGR** | Annualized price growth over 3 years |
| **Trend sparkline** | Mini price chart for the selected period |

#### 🏆 Best Buy Leaderboard
A composite **0–100 score** blending momentum, forecast upside, analyst upside, Sharpe ratio, valuation, and RSI.
Higher = stronger buy case. Use the **quick-sort presets** (Top 1Y Gainers, Hot 3M…) or click any column header.

#### ⚠️ Risk metrics (what good looks like)
| Metric | Plain meaning | Rule of thumb |
|---|---|---|
| **Volatility** | How much the price swings (annualized) | Lower = calmer; > 40% = high |
| **Max Drawdown** | Worst peak-to-trough fall | Smaller is safer |
| **Beta** | Sensitivity vs the benchmark | > 1 = more volatile than market |
| **Sharpe** | Return per unit of risk | > 1 is good, > 2 excellent |
| **RSI (14)** | Momentum gauge (0–100) | > 70 overbought, < 30 oversold |

#### 📈 CAGR & growth
**CAGR** = the smoothed annual growth rate (1Y/3Y/5Y). It strips out the noise of a single good or bad year —
the cleanest way to compare long-run performance across very different stocks.

#### 🔭 Other sections
- **Comparison chart** — every stock rebased to 100 so you can see relative performance regardless of price.
- **Sector heatmap** — today's moves grouped by sector (red→green), sized by trading value.
- **Speculative Watch** — penny/micro caps and high-volatility mid caps (high risk, high reward).
- **Forecast** — a *statistical* 30-day projection of the trend, **not** a guarantee.

> Tip: high momentum + high score is exciting, but always sanity-check value on the **💰 Stock Valuation** page.
> Educational tool, not investment advice.
        """
    )


# ---- Compute metrics for every ticker (parallelized) -------------------
def _compute_one(t: str) -> StockMetrics | None:
    if t not in prices_df.columns:
        return None
    vols = volumes_df[t] if (not volumes_df.empty and t in volumes_df.columns) else None
    return compute_metrics(t, universe.get(t, t), prices_df[t], bench_returns, vols)


all_metrics: list[StockMetrics] = []
n_tickers = len(chosen_tickers)
progress_bar = st.progress(0.0, text=f"Computing metrics for {n_tickers} tickers…")
# Cap workers to avoid hammering Yahoo and tripping its rate limiter.
max_workers = min(12, max(4, n_tickers))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(_compute_one, t): t for t in chosen_tickers}
    for i, fut in enumerate(as_completed(futures)):
        try:
            result = fut.result()
        except Exception:
            result = None
        if result is not None:
            all_metrics.append(result)
        progress_bar.progress((i + 1) / n_tickers, text=f"Computing metrics… {i + 1}/{n_tickers}")
progress_bar.empty()

# Preserve user's chosen order rather than completion order.
ticker_order = {t: i for i, t in enumerate(chosen_tickers)}
all_metrics.sort(key=lambda m: ticker_order.get(m.ticker, 1e9))

if not all_metrics:
    st.error("No metrics could be computed for the selected tickers.")
    st.stop()


# ---- Live quote tiles (auto-refresh fragment) ---------------------------
def _sparkline_svg(values, width: int = 140, height: int = 36, max_points: int = 80) -> str:
    """Inline SVG sparkline with a soft gradient area fill. Green if up, red if down."""
    vals = [float(v) for v in values if v == v]  # drop NaN
    if len(vals) < 2:
        return '<div style="height:36px;display:flex;align-items:center;color:#475569;font-size:0.65rem;">no data</div>'

    # Downsample evenly so 5Y daily (~1250 pts) stays light across many cards.
    if len(vals) > max_points:
        step = len(vals) / max_points
        vals = [vals[int(i * step)] for i in range(max_points)]

    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = (i / (n - 1)) * (width - 2) + 1
        y = (height - 2) - ((v - lo) / rng) * (height - 4) + 1
        pts.append((x, y))

    up = vals[-1] >= vals[0]
    color = "#10b981" if up else "#ef4444"
    gid = f"sg{abs(hash((round(vals[0], 4), round(vals[-1], 4), n))) % 100000}"
    line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_pts = f"1,{height-1} " + line_pts + f" {width-1},{height-1}"
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="width:100%;height:{height}px;display:block;">'
        f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<polygon points="{area_pts}" fill="url(#{gid})" stroke="none"/>'
        f'<polyline points="{line_pts}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def _growth_line(label: str, value: float | None) -> str:
    """Compact "Label · +12.3%" row, color-coded green/red."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (
            f"<div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#94a3b8;'>"
            f"<span>{label}</span><span>—</span></div>"
        )
    pct = value * 100
    color = "#10b981" if pct >= 0 else "#ef4444"
    return (
        f"<div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#94a3b8;'>"
        f"<span>{label}</span><span style='color:{color};font-weight:600;'>{pct:+.1f}%</span></div>"
    )


@st.fragment(run_every=30 if auto_refresh else None)
def live_quote_tiles() -> None:
    intraday = fetch_intraday(tickers_tuple)
    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.subheader(f"🟢 Live Quotes — {datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S %Z')}")
    with head_r:
        # Period toggle — controls which price-change % shows on every card.
        try:
            sel_period = st.segmented_control(
                "Price change period",
                options=["Today", "3M", "6M", "1Y", "3Y", "5Y"],
                default="Today",
                key="card_change_period",
                label_visibility="collapsed",
            )
        except Exception:
            sel_period = st.radio(
                "Price change period",
                options=["Today", "3M", "6M", "1Y", "3Y", "5Y"],
                horizontal=True, index=0, key="card_change_period_radio",
                label_visibility="collapsed",
            )
    sel_period = sel_period or "Today"

    cols_per_row = 3
    rows = math.ceil(len(all_metrics) / cols_per_row)
    for r in range(rows):
        cols = st.columns(cols_per_row, gap="small")
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= len(all_metrics):
                break
            m = all_metrics[idx]
            with cols[c]:
                # Latest intraday close if available; else daily close.
                live_price = m.price
                if not intraday.empty and m.ticker in intraday.columns:
                    last = intraday[m.ticker].dropna()
                    if not last.empty:
                        live_price = float(last.iloc[-1])

                color = _color(m.pct_change_1d)
                safe_name = (m.name or "").replace("<", "&lt;").replace(">", "&gt;")

                # Selected-period price change for the highlighted strip.
                if sel_period == "Today":
                    period_val = m.pct_change_1d
                else:
                    period_val = m.period_returns.get(sel_period, float("nan"))
                period_color = _color(period_val)
                period_str = _fmt_pct(period_val) if not (period_val is None or (isinstance(period_val, float) and math.isnan(period_val))) else "—"

                # Build the sparkline series for the selected period.
                if sel_period == "Today" and not intraday.empty and m.ticker in intraday.columns:
                    spark_vals = intraday[m.ticker].dropna().tolist()
                elif m.ticker in prices_df.columns:
                    s = prices_df[m.ticker].dropna()
                    if sel_period != "Today":
                        _days = {"3M": 91, "6M": 182, "1Y": 365, "3Y": 1095, "5Y": 1825}[sel_period]
                        if not s.empty:
                            s = s[s.index >= s.index[-1] - pd.Timedelta(days=_days)]
                    spark_vals = s.tolist()
                else:
                    spark_vals = []
                spark_html = _sparkline_svg(spark_vals)

                left_html = (
                    f'<div style="font-weight:600;font-size:0.92rem;color:#f3f4f6;letter-spacing:-0.01em;line-height:1.2;">{m.ticker}</div>'
                    f'<div title="{safe_name}" style="color:#cbd5e1;font-size:0.74rem;margin:4px 0 10px 0;line-height:1.35;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;min-height:2.05em;">{safe_name}</div>'
                    f'<div style="font-size:1.55rem;font-weight:700;color:#f3f4f6;letter-spacing:-0.02em;line-height:1.1;">{live_price:,.2f}</div>'
                    f'<div style="color:{color};font-size:0.85rem;font-weight:600;margin-top:3px;">{_fmt_pct(m.pct_change_1d)} today</div>'
                    '<div style="margin-top:8px;padding:6px 8px;border-radius:8px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.18);">'
                    f'<span style="font-size:0.64rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em;">{sel_period} change</span><br>'
                    f'<span style="font-size:1.1rem;font-weight:700;color:{period_color};">{period_str}</span>'
                    '</div>'
                    '<div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;">'
                    f'<span class="gt-pill" style="background:{_risk_color(m.risk_level)}22;color:{_risk_color(m.risk_level)};border:1px solid {_risk_color(m.risk_level)}55;">{m.risk_level}</span>'
                    f'<span class="gt-pill" style="background:{_rec_color(m.recommendation)}22;color:{_rec_color(m.recommendation)};border:1px solid {_rec_color(m.recommendation)}55;">{m.recommendation}</span>'
                    '</div>'
                )

                right_html = (
                    '<div style="font-size:0.66rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Revenue (TTM)</div>'
                    f'<div style="font-size:0.95rem;font-weight:600;color:#f3f4f6;line-height:1.1;">{_money(m.annual_revenue)}</div>'
                    f"{_growth_line('YoY growth', m.revenue_yoy)}"
                    '<div style="font-size:0.66rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-top:10px;margin-bottom:4px;">Net Income</div>'
                    f'<div style="font-size:0.95rem;font-weight:600;color:#f3f4f6;line-height:1.1;">{_money(m.net_income)}</div>'
                    f"{_growth_line('YoY growth', m.earnings_yoy)}"
                    '<div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);">'
                    f"{_growth_line('3Y CAGR', m.cagr_3y)}"
                    f"{_growth_line('QoQ sales', m.sales_growth_qoq)}"
                    '</div>'
                )

                spark_strip = (
                    '<div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);">'
                    f'<div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{sel_period} trend</div>'
                    f'{spark_html}'
                    '</div>'
                )

                card_html = (
                    '<div class="gt-card" style="display:flex;flex-direction:column;">'
                    '<div style="display:flex;gap:14px;">'
                    f'<div style="flex:1;min-width:0;">{left_html}</div>'
                    f'<div style="flex:1;min-width:0;border-left:1px solid rgba(255,255,255,0.08);padding-left:14px;">{right_html}</div>'
                    '</div>'
                    f'{spark_strip}'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                if st.button("📊 View detailed analysis", key=f"detail_{m.ticker}_{idx}", use_container_width=True):
                    show_company_detail(m.ticker, m.name)


live_quote_tiles()
st.divider()


# ---- Price Alerts banner -------------------------------------------------
alert_defs = st.session_state.user_state.get("alerts", [])
if alert_defs:
    alert_tickers = tuple({a["ticker"] for a in alert_defs})
    alert_prices_df = fetch_history(alert_tickers, period="5d")
    triggered: list[str] = []
    for a in alert_defs:
        tk = a["ticker"]
        if tk not in alert_prices_df.columns:
            continue
        series = alert_prices_df[tk].dropna()
        if series.empty:
            continue
        cur = float(series.iloc[-1])
        if a["op"] == "above" and cur >= a["price"]:
            triggered.append(f"🟢 **{tk}** at **{cur:,.2f}** ≥ {a['price']:.2f}")
        elif a["op"] == "below" and cur <= a["price"]:
            triggered.append(f"🔴 **{tk}** at **{cur:,.2f}** ≤ {a['price']:.2f}")
    if triggered:
        st.warning("**🔔 Price alerts triggered:** " + " · ".join(triggered))
    else:
        st.success(f"🔔 {len(alert_defs)} alert(s) armed — none triggered.")


# ---- Portfolio P&L -------------------------------------------------------
holdings = st.session_state.user_state.get("portfolio", [])
if holdings:
    st.subheader("💼 My Portfolio")
    pf_tickers = tuple({h["ticker"] for h in holdings})
    pf_prices_df = fetch_history(pf_tickers, period="5d")

    rows = []
    total_value = total_cost = total_day_change = 0.0
    for h in holdings:
        tk = h["ticker"]
        if tk in pf_prices_df.columns:
            series = pf_prices_df[tk].dropna()
            cur = float(series.iloc[-1]) if not series.empty else float("nan")
            prev = float(series.iloc[-2]) if len(series) >= 2 else cur
        else:
            cur = float("nan")
            prev = float("nan")
        qty = h["qty"]
        cost_basis = h["cost"] * qty
        market_value = cur * qty if not math.isnan(cur) else float("nan")
        unrealized = market_value - cost_basis if not math.isnan(market_value) else float("nan")
        ret_pct = (cur / h["cost"] - 1) * 100 if h["cost"] > 0 and not math.isnan(cur) else float("nan")
        day_change_dollar = (cur - prev) * qty if not math.isnan(cur) and not math.isnan(prev) else 0.0

        if not math.isnan(market_value):
            total_value += market_value
            total_cost += cost_basis
            total_day_change += day_change_dollar

        rows.append({
            "Ticker": tk,
            "Qty": qty,
            "Avg Cost": h["cost"],
            "Current Price": cur,
            "Market Value": market_value,
            "Cost Basis": cost_basis,
            "Unrealized P&L": unrealized,
            "Return %": ret_pct,
            "Day Change $": day_change_dollar,
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else float("nan")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Value", f"{total_value:,.2f}")
    k2.metric("Total Cost", f"{total_cost:,.2f}")
    k3.metric(
        "Unrealized P&L",
        f"{total_pnl:,.2f}",
        delta=f"{total_pnl_pct:+.2f}%" if not math.isnan(total_pnl_pct) else None,
    )
    k4.metric("Today's Change", f"{total_day_change:+,.2f}")

    pf_df = pd.DataFrame(rows)
    st.dataframe(
        pf_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg Cost": st.column_config.NumberColumn(format="%.2f"),
            "Current Price": st.column_config.NumberColumn(format="%.2f"),
            "Market Value": st.column_config.NumberColumn(format="%.2f"),
            "Cost Basis": st.column_config.NumberColumn(format="%.2f"),
            "Unrealized P&L": st.column_config.NumberColumn(format="%+.2f"),
            "Return %": st.column_config.NumberColumn(format="%+.2f"),
            "Day Change $": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    alloc_df = pf_df.dropna(subset=["Market Value"])
    if not alloc_df.empty and alloc_df["Market Value"].sum() > 0:
        alloc_fig = px.pie(
            alloc_df, values="Market Value", names="Ticker",
            title="Allocation by current market value", hole=0.4,
        )
        alloc_fig.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(alloc_fig, use_container_width=True)

    st.divider()


# ---- Best Buy Leaderboard -----------------------------------------------
st.subheader("🏆 Best Buy Leaderboard")
st.caption(
    "Composite score (0–100) blending 3M momentum, 30-day forecast upside, analyst upside, "
    "Sharpe ratio, valuation, and RSI extremes. Higher = stronger buy case."
)
st.caption("💡 Use a quick-sort preset below, or click any column header to sort manually.")

# Sort presets — one-click reordering. (col, ascending)
_LB_SORT_PRESETS = {
    "🏆 Highest Score": ("Best Buy Score", False),
    "📈 Top 1Y Gainers": ("1Y %", False),
    "📉 Top 1Y Losers": ("1Y %", True),
    "🔥 Hot 3M": ("3M %", False),
    "🎯 Best Forecast": ("Forecast 30d %", False),
}
if "_lb_sort" not in st.session_state:
    st.session_state["_lb_sort"] = ("Best Buy Score", False)

_preset_cols = st.columns(len(_LB_SORT_PRESETS))
for _col, (_label, _cfg) in zip(_preset_cols, _LB_SORT_PRESETS.items()):
    if _col.button(_label, use_container_width=True, key=f"lb_sort_{_label}"):
        st.session_state["_lb_sort"] = _cfg


def _pct100(x) -> float:
    return (x * 100) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else float("nan")


def _trend_series(ticker: str, days: int = 365, max_points: int = 60) -> list[float]:
    """Downsampled 1Y price series for the inline LineChartColumn sparkline."""
    if ticker not in prices_df.columns:
        return []
    s = prices_df[ticker].dropna()
    if s.empty:
        return []
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=days)]
    vals = s.tolist()
    if len(vals) > max_points:
        step = len(vals) / max_points
        vals = [vals[int(i * step)] for i in range(max_points)]
    return vals


leaderboard = pd.DataFrame(
    [
        {
            "Ticker": m.ticker,
            "Name": m.name,
            "Price": m.price,
            "Best Buy Score": m.best_buy_score,
            "Recommendation": m.recommendation,
            "Risk": m.risk_level,
            "1Y Trend": _trend_series(m.ticker),
            "3M %": _pct100(m.period_returns.get("3M")),
            "6M %": _pct100(m.period_returns.get("6M")),
            "1Y %": _pct100(m.period_returns.get("1Y")),
            "3Y %": _pct100(m.period_returns.get("3Y")),
            "5Y %": _pct100(m.period_returns.get("5Y")),
            "Forecast 30d %": _pct100(m.forecast_upside),
            "Analyst Upside %": _pct100(m.analyst_upside),
        }
        for m in all_metrics
    ]
)
_sort_col, _sort_asc = st.session_state["_lb_sort"]
if _sort_col not in leaderboard.columns:
    _sort_col, _sort_asc = "Best Buy Score", False
leaderboard = leaderboard.sort_values(_sort_col, ascending=_sort_asc, na_position="last")
st.caption(f"Sorted by **{_sort_col}** ({'ascending' if _sort_asc else 'descending'}).")

st.dataframe(
    leaderboard,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="%.2f"),
        "Best Buy Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
        "1Y Trend": st.column_config.LineChartColumn("1Y Trend", width="small"),
        "3M %": st.column_config.NumberColumn(format="%+.1f"),
        "6M %": st.column_config.NumberColumn(format="%+.1f"),
        "1Y %": st.column_config.NumberColumn(format="%+.1f"),
        "3Y %": st.column_config.NumberColumn(format="%+.1f"),
        "5Y %": st.column_config.NumberColumn(format="%+.1f"),
        "Forecast 30d %": st.column_config.NumberColumn(format="%+.1f"),
        "Analyst Upside %": st.column_config.NumberColumn(format="%+.1f"),
    },
)


# ---- Comparison chart ---------------------------------------------------
st.subheader("📊 Normalized Performance Comparison")
st.caption("All series rebased to 100 at the start of the window. Lets you see relative performance across markets and currencies.")
norm = prices_df[chosen_tickers].dropna(how="all").ffill()
norm = norm.div(norm.iloc[0]).mul(100)
fig_cmp = px.line(
    norm,
    labels={"value": "Indexed price (start = 100)", "Date": "Date"},
    title=None,
)
fig_cmp.update_layout(
    legend_title_text="Ticker", height=450, hovermode="x unified",
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
)
st.plotly_chart(fig_cmp, use_container_width=True)


# ---- Sector Heatmap ------------------------------------------------------
st.subheader("🗺️ Sector Heatmap")
st.caption("Treemap sized by avg dollar volume (proxy for market significance), colored by today's % change.")

heatmap_rows = []
for m in all_metrics:
    sector, industry = fetch_sector(m.ticker)
    heatmap_rows.append({
        "Ticker": m.ticker,
        "Name": m.name,
        "Sector": sector,
        "Industry": industry,
        "Change %": (m.pct_change_1d * 100) if not math.isnan(m.pct_change_1d) else 0.0,
        "Size": m.avg_dollar_volume if not math.isnan(m.avg_dollar_volume) else 1.0,
    })

heatmap_df = pd.DataFrame(heatmap_rows)
if not heatmap_df.empty:
    hm_fig = px.treemap(
        heatmap_df,
        path=[px.Constant("All sectors"), "Sector", "Ticker"],
        values="Size",
        color="Change %",
        color_continuous_scale=["#dc2626", "#f3f4f6", "#16a34a"],
        color_continuous_midpoint=0,
        hover_data={"Name": True, "Industry": True, "Change %": ":.2f"},
    )
    hm_fig.update_layout(
        height=500, margin=dict(t=10, l=10, r=10, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(hm_fig, use_container_width=True)

st.divider()


# ---- Speculative Watch — Penny stocks & High-Vol Mid Caps ---------------
st.subheader("⚡ Speculative Watch")
st.caption(
    "Two high-risk / high-reward buckets. Penny & Micro caps are illiquid and volatile by nature. "
    "Mid-cap stocks with elevated volatility can deliver outsized moves — both directions."
)

vol_threshold = st.slider(
    "High-volatility threshold (annualized %)",
    min_value=20, max_value=120, value=40, step=5,
    help="A stock is flagged 'high volatility' if its annualized stdev exceeds this.",
    key="vol_threshold",
)

penny_stocks = [m for m in all_metrics if m.cap_category in ("Penny/Nano", "Micro Cap")]
high_vol_midcaps = [
    m for m in all_metrics
    if m.cap_category == "Mid Cap"
    and not math.isnan(m.volatility)
    and m.volatility * 100 >= vol_threshold
]

spec_left, spec_right = st.columns(2)

with spec_left:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span class='gt-pill' style='background:#ef444422;color:#ef4444;border:1px solid #ef444455;'>"
        f"PENNY & MICRO</span> "
        f"<span style='color:#9ca3af;font-size:0.85rem;'>{len(penny_stocks)} matches</span></div>",
        unsafe_allow_html=True,
    )
    if penny_stocks:
        ps_rows = [{
            "Ticker": m.ticker,
            "Name": m.name,
            "Price": m.price,
            "Cap": m.market_cap,
            "Cap Bucket": m.cap_category,
            "1Y Trend": _trend_series(m.ticker),
            "1D %": m.pct_change_1d * 100 if not math.isnan(m.pct_change_1d) else float("nan"),
            "1M %": m.pct_change_1m * 100 if not math.isnan(m.pct_change_1m) else float("nan"),
            "Vol %": m.volatility * 100 if not math.isnan(m.volatility) else float("nan"),
            "Max DD %": m.max_drawdown * 100 if not math.isnan(m.max_drawdown) else float("nan"),
            "RSI": m.rsi_14,
            "Risk": m.risk_level,
        } for m in penny_stocks]
        ps_df = pd.DataFrame(ps_rows).sort_values("Vol %", ascending=False)
        st.dataframe(
            ps_df, use_container_width=True, hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.4f"),
                "Cap": st.column_config.NumberColumn(format="%.0f"),
                "1Y Trend": st.column_config.LineChartColumn("1Y Trend", width="small"),
                "1D %": st.column_config.NumberColumn(format="%+.2f"),
                "1M %": st.column_config.NumberColumn(format="%+.2f"),
                "Vol %": st.column_config.NumberColumn(format="%.1f"),
                "Max DD %": st.column_config.NumberColumn(format="%.1f"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        picked = st.selectbox(
            "Detailed analysis →", [""] + [m.ticker for m in penny_stocks],
            key="penny_pick", format_func=lambda x: x if x else "Pick a ticker…",
        )
        if picked:
            m = next((x for x in penny_stocks if x.ticker == picked), None)
            if m:
                show_company_detail(m.ticker, m.name)
    else:
        st.caption("No Penny/Micro caps in your current watchlist. Try adding small-cap tickers or loading an index that includes them.")

with spec_right:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span class='gt-pill' style='background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b55;'>"
        f"HIGH-VOL MID CAPS</span> "
        f"<span style='color:#9ca3af;font-size:0.85rem;'>{len(high_vol_midcaps)} matches (vol ≥ {vol_threshold}%)</span></div>",
        unsafe_allow_html=True,
    )
    if high_vol_midcaps:
        hv_rows = [{
            "Ticker": m.ticker,
            "Name": m.name,
            "Price": m.price,
            "Cap": m.market_cap,
            "1Y Trend": _trend_series(m.ticker),
            "Vol %": m.volatility * 100 if not math.isnan(m.volatility) else float("nan"),
            "3M %": m.pct_change_3m * 100 if not math.isnan(m.pct_change_3m) else float("nan"),
            "1Y CAGR %": m.cagr_1y * 100 if not math.isnan(m.cagr_1y) else float("nan"),
            "Max DD %": m.max_drawdown * 100 if not math.isnan(m.max_drawdown) else float("nan"),
            "Beta": m.beta,
            "RSI": m.rsi_14,
            "Score": m.best_buy_score,
        } for m in high_vol_midcaps]
        hv_df = pd.DataFrame(hv_rows).sort_values("Vol %", ascending=False)
        st.dataframe(
            hv_df, use_container_width=True, hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Cap": st.column_config.NumberColumn(format="%.0f"),
                "1Y Trend": st.column_config.LineChartColumn("1Y Trend", width="small"),
                "Vol %": st.column_config.NumberColumn(format="%.1f"),
                "3M %": st.column_config.NumberColumn(format="%+.2f"),
                "1Y CAGR %": st.column_config.NumberColumn(format="%+.2f"),
                "Max DD %": st.column_config.NumberColumn(format="%.1f"),
                "Beta": st.column_config.NumberColumn(format="%.2f"),
                "RSI": st.column_config.NumberColumn(format="%.0f"),
                "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            },
        )
        picked_hv = st.selectbox(
            "Detailed analysis →", [""] + [m.ticker for m in high_vol_midcaps],
            key="hv_pick", format_func=lambda x: x if x else "Pick a ticker…",
        )
        if picked_hv:
            m = next((x for x in high_vol_midcaps if x.ticker == picked_hv), None)
            if m:
                show_company_detail(m.ticker, m.name)
    else:
        st.caption(f"No mid-caps in your watchlist exceed the {vol_threshold}% volatility threshold. Lower the slider or add more mid-caps.")

st.divider()


# ---- CAGR + Risk tables --------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 CAGR Growth (annualized)")
    cagr_df = pd.DataFrame(
        [
            {
                "Ticker": m.ticker,
                "Name": m.name,
                "1Y CAGR %": m.cagr_1y * 100 if not math.isnan(m.cagr_1y) else float("nan"),
                "3Y CAGR %": m.cagr_3y * 100 if not math.isnan(m.cagr_3y) else float("nan"),
                "5Y CAGR %": m.cagr_5y * 100 if not math.isnan(m.cagr_5y) else float("nan"),
                "YTD %": m.pct_change_ytd * 100 if not math.isnan(m.pct_change_ytd) else float("nan"),
            }
            for m in all_metrics
        ]
    ).sort_values("3Y CAGR %", ascending=False)
    st.dataframe(
        cagr_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "1Y CAGR %": st.column_config.NumberColumn(format="%+.2f"),
            "3Y CAGR %": st.column_config.NumberColumn(format="%+.2f"),
            "5Y CAGR %": st.column_config.NumberColumn(format="%+.2f"),
            "YTD %": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

with col_right:
    st.subheader("⚠️ Risk Profile")
    beta_col = f"Beta vs {benchmark_label}"
    risk_df = pd.DataFrame(
        [
            {
                "Ticker": m.ticker,
                "Risk Level": m.risk_level,
                "Volatility % (ann.)": m.volatility * 100 if not math.isnan(m.volatility) else float("nan"),
                "Max Drawdown %": m.max_drawdown * 100 if not math.isnan(m.max_drawdown) else float("nan"),
                beta_col: m.beta,
                "Sharpe": m.sharpe,
                "RSI(14)": m.rsi_14,
            }
            for m in all_metrics
        ]
    ).sort_values("Volatility % (ann.)", ascending=True)
    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Volatility % (ann.)": st.column_config.NumberColumn(format="%.2f"),
            "Max Drawdown %": st.column_config.NumberColumn(format="%.2f"),
            beta_col: st.column_config.NumberColumn(format="%.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "RSI(14)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


# ---- Order Flow Proxy / Volume Movement ---------------------------------
st.subheader("🔁 Order Flow Proxy (Volume & Dollar Volume)")
st.caption(
    f"True per-order counts require a paid Level-2 feed. Volume window scales with chosen period — "
    f"currently using **{vol_period}** of history. Spike = today's volume ÷ 20-day average."
)

flow_rows = []
for m in all_metrics:
    if m.ticker not in volumes_df.columns:
        continue
    vols = volumes_df[m.ticker].dropna()
    if vols.empty:
        continue
    last_vol = float(vols.iloc[-1])
    avg_vol = float(vols.tail(20).mean())
    spike = last_vol / avg_vol if avg_vol > 0 else float("nan")
    move_today = m.pct_change_1d if not math.isnan(m.pct_change_1d) else 0
    flow_rows.append(
        {
            "Ticker": m.ticker,
            "Last Volume": last_vol,
            "20D Avg Volume": avg_vol,
            "Volume Spike (×avg)": spike,
            "Avg $ Volume (20D)": m.avg_dollar_volume,
            "Today's Move %": move_today * 100,
            "Signal": (
                "🔥 Heavy buying" if (not math.isnan(spike) and spike > 1.5 and move_today > 0) else
                "❄️ Heavy selling" if (not math.isnan(spike) and spike > 1.5 and move_today < 0) else
                "Normal flow"
            ),
        }
    )

flow_df = pd.DataFrame(flow_rows).sort_values("Volume Spike (×avg)", ascending=False)
st.dataframe(
    flow_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Last Volume": st.column_config.NumberColumn(format="%.0f"),
        "20D Avg Volume": st.column_config.NumberColumn(format="%.0f"),
        "Volume Spike (×avg)": st.column_config.NumberColumn(format="%.2fx"),
        "Avg $ Volume (20D)": st.column_config.NumberColumn(format="%.0f"),
        "Today's Move %": st.column_config.NumberColumn(format="%+.2f"),
    },
)


# ---- Forecast section ---------------------------------------------------
st.subheader("🔮 30-Day Price Forecast")
st.caption(
    "Holt-Winters exponential smoothing where available, with a log-linear fallback. "
    "This is a *statistical projection of historical trend*, not a guarantee. "
    "Use the upside number as a directional signal alongside fundamentals."
)

forecast_picks = st.multiselect(
    "Show forecast chart for:",
    options=[m.ticker for m in all_metrics],
    default=[m.ticker for m in all_metrics[:4]],
)

if forecast_picks:
    fc_fig = make_subplots(
        rows=math.ceil(len(forecast_picks) / 2),
        cols=2,
        subplot_titles=forecast_picks,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for i, tk in enumerate(forecast_picks):
        if tk not in prices_df.columns:
            continue
        prices = prices_df[tk].dropna()
        if prices.empty:
            continue
        _, path = _forecast(prices, horizon=30)
        if len(path) == 0:
            continue
        future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=len(path))
        row = i // 2 + 1
        col = i % 2 + 1
        fc_fig.add_trace(
            go.Scatter(x=prices.index[-180:], y=prices.values[-180:], name=f"{tk} actual", line=dict(color="#2563eb")),
            row=row, col=col,
        )
        fc_fig.add_trace(
            go.Scatter(x=future_dates, y=path, name=f"{tk} forecast", line=dict(color="#f97316", dash="dash")),
            row=row, col=col,
        )
    fc_fig.update_layout(
        height=300 * math.ceil(len(forecast_picks) / 2), showlegend=False,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    )
    st.plotly_chart(fc_fig, use_container_width=True)


# ---- Forecasted Profit / Analyst Targets --------------------------------
st.subheader("💰 Forecasted Profits & Analyst Targets")
st.caption("Per-share gain implied by both our 30-day statistical forecast and the consensus analyst price target.")

profit_df = pd.DataFrame(
    [
        {
            "Ticker": m.ticker,
            "Name": m.name,
            "Current Price": m.price,
            "30D Forecast": m.forecast_30d,
            "30D Profit/Share": (m.forecast_30d - m.price) if not math.isnan(m.forecast_30d) else float("nan"),
            "Analyst Target": m.analyst_target if m.analyst_target else float("nan"),
            "Analyst Profit/Share": ((m.analyst_target - m.price) if m.analyst_target else float("nan")),
            "P/E": m.pe_ratio if m.pe_ratio else float("nan"),
        }
        for m in all_metrics
    ]
).sort_values("30D Profit/Share", ascending=False)

st.dataframe(
    profit_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Current Price": st.column_config.NumberColumn(format="%.2f"),
        "30D Forecast": st.column_config.NumberColumn(format="%.2f"),
        "30D Profit/Share": st.column_config.NumberColumn(format="%.2f"),
        "Analyst Target": st.column_config.NumberColumn(format="%.2f"),
        "Analyst Profit/Share": st.column_config.NumberColumn(format="%.2f"),
        "P/E": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.divider()


# ---- News Headlines ------------------------------------------------------
st.subheader("📰 Latest News")
st.caption("Top headlines per ticker from Yahoo Finance. Click any link to open the source article.")

news_tickers = st.multiselect(
    "Show news for:",
    options=[m.ticker for m in all_metrics],
    default=[m.ticker for m in all_metrics[:5]],
    key="news_picker",
)

if news_tickers:
    news_cols = st.columns(min(len(news_tickers), 3))
    for i, tk in enumerate(news_tickers):
        col = news_cols[i % len(news_cols)]
        with col:
            st.markdown(f"**{tk}** · {universe.get(tk, tk)}")
            items = fetch_news(tk, limit=4)
            if not items:
                st.caption("_No recent news._")
                continue
            for n in items:
                title_line = (
                    f"[{n['title']}]({n['link']})" if n["link"] else n["title"]
                )
                st.markdown(
                    f"- {title_line}  \n  _{n['publisher']} · {n['time']}_"
                )

st.divider()
st.caption(
    "**CAPRA Finance** · For research and education only. Quotes may be delayed up to "
    "15 minutes on most exchanges. Forecasts are statistical projections, not investment advice. "
    "Always verify with primary sources before trading."
)
