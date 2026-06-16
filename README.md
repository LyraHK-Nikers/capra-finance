# CAPRA Finance

A global market-intelligence dashboard built with Streamlit.

- **📈 Global Stocks** — live watchlist with risk metrics, CAGR, best-buy scoring, sparklines, sector heatmap, forecasts, and news. Covers USA, Hong Kong, India, Europe, and Asia.
- **🚀 Top Movers** — live top 50 gainers/losers per market with analyst ratings (Yahoo screener).
- **💰 Stock Valuation** — 5-year DCF with reverse-DCF (market-implied growth), CAPM-based WACC, cash-flow projection, value bridge, sensitivity heatmaps, and peer comps.

Data via Yahoo Finance. For research and education only — not investment advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Pick this repo, branch `main`, main file `app.py`.
4. Deploy.
