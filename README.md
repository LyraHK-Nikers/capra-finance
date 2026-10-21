# CAPRA Finance

A global market-intelligence dashboard built with Streamlit.

- **📈 Global Stocks** — live watchlist with risk metrics, CAGR, best-buy scoring, sparklines, sector heatmap, forecasts, and news. Covers USA, Hong Kong, India, Europe, and Asia.
- **🚀 Top Movers** — live top 50 gainers/losers per market with analyst ratings (Yahoo screener).
- **💰 Stock Valuation** — 5-year DCF with reverse-DCF (market-implied growth), CAPM-based WACC, cash-flow projection, value bridge, sensitivity heatmaps, and peer comps.

Data via Yahoo Finance. For research and education only — not investment advice.

## Valuation behavior

- DCF inputs remain unavailable when financial data is missing. Net income, debt,
  cash and growth are never invented from generic margins or earnings growth.
- D&A and capital-spending starting ratios use the same annual statement period.
  That period is displayed. Other forecast assumptions remain editable.
- Negative initial revenue growth stays negative and approaches terminal growth
  gradually. The main model, sensitivity tables and reverse DCF use one calculation.
- A missing beta disables the CAPM estimate. Users can explicitly select a manual
  discount rate. Currency mismatches require conversion before valuation.
- A nonpositive terminal cash flow needs a recovery model. A perpetual-growth
  target is not generated. The discount rate must exceed terminal growth by 0.5pp.
- Health verdicts and buy labels are withheld when financial coverage is incomplete.
- The DCF comparison describes selected assumptions, not an automatic recommendation.

Offline regression tests:

```bash
python -m unittest test_valuation test_valuation_data -v
```

With the app dependencies installed, also run `python -m unittest test_valuation_ui -v`
to verify the real Streamlit valuation controls with offline fixtures.

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
# October 2026 reliability and comparison update

Follow-up audit fixes: terminal cash flow is rebuilt in year six with stable
growth and matching working capital, rather than growing the previous year's
cash flow. See [terminal value methodology](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm).
Pence quotes convert correctly to pounds; portfolio totals are grouped by currency
and suppressed if quotes are incomplete. Short histories no longer masquerade as
full multi-year CAGR. A GitHub Actions workflow runs the offline regression suite
on each push and pull request.

- Compare Stocks is the landing page, with two companies alongside each other, a
  four-measure business preference, three fiscal years of P&L and cash flow, YoY
  growth and downloadable CSV tables. Each company's fiscal dates are visible.
- Partial summary-feed failures fall back to date-aligned annual statements.
  Annual cash flows and profits are kept together; missing figures remain blank.
- Scenario entry levels use a common discount rate and an explicit margin of
  safety. Missing financial currency requires verification before a target appears.
- Price-trend scores use signal labels rather than automatic Buy/Sell instructions.
- Portfolios and alerts are private to the Streamlit session. The former shared
  server file is no longer read or written; there is no per-person identity in
  shared-password mode, so persistent private portfolios require a separate store.
- Run `python -m unittest discover -p "test_*.py"` for valuation, statement fallback,
  comparison and Streamlit UI checks.

