"""Portfolio arithmetic grouped by currency, with explicit quote coverage."""
from collections import defaultdict
from valuation import _finite, _quote_unit


def portfolio_rows(holdings, quotes):
    rows = []
    groups = defaultdict(list)
    for h in holdings:
        quote = quotes.get(h['ticker'], {})
        currency, factor = _quote_unit(quote.get('currency'))
        qty, cost = h.get('qty'), h.get('cost')
        if not all(_finite(v) for v in [qty, cost]) or qty <= 0 or cost < 0:
            continue
        price, prev = quote.get('price'), quote.get('previous')
        price = price if _finite(price) and price > 0 else None
        prev = prev if _finite(prev) and prev > 0 else None
        value = price * qty * factor if price is not None else None
        basis = cost * qty * factor
        pnl = value - basis if value is not None else None
        day = (price - prev) * qty * factor if price is not None and prev is not None else None
        row = {'Ticker': h['ticker'], 'Qty': qty, 'Quote unit': quote.get('currency') or 'Unknown',
               'Avg Cost': cost, 'Current Price': price, 'Value currency': currency or 'Unknown',
               'Market Value': value, 'Cost Basis': basis, 'Unrealized P&L': pnl,
               'Return %': (value / basis - 1) * 100 if value is not None and basis > 0 else None,
               'Day Change': day}
        rows.append(row)
        if currency:
            groups[currency].append(row)
    totals = []
    for currency, group in sorted(groups.items()):
        complete = all(r['Market Value'] is not None for r in group)
        day_complete = all(r['Day Change'] is not None for r in group)
        cost = sum(r['Cost Basis'] for r in group)
        value = sum(r['Market Value'] for r in group) if complete else None
        totals.append({'Currency': currency, 'Quote coverage': f"{sum(r['Market Value'] is not None for r in group)}/{len(group)}",
                       'Cost': cost, 'Value': value, 'P&L': value-cost if value is not None else None,
                       'Return %': (value/cost-1)*100 if value is not None and cost>0 else None,
                       'Day Change': sum(r['Day Change'] for r in group) if day_complete else None})
    return rows, totals


def render_portfolio(holdings, fetch_history, fetch_info):
    import pandas as pd
    import streamlit as st
    st.subheader('My Portfolio')
    history = fetch_history(tuple(dict.fromkeys(h['ticker'] for h in holdings)), period='5d')
    quotes = {}
    for ticker in dict.fromkeys(h['ticker'] for h in holdings):
        values = history[ticker].dropna() if ticker in history else pd.Series(dtype=float)
        quotes[ticker] = {'currency': fetch_info(ticker).get('currency'),
                          'price': float(values.iloc[-1]) if len(values) else None,
                          'previous': float(values.iloc[-2]) if len(values) > 1 else None}
    rows, totals = portfolio_rows(holdings, quotes)
    st.caption('Totals are separate by currency. No FX conversion is assumed. Avg cost and price use the quote unit; values use the currency shown. Missing quotes suppress the affected total and P&L.')
    st.dataframe(pd.DataFrame(totals), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if any(r['Value currency']=='Unknown' for r in rows):
        st.info('Positions with an unknown currency are excluded from totals.')
    st.divider()
