"""Date-aligned reported statement fallbacks; never replace missing data with zero."""
import math
from numbers import Real
import pandas as pd
from valuation import _quote_unit


def number(x):
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def series(frame, *labels):
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    # Combine aliases by date rather than choosing an incomplete row globally.
    result = pd.Series(dtype=float)
    for label in labels:
        if label in frame.index:
            row = frame.loc[label].copy()
            row.index = pd.to_datetime(row.index)
            row = pd.to_numeric(row, errors='coerce').replace([float('inf'), -float('inf')], float('nan'))
            result = result.combine_first(row)
    return result.dropna().sort_index()


def annual_rows(fin):
    inc, cf = fin.get('income'), fin.get('cashflow')
    rows = {
        'Revenue': series(inc, 'Total Revenue', 'Operating Revenue'),
        'Gross profit': series(inc, 'Gross Profit'),
        'Operating profit': series(inc, 'Operating Income', 'Total Operating Income As Reported'),
        'Net income': series(inc, 'Net Income Common Stockholders', 'Net Income'),
        'Operating cash flow': series(cf, 'Operating Cash Flow', 'Total Cash From Operating Activities'),
        'Capital expenditure': series(cf, 'Capital Expenditure', 'Purchase Of PPE').abs(),
        'Free cash flow': series(cf, 'Free Cash Flow'),
    }
    calculated = (rows['Operating cash flow'] - rows['Capital expenditure']).dropna()
    rows['Free cash flow'] = rows['Free cash flow'].combine_first(calculated).sort_index()
    return rows


def growth(s):
    if len(s) < 2:
        return None
    end = s.index[-1]
    prior = s[[330 <= (end - d).days <= 400 for d in s.index]]
    if prior.empty or prior.iloc[-1] <= 0:
        return None  # loss-to-profit transitions have no meaningful growth percentage
    return float(s.iloc[-1] / prior.iloc[-1] - 1)


def enrich_info(info, fin, fast=None):
    out = {k: (None if isinstance(v, Real) and not number(v) else v) for k, v in info.items()}
    derived = []
    def put(key, value):
        if out.get(key) is None and value is not None and (not isinstance(value, Real) or number(value)):
            out[key] = value
            derived.append(key)
    for k, v in (fast or {}).items():
        put(k, v)
    rows = annual_rows(fin)
    # All flow ratios and amounts use one latest income/cash-flow fiscal date.
    common = rows['Revenue'].index.intersection(rows['Operating cash flow'].index)
    date = common.max() if len(common) else (rows['Revenue'].index.max() if len(rows['Revenue']) else None)
    def flow(label):
        s = rows[label]
        return float(s.loc[date]) if date is not None and date in s.index else None
    rev, ni = flow('Revenue'), flow('Net income')
    flow_keys = ['totalRevenue', 'netIncomeToCommon', 'operatingCashflow', 'freeCashflow']
    use_annual = rev is not None and any(not number(out.get(k)) for k in flow_keys)
    if use_annual:
        # A partial summary must not pair annual cash with trailing profits.
        for key in flow_keys + ['grossMargins', 'operatingMargins', 'profitMargins', 'revenueGrowth', 'earningsGrowth']:
            out[key] = None
    for key, label in [('totalRevenue','Revenue'), ('netIncomeToCommon','Net income'),
                       ('operatingCashflow','Operating cash flow'), ('freeCashflow','Free cash flow')]:
        put(key, flow(label))
    for key, label in [('grossMargins','Gross profit'), ('operatingMargins','Operating profit'), ('profitMargins','Net income')]:
        v = flow(label)
        put(key, v / rev if v is not None and rev and rev > 0 else None)
    put('revenueGrowth', growth(rows['Revenue'].loc[:date]) if date is not None else None)
    put('earningsGrowth', growth(rows['Net income'].loc[:date]) if date is not None else None)
    bs = fin.get('balance')
    def bal(*labels):
        s = series(bs, *labels)
        return (float(s.iloc[-1]), s.index[-1]) if len(s) else (None, None)
    debt, bd = bal('Total Debt')
    cash, cd = bal('Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents')
    equity, ed = bal('Stockholders Equity', 'Common Stock Equity')
    assets, ad = bal('Current Assets')
    liabilities, ld = bal('Current Liabilities')
    # Fill the net-debt pair from a common balance date, never two different years.
    if out.get('totalDebt') is None or out.get('totalCash') is None:
        debts = series(bs, 'Total Debt')
        cashes = series(bs, 'Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents')
        common_balance = debts.index.intersection(cashes.index)
        if len(common_balance):
            balance_date = common_balance.max()
            out['totalDebt'] = float(debts.loc[balance_date])
            out['totalCash'] = float(cashes.loc[balance_date])
            derived.extend(['totalDebt','totalCash'])
            out['_balance_period'] = str(balance_date)[:10]
    put('debtToEquity', debt / equity * 100 if debt is not None and equity and equity > 0 and bd == ed else None)
    put('currentRatio', assets / liabilities if assets is not None and liabilities and liabilities > 0 and ad == ld else None)
    put('sharesOutstanding', bal('Ordinary Shares Number')[0])
    price, shares = out.get('currentPrice') or out.get('regularMarketPrice'), out.get('sharesOutstanding')
    if number(price) and number(shares) and price > 0 and shares > 0:
        put('marketCap', price * shares * _quote_unit(out.get('currency'))[1])
    # Derived multiples explicitly use the reported annual period, not TTM EPS.
    cap = out.get('marketCap')
    currency_match = out.get('financialCurrency') and out.get('financialCurrency') == _quote_unit(out.get('currency'))[0]
    put('annualPE', cap / ni if currency_match and number(cap) and ni is not None and ni > 0 else None)
    if currency_match and number(cap) and equity and equity > 0:
        put('priceToBook', cap / equity)
    out['_statement_period'] = str(date)[:10] if date is not None else None
    out['_flow_basis'] = 'annual' if use_annual else 'summary'
    out['_derived_fields'] = derived
    out['_statement_rows'] = rows
    return out


def business_view(rows):
    """Comparable four-measure business screen; no price-based buy recommendation."""
    common = rows['Revenue'].index
    for label in ['Operating profit', 'Net income', 'Free cash flow']:
        common = common.intersection(rows[label].index)
    if not len(common):
        return None
    d = common.max()
    latest_revenue = rows['Revenue'].index.max()
    if (latest_revenue - d).days > 120:
        return None  # do not silently rank using older complete statements
    rev = rows['Revenue'].loc[d]
    if rev <= 0:
        return None
    metrics = [growth(rows['Revenue'].loc[:d]), rows['Operating profit'].loc[d] / rev,
               rows['Net income'].loc[d] / rev, rows['Free cash flow'].loc[d] / rev]
    return {'date': d, 'metrics': metrics} if all(number(v) for v in metrics) else None
