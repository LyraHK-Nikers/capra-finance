"""Transparent stock-idea screening. No fabricated data, targets or catalysts."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pandas as pd
import streamlit as st
from fundamentals import annual_rows, series, growth, number, business_view
from valuation import DCF_DEFAULTS, _dcf, _dcf_missing_inputs, _quote_unit

UNIVERSE = {
    'PYPL':'PayPal', 'GOOGL':'Alphabet', 'META':'Meta Platforms', 'MSFT':'Microsoft',
    'AAPL':'Apple', 'AMZN':'Amazon', 'ADBE':'Adobe', 'CRM':'Salesforce',
    'CSCO':'Cisco', 'QCOM':'Qualcomm', 'AMD':'AMD', 'INTC':'Intel', 'NVDA':'NVIDIA',
    'UBER':'Uber', 'DIS':'Disney', 'NKE':'Nike', 'SBUX':'Starbucks', 'PFE':'Pfizer',
    'MRK':'Merck', 'JNJ':'Johnson & Johnson', 'BMY':'Bristol Myers Squibb',
    'GM':'General Motors', 'F':'Ford', 'FSLR':'First Solar', 'NVAX':'Novavax',
    'TDOC':'Teladoc', 'XOM':'Exxon Mobil', 'CVX':'Chevron', 'UNH':'UnitedHealth',
    'PG':'Procter & Gamble', 'KO':'Coca-Cola', 'NFLX':'Netflix',
}
PROFILES = ['Balanced', 'Conservative', 'Higher-risk Recovery']


def currency_from_payload(payload):
    points=[]
    for result in (payload.get('timeseries') or {}).get('result') or []:
        for key, values in result.items():
            if key.endswith('TotalRevenue') and isinstance(values,list):
                for value in values:
                    if value.get('currencyCode') and value.get('asOfDate'):
                        points.append((value['asOfDate'],value['currencyCode']))
    if not points:
        return None
    latest=max(d for d,c in points)
    codes={c for d,c in points if d==latest}
    return next(iter(codes)) if len(codes)==1 else None


def quarter_growth(values):
    if values.empty:
        return None
    latest = values.index[-1]
    prior = values[[320 <= (latest-d).days <= 410 for d in values.index]]
    return float(values.iloc[-1] / prior.iloc[-1] - 1) if len(prior) and prior.iloc[-1] > 0 else None


def _clip(value, scale):
    return min(1., max(0., value / scale)) if number(value) else 0.


def evaluate(snapshot, history, asof=None):
    """Turn one dated snapshot into facts shared by all three risk profiles."""
    now = pd.Timestamp(asof or datetime.now(timezone.utc)).tz_localize(None).normalize()
    info, fin = snapshot.get('info', {}), snapshot.get('financials', {})
    rows = annual_rows(fin)
    view = business_view(rows)
    issues = []
    if view is None:
        return {'symbol':snapshot['symbol'], 'eligible':False, 'issues':['Incomplete or mismatched annual statements']}
    date = view['date']
    if not 0 <= (now-date).days <= 550:
        issues.append('Annual statements are stale or future dated')
    qrev = series(snapshot.get('quarterly_income'), 'Total Revenue', 'Operating Revenue')
    qni = series(snapshot.get('quarterly_income'), 'Net Income Common Stockholders', 'Net Income')
    qgrowth = quarter_growth(qrev)
    if qrev.empty or not 0 <= (now-qrev.index[-1]).days <= 200 or not number(qgrowth):
        issues.append('Recent quarterly revenue and year-ago comparison are unavailable')
    revenue = float(rows['Revenue'].loc[date])
    profit = float(rows['Net income'].loc[date])
    fcf = float(rows['Free cash flow'].loc[date])
    ocf = rows['Operating cash flow'].get(date)
    debt, cash = info.get('totalDebt'), info.get('totalCash')
    if not all(number(v) and v >= 0 for v in [debt, cash]):
        issues.append('Debt or cash is unavailable')
    balance_date=snapshot.get('balance_date')
    if not balance_date or not 0 <= (now-pd.Timestamp(balance_date)).days <= 400:
        issues.append('A sufficiently recent common cash / debt balance date is unavailable')
    currency = info.get('financialCurrency') or snapshot.get('statement_currency')
    quote_currency = info.get('currency')
    if not currency or currency != _quote_unit(quote_currency)[0]:
        issues.append('Quote / statement currency is unverified or incompatible')
    prices = history.dropna() if history is not None else pd.Series(dtype=float)
    if prices.empty or len(prices)<60:
        issues.append('Insufficient recent price history')
        price = None
    else:
        price = float(prices.iloc[-1])
        age = (now-pd.Timestamp(prices.index[-1]).tz_localize(None).normalize()).days
        if age > 7 or age < 0:
            issues.append('Quote history is stale or future dated')
    shares = info.get('sharesOutstanding')
    cap = price * shares * _quote_unit(quote_currency)[1] if number(price) and number(shares) and shares>0 else None
    if not number(cap) or cap<=0:
        issues.append('Market capitalisation cannot be verified')
    metrics = view['metrics']
    annual_growth, opmargin, netmargin, fcfmargin = metrics
    leverage = (debt-cash)/fcf if number(debt) and number(cash) and fcf>0 else None
    cash_years = cash/abs(fcf) if number(cash) and fcf<0 else None
    fcfyield = fcf/cap if number(cap) and cap>0 else None
    earningsyield = profit/cap if number(cap) and cap>0 else None
    drawdown = 1-price/prices.max() if len(prices) and number(price) and prices.max()>0 else None
    ni_history = rows['Net income'].loc[:date]
    stable_profit = len(ni_history)>=3 and all(v>0 for v in ni_history.tail(3))
    qprofit = float(qni.iloc[-1]) if len(qni) and len(qrev) and qni.index[-1]==qrev.index[-1] else None
    fcf_values = rows['Free cash flow'].loc[:date]
    fcf_improving = len(fcf_values)>=2 and fcf_values.iloc[-1]>fcf_values.iloc[-2]
    qni_prior = qni[[320 <= (qni.index[-1]-d).days <= 410 for d in qni.index]] if len(qni) else qni
    loss_improving = len(qni_prior)>0 and number(qprofit) and qprofit>qni_prior.iloc[-1]
    score = (25*_clip(fcfyield,.08) + 15*_clip(earningsyield,.08) + 20*_clip(annual_growth,.20)
             + 15*_clip(fcfmargin,.20) + 10*_clip(opmargin,.25)
             + (15*max(0.,1-max(0.,leverage)/6) if number(leverage) else 0.))
    fact = dict(symbol=snapshot['symbol'], name=UNIVERSE.get(snapshot['symbol'],snapshot['symbol']),
                eligible=not issues, issues=issues, date=str(date)[:10], quarter=str(qrev.index[-1])[:10] if len(qrev) else None,
                price=price, currency=quote_currency, financial_currency=currency, revenue=revenue, profit=profit,
                fcf=fcf, ocf=ocf, growth=annual_growth, qgrowth=qgrowth, opmargin=opmargin, netmargin=netmargin,
                fcfmargin=fcfmargin, fcfyield=fcfyield, earningsyield=earningsyield, leverage=leverage,
                cash_years=cash_years, drawdown=drawdown, stable_profit=stable_profit, qprofit=qprofit,
                improving=bool(fcf_improving or loss_improving or (number(qgrowth) and qgrowth>0)), score=score,
                balance_date=balance_date, balance_age=(now-pd.Timestamp(balance_date)).days if balance_date else None,
                base=snapshot.get('base'))
    return fact


def profile_reasons(fact, profile):
    if not fact.get('eligible'):
        return fact.get('issues', ['Incomplete data'])
    failures = []
    if profile in ['Balanced','Conservative']:
        if fact['profit']<=0 or fact['opmargin']<=0:
            failures.append('Annual operating profit / net profit is not positive')
        if fact['fcf']<=0 or not number(fact['ocf']) or fact['ocf']<=0:
            failures.append('Positive operating and free cash flow required')
        if not number(fact['qprofit']) or fact['qprofit']<0:
            failures.append('Latest reported quarter is loss-making or unavailable')
        growth_floor, quarter_floor, debt_limit = ((0.,0.,3.) if profile=='Conservative' else (-.05,-.10,6.))
        if fact['growth']<growth_floor or fact['qgrowth']<quarter_floor:
            failures.append('Annual / quarterly sales trend is below this profile’s limit')
        if not number(fact['leverage']) or fact['leverage']>debt_limit:
            failures.append('Net debt is too high relative to free cash flow')
        if profile=='Conservative' and not fact['stable_profit']:
            failures.append('Three consecutive profitable annual periods required')
    else:
        if not number(fact['drawdown']) or fact['drawdown']<.30:
            failures.append('Price is less than 30% below its available five-year high')
        if not fact['improving']:
            failures.append('No improvement in quarterly profit, sales or annual cash flow')
        if fact['fcf']<=0 and (not number(fact['cash_years']) or fact['cash_years']<2):
            failures.append('Cash generation or at least two years of annualised cash-burn coverage required')
        if fact['fcf']<=0 and fact.get('balance_age',999)>200:
            failures.append('Cash-burning recoveries require a more recent balance sheet')
        if fact['qgrowth']<-.50:
            failures.append('Latest quarterly sales fell more than 50%; investigate one-off/milestone revenue')
    return failures


def shortlist(facts, profile, count=5):
    eligible = [f for f in facts if not profile_reasons(f,profile)]
    def rank(f):
        return f['score']*.75 + 25*_clip(f['drawdown'],.75) if profile=='Higher-risk Recovery' else f['score']
    return sorted(eligible,key=lambda f:(-rank(f),f['symbol']))[:count]


def entry_range(fact):
    base = fact.get('base')
    if not base or fact.get('opmargin',0)<=0:
        return None, 'A recovery forecast is needed before estimating an entry price.'
    base = dict(base, financial_currency=fact['financial_currency'])
    if _dcf_missing_inputs(base):
        return None, 'Valuation inputs are incomplete.'
    values=[]
    for rate in [.10,.12]:
        a=dict(DCF_DEFAULTS,da=base['da'],capex=base['capex'],wacc=rate,termg=.025)
        model=_dcf(base,a)
        if model['error'] or not number(model['ps']) or model['ps']<=0:
            return None, 'The cash-flow scenario does not support a positive target.'
        values.append(model['ps']*.8)
    return (min(values),max(values)), '10–12% discount rates, 2.5% terminal growth, 20% margin of safety; scenario thresholds, not guaranteed targets.'


@st.cache_data(ttl=1800, show_spinner=False)
def load_snapshots(_fetch_snapshot):
    output=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(_fetch_snapshot,s):s for s in UNIVERSE}
        for future in as_completed(futures):
            symbol=futures[future]
            try:
                output.append(future.result())
            except Exception as exc:
                output.append(dict(symbol=symbol,error=type(exc).__name__))
    return output


def render_stock_ideas(fetch_snapshot,fetch_history):
    st.subheader('Top 5 Buy Ideas — three investing styles')
    st.caption('A US-stock research shortlist, refreshed from reported financials and prices. A large price fall or an EPS beat alone is not enough to qualify.')
    st.write('**Balanced:** profitable businesses at reasonable valuations. **Conservative:** stronger cash generation and balance sheets. **Recovery:** depressed prices with evidence of improvement; substantial loss risk.')
    if st.button('Refresh stock ideas'):
        st.cache_data.clear()
        st.rerun()
    with st.spinner(f'Screening {len(UNIVERSE)} companies — the first scan may take a minute…'):
        snapshots=load_snapshots(fetch_snapshot)
        history=fetch_history(tuple(UNIVERSE),period='5y')
        facts=[evaluate(s,history[s['symbol']] if s['symbol'] in history else None) for s in snapshots]
    st.caption(f"Screen run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · {sum(f.get('eligible',False) for f in facts)}/{len(UNIVERSE)} companies have sufficient verified data. Rankings can overlap across styles. This is a limited universe, not the entire US market.")
    for tab,profile in zip(st.tabs(PROFILES),PROFILES):
        with tab:
            picks=shortlist(facts,profile)
            if len(picks)<5:
                st.info(f'{len(picks)} stocks meet this profile right now. Missing slots stay empty rather than relaxing the checks.')
            if picks:
                table=[]
                for rank,f in enumerate(picks,1):
                    zone,note=entry_range(f)
                    entry=f'{zone[0]:,.2f}–{zone[1]:,.2f}' if zone else 'Needs further valuation'
                    status='Valuation pending' if not zone else 'At/below entry range' if f['price']<=zone[1] else 'Wait for a lower price'
                    why=f"FCF yield {f['fcfyield']:.1%}; annual sales {f['growth']:+.1%}; latest-quarter sales {f['qgrowth']:+.1%}"
                    risk=('Recovery may fail; cash-burn coverage is only a rough annualised estimate.' if profile=='Higher-risk Recovery' else
                          'Sales are slowing; verify the next earnings report.' if f['qgrowth']<f['growth'] else
                          'Valuation depends on future cash generation; results can disappoint.')
                    table.append({'Rank':rank,'Stock':f"{f['symbol']} · {f['name']}",'Price':f"{f['price']:,.2f} {f['currency']}",
                                  'Research entry range':entry,'Indication':status,'Why it qualifies':why,'Main risk':risk})
                st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
                for rank,f in enumerate(picks,1):
                    with st.expander(f"{rank}. {f['symbol']} — why, risks and price assumptions"):
                        zone,note=entry_range(f)
                        st.write(f"Annual operating margin {f['opmargin']:.1%}; FCF margin {f['fcfmargin']:.1%}; net debt / FCF {f['leverage']:.1f}×." if number(f['leverage']) else f"Annual FCF margin {f['fcfmargin']:.1%}; cash / annual cash burn {f['cash_years']:.1f} years.")
                        st.write(note)
                        st.caption(f"Annual period: {f['date']} · latest quarter: {f['quarter']} · balance sheet: {f['balance_date']} · drawdown from available five-year high: {f['drawdown']:.1%}. Qualitative catalysts and one-off earnings require review.")
                        st.markdown(f"[Financial statements](https://finance.yahoo.com/quote/{f['symbol']}/financials/) · [SEC filings](https://www.sec.gov/edgar/browse/?CIK={f['symbol']}&owner=exclude)")
    with st.expander('Why NVAX / PayPal / TDOC did or did not qualify'):
        for symbol in ['NVAX','PYPL','TDOC']:
            f=next((x for x in facts if x['symbol']==symbol),None)
            if f:
                st.markdown(f'**{symbol}**')
                for profile in PROFILES:
                    reasons=profile_reasons(f,profile)
                    st.write(f"{profile}: " + ('Meets the screen; ranking determines whether it reaches the top five.' if not reasons else '; '.join(reasons)))
    with st.expander('Screening rules and full coverage'):
        st.write('Score weights: free-cash-flow yield 25%, earnings yield 15%, annual sales growth 20%, FCF margin 15%, operating margin 10%, net-debt coverage 15%. Recovery adds a drawdown component after improvement/liquidity checks. Scores are heuristics, not probabilities of profit.')
        st.write('Conservative requires three profitable annual periods, non-declining annual and quarterly sales, positive cash flow, and net debt no greater than three years of FCF. Balanced allows small sales declines and up to six years of net debt/FCF. Recovery requires a ≥30% drawdown and operating improvement, plus positive FCF or two years of cash-burn coverage.')
        st.dataframe(pd.DataFrame([{'Ticker':f['symbol'],'Data status':'Complete' if f.get('eligible') else '; '.join(f['issues'])} for f in sorted(facts,key=lambda x:x['symbol'])]),hide_index=True,use_container_width=True)
