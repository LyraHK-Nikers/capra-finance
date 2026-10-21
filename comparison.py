"""A compact two-company review with fiscal-year P&L and cash flows alongside."""
import pandas as pd
import streamlit as st
from fundamentals import annual_rows, business_view, growth, number
from valuation import DCF_DEFAULTS, _dcf, _dcf_missing_inputs, _quote_unit


def money(v):
    return f'{v / 1e6:,.1f}' if number(v) else '—'


def pct(v):
    return f'{v:+.1%}' if number(v) else '—'


def preference(left, right, names):
    views = [business_view(left), business_view(right)]
    if any(v is None for v in views):
        return 'No preference yet — complete annual revenue, profit and cash-flow history is needed.'
    today = pd.Timestamp.now().normalize()
    if any((today - v['date']).days > 730 for v in views):
        return 'No preference yet — financial statements are more than two years old.'
    if abs((views[0]['date'] - views[1]['date']).days) > 120:
        return 'No preference yet — the latest fiscal years are too far apart for a fair comparison.'
    wins = [0, 0]
    for a, b in zip(views[0]['metrics'], views[1]['metrics']):
        if abs(a - b) > .005:
            wins[0 if a > b else 1] += 1
    if max(wins) < 3:
        return 'Mixed result — neither company leads on at least three of the four business measures.'
    best = 0 if wins[0] > wins[1] else 1
    return (f'{names[best]} is the stronger business on this screen: it leads on {wins[best]} of 4 measures '
            '(revenue growth, operating margin, net margin and free-cash-flow margin). '
            'Review price, debt and business risks before an investment decision.')


def statement_table(histories, symbols, labels):
    years = sorted(set().union(*(set(d.year for d in s.index) for rows in histories for s in rows.values())), reverse=True)[:3]
    output = []
    for year in years:
        for label in labels:
            row = {'Fiscal year': str(year), 'Metric': label}
            for sym, rows in zip(symbols, histories):
                values = rows[label]
                dates = [d for d in values.index if d.year == year]
                date = max(dates) if dates else None
                v = values.get(date) if date is not None else None
                prior = values.loc[:date] if date is not None else values.iloc[:0]
                change = growth(prior) if date is not None else None
                row[f'{sym} period end'] = str(date)[:10] if date is not None else '—'
                row[sym] = money(v)
                row[f'{sym} YoY'] = pct(change)
            output.append(row)
    return pd.DataFrame(output)


def render_comparison(fetch_info, fetch_statements, fetch_base):
    st.subheader('Compare two stocks')
    st.caption('Start with the summary. Check annual profits and cash flow side by side below.')
    with st.form('compare_symbols'):
        a, b = st.columns(2)
        left = a.text_input('First ticker', value='PYPL').strip().upper()
        right = b.text_input('Second ticker', value='TDOC').strip().upper()
        st.form_submit_button('Compare', use_container_width=True)
    symbols = [left, right]
    if any(not s or len(s) > 20 or not all(c.isalnum() or c in '.-^=' for c in s) for s in symbols):
        st.info('Enter two valid ticker symbols, such as PYPL and TDOC.')
        return
    if left == right:
        st.info('Choose two different stocks.')
        return
    if st.button('Refresh financial data'):
        for fn in [fetch_info, fetch_statements, fetch_base]:
            if hasattr(fn, 'clear'):
                fn.clear()
        st.rerun()
    with st.spinner('Loading reported financial statements…'):
        infos = [fetch_info(s) for s in symbols]
        histories = [annual_rows(fetch_statements(s)) for s in symbols]
        bases = [fetch_base(s) for s in symbols]
    st.markdown('**' + preference(*histories, symbols) + '**')
    st.caption('This screen compares different businesses using four simple measures. It does not judge competitive advantage, future execution or investment suitability.')
    tabs = st.tabs(['Summary', 'P&L side by side', 'Cash flow side by side', 'Value & entry price'])
    with tabs[0]:
        columns = st.columns(2)
        for col, sym, info, rows in zip(columns, symbols, infos, histories):
            with col:
                st.markdown(f'### {sym}')
                st.caption(info.get('longName') or info.get('shortName') or sym)
                currency = info.get('currency') or 'Currency unavailable'
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                st.metric(f'Price ({currency})', f'{price:,.2f}' if number(price) else '—')
                view = business_view(rows)
                if view:
                    st.caption(f"Annual period ending {view['date']:%Y-%m-%d}")
                    st.dataframe(pd.DataFrame({'Measure': ['Revenue growth', 'Operating margin', 'Net margin', 'FCF margin'],
                                               'Result': [pct(v) for v in view['metrics']]}), hide_index=True, use_container_width=True)
                    ni, fcf = rows['Net income'].get(view['date']), rows['Free cash flow'].get(view['date'])
                    st.write('Profitable' if ni > 0 else 'Loss-making')
                    st.write('Generates spare cash' if fcf > 0 else 'Uses cash after investment')
                else:
                    st.info('Financial history is incomplete. No business ranking generated.')
    for tab, labels in [(tabs[1], ['Revenue','Gross profit','Operating profit','Net income']),
                        (tabs[2], ['Operating cash flow','Capital expenditure','Free cash flow'])]:
        with tab:
            st.caption('Amounts in millions of each company’s reported currency. Fiscal dates are shown explicitly; a dash means unavailable. YoY requires consecutive annual periods and a positive prior value.')
            frame = statement_table(histories, symbols, labels)
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button('Download table (CSV)', frame.to_csv(index=False).encode(),
                               file_name=f'{left}_{right}_{labels[0].replace(" ","_")}.csv', mime='text/csv', key='csv_'+labels[0])
            if labels[0] == 'Operating cash flow':
                st.caption('Capital expenditure is shown as spending (positive). Free cash flow = operating cash flow − capital expenditure. Profit can differ from cash because of noncash costs and working capital.')
    with tabs[3]:
        st.write('Use the same discount rate and terminal growth to compare scenarios.')
        x, y, z = st.columns(3)
        wacc = x.number_input('Discount rate (%)', min_value=.5, max_value=40., value=10., step=.5) / 100
        terminal = y.number_input('Long-run growth (%)', min_value=0., max_value=5., value=2.5, step=.5) / 100
        cushion = z.slider('Margin of safety (%)', 0, 50, 20, 5) / 100
        st.caption('Entry level = scenario value × (1 − margin of safety). This is a research threshold, not a buy order. Annual D&A and capex ratios, reported revenue growth and operating margin seed the model; tax is assumed at 18%, working capital at 2% of revenue change.')
        for col, sym, base in zip(st.columns(2), symbols, bases):
            with col:
                st.markdown(f'**{sym}**')
                if not base:
                    st.info('Valuation inputs unavailable.')
                    continue
                missing = _dcf_missing_inputs(base)
                if not base.get('financial_currency'):
                    if not st.checkbox(f"I verified {sym} statements are in {_quote_unit(base.get('currency'))[0] or 'the quote currency'}", key='cmp_currency_'+sym):
                        missing.append('verified financial currency')
                    else:
                        base = dict(base, financial_currency=_quote_unit(base.get('currency'))[0])
                if missing:
                    st.info('No price target. Missing: ' + ', '.join(missing))
                    continue
                assumptions = dict(DCF_DEFAULTS, da=base['da'], capex=base['capex'], wacc=wacc, termg=terminal)
                model = _dcf(base, assumptions)
                if model['error'] or model['ps'] <= 0:
                    st.info(model['error'] or 'No positive equity value in this scenario.')
                    continue
                st.metric(f"Scenario value ({base['currency']})", f"{model['ps']:,.2f}")
                st.metric('Research entry level', f"{model['ps'] * (1-cushion):,.2f}")
                st.caption(f"{model['tvShare']:.0%} of enterprise value comes after year five. Annual calibration: {base.get('calibration_period')}. Growth {base['g1']:.1%}; operating margin {base['margin']:.1%}. Confirm cash excludes customer funds and required operating balances.")
