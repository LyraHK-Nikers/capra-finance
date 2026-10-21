"""Provider-failure regressions without Yahoo requests or running the app UI."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd
import numpy as np
import valuation


class ProviderTests(unittest.TestCase):
    def load(self, info, income=None, cashflow=None, balance=None):
        ticker = SimpleNamespace(info=info, fast_info={}, income_stmt=income,
                                 cashflow=cashflow, quarterly_balance_sheet=balance,
                                 balance_sheet=balance)
        tree = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fetch_valuation_base')
        fn.decorator_list = []
        scope = dict(vars(valuation), yf=SimpleNamespace(Ticker=lambda _: ticker),
                     fetch_history=lambda *args, **kwargs: pd.DataFrame())
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), scope)
        return scope['fetch_valuation_base']('TEST')

    def fixture(self):
        dates = pd.to_datetime(['2025-12-31', '2024-12-31'])
        inc = pd.DataFrame([[1000,900],[170,150]],
                           index=['Total Revenue','Operating Income'], columns=dates)
        cf = pd.DataFrame([[30,25],[-25,-20]],
                          index=['Depreciation And Amortization','Capital Expenditure'], columns=dates)
        return inc, cf

    def test_empty_provider_never_fabricates_financials(self):
        base = self.load({})
        self.assertTrue(valuation._dcf_missing_inputs(base))
        self.assertIsNone(base['margin'])
        self.assertIsNone(base['g1'])
        self.assertIsNone(base['cash'])
        self.assertIsNone(base['debt'])
        self.assertTrue(pd.isna(base['ni']))

    def test_reported_zero_financials_are_preserved(self):
        inc, cf = self.fixture()
        base = self.load(dict(currentPrice=10., sharesOutstanding=50., totalRevenue=1000.,
                              operatingMargins=0., revenueGrowth=0., totalDebt=0., totalCash=0.,
                              netIncomeToCommon=0., currency='USD', financialCurrency='USD'), inc, cf)
        self.assertEqual(valuation._dcf_missing_inputs(base), [])
        for field in ['margin','g1','debt','cash','ni']:
            self.assertEqual(base[field], 0)

    def test_annual_ratios_replace_generic_defaults(self):
        inc, cf = self.fixture()
        base = self.load(dict(currentPrice=10., sharesOutstanding=50., totalRevenue=1000.,
                              operatingMargins=.17, revenueGrowth=.03, totalDebt=100., totalCash=50.,
                              currency='USD', financialCurrency='USD'), inc, cf)
        self.assertAlmostEqual(base['da'], .03)
        self.assertAlmostEqual(base['capex'], .025)
        self.assertTrue(pd.isna(base['ni']))  # no synthetic P/E from operating margin

    def test_missing_cashflow_blocks_valuation(self):
        inc, _ = self.fixture()
        base = self.load(dict(currentPrice=10., sharesOutstanding=50., totalRevenue=1000.,
                              operatingMargins=.17, revenueGrowth=.03, totalDebt=100., totalCash=50.,
                              currency='USD'), inc)
        self.assertIn('da', valuation._dcf_missing_inputs(base))
        self.assertIn('capex', valuation._dcf_missing_inputs(base))

    def test_balance_fallback_uses_a_common_date(self):
        bs = pd.DataFrame([[100,80],[25,15]], index=['Total Debt','Cash And Cash Equivalents'],
                          columns=pd.to_datetime(['2026-06-30','2025-12-31']))
        base = self.load(dict(totalDebt=99), balance=bs)
        self.assertEqual((base['debt'],base['cash']), (100,25))

    def metrics(self, info):
        tree = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_metrics')
        scope = dict(vars(valuation), pd=pd, np=np, StockMetrics=SimpleNamespace,
                     fetch_info=lambda _: info, _max_drawdown=lambda _: -.1,
                     _beta=lambda *args: 1., _rsi=lambda _: 50.,
                     _forecast=lambda series, **kwargs: (series.iloc[-1]*1.1, None),
                     _cagr=lambda *args: .1, _risk_level=lambda *args: 'Low',
                     _cap_category=lambda _: 'Large', _best_buy_score=lambda _: (80., 'Strong Buy'))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), scope)
        series = pd.Series(np.linspace(10,20,400), index=pd.date_range('2025-01-01', periods=400))
        return scope['compute_metrics']('TEST', 'TEST', series, series.pct_change().dropna(), None)

    def test_price_signals_cannot_generate_buy_with_missing_financials(self):
        self.assertEqual(self.metrics({}).recommendation, 'Insufficient data')

    def test_reported_zero_profit_not_lost_on_dashboard(self):
        info = dict(totalRevenue=1000., operatingCashflow=10., netIncomeToCommon=0., quarterlyRevenueGrowth=0.)
        result = self.metrics(info)
        self.assertEqual(result.net_income, 0)
        self.assertEqual(result.sales_growth_qoq, 0)

    def test_earnings_growth_is_not_sales_growth(self):
        self.assertIsNone(self.metrics({'earningsQuarterlyGrowth': .2}).sales_growth_qoq)

    def test_small_yahoo_debt_equity_percentage_is_scaled(self):
        tree = ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fetch_analyzer_data')
        fn.decorator_list = []
        scope = dict(vars(valuation), pd=pd, fetch_info=lambda _: {'currentPrice':10., 'debtToEquity':4.},
                     fetch_financial_statements=lambda _: {}, fetch_history=lambda *a, **k: pd.DataFrame())
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'app.py', 'exec'), scope)
        self.assertAlmostEqual(scope['fetch_analyzer_data']('TEST')['debt_to_equity'], .04)


if __name__ == '__main__':
    unittest.main()
