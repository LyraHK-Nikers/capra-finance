import ast
from pathlib import Path
import unittest
import pandas as pd
from portfolio import portfolio_rows


class PortfolioTests(unittest.TestCase):
    def test_currencies_are_never_added_together(self):
        holdings=[dict(ticker='A',qty=2.,cost=10.),dict(ticker='B',qty=3.,cost=10.)]
        rows, totals=portfolio_rows(holdings,{'A':dict(price=12.,previous=11.,currency='USD'),
                                            'B':dict(price=20.,previous=19.,currency='HKD')})
        self.assertEqual({t['Currency']:t['Value'] for t in totals},{'HKD':60.,'USD':24.})
    def test_missing_quote_suppresses_aggregate_pnl(self):
        h=[dict(ticker='A',qty=2.,cost=10.),dict(ticker='B',qty=3.,cost=10.)]
        rows, totals=portfolio_rows(h,{'A':dict(price=12.,previous=11.,currency='USD'),
                                      'B':dict(price=None,previous=None,currency='USD')})
        self.assertEqual(totals[0]['Cost'],50)
        self.assertIsNone(totals[0]['P&L'])
        self.assertEqual(totals[0]['Quote coverage'],'1/2')
    def test_pence_converted_to_pounds_for_valuation(self):
        rows, totals=portfolio_rows([dict(ticker='A',qty=10.,cost=100.)],
                                   {'A':dict(price=120.,previous=110.,currency='GBp')})
        self.assertEqual((totals[0]['Currency'],totals[0]['Value'],totals[0]['P&L']),('GBP',12.,2.))
        self.assertEqual(rows[0]['Current Price'],120.)
    def test_unknown_currency_excluded_from_totals(self):
        rows, totals=portfolio_rows([dict(ticker='A',qty=1.,cost=1.)],{'A':dict(price=2.)})
        self.assertEqual(totals,[])
        self.assertEqual(rows[0]['Value currency'],'Unknown')
    def test_missing_previous_close_is_not_zero_daily_change(self):
        rows, totals=portfolio_rows([dict(ticker='A',qty=1.,cost=1.)],{'A':dict(price=2.,currency='USD')})
        self.assertIsNone(totals[0]['Day Change'])
    def test_short_history_cannot_be_labelled_five_year_cagr(self):
        tree=ast.parse(Path(__file__).with_name('app.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cagr')
        scope={'pd':pd}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'app.py','exec'),scope)
        prices=pd.Series([10.,11.,12.],index=pd.to_datetime(['2025-01-01','2025-02-01','2025-03-01']))
        self.assertTrue(pd.isna(scope['_cagr'](prices,5)))


if __name__=='__main__':
    unittest.main()
