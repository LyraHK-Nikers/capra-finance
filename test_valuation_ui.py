"""Streamlit UI regressions with deterministic provider fixtures, no network."""
from pathlib import Path
import ast
import unittest

from streamlit.testing.v1 import AppTest


class ValuationUITests(unittest.TestCase):
    def app(self, changes=''):
        # Load the actual function definitions, stopping before the global dashboard
        # starts fetching every watchlist ticker. Render the real valuation page.
        root = Path(__file__).parent
        tree = ast.parse(root.joinpath('app.py').read_text(encoding='utf-8'))
        names = {'_money', '_fmt_pct', '_color', '_sparkline_svg',
                 '_sensitivity_heatmap', 'render_valuation_body'}
        source = '\n'.join(ast.unparse(n) for n in tree.body
                           if isinstance(n, ast.FunctionDef) and n.name in names)
        source = '''import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import valuation
globals().update({k: v for k, v in vars(valuation).items() if not k.startswith('__')})
''' + source
        fixture = '''
BASE = dict(name='Test company', price=57.14, shares=862e6, rev=33e9,
            margin=.17, ni=4e9, debt=13.4e9, cash=11.256e9, g1=.04,
            currency='USD', financial_currency='USD', beta=1.1, da=.028,
            capex=.026, calibration_period='2025-12-31', pe=12,
            ev_ebitda=10, sector='Payments')
'''
        fixture += changes + '''
def fetch_valuation_base(ticker):
    return dict(BASE)
fetch_valuation_base.clear = lambda: None
fetch_history = lambda *args, **kwargs: pd.DataFrame()
render_valuation_body('TEST')
'''
        return AppTest.from_string('import sys\nsys.path.insert(0, '+repr(str(root))+')\n'+source+fixture,
                                   default_timeout=30).run()

    def test_complete_inputs_render_and_recalculate(self):
        app = self.app()
        self.assertFalse(app.exception)
        before = app.metric[1].value
        app.number_input(key='val_margin_TEST').set_value(12.).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(before, app.metric[1].value)
        self.assertFalse(any('Strong Buy' in m.value for m in app.markdown))

    def test_missing_inputs_suppress_target(self):
        app = self.app("BASE['cash'] = None\n")
        self.assertFalse(app.exception)
        self.assertTrue(any('Missing or incompatible' in m.value for m in app.warning))
        self.assertEqual(len(app.metric), 0)

    def test_unknown_statement_currency_requires_verification(self):
        app = self.app("BASE['financial_currency'] = None\n")
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        app.checkbox(key='val_currency_TEST').check().run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.metric), 0)

    def test_missing_beta_uses_explicit_manual_discount_rate(self):
        app = self.app("BASE['beta'] = None\n")
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key='val_use_capm_TEST').disabled)
        self.assertEqual(app.number_input(key='val_wacc_TEST').value, 9.)

    def test_invalid_terminal_spread_suppresses_target(self):
        app = self.app("BASE['beta'] = None\n")
        app.number_input(key='val_wacc_TEST').set_value(2.5).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('must exceed terminal' in m.value for m in app.warning))
        self.assertEqual(len(app.metric), 0)

    def test_loss_maker_does_not_get_perpetuity_target(self):
        app = self.app("BASE['margin'] = -.10\n")
        self.assertFalse(app.exception)
        self.assertTrue(any('recovery model' in m.value for m in app.warning))
        self.assertEqual(len(app.metric), 0)


if __name__ == '__main__':
    unittest.main()
