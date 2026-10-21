"""Run the whole application's default view with deterministic financial providers."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class StartupTests(unittest.TestCase):
    def test_whole_app_starts_with_statement_provider(self):
        root=Path(__file__).parent
        source=f'''import sys, types, runpy
sys.path.insert(0,{str(root)!r})
from test_fundamentals import fixture
class Company:
    def __init__(self,symbol):
        self.info=dict(shortName=symbol, currentPrice=10.,sharesOutstanding=50.,currency='USD',financialCurrency='USD')
        self.fast_info=dict(last_price=10.,shares=50.,currency='USD')
        data=fixture(100 if symbol=='PYPL' else -100,130 if symbol=='PYPL' else -50)
        self.income_stmt=data['income']
        self.cashflow=data['cashflow']
        self.balance_sheet=data['balance']
        self.quarterly_balance_sheet=data['balance']
provider=types.ModuleType('yfinance')
provider.Ticker=Company
previous_provider=sys.modules.get('yfinance')
sys.modules['yfinance']=provider
try:
    runpy.run_path({str(root/'app.py')!r},run_name='__main__')
finally:
    if previous_provider is None:
        sys.modules.pop('yfinance',None)
    else:
        sys.modules['yfinance']=previous_provider
'''
        app=AppTest.from_string(source,default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('PYPL is the stronger' in m.value for m in app.markdown))
        self.assertEqual(app.radio[0].value,'⚖️ Compare Stocks')


if __name__=='__main__':
    unittest.main()
