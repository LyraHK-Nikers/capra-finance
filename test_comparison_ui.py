from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class ComparisonUI(unittest.TestCase):
    def app(self,empty=False):
        root=str(Path(__file__).parent)
        source=f'''import sys
sys.path.insert(0, {root!r})
import streamlit as st
from comparison import render_comparison
from fundamentals import enrich_info
from test_fundamentals import fixture
def fin(symbol):
    return {{}} if {empty!r} else fixture(100 if symbol=='PYPL' else -100, 130 if symbol=='PYPL' else -50)
def info(symbol):
    return enrich_info({{}},fin(symbol),dict(currentPrice=10.,currency='USD',sharesOutstanding=50.))
def base(symbol):
    return None
render_comparison(info,fin,base)
'''
        return AppTest.from_string(source,default_timeout=30).run()
    def test_side_by_side_and_business_preference(self):
        app=self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any('PYPL is the stronger' in x.value for x in app.markdown))
        self.assertEqual([t.label for t in app.tabs],['Summary','P&L side by side','Cash flow side by side','Value & entry price'])
        self.assertTrue(any('PYPL YoY' in d.value.columns and 'TDOC YoY' in d.value.columns for d in app.dataframe))
    def test_empty_feed_renders_without_false_preference(self):
        app=self.app(empty=True)
        self.assertFalse(app.exception)
        self.assertTrue(any('No preference' in x.value for x in app.markdown))
    def test_same_symbol_does_not_rank_against_itself(self):
        app=self.app()
        app.text_input[1].set_value('PYPL')
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('two different' in x.value for x in app.info))


if __name__=='__main__':
    unittest.main()
