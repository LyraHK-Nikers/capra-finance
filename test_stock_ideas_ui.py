from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class IdeasUI(unittest.TestCase):
    def test_three_profiles_and_five_rows_per_profile(self):
        root=Path(__file__).parent
        code=f'''import sys
sys.path.insert(0,{str(root)!r})
import pandas as pd
import stock_ideas
from test_stock_ideas import snapshot,prices
stock_ideas.load_snapshots.clear()
original_universe=stock_ideas.UNIVERSE
original_evaluate=stock_ideas.evaluate
stock_ideas.UNIVERSE=dict(list(stock_ideas.UNIVERSE.items())[:6])
stock_ideas.evaluate=lambda s,h: original_evaluate(s,h,asof='2026-10-21')
def fetch(s): return snapshot(s)
def history(symbols,period): return pd.DataFrame({{s:prices()*.5+5 for s in symbols}})
try:
    stock_ideas.render_stock_ideas(fetch,history)
finally:
    stock_ideas.UNIVERSE=original_universe
    stock_ideas.evaluate=original_evaluate
    stock_ideas.load_snapshots.clear()
'''
        app=AppTest.from_string(code,default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs],['Balanced','Conservative','Higher-risk Recovery'])
        self.assertTrue(all(len(d.value)==5 for d in app.dataframe[:3]))


if __name__=='__main__':
    unittest.main()
