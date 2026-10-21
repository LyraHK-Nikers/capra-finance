import unittest
import numpy as np
import pandas as pd
from stock_ideas import (evaluate,profile_reasons,shortlist,entry_range,currency_from_payload,PROFILES)
from test_fundamentals import fixture
from test_valuation import BASE


def snapshot(symbol='PYPL'):
    qdates=pd.to_datetime(['2026-06-30','2026-03-31','2025-12-31','2025-09-30','2025-06-30'])
    quarterly=pd.DataFrame([[270,260,250,245,240],[25,24,23,22,20]],
                           index=['Total Revenue','Net Income'],columns=qdates)
    return dict(symbol=symbol,info=dict(currency='USD',financialCurrency='USD',sharesOutstanding=50.,totalDebt=200.,totalCash=50.),
                financials=fixture(),quarterly_income=quarterly,base=BASE,balance_date='2026-06-30')


def prices():
    return pd.Series(np.linspace(20,10,1300),index=pd.bdate_range(end='2026-10-20',periods=1300))


class IdeaTests(unittest.TestCase):
    def fact(self):
        return evaluate(snapshot(),prices(),asof='2026-10-21')
    def test_complete_company_qualifies_in_each_profile(self):
        f=self.fact()
        self.assertTrue(f['eligible'])
        for profile in PROFILES:
            self.assertEqual(profile_reasons(f,profile),[])
    def test_empty_provider_never_generates_idea(self):
        f=evaluate(dict(symbol='X'),None,asof='2026-10-21')
        for profile in PROFILES:
            self.assertTrue(profile_reasons(f,profile))
    def test_currency_must_be_verified(self):
        s=snapshot();s['info'].pop('financialCurrency')
        f=evaluate(s,prices(),asof='2026-10-21')
        self.assertFalse(f['eligible'])
        s['statement_currency']='USD'
        self.assertTrue(evaluate(s,prices(),asof='2026-10-21')['eligible'])
    def test_quarterly_decline_excludes_balanced_and_conservative(self):
        f=dict(self.fact(),qgrowth=-.30)
        self.assertTrue(profile_reasons(f,'Balanced'))
        self.assertTrue(profile_reasons(f,'Conservative'))
    def test_drawdown_alone_does_not_qualify(self):
        f=dict(self.fact(),improving=False,drawdown=.90)
        self.assertTrue(profile_reasons(f,'Higher-risk Recovery'))
    def test_large_milestone_revenue_drop_needs_manual_review(self):
        f=dict(self.fact(),qgrowth=-.76)
        self.assertTrue(any('50%' in r for r in profile_reasons(f,'Higher-risk Recovery')))
    def test_recovery_cash_burn_requires_liquidity(self):
        f=dict(self.fact(),fcf=-100.,leverage=None,cash_years=.5)
        self.assertTrue(profile_reasons(f,'Higher-risk Recovery'))
        f['cash_years']=2.5
        self.assertEqual(profile_reasons(f,'Higher-risk Recovery'),[])
    def test_loss_companies_not_conservative(self):
        self.assertTrue(profile_reasons(dict(self.fact(),profit=-1.),'Conservative'))
    def test_conservative_requires_three_profitable_years(self):
        self.assertTrue(profile_reasons(dict(self.fact(),stable_profit=False),'Conservative'))
    def test_no_fabricated_slots(self):
        self.assertEqual(len(shortlist([self.fact()],'Balanced')),1)
    def test_top_five_are_ranked_not_first_five(self):
        facts=[dict(self.fact(),symbol=str(i),score=float(i)) for i in range(8)]
        self.assertEqual([f['symbol'] for f in shortlist(facts,'Balanced')],['7','6','5','4','3'])
    def test_entry_price_range_and_loss_recovery_suppression(self):
        zone,note=entry_range(self.fact())
        self.assertGreater(zone[0],0)
        self.assertLessEqual(zone[0],zone[1])
        self.assertIsNone(entry_range(dict(self.fact(),opmargin=-.1))[0])
    def test_stale_quote_excluded(self):
        f=evaluate(snapshot(),prices().iloc[:-20],asof='2026-10-21')
        self.assertFalse(f['eligible'])
    def test_stale_quarter_excluded(self):
        s=snapshot();s['quarterly_income'].columns-=pd.Timedelta(days=365)
        self.assertFalse(evaluate(s,prices(),asof='2026-10-21')['eligible'])
    def test_future_dated_accounts_excluded(self):
        s=snapshot()
        for frame in s['financials'].values():
            frame.columns+=pd.Timedelta(days=730)
        self.assertFalse(evaluate(s,prices(),asof='2026-10-21')['eligible'])
    def test_currency_payload_parser(self):
        p={'timeseries':{'result':[{'annualTotalRevenue':[{'asOfDate':'2025-12-31','currencyCode':'USD','reportedValue':{'raw':100}}]}]}}
        self.assertEqual(currency_from_payload(p),'USD')
        self.assertIsNone(currency_from_payload({}))


if __name__=='__main__':
    unittest.main()
