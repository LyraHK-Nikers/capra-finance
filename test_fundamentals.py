import unittest
import pandas as pd
from fundamentals import annual_rows, enrich_info, growth, series
from comparison import preference, statement_table


def fixture(profit=100, cash=130):
    dates = pd.to_datetime(['2025-12-31','2024-12-31','2023-12-31'])
    inc = pd.DataFrame([[1000,900,800],[500,450,400],[profit,profit*.9,profit*.8],
                        [profit,profit*.9,profit*.8]],
                       index=['Total Revenue','Gross Profit','Operating Income','Net Income'],columns=dates)
    cf = pd.DataFrame([[cash,cash*.9,cash*.8],[-30,-25,-20]],
                      index=['Operating Cash Flow','Capital Expenditure'],columns=dates)
    bs = pd.DataFrame([[200,220,240],[50,40,30],[400,380,350],[300,280,250],[100,90,80]],
                      index=['Total Debt','Cash And Cash Equivalents','Stockholders Equity','Current Assets','Current Liabilities'],columns=dates)
    return dict(income=inc,cashflow=cf,balance=bs)


class FundamentalsTests(unittest.TestCase):
    def test_summary_outage_still_produces_reported_metrics(self):
        info = enrich_info({'financialCurrency':'USD'},fixture(),{'currentPrice':10.,'sharesOutstanding':50.,'currency':'USD'})
        self.assertEqual(info['totalRevenue'],1000)
        self.assertEqual(info['freeCashflow'],100)
        self.assertEqual(info['currentRatio'],3)
        self.assertEqual(info['debtToEquity'],50)
        self.assertEqual(info['annualPE'],5)
        self.assertEqual(info['_flow_basis'],'annual')
    def test_partial_summary_does_not_mix_annual_cash_with_ttm_profit(self):
        info=enrich_info({'totalRevenue':2000,'netIncomeToCommon':250},fixture())
        self.assertEqual((info['totalRevenue'],info['netIncomeToCommon']),(1000,100))
    def test_reported_zero_profit_preserved(self):
        info=enrich_info({},fixture(profit=0))
        self.assertEqual(info['netIncomeToCommon'],0)
        self.assertIsNone(info.get('annualPE'))
    def test_negative_equity_not_labelled_low_debt(self):
        fin=fixture()
        fin['balance'].loc['Stockholders Equity']=-100
        self.assertIsNone(enrich_info({},fin).get('debtToEquity'))
    def test_currency_mismatch_does_not_generate_multiple(self):
        info=enrich_info({'financialCurrency':'USD'},fixture(),{'currency':'HKD','currentPrice':10.,'sharesOutstanding':50.})
        self.assertIsNone(info.get('annualPE'))
    def test_pence_market_cap_is_in_pounds(self):
        info=enrich_info({'financialCurrency':'GBP'},fixture(),{'currency':'GBp','currentPrice':1000.,'sharesOutstanding':50.})
        self.assertEqual(info['marketCap'],500.)
        self.assertEqual(info['annualPE'],5.)
    def test_stale_complete_year_not_ranked_over_latest_income(self):
        fin=fixture()
        fin['cashflow']=fin['cashflow'].drop(columns=pd.Timestamp('2025-12-31'))
        self.assertIsNone(__import__('fundamentals').business_view(annual_rows(fin)))
    def test_net_debt_fallback_never_mixes_periods(self):
        fin=fixture()
        fin['balance'].loc['Total Debt',pd.Timestamp('2025-12-31')]=float('nan')
        info=enrich_info({},fin)
        self.assertEqual(info['_balance_period'],'2024-12-31')
        self.assertEqual((info['totalDebt'],info['totalCash']),(220,40))
    def test_growth_matches_chosen_flow_period(self):
        fin=fixture()
        fin['cashflow']=fin['cashflow'].drop(columns=pd.Timestamp('2025-12-31'))
        info=enrich_info({},fin)
        self.assertEqual(info['_statement_period'],'2024-12-31')
        self.assertAlmostEqual(info['revenueGrowth'],900/800-1)
    def test_no_data_no_zero_fabrication(self):
        self.assertIsNone(enrich_info({},{}).get('freeCashflow'))
    def test_fcf_requires_matching_dates(self):
        fin=fixture()
        fin['cashflow'].loc['Capital Expenditure',pd.Timestamp('2025-12-31')]=float('nan')
        self.assertNotIn(pd.Timestamp('2025-12-31'),annual_rows(fin)['Free cash flow'].index)
    def test_aliases_fill_missing_periods(self):
        df=pd.DataFrame([[None,90],[100,None]],index=['Total Revenue','Operating Revenue'],columns=pd.to_datetime(['2025-12-31','2024-12-31']))
        self.assertEqual(series(df,'Total Revenue','Operating Revenue').tolist(),[90,100])
    def test_loss_to_profit_has_no_percentage_growth(self):
        self.assertIsNone(growth(pd.Series([-10,10],index=pd.to_datetime(['2024-12-31','2025-12-31']))))
    def test_missing_year_has_no_growth(self):
        self.assertIsNone(growth(pd.Series([10,20],index=pd.to_datetime(['2023-12-31','2025-12-31']))))
    def test_preference_needs_real_profit_and_cash(self):
        left,right=annual_rows(fixture()),annual_rows(fixture(-100,-50))
        self.assertIn('PYPL is the stronger',preference(left,right,['PYPL','TDOC']))
        self.assertIn('No preference',preference(annual_rows({}),right,['PYPL','TDOC']))
    def test_dates_retained_when_fiscal_years_differ(self):
        left=fixture();right=fixture()
        for frame in right.values():
            frame.columns=frame.columns-pd.Timedelta(days=90)
        table=statement_table([annual_rows(left),annual_rows(right)],['A','B'],['Revenue'])
        self.assertEqual(len(table),3)
        self.assertNotEqual(table.iloc[0]['A period end'],table.iloc[0]['B period end'])
    def test_preference_rejects_stale_comparison(self):
        left=fixture();right=fixture()
        for frame in right.values():
            frame.columns=frame.columns-pd.Timedelta(days=365)
        self.assertIn('too far apart',preference(annual_rows(left),annual_rows(right),['A','B']))


if __name__=='__main__':
    unittest.main()
