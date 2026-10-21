"""Offline regression checks. Run: python -m unittest test_valuation -v"""
import math
import unittest
from datetime import datetime

from valuation import (_annual_calibration, _capm_wacc, _dcf, _dcf_missing_inputs,
                       _dcf_price_gm, _dcf_price_w, _dcf_project, _health_verdict,
                       _implied_growth)

BASE = dict(rev=1000., margin=.2, g1=.05, debt=100., cash=40., shares=50.,
            price=30., beta=1.2, da=.03, capex=.04, currency='USD', financial_currency='USD')
ASSUMPTIONS = dict(tax=.2, wacc=.1, termg=.025, da=.03, capex=.04, nwc=.02, taper=.01)


class DCFTests(unittest.TestCase):
    def test_negative_growth_is_preserved(self):
        rows = _dcf_project(dict(BASE, g1=-.12), ASSUMPTIONS)
        self.assertAlmostEqual(rows[0]['g'], -.12)
        self.assertAlmostEqual(rows[0]['rev'], 880)
        self.assertAlmostEqual(rows[1]['g'], -.11)

    def test_terminal_rebuild_uses_stable_year_six_working_capital(self):
        a = dict(ASSUMPTIONS, taper=0., nwc=.2)
        d = _dcf(dict(BASE, g1=.20), a)
        last = d['pr'][-1]
        rev6 = last['rev'] * 1.025
        expected = rev6 * (.2 * .8 + .03 - .04) - (rev6-last['rev'])*.2
        self.assertAlmostEqual(d['terminal_fcf'], expected)
        self.assertAlmostEqual(d['pvtv'], expected/.075/1.1**5)
        self.assertNotAlmostEqual(expected, last['fcf']*1.025)

    def test_pence_quotes_match_pound_economic_value(self):
        pounds = dict(BASE, currency='GBP', financial_currency='GBP')
        pence = dict(pounds, currency='GBp', price=BASE['price']*100)
        self.assertEqual(_dcf_missing_inputs(pence), [])
        self.assertAlmostEqual(_dcf(pence, ASSUMPTIONS)['ps'], _dcf(pounds, ASSUMPTIONS)['ps']*100)
        self.assertAlmostEqual(_dcf(pence, ASSUMPTIONS)['upside'], _dcf(pounds, ASSUMPTIONS)['upside'])
        self.assertAlmostEqual(_capm_wacc(pounds,.04,.05,.2), _capm_wacc(pence,.04,.05,.2))

    def test_arithmetic_api_rejects_currency_mismatch(self):
        self.assertTrue(_dcf(dict(BASE, financial_currency='EUR'), ASSUMPTIONS)['error'])

    def test_extreme_growth_does_not_render_infinite_target(self):
        self.assertTrue(_dcf(dict(BASE, g1=1e308), ASSUMPTIONS)['error'])

    def test_growth_tapers_from_either_direction(self):
        self.assertEqual([round(r['g'], 3) for r in _dcf_project(BASE, ASSUMPTIONS)],
                         [.05, .04, .03, .025, .025])

    def test_no_assumed_tax_refund_for_losses(self):
        row = _dcf_project(dict(BASE, margin=-.1), ASSUMPTIONS)[0]
        expected = row['ebit'] + row['da'] - row['capex'] - row['nwc']
        self.assertAlmostEqual(row['fcf'], expected)
        self.assertEqual(row['nopat'], row['ebit'])

    def test_sensitivity_center_matches_main_model(self):
        result = _dcf(BASE, ASSUMPTIONS)['ps']
        self.assertAlmostEqual(result, _dcf_price_gm(BASE, ASSUMPTIONS, BASE['g1'], BASE['margin']))
        self.assertAlmostEqual(result, _dcf_price_w(BASE, ASSUMPTIONS, .1, .025))

    def test_terminal_growth_sensitivity_rebuilds_forecast(self):
        result = _dcf(BASE, dict(ASSUMPTIONS, termg=.01))['ps']
        self.assertAlmostEqual(result, _dcf_price_w(BASE, ASSUMPTIONS, .1, .01))

    def test_higher_discount_rate_lowers_value(self):
        self.assertGreater(_dcf(BASE, ASSUMPTIONS)['ps'], _dcf_price_w(BASE, ASSUMPTIONS, .12, .025))

    def test_value_bridge_and_cash_sign(self):
        d = _dcf(BASE, ASSUMPTIONS)
        self.assertAlmostEqual(d['eq'], d['ev'] - BASE['debt'] + BASE['cash'])
        self.assertAlmostEqual(d['ps'] * BASE['shares'], d['eq'])
        cash = _dcf(dict(BASE, debt=0, cash=100), ASSUMPTIONS)
        self.assertAlmostEqual(cash['eq'], cash['ev'] + 100)

    def test_invalid_spread_suppresses_target(self):
        d = _dcf(BASE, dict(ASSUMPTIONS, wacc=.025))
        self.assertTrue(d['error'])
        self.assertTrue(math.isnan(d['ps']))

    def test_negative_terminal_cashflow_requires_recovery_model(self):
        d = _dcf(dict(BASE, margin=-.2), ASSUMPTIONS)
        self.assertIn('recovery model', d['error'])
        self.assertTrue(math.isnan(d['ps']))

    def test_nonfinite_and_zero_shares_do_not_produce_target(self):
        for value in [0, None, float('nan'), float('inf')]:
            self.assertTrue(_dcf(dict(BASE, shares=value), ASSUMPTIONS)['error'])

    def test_reverse_dcf_recovers_main_model_initial_growth(self):
        market = dict(BASE, price=_dcf(BASE, ASSUMPTIONS)['ps'])
        g, edge = _implied_growth(market, ASSUMPTIONS)
        self.assertEqual(edge, 'exact')
        self.assertAlmostEqual(g, BASE['g1'])

    def test_beta_zero_is_not_replaced_by_one(self):
        self.assertLess(_capm_wacc(dict(BASE, beta=0), .04, .05, .2),
                        _capm_wacc(dict(BASE, beta=1), .04, .05, .2))

    def test_missing_beta_disables_capm(self):
        self.assertTrue(math.isnan(_capm_wacc(dict(BASE, beta=None), .04, .05, .2)))

    def test_missing_financials_not_assumed_zero(self):
        for field in ['cash', 'debt', 'da', 'capex', 'margin', 'g1']:
            self.assertIn(field, _dcf_missing_inputs(dict(BASE, **{field: None})))

    def test_genuine_zero_inputs_are_valid(self):
        self.assertEqual(_dcf_missing_inputs(dict(BASE, cash=0, debt=0, da=0, capex=0, margin=0, g1=0)), [])

    def test_currency_mismatch_blocks_valuation(self):
        self.assertTrue(_dcf_missing_inputs(dict(BASE, financial_currency='EUR')))

    def test_empty_health_coverage_not_healthy(self):
        self.assertIn('insufficient', _health_verdict('PYPL', dict.fromkeys(['p','b','c','v','q'], 'na')))
        self.assertIn('insufficient', _health_verdict('PYPL', {'p':'na','b':'na','c':'na','v':'na','q':'okay'}))

    def test_available_health_findings(self):
        self.assertIn('strong', _health_verdict('X', dict.fromkeys(range(5), 'good')))
        self.assertIn('warning', _health_verdict('X', dict.fromkeys(range(5), 'bad')))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.pd = pd
        self.dates = pd.to_datetime(['2025-12-31', '2024-12-31'])
        self.inc = pd.DataFrame([[1000,900],[200,150]], index=['Total Revenue','Operating Income'], columns=self.dates)
        self.cf = pd.DataFrame([[30,25],[-40,-35]], index=['Depreciation And Amortization','Capital Expenditure'], columns=self.dates)

    def test_matches_period_and_capex_sign(self):
        result = _annual_calibration(self.inc, self.cf)
        self.assertAlmostEqual(result['da'], .03)
        self.assertAlmostEqual(result['capex'], .04)
        self.assertAlmostEqual(result['growth'], 1000/900-1)
        self.assertEqual(result['calibration_period'], '2025-12-31')

    def test_nonmatching_periods_do_not_mix(self):
        self.cf.columns = self.pd.to_datetime(['2023-12-31','2022-12-31'])
        self.assertEqual(_annual_calibration(self.inc, self.cf), {})

    def test_zero_capex_and_da_are_valid(self):
        self.cf.loc[:, :] = 0
        result = _annual_calibration(self.inc, self.cf)
        self.assertEqual(result['da'], 0)
        self.assertEqual(result['capex'], 0)

    def test_missing_year_not_called_yoy_growth(self):
        self.inc.columns = self.pd.to_datetime(['2025-12-31','2023-12-31'])
        self.assertNotIn('growth', _annual_calibration(self.inc, self.cf))


if __name__ == '__main__':
    unittest.main()
