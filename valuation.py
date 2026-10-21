"""DCF arithmetic shared by the main valuation, sensitivities and reverse DCF.

No network or UI imports. Missing financial data must never become invented profit.
"""
import math
from numbers import Real

DCF_DEFAULTS = {"tax": 0.18, "wacc": 0.09, "termg": 0.025,
                "nwc": 0.02, "taper": 0.02}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _first_finite(*values):
    return next((float(v) for v in values if _finite(v)), None)


def _annual_calibration(income, cashflow):
    """Use one matching annual period for operating margin, D&A and capex ratios."""
    if income is None or cashflow is None or income.empty or cashflow.empty:
        return {}

    def value(frame, col, names):
        for name in names:
            if name in frame.index:
                candidate = frame.loc[name, col]
                if _finite(candidate):
                    return float(candidate)
        return None

    for col in sorted(set(income.columns) & set(cashflow.columns), reverse=True):
        rev = value(income, col, ["Total Revenue", "Operating Revenue"])
        op = value(income, col, ["Operating Income", "Total Operating Income As Reported"])
        da = value(cashflow, col, ["Depreciation Depletion And Amortization",
                                  "Depreciation And Amortization", "Depreciation Amortization Depletion"])
        capex = value(cashflow, col, ["Capital Expenditure", "Purchase Of PPE"])
        if rev is None or rev <= 0 or op is None or da is None or da < 0 or capex is None:
            continue
        out = {"margin": op / rev, "da": da / rev, "capex": abs(capex) / rev,
               "calibration_period": str(col)[:10], "annual_revenue": rev}
        # Annual growth only from consecutive fiscal years, never across a missing year.
        previous = [c for c in income.columns if 330 <= (col - c).days <= 400]
        if previous:
            prior = value(income, max(previous), ["Total Revenue", "Operating Revenue"])
            if prior is not None and prior > 0:
                out["growth"] = rev / prior - 1
        return out
    return {}


def _dcf_missing_inputs(base):
    missing = []
    for key in ["price", "shares", "rev", "margin", "g1", "debt", "cash", "da", "capex"]:
        val = base.get(key)
        if not _finite(val) or (key in ["price", "shares", "rev"] and val <= 0):
            missing.append(key)
        elif key in ["debt", "cash", "da", "capex"] and val < 0:
            missing.append(key)
    if not base.get("currency"):
        missing.append("quote currency")
    if (base.get("financial_currency") and base.get("currency") and
            base["financial_currency"] != base["currency"]):
        missing.append("matching quote / financial currencies (FX conversion required)")
    return missing


def _dcf_project(base, a):
    rows, prev = [], base["rev"]
    for t in range(1, 6):
        # Approach the terminal rate from either side. Never floor declining sales
        # at a positive perpetual-growth assumption in year one.
        gap = a["termg"] - base["g1"]
        g = base["g1"] + math.copysign(min(abs(gap), (t - 1) * a["taper"]), gap)
        rev = prev * (1 + g)
        ebit = rev * base["margin"]
        nopat = ebit - max(ebit, 0) * a["tax"]  # no assumed tax refund on losses
        da, capex = rev * a["da"], rev * a["capex"]
        nwc = (rev - prev) * a["nwc"]
        rows.append(dict(t=t, g=g, rev=rev, ebit=ebit, nopat=nopat, da=da, capex=capex,
                         nwc=nwc, fcf=nopat + da - capex - nwc))
        prev = rev
    return rows


def _dcf(base, a):
    nan = float("nan")
    out = dict(pr=[], sumpv=nan, pvtv=nan, ev=nan, netDebt=nan, eq=nan,
               ps=nan, upside=nan, tvShare=nan, error="")
    if any(not _finite(base.get(k)) for k in ["rev", "margin", "g1", "debt", "cash", "shares", "price"]):
        out["error"] = "Missing or non-finite financial input."
        return out
    if any(not _finite(a.get(k)) for k in ["tax", "wacc", "termg", "da", "capex", "nwc", "taper"]):
        out["error"] = "Missing or non-finite forecast assumption."
        return out
    if (min(base["rev"], base["shares"], base["price"]) <= 0 or min(base["debt"], base["cash"]) < 0
            or not 0 <= a["tax"] <= 1 or min(a["da"], a["capex"], a["taper"]) < 0
            or min(base["g1"], a["termg"]) <= -1 or a["wacc"] <= 0):
        out["error"] = "Invalid financial input or forecast assumption."
        return out
    if a["wacc"] - a["termg"] < 0.005 - 1e-12:
        out["error"] = "Discount rate must exceed terminal growth by at least 0.5 percentage points."
        return out
    pr = _dcf_project(base, a)
    out["pr"] = pr
    if pr[-1]["fcf"] <= 0:
        out["error"] = "Terminal cash flow is not positive. A recovery model is required before using perpetual growth."
        return out
    sumpv = sum(r["fcf"] / (1 + a["wacc"]) ** r["t"] for r in pr)
    pvtv = pr[-1]["fcf"] * (1 + a["termg"]) / (a["wacc"] - a["termg"]) / (1 + a["wacc"]) ** 5
    ev = sumpv + pvtv
    nd = base["debt"] - base["cash"]
    eq = ev - nd
    ps = eq / base["shares"]
    out.update(sumpv=sumpv, pvtv=pvtv, ev=ev, netDebt=nd, eq=eq, ps=ps,
               upside=ps / base["price"] - 1, tvShare=pvtv / ev if ev else nan)
    return out


def _dcf_price_gm(base, a, g, m):
    return _dcf(dict(base, g1=g, margin=m), a)["ps"]


def _dcf_price_w(base, a, w, tg):
    return _dcf(base, dict(a, wacc=w, termg=tg))["ps"]


def _capm_wacc(base, rf, erp, tax):
    beta = base.get("beta")
    if not all(_finite(x) for x in [beta, rf, erp, tax, base.get("price"), base.get("shares"), base.get("debt")]):
        return float("nan")
    ke = rf + beta * erp
    e, d = base["price"] * base["shares"], max(0.0, base["debt"])
    if e <= 0 or not 0 <= tax <= 1:
        return float("nan")
    # Credit spread is an explicit approximation, not a quoted borrowing yield.
    return (e * ke + d * (rf + 0.015) * (1 - tax)) / (e + d)


def _implied_growth(base, a, lo=-0.5, hi=1.0):
    """Solve initial growth using the SAME tapered forecast as the headline DCF."""
    def price(g):
        return _dcf_price_gm(base, a, g, base["margin"])
    points = [price(lo + (hi - lo) * i / 20) for i in range(21)]
    if any(not _finite(p) for p in points) or any(y <= x for x, y in zip(points, points[1:])):
        return float("nan"), "unavailable"
    if points[0] > base["price"]:
        return lo, "below"
    if points[-1] < base["price"]:
        return hi, "above"
    for _ in range(60):
        mid = (lo + hi) / 2
        if price(mid) > base["price"]:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, "exact"


def _health_verdict(name, ratings):
    greens = sum(v == "good" for v in ratings.values())
    reds = sum(v == "bad" for v in ratings.values())
    covered = sum(v != "na" for v in ratings.values())
    if covered < 4:
        return f"{name}: insufficient data to assess overall financial health."
    if greens >= 4 and reds == 0:
        return f"{name} looks financially strong on the available measures."
    if reds >= 3:
        return f"{name} shows several warning signs."
    if greens >= reds and reds <= 1:
        return f"{name} looks reasonably healthy, with areas to review."
    return f"{name} has a mixed financial picture."
