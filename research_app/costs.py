"""Aggregate execution-cost evidence; no reconstructed daily or Sharpe estimates."""

import math

DEFAULT_COST_BPS = (5.0, 10.0, 20.0)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def evidence(metrics, scope, cost_bps=None):
    """Interpret only the supplied period's metrics, in dollars per traded dollar.

    Margin is the platform raw ratio; one basis point is 0.0001. Turnover is
    the reported traded-value/book-value ratio, not a daily traded-dollar path.
    Costs are one-way bps per dollar traded, so aggregate net margin is margin
    minus cost. No assumed turnover frequency or capital is used.
    """
    scenarios = DEFAULT_COST_BPS if cost_bps is None else cost_bps
    if isinstance(scenarios, (int, float)):
        scenarios = [scenarios]
    values = []
    for value in scenarios:
        number = _number(value)
        if number is None or number < 0:
            raise ValueError("cost bps must be finite non-negative numbers")
        if number not in values:
            values.append(number)
    metrics = metrics or {}
    margin = _number(metrics.get("margin"))
    turnover = _number(metrics.get("turnover"))
    if turnover is not None and turnover < 0:
        turnover = None
    margin_bps = None if margin is None else _number(margin * 10000.0)
    estimates = []
    for value in values:
        net = None if margin_bps is None else _number(margin_bps - value)
        estimates.append({"cost_bps": value, "net_margin_bps": net,
                          "state": "known" if net is not None else "unknown_margin"})
    return {
        "scope": scope,
        "period": {"start": metrics.get("startDate"), "end": metrics.get("endDate")},
        "raw_margin": margin,
        "margin_bps": margin_bps,
        "margin_unit": "dollars_profit_per_dollar_traded",
        "margin_bps_unit": "basis_points_per_dollar_traded",
        "turnover": turnover,
        "turnover_unit": "traded_value_over_book_value",
        "turnover_period": metrics.get("turnoverPeriod") or "unknown",
        "state": "known" if margin_bps is not None else "unknown_margin",
        "cost_basis": "one_way_per_dollar_traded",
        "scenarios": estimates,
        "net_sharpe": None,
        "daily_costs": None,
        "limitations": "Aggregate metrics do not identify net Sharpe or daily traded dollars/costs.",
    }
