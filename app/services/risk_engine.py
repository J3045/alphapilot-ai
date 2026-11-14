from app.models.portfolio import Portfolio


def assess_portfolio_risk(
    portfolio: Portfolio,
    max_single_holding_pct: float = 40.0,
    min_cash_pct: float = 5.0,
) -> dict:
    total_value = portfolio.total_value

    if total_value <= 0:
        raise ValueError("Portfolio total value must be positive.")

    cash_pct = portfolio.cash / total_value * 100

    aggregated_values: dict[str, float] = {}
    display_symbols: dict[str, str] = {}
    for holding in portfolio.holdings:
        identity_key = holding.instrument_id or f"symbol:{holding.symbol}"
        aggregated_values[identity_key] = aggregated_values.get(identity_key, 0.0) + holding.market_value
        display_symbols.setdefault(identity_key, holding.symbol)

    concentrations = [
        {
            "symbol": display_symbols[identity_key],
            "weight_pct": market_value / total_value * 100,
        }
        for identity_key, market_value in aggregated_values.items()
    ]

    violations = []

    for holding in concentrations:
        if holding["weight_pct"] > max_single_holding_pct:
            violations.append(
                f"{holding['symbol']} exceeds the "
                f"{max_single_holding_pct:.1f}% single-holding limit."
            )

    if cash_pct < min_cash_pct:
        violations.append(
            f"Cash is below the {min_cash_pct:.1f}% minimum."
        )

    return {
        "portfolio_id": portfolio.portfolio_id,
        "total_value": round(total_value, 2),
        "cash_pct": round(cash_pct, 2),
        "concentrations": concentrations,
        "passed": len(violations) == 0,
        "violations": violations,
    }
