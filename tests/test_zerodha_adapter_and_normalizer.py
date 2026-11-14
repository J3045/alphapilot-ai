"""Unit tests for Day 17–18 Zerodha Kite Connect v3 adapter and portfolio normalizer."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from app.integrations.brokers.exceptions import (
    BrokerAuthenticationError,
    BrokerPermissionError,
    BrokerRateLimitError,
    BrokerResponseValidationError,
    BrokerTimeoutError,
    BrokerTokenExpiredError,
    BrokerUnavailableError,
)
from app.integrations.brokers.models import (
    BrokerBalance,
    BrokerHoldingItem,
    BrokerProfile,
)
from app.integrations.brokers.zerodha import (
    ZerodhaBrokerAdapter,
    compute_kite_checksum,
    compute_kite_session_expiry,
)
from app.integrations.normalizers.portfolio_normalizer import ZerodhaPortfolioNormalizer


def _step_no_io(coro: Any) -> Any:
    """Drive purely in-memory coroutines without creating OS event loops."""
    try:
        while True:
            result = coro.send(None)
            if result is not None:
                raise RuntimeError("Coroutine yielded unexpected async wait primitive in sync harness.")
    except StopIteration as stop:
        return stop.value


def test_kite_checksum_and_daily_session_expiry_calculation():
    expected = hashlib.sha256(b"k123r456s789").hexdigest()
    assert compute_kite_checksum("k123", "r456", "s789") == expected

    # Before 07:30 IST (02:00 UTC) -> expires today 02:00 UTC (07:30 IST)
    early_utc = datetime(2026, 11, 12, 1, 0, tzinfo=timezone.utc)
    exp_early = compute_kite_session_expiry(early_utc)
    assert exp_early == datetime(2026, 11, 12, 2, 0, tzinfo=timezone.utc)

    # After 07:30 IST (e.g. 05:00 UTC = 10:30 IST) -> expires tomorrow 02:00 UTC
    midday_utc = datetime(2026, 11, 12, 5, 0, tzinfo=timezone.utc)
    exp_next = compute_kite_session_expiry(midday_utc)
    assert exp_next == datetime(2026, 11, 13, 2, 0, tzinfo=timezone.utc)


def test_zerodha_adapter_session_exchange_and_data_endpoints_with_mock_transport():
    captured_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "headers": dict(request.headers),
                "body": request.content.decode("utf-8") if request.content else "",
                "query": dict(request.url.params),
            }
        )
        if request.method == "POST" and request.url.path == "/session/token":
            form = parse_qs(request.content.decode("utf-8"))
            expected_sum = compute_kite_checksum("my_api_key", "req_tok_1", "my_api_secret")
            assert form["api_key"] == ["my_api_key"]
            assert form["request_token"] == ["req_tok_1"]
            assert form["checksum"] == [expected_sum]
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "user_id": "AB1234",
                        "user_name": "Test Investor",
                        "email": "investor@example.com",
                        "access_token": "kite_access_xyz",
                        "exchanges": ["NSE", "BSE"],
                        "products": ["CNC", "MIS"],
                        "login_time": "2026-11-12 09:15:00",
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/user/profile":
            assert request.headers["authorization"] == "token my_api_key:kite_access_xyz"
            assert request.headers["x-kite-version"] == "3"
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "user_id": "AB1234",
                        "user_name": "Test Investor",
                        "email": "investor@example.com",
                        "user_type": "individual",
                        "exchanges": ["NSE"],
                        "products": ["CNC"],
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/portfolio/holdings":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": [
                        {
                            "tradingsymbol": "INFY",
                            "exchange": "NSE",
                            "instrument_token": 408065,
                            "isin": "INE009A01021",
                            "product": "CNC",
                            "quantity": 10,
                            "t1_quantity": 2,
                            "average_price": 1500.0,
                            "last_price": 1820.5,
                            "pnl": 3846.0,
                        }
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/portfolio/positions":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "net": [
                            {
                                "tradingsymbol": "TCS",
                                "exchange": "NSE",
                                "instrument_token": 2953217,
                                "product": "MIS",
                                "quantity": 1,
                                "average_price": 4100.0,
                                "last_price": 4125.0,
                                "pnl": 25.0,
                            }
                        ],
                        "day": [],
                    },
                },
            )
        if request.method == "GET" and request.url.path == "/user/margins":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "equity": {
                            "enabled": True,
                            "net": 25500.75,
                            "available": {"cash": 25000.0, "live_balance": 25500.75, "collateral": 0},
                            "utilised": {"debits": 100.0},
                        }
                    },
                },
            )
        if request.method == "DELETE" and request.url.path == "/session/token":
            assert request.url.params["api_key"] == "my_api_key"
            assert request.url.params["access_token"] == "kite_access_xyz"
            return httpx.Response(200, json={"status": "success", "data": True})
        raise AssertionError(f"Unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    adapter = ZerodhaBrokerAdapter(
        api_key="my_api_key",
        api_secret="my_api_secret",
        transport=transport,
        max_retries=0,
    )

    session_token = _step_no_io(adapter.exchange_request_token("req_tok_1"))
    assert session_token.broker_account_id == "AB1234"
    assert session_token.access_token.get_secret_value() == "kite_access_xyz"

    profile = _step_no_io(adapter.get_profile("kite_access_xyz"))
    assert profile.broker_account_id == "AB1234"
    assert profile.user_name == "Test Investor"

    holdings = _step_no_io(adapter.get_holdings("kite_access_xyz"))
    assert len(holdings) == 1
    assert holdings[0].symbol == "INFY"
    assert holdings[0].quantity == Decimal("10")
    assert holdings[0].t1_quantity == Decimal("2")
    assert holdings[0].last_price == Decimal("1820.5")

    positions = _step_no_io(adapter.get_positions("kite_access_xyz"))
    assert len(positions) == 1
    assert positions[0].symbol == "TCS"

    balance = _step_no_io(adapter.get_balance("kite_access_xyz"))
    assert balance.available_cash == Decimal("25500.75")
    assert balance.currency == "INR"

    _step_no_io(adapter.revoke_session("kite_access_xyz"))
    _step_no_io(adapter.aclose())


def test_zerodha_adapter_error_mappings_and_sanitization():
    def run_with_response(response: httpx.Response | Exception, *, is_exchange: bool = False):
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(response, Exception):
                raise response
            return response

        adapter = ZerodhaBrokerAdapter(
            api_key="api_k",
            api_secret="api_s",
            transport=httpx.MockTransport(handler),
            max_retries=0,
        )
        if is_exchange:
            return _step_no_io(adapter.exchange_request_token("bad_req"))
        return _step_no_io(adapter.get_holdings("expired_tok_secret"))

    with pytest.raises(BrokerTokenExpiredError) as exc_tok:
        run_with_response(
            httpx.Response(
                403,
                json={"status": "error", "error_type": "TokenException", "message": "Invalid token expired_tok_secret"},
            )
        )
    assert "TokenException" in (exc_tok.value.error_type or "")

    with pytest.raises(BrokerAuthenticationError):
        run_with_response(
            httpx.Response(
                403,
                json={"status": "error", "error_type": "TokenException", "message": "Bad request token"},
            ),
            is_exchange=True,
        )

    with pytest.raises(BrokerPermissionError):
        run_with_response(
            httpx.Response(
                403,
                json={"status": "error", "error_type": "PermissionException", "message": "Insufficient scope"},
            )
        )

    with pytest.raises(BrokerRateLimitError):
        run_with_response(
            httpx.Response(
                429,
                json={"status": "error", "error_type": "TooManyRequests", "message": "Too many requests"},
            )
        )

    with pytest.raises(BrokerUnavailableError):
        run_with_response(httpx.Response(502, text="<html><body>502 Bad Gateway</body></html>"))

    with pytest.raises(BrokerTimeoutError):
        run_with_response(httpx.ReadTimeout("timed out"))

    with pytest.raises(BrokerResponseValidationError):
        run_with_response(httpx.Response(200, json={"status": "success", "data": "not-a-list"}))


def test_normalizer_handles_empty_unpriced_partial_and_currency_separation():
    normalizer = ZerodhaPortfolioNormalizer()
    profile = BrokerProfile(broker_name="zerodha", broker_account_id="AB1234", user_name="Investor")
    balance = BrokerBalance(available_cash=Decimal("10000"), currency="INR")

    # 1. Empty account succeeds honestly
    empty_res = normalizer.normalize(profile=profile, raw_holdings=[], balance=balance)
    assert empty_res.status == "completed"
    assert empty_res.cash_balance == Decimal("10000")
    assert empty_res.valuation_available is True
    assert empty_res.currency == "INR"
    domain_empty, notes = empty_res.to_domain_portfolio("P-EMPTY")
    assert domain_empty is not None
    assert domain_empty.cash == 10000.0
    assert domain_empty.holdings == []
    assert any("Sector classification" in note for note in notes)

    # 2. Missing last_price preserves average_price separately and refuses silent cost fallback
    unpriced = [
        BrokerHoldingItem(
            symbol="SBIN",
            exchange="NSE",
            quantity=Decimal("10"),
            average_price=Decimal("700.0"),
            last_price=Decimal("0"),
        )
    ]
    unpriced_res = normalizer.normalize(profile=profile, raw_holdings=unpriced, balance=balance)
    assert unpriced_res.status == "completed"
    assert unpriced_res.valuation_available is False
    assert unpriced_res.holdings[0].average_purchase_price == Decimal("700.0")
    assert unpriced_res.holdings[0].last_price is None
    assert unpriced_res.holdings[0].price_source == "unavailable"
    blocked_domain, block_reasons = unpriced_res.to_domain_portfolio("P-UNPRICED")
    assert blocked_domain is None
    assert any("average purchase cost is intentionally not treated as market value" in r for r in block_reasons)

    # 3. Malformed rows mark partial (never completed) under mark_partial, and raise under fail
    mixed_raw = [
        {
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "quantity": 5,
            "t1_quantity": 1,
            "average_price": 1500.0,
            "last_price": 1800.0,
        },
        {
            "tradingsymbol": "BADQTY",
            "exchange": "NSE",
            "quantity": -3,
            "average_price": 100.0,
            "last_price": 110.0,
        },
    ]
    partial_res = normalizer.normalize(
        profile=profile,
        raw_holdings=mixed_raw,
        balance=balance,
        partial_policy="mark_partial",
    )
    assert partial_res.status == "partial"
    assert len(partial_res.malformed_records) == 1
    assert partial_res.holdings[0].quantity == Decimal("6")
    assert partial_res.valuation_available is False
    partial_domain, _ = partial_res.to_domain_portfolio("P-PARTIAL")
    assert partial_domain is None

    with pytest.raises(BrokerResponseValidationError):
        normalizer.normalize(
            profile=profile,
            raw_holdings=mixed_raw,
            balance=balance,
            partial_policy="fail",
        )

    # 4. Valid priced account converts cleanly into core AlphaPilot Portfolio
    valid = [
        BrokerHoldingItem(
            symbol="INFY",
            exchange="NSE",
            quantity=Decimal("8"),
            t1_quantity=Decimal("2"),
            average_price=Decimal("1500.0"),
            last_price=Decimal("1800.0"),
        )
    ]
    valid_res = normalizer.normalize(profile=profile, raw_holdings=valid, balance=balance)
    assert valid_res.status == "completed"
    assert valid_res.valuation_available is True
    assert valid_res.holdings[0].quantity == Decimal("10")
    domain_valid, _ = valid_res.to_domain_portfolio("P-VALID")
    assert domain_valid is not None
    assert domain_valid.holdings[0].symbol == "INFY"
    assert domain_valid.holdings[0].quantity == 10.0
    assert domain_valid.holdings[0].price == 1800.0

