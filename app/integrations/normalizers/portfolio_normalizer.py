"""Canonical portfolio normalizer separating broker wire formats from core portfolio logic."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Mapping, Sequence

from app.integrations.brokers.exceptions import BrokerResponseValidationError
from app.integrations.brokers.models import (
    BrokerBalance,
    BrokerHoldingItem,
    BrokerPositionItem,
    BrokerProfile,
    MalformedBrokerRecord,
    NormalizedHolding,
    NormalizedPortfolioResult,
    POSITIONS_EXCLUSION_LIMITATION,
    SnapshotStatus,
)
from app.logging_config import redact_sensitive_text


def _sanitize_malformed_entry(
    record: MalformedBrokerRecord | Mapping[str, Any],
    *,
    default_type: Literal["holding", "position"] = "holding",
) -> dict[str, Any]:
    if isinstance(record, MalformedBrokerRecord):
        return {
            "record_type": record.record_type,
            "index": record.index,
            "symbol": redact_sensitive_text(record.symbol),
            "exchange": record.exchange,
            "reason": redact_sensitive_text(record.reason)[:500],
        }
    raw_reason = str(record.get("reason") or "Malformed broker record.")
    return {
        "record_type": str(record.get("record_type") or default_type),
        "index": int(record.get("index") or 0),
        "symbol": redact_sensitive_text(str(record.get("symbol") or "UNKNOWN")),
        "exchange": str(record["exchange"]) if record.get("exchange") else None,
        "reason": redact_sensitive_text(raw_reason)[:500],
    }


class ZerodhaPortfolioNormalizer:
    """Normalizes Zerodha holdings, positions, and cash balances into AlphaPilot domain snapshots."""

    DEFAULT_CURRENCY = "INR"
    SECTOR_LIMITATION_NOTE = (
        "Sector classification is not provided by Zerodha holdings APIs; sector concentration analysis "
        "is unavailable without external instrument metadata."
    )

    def normalize(
        self,
        *,
        profile: BrokerProfile,
        raw_holdings: Sequence[BrokerHoldingItem | Mapping[str, Any]],
        balance: BrokerBalance | None,
        raw_positions: Sequence[BrokerPositionItem] | None = None,
        pre_malformed_holdings: Sequence[MalformedBrokerRecord | Mapping[str, Any]] | None = None,
        pre_malformed_positions: Sequence[MalformedBrokerRecord | Mapping[str, Any]] | None = None,
        fetched_at: datetime | None = None,
        partial_policy: Literal["fail", "mark_partial"] = "mark_partial",
    ) -> NormalizedPortfolioResult:
        """Convert Zerodha holdings and balances into a validated NormalizedPortfolioResult.

        Enforces strict guarantees:
        - Rejects negative, blank, or non-finite quantities/prices cleanly without silent coercion.
        - Explicitly separates `average_purchase_price` from current market `last_price`.
        - Tracks malformed holding and position records explicitly (`partial_policy='fail'` raises;
          `'mark_partial'` flags snapshot status='partial' and never promotes it to 'completed').
        - Preserves valid holdings/positions and empty holding accounts cleanly with explicit cash valuation.
        """
        timestamp = fetched_at or datetime.now(timezone.utc)
        limitations: list[str] = [self.SECTOR_LIMITATION_NOTE]
        malformed_records: list[dict[str, Any]] = [
            _sanitize_malformed_entry(item, default_type="holding")
            for item in (pre_malformed_holdings or ())
        ]
        malformed_records.extend(
            _sanitize_malformed_entry(item, default_type="position")
            for item in (pre_malformed_positions or ())
        )
        normalized_holdings: list[NormalizedHolding] = []
        all_holdings_valued = True

        for idx, item in enumerate(raw_holdings):
            parsed: BrokerHoldingItem | None = None
            if isinstance(item, BrokerHoldingItem):
                parsed = item
            elif isinstance(item, Mapping):
                try:
                    parsed = BrokerHoldingItem(
                        symbol=str(item.get("tradingsymbol") or item.get("symbol") or ""),
                        exchange=str(item.get("exchange") or "NSE"),
                        instrument_identifier=(
                            str(item["instrument_token"])
                            if item.get("instrument_token") is not None
                            else item.get("instrument_identifier")
                        ),
                        isin=item.get("isin"),
                        product=item.get("product"),
                        quantity=item.get("quantity"),
                        t1_quantity=item.get("t1_quantity", 0),
                        average_price=item.get("average_price"),
                        last_price=item.get("last_price"),
                        pnl=item.get("pnl"),
                        currency=str(item.get("currency") or self.DEFAULT_CURRENCY),
                    )
                except Exception as exc:
                    malformed_records.append(
                        {
                            "record_type": "holding",
                            "index": idx,
                            "symbol": redact_sensitive_text(
                                str(item.get("tradingsymbol") or item.get("symbol") or "UNKNOWN")
                            ),
                            "exchange": str(item.get("exchange")) if item.get("exchange") else None,
                            "reason": redact_sensitive_text(str(exc))[:500],
                        }
                    )
                    continue
            else:
                malformed_records.append(
                    {
                        "record_type": "holding",
                        "index": idx,
                        "symbol": "UNKNOWN",
                        "exchange": None,
                        "reason": "Unsupported holding entry type.",
                    }
                )
                continue

            total_qty = parsed.quantity + parsed.t1_quantity
            if parsed.last_price is not None and parsed.last_price > Decimal("0"):
                market_price: Decimal | None = parsed.last_price
                price_source = "broker_ltp"
                item_valuation_available = True
            else:
                market_price = None
                price_source = "unavailable"
                item_valuation_available = False
                if total_qty > Decimal("0"):
                    all_holdings_valued = False
                    limitations.append(
                        f"Market price (last_price) unavailable for {parsed.symbol} ({parsed.exchange}); "
                        "average purchase price is stored separately and not used as market value."
                    )

            metadata: dict[str, Any] = {
                "product": parsed.product,
                "settled_quantity": str(parsed.quantity),
                "t1_quantity": str(parsed.t1_quantity),
                "price_source": price_source,
            }
            if parsed.pnl is not None:
                metadata["broker_reported_pnl"] = str(parsed.pnl)
            if parsed.raw_metadata:
                metadata.update(parsed.raw_metadata)

            normalized_holdings.append(
                NormalizedHolding(
                    symbol=parsed.symbol,
                    exchange=parsed.exchange,
                    instrument_identifier=parsed.instrument_identifier,
                    isin=parsed.isin,
                    quantity=total_qty,
                    settled_quantity=parsed.quantity,
                    t1_quantity=parsed.t1_quantity,
                    average_purchase_price=parsed.average_price,
                    last_price=market_price,
                    currency=parsed.currency or self.DEFAULT_CURRENCY,
                    price_source=price_source,
                    valuation_available=item_valuation_available,
                    metadata=metadata,
                )
            )

        if malformed_records and partial_policy == "fail":
            raise BrokerResponseValidationError(
                f"Broker holdings ingestion rejected {len(malformed_records)} malformed holding record(s).",
                broker_name="zerodha",
            )

        cash_balance: Decimal | None = None
        if balance is None:
            limitations.append("Broker cash balance unavailable from margins endpoint.")
        else:
            cash_balance = balance.available_cash

        positions_list = list(raw_positions or [])
        if positions_list and POSITIONS_EXCLUSION_LIMITATION not in limitations:
            limitations.append(POSITIONS_EXCLUSION_LIMITATION)

        status: SnapshotStatus
        if malformed_records:
            status = "partial"
            limitations.append(
                f"{len(malformed_records)} malformed broker record(s) excluded; snapshot marked partial."
            )
            valuation_available = False
        elif cash_balance is None:
            status = "partial"
            valuation_available = False
        else:
            status = "completed"
            valuation_available = all_holdings_valued

        return NormalizedPortfolioResult(
            broker_name="zerodha",
            broker_account_id=profile.broker_account_id,
            currency=balance.currency if balance is not None else self.DEFAULT_CURRENCY,
            cash_balance=cash_balance,
            holdings=normalized_holdings,
            positions=positions_list,
            status=status,
            valuation_available=valuation_available,
            malformed_records=malformed_records,
            limitations=limitations,
            fetched_at=timestamp,
            profile=profile,
        )

