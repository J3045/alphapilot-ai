"""User-owned portfolio CRUD, manual broker refresh, immutable snapshot history, and AI analysis routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.agents.portfolio_graph import run_llm_portfolio_workflow_async
from app.api.dependencies import get_portfolio_ingestion_service, translate_service_exception
from app.api.schemas import (
    LatestPortfolioResponse,
    PortfolioAnalyzeRequest,
    PortfolioAnalyzeResponse,
    PortfolioCreateRequest,
    PortfolioRefreshRequest,
    PortfolioRiskSummaryResponse,
    PortfolioSnapshotResponse,
    PortfolioSummaryResponse,
    SnapshotHoldingResponse,
    SnapshotPositionResponse,
)
from app.auth.dependencies import get_current_user
from app.auth.models import AuthenticatedUser
from app.db.broker_repository import PortfolioSnapshotEntity, UserPortfolioEntity
from app.services.portfolio_ingestion_service import PortfolioIngestionService

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


def _to_portfolio_summary(entity: UserPortfolioEntity) -> PortfolioSummaryResponse:
    return PortfolioSummaryResponse(
        portfolio_id=entity.portfolio_id,
        name=entity.name,
        broker_name=entity.broker_name,
        broker_connection_id=entity.broker_connection_id,
        currency=entity.currency,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _to_snapshot_response(entity: PortfolioSnapshotEntity) -> PortfolioSnapshotResponse:
    meta = dict(entity.metadata or {})
    holdings = [
        SnapshotHoldingResponse(
            holding_id=h.holding_id,
            symbol=h.symbol,
            exchange=h.exchange,
            instrument_identifier=h.instrument_identifier,
            isin=h.isin,
            quantity=h.quantity,
            average_purchase_price=h.average_purchase_price,
            last_price=h.last_price,
            currency=h.currency,
            price_source=str(h.metadata.get("price_source") or "unavailable"),
            valuation_available=bool(h.metadata.get("valuation_available", False)),
        )
        for h in entity.holdings
    ]
    positions = [
        SnapshotPositionResponse(
            position_id=p.position_id,
            symbol=p.symbol,
            exchange=p.exchange,
            instrument_identifier=p.instrument_identifier,
            product=p.product,
            quantity=p.quantity,
            average_price=p.average_price,
            last_price=p.last_price,
            pnl=p.pnl,
            unrealised=p.unrealised,
            realised=p.realised,
            currency=p.currency,
        )
        for p in entity.positions
    ]
    return PortfolioSnapshotResponse(
        snapshot_id=entity.snapshot_id,
        portfolio_id=entity.portfolio_id,
        broker_name=entity.broker_name,
        broker_account_id=entity.broker_account_id,
        fetched_at=entity.fetched_at,
        status=entity.status,
        idempotency_key=entity.idempotency_key,
        cash_balance=entity.cash_balance,
        currency=entity.currency,
        valuation_available=entity.valuation_available,
        holdings_count=entity.holdings_count,
        positions_count=len(positions),
        limitations=[str(x) for x in meta.get("limitations", [])],
        malformed_records=[dict(x) for x in meta.get("malformed_records", []) if isinstance(x, dict)],
        error_detail=entity.error_detail,
        created_at=entity.created_at,
        holdings=holdings,
        positions=positions,
    )


@router.post("", response_model=PortfolioSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    payload: PortfolioCreateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> PortfolioSummaryResponse:
    """Create a logical brokerage portfolio owned by the authenticated user."""
    try:
        created = await service.create_portfolio(
            user_id=current_user.user_id,
            name=payload.name,
            broker_name=payload.broker_name,
            broker_connection_id=payload.broker_connection_id,
            currency=payload.currency,
        )
        return _to_portfolio_summary(created)
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get("", response_model=list[PortfolioSummaryResponse])
async def list_portfolios(
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> list[PortfolioSummaryResponse]:
    """List all portfolios owned by the authenticated user."""
    try:
        items = await service.list_portfolios(user_id=current_user.user_id)
        return [_to_portfolio_summary(item) for item in items]
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.post("/{portfolio_id}/refresh", response_model=PortfolioSnapshotResponse)
async def refresh_portfolio_manual(
    portfolio_id: str,
    payload: PortfolioRefreshRequest | None = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> PortfolioSnapshotResponse:
    """Manually trigger brokerage fetch, normalization, and immutable snapshot persistence."""
    try:
        snapshot = await service.refresh_portfolio(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
            idempotency_key=payload.idempotency_key if payload else None,
            partial_policy=payload.partial_policy if payload else "mark_partial",
        )
        return _to_snapshot_response(snapshot)
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get("/{portfolio_id}", response_model=LatestPortfolioResponse)
async def get_portfolio_with_latest_snapshot(
    portfolio_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> LatestPortfolioResponse:
    """Return user's portfolio and its latest completed immutable snapshot."""
    try:
        portfolio, latest = await service.get_latest_portfolio_with_snapshot(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
        )
        return LatestPortfolioResponse(
            portfolio=_to_portfolio_summary(portfolio),
            latest_snapshot=_to_snapshot_response(latest) if latest else None,
        )
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get("/{portfolio_id}/snapshots", response_model=list[PortfolioSnapshotResponse])
async def list_portfolio_snapshots(
    portfolio_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> list[PortfolioSnapshotResponse]:
    """Return reverse-chronological historical snapshots for the authenticated owner's portfolio."""
    try:
        snapshots = await service.list_snapshots(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
            limit=limit,
        )
        return [_to_snapshot_response(s) for s in snapshots]
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get("/{portfolio_id}/snapshots/{snapshot_id}", response_model=PortfolioSnapshotResponse)
async def get_portfolio_snapshot(
    portfolio_id: str,
    snapshot_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> PortfolioSnapshotResponse:
    """Retrieve a specific immutable snapshot belonging to the authenticated user's portfolio."""
    try:
        snap = await service.get_snapshot(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
            snapshot_id=snapshot_id,
        )
        return _to_snapshot_response(snap)
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.get("/{portfolio_id}/risk", response_model=PortfolioRiskSummaryResponse)
async def evaluate_latest_snapshot_risk(
    portfolio_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> PortfolioRiskSummaryResponse:
    """Run deterministic AlphaPilot risk rules against the latest completed portfolio snapshot."""
    try:
        ctx = await service.build_intelligence_context(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
        )
        return PortfolioRiskSummaryResponse(
            provenance=ctx.provenance,
            risk_assessment=ctx.risk_assessment,
            limitations=ctx.limitations,
        )
    except Exception as exc:
        raise translate_service_exception(exc) from exc


@router.post("/{portfolio_id}/analyze", response_model=PortfolioAnalyzeResponse)
async def analyze_synced_portfolio(
    portfolio_id: str,
    payload: PortfolioAnalyzeRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    service: PortfolioIngestionService = Depends(get_portfolio_ingestion_service),
) -> PortfolioAnalyzeResponse:
    """Feed the latest completed snapshot into Risk Engine + LangGraph without exposing broker secrets."""
    try:
        ctx = await service.build_intelligence_context(
            user_id=current_user.user_id,
            portfolio_id=portfolio_id,
        )
    except Exception as exc:
        raise translate_service_exception(exc) from exc

    if ctx.snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No completed portfolio snapshot available to analyze; run a manual refresh first.",
        )

    llm_override = getattr(request.app.state, "agent_llm_client", None)
    state = await run_llm_portfolio_workflow_async(
        user_query=payload.query,
        llm_client=llm_override,
        portfolio_context=ctx.domain_portfolio,
        snapshot_provenance=ctx.provenance,
        portfolio_limitations=ctx.limitations,
        precomputed_assessment=ctx.risk_assessment,
    )
    return PortfolioAnalyzeResponse(
        status=str(state.get("status", "unknown")),
        provenance=ctx.provenance,
        risk_assessment=state.get("assessment_result") or ctx.risk_assessment,
        limitations=ctx.limitations,
        final_response=state.get("final_response"),
        error=state.get("error"),
    )

