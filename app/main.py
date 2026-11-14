
import asyncio
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException

from app.agents.portfolio_graph import run_llm_portfolio_workflow_async
from app.api.auth_routes import router as auth_router
from app.api.broker_routes import router as broker_router
from app.api.portfolio_routes import router as portfolio_router
from app.config import settings
from app.db.session import (
    dispose_default_database_manager,
    get_default_database_manager,
    has_database_configuration,
)
from app.logging_config import configure_logging
from app.models.agent import AgentRunRequest, AgentRunResponse
from app.models.portfolio import Portfolio
from app.models.rag import RagAskRequest, RagAskResponse
from app.services.rag_service import get_default_rag_service, reset_default_rag_service
from app.services.resilience import (
    get_default_llm_limiter,
    reset_default_llm_limiter,
    supports_async_method,
)
from app.services.risk_engine import assess_portfolio_risk


configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service = get_default_rag_service()
    llm_limiter = get_default_llm_limiter()
    app.state.rag_service = rag_service
    app.state.llm_limiter = llm_limiter
    if has_database_configuration():
        app.state.db_manager = get_default_database_manager()

    logger.info(
        "application_starting",
        extra={
            "pid": os.getpid(),
            "max_concurrent_llm_requests": settings.max_concurrent_llm_requests,
            "rag_prewarm_on_startup": settings.rag_prewarm_on_startup,
            "rag_storage_backend": settings.rag_storage_backend,
            "database_configured": has_database_configuration(),
        },
    )

    if settings.rag_prewarm_on_startup:
        await rag_service._ensure_initialized_async()

    try:
        yield
    finally:
        if supports_async_method(rag_service, "aclose"):
            await rag_service.aclose()
        registry = getattr(app.state, "broker_registry", None)
        if registry is not None and supports_async_method(registry, "aclose_all"):
            await registry.aclose_all()
        await dispose_default_database_manager()
        reset_default_rag_service()
        reset_default_llm_limiter()
        logger.info("application_stopping", extra={"pid": os.getpid()})


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(auth_router)
app.include_router(broker_router)
app.include_router(portfolio_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "application": settings.app_name,
        "environment": settings.environment,
    }


@app.post("/portfolio/assess")
async def assess_portfolio(portfolio: Portfolio):
    try:
        result = assess_portfolio_risk(portfolio)
        logger.info(
            "portfolio_assessed",
            extra={"portfolio_id": portfolio.portfolio_id},
        )
        return result
    except ValueError as exc:
        logger.warning("portfolio_assessment_rejected: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc




@app.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    try:
        query = request.query
        if request.portfolio:
            query = (
                f"{request.query}\n\n"
                f"Portfolio Context:\n{request.portfolio.model_dump_json(indent=2)}"
            )

        state = await run_llm_portfolio_workflow_async(user_query=query)

        logger.info(
            "agent_workflow_executed",
            extra={"status": state.get("status")},
        )

        return AgentRunResponse(
            status=state.get("status", "unknown"),
            assessment_result=state.get("assessment_result"),
            final_response=state.get("final_response"),
            error=state.get("error"),
        )
    except Exception as exc:
        logger.error("agent_workflow_unhandled_failure: %s", exc)
        return AgentRunResponse(
            status="failed",
            error="An error occurred while executing the agent workflow.",
        )


@app.post("/rag/ask", response_model=RagAskResponse)
async def ask_rag(request: RagAskRequest):
    """Answer research and policy questions grounded in synthetic document evidence."""
    rag_service = get_default_rag_service()
    try:
        if supports_async_method(rag_service, "answer_question_async"):
            result = await rag_service.answer_question_async(
                question=request.question,
                top_k=request.top_k,
            )
        else:
            result = await asyncio.to_thread(
                rag_service.answer_question,
                question=request.question,
                top_k=request.top_k,
            )

        if result.status == "failed":
            logger.error("rag_ask_failed: %s", result.error)
            raise HTTPException(
                status_code=503,
                detail=result.error or "RAG service temporarily unavailable.",
            )

        return RagAskResponse(
            status=result.status,
            answer=result.answer,
            sources=result.sources,
            error=result.error,
        )
    except ValueError as exc:
        logger.warning("rag_ask_invalid_request: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


