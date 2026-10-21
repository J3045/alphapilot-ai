
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from app.agents.portfolio_graph import run_llm_portfolio_workflow
from app.config import settings
from app.logging_config import configure_logging
from app.models.portfolio import Portfolio
from app.services.risk_engine import assess_portfolio_risk


configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_starting")
    yield
    logger.info("application_stopping")


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    lifespan=lifespan,
)


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


class AgentRunRequest(BaseModel):
    query: str
    portfolio: Portfolio | None = None


class AgentRunResponse(BaseModel):
    status: str
    assessment_result: dict | None = None
    final_response: str | None = None
    error: str | None = None


@app.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    try:
        query = request.query
        if request.portfolio:
            query = (
                f"{request.query}\n\n"
                f"Portfolio Context:\n{request.portfolio.model_dump_json(indent=2)}"
            )

        state = run_llm_portfolio_workflow(user_query=query)

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

