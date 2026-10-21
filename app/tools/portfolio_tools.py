from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.models.portfolio import Portfolio
from app.services.risk_engine import assess_portfolio_risk


class HoldingConcentration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    weight_pct: float


class PortfolioAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_id: str
    total_value: float
    cash_pct: float
    concentrations: list[HoldingConcentration]
    passed: bool
    violations: list[str]


class PortfolioAssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio: Portfolio
    max_single_holding_pct: float = Field(default=40.0, gt=0, le=100)
    min_cash_pct: float = Field(default=5.0, ge=0, le=100)


def assess_portfolio_risk_tool(
    input_data: PortfolioAssessmentInput | dict[str, Any] | None = None,
    **kwargs: Any,
) -> PortfolioAssessmentResult:
    """Execute deterministic portfolio risk assessment with strict input validation.

    Accepts either a validated `PortfolioAssessmentInput`, a dictionary representing the input,
    or keyword arguments matching `PortfolioAssessmentInput`. Unexpected fields are rejected.
    """
    if input_data is not None:
        if kwargs:
            raise ValueError("Provide either input_data or keyword arguments, not both.")
        if isinstance(input_data, PortfolioAssessmentInput):
            validated = input_data
        elif isinstance(input_data, dict):
            validated = PortfolioAssessmentInput.model_validate(input_data)
        else:
            raise TypeError(
                f"Expected PortfolioAssessmentInput or dict, got {type(input_data).__name__}"
            )
    else:
        validated = PortfolioAssessmentInput.model_validate(kwargs)

    raw_result = assess_portfolio_risk(
        portfolio=validated.portfolio,
        max_single_holding_pct=validated.max_single_holding_pct,
        min_cash_pct=validated.min_cash_pct,
    )
    return PortfolioAssessmentResult.model_validate(raw_result)


# Controlled tool execution registry to prevent arbitrary code execution
AVAILABLE_TOOLS: dict[str, Any] = {
    "assess_portfolio_risk": assess_portfolio_risk_tool,
}


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Controlled tool dispatcher.

    Guarantees callers can only invoke explicitly registered tools,
    rejecting arbitrary functions or command execution attempts.
    Returns a predictable, JSON-serializable dictionary.
    """
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Tool '{tool_name}' is not permitted or does not exist.")

    tool_fn = AVAILABLE_TOOLS[tool_name]
    result: PortfolioAssessmentResult = tool_fn(arguments)
    return result.model_dump()


try:
    from langchain_core.tools import tool

    @tool("assess_portfolio_risk", args_schema=PortfolioAssessmentInput)
    def assess_portfolio_risk_langchain_tool(
        portfolio: Portfolio,
        max_single_holding_pct: float = 40.0,
        min_cash_pct: float = 5.0,
    ) -> dict[str, Any]:
        """Assess portfolio risk for single-holding concentration and minimum cash limits."""
        res = assess_portfolio_risk_tool(
            portfolio=portfolio,
            max_single_holding_pct=max_single_holding_pct,
            min_cash_pct=min_cash_pct,
        )
        return res.model_dump()

    APPROVED_LANGCHAIN_TOOLS = [assess_portfolio_risk_langchain_tool]
except ImportError:  # pragma: no cover
    APPROVED_LANGCHAIN_TOOLS = []
