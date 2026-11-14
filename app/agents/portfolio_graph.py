import json
from typing import Any, Literal, TypedDict
from pydantic import ValidationError

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from app.config import settings
from app.observability import resolve_model_name, track_llm_operation
from app.services.resilience import (
    LLMConcurrencyLimiter,
    WorkflowDeadline,
    WorkflowDeadlineExceededError,
    execute_with_retry,
    invoke_llm_with_resilience_async,
    sanitize_provider_error,
)
from app.tools.portfolio_tools import (
    APPROVED_LANGCHAIN_TOOLS,
    AVAILABLE_TOOLS,
    execute_tool,
)


class PortfolioWorkflowState(TypedDict, total=False):
    """Explicit workflow state representation for portfolio assessment."""

    portfolio_input: dict[str, Any]
    assessment_result: dict[str, Any] | None
    error: str | None
    status: Literal["pending", "completed", "failed"]


def assess_portfolio_node(state: PortfolioWorkflowState) -> dict[str, Any]:
    """Execute deterministic portfolio assessment using the approved tool."""
    portfolio_input = state.get("portfolio_input")

    if portfolio_input is None:
        return {
            "status": "failed",
            "error": "Missing 'portfolio_input' in workflow state.",
            "assessment_result": None,
        }

    if not isinstance(portfolio_input, dict):
        return {
            "status": "failed",
            "error": f"Invalid 'portfolio_input' format: expected dict, got {type(portfolio_input).__name__}.",
            "assessment_result": None,
        }

    try:
        result = execute_tool("assess_portfolio_risk", portfolio_input)
        return {
            "status": "completed",
            "assessment_result": result,
            "error": None,
        }
    except (ValidationError, ValueError) as exc:
        return {
            "status": "failed",
            "error": f"Portfolio assessment rejected: {str(exc)}",
            "assessment_result": None,
        }


def build_portfolio_graph():
    """Construct and compile the minimal deterministic portfolio state machine."""
    workflow = StateGraph(PortfolioWorkflowState)

    workflow.add_node("assess_portfolio", assess_portfolio_node)

    workflow.add_edge(START, "assess_portfolio")
    workflow.add_edge("assess_portfolio", END)

    return workflow.compile()


def run_portfolio_workflow(portfolio_input: dict[str, Any]) -> PortfolioWorkflowState:
    """Execute the portfolio assessment graph synchronously."""
    graph = build_portfolio_graph()
    initial_state: PortfolioWorkflowState = {
        "portfolio_input": portfolio_input,
        "assessment_result": None,
        "error": None,
        "status": "pending",
    }
    return graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# LLM Tool-Calling Portfolio Workflow
# ---------------------------------------------------------------------------


_SAFE_PROVENANCE_KEYS = (
    "portfolio_id",
    "portfolio_name",
    "broker_name",
    "snapshot_id",
    "fetched_at",
    "currency",
    "valuation_available",
    "holdings_count",
    "snapshot_available",
)


def sanitize_snapshot_provenance(provenance: dict[str, Any] | None) -> dict[str, Any] | None:
    """Allowlist-sanitize snapshot provenance so credentials or tokens never enter graph state."""
    if not provenance:
        return None
    return {
        key: provenance[key]
        for key in _SAFE_PROVENANCE_KEYS
        if key in provenance
    }


class LLMPortfolioState(TypedDict, total=False):
    """Workflow state representation for the LLM tool-calling agent."""

    user_query: str
    messages: list[BaseMessage]
    portfolio_input: dict[str, Any] | None
    assessment_result: dict[str, Any] | None
    final_response: str | None
    status: Literal["pending", "completed", "failed"]
    error: str | None
    iteration_count: int
    max_iterations: int
    deadline: WorkflowDeadline | None
    snapshot_provenance: dict[str, Any] | None
    portfolio_limitations: list[str] | None


def _check_model_node_preconditions(
    state: LLMPortfolioState,
) -> tuple[int, WorkflowDeadline | None, dict[str, Any] | None]:
    """Validate iteration count and workflow deadline before calling the LLM."""
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", settings.max_agent_iterations)
    deadline = state.get("deadline")

    if deadline is not None and deadline.is_expired():
        return iteration_count, deadline, {
            "status": "failed",
            "error": "LLM provider error: Workflow execution deadline exceeded.",
            "iteration_count": iteration_count + 1,
            "assessment_result": None,
            "final_response": None,
        }

    if iteration_count >= max_iterations:
        return iteration_count, deadline, {
            "status": "failed",
            "error": f"Execution limit reached: model reached maximum invocations limit ({max_iterations}).",
            "iteration_count": iteration_count,
        }

    return iteration_count, deadline, None


def _resolve_agent_llm_client(
    llm_client: Any,
    iteration_count: int,
) -> tuple[Any, dict[str, Any] | None]:
    """Return provided LLM client or build default ChatOpenAI bound to approved tools."""
    if llm_client is not None:
        return llm_client, None

    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        return None, {
            "status": "failed",
            "error": "LLM provider error: OpenAI API key is not configured.",
            "iteration_count": iteration_count + 1,
        }

    client = ChatOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        timeout=settings.openai_timeout_seconds,
        max_retries=0,
    ).bind_tools(APPROVED_LANGCHAIN_TOOLS)
    return client, None


def _build_model_error_update(exc: Exception, iteration_count: int) -> dict[str, Any]:
    is_deadline = isinstance(exc, WorkflowDeadlineExceededError) or "deadline" in str(exc).lower()
    return {
        "status": "failed",
        "error": f"LLM provider error: {sanitize_provider_error(exc)}",
        "iteration_count": iteration_count + 1,
        **({"assessment_result": None, "final_response": None} if is_deadline else {}),
    }


def make_call_model_node(llm_client: Any = None):
    """Factory creating the synchronous model invocation node with bounded iterations and error isolation."""

    def call_model_node(state: LLMPortfolioState) -> dict[str, Any]:
        iteration_count, deadline, pre_error = _check_model_node_preconditions(state)
        if pre_error is not None:
            return pre_error

        client, client_error = _resolve_agent_llm_client(llm_client, iteration_count)
        if client_error is not None:
            return client_error

        model_name = resolve_model_name(
            client=client,
            fallback=settings.openai_model,
        )

        try:
            with track_llm_operation(operation="portfolio_agent_model", model=model_name) as tracker:
                response = execute_with_retry(
                    lambda: client.invoke(state.get("messages", [])),
                    deadline=deadline,
                )
                tracker.record_success(response=response, outcome="success")
                return {
                    "messages": state.get("messages", []) + [response],
                    "iteration_count": iteration_count + 1,
                }
        except Exception as exc:
            return _build_model_error_update(exc, iteration_count)

    return call_model_node


def make_async_call_model_node(
    llm_client: Any = None,
    concurrency_limiter: LLMConcurrencyLimiter | None = None,
):
    """Factory creating the async model invocation node with bounded concurrency and error isolation."""

    async def call_model_node_async(state: LLMPortfolioState) -> dict[str, Any]:
        iteration_count, deadline, pre_error = _check_model_node_preconditions(state)
        if pre_error is not None:
            return pre_error

        client, client_error = _resolve_agent_llm_client(llm_client, iteration_count)
        if client_error is not None:
            return client_error

        model_name = resolve_model_name(
            client=client,
            fallback=settings.openai_model,
        )

        try:
            with track_llm_operation(operation="portfolio_agent_model", model=model_name) as tracker:
                response = await invoke_llm_with_resilience_async(
                    client=client,
                    messages=state.get("messages", []),
                    deadline=deadline,
                    concurrency_limiter=concurrency_limiter,
                )
                tracker.record_success(response=response, outcome="success")
                return {
                    "messages": state.get("messages", []) + [response],
                    "iteration_count": iteration_count + 1,
                }
        except Exception as exc:
            return _build_model_error_update(exc, iteration_count)

    return call_model_node_async


def route_after_model(state: LLMPortfolioState) -> str:
    """Route based on model output: tool call, plain text, or failure."""
    if state.get("status") == "failed":
        return END

    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls and len(tool_calls) > 0:
        return "execute_tool_node"

    return "finalize_text_node"


def execute_tool_node(state: LLMPortfolioState) -> dict[str, Any]:
    """Execute the model-requested tool using strict allowlist and validation."""
    deadline = state.get("deadline")
    if deadline is not None and deadline.is_expired():
        return {
            "status": "failed",
            "error": "Workflow execution deadline exceeded.",
            "assessment_result": None,
            "final_response": None,
        }

    messages = state.get("messages", [])
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    call = tool_calls[0]

    tool_name = call.get("name")
    tool_args = call.get("args")
    tool_call_id = call.get("id", "call_default")

    if tool_name not in AVAILABLE_TOOLS:
        return {
            "status": "failed",
            "error": f"Controlled tool execution rejected: unapproved tool '{tool_name}'.",
            "assessment_result": None,
        }

    if not isinstance(tool_args, dict):
        return {
            "status": "failed",
            "error": "Controlled tool execution rejected: malformed tool arguments (expected dictionary).",
            "assessment_result": None,
        }

    try:
        assessment = execute_tool(tool_name, tool_args)
        tool_message = ToolMessage(
            content=json.dumps(assessment),
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        return {
            "portfolio_input": tool_args,
            "assessment_result": assessment,
            "messages": messages + [tool_message],
            "status": "completed",
            "error": None,
        }
    except PermissionError as exc:
        return {
            "status": "failed",
            "error": f"Controlled tool execution rejected: {str(exc)}",
            "portfolio_input": tool_args,
            "assessment_result": None,
        }
    except (ValidationError, ValueError) as exc:
        return {
            "status": "failed",
            "error": f"Controlled tool execution rejected: malformed tool arguments: {str(exc)}",
            "portfolio_input": tool_args,
            "assessment_result": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"Controlled tool execution rejected: tool failed ({type(exc).__name__}).",
            "portfolio_input": tool_args,
            "assessment_result": None,
        }


def route_after_tool(state: LLMPortfolioState) -> str:
    """Route after tool execution: abort if failed, otherwise loop back for model synthesis."""
    if state.get("status") == "failed":
        return END
    return "call_model"


def finalize_text_node(state: LLMPortfolioState) -> dict[str, Any]:
    """Capture final plain-text response from the model when no tool calls remain."""
    messages = state.get("messages", [])
    content = messages[-1].content if messages else ""
    return {
        "final_response": content if isinstance(content, str) else str(content),
        "status": "completed",
        "error": None,
    }


def _build_graph_with_call_node(call_node: Any):
    workflow = StateGraph(LLMPortfolioState)

    workflow.add_node("call_model", call_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("finalize_text", finalize_text_node)

    workflow.add_edge(START, "call_model")

    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool_node": "execute_tool",
            "finalize_text_node": "finalize_text",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "call_model": "call_model",
            END: END,
        },
    )

    workflow.add_edge("finalize_text", END)

    return workflow.compile()


def build_llm_portfolio_graph(llm_client: Any = None):
    """Construct and compile the synchronous LLM tool-calling portfolio state machine."""
    return _build_graph_with_call_node(make_call_model_node(llm_client))


def build_async_llm_portfolio_graph(
    llm_client: Any = None,
    concurrency_limiter: LLMConcurrencyLimiter | None = None,
):
    """Construct and compile the asynchronous LLM tool-calling portfolio state machine."""
    return _build_graph_with_call_node(
        make_async_call_model_node(
            llm_client=llm_client,
            concurrency_limiter=concurrency_limiter,
        )
    )


def _create_initial_llm_state(
    user_query: str,
    max_iterations: int | None,
    system_prompt: str | None,
    workflow_timeout_seconds: float | None,
    deadline: WorkflowDeadline | None,
    portfolio_context: Any = None,
    snapshot_provenance: dict[str, Any] | None = None,
    portfolio_limitations: list[str] | None = None,
    precomputed_assessment: dict[str, Any] | None = None,
) -> LLMPortfolioState:
    if not user_query or not user_query.strip():
        raise ValueError("user_query cannot be empty or whitespace-only.")
    if len(user_query.strip()) > settings.max_query_length:
        raise ValueError(
            f"user_query character length ({len(user_query.strip())}) exceeds "
            f"maximum permitted limit ({settings.max_query_length})."
        )

    limit = max_iterations if max_iterations is not None else settings.max_agent_iterations
    safe_provenance = sanitize_snapshot_provenance(snapshot_provenance)
    safe_limitations = list(portfolio_limitations or []) if portfolio_limitations else None

    prompt_text = user_query
    context_sections: list[str] = []
    if safe_provenance:
        context_sections.append(
            f"Snapshot Provenance (Static Broker Refresh):\n{json.dumps(safe_provenance, indent=2)}"
        )
    if portfolio_context is not None:
        if hasattr(portfolio_context, "model_dump_json"):
            portfolio_json = portfolio_context.model_dump_json(indent=2, exclude_none=True)
        else:
            portfolio_json = json.dumps(portfolio_context, indent=2, default=str)
        context_sections.append(f"Portfolio Context:\n{portfolio_json}")
    if precomputed_assessment is not None:
        context_sections.append(
            f"Deterministic Risk Assessment:\n{json.dumps(precomputed_assessment, indent=2)}"
        )
    if safe_limitations:
        bullet_notes = "\n".join(f"- {note}" for note in safe_limitations)
        context_sections.append(f"Known Data Limitations:\n{bullet_notes}")

    if context_sections:
        prompt_text = f"{user_query}\n\n" + "\n\n".join(context_sections)

    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt_text))

    active_deadline = deadline
    if active_deadline is None:
        timeout = (
            workflow_timeout_seconds
            if workflow_timeout_seconds is not None
            else settings.workflow_timeout_seconds
        )
        active_deadline = WorkflowDeadline(timeout_seconds=timeout)

    portfolio_input_dict = (
        portfolio_context.model_dump(exclude_none=True)
        if hasattr(portfolio_context, "model_dump")
        else (portfolio_context if isinstance(portfolio_context, dict) else None)
    )

    return {
        "user_query": user_query,
        "messages": messages,
        "portfolio_input": portfolio_input_dict,
        "assessment_result": precomputed_assessment,
        "final_response": None,
        "status": "pending",
        "error": None,
        "iteration_count": 0,
        "max_iterations": limit,
        "deadline": active_deadline,
        "snapshot_provenance": safe_provenance,
        "portfolio_limitations": safe_limitations,
    }


def run_llm_portfolio_workflow(
    user_query: str,
    llm_client: Any = None,
    max_iterations: int | None = None,
    system_prompt: str | None = None,
    workflow_timeout_seconds: float | None = None,
    deadline: WorkflowDeadline | None = None,
    portfolio_context: Any = None,
    snapshot_provenance: dict[str, Any] | None = None,
    portfolio_limitations: list[str] | None = None,
    precomputed_assessment: dict[str, Any] | None = None,
) -> LLMPortfolioState:
    """Execute the LLM portfolio workflow synchronously."""
    initial_state = _create_initial_llm_state(
        user_query=user_query,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
        workflow_timeout_seconds=workflow_timeout_seconds,
        deadline=deadline,
        portfolio_context=portfolio_context,
        snapshot_provenance=snapshot_provenance,
        portfolio_limitations=portfolio_limitations,
        precomputed_assessment=precomputed_assessment,
    )
    graph = build_llm_portfolio_graph(llm_client=llm_client)
    return graph.invoke(initial_state)


async def run_llm_portfolio_workflow_async(
    user_query: str,
    llm_client: Any = None,
    max_iterations: int | None = None,
    system_prompt: str | None = None,
    workflow_timeout_seconds: float | None = None,
    deadline: WorkflowDeadline | None = None,
    concurrency_limiter: LLMConcurrencyLimiter | None = None,
    portfolio_context: Any = None,
    snapshot_provenance: dict[str, Any] | None = None,
    portfolio_limitations: list[str] | None = None,
    precomputed_assessment: dict[str, Any] | None = None,
) -> LLMPortfolioState:
    """Execute the LLM portfolio workflow asynchronously with bounded concurrency."""
    initial_state = _create_initial_llm_state(
        user_query=user_query,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
        workflow_timeout_seconds=workflow_timeout_seconds,
        deadline=deadline,
        portfolio_context=portfolio_context,
        snapshot_provenance=snapshot_provenance,
        portfolio_limitations=portfolio_limitations,
        precomputed_assessment=precomputed_assessment,
    )
    graph = build_async_llm_portfolio_graph(
        llm_client=llm_client,
        concurrency_limiter=concurrency_limiter,
    )
    return await graph.ainvoke(initial_state)

