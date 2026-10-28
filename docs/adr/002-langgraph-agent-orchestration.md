# ADR 002: LangGraph for Agent Workflow Orchestration

## Status
Accepted

## Context
The portfolio intelligence agent requires a multi-step execution loop: receiving a user query, invoking the LLM to select an action, dispatching controlled tools, inspecting tool outputs, and generating a grounded response. Unconstrained while-loops in vanilla Python lack state validation, breakpoint inspection, and cyclical safety guarantees.

## Decision
We use **LangGraph** (`StateGraph`) to construct a deterministic state machine for orchestrating the LLM tool-calling workflow (`app.agents.portfolio_graph.py`). Workflow state is explicitly defined as a typed dictionary (`LLMPortfolioState`), and node transitions are controlled via explicit conditional routing functions (`route_after_model`, `route_after_tool`).

## Benefits
- **Explicit State Transitions**: Transitions between model invocation, tool execution, and finalization are formal graph edges, preventing uncontrolled execution jumps.
- **Bounded Iteration Cycles**: Hard iteration ceilings (`iteration_count >= max_iterations`) and workflow deadline checks are enforced at node boundaries.
- **Traceability**: Every state update is explicit, making debugging and test-driven mocking straightforward.

## Trade-offs
- Introduces a dependency on `langgraph` and `langchain-core`.
- Graph compilation overhead (negligible for small graphs).

## Alternatives Considered
- *Vanilla Python While-Loops*: Harder to maintain clean state boundaries and test intermediate execution steps.
- *Autonomous Multi-Agent Frameworks (CrewAI, AutoGen)*: Too heavy for an MVP, introduced unnecessary abstractions, and increased vulnerability to multi-agent prompt loops.

## Conditions for Revisit
If the agent grows to require multi-agent collaboration (e.g., separate research and compliance agents) or human-in-the-loop approvals before executing actions, LangGraph's checkpointer features can be leveraged.
