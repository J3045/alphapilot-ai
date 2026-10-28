# ADR 001: Separation of Deterministic Financial Calculations from LLM Reasoning

## Status
Accepted

## Context
AlphaPilot AI analyzes investment portfolios against risk guidelines (such as single-security concentration limits and minimum cash buffer requirements). Large Language Models (LLMs) are probabilistic systems prone to arithmetic hallucinations, floating-point rounding errors, and non-deterministic logic. In financial applications, regulatory compliance and fiduciary responsibility require exact, audit-verifiable calculations.

## Decision
We strictly isolate all financial arithmetic and compliance rule enforcement into deterministic Python functions (`app.services.risk_engine` and `app.tools.portfolio_tools`). The LLM is never permitted to calculate sums, percentages, or compliance pass/fail statuses directly; it may only propose arguments to deterministic tools and synthesize natural-language summaries based on tool results.

## Benefits
- **Auditability & Accuracy**: Calculations are 100% reproducible and verifiable via standard unit tests.
- **Security & Integrity**: Prompt injections or adversarial inputs cannot trick the engine into reporting that an overconcentrated portfolio passed compliance.
- **Separation of Concerns**: Mathematical logic can be optimized, tested, and updated independently of LLM prompt templates.

## Trade-offs
- Requires structured Pydantic models for both tool inputs and outputs.
- Adds an additional step in the agent workflow where tool calls must be dispatched, validated, and returned to the model context.

## Alternatives Considered
- *Direct Prompt Arithmetic*: Asking the model to compute percentages in the prompt. Rejected due to hallucination risks and lack of precision guarantees.
- *LLM Code Interpreter*: Permitting the LLM to write and execute arbitrary Python code. Rejected due to extreme security risks (arbitrary code execution / sandbox escape).

## Conditions for Revisit
This architectural separation will not be revisited for mathematical risk calculations. If qualitative risk evaluations (e.g., ESG qualitative impact assessment) are introduced, they may be handled via dedicated RAG policies while keeping quantitative limits deterministic.
