"""Synthetic research documents for AlphaPilot AI RAG pipeline.

NOTICE: All documents in this module are purely synthetic mock guidelines created
strictly for demonstration, testing, and automated evaluation of AlphaPilot AI.
They DO NOT represent real financial advice, investment strategies, or actual
regulatory compliance mandates.
"""

from app.models.document import ResearchDocument


def get_sample_documents() -> list[ResearchDocument]:
    """Return a fresh list of synthetic research documents for RAG ingestion."""
    return [
        ResearchDocument(
            document_id="SYNTH-DOC-001",
            title="Synthetic Policy: Portfolio Concentration Limits & Allocation Ceilings",
            content=(
                "[SYNTHETIC TEST DOCUMENT - NOT FINANCIAL ADVICE]\n"
                "Internal Policy Framework: Synthetic Portfolio Concentration Limits\n\n"
                "1. Single-Security Allocation Ceiling\n"
                "Under standard risk mandates, no individual equity holding or single-issuer "
                "position may exceed 20.0% of total portfolio market value. For aggressive growth "
                "or concentrated strategy mandates, a strict absolute cap of 40.0% applies. "
                "If market price appreciation drives a position over its threshold, portfolio "
                "managers are granted a three-business-day grace period to initiate orderly rebalancing.\n\n"
                "2. Sector Exposure Boundaries\n"
                "Aggregate market exposure to any single GICS economic sector (such as Technology, "
                "Healthcare, Financials, or Energy) must remain strictly at or below 40.0% of the total "
                "portfolio valuation. Dual-industry conglomerates must be classified based on primary "
                "trailing twelve-month GAAP revenue source.\n\n"
                "3. Remediation Protocols for Concentration Breaches\n"
                "When automated risk scans flag an unauthorized concentration breach, an alert is "
                "dispatched to the portfolio manager and compliance desk. The position must either be "
                "trimmed back below the target threshold or an approved temporary variance must be "
                "obtained from the Risk Oversight Committee within five business days.\n\n"
                "DISCLAIMER: This text is entirely synthetic and designed solely for testing "
                "AlphaPilot AI automated analysis. It does not constitute real-world financial advice."
            ),
            metadata={
                "source": "synthetic_policy_manual",
                "category": "concentration_limits",
                "version": "1.0",
                "synthetic": True,
                "author": "Synthetic Risk Committee",
            },
        ),
        ResearchDocument(
            document_id="SYNTH-DOC-002",
            title="Synthetic Policy: Cash Reserve and Liquidity Standards",
            content=(
                "[SYNTHETIC TEST DOCUMENT - NOT FINANCIAL ADVICE]\n"
                "Internal Operational Standard: Synthetic Cash Reserve and Liquidity Requirements\n\n"
                "1. Minimum Cash Reserve Floor\n"
                "Every active portfolio account must maintain an unencumbered cash balance of at least "
                "5.0% of total portfolio market value. This liquid reserve is mandatory to accommodate "
                "pending trade settlements, scheduled client distributions, and market volatility buffers.\n\n"
                "2. Cash Deployment and Staged Execution\n"
                "To mitigate slippage and market-impact risk, deployments of accumulated cash that exceed "
                "15.0% of aggregate account value must be staged across a minimum of two distinct trading "
                "sessions. Direct lump-sum single-order deployment of large cash blocks is prohibited without "
                "prior desk supervisor sign-off.\n\n"
                "3. Eligible Liquidity Instruments\n"
                "Permissible cash equivalents include high-grade overnight sovereign repo, AAA-rated "
                "government money market funds, and demand deposit accounts held at designated custodian "
                "institutions with short-term ratings of A-1/P-1 or equivalent.\n\n"
                "DISCLAIMER: This text is entirely synthetic and designed solely for testing "
                "AlphaPilot AI automated analysis. It does not constitute real-world financial advice."
            ),
            metadata={
                "source": "synthetic_treasury_manual",
                "category": "cash_allocation",
                "version": "1.0",
                "synthetic": True,
                "author": "Synthetic Treasury Desk",
            },
        ),
        ResearchDocument(
            document_id="SYNTH-DOC-003",
            title="Synthetic Procedure: Risk Review Cadence and Escalation Protocol",
            content=(
                "[SYNTHETIC TEST DOCUMENT - NOT FINANCIAL ADVICE]\n"
                "Internal Operational Protocol: Synthetic Risk Review Cadence and Escalation Hierarchy\n\n"
                "1. Daily Automated Exception Screening\n"
                "Each trading day at 16:30 EST post-market close, the automated risk engine executes batch "
                "evaluations across all active portfolios. The engine verifies compliance against single-stock "
                "caps, sector ceilings, and cash reserve thresholds, publishing an exception log.\n\n"
                "2. Weekly Governance Review\n"
                "The Risk Oversight Committee convenes every Tuesday morning to review open exceptions, "
                "macro stress-test simulations, and liquidity profiles. Any portfolio running a variance "
                "exceeding 1.25x allowable concentration limits requires mandatory committee review.\n\n"
                "3. Multi-Tier Escalation Path\n"
                "Exceptions remaining unaddressed after three business days trigger Level-2 notification "
                "to the Managing Director. If an active breach persists past five business days without "
                "an approved remediation plan, the Chief Investment Officer possesses sole discretion "
                "to execute forced rebalancing liquidations.\n\n"
                "DISCLAIMER: This text is entirely synthetic and designed solely for testing "
                "AlphaPilot AI automated analysis. It does not constitute real-world financial advice."
            ),
            metadata={
                "source": "synthetic_governance_manual",
                "category": "risk_review",
                "version": "1.0",
                "synthetic": True,
                "author": "Synthetic Compliance Oversight",
            },
        ),
    ]


SAMPLE_DOCUMENTS: list[ResearchDocument] = get_sample_documents()
