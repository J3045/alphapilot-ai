"""Golden evaluation dataset for AlphaPilot AI RAG pipeline.

NOTICE: All questions, expected answers, and documents in this dataset are purely
synthetic test fixtures for demonstration and automated testing. They DO NOT
represent real financial advice or actual regulatory policies.
"""

from app.evaluation.models import EvalTestCase


def get_golden_dataset() -> list[EvalTestCase]:
    """Return the version-controlled golden evaluation dataset."""
    return [
        # --- Concentration Limits (SYNTH-DOC-001) ---
        EvalTestCase(
            case_id="EVAL-001",
            question="What is the maximum allowable single-security concentration limit under standard mandates?",
            category="concentration_limits",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-001"],
            expected_keywords=["20.0%"],
            notes="SYNTH-DOC-001 Section 1 defines 20.0% standard single-security cap.",
        ),
        EvalTestCase(
            case_id="EVAL-002",
            question="How many business days do portfolio managers have to rebalance after a price appreciation concentration breach?",
            category="concentration_limits",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-001"],
            expected_keywords=["three"],
            notes="SYNTH-DOC-001 Section 1 provides a 3-business-day grace period.",
        ),
        EvalTestCase(
            case_id="EVAL-003",
            question="What is the maximum portfolio exposure permitted for any single economic sector?",
            category="concentration_limits",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-001"],
            expected_keywords=["40.0%"],
            notes="SYNTH-DOC-001 Section 2 defines the 40.0% sector concentration ceiling.",
        ),

        # --- Cash Allocation & Liquidity (SYNTH-DOC-002) ---
        EvalTestCase(
            case_id="EVAL-004",
            question="What is the minimum cash reserve buffer that active portfolios must maintain?",
            category="cash_allocation",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-002"],
            expected_keywords=["5.0%"],
            notes="SYNTH-DOC-002 Section 1 establishes a 5.0% unencumbered cash floor.",
        ),
        EvalTestCase(
            case_id="EVAL-005",
            question="What staging rule applies when deploying accumulated cash exceeding 15 percent of account value?",
            category="cash_allocation",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-002"],
            expected_keywords=["two"],
            notes="SYNTH-DOC-002 Section 2 mandates staging across at least two trading sessions.",
        ),
        EvalTestCase(
            case_id="EVAL-006",
            question="What short-term credit rating must custodian banks possess for demand deposit accounts?",
            category="cash_allocation",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-002"],
            expected_keywords=["A-1/P-1"],
            notes="SYNTH-DOC-002 Section 3 requires short-term ratings of A-1/P-1.",
        ),

        # --- Risk Review Cadence & Escalation (SYNTH-DOC-003) ---
        EvalTestCase(
            case_id="EVAL-007",
            question="At what time each day does the automated risk engine execute batch evaluations?",
            category="risk_review",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-003"],
            expected_keywords=["16:30"],
            notes="SYNTH-DOC-003 Section 1 specifies daily automated screening at 16:30 EST.",
        ),
        EvalTestCase(
            case_id="EVAL-008",
            question="When does the Risk Oversight Committee meet and what variance level triggers mandatory review?",
            category="risk_review",
            expected_status="answered",
            expected_document_ids=["SYNTH-DOC-003"],
            expected_keywords=["Tuesday", "1.25x"],
            notes="SYNTH-DOC-003 Section 2 specifies weekly Tuesday meetings and 1.25x variance threshold.",
        ),

        # --- Unanswerable / Insufficient Evidence Questions ---
        EvalTestCase(
            case_id="EVAL-009",
            question="What is the maximum allowable leverage ratio permitted for derivative margin accounts?",
            category="unanswerable_leverage",
            expected_status="insufficient_evidence",
            expected_document_ids=[],
            expected_keywords=[],
            notes="Unanswerable: Synthetic documents contain no rules regarding margin leverage ratios.",
        ),
        EvalTestCase(
            case_id="EVAL-010",
            question="What cold storage security protocols must be followed for cryptocurrency asset custody?",
            category="unanswerable_crypto",
            expected_status="insufficient_evidence",
            expected_document_ids=[],
            expected_keywords=[],
            notes="Unanswerable: Synthetic documents contain no cryptocurrency or digital asset guidelines.",
        ),
    ]


GOLDEN_EVAL_DATASET: list[EvalTestCase] = get_golden_dataset()
