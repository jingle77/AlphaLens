"""
Prompt templates and analysis options for AlphaLens.

This module converts deterministic evidence packages into structured prompts
for OpenAI text generation.

Important:
AlphaLens does not use semantic RAG in v1. The selected analysis type maps to
a deterministic evidence recipe, and the LLM synthesizes the provided structured
financial evidence into a balanced research-style narrative.
"""

from __future__ import annotations

import json
from typing import Any

from src.evidence import (
    ANALYSIS_BEAR,
    ANALYSIS_BULL,
    ANALYSIS_IMPROVEMENT,
    ANALYSIS_OVERALL,
    ANALYSIS_QUALITY,
    SUPPORTED_ANALYSIS_TYPES,
)


ANALYSIS_OPTIONS = list(SUPPORTED_ANALYSIS_TYPES)


EVIDENCE_RECIPES: dict[str, dict[str, Any]] = {
    ANALYSIS_OVERALL: {
        "goal": "Evaluate the overall evidence-weighted outperformance thesis.",
        "primary_evidence_sections": [
            "company_overview",
            "financial_snapshot",
            "market_snapshot",
            "recent_news",
            "deterministic_signal_summary",
        ],
        "expected_balance": "balanced",
        "recommended_output_focus": [
            "outperformance thesis",
            "supportive evidence",
            "weakening evidence",
            "key uncertainty",
            "evidence-weighted conclusion",
        ],
    },
    ANALYSIS_BULL: {
        "goal": "Develop the strongest reasonable bull case for outperformance.",
        "primary_evidence_sections": [
            "company_overview",
            "financial_snapshot",
            "market_snapshot",
            "recent_news",
            "primary_supporting_signals",
            "important_caveats",
        ],
        "expected_balance": "constructive but caveated",
        "recommended_output_focus": [
            "strongest supporting evidence",
            "why the setup could support outperformance",
            "conditions needed for the bull case to hold",
            "important caveats",
        ],
    },
    ANALYSIS_BEAR: {
        "goal": "Develop the strongest reasonable bear case against outperformance.",
        "primary_evidence_sections": [
            "company_overview",
            "financial_snapshot",
            "market_snapshot",
            "recent_news",
            "primary_weakening_signals",
            "offsetting_supportive_signals",
            "unresolved_uncertainties",
        ],
        "expected_balance": "skeptical but fair",
        "recommended_output_focus": [
            "evidence weakening the thesis",
            "risk and quality concerns",
            "relative performance concerns",
            "offsetting evidence",
            "what would change the bear case",
        ],
    },
    ANALYSIS_QUALITY: {
        "goal": "Assess the company's financial quality and how it affects the outperformance thesis.",
        "primary_evidence_sections": [
            "company_overview",
            "financial_snapshot",
            "deterministic_signal_summary",
        ],
        "expected_balance": "fundamental-quality focused",
        "recommended_output_focus": [
            "growth quality",
            "margin quality",
            "cash generation",
            "balance sheet strength",
            "quality-related uncertainty",
        ],
    },
    ANALYSIS_IMPROVEMENT: {
        "goal": "Identify what would need to improve for the outperformance thesis to become more credible.",
        "primary_evidence_sections": [
            "company_overview",
            "financial_snapshot",
            "market_snapshot",
            "current_weakening_signals",
            "current_uncertainties",
            "existing_supportive_signals",
        ],
        "expected_balance": "diagnostic and improvement-oriented",
        "recommended_output_focus": [
            "specific weak points",
            "needed financial improvements",
            "needed market-performance improvements",
            "needed risk improvements",
            "follow-up research questions",
        ],
    },
}


SYSTEM_PROMPT = """
You are AlphaLens, an equity research assistant.

Your task is to synthesize structured financial evidence into balanced,
research-style analysis about whether a selected stock has a credible setup to
outperform the S&P 500 benchmark.

You must follow these rules:

1. Do not provide financial advice.
2. Do not tell the user to buy, sell, hold, short, or avoid a stock.
3. Do not make hard predictions.
4. Do not claim that a stock will or will not outperform.
5. Use evidence-weighted language, such as:
   - "the evidence supports"
   - "the evidence weakens"
   - "the outperformance thesis"
   - "key uncertainty"
   - "the setup appears"
   - "the case is mixed"
6. Interpret only the evidence provided.
7. Do not invent missing financial data.
8. If evidence is missing or incomplete, explicitly identify the uncertainty.
9. Weigh conflicting signals rather than forcing a one-sided conclusion.
10. Keep the tone professional, analytical, and suitable for an equity research note.

Output format:
- Write 3 to 5 concise paragraphs.
- Use clear paragraph breaks.
- Do not use bullet points unless the requested analysis type specifically benefits from a short list.
- End with a balanced evidence-weighted conclusion.
""".strip()


def validate_analysis_type(analysis_type: str) -> None:
    """
    Validate that an analysis type is supported.

    Args:
        analysis_type: Selected analysis type.

    Raises:
        ValueError: If analysis_type is unsupported.
    """

    if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported analysis_type: {analysis_type}. "
            f"Expected one of: {SUPPORTED_ANALYSIS_TYPES}"
        )


def get_analysis_options() -> list[str]:
    """
    Return supported analysis options for the Streamlit dropdown.

    Returns:
        List of analysis option strings.
    """

    return ANALYSIS_OPTIONS.copy()


def get_evidence_recipe(analysis_type: str) -> dict[str, Any]:
    """
    Return the deterministic evidence recipe for an analysis type.

    Args:
        analysis_type: Selected analysis type.

    Returns:
        Evidence recipe dictionary.
    """

    validate_analysis_type(analysis_type)

    return EVIDENCE_RECIPES[analysis_type]


def evidence_to_json(evidence_package: dict[str, Any]) -> str:
    """
    Convert an evidence package to stable, readable JSON.

    Args:
        evidence_package: Dictionary returned by src.evidence.build_evidence_package.

    Returns:
        Pretty-printed JSON string.
    """

    return json.dumps(
        evidence_package,
        indent=2,
        sort_keys=True,
        default=str,
    )


def build_user_prompt(
    evidence_package: dict[str, Any],
) -> str:
    """
    Build the user prompt for OpenAI synthesis.

    Args:
        evidence_package: Dictionary returned by src.evidence.build_evidence_package.

    Returns:
        User prompt string.
    """

    analysis_type = evidence_package.get("analysis_type")

    if not analysis_type:
        raise ValueError("evidence_package must include an 'analysis_type' field.")

    validate_analysis_type(analysis_type)

    recipe = get_evidence_recipe(analysis_type)
    evidence_json = evidence_to_json(evidence_package)
    recommended_focus_json = json.dumps(
        recipe["recommended_output_focus"],
        indent=2,
    )

    prompt = f"""
Analysis type:
{analysis_type}

Research goal:
{recipe["goal"]}

Expected analytical posture:
{recipe["expected_balance"]}

Recommended output focus:
{recommended_focus_json}

Evidence routing note:
The evidence below was selected using deterministic evidence routing from the
user's dropdown choice. This is not embedding-based or semantic RAG.

Structured evidence JSON:
{evidence_json}

Write the AlphaLens analysis.

Requirements:
- Use the selected analysis type as the framing.
- Discuss the evidence that supports the outperformance thesis.
- Discuss the evidence that weakens the outperformance thesis.
- Identify at least one key uncertainty when applicable.
- Avoid financial advice or recommendation language.
- Do not use "buy", "sell", "hold", "price target", or "guaranteed".
- Do not calculate new metrics; use the metrics provided.
- If the evidence is mixed, say so directly.
""".strip()

    return prompt

def build_prompt_messages(
    evidence_package: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build OpenAI-compatible prompt messages.

    Args:
        evidence_package: Dictionary returned by src.evidence.build_evidence_package.

    Returns:
        List of message dictionaries.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(evidence_package),
        },
    ]


def build_prompt_payload(
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete prompt payload for the LLM client.

    Args:
        evidence_package: Dictionary returned by src.evidence.build_evidence_package.

    Returns:
        Prompt payload dictionary.
    """

    analysis_type = evidence_package.get("analysis_type")

    if not analysis_type:
        raise ValueError("evidence_package must include an 'analysis_type' field.")

    validate_analysis_type(analysis_type)

    return {
        "analysis_type": analysis_type,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(evidence_package),
        "messages": build_prompt_messages(evidence_package),
        "evidence_recipe": get_evidence_recipe(analysis_type),
    }