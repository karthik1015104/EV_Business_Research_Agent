# ============================================================
# EV Business Research Agent
# ============================================================

"""
Single-agent, multi-tool business research system for
India's electric passenger-vehicle market.

Tools:
1. Market Analytics
2. Competitor Analytics
3. Research / RAG
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_FILE,
    override=True
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from analytics import (
    market_analytics,
    competitor_analytics
)

from rag import (
    load_vector_database,
    research_rag_tool
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

MAX_OUTPUT_TOKENS = 800

REASONING_EFFORT = "low"


# ============================================================
# MARKET ANALYTICS TOOL
# ============================================================

@tool
def market_analytics_tool(
    metric: Literal[
        "ev_registrations",
        "total_cars",
        "ev_penetration",
        "yoy_growth",
        "cagr",
        "summary"
    ],
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    """
    Analyze India's electric passenger-car market.

    Use this tool for ALL numerical market questions,
    including:
    - EV registrations
    - total passenger-car registrations
    - EV penetration
    - YoY growth
    - CAGR
    - market trends

    IMPORTANT:
    Never estimate or calculate market numbers manually
    when the required metric is available through this tool.

    For yearly metrics, provide year.

    For CAGR, provide start_year and end_year.

    For period summaries, provide start_year and end_year.
    """

    return market_analytics(
        metric=metric,
        year=year,
        start_year=start_year,
        end_year=end_year
    )


# ============================================================
# COMPETITOR ANALYTICS TOOL
# ============================================================

@tool
def competitor_analytics_tool(
    metric: Literal[
        "ev_sales",
        "yoy_growth",
        "compare",
        "trend",
        "summary"
    ],
    company: Optional[str] = None,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    """
    Analyze reported EV sales for Tata Motors,
    Mahindra and Hyundai.

    Use this tool for ALL numerical competitor questions,
    including:
    - company EV sales
    - competitor comparisons
    - company sales trends
    - company-level YoY growth
    - latest available data

    IMPORTANT:
    Never invent, estimate or manually calculate company
    sales figures when the required data is available
    through this tool.

    Company metrics may use different definitions.

    Missing data does not mean zero sales.
    """

    return competitor_analytics(
        metric=metric,
        company=company,
        year=year,
        start_year=start_year,
        end_year=end_year
    )


# ============================================================
# RESEARCH / RAG TOOL
# ============================================================

@tool
def research_tool(
    query: str,
    k: int = 5
):
    """
    Search the EV business research document collection.

    Use for qualitative and document-supported information
    about:
    - company strategy
    - competitive position
    - products
    - technology
    - manufacturing
    - localization
    - charging infrastructure
    - government policy
    - industry trends
    - risks
    - opportunities
    - strategic priorities

    Results contain document names and page numbers.

    Numerical claims from research documents may be quoted
    only when they are explicitly present in the retrieved
    evidence.
    """

    if not query or not query.strip():
        raise ValueError(
            "Research query cannot be empty."
        )

    k = max(
        1,
        min(int(k), 8)
    )

    vector_db = load_vector_database()

    return research_rag_tool(
        query=query.strip(),
        k=k,
        vector_db=vector_db
    )


# ============================================================
# TOOL LIST
# ============================================================

TOOLS = [
    market_analytics_tool,
    competitor_analytics_tool,
    research_tool
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a business research and decision-support agent
focused on India's electric passenger-vehicle market.

Your job is to answer business questions using the
available analytical and research tools.

AVAILABLE TOOLS
----------------

1. market_analytics_tool

Use for numerical Indian EV market information:
- EV registrations
- total passenger-car registrations
- EV penetration
- YoY growth
- CAGR
- market trends

2. competitor_analytics_tool

Use for numerical reported EV sales of:
- Tata Motors
- Mahindra
- Hyundai

Use for:
- company EV sales
- competitor comparisons
- company trends
- company-level YoY growth

3. research_tool

Use for document-backed qualitative evidence about:
- company strategy
- competitive position
- product plans
- technology
- manufacturing
- localization
- charging infrastructure
- government policy
- industry trends
- risks
- opportunities
- strategic priorities


============================================================
NUMERICAL DATA RULE
============================================================

ALL numerical market and competitor values MUST come from
the analytics tools.

Do not estimate, guess, remember or manually invent numbers.

Use:
- market_analytics_tool for market-level numbers
- competitor_analytics_tool for company sales numbers

If the required numerical value is unavailable, state:

"Data not available in the current dataset."

Do not fill missing values with zero.

If a numerical fact appears in research documents, only use
it when it is explicitly returned by research_tool and clearly
identify it as research evidence.


============================================================
NO INVENTED FACTS
============================================================

Use tools instead of guessing.

Never invent:
- numbers
- market statistics
- company facts
- products
- partnerships
- strategies
- management plans
- policy details
- sources
- page numbers

If evidence is insufficient, say so.

Do not make causal claims unless the retrieved evidence
explicitly supports the causal relationship.

Do not turn a reasonable assumption into a stated fact.


============================================================
CRITICAL MARKET-SHARE RULE
============================================================

NEVER calculate Tata Motors' reported EV sales as a percentage
of India's total EV registrations.

NEVER describe Tata's reported EV sales as its share of the
entire Indian EV market.

Tata's FY2025 reported EV sales include International Business
(IB) + Domestic sales.

The market dataset represents Indian EV registrations.

These are different measurement bases.

Therefore, do not divide Tata's reported EV sales by India's
EV registrations to produce an Indian market-share percentage.

Similarly, do not describe the Tata/Mahindra/Hyundai dataset
as representing the entire Indian EV market.


============================================================
COMPETITOR COMPARABILITY
============================================================

Competitor sales figures may use different definitions.

Always preserve the metric definition returned by
competitor_analytics_tool.

When comparing companies:

- report the values
- state the metric definitions when relevant
- mention comparability limitations
- do not imply a precise market share unless the definitions
  are genuinely compatible

Missing company-year data is NOT zero.


============================================================
WHEN TO USE RESEARCH
============================================================

Use research_tool whenever the question involves:

- company strategy
- competitive positioning
- strategic opportunities
- strategic risks
- management priorities
- recommendations
- qualitative business evidence

For strategic questions, combine analytics and research
when numerical and qualitative evidence are both required.


============================================================
MARKET PERIODS
============================================================

When the user specifies a period such as FY2022-FY2025,
use exactly that period.

Do not substitute an unrelated period.

For CAGR questions, use the specified start and end years.

For market-evolution questions, use relevant metrics from
the requested period.


============================================================
RESEARCH CITATIONS
============================================================

When using evidence from research_tool, preserve the source
and page information.

Use this exact citation format:

(Source: document name, p. X)

Only cite page numbers actually returned by research_tool.

NEVER invent page numbers.

If a claim comes from analytics tools, identify the relevant
analytics tool when useful.

Do not create fake citations or source names.


============================================================
FACT vs INFERENCE vs RECOMMENDATION
============================================================

Clearly distinguish between evidence and interpretation.

Use:

FACT:
A statement directly supported by a tool result.

AGENT INFERENCE:
A reasonable interpretation derived from the available
evidence. It must not introduce unsupported facts.

RECOMMENDATION:
A proposed business action based on the evidence and
agent inference.


============================================================
RECOMMENDATION RULE
============================================================

Whenever giving a recommendation, explicitly label the
recommendation section:

**Agent inference / recommendation**

Do not present recommendations as facts.

Use this structure when appropriate:

Evidence:
- 1–3 important evidence points.

Agent inference:
- What the evidence means strategically.

**Agent inference / recommendation:**
- The recommended business action.

Limitation:
- Important uncertainty or data limitation.


============================================================
OUTPUT STYLE
============================================================

Keep the final answer concise and business-oriented.

Target approximately 500–700 tokens and never intentionally
write a long report.

Prefer:
- short headings
- concise bullets
- compact tables when useful

Avoid:
- unnecessary repetition
- long explanations
- repeating the same evidence
- unsupported assumptions

For simple factual questions, answer directly rather than
using the full business-analysis structure.

For strategic questions, prioritize:
1. Market assessment
2. Competitive position
3. Evidence
4. Agent inference
5. Recommendation
6. Key limitation

Accuracy and evidence quality are more important than length.
"""


# ============================================================
# CREATE BUSINESS AGENT
# ============================================================

def create_business_agent():
    """
    Create and return the single EV Business Research Agent.
    """

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not found.\n\n"
            f"Expected .env file at:\n{ENV_FILE}\n\n"
            "Make sure the file contains:\n"
            "GROQ_API_KEY=your_api_key"
        )

    # --------------------------------------------------------
    # INITIALIZE LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        max_retries=2,
        api_key=groq_api_key
    )

    # --------------------------------------------------------
    # CREATE SINGLE AGENT
    # --------------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


# ============================================================
# AGENT INFORMATION
# ============================================================

def get_agent_info():
    """
    Return the agent configuration.
    """

    return {
        "model": MODEL_NAME,
        "provider": "Groq",
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "tool_count": len(TOOLS),
        "tools": [
            tool_item.name
            for tool_item in TOOLS
        ]
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EV BUSINESS RESEARCH AGENT")
    print("=" * 70)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nEnvironment file:\n{ENV_FILE}"
    )

    print(
        f"\nEnvironment file exists: "
        f"{ENV_FILE.exists()}"
    )

    print(
        "\nGroq API key: "
        f"{'FOUND' if os.getenv('GROQ_API_KEY') else 'NOT FOUND'}"
    )

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Reasoning effort: "
        f"{REASONING_EFFORT}"
    )

    print(
        f"Maximum output tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )

    print("\nTools:")

    for tool_item in TOOLS:
        print(
            f"  ✓ {tool_item.name}"
        )

    print("\n" + "=" * 70)

    try:

        test_agent = create_business_agent()

        print(
            "\n✓ Agent initialized successfully."
        )

    except Exception as e:

        print(
            f"\n✗ Agent initialization failed:\n{e}"
        )