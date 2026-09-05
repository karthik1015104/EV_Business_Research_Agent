# ============================================================
# app.py
# AI Business Research & Decision Support Agent
# Indian Electric Passenger-Vehicle Market
# ============================================================

import streamlit as st
import os
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Business Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #9aa0a6;
        margin-bottom: 2rem;
    }

    .tool-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: rgba(40, 167, 69, 0.18);
        border: 1px solid rgba(40, 167, 69, 0.35);
        margin-bottom: 0.7rem;
    }

    .tool-card-active {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: rgba(40, 167, 69, 0.25);
        border: 1px solid rgba(40, 167, 69, 0.55);
        margin-bottom: 0.7rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        text-align: center;
    }

    .small-text {
        color: #9aa0a6;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# SESSION STATE
# ============================================================

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# LOAD AGENT
# ============================================================

@st.cache_resource
def load_agent():

    try:

        from src.agent import create_business_agent

        agent = create_business_agent()

        return agent

    except Exception as e:

        raise RuntimeError(
            f"Unable to initialize the agent.\n\n{e}"
        )


# ============================================================
# EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(response):

    if response is None:
        return "No response was generated."

    # LangChain agent response
    if isinstance(response, dict):

        messages = response.get("messages", [])

        # Search backwards for final AI response
        for message in reversed(messages):

            message_type = getattr(
                message,
                "type",
                ""
            )

            if message_type == "ai":

                content = getattr(
                    message,
                    "content",
                    None
                )

                if isinstance(content, str) and content.strip():
                    return content.strip()

        # Fallback
        if "output" in response:
            return str(response["output"])

    # Direct string
    if isinstance(response, str):
        return response.strip()

    return str(response)


# ============================================================
# EXTRACT TOOL USAGE
# ============================================================

def extract_tools_used(response):

    tools_used = []

    if not isinstance(response, dict):
        return tools_used

    messages = response.get("messages", [])

    for message in messages:

        tool_calls = getattr(
            message,
            "tool_calls",
            []
        )

        if tool_calls:

            for call in tool_calls:

                name = call.get(
                    "name"
                )

                if name and name not in tools_used:
                    tools_used.append(name)

    return tools_used


# ============================================================
# RUN AGENT
# ============================================================

def run_business_analysis(
    agent,
    question
):

    # Important business-analysis instructions.
    # These are deliberately included at application level
    # so the Streamlit demo follows the project's data rules.

    enhanced_question = f"""
You are answering a business research question about
India's electric passenger-vehicle market.

USER QUESTION:
{question}

IMPORTANT DATA RULES:

1. Use the analytical tools for numerical claims.
2. Use the research tool for company strategy,
   policy, technology, infrastructure and qualitative evidence.
3. Never invent data.
4. Never calculate or describe a company's reported EV
   sales as its share of India's total EV registrations
   unless the sales metric and market metric are explicitly
   comparable.
5. Tata Motors' reported EV sales may include
   International Business + Domestic sales.
6. Mahindra and Hyundai figures may use different
   reporting definitions.
7. Therefore, do NOT calculate percentages such as:
   Tata EV sales / Indian EV registrations.
8. Do not call the three-company dataset "Indian market share."
9. Clearly state important comparability limitations.
10. Separate factual evidence from strategic inference
    and recommendations.
11. For research evidence, cite:
    (Source: document name, p. X)

ANSWER FORMAT:

1. Market assessment
2. Competitive position
3. Strategic implications
4. Key opportunities
5. Key risk / limitation
6. Final recommendation

Keep the answer concise and business-oriented.
Use tables or bullet points where useful.
Do not unnecessarily repeat the question.
Finish the recommendation completely.
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": enhanced_question
                }
            ]
        }
    )

    answer = extract_final_answer(response)

    tools_used = extract_tools_used(response)

    return {
        "answer": answer,
        "tools_used": tools_used,
        "raw_response": response
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">⚡ EV Business Research Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered business research and decision support for
    India's electric passenger-vehicle market
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Agent Architecture")

    st.markdown(
        """
        This application uses a **single AI agent** that
        dynamically selects among three specialized tools.
        """
    )

    st.divider()

    st.subheader("Available Tools")

    st.markdown(
        """
        📊 **Market Analytics**

        India's EV registrations, penetration,
        YoY growth and CAGR.
        """
    )

    st.markdown(
        """
        🏢 **Competitor Analytics**

        Reported EV sales for Tata Motors,
        Mahindra and Hyundai.
        """
    )

    st.markdown(
        """
        📚 **Research / RAG**

        Company strategy, industry trends,
        policy, technology and infrastructure.
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**OpenAI GPT-OSS 120B**")
    st.write("Provider: Groq")
    st.write("Reasoning: Low")

    st.divider()

    st.caption(
        "Business research agent • India EV market"
    )


# ============================================================
# INITIALIZE AGENT
# ============================================================

if st.session_state.agent is None:

    with st.spinner(
        "Initializing research agent..."
    ):

        try:

            st.session_state.agent = load_agent()

            st.session_state.agent_error = None

        except Exception as e:

            st.session_state.agent_error = str(e)


# ============================================================
# INITIALIZATION ERROR
# ============================================================

if st.session_state.agent_error:

    st.error(
        "Unable to initialize the agent."
    )

    st.code(
        st.session_state.agent_error
    )

    st.info(
        """
        Check the following:

        1. Your `.env` file contains GROQ_API_KEY.
        2. The `.env` file is in the project root.
        3. `chroma_db` exists in the project root.
        4. All packages are installed in the active `.venv`.
        """
    )

    st.stop()


# ============================================================
# AGENT STATUS
# ============================================================

st.success(
    "Agent initialized successfully"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("Ask a Business Question")

st.markdown(
    "Examples:"
)

example_columns = st.columns(3)

example_questions = [
    (
        "📈 Market Growth",
        "How has India's electric passenger-car market evolved from FY2022 to FY2025?"
    ),
    (
        "🏢 Tata Strategy",
        "What are Tata Motors' key EV strategy priorities?"
    ),
    (
        "🎯 Competitive Analysis",
        "Assess Tata Motors' competitive position in India's electric passenger-car market and recommend two strategic priorities."
    )
]

for column, (title, question) in zip(
    example_columns,
    example_questions
):

    with column:

        if st.button(
            title,
            use_container_width=True
        ):

            st.session_state.selected_question = question


# ============================================================
# QUESTION INPUT
# ============================================================

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


question = st.text_area(
    "Business question",
    value=st.session_state.selected_question,
    height=120,
    placeholder=(
        "Example: How should Tata Motors strengthen "
        "its position in India's EV market?"
    )
)


# ============================================================
# RUN BUTTON
# ============================================================

run_button = st.button(
    "🚀 Run Business Analysis",
    type="primary",
    use_container_width=True
)


# ============================================================
# EXECUTE ANALYSIS
# ============================================================

if run_button:

    if not question.strip():

        st.warning(
            "Please enter a business question."
        )

        st.stop()

    with st.spinner(
        "Agent is researching and analyzing..."
    ):

        try:

            result = run_business_analysis(
                st.session_state.agent,
                question
            )

            st.session_state.last_result = result

        except Exception as e:

            st.error(
                "The agent encountered an error."
            )

            st.exception(e)

            st.stop()


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.last_result:

    result = st.session_state.last_result

    answer = result["answer"]

    tools_used = result["tools_used"]

    st.divider()

    st.header("Business Analysis")

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    st.markdown(answer)

    # --------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------

    st.divider()

    st.header("Agent Tool Selection")

    tool_columns = st.columns(3)

    tool_information = [
        (
            "📊",
            "Market Analytics",
            "market_analytics_tool"
        ),
        (
            "🏢",
            "Competitor Analytics",
            "competitor_analytics_tool"
        ),
        (
            "📚",
            "Research / RAG",
            "research_tool"
        )
    ]

    for column, (
        icon,
        display_name,
        tool_name
    ) in zip(
        tool_columns,
        tool_information
    ):

        with column:

            if tool_name in tools_used:

                st.markdown(
                    f"""
                    <div class="tool-card-active">
                        <strong>{icon} {display_name}</strong>
                        <br>
                        <span class="small-text">
                        ✓ Selected by agent
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="tool-card">
                        <strong>{icon} {display_name}</strong>
                        <br>
                        <span class="small-text">
                        Not required for this question
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # --------------------------------------------------------
    # Execution details
    # --------------------------------------------------------

    with st.expander(
        "View agent execution details"
    ):

        st.write(
            "**Tools selected by the agent:**"
        )

        if tools_used:

            for tool in tools_used:
                st.write(
                    f"✓ `{tool}`"
                )

        else:

            st.write(
                "No tool calls detected."
            )

        st.write(
            f"**Number of tools used:** "
            f"{len(tools_used)}"
        )

        st.write(
            "**Agent type:** Single-agent architecture"
        )

        st.write(
            "**Model:** openai/gpt-oss-120b"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EV Business Research & Decision Support Agent | "
    "Market Analytics + Competitor Analytics + Research/RAG"
)