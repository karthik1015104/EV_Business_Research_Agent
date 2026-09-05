# AI Business Research & Decision Support Agent

> **An agentic AI system for business research and strategic decision support in the Indian electric passenger-vehicle market.**

This project implements a **single-agent AI system** that answers business and competitive-strategy questions by dynamically combining:

- **Structured market analytics**
- **Competitor sales analytics**
- **Research-document retrieval (RAG)**

Instead of relying on a single LLM response, the agent determines which information sources are required, invokes the appropriate tools, validates the available evidence, and synthesizes the results into a concise business-oriented analysis.

---

## 1. Project Overview

Business research often requires combining different types of information:

- Numerical market trends
- Competitor performance
- Company strategy
- Industry and policy developments
- Qualitative evidence from annual reports and investor presentations

A conventional RAG chatbot is well suited for retrieving information from documents, but it is not sufficient when a question requires both **documentary evidence and numerical analysis**.

This project addresses that problem using a **single agent with specialized tools**.

### Example question

> **"How is India's electric passenger-car market evolving, what is Tata Motors' competitive position, and what strategic opportunities follow from the available evidence?"**

The agent can independently determine that it needs:

1. Market growth data → **Market Analytics Tool**
2. Competitor sales data → **Competitor Analytics Tool**
3. Company strategy and industry evidence → **Research / RAG Tool**

It then combines the evidence into a structured business analysis.

---

# 2. System Architecture

```text
                         ┌─────────────────────┐
                         │        USER         │
                         │ Business Question   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    SINGLE AI AGENT  │
                         │                     │
                         │ Tool selection      │
                         │ Reasoning            │
                         │ Evidence synthesis   │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
   ┌──────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ Market Analytics │  │ Competitor         │  │ Research / RAG     │
   │ Tool             │  │ Analytics Tool     │  │ Tool               │
   │                  │  │                    │  │                    │
   │ EV registrations │  │ Tata Motors        │  │ Annual reports     │
   │ EV penetration   │  │ Mahindra           │  │ Investor decks     │
   │ YoY growth       │  │ Hyundai            │  │ Industry reports   │
   │ CAGR             │  │ EV sales trends    │  │ Policy documents   │
   └────────┬─────────┘  └─────────┬──────────┘  └─────────┬──────────┘
            │                      │                       │
            └──────────────────────┼───────────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Evidence Synthesis  │
                         │                     │
                         │ Facts               │
                         │ Comparisons         │
                         │ Implications        │
                         │ Agent inference     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Business Analysis   │
                         │ / Recommendation    │
                         └─────────────────────┘
```

---

# 3. Key Design Decision: Single Agent + Specialized Tools

The system intentionally uses a **single AI agent** rather than multiple collaborating agents.

The agent acts as the central decision-maker and has access to three specialized tools.

### Why a single agent?

The goal is to demonstrate that an LLM can:

- Understand a business question
- Decompose the information requirement
- Select appropriate tools
- Combine structured and unstructured evidence
- Respect data limitations
- Produce a decision-oriented response

This keeps the architecture understandable while still demonstrating meaningful **agentic behavior**.

---

# 4. Agent Tools

## 4.1 Market Analytics Tool

The Market Analytics Tool operates on a structured dataset of Indian electric passenger-car registrations.

### Capabilities

- EV registrations by year
- Total passenger-car registrations
- EV penetration
- Year-over-year EV growth
- CAGR
- Market-period summaries

### Example

```text
Question:
What was India's EV registration CAGR from FY2022 to FY2025?

Agent action:
→ market_analytics_tool
→ metric = CAGR
→ start_year = 2022
→ end_year = 2025

Result:
≈ 75.9% CAGR
```

The agent is instructed that **numerical market values must come from the analytics tool rather than being generated from memory or inference**.

---

# 5. Competitor Analytics Tool

The Competitor Analytics Tool operates on reported EV sales data for:

- Tata Motors
- Mahindra
- Hyundai

### Capabilities

- Company EV sales
- Year-over-year growth
- Competitor comparison
- Company sales trends
- Latest available reported values

### Important data-quality principle

The companies do not always report EV sales using identical definitions.

For example:

- Tata Motors' reported EV sales may include domestic and international business.
- Mahindra's reported electric 4-wheeler figures may represent a different reporting scope.
- Hyundai has limited comparable annual data in the assembled dataset.

Therefore, the system explicitly preserves **metric definitions and data-quality metadata**.

The agent is instructed **not to manufacture a market-share calculation when the underlying metrics are not directly comparable**.

---

# 6. Research / RAG Tool

The Research Tool provides qualitative evidence from a curated document corpus.

### Document categories

```text
Documents/
├── hyundai/
├── industry/
├── mahindra/
└── tata/
```

The corpus contains:

- Annual reports
- Investor presentations
- Corporate presentations
- Investor-day presentations
- Industry reports
- Government / policy documents

### Example sources

The research corpus includes documents from:

- Tata Motors
- Mahindra
- Hyundai
- International Energy Agency
- Government of India EV policy initiatives

---

# 7. RAG Pipeline

The research component follows a standard retrieval-augmented generation pipeline:

```text
PDF Documents
      │
      ▼
PDF Text Extraction
      │
      ▼
Page-level Documents
      │
      ▼
Recursive Text Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
Chroma Vector Database
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Evidence
      │
      ▼
Single AI Agent
```

### Embedding model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Vector database

```text
Chroma
```

### Chunking

The documents are split using recursive character-based chunking with overlap to preserve contextual continuity between chunks.

Document metadata is preserved during chunking, including:

```text
category
file_name
page
source
```

This enables the system to provide page-level research provenance.

---

# 8. Evidence & Citation Handling

One of the main design goals of the project is **traceability**.

Research-derived statements are expected to preserve their document provenance:

```text
(Source: tata-motor-IAR-2024-25.pdf, p. 36)
```

This allows a user to distinguish:

- Information retrieved from company documents
- Numerical information returned by analytics tools
- Strategic conclusions inferred by the agent

The agent is explicitly instructed not to invent missing evidence.

When the available sources do not establish a claim, the system should indicate that the claim is not established by the available data.

---

# 9. Agent Reasoning Workflow

For a business question, the agent follows this general process:

```text
1. Understand the business question
                ↓
2. Identify required information
                ↓
3. Select appropriate tools
                ↓
4. Retrieve numerical / documentary evidence
                ↓
5. Check definitions and limitations
                ↓
6. Synthesize evidence
                ↓
7. Separate facts from inference
                ↓
8. Produce concise business recommendation
```

For example:

```text
User:
"Assess Tata Motors' position in India's EV market
and identify strategic opportunities."

                    ↓

        ┌───────────────────────┐
        │    Single AI Agent    │
        └───────────┬───────────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
      Market    Competitor     Research
     Analytics  Analytics       RAG
        │           │            │
        └───────────┼────────────┘
                    ▼
            Evidence synthesis
                    │
                    ▼
          Strategic implications
                    │
                    ▼
          Agent inference / advice
```

---

# 10. Business Output

The application presents the agent's output as a structured business analysis rather than a generic chatbot response.

Typical sections include:

### Market assessment

- Market growth
- EV penetration
- Market scale

### Competitive position

- Reported company EV sales
- Metric definitions
- Data comparability

### Strategic implications

- Evidence-based observations
- Business implications

### Key opportunities

- Evidence
- Why the opportunity matters

### Key risks / limitations

- Data limitations
- Market uncertainties
- Reporting differences

### Final recommendation

Recommendations are explicitly labelled:

> **Agent inference:** ...

This prevents inferred strategic advice from being confused with factual information contained in the source documents.

---

# 11. Example Analysis

A representative analysis can combine:

```text
Market Analytics
────────────────
FY2022 EV registrations : 21,194
FY2025 EV registrations : 115,315
FY2022–FY2025 CAGR      : ≈75.9%

        +

Competitor Analytics
─────────────────────
Tata Motors : 64,276
Mahindra    : 14,183
Hyundai     : 3,969

        +

Research / RAG
───────────────
Tata EV strategy
Localization
Charging ecosystem
Product expansion
Corporate EV initiatives

        ↓

Strategic synthesis

        ↓

Agent inference
```

The exact numerical values shown by the application are always retrieved from the project's structured analytics datasets rather than manually generated by the LLM.

---

# 12. Data Sources

The project uses a combination of structured datasets and primary research documents.

### Market data

Indian vehicle registration data was derived from the Vahan / Transport Data Commons dataset and filtered for electric passenger cars.

### Company data

Competitor EV sales data was compiled from official company disclosures including:

- Tata Motors sales releases
- Mahindra annual reports
- Hyundai investor presentations

### Research documents

The document corpus contains company annual reports, investor presentations and relevant industry / policy documents.

The project prioritizes **primary sources** wherever possible.

---

# 13. Data Quality & Limitations

An important part of the system is recognizing where the data cannot support a conclusion.

### 13.1 Different reporting definitions

Company-reported EV sales are not necessarily identical metrics.

For example, Tata Motors' reported figure can include international and domestic EV deliveries, while another company's figure may represent domestic electric 4-wheeler sales.

Therefore:

> Company-reported EV sales should not automatically be interpreted as directly comparable Indian market shares.

---

### 13.2 Missing data

Missing company data is treated as:

```text
Missing ≠ Zero
```

If a company did not have a comparable reported value for a particular year, the system does not replace it with zero.

---

### 13.3 Market-share calculations

The system does **not** calculate or claim market share when the underlying definitions are incompatible.

The three-company competitor dataset also does not represent the entire Indian EV market.

---

### 13.4 RAG limitations

Document retrieval quality depends on:

- PDF text extraction
- Document structure
- Chunking strategy
- Embedding quality
- Retrieval relevance

Tables and highly visual PDF content may not always be perfectly represented by text extraction.

---

### 13.5 LLM limitations

The LLM can still make reasoning errors.

The system therefore uses tool constraints and explicit instructions to:

- Ground numerical claims in analytics tools
- Ground qualitative claims in retrieved evidence
- Preserve citations
- Avoid unsupported market-share claims
- Label strategic recommendations as agent inference

---

# 14. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | OpenAI GPT-OSS 120B via Groq |
| Agent framework | LangChain |
| Agent architecture | Single tool-using agent |
| RAG framework | LangChain |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | Chroma |
| PDF processing | PyPDF |
| Data analysis | Pandas / NumPy |
| Frontend | Streamlit |
| Configuration | python-dotenv |
| Development | Jupyter / VS Code |

---

# 15. Project Structure

```text
EV_Business_Research_Agent/
│
├── Data/
│   ├── raw/
│   └── processed/
│       ├── india_ev_market.csv
│       └── competitor_ev_sales.csv
│
├── Documents/
│   ├── hyundai/
│   ├── industry/
│   ├── mahindra/
│   └── tata/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_agent_development.ipynb
│
├── src/
│   ├── analytics.py
│   ├── rag.py
│   └── agent.py
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# 16. Running the Project Locally

## Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EV_Business_Research_Agent
```

---

## Step 2 — Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

## Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 4 — Configure the API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

**Never commit the `.env` file to GitHub.**

It should be included in `.gitignore`.

---

## Step 5 — Build the vector database

The first run may require creating the Chroma vector database from the research documents.

The RAG pipeline:

```text
Documents
    ↓
PDF extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Chroma
```

Once the vector database has been created, the application can load the stored collection.

---

## Step 6 — Run the Streamlit application

```powershell
streamlit run app.py
```

The application will open in the browser.

---

# 17. Example Questions

The agent can answer questions such as:

### Market analysis

```text
How has India's electric passenger-car market evolved
from FY2022 to FY2025?
```

### Growth analysis

```text
What was the CAGR of Indian EV passenger-car
registrations between FY2022 and FY2025?
```

### Competitor analysis

```text
Compare the reported FY2025 EV sales of Tata Motors,
Mahindra and Hyundai.
```

### Company research

```text
What are Tata Motors' key EV strategy priorities?
```

### Multi-tool business analysis

```text
How is India's electric passenger-car market evolving,
and what does this mean for Tata Motors' competitive position?
```

### Strategic decision support

```text
What strategic opportunities and risks should a new
entrant consider in India's electric passenger-car market?
```

---

# 18. Evaluation

The agent was tested for both **tool routing** and **multi-source reasoning**.

### Individual tool-routing tests

| Test | Task | Expected Tool |
|---|---|---|
| E1 | Market growth / CAGR | Market Analytics |
| E2 | Competitor EV sales | Competitor Analytics |
| E3 | Tata EV strategy | Research / RAG |

### Multi-tool test

A business question requiring market, competitor and qualitative research evidence was used to verify that the agent could select:

```text
✓ Market Analytics
✓ Competitor Analytics
✓ Research / RAG
```

The evaluation therefore tests more than whether the LLM can answer a question — it tests whether it can **select and combine the appropriate information sources**.

---

# 19. Engineering Considerations

Several design decisions were made specifically to improve reliability.

### Tool-grounded numerical reasoning

The LLM does not act as the source of numerical market data.

Instead:

```text
Question
   ↓
Agent
   ↓
Analytics Tool
   ↓
Structured numerical result
   ↓
LLM synthesis
```

This reduces the risk of fabricated numerical values.

---

### Provenance-aware retrieval

Each retrieved research chunk retains:

```text
document
page
category
source
```

This enables source-aware answers instead of anonymous retrieved text.

---

### Explicit data limitations

The agent is instructed to recognize:

```text
Different metric definitions
        +
Missing company data
        +
Incomplete competitor coverage
        ↓
Avoid unsupported conclusions
```

This is particularly important for business-analysis systems, where presenting an incorrect market-share number can lead to misleading decisions.

---

### Separation of fact and inference

The system distinguishes:

```text
FACT
↓
Evidence retrieved from data / documents

IMPLICATION
↓
Interpretation of the evidence

AGENT INFERENCE
↓
Strategic recommendation
```

This separation makes the generated analysis more transparent.

---

# 20. Why This Project Is Agentic AI

The project is not simply a RAG chatbot.

A conventional RAG system typically follows:

```text
Question
   ↓
Retrieve documents
   ↓
Generate answer
```

This project follows:

```text
Business Question
       ↓
    AI Agent
       ↓
Determine information requirements
       ↓
Select appropriate tools
       ↓
┌────────────┬──────────────┬─────────────┐
│ Market     │ Competitor   │ Research    │
│ Analytics  │ Analytics    │ RAG         │
└────────────┴──────────────┴─────────────┘
       ↓
Combine evidence
       ↓
Check limitations
       ↓
Generate business analysis
       ↓
Agent inference / recommendation
```

The agent therefore performs **tool selection and multi-source reasoning**, which is the central agentic behavior demonstrated by this project.

---

# 21. Future Improvements

Potential extensions include:

### Automated evaluation

Introduce a larger benchmark containing:

- Tool-selection accuracy
- Retrieval relevance
- Numerical accuracy
- Citation correctness
- Hallucination rate
- Recommendation quality

### More structured outputs

Introduce typed response schemas for:

```text
Evidence
Implication
Recommendation
Confidence
Sources
```

### Expanded competitor coverage

Add additional EV manufacturers and normalize reporting definitions where possible.

### Time-series forecasting

Add forecasting tools for:

- EV registrations
- Market penetration
- Company sales

### Additional external data

Integrate:

- Charging infrastructure data
- Battery prices
- Regional EV adoption
- Government incentive changes

### Production deployment

Potential production architecture:

```text
Streamlit / Web UI
        ↓
API Layer
        ↓
Agent Service
        ↓
Analytics + RAG Services
        ↓
Data / Vector Stores
```

---

# 22. Key Takeaways

This project demonstrates the integration of:

- **LLM-based reasoning**
- **Tool calling**
- **Agentic workflows**
- **Retrieval-Augmented Generation**
- **Structured data analytics**
- **Competitor intelligence**
- **Source provenance**
- **Data-quality awareness**
- **Business decision support**

The central design principle is:

> **The LLM reasons over evidence; it does not replace the evidence.**

Numerical claims are grounded in structured analytics, qualitative claims are grounded in retrieved research, and strategic recommendations are explicitly identified as **agent inference**.

---

# 23. Author

**Karthik**

Built as a portfolio project demonstrating practical applications of:

```text
Generative AI
Agentic AI
RAG
LLM Tool Use
Data Analytics
Business Intelligence
Decision Support Systems
```

---

## Disclaimer

This project is intended for educational and portfolio demonstration purposes.

The analysis is based on the project's curated datasets and research corpus. Company-reported metrics may use different definitions and should not automatically be interpreted as directly comparable market-share figures. Strategic recommendations generated by the agent represent **agent inference** and should not be treated as professional investment or business advice.