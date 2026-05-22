# AlphaLens

AlphaLens is a Streamlit-based equity research assistant that evaluates the evidence-weighted case that a selected stock could outperform the S&P 500.

The app uses:

- Python
- Streamlit
- Financial Modeling Prep API
- OpenAI Responses API
- Deterministic financial metric calculation
- Deterministic evidence routing
- LLM-based research synthesis

AlphaLens is designed as a research assistant, not a trading system or financial advisor. It does not issue buy, sell, hold, short, or price-target recommendations.

---

## Project Goal

AlphaLens answers one central research question:

> Does the available evidence support a credible setup for this stock to outperform the S&P 500?

The app produces balanced, research-style analysis using language such as:

- "the evidence supports"
- "the evidence weakens"
- "the outperformance thesis"
- "key uncertainty"
- "the setup appears mixed"
- "the evidence-weighted case"

It avoids hard prediction language such as:

- "this stock will outperform"
- "buy"
- "sell"
- "hold"
- "guaranteed"
- "price target"

---

## Current v1 Features

The current version supports:

- Ticker selection or manual ticker entry
- Benchmark comparison against SPY by default
- Five analysis angles:
  1. Overall outperformance thesis
  2. Bull case for outperformance
  3. Bear case against outperformance
  4. Financial quality assessment
  5. What would need to improve?
- FMP data retrieval
- Deterministic financial metric calculation
- Deterministic market metric calculation
- Prompt-specific evidence routing
- OpenAI-powered research synthesis
- Streamlit metric cards
- Expandable financial metrics
- Expandable market metrics
- Expandable evidence package
- Short-term Streamlit caching to avoid duplicate API calls during reruns

---

## Architecture

AlphaLens v1 intentionally does not use embeddings, vector search, or semantic RAG.

The architecture is better described as:

> Deterministic evidence routing plus LLM synthesis over structured financial evidence.

The pipeline is:

```text
Prompt dropdown selection
→ evidence recipe lookup
→ fetch FMP data
→ clean data
→ calculate deterministic metrics
→ assemble prompt-specific evidence
→ build OpenAI prompt
→ generate research-style analysis
→ display result in Streamlit
```

This keeps the application explainable and easier to debug.

---

## Why This Is Not Strict RAG

Traditional retrieval-augmented generation often includes:

- Document chunking
- Embedding generation
- Vector database storage
- Semantic similarity search
- Dynamic retrieval at inference time

AlphaLens v1 does not use those components.

Instead, the user selects a predefined analysis type from a dropdown. That selection maps to a deterministic evidence recipe. The app then assembles the relevant financial evidence and sends that structured evidence to the LLM for synthesis.

This makes the system simpler, more transparent, and appropriate for a first portfolio-quality LLM finance application.

---

## Role of Python vs. OpenAI

AlphaLens uses Python for deterministic work:

- FMP API calls
- Data cleaning
- Date parsing
- Numeric conversion
- Revenue growth
- Margins
- Free cash flow
- Leverage metrics
- Stock returns
- Benchmark returns
- Relative returns
- Volatility
- Max drawdown
- Evidence package construction

AlphaLens uses OpenAI for interpretive work:

- Synthesizing conflicting signals
- Weighing evidence
- Framing bull and bear cases
- Explaining uncertainties
- Producing polished research-style narrative

The LLM is not responsible for calculating financial metrics.

---

## Data Sources

AlphaLens v1 uses Financial Modeling Prep data for:

- Company profile
- Income statement
- Balance sheet statement
- Cash flow statement
- Adjusted historical price data
- SPY adjusted historical price data
- Recent stock news

The adjusted price history is standardized into a canonical column:

```text
close_for_returns
```

This allows the metrics layer to calculate returns without depending on the exact FMP adjusted close column name.

---

## Metrics Calculated

### Financial Metrics

- Revenue
- Prior revenue
- Revenue growth
- Gross margin
- Operating margin
- Net margin
- Operating cash flow
- Free cash flow
- Free cash flow margin
- Cash and equivalents
- Total debt
- Total equity
- Debt-to-equity

### Market Metrics

- 1-month stock return
- 3-month stock return
- 6-month stock return
- 1-year stock return
- SPY returns over the same windows
- Relative return versus SPY
- Annualized volatility
- Benchmark annualized volatility
- Max drawdown
- Benchmark max drawdown

---

## Project Structure

```text
alphalens/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── rate_limiter.py
    ├── fmp_client.py
    ├── data_fetcher.py
    ├── data_cleaner.py
    ├── metrics.py
    ├── evidence.py
    ├── prompts.py
    ├── llm_client.py
    └── research_assistant.py
```

---

## File Responsibilities

### `app.py`

Streamlit frontend only.

Responsible for:

- Rendering the app title and sidebar
- Accepting ticker and analysis type inputs
- Calling the backend orchestration function
- Displaying analysis text and metrics

It should not contain core API, metric, evidence, prompt, or LLM logic.

---

### `src/config.py`

Central configuration.

Responsible for:

- Loading environment variables
- Reading API keys
- Defining model names
- Defining FMP base URL
- Setting default benchmark
- Setting default request limits
- Setting FMP rate-limit safety buffer
- Defining default ticker list

---

### `src/rate_limiter.py`

Rolling-window FMP API rate limiter.

Responsible for:

- Enforcing request pacing
- Supporting a safety buffer below the FMP plan limit
- Remaining thread-safe for future parallel requests

The default configuration uses approximately 700 requests per minute.

---

### `src/fmp_client.py`

Low-level FMP request engine.

Responsible for:

- Reusing a `requests.Session`
- Injecting the FMP API key
- Applying the rate limiter
- Retrying temporary failures
- Handling HTTP errors
- Parsing JSON
- Validating basic response payloads

It does not know about specific finance endpoints.

---

### `src/data_fetcher.py`

Endpoint-specific FMP functions.

Responsible for fetching:

- Company profile
- Income statement
- Balance sheet
- Cash flow statement
- Stock price history
- Benchmark price history
- Recent stock news

---

### `src/data_cleaner.py`

Data cleaning utilities.

Responsible for:

- Converting raw FMP responses into pandas DataFrames
- Parsing date columns
- Converting numeric columns
- Sorting financial statements newest-to-oldest
- Sorting price history oldest-to-newest
- Creating `close_for_returns`

---

### `src/metrics.py`

Deterministic metric calculation.

Responsible for calculating:

- Financial metrics
- Return windows
- Relative benchmark performance
- Annualized volatility
- Max drawdown

The LLM does not calculate these metrics.

---

### `src/evidence.py`

Deterministic evidence routing.

Responsible for:

- Building prompt-specific evidence packages
- Creating financial and market signal summaries
- Selecting evidence based on analysis type
- Making clear that AlphaLens v1 does not use semantic RAG

---

### `src/prompts.py`

Prompt templates and evidence recipes.

Responsible for:

- Storing analysis options
- Storing evidence recipes
- Building the system prompt
- Building the user prompt
- Building the prompt payload for the LLM client

---

### `src/llm_client.py`

OpenAI Responses API wrapper.

Responsible for:

- Creating the OpenAI client
- Sending prompts to the Responses API
- Returning generated text
- Handling LLM response errors

It does not contain finance logic.

---

### `src/research_assistant.py`

Main backend orchestration layer.

Responsible for:

```text
fetch data
→ clean data
→ calculate metrics
→ build evidence
→ build prompt
→ call OpenAI
→ return structured result
```

The Streamlit app calls this layer directly.

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AlphaLens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create a local environment file

```bash
cp .env.example .env
```

Edit `.env` and add your real API keys:

```bash
FMP_API_KEY=your_real_fmp_api_key
OPENAI_API_KEY=your_real_openai_api_key
```

Optional settings:

```bash
OPENAI_MODEL=gpt-4.1-mini
DEFAULT_BENCHMARK_SYMBOL=SPY
DEFAULT_STATEMENT_LIMIT=5
DEFAULT_NEWS_LIMIT=10
DEFAULT_PRICE_HISTORY_DAYS=400
FMP_MAX_REQUESTS_PER_MINUTE=700
```

Do not commit `.env`.

---

## Running the App

From the project root:

```bash
streamlit run app.py
```

Then open the forwarded Streamlit URL.

Recommended first test:

```text
Ticker: AAPL
Analysis angle: Overall outperformance thesis
Benchmark: SPY
Financial statement periods: 3
Recent news articles: 3
Price history lookback days: 420
```

---

## Testing Individual Modules

### Syntax check

```bash
python -m compileall src app.py
```

### Test configuration

```bash
python - <<'PY'
from src.config import settings, validate_required_settings

print(settings.openai_model)
print(settings.default_benchmark_symbol)
validate_required_settings()
print("Settings are valid.")
PY
```

### Test FMP fetch

```bash
python - <<'PY'
from src.data_fetcher import fetch_research_data

raw = fetch_research_data(
    symbol="AAPL",
    statement_limit=2,
    news_limit=3,
    price_history_days=60,
)

print(raw.symbol)
print(len(raw.company_profile))
print(len(raw.price_history))
print("FMP fetch works.")
PY
```

### Test full backend pipeline

```bash
python - <<'PY'
from src.evidence import ANALYSIS_OVERALL
from src.research_assistant import generate_equity_analysis

result = generate_equity_analysis(
    symbol="AAPL",
    analysis_type=ANALYSIS_OVERALL,
    statement_limit=3,
    news_limit=3,
    price_history_days=420,
)

print(result.analysis_text)
print("Backend pipeline works.")
PY
```

---

## Current Limitations

AlphaLens v1 is intentionally lightweight.

Current limitations include:

- No valuation model
- No analyst estimate integration
- No earnings call transcript analysis
- No peer-group comparison
- No portfolio construction
- No backtesting
- No vector database
- No embeddings
- No semantic retrieval
- No user authentication
- No persistent database
- No saved research history

---

## Future Improvements

Potential next steps:

- Add peer comparison
- Add valuation multiples
- Add analyst estimates
- Add earnings call transcript ingestion
- Add SEC filing support
- Add charting for relative returns and drawdowns
- Add watchlist support
- Add saved research runs
- Add PDF export
- Add deterministic scoring rubric
- Add optional semantic RAG for long-form filings and transcripts
- Add deployment instructions for Streamlit Community Cloud or AWS

---

## Disclaimer

AlphaLens is for educational and research purposes only.

It does not provide investment advice, financial advice, trading recommendations, portfolio recommendations, or price targets.

Users should perform their own due diligence and consult qualified financial professionals before making investment decisions.

---

## Status

AlphaLens v1 is a working MVP.

The current version demonstrates:

- Modular Python application architecture
- API integration
- Financial data processing
- Deterministic metric calculation
- LLM application design
- Streamlit UI development
- Evidence-routed prompt construction