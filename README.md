# Multi-Agent Stock Research Desk

**Week 4 Capstone Project**

it is a multi-agent system that researches a stock the way a real equity research desk does: three independent analysts each file their own verdict from their own slice of data, and a portfolio manager reconciles their opinions into one final call instead of one model blending everything into a shallow, single-lens answer.

---

## Problem Statement

Ask a single LLM to "analyse this stock" and it tends to blend technical, fundamental, and risk reasoning into one shallow paragraph because it never actually separates the reasoning. In real research desks, a technical analyst, a fundamentals analyst, and a risk officer each file **independent** notes without seeing each other's work, and a portfolio manager is the only one who reconciles all three into a final call. That separation is what surfaces genuine disagreement (e.g. "the chart looks great but the risk profile is ugly") instead of averaging it away.

This project rebuilds that structure as a multi-agent system.

## Architecture — Parallel + Aggregator

```
                        ┌───────────────────┐
                        │    fetch_data       │   (single shared data-pull node)
                        └─────────┬──────────┘
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
            ┌────────────┐ ┌────────────┐ ┌────────────┐
            │ Technical  │ │Fundamental │ │   Risk     │      <- run in
            │  Analyst   │ │  Analyst   │ │  Analyst   │         PARALLEL
            └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                     └────────────┼────────────┘
                                  ▼
                        ┌───────────────────┐
                        │    Aggregator       │   (fan-in / synthesis)
                        └─────────┬──────────┘
                                  ▼
                              FINAL BRIEF
```

Built as a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`:

1. **`fetch_data`** — a single shared node pulls price history (via `yfinance`), benchmark data, and company info once, so the three analysts don't redundantly hit the API three times each.
2. **Fan-out** — `fetch_data` has an edge to all three analyst nodes. Since none of the analysts depend on each other, LangGraph executes them **concurrently**.
3. Each analyst writes to its **own key** in shared state (`technical_report`, `fundamental_report`, `risk_report`) — no write conflicts, no shared mutable state.
4. **Fan-in** — the `aggregator` node only fires once **all three** analyst branches have completed.

| Agent | Sees only | Model |
|---|---|---|
| **Technical Analyst** | RSI(14), SMA20/SMA50, recent volatility, period return | Groq — `llama-3.3-70b-versatile` |
| **Fundamental Analyst** | P/E, market cap, sector, margins, revenue growth, analyst rating | Groq — `llama-3.3-70b-versatile` |
| **Risk Analyst** | Annualized volatility, max drawdown, beta vs. benchmark, Sharpe ratio | Groq — `llama-3.3-70b-versatile` |
| **Aggregator** | The three finished verdicts (not the raw numbers) | Gemini 2.5 Flash |

The aggregator is explicitly prompted to (1) summarize where the analysts **agree**, (2) flag where they **disagree** and why, and (3) give one final stance (Bullish/Bearish/Neutral) with a confidence level — so it's forced to reconcile conflicting views rather than just average them.

## Why Two Different LLM Providers

- The three analysts are cheap, high-volume, "read the numbers and give a verdict" calls — Groq's free tier (fast LPU inference) is a great fit.
- The aggregator only runs once per query but does the hardest reasoning step (reconciling 3 independent, sometimes-conflicting opinions), so it's routed to a different model — Gemini 2.5 Flash.
- This also demonstrates that the orchestration layer (LangGraph) doesn't care which model backs a given node — agents are freely swappable, which is the whole point of building on an agentic framework instead of one long prompt.

## Tech Stack

- **Orchestration:** LangGraph (`StateGraph`, parallel fan-out/fan-in)
- **Sub-agent LLM:** Groq (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Aggregator LLM:** Google Gemini 2.5 Flash via `langchain-google-genai`
- **Market data:** `yfinance` (free, no API key required)
- **Numerics:** `numpy`, `pandas`
- **Environment:** Jupyter Notebook

## Project Structure

```
.
├── multi_agent_stock_research.ipynb   # the entire system — one notebook, one file
└── README.md                          # this file
```

Everything — state schema, data fetching, indicator math, all four agents, the rate limiter, and the graph — lives in a single notebook, organized into clearly labeled, sequential cells (see the notebook's own table of contents in its first markdown cell).

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain-groq langchain-google-genai yfinance numpy pandas
```

(The notebook's first code cell also runs this via `%pip install`.)

### 2. Get free API keys

| Provider | Where to get a free key |
|---|---|
| Groq | https://console.groq.com/keys |
| Gemini | https://aistudio.google.com/apikey |

### 3. Set your keys

Open the notebook's **"API Keys"** cell and paste them directly:

```python
os.environ["GROQ_API_KEY"] = "gsk_..."
os.environ["GOOGLE_API_KEY"] = "AIza..."
```

> ⚠️ **Before pushing to GitHub:** clear the output of this cell and blank out the key strings. Never commit real API keys to a public repo. If you've accidentally pushed a key, rotate/regenerate it immediately.

## Configuration

The only cell you need to touch to research a different stock:

```python
TICKER = "RELIANCE.NS"   # Indian tickers need .NS (NSE) or .BO (BSE) suffix
PERIOD = "6mo"           # history window: "3mo", "6mo", "1y", etc.
BENCHMARK = "^NSEI"      # Nifty 50, used for beta calculation
```

Model and rate-limit settings are also configurable in the same cell:

```python
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

GROQ_RPM_LIMIT, GROQ_RPD_LIMIT = 30, 1000
GEMINI_RPM_LIMIT, GEMINI_RPD_LIMIT = 10, 250
```

## Running the Notebook

Run all cells top to bottom. You'll see a live trace of the whole process, e.g.:

```
######################################################################
#  MULTI-AGENT STOCK RESEARCH DESK -- Researching RELIANCE.NS
######################################################################

======================================================================
STEP 1 — DATA COLLECTION
======================================================================
[fetch_data] Pulling 6mo of price history for RELIANCE.NS and benchmark ^NSEI ...
[fetch_data] Got 124 trading days for RELIANCE.NS (2025-01-06 -> 2025-07-03)
[fetch_data] Latest close -> RELIANCE.NS: 1298.30 | ^NSEI: 25453.40
[fetch_data] Company info -> sector: Energy, industry: Oil & Gas Refining & Marketing

[fetch_data] Handing off to 3 analyst agents in parallel...

[Technical Analyst] Reading price action & indicators for RELIANCE.NS...
[Technical Analyst] Computed metrics: {'last_price': 1298.3, 'sma20': 1305.44, ...}
[Technical Analyst] Consulting Groq (llama-3.3-70b-versatile)...
    [Groq] 1/30 RPM used | 1/1000 RPD used
[Technical Analyst] Done in 0.8s. Verdict:
    ... Bearish.

[Fundamental Analyst] Reviewing company fundamentals for RELIANCE.NS...
    ...
[Risk Analyst] Assessing volatility, drawdown & beta for RELIANCE.NS...
    ...

======================================================================
STEP 3 — SYNTHESIS
======================================================================
[Aggregator / Portfolio Manager] All 3 analyst notes received. Reconciling independent opinions...
[Aggregator] Consulting Gemini 2.5 Flash for final synthesis...
    [Gemini] 1/10 RPM used | 1/250 RPD used
[Aggregator] Final brief ready in 1.4s.

======================================================================
ALL AGENTS FINISHED -- total run time: 4.2s
======================================================================
```

Followed by the final formatted brief and each analyst's individual note, for full transparency.

## Sample Output

For `RELIANCE.NS` over a recent 6-month window, the three analysts disagreed in an interesting, realistic way:

- **Technical Analyst → Bearish** (price below its 50-day SMA, ~12% period drawdown)
- **Fundamental Analyst → Bullish** (reasonable P/E, 12.5% revenue growth, "strong buy" rating)
- **Risk Analyst → Moderate Risk** (beta 0.81, but a negative Sharpe ratio)

The Aggregator's job was to reconcile a bearish chart against bullish fundamentals — exactly the kind of disagreement a single blended-lens model would have papered over.

## Rate Limit Monitoring

Both Groq's and Gemini's free tiers cap requests-per-minute (RPM) *and* requests-per-day (RPD). A `RateLimitMonitor` class tracks a rolling 60-second window of call timestamps plus a running daily count, **per provider**:

- Before every LLM call, it checks whether making the call would breach the RPM cap — if so, it **sleeps** just long enough to stay under the limit.
- If the daily cap is reached, it **raises a clear error** instead of letting the request fail with a confusing 429.
- It prints live usage after every call, e.g. `[Groq] 3/30 RPM used | 3/1000 RPD used`.

Every single LLM call in the notebook goes through one function, `call_llm()`, which wraps `monitor.before_call()` → `llm.invoke()` → `monitor.record_call()` — this is the one choke point that guarantees the system never silently blows past either provider's free-tier limits.

Default limits (`GROQ_RPM_LIMIT=30`, `GROQ_RPD_LIMIT=1000`, `GEMINI_RPM_LIMIT=10`, `GEMINI_RPD_LIMIT=250`) are current best estimates for each provider's free tier as of mid-2026 — these numbers do shift over time, so they're exposed as config variables rather than hardcoded. Check current figures at:
- https://console.groq.com/docs/rate-limits
- https://ai.google.dev/gemini-api/docs/rate-limits

## A Gemini Gotcha We Hit (and Fixed)

Early runs occasionally produced a **truncated** final brief (cutting off mid-sentence). This turned out to have nothing to do with rate limits — it's a documented quirk of Gemini 2.5 models: they perform internal "thinking" before writing a visible answer, and those thinking tokens are deducted from the *same* `max_tokens` budget as the visible output. With a low `max_tokens`, the model can burn its entire budget on invisible reasoning and get cut off before finishing the part you actually see.

**Fix:** since the aggregator's task (reconciling three short analyst notes) doesn't need deep step-by-step reasoning, thinking is explicitly disabled and the token budget is raised as a safety margin:

```python
aggregator_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=0.3,
    max_tokens=2048,
    thinking_budget=0,
)
```

## Design Decisions & Trade-offs

- **Shared `fetch_data` node vs. each analyst fetching its own data:** a single fetch avoids 3x redundant API calls to `yfinance`, at the cost of the three analysts technically not being fully "black-box independent" data-wise (they all read from the same pre-fetched DataFrame). Each analyst still only computes and reasons over *its own slice* of that shared data.
- **Two providers instead of one:** deliberately shows LangGraph's model-agnostic node design, and matches workload shape (many cheap calls vs. one expensive reasoning call) to the right free tier.
- **Rate limiting built in-house rather than relying on provider SDK retries:** gives predictable, proactive behavior (wait before you would be throttled) instead of reactive retry-after-failure handling.

## Known Limitations

- `yfinance`'s `.info` field can be sparse or missing for smaller/less-covered tickers — the fundamentals analyst will note fields as "N/A" in that case rather than fail.
- Analyst verdicts are only as good as the underlying free-tier models (Groq's Llama 3.3 and Gemini 2.5 Flash) — this is a demonstration of orchestration architecture, not a substitute for professional investment research.
- This is not financial advice; it's a capstone project demonstrating multi-agent orchestration patterns.

## Judging Criteria Mapping

| Criterion | Weight | How this project addresses it |
|---|---|---|
| Problem originality & relevance | 25% | Grounded in a real research-desk workflow rather than a toy example |
| Technical execution | 30% | Real parallel execution (not sequential-disguised-as-parallel), custom rate-limit monitoring, two-provider integration, live-traced execution |
| Orchestration design | 25% | Textbook Parallel + Aggregator pattern with clean state separation and proper fan-in |
| Clarity & presentation | 20% | Fully commented single notebook, live narration of each agent's reasoning, final brief + individual notes shown separately |

---

**Course:** Agentic AI — Learners' Space 2026, Week 4 Capstone
**Author:** Sonu (IIT Bombay, Roll No. 25B2140)
