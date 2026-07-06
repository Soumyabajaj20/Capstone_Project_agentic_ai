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

Built as a LangGraph:

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

The aggregator is explicitly prompted to summarize where the analysts **agree** and flag where they **disagree** and why, and give one final stance (Bullish/Bearish/Neutral) with a confidence level so it's forced to reconcile conflicting views rather than just average them.

## Tech Stack

- **Orchestration:** LangGraph
- **Sub-agent LLM:** Groq (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Aggregator LLM:** Google Gemini 2.5 Flash via `langchain-google-genai`
- **Market data:** `yfinance`

## Setup

### Install dependencies

```bash
pip install langgraph langchain-groq langchain-google-genai yfinance numpy pandas
```

## Configuration

```python
TICKER = "RELIANCE.NS"   # Indian tickers need .NS (NSE) or .BO (BSE) suffix
PERIOD = "1y"           # history window: "3mo", "6mo", "1y", etc.
BENCHMARK = "^NSEI"      # Nifty 50, used for beta calculation
```

Model and rate-limit settings are also configurable in the same cell:

```python
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

GROQ_RPM_LIMIT, GROQ_RPD_LIMIT = 30, 1000
GEMINI_RPM_LIMIT, GEMINI_RPD_LIMIT = 10, 250
```

## Sample Output

For `RELIANCE.NS` over a recent 1-year window, the three analysts disagreed in an interesting, realistic way:

- **Technical Analyst → Bearish** (price below its 50-day SMA, ~12% period drawdown)
- **Fundamental Analyst → Bullish** (reasonable P/E, 12.5% revenue growth, "strong buy" rating)
- **Risk Analyst → Moderate Risk** (beta 0.65, but a negative Sharpe ratio)

The Aggregator's job was to reconcile a bearish chart against bullish fundamentals exactly the kind of disagreement a single blended-lens model would have papered over.

## Design Decisions & Trade-offs

- **Shared `fetch_data` node vs. each analyst fetching its own data:** a single fetch avoids 3x redundant API calls to `yfinance`, at the cost of the three analysts technically not being fully black-box independen" data-wise (they all read from the same pre-fetched DataFrame).

## Judging Criteria Mapping

| Criterion | How this project addresses it |
|---|---|
| Problem originality & relevance | Grounded in a real research-desk workflow |
| Technical execution | Real parallel execution (not sequential-disguised-as-parallel), two-provider integration, live-traced execution |
| Orchestration design | Textbook Parallel + Aggregator pattern with clean state separation and proper fan-in |
| Clarity & presentation | Fully live narration of each agent's reasoning, final brief + individual notes shown separately |

---

**Course:** Agentic AI — Learners' Space 2026, Week 4 Capstone
**Author:** Sonu (IIT Bombay, Roll No. 25B2140)
