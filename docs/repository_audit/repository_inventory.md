# MarketAI Repository Inventory

Audit date: 2026-09-10
Account: `startup01012`
Audit branch: `audit/repository-intelligence-2026-09-10`

## Scope

This Phase-0 inventory covers the 42 repositories named by the MarketAI master specification. Evidence is drawn from the connected GitHub account, repository metadata, current/default branches, README files and selected repository history/source where inspected. **This document is an intelligence baseline, not a claim that official end-to-end examples have been reproduced.** Reproduction status is recorded separately.

## Key account-level finding

`startup01012/EarningsOS` is **not** a GitHub fork (`fork=false`). It is an original repository and therefore must be treated as MarketAI's primary integration repository, with its existing work preserved. A dedicated audit branch was created before audit artifacts were written.

## Repository-by-repository inventory

| # | Repository | Role / method | Inputs | Outputs | Current MarketAI disposition | Evidence status |
|---|---|---|---|---|---|---|
| 1 | EarningsOS | Event-study and earnings ML/data research stack; PIT-oriented dataset engineering | Earnings events, OHLCV, fundamentals, benchmark data | Features, targets, audits, ML-ready parquet | **CORE** | Direct GitHub inspection; dataset/audit artifacts inspected |
| 2 | OpenAlice | Local agent workspace for market research, persistent issues, memory graph and approval-gated trading | Agent prompts, files, market/news/account tools | Reports, issue artifacts, staged trading actions | **ADAPT** for agent UX/governance | README + account metadata inspected |
| 3 | Lumibot | Python strategy/backtesting/live execution framework with AI-agent extensions | Historical data, broker APIs, strategy code | Orders, fills, backtest artifacts, strategy state | **RESEARCH / ADAPT** | README + current branch metadata inspected |
| 4 | NautilusTrader | Rust-native event-driven deterministic simulation/live trading engine with Python control plane | Tick/order book/bar/custom data, venue feeds | Events, orders, fills, positions, deterministic backtest/live execution | **PRIMARY EXECUTION/SIMULATION CANDIDATE** | README + current branch metadata + recent code-history inspection |
| 5 | Hummingbot | Exchange connector + strategy/executor framework for CEX/DEX and market making | REST/WebSocket market/account data | Orders, fills, strategy state | **CRYPTO/CONNECTOR OPTIONAL** | README inspected |
| 6 | Freqtrade | Python crypto bot with strategy framework and backtesting/hyperoptimization | Exchange candles/trades, strategies, indicators | Signals, orders, backtest reports | **CRYPTO OPTIONAL** | Repository metadata; deeper code pass pending |
| 7 | Vibe-Trading | Trading/agent research project | Project-specific market/agent inputs | Strategy/agent outputs | **RESEARCH ONLY** | Metadata; source-level pass pending |
| 8 | MIT-TradingAgents | Multi-agent trading/research architecture | Market/news/research inputs | Agent decisions/reasoning | **RESEARCH ONLY** | Metadata; source-level pass pending |
| 9 | FinRL | Reinforcement-learning financial trading framework | Market time series, environment state | Policies, actions, backtest results | **RESEARCH ONLY** | Repository metadata; source-level pass pending |
| 10 | Backtrader | Python event-driven backtesting/trading framework | Historical market feeds, strategies | Trade/fill/portfolio results | **BENCHMARK / LEGACY ADAPTER** | Repository metadata; source-level pass pending |
| 11 | CCXT | Unified exchange API abstraction | Exchange REST/WebSocket APIs | Normalized market/account/exchange calls | **CONNECTOR ONLY** | Repository metadata; source-level pass pending |
| 12 | Polymarket API | Prediction-market API tooling | Market/event API data | Market prices/order interactions | **ALTERNATIVE SIGNAL RESEARCH** | Repository metadata; source-level pass pending |
| 13 | OpenWA | WhatsApp automation/gateway-style integration | WhatsApp/session messages | Messages/events | **NOT CORE; ISOLATE** | Repository metadata + recent upstream-derived history signal |
| 14 | Chronos | Probabilistic time-series foundation models | Time-series context | Forecast distributions | **FORECAST BENCHMARK** | Current fork metadata/history inspected |
| 15 | TimesFM | Time-series foundation model | Univariate/multivariate time-series context | Forecasts | **FORECAST BENCHMARK** | Repository metadata; source-level pass pending |
| 16 | FinCast-fts | Financial time-series foundation-model research | Financial time series | Forecasts/embeddings | **FORECAST RESEARCH** | Recent fork commit history inspected |
| 17 | Uni2TS | General time-series foundation model/training ecosystem | Time series + configs | Forecasts/models | **FORECAST RESEARCH** | Repository metadata; source-level pass pending |
| 18 | Timer-XL | Long-context time-series forecasting research | Long historical sequences | Forecasts | **FORECAST RESEARCH** | Repository metadata; source-level pass pending |
| 19 | Financial-Time-Series | Financial forecasting/data research collection | Financial time series/datasets | Experiments/forecasts | **BENCHMARK RESEARCH** | Large fork footprint; source-level pass pending |
| 20 | StockMixer | Financial stock prediction model | Stock market features/returns | Direction/return predictions | **BENCHMARK RESEARCH** | Repository metadata; source-level pass pending |
| 21 | StockCL | Contrastive-learning stock representation/prediction research | Market sequences/features | Representations/predictions | **RESEARCH BENCHMARK** | Repository metadata; source-level pass pending |
| 22 | Kronos | Financial K-line foundation model using hierarchical tokenization + autoregressive Transformer | OHLCV/amount K-line sequences | Probabilistic forecasts | **FORECAST BENCHMARK / ADAPT** | README and current fork history inspected |
| 23 | Kronos1 | Fork variant / comparison target for Kronos lineage | K-line sequences | Forecasts | **RESEARCH / COMPARE** | Metadata/history inspected |
| 24 | Conductor | Maintained distribution of Kronos with packaging, fixed checkpoints, regression tests and CI | K-line sequences | Forecasts | **ADAPT; treat model as Kronos lineage** | Fork-specific commit diff inspected |
| 25 | FinGPT | Financial LLM training/inference toolkit and notebooks | Financial text + prompts | Generated text/classification/analysis | **NLP RESEARCH** | README/tree inspected |
| 26 | FinLLMs | Financial LLM/model research collection | Financial text/tasks | Model outputs | **NLP RESEARCH / BENCHMARK** | Metadata; source-level pass pending |
| 27 | PIXIU | Financial instruction-tuning/LLM benchmark ecosystem | Financial text, datasets, tasks | LLM responses/evaluations | **NLP BENCHMARK** | Metadata; source-level pass pending |
| 28 | FLANG | Financial language-model research | Financial language corpora/tasks | Financial NLP outputs | **NLP BENCHMARK** | Metadata; source-level pass pending |
| 29 | InvestLM | Investment-focused LLM research | Financial documents/prompts | Investment-analysis text | **RESEARCH ONLY** | Metadata; source-level pass pending |
| 30 | FinVis-GPT | Financial multimodal/visual-language research | Financial charts/text | Explanations/outputs | **RESEARCH ONLY** | Small fork footprint; source-level pass pending |
| 31 | FinBERT | BERT further trained on financial text + 3-class sentiment classifier | Financial sentences | Positive/neutral/negative probabilities + score | **NLP BASELINE** | README inspected; training and dataset constraints documented |
| 32 | Kimi-K3 | Kimi K3 model/source project | Large-model inputs | Text generation | **LLM RESEARCH ONLY** | Metadata/history inspected |
| 33 | Kimi-K3-in-C | C/C++ local inference engine with checkpoint streaming/quantization/performance engineering | Large model checkpoint + prompts | Token generation | **LOCAL INFERENCE RESEARCH** | Extensive commit history inspected |
| 34 | Crucix | Specialized intelligence/automation project | Project-specific data | Research/agent outputs | **REQUIRES DEEP CODE AUDIT** | Metadata only so far |
| 35 | Hermes-Agent | Multi-provider agent runtime, tools, durable state and governance patterns | Prompts, tools, provider sessions | Agent actions/reports | **SANDBOX AGENT INFRA** | Extensive current fork history inspected |
| 36 | Shannon | Agentic security/SAST and exploitation workflow | Source code and scan inputs | Findings, SARIF/PDF/report artifacts | **SECURITY TOOLCHAIN ONLY** | Extensive current fork history inspected |
| 37 | Numbat | Supporting intelligence/security/infrastructure project | Project-specific security/agent inputs | Security/agent outputs | **SECURITY/OBSERVABILITY RESEARCH** | Metadata; deep pass pending |
| 38 | skfolio | Portfolio optimization and risk-analysis library | Returns, constraints, portfolio definitions | Weights, risk/optimization outputs | **RISK/PORTFOLIO CORE CANDIDATE** | Repository metadata inspected |
| 39 | vectorbt | Vectorized research/backtesting/analysis framework | Historical arrays/dataframes | Signals, stats, backtests | **RESEARCH/BACKTESTING** | Repository metadata inspected |
| 40 | Ornith-1 | AI/intelligence research project | Project-specific model/data inputs | Agent/model outputs | **RESEARCH ONLY** | Metadata only so far |
| 41 | FreeDomain | Domain/infrastructure utility | Domain names/config | Domain-related state | **REJECT FROM CORE** | Metadata only; no demonstrated quant value |
| 42 | turbo-fieldfare | Local model/inference/engineering support project | Model/runtime inputs | Model/runtime outputs | **OPTIONAL INFRASTRUCTURE** | Metadata/history inspected |

## Reproduction status

No repository is marked `REPRODUCED` merely because a README contains a runnable command. Official-example execution requires an actual environment and dataset/checkpoint availability. Until that execution is performed, the conservative state is `NOT_REPRODUCIBLE_YET` or `BLOCKED`, not `REPRODUCED`.

## Immediate conclusions

1. EarningsOS is the system of record for the initial India/NSE research problem.
2. NautilusTrader is the strongest candidate for one authoritative event-driven execution/simulation core; Backtrader and vectorbt should not become parallel production engines.
3. Chronos/TimesFM/Kronos/FinCast/Uni2TS/Timer-XL/StockMixer/StockCL belong in a controlled forecasting tournament, not automatic production deployment.
4. FinBERT is a sensible first financial-NLP baseline because its training/evaluation task is explicit and its outputs are simple to calibrate.
5. Conductor should not be counted as a distinct scientific forecasting model; its fork history identifies it as a maintained Kronos distribution with engineering/test corrections.
6. Agent repositories are infrastructure for research/evidence workflows, not substitutes for validated quantitative prediction.
7. Crypto-specific stacks remain isolated from the initial NSE/NIFTY-50 scope until they demonstrate measurable benefit.
