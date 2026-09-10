# Repository Architecture Synthesis

## Audit principle

The 42 repositories are not one software system and must not be merged wholesale. They fall into distinct architectural families.

## 1. Quant/data core

**EarningsOS** is the system-of-record candidate for the initial MarketAI/NIFTY-50 research layer. Its existing artifacts include ML-ready earnings data and temporal/leakage audit outputs. Preserve this work and extend it behind versioned canonical schemas.

**NautilusTrader** is the strongest candidate for the authoritative event-driven simulation/execution semantics because its architecture uses a Rust-native core with Python as a control/orchestration plane and explicitly targets deterministic research/live parity.

**VectorBT** is useful for fast vectorized research and screening. It should not be treated as a second live execution authority.

**Backtrader** remains a useful comparison/reference engine but should not create a second production execution path.

**Skfolio** belongs after prediction: it can supply portfolio construction/risk optimization, subject to MarketAI's constraints and independent validation.

## 2. Forecasting family

Chronos, TimesFM, FinCast, Uni2TS, Timer-XL, Kronos/Kronos1, StockMixer, StockCL and Financial-Time-Series should enter a common forecasting adapter interface. They must compete against simple baselines on identical point-in-time datasets and walk-forward splits.

Canonical interface:

```text
ForecastModel
  fit(training_snapshot)
  predict(feature_snapshot, horizon)
  metadata()
```

The adapter must carry model version, training snapshot, feature version, prediction cutoff and inference timestamp.

Conductor should be evaluated as a Kronos-lineage engineering distribution rather than as an independent scientific model.

## 3. Financial NLP family

FinBERT should be the initial auditable sentiment baseline. Its repository describes a BERT model further trained in the finance domain and fine-tuned for three-class financial sentiment; prediction outputs include per-class softmax probabilities and a positive-minus-negative score.

FinGPT, FinLLMs, PIXIU, FLANG, InvestLM and FinVis-GPT should plug into task-specific interfaces for document classification, event extraction, retrieval, summarization or reasoning. They should not become the default numerical price predictor.

## 4. Agent family

OpenAlice, MIT-TradingAgents, Vibe-Trading, Hermes-Agent, Ornith-1 and related agent repositories are best represented as bounded research/workflow components. The MarketAI agent interface must require evidence IDs and explicit tool permissions.

No agent should directly bypass the prediction/risk/approval gates.

## 5. Execution/connectivity family

CCXT, Hummingbot and Freqtrade contain exchange connectivity and trading abstractions. Their initial relevance is limited by MarketAI's NSE/BSE equity scope. They should be isolated behind connector interfaces rather than coupled to the prediction engine.

## 6. Security/support family

Shannon and Numbat are not trading models. Their useful role is security testing/observability research. Kimi-K3-in-C and Turbo-Fieldfare are runtime/inference engineering references, not financial models.

## 7. Target architecture

```text
Canonical Events + PIT Market Data
          |
          v
Temporal Feature Store + Dataset Snapshots
          |
   +------+------+
   |             |
   v             v
NLP/Event      Quant Features
Engine         + EarningsOS
   |             |
   +------+------+
          |
          v
Forecast Adapter Tournament
          |
          v
Calibration + Agreement + Abstention
          |
          v
Evidence Graph + Explanation
          |
          v
Risk / Portfolio Layer
          |
          v
Event-driven Simulator / Execution Boundary
          |
          v
API + Web Application
```

## Explicit non-goals

- No automatic adoption of every foundation model.
- No multiple live execution engines.
- No autonomous agent trading authority.
- No vector database until retrieval requirements justify it.
- No Kafka/Kubernetes by default.
- No production deployment from research notebooks.
