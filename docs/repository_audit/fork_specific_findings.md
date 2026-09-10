# Fork-Specific Findings

Audit date: 2026-09-10

This document records observations that materially change the integration decision compared with a generic upstream-repository review.

## 1. EarningsOS is original, not a fork

The account repository is an original project rather than a GitHub fork. It contains the initial earnings research/data assets and therefore remains the primary MarketAI integration target.

## 2. Conductor is engineering around Kronos

The account's Conductor commit `a2cdb8f...` replaces dead `NeoQuasar/Conductor-*` checkpoint references with real Kronos checkpoints, adds packaging/tests/CI, removes committed runtime artifacts, and corrects README attribution. This is strong engineering value but does not establish a new independent forecasting algorithm.

## 3. Chronos fork has current compatibility work

The account Chronos fork contains current Chronos-2 compatibility fixes, including Transformers 5 `is_decoder` handling and PEFT 0.20 import-allowlist handling. These changes make the fork useful as a maintained compatibility branch, but the forecasting method itself remains upstream Chronos.

## 4. Kimi-K3-in-C has substantial systems engineering value

The account fork contains reproducibility and correctness work around a 1.56 TB checkpoint: immutable revision/checksum verification, C runtime numeric kernels, model/config validation, tokenizer parity, adversarial fixtures, sanitizers, memory-budget guards, streaming I/O, expert caching, and regression tests. It also documents a serious class of concurrency corruption where a one-slot reader overwrote live data and was disabled.

MarketAI should reuse patterns for checkpoint provenance, fail-closed model loading, fixture design and inference validation, not import Kimi-K3 as a financial model.

## 5. Hermes-Agent is strongest as governance/infrastructure reference

The account fork has extensive current work on provider/model-selection correctness, authentication evidence, profile-scoped secrets, profile-safe webhook/MCP/session state, fail-closed gateway shutdown and durable automation. These patterns are relevant to MarketAI's agent governance, but direct coupling would introduce unnecessary runtime/security complexity.

## 6. OpenAlice is useful but legally and operationally constrained

The account fork describes a local trading workspace, durable issues/memory and approval-gated trading operations. It explicitly labels the trading layer beta/experimental. The repository is AGPL-3.0. Use it for UX/workflow patterns and evidence-oriented agent operation, not as the production trading core.

## 7. Lumibot is useful but GPL-constrained

The account fork is GPL-3.0 and combines strategy, backtest and broker abstractions with AI-agent examples. Use as a research/reference adapter rather than merging its code into a differently licensed production core.

## 8. NautilusTrader remains the execution/simulation candidate

The account fork tracks the current upstream develop line and retains the Rust-native deterministic architecture. Its strongest MarketAI value is one authoritative event-driven semantics layer for simulation and eventual execution, provided India-specific market rules, data adapters and risk constraints are implemented and independently tested.

## 9. CCXT/OpenWA activity is mostly upstream functionality

Recent CCXT history is dominated by upstream-generated multilingual/API maintenance and exchange-specific fixes; OpenWA history inspected is messaging/dashboard work. Neither is a MarketAI intelligence core.

## 10. TimesFM and FinCast are forecast candidates, not production choices yet

TimesFM's fork README documents TimesFM 2.5, 200M parameters, long context and quantile output. FinCast documents a decoder-only transformer with PQ-Loss and MoE trained on >20B financial time points. Both are compelling benchmark candidates, but their research disclaimers and transfer-to-NIFTY validity require independent MarketAI evaluation.
