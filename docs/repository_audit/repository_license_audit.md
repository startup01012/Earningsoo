# Repository License / Artifact Risk Audit

Audit date: 2026-09-10

This is a screening document, not legal advice. It separates repository software licensing from model/checkpoint/dataset terms, which must be reviewed independently before redistribution or hosted use.

## High-priority findings

### OpenAlice
The account fork is AGPL-3.0. The README also describes the project as experimental and the trading layer as beta. Because MarketAI is intended to have a web/API product surface, AGPL network-service obligations require legal review before any code is copied into a proprietary or differently licensed production distribution. Treat as reference/adaptation source unless counsel approves the exact use.

### Lumibot
The account fork metadata reports GPL-3.0. This is materially different from the permissive/MIT assumption sometimes made about trading libraries. Do not copy Lumibot modules into a separately licensed proprietary MarketAI component without a deliberate GPL compatibility decision. Prefer interface-level integration or isolated research use.

### NautilusTrader
The account fork metadata reports LGPL-3.0. This is potentially usable as an isolated library/component, subject to the exact source files, dynamic/static linking arrangement, modifications, and upstream notices. The repository itself also documents license checks against LGPL-compatible dependencies. Keep it isolated behind a narrow execution/simulation boundary and preserve notices.

### FinBERT
The repository README describes the implementation and model training datasets, including Reuters TRC2 for domain-adaptive pretraining and Financial PhraseBank for sentiment. Dataset access/terms and the model/checkpoint's own terms must be checked separately from the repository license before commercial deployment.

### Kimi-K3-in-C
The fork's project itself declares Apache-2.0 and records third-party vendored components/modifications in NOTICE. However, the 1.56 TB model checkpoint is an external artifact and must not be assumed to inherit the repository's Apache-2.0 terms. Treat model weights as separately governed/untrusted artifacts.

## General rule for MarketAI

For every candidate component track at least four independent legal objects:

1. repository/source-code license;
2. pretrained model/checkpoint license;
3. dataset license/access terms;
4. dependency and vendored-component licenses.

The production build must keep a machine-readable SBOM/license inventory and reject components whose exact redistribution/hosted-use terms are unresolved.

## Prohibited shortcut

A repository being public, a GitHub fork existing under `startup01012`, or a README saying "open source" does **not** establish commercial compatibility for MarketAI.
