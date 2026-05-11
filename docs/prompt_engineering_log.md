# Prompt Engineering Log

**Project:** Noisy Sine Signal Dataset Generator
**AI Model:** claude-sonnet-4-6
**Start Date:** 2026-05-10

---

## Prompt 1 — Initial Project Description

**Date:** 2026-05-10
**Prompt Summary:** User described the project requirements: generate a labeled dataset from 4 sine
waves (5 Hz, 15 Hz, 50 Hz, 100 Hz) with Gaussian + burst noise, windowed with a one-hot context
selector C of length 5, and stored as .npz splits (70/15/15). Labels are scalar clean values.

**Outcome:** Understood project scope. Identified need for PRD, PLAN, and TODO docs.

---

## Prompt 2 — PRD / PLAN / TODO Generation

**Date:** 2026-05-10
**Prompt Summary:** Requested generation of docs/PRD.md, docs/PLAN.md, docs/TODO.md per the
submission guidelines (Dr. Segal Yoram V3). Emphasis on:
- One-hot C vector of length 5 (not W)
- Windows sourced exclusively from s5_noisy
- Labels = scalar clean value of C-selected signal at window center

**Outcome:** Full PRD.md, PLAN.md, and TODO.md generated. Phase 0 verification confirmed
all formulas, noise model, and dataset structure documented correctly.

---

## Prompt 3 — Full TODO Execution

**Date:** 2026-05-10
**Prompt Summary:** "Start executing the todos in the TODO file one by one." Executed all 31
phases from environment setup through final checklist.

**Key decisions during execution:**
- Fixed `requires-python` from `>=3.10` to `>=3.11` (numpy 2.4.4 requires 3.11+)
- Adjusted s4 amplitude test to use RMS instead of peak (100 Hz at 1000 Hz sampling cannot
  hit exact peak due to discretization — only 10 samples/cycle)
- Added `# noqa: N803, N806` comments for ML-convention variable names (X, C) in source
- Renamed test functions to lowercase to satisfy ruff N802 rule
- Refactored SDK mock tests into a shared `_run_sdk_with_mocks()` helper to eliminate
  code duplication and fix ruff E702 violations

**Outcome:** 244 tests pass, 97.79% coverage, ruff 0 errors, all 7 visualizations generated.

---

## Iterative Refinements

| Issue | Correction Applied |
|-------|--------------------|
| `uv init` created subdirectory | Re-ran `uv init --no-workspace` at project root |
| Lazy imports in sdk.py prevented mock patching | Moved all imports to module level |
| N802/N806 ruff violations in tests | Renamed functions + added noqa comments |
| B017 blind exception catch | Narrowed to `(ValueError, json.JSONDecodeError)` |
| s4 peak amplitude test failure | Switched to RMS-based amplitude verification |
| Integration test scope mismatch | Added module-scoped fixture overrides in integration test |

---

---

## Prompt 4 — Neural Signal Pipeline (Phases 10–20)

**Date:** 2026-05-10
**Prompt Summary:** Extended project with FCN / RNN / LSTM training pipeline. Phases 10–20 covered:
- Phase 10: FCN model (`Dense 128→64→1`, Dropout 0.1)
- Phase 11: Vanilla RNN (`hidden=64, tanh`)
- Phase 12: LSTM (`hidden=64, Dense 32→1`)
- Phase 13: DataLoaderService (FCN concat `[X, C]`, RNN reshape `[N, 10, 1]`)
- Phase 14: Preprocessor (z-score on X_train, save params to JSON)
- Phase 15: Trainer (Adam, MSE, early stopping, CSV log)
- Phase 16: Evaluator (cold-load checkpoint, compute train/val/test MSE, save CSV)
- Phase 17: Visualizer (5 plot types: clean/noisy/predicted, loss curves, MSE bar, residuals, scatter)
- Phase 18: NeuralSignalSDK (orchestrator, single entry point)
- Phase 19: ConfigManager for neural_signal (AppConfig with FCNConfig, RNNConfig, LSTMConfig)
- Phase 20: ApiGatekeeper integration, rate_limits.json

**Key decisions:**
- FCN uses `weight_decay=0.0001`; RNN and LSTM use `weight_decay=0.0`
- Evaluator does cold-load from checkpoint path before computing MSE
- Visualizer uses Agg backend (headless), always saves PNG
- ConfigManager re-exports all dataclass types for backward compat

**Outcome:** 377 tests, 98.37% coverage, 0 ruff errors.

---

## Prompt 5 — Gap Fixes & Phase 21–30 (2026-05-11)

**Date:** 2026-05-11
**Prompt Summary:** Audited TODO.md, identified 12 gaps and pending phases. Implemented:
- File size violations: split sdk.py (VizMixin), config.py (config_types), conftest.py, test_config.py
- Gap 3: gradient clipping in Trainer (`cfg.gradient_clip_norm`)
- Gap 4: DataLoader reproducibility seed (torch.Generator per loader)
- Gap 8: training_log paths moved from hardcoded strings to `OutputConfig` fields
- Gap 10: device handling in Trainer and Evaluator (`cfg.training.device`)
- Gap 11: val MSE divergence integration tests
- Gap 12: SensitivityAnalyzer OAT service + 13 unit tests
- Phase 21: argparse CLI (`--model`, `--config`, `--rate-limits`, `--verbose`)
- Phase 22: per-algorithm PRDs (PRD_fcn.md, PRD_rnn.md, PRD_lstm.md)
- Phase 23: cold-load integration tests for all 3 models
- Gap 6: line style/color tests in test_visualizer.py
- Gap 7: docs/api_docs.md
- Phase 26: notebooks/results_analysis.ipynb

**Key decisions:**
- VizMixin uses `getattr(self, "_train_results", {})` to avoid circular dependency
- DataLoader generator only set for shuffle=True loaders (val/test stay deterministic)
- Evaluator accepts optional `device` param (default "cpu") for forward compatibility

**Outcome:** 403 tests, 98.50% coverage, 0 ruff errors.

---

## AI Model Used

- **Model:** claude-sonnet-4-6
- **Context:** All code generation, architecture decisions, and TDD cycles performed
  interactively with the model across multiple sessions.
