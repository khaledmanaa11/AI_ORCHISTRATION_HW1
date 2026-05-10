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

## AI Model Used

- **Model:** claude-sonnet-4-6
- **Context:** All code generation, architecture decisions, and TDD cycles performed
  interactively with the model in a single session.
