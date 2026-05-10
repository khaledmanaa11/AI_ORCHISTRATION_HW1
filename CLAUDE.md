# Project Guidelines — Dr. Segal Yoram V3

Source: `software_submission_guidelines-V3.pdf` (39 pages, v3.00, 2026-03-26)
All code in this project must comply with these rules.

---

## Mandatory Work Process (in order)
1. Create `docs/PRD.md` → get approval before proceeding
2. Create `docs/PLAN.md` (architecture)
3. Create `docs/TODO.md` (task list)
4. Create per-algorithm PRD files (`docs/PRD_<mechanism>.md`)
5. Approve all docs before writing any code
6. Update `TODO.md` during development
7. Save results, create visualizations, update `README.md`

---

## Mandatory Project Structure
```
project-root/
├── src/<package>/
│   ├── __init__.py          # exports __all__, __version__
│   ├── sdk/sdk.py           # SDK layer — single entry point for ALL logic
│   ├── services/            # Business logic
│   ├── shared/
│   │   ├── gatekeeper.py    # API Gatekeeper (mandatory)
│   │   ├── config.py        # Configuration manager
│   │   └── version.py       # Version tracking (start at 1.00)
│   ├── constants.py
│   └── main.py
├── tests/
│   ├── unit/                # mirrors src/ structure
│   ├── integration/
│   └── conftest.py          # shared fixtures
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PRD_<mechanism>.md   # one per algorithm/central mechanism
├── config/
│   ├── setup.json
│   └── rate_limits.json
├── data/
├── results/
├── assets/
├── notebooks/
├── README.md                # MANDATORY
├── pyproject.toml           # single source of truth for deps
├── uv.lock
├── .env-example             # committed, with dummy values
└── .gitignore
```

---

## Code Rules
- **Max 150 lines per file** — split if exceeded (helper functions, mixins, constants, models)
- **SDK Architecture**: All logic through SDK layer. No business logic in GUI/CLI.
  Flow: External Consumer → SDK → Domain Services → Infrastructure
- **OOP — No code duplication**:
  - Same logic in 2+ files → extract to shared module or mixin
  - Same try/except in 3+ files → create wrapper function
- **Mixin rules**: each mixin handles ONE concern, mixins don't depend on each other, independently testable
- **Docstrings**: explain "why" and "what" for every function, class, module
- **Naming**: descriptive, theoretical/descriptive functions, single responsibility, DRY
- **No hardcoded values**: all configs from config files (`cfg.get("key")` not `value = 10`)

---

## API Gatekeeper (mandatory)
- ALL API calls must go through `ApiGatekeeper` class in `shared/gatekeeper.py`
- Handles: rate limits, retries, FIFO queue, logging all calls
- Rate limits live in `config/rate_limits.json` — never hardcoded
- Queue: FIFO, configurable max depth, backpressure on full queue

```python
class ApiGatekeeper:
    """Centralized API call manager."""
    def __init__(self, config: RateLimitConfig): ...
    def execute(self, api_call, *args, **kwargs): ...
    def get_queue_status(self) -> QueueStatus: ...
```

`config/rate_limits.json` structure:
```json
{
  "rate_limits": {
    "version": "1.00",
    "services": {
      "default": {
        "requests_per_minute": 30,
        "requests_per_hour": 500,
        "concurrent_max": 5,
        "retry_after_seconds": 30,
        "max_retries": 3
      }
    }
  }
}
```

---

## Testing — TDD (mandatory)
- Follow **RED → GREEN → REFACTOR**
- Every module needs a matching test file
- Every function/method needs at least one test
- **Minimum 85% coverage** — build fails if below
- Mock external dependencies (APIs, files, DBs)
- Test files also max 150 lines
- `conftest.py` for shared fixtures

```toml
[tool.coverage.run]
source = ["src"]
omit = ["src/main.py", "*/tests/*", "src/**/gui/*"]

[tool.coverage.report]
fail_under = 85
```

---

## Linter: Ruff (zero errors — mandatory)
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E","F","W","I","N","UP","B","C4","SIM"]
ignore = ["E501"]
```
Run: `uv run ruff check` — must pass with **0 errors**.

---

## Configuration & Secrets
- No API keys, tokens, or secrets in source code
- Use only `os.environ.get("API_KEY")`
- `.env` → git-ignored (real secrets)
- `.env-example` → committed (dummy placeholder values)
- `.gitignore` must include: `.env`, `*.key`, `*.pem`, `credentials.json`

---

## Package Manager: uv (mandatory — no pip/venv/virtualenv)

| Task | Correct (uv) | Forbidden |
|------|-------------|-----------|
| Install deps | `uv sync` | `pip install` |
| Add package | `uv add <pkg>` | `pip install <pkg>` |
| Run script | `uv run python script.py` | `python script.py` |
| Run tests | `uv run pytest tests/` | `python -m pytest` |
| Lock deps | `uv lock` | `pip freeze` |

- `pyproject.toml` is the single source for dependencies (no `requirements.txt`)
- `uv.lock` must exist and be committed

---

## Version Management
- Start at **1.00** in `src/<pkg>/shared/version.py` and config JSONs
- Git: structured commits with clear messages, branches for features, Pull Requests, tagging for major versions
- Maintain a **Prompt Engineering Log** documenting all AI prompts used

---

## Parallel Processing
- **Multiprocessing** → CPU-bound (math, models, image processing)
- **Multithreading** → I/O-bound (network, file read/write)
- Thread safety: use `queue.Queue`, context managers, avoid shared mutable state

---

## Building Blocks Design
Each class/component defines:
- **Input Data**: input types, task scope, external dependencies
- **Output Data**: output types, format, edge case behavior
- **Setup Data**: parameters with defaults, config, validation

Principles: Single Responsibility, Separation of Concerns, Reusability, Testability

---

## Research & Results (if applicable)
- Parameter sensitivity analysis (OAT or equivalent)
- Results notebook (Jupyter): comparisons, statistical analysis
- Charts: Bar, Line, Scatter, Heatmap, Box plot, Waterfall
- Tools: Matplotlib, Seaborn, Plotly, Tableau, D3.js

---

## Cost & Budget Tracking
- Track API tokens (input/output) per model
- Cost breakdown table per model
- Optimization: batch processing, token reduction, model selection by cost-efficiency

---

## Final Submission Checklist
1. **Docs**: PRD, PLAN (architecture), README, API docs, prompt log
2. **Code**: modular, ≤150 lines/file, docstrings, naming conventions, OOP, no duplication
3. **API**: all calls through Gatekeeper, rate limits from config, queue management
4. **Tests**: TDD, 85%+, `ruff check` zero errors, edge cases, automated reports
5. **Config**: separate files, `.env-example`, no secrets in code, `.gitignore` correct
6. **Package**: uv only, `pyproject.toml`, `uv.lock` committed
7. **Research**: parameter experiments, results notebook, visualizations
8. **UX**: Nielsen's 10 heuristics, all screens documented with screenshots
9. **Versions**: Git history, tagging, deployment notes, license & credits
