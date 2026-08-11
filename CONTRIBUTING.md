# Contributing

Contributions welcome — this project is meant to grow as a shared
reference point for "does a VLM or an OCR+SLM pipeline work better for
*this* kind of document", not a one-off personal benchmark.

## Good first contributions

- Implement one of the stubbed dataset loaders (`funsd`, `xfund`, `docile`,
  `docvqa`, `kleister` — see [`docs/ADDING_A_DATASET.md`](docs/ADDING_A_DATASET.md)).
- Add a model you're curious about to `configs/models.yaml` (see
  [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md)) and share the
  resulting `results/leaderboard.md` entry in your PR description.
- Add a `llama_cpp` runtime backend (see the note at the end of
  `docs/ADDING_A_MODEL.md`).
- Improve `evaluation/metrics.py` — e.g. CER/WER for free-text fields,
  currency-aware numeric tolerance, or a softer partial-credit metric for
  line-item lists.

## Setup

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## PR guidelines

- Keep loader modules self-contained — don't add dataset-specific special
  cases into shared code (`pipeline.py`, `metrics.py`, `base.py`). If a
  dataset genuinely needs a shared-code change, explain why in the PR.
- If you add a model, please note in the PR whether you actually ran it
  (and against what) — "added but never run" entries erode trust in the
  registry.
- If a model or dataset URL you're touching turns out to have moved or
  gone stale, fix or flag it in the same PR rather than leaving it silently
  broken for the next contributor.
