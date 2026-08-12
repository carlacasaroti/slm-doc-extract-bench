# Architecture

## Design goals

1. **Config over code.** Adding a model or a dataset should almost never
   require touching the pipeline, evaluator, or CLI — just a YAML entry
   (models/tasks) or a new loader module following a fixed contract
   (datasets).
2. **One normalized document shape.** Every dataset loader outputs
   `DocumentSample` (`src/slmbench/datasets/base.py`). Every extractor
   consumes the same shape regardless of whether it's a VLM or an OCR+SLM
   pipeline. This is what makes the multimodal-vs-text_ocr comparison
   apples-to-apples instead of two different codebases.
3. **Fair prompting.** One prompt template per task (not per model) — see
   `src/slmbench/extraction/prompts/builder.py`. If a model needs
   hand-tuned prompting to do well, that's a finding worth reporting, not
   something to hide by prompt-engineering only the models you like.
4. **Field-level scoring, not blob matching.** See
   `src/slmbench/evaluation/metrics.py` — a model that nails 8/10 fields
   should not score the same as one that nails 0/10.

## Data flow

```
configs/datasets.yaml  ──▶ datasets/registry.py ──▶ datasets/loaders/<x>.py ──▶ DocumentSample[]
configs/models.yaml     ──▶ extraction/pipeline.py::build_extractors() ──▶ BaseExtractor[]
configs/tasks.yaml       ──▶ extraction/schema.py + prompts/builder.py

for extractor in extractors:
    for sample in DocumentSample[]:
        result = extractor.extract(sample.image_path, schema, prompt)  # ExtractionResult
        score  = evaluation/metrics.py::score_sample(result, sample.ground_truth)

aggregate(scores) ──▶ evaluation/report.py ──▶ results/leaderboard.md
```

## Key modules

- `datasets/base.py` — `DocumentSample`, `ExtractionResult`. The two
  contracts everything else depends on.
- `datasets/registry.py` — reads `configs/datasets.yaml`, dispatches to the
  right loader by `loader:` field.
- `datasets/loaders/*.py` — one module per dataset. Contract in
  `_template.py`. `sroie.py` and `cord.py` are complete reference
  implementations; `synthetic.py` needs no external data at all.
- `models/base.py` — `BaseExtractor` abstract class + best-effort JSON
  extraction from raw model output (`_extract_json`) shared by both
  families.
- `models/multimodal/extractor.py` — VLM extractor, routes to
  `models/runtime/{ollama,transformers}_backend.py` based on the model's
  `backend:` field in `configs/models.yaml`.
- `models/text_ocr/extractor.py` — runs an OCR engine
  (`models/ocr/engines.py`) then a text SLM via the same runtime backends.
- `extraction/pipeline.py` — `build_extractors()` turns `configs/models.yaml`
  into a list of ready-to-run extractors; `run_extractor_on_samples()` runs
  one extractor over a dataset.
- `evaluation/metrics.py` — `score_sample()` (per-document field scoring),
  `aggregate()` (per-model summary stats).
- `evaluation/report.py` — writes `results/<run>.json` and appends a table
  to `results/leaderboard.md`.

## Extending the metrics

`score_sample()` in `metrics.py` is intentionally the one place that knows
how to compare a predicted value to a ground-truth value
(`_values_match()`). If you need e.g. a stricter numeric tolerance for
currency fields, or CER/WER instead of fuzzy string ratio for free-text
fields, that's the function to change — it's used identically regardless
of model family or dataset.

## Extending to layout-aware text_ocr

The current `text_ocr` family deliberately uses **plain OCR text**, no
bounding boxes — this measures "how much does the visual layout actually
matter" against the multimodal family. If you want a fairer/stronger
text_ocr baseline, the natural extension point is
`DocumentSample.ocr_text`: change `models/ocr/engines.py` to return
layout-annotated text (e.g. tab-separated columns reconstructed from boxes)
and feed that through `text_ocr/extractor.py` instead.

## Controlled experiments

Sometimes you want to isolate one variable while holding the underlying
document fixed — e.g. "do small models drop these specific fields because
they're inherently hard, or because they're competing with 5 other fields
in the same schema?" The pattern for this:

1. Add a second task in `configs/tasks.yaml` with a narrower schema (see
   `invoice_fields_reduced`).
2. Register a second dataset entry pointing at the **same loader** with the
   new `task:` (see `synthetic_invoices_ptbr_reduced`).
3. Make sure the loader filters `ground_truth` down to
   `get_schema(task_id)["properties"].keys()` and sets
   `DocumentSample.task_id` from the passed-in `task_id` rather than
   hardcoding it — this is exactly what `loaders/synthetic.py` does, and
   it's why the same rng seed produces byte-identical underlying documents
   across both dataset ids, differing only in which fields are scored.

Run both and compare field-level F1 for the shared fields — a jump when
isolated points to schema/field-count fatigue; no change points to a
genuine per-field extraction limitation.

This pattern extends naturally to more than two points: `invoice_fields_2`
/ `_4` / `_6` / `_8` (plus the full 10-field `invoice_fields`) form a
nested sequence over one fixed field order, giving a full degradation
curve instead of a single before/after comparison — see the "field-count
degradation curve" section in the README for how to run and plot it.

## Why Ollama AND transformers?

- **Ollama** — the easiest path for anyone without a Python ML background
  to reproduce results: `ollama pull <model>` and it just runs. Best
  default for anyone cloning this repo just to compare a few models.
- **transformers** — needed for models not (yet) packaged for Ollama, or
  when you want more control (custom generation params, batching,
  quantization strategy).

Both are optional extras (`pip install -e ".[transformers]"` /
`".[ocr]"`) so a minimal install stays light.
