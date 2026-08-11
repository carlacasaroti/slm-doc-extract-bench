# slmbench — Small Language Models for Document Extraction

A local-first framework to benchmark **Small Language Models (SLMs)** on
**document information extraction**, comparing two approaches head-to-head
across multiple document domains:

| Family | How it works |
|---|---|
| **Multimodal (VLM)** | The model reads the document image directly (Qwen2.5/3-VL, InternVL3, SmolVLM2, Phi-4-multimodal, ...) |
| **Text + OCR** | An OCR engine (Tesseract / PaddleOCR / EasyOCR) reads the text first, then a text-only SLM (Phi-4-mini, Qwen2.5, Gemma3, Llama 3.2, ...) extracts structured fields |

Everything runs **100% locally** — no API keys, no cloud calls — via
[Ollama](https://ollama.com) or HuggingFace `transformers`.

```
┌──────────────┐        ┌──────────────────┐
│  Document     │──────▶│  Multimodal VLM   │──▶ JSON
│  (image/PDF)  │        └──────────────────┘
│              │        ┌──────────────────┐
│              │──────▶│ OCR engine │─▶│ Text SLM │──▶ JSON
└──────────────┘        └──────────────────┘
                                │
                                ▼
                     Evaluator (field-level F1,
                     exact match, latency, JSON validity)
                                │
                                ▼
                        results/leaderboard.md
```

## Why this exists

VLMs get most of the attention for document extraction right now, but a
classic OCR-then-SLM pipeline is often cheaper, faster, and sometimes just
as accurate — especially on clean, machine-printed documents. This repo
exists to make that comparison **reproducible, local, and dataset-specific**
instead of anecdotal: the right answer usually depends on your document
domain, not on which model is trending.

## Quickstart

```bash
git clone https://github.com/<you>/slm-doc-extract-bench
cd slm-doc-extract-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 1. Pull a couple of local models via Ollama (https://ollama.com/download)
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:3b

# 2. Smoke test with the zero-setup synthetic dataset (generated on the fly)
slmbench run --dataset synthetic_invoices_ptbr --limit 10 \
  --multimodal qwen2.5-vl-3b --text-slm qwen2.5-3b --ocr-engine tesseract

# 3. See the leaderboard
cat results/leaderboard.md
```

No GPU? Everything above also runs on CPU (slower) — just pick smaller
models (`smolvlm2-2.2b`, `gemma3-1b`) from `configs/models.yaml`.

## Adding real datasets

The synthetic PT-BR invoice dataset needs nothing beyond the quickstart.
For real benchmark datasets (SROIE, CORD, FUNSD, DocILE, DocVQA, Kleister),
see [`configs/datasets.yaml`](configs/datasets.yaml) for what's registered
and:

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset cord
```

Some datasets (SROIE, DocILE) require free registration on the host site —
the script tells you exactly what to do when that's the case.

## Adding a model

Multimodal and text SLMs are added purely through
[`configs/models.yaml`](configs/models.yaml) — no code changes needed for
any model already supported by Ollama or `transformers`. See
[`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md).

## Adding a dataset / document domain

See [`docs/ADDING_A_DATASET.md`](docs/ADDING_A_DATASET.md). Two datasets
(`sroie`, `cord`) have complete reference loader implementations to copy
from; several others (`funsd`, `xfund`, `docile`, `docvqa`, `kleister`) are
scaffolded as good first contributions.

## Metrics

- **Field-level F1** — per-field correctness, not all-or-nothing on the
  whole JSON (numbers matched with tolerance, strings with fuzzy matching,
  line-item lists matched order-independently).
- **Exact match rate** — fraction of documents with every field correct.
- **JSON validity rate** — how often the model returns parseable JSON at all
  (this alone is often where multimodal vs. text_ocr diverges).
- **Latency** (mean, p95) — measured per document, per model.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and
how to extend the metrics.

## Running with Docker

```bash
docker compose -f docker/docker-compose.yml up -d ollama
docker compose -f docker/docker-compose.yml run slmbench \
  slmbench run --dataset synthetic_invoices_ptbr --limit 10
```

## Project structure

```
configs/            model / dataset / task registries (YAML — edit these first)
src/slmbench/
  datasets/          loaders per dataset -> common DocumentSample
  models/            multimodal + text_ocr extractors, OCR engines, runtime backends
  extraction/        prompt building, task schemas, pipeline orchestration
  evaluation/        metrics + leaderboard report generation
  cli.py             `slmbench` command
scripts/             dataset download helper, synthetic data generator
docs/                architecture + contribution guides
results/             leaderboard.md + per-run JSON (gitignored except leaderboard)
```

## Contributing

PRs adding a dataset loader, a model config entry, or a new metric are all
welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
