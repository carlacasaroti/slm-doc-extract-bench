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

## Reproducing the real-document (PDF invoice) experiments

The synthetic dataset needs no setup. To benchmark on your **own real
documents** (e.g. scanned or digital PDF invoices), follow the steps below —
this is the exact workflow behind the `real_waste_invoices` dataset:
multimodal VLM vs OCR+SLM vs a cloud LLM, plus a field-count curve.

### 1. Extra dependencies

```bash
# PDF -> image rasterization (required)
pip install pymupdf

# Tesseract OCR (required for the text_ocr family) — system binary:
#   macOS:  brew install tesseract
#   Ubuntu: sudo apt-get install tesseract-ocr

# EasyOCR (optional, layout-robust alternative OCR). Pin numpy/opencv so the
# PyTorch/OpenCV wheels stay on a compatible NumPy 1.x build:
pip install easyocr "numpy==1.26.4" "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80"

# Cloud LLM (optional, Anthropic Claude as a ceiling reference):
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."   # your key; note the image is sent to the API
```

### 2. Add your documents

Drop each PDF and its ground-truth JSON (same basename) into
`data/raw/real_waste_invoices/`:

```
data/raw/real_waste_invoices/
  my_invoice_001.pdf
  my_invoice_001.json    # ground truth matching the task schema
  ...
```

The `real_pdf` loader rasterizes page 1 to a cached PNG on first run and
filters each ground-truth JSON to the fields the task requests. Target
fields are defined by the `waste_invoice_fields` task in
`configs/tasks.yaml` — edit it to match your documents.

### 3. Run the comparisons

```bash
# Start Ollama and pull the local models (once):
ollama serve &
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b

# OCR + text SLM — Tesseract vs EasyOCR, model held fixed:
slmbench run --dataset real_waste_invoices --multimodal none \
  --text-slm qwen2.5-3b --ocr-engine tesseract --ocr-engine easyocr

# Local multimodal VLM (full-page image; allow more time on CPU):
SLMBENCH_OLLAMA_TIMEOUT=600 slmbench run --dataset real_waste_invoices \
  --multimodal qwen2.5-vl-3b --text-slm none

# Cloud LLM ceiling (Claude):
slmbench run --dataset real_waste_invoices --multimodal claude-opus-5 --text-slm none

# Read results:
cat results/leaderboard.md
```

`--multimodal none` / `--text-slm none` is a convenience: an unknown id
disables that whole family, so you can run one family at a time.

### 4. Field-count degradation curve (optional)

Same PDFs, growing scalar field set — isolates "how many fields" from "which
fields". The curve datasets share the base data dir via symlinks:

```bash
for n in 2f 4f 5f 6f 7f; do
  ln -sfn real_waste_invoices "data/raw/real_waste_invoices_$n"
done

for d in real_waste_invoices_2f real_waste_invoices_4f real_waste_invoices_5f \
         real_waste_invoices_6f real_waste_invoices_7f; do
  slmbench run --dataset "$d" --multimodal none --text-slm qwen2.5-3b --ocr-engine tesseract
done
```

### Tunable environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SLMBENCH_OLLAMA_TIMEOUT` | 300 | HTTP timeout (s) for Ollama — raise for slow CPU VLM runs |
| `SLMBENCH_OLLAMA_NUM_CTX` | 8192 | Ollama context window — raise if a full-page image overflows it |
| `SLMBENCH_VLM_MAX_DIM` | 1400 | Longest edge (px) the VLM image is downscaled to — raise for small text |
| `SLMBENCH_ANTHROPIC_MAX_TOKENS` | 8192 | Max output tokens for the Anthropic backend |

> **Reproducibility note:** all backends set `temperature: 0`, but on CPU that
> does not fully guarantee determinism (parallel float-reduction order can flip
> near-tied tokens). For stable numbers, run each configuration several times
> and/or set `OLLAMA_NUM_THREADS=1` — and label enough documents (n=2 is a
> smoke test, not a measurement).


## Contributing

PRs adding a dataset loader, a model config entry, or a new metric are all
welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
