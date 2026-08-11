# Adding a model

In almost all cases this is a **config-only change** — no Python needed.

## Adding a multimodal (VLM) model

Add an entry under `multimodal:` in `configs/models.yaml`:

```yaml
  - id: my-new-vlm
    backend: ollama            # or: transformers
    ollama_tag: my-new-vlm:7b  # required if backend is ollama
    # hf_repo: org/my-new-vlm  # required if backend is transformers
    params: 7B
    notes: Why this model is interesting to benchmark.
```

Then:

```bash
ollama pull my-new-vlm:7b   # if backend: ollama
slmbench run --dataset synthetic_invoices_ptbr --limit 5 --multimodal my-new-vlm
```

## Adding a text SLM (for the text_ocr family)

Add an entry under `text_ocr.slms:` in `configs/models.yaml`, same shape as
above. It will automatically be benchmarked against every registered OCR
engine (`text_ocr.ocr_engines:`) unless you restrict with `--ocr-engine`.

## Adding a new OCR engine

Add an entry under `text_ocr.ocr_engines:`, then implement the actual call
in `src/slmbench/models/ocr/engines.py::run_ocr()` — this is the one part
of "adding a model" that does need a small code change, since each OCR
library has a different Python API.

## Adding a new backend (beyond ollama / transformers)

This does require code:

1. Add a module in `src/slmbench/models/runtime/`, exposing
   `generate_vision(...)` and/or `generate_text(...)` matching the
   signatures in `ollama_backend.py` / `transformers_backend.py`.
2. Add a branch for it in `models/multimodal/extractor.py::_generate()`
   and/or `models/text_ocr/extractor.py::_generate()`.
3. Document the new `backend:` value here.

`llama_cpp` (raw GGUF via `llama-cpp-python`, useful for models Ollama
hasn't packaged yet) is a natural next backend to add — contributions
welcome.

## A note on keeping the model list current

Small/local models move fast — a model that's SOTA-for-its-size this
quarter may be superseded within months. `configs/models.yaml` includes a
`notes:` field for exactly this reason: if a model in the registry is
stale, outdated, or its Ollama tag no longer resolves, please open a PR
updating or removing it rather than assuming it still reflects the current
landscape.
