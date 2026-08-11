# Adding a dataset

Adding a dataset has two parts: a **registry entry** (config) and a
**loader module** (a little bit of code — this part is dataset-specific by
nature, since every dataset ships its raw annotations in its own format).

## 1. Register it in `configs/datasets.yaml`

```yaml
  - id: my_dataset
    domain: invoices              # receipts | forms | invoices | contracts | general
    task: invoice_fields          # must match an entry in configs/tasks.yaml
    loader: my_dataset            # matches datasets/loaders/my_dataset.py
    source: https://example.com/my-dataset
    license: CC BY 4.0
    n_samples: ~1000
    notes: Why this dataset is worth including.
```

If none of the existing tasks in `configs/tasks.yaml` fit your dataset's
target schema, add a new one there first (it's just a JSON Schema + a
`description:` used in the prompt).

## 2. Implement the loader

Copy `src/slmbench/datasets/loaders/_template.py` to
`src/slmbench/datasets/loaders/my_dataset.py` and implement `load()`:

```python
def load(raw_dir: Path, split: str, limit: int | None, dataset_id: str) -> list[DocumentSample]:
    ...
```

Your job is to map the dataset's raw annotation format into
`DocumentSample.ground_truth`, shaped to match the JSON Schema of the task
you registered. Two complete reference implementations:

- `loaders/sroie.py` — simple flat key-value ground truth (receipt header
  fields).
- `loaders/cord.py` — nested ground truth with a list of line items,
  showing how to map a dataset's own annotation schema into ours.

`loaders/synthetic.py` is a third pattern worth looking at: a loader that
*generates* data instead of parsing a download, useful if you want a
controlled dataset for ablations (rotation, blur, language, layout noise)
rather than a fixed real-world one.

## 3. Wire up the download step (if the data has a stable URL)

If the dataset can be fetched with a plain URL, add it to
`DIRECT_DOWNLOAD_URLS` in `scripts/download_datasets.py`. If it requires
manual registration (like SROIE or DocILE), add a short explanation to
`MANUAL_INSTRUCTIONS` in the same file instead — don't try to automate
around a host's registration wall.

## 4. Verify

```bash
python scripts/download_datasets.py --dataset my_dataset
slmbench run --dataset my_dataset --limit 5 --multimodal smolvlm2-2.2b
```

If `load()` raises, the CLI will show the traceback — the most common
mistake is a ground-truth shape that doesn't quite match the task schema's
field names (the evaluator will just silently score those fields as
"missing" rather than crash, so double-check field names carefully against
`configs/tasks.yaml`).

## Currently-stubbed datasets (good first contributions)

`funsd`, `xfund`, `docile`, `docvqa`, `kleister` are registered in
`configs/datasets.yaml` but their loaders raise `NotImplementedError`.
Picking one of these up is a great, self-contained first PR — the dataset
research is already done (see the `source:` field for each), only the
parsing logic is missing.
