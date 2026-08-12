"""slmbench CLI.

    slmbench list-datasets
    slmbench list-models
    slmbench run --dataset synthetic_invoices_ptbr --limit 20
    slmbench run --dataset cord --multimodal qwen2.5-vl-3b --multimodal internvl3-2b
    slmbench report results/20260811T120000Z_cord.json
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from slmbench.datasets.registry import list_datasets, load_dataset
from slmbench.evaluation.metrics import aggregate, score_sample
from slmbench.evaluation.report import write_report
from slmbench.extraction.pipeline import build_extractors, run_extractor_on_samples

app = typer.Typer(add_completion=False)
console = Console()


@app.command("list-datasets")
def list_datasets_cmd() -> None:
    for dataset_id in list_datasets():
        console.print(f"  • {dataset_id}")


@app.command("list-models")
def list_models_cmd() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    console.print("[bold]multimodal[/bold]")
    for entry in cfg["multimodal"]:
        console.print(f"  • {entry['id']}  ({entry['params']}, backend={entry['backend']})")

    console.print("\n[bold]text_ocr — ocr_engines[/bold]")
    for entry in cfg["text_ocr"]["ocr_engines"]:
        console.print(f"  • {entry['id']}")

    console.print("\n[bold]text_ocr — slms[/bold]")
    for entry in cfg["text_ocr"]["slms"]:
        console.print(f"  • {entry['id']}  ({entry['params']}, backend={entry['backend']})")


@app.command("run")
def run_cmd(
    dataset: str = typer.Option(..., help="Dataset id from configs/datasets.yaml"),
    split: str = typer.Option("test"),
    limit: int | None = typer.Option(None, help="Max samples (useful for a quick smoke test)"),
    multimodal: list[str] = typer.Option(None, help="Restrict to these multimodal model ids"),
    text_slm: list[str] = typer.Option(None, help="Restrict to these text_ocr SLM ids"),
    ocr_engine: list[str] = typer.Option(None, help="Restrict to these OCR engine ids"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-field mismatch details"),
) -> None:
    console.print(f"[bold]Loading dataset[/bold] {dataset} (split={split}, limit={limit})")
    samples = load_dataset(dataset, split=split, limit=limit)
    console.print(f"  {len(samples)} samples loaded")

    extractors = build_extractors(
        multimodal_ids=multimodal or None,
        text_slm_ids=text_slm or None,
        ocr_engine_ids=ocr_engine or None,
    )
    console.print(f"[bold]Running {len(extractors)} extractor(s)[/bold]: "
                  f"{', '.join(e.model_id for e in extractors)}")

    all_scores = []
    for extractor in extractors:
        console.print(f"\n[cyan]→ {extractor.model_id}[/cyan]")
        for result, sample in zip(
            run_extractor_on_samples(extractor, samples), samples, strict=True
        ):
            score = score_sample(
                sample_id=result.sample_id,
                model_id=result.model_id,
                predicted=result.parsed_output,
                expected=sample.ground_truth,
                json_valid=result.json_valid,
                latency_seconds=result.latency_seconds,
            )
            all_scores.append(score)
            status = "✓" if result.json_valid else "✗"
            console.print(f"    {status} {sample.sample_id}  f1={score.field_f1:.2f}"
                          f"  ({result.latency_seconds:.1f}s)"
                          + (f"  [red]{result.error}[/red]" if result.error else ""))

            mismatches = [f for f in score.field_scores if not f.match]
            if mismatches and result.json_valid and not quiet:
                for f in mismatches:
                    console.print(
                        f"        [dim]· {f.field}: got {f.predicted!r}, expected {f.expected!r}[/dim]"
                    )

    summary = aggregate(all_scores)
    _print_summary_table(summary)
    out_path = write_report(summary, dataset_id=dataset)
    console.print(f"\n[bold green]Saved[/bold green] {out_path}")
    console.print("[bold green]Updated[/bold green] results/leaderboard.md")


@app.command("report")
def report_cmd(results_json: Path) -> None:
    import json

    summary = json.loads(results_json.read_text())
    _print_summary_table(summary)


def _print_summary_table(summary: dict) -> None:
    table = Table(title="Results")
    for col in ["Model", "Field F1", "Exact Match", "JSON Valid %", "Mean Latency (s)", "p95 (s)"]:
        table.add_column(col)

    rows = sorted(summary.items(), key=lambda kv: kv[1]["mean_field_f1"], reverse=True)
    for model_id, stats in rows:
        table.add_row(
            model_id,
            f"{stats['mean_field_f1']:.3f}",
            f"{stats['exact_match_rate']:.3f}",
            f"{stats['json_valid_rate'] * 100:.1f}%",
            f"{stats['mean_latency_seconds']:.2f}",
            f"{stats['p95_latency_seconds']:.2f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()