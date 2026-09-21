"""``any2jev`` command line: convert -> train -> calibrate -> eval -> serve -> ask."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Turn any open model into a Jev-style System One decision model.", no_args_is_help=True)
data_app = typer.Typer(help="Build training data.", no_args_is_help=True)
app.add_typer(data_app, name="data")
console = Console()


@app.command()
def convert(
    base: str = typer.Argument(..., help="Hugging Face model id or local path of a causal LM"),
    out: Path = typer.Option(..., "--out", "-o", help="checkpoint directory to create"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    head_dim: int = typer.Option(256, help="pointer head dimension"),
    delimiters: str = typer.Option("auto", help="auto | reuse | add: how to obtain the 5 delimiter tokens"),
    dtype: str = typer.Option("fp32", help="fp32 | bf16 | fp16 backbone weights"),
    max_state: int = typer.Option(1024),
    max_branch: int = typer.Option(1024),
):
    """Model surgery only: drop the LM head, attach LoRA + pointer head, save an untrained checkpoint."""
    from .model import DecisionModel

    m = DecisionModel.from_base(base, lora_r=lora_r, head_dim=head_dim, delimiters=delimiters, dtype=dtype,
                                max_state=max_state, max_branch=max_branch)
    p = m.count_parameters()
    m.save(out)
    console.print(f"[green]converted[/] {base} -> {out}  mode={m.mode}  delimiters={m.delims.tokens} "
                  f"trainable={p['trainable']:,}/{p['total']:,}")


@app.command()
def train(
    base: str = typer.Option(..., help="base model id, or an any2jev checkpoint to continue from"),
    data: Path = typer.Option(..., help="train JSONL (labelled requests)"),
    out: Path = typer.Option(..., "--out", "-o"),
    val: Optional[Path] = typer.Option(None, help="validation JSONL; enables temperature scaling"),
    epochs: float = typer.Option(2.0),
    lr: float = typer.Option(2e-4),
    head_lr: float = typer.Option(1e-3),
    batch_size: int = typer.Option(4),
    grad_accum: int = typer.Option(1),
    lora_r: int = typer.Option(16),
    head_dim: int = typer.Option(256),
    dtype: str = typer.Option("fp32"),
    max_state: int = typer.Option(1024),
    max_branch: int = typer.Option(1024),
    brier_weight: float = typer.Option(0.0, help="auxiliary multiclass Brier loss weight"),
    ordinal_weight: float = typer.Option(0.0, help="ranked-probability-score weight for Score questions"),
    delimiters: str = typer.Option("auto"),
    seed: int = typer.Option(0),
    max_steps: Optional[int] = typer.Option(None),
    no_calibrate: bool = typer.Option(False, help="skip temperature scaling even when --val is given"),
):
    """Fine-tune LoRA + pointer head on labelled requests; fit a temperature on --val."""
    from .train import TrainConfig
    from .train import train as _train

    cfg = TrainConfig(base=base, data=str(data), out=str(out), val=str(val) if val else None, epochs=epochs, lr=lr,
                      head_lr=head_lr, batch_size=batch_size, grad_accum=grad_accum, lora_r=lora_r, head_dim=head_dim,
                      dtype=dtype, max_state=max_state, max_branch=max_branch, brier_weight=brier_weight,
                      ordinal_weight=ordinal_weight, delimiters=delimiters, seed=seed, max_steps=max_steps,
                      calibrate=not no_calibrate)
    _train(cfg, log=console.print)


@app.command()
def calibrate(model_dir: Path = typer.Argument(...), data: Path = typer.Option(..., help="validation JSONL")):
    """Fit temperature scaling on held-out data and store it in the checkpoint."""
    from .train import calibrate as _calibrate

    _calibrate(model_dir, data, log=console.print)


@app.command("eval")
def eval_cmd(
    model_dir: Path = typer.Argument(...),
    data: Path = typer.Option(..., help="test JSONL"),
    out: Optional[Path] = typer.Option(None, help="write the full report as JSON"),
    n_perm: int = typer.Option(4, help="option orders per Choice question for the permutation test (0 = skip)"),
    dtype: Optional[str] = typer.Option(None),
):
    """Accuracy, NLL, Brier, ECE, AURC per question type, plus order-sensitivity and isolation checks."""
    from .evaluate import evaluate_checkpoint

    rep = evaluate_checkpoint(model_dir, data, out, dtype=dtype, n_perm=n_perm)
    table = Table(title=f"any2jev eval  (T={rep['temperature']:.3f}, {rep['n_questions']} questions)")
    for c in ("group", "n", "acc", "nll", "brier", "ECE", "AURC", "cov@5%"):
        table.add_column(c, justify="right" if c != "group" else "left")
    for g, m in rep["metrics"].items():
        table.add_row(g, str(m["n"]), f"{m['accuracy']:.3f}", f"{m['nll']:.3f}", f"{m['brier']:.3f}", f"{m['ece']:.3f}",
                      f"{m['aurc']:.3f}", f"{m['coverage_at_5pct_risk']:.2f}")
    console.print(table)
    if "permutation" in rep and rep["permutation"].get("n"):
        p = rep["permutation"]
        console.print(f"option-order test: argmax stable {p['argmax_stable_rate']:.0%}, mean max spread {p['mean_max_spread']:.3f}")
    if "isolation" in rep:
        console.print(f"isolation check: max |packed - separate| = {rep['isolation']['max_abs_prob_diff']:.2e}")


@app.command()
def serve(
    model_dir: str = typer.Argument(..., help="checkpoint directory, or hf://user/repo"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8009),
    dtype: Optional[str] = typer.Option(None, help="override backbone dtype, e.g. bf16 for serving"),
    device: Optional[str] = typer.Option(None),
    model_name: str = typer.Option("any2jev-latest"),
):
    """Serve a Jev-compatible API (POST /v1/systemone, GET /v1/models)."""
    from .hub import resolve_model_dir
    from .serve import serve as _serve

    _serve(resolve_model_dir(model_dir), host=host, port=port, device=device, dtype=dtype, model_name=model_name)


@app.command()
def push(
    model_dir: Path = typer.Argument(...),
    repo: str = typer.Option(..., help="Hugging Face repo id, e.g. user/any2jev-qwen3-0.6b"),
    private: bool = typer.Option(False),
):
    """Upload a checkpoint (adapter, head, tokenizer, config, model card) to the Hugging Face Hub."""
    from .hub import push as _push

    console.print(f"[green]pushed[/] {_push(model_dir, repo, private)}")


def _parse_q(spec: str, qtype: str) -> dict:
    instr, _, opts = spec.partition("|")
    q: dict = {"type": qtype, "instructions": instr.strip()}
    items = [o.strip() for o in opts.split(",") if o.strip()]
    if qtype == "choice":
        q["criteria"] = {o: None for o in items}
    elif qtype == "score":
        q["criteria"] = items
    return q


@app.command()
def ask(
    model_dir: str = typer.Argument(..., help="checkpoint directory, or hf://user/repo"),
    state: str = typer.Option("", help="the state text (or a path to a .txt/.json file)"),
    choice: list[str] = typer.Option([], help='"instructions | opt1, opt2, ..." (repeatable)'),
    noul: list[str] = typer.Option([], help='"yes/no question" (repeatable)'),
    score: list[str] = typer.Option([], help='"instructions | level0, level1, ..." (repeatable)'),
    request: Optional[Path] = typer.Option(None, help="a full /v1/systemone request JSON file instead of flags"),
    dtype: Optional[str] = typer.Option(None),
):
    """Ask a checkpoint directly, without starting a server."""
    from .hub import resolve_model_dir
    from .model import DecisionModel

    model_dir = resolve_model_dir(model_dir)
    if request:
        body = json.loads(Path(request).read_text(encoding="utf-8"))
    else:
        if not state:
            raise typer.BadParameter("--state is required unless --request is given")
        p = Path(state)
        st = (json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else p.read_text(encoding="utf-8")) if p.is_file() else state
        qs: dict = {}
        for i, c in enumerate(choice):
            qs[f"choice_{i}"] = _parse_q(c, "choice")
        for i, n in enumerate(noul):
            qs[f"noul_{i}"] = {"type": "noul", "instructions": n}
        for i, s in enumerate(score):
            qs[f"score_{i}"] = _parse_q(s, "score")
        if not qs:
            raise typer.BadParameter("give at least one --choice / --noul / --score, or --request")
        body = {"state": st, "questions": qs}
    m = DecisionModel.load(model_dir, dtype=dtype)
    t0 = time.perf_counter()
    answers, n_in = m.decide(body)
    ms = (time.perf_counter() - t0) * 1000
    console.print_json(json.dumps({"answers": answers, "usage": {"input_tokens": n_in}, "latency_ms": round(ms, 1)}))


@data_app.command("synthetic")
def data_synthetic(
    out: Path = typer.Option(Path("data"), help="directory for train/val/test JSONL"),
    n: int = typer.Option(2000, help="total records"),
    seed: int = typer.Option(0),
):
    """Generate synthetic support-ticket decisions (no download): 80/10/10 train/val/test."""
    from .data import save_jsonl, synthetic_records

    recs = synthetic_records(n, seed)
    n_val = n_test = max(1, n // 10)
    save_jsonl(out / "train.jsonl", recs[: n - n_val - n_test])
    save_jsonl(out / "val.jsonl", recs[n - n_val - n_test : n - n_test])
    save_jsonl(out / "test.jsonl", recs[n - n_test :])
    console.print(f"[green]wrote[/] {out}/train.jsonl val.jsonl test.jsonl ({n} records)")


@data_app.command("build")
def data_build(
    sources: str = typer.Option("boolq,ag_news,yelp", help="comma-separated public sources"),
    out: Path = typer.Option(Path("data")),
    n_per_source: int = typer.Option(1000),
    n_test: int = typer.Option(200, help="test records per source (from the source's test split)"),
    seed: int = typer.Option(0),
):
    """Convert public labelled datasets (needs `pip install any2jev[data]`)."""
    from .data import build_public, save_jsonl, split_records

    srcs = [s.strip() for s in sources.split(",") if s.strip()]
    train_val = build_public(srcs, "train", n_per_source, seed)
    train, val = split_records(train_val, 0.1, seed)
    test = build_public(srcs, "test", n_test, seed + 1)
    save_jsonl(out / "train.jsonl", train)
    save_jsonl(out / "val.jsonl", val)
    save_jsonl(out / "test.jsonl", test)
    console.print(f"[green]wrote[/] {len(train)} train / {len(val)} val / {len(test)} test records to {out}")


if __name__ == "__main__":
    app()
