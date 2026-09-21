# How every number in the README is produced

The README carries no figure that a script in this repository did not write. This page lists the scripts,
what each measures, and where its output lands. `scripts/fill_readme.py` reads those outputs and rewrites the
blocks between `<!-- RESULTS:… -->` markers in both READMEs; `scripts/after_training.sh` runs the whole chain.

| README block | script | measures | output |
|---|---|---|---|
| hero GIF | `examples/record_vs.py` → `examples/render_vs.py` | the same base weights answering one 3-question ticket: (a) prompted for JSON and decoded with `generate()`, per-token timestamps from a streamer; (b) label-logit reading, one prefill per question; (c) any2jev, one forward pass. Median of repeated runs, peak VRAM per method. | `runs/vs_recording.json`, `docs/vs.gif` |
| hero table | `examples/baseline_generate_json.py` | the base model prompted for a one-key JSON answer on every test question: format failure rate, accuracy, p50 latency | `runs/…/baseline_generate_json.json` |
| | `examples/baseline_zeroshot.py` | label-logit reading on every test question, with and without a temperature fitted on validation | `runs/…/baseline_zeroshot.json` |
| | `any2jev eval` | the converted model: accuracy, NLL, Brier, ECE, AURC, coverage at 5% risk, option-order sensitivity, isolation check | `runs/…/eval.json` |
| | `examples/bench_cloud.py` | a hosted model through any OpenAI-compatible endpoint, same prompt and parser as the local baseline. Not run by the maintainers; bring your own key. | `runs/…/baseline_cloud_<model>.json` |
| latency table | `examples/bench_latency.py` | steady-state single-request latency, p50/p95 over 60 runs, several dtypes and sequence lengths | `runs/latency.json` |
| Snake | `examples/snake.py --generate`, `any2jev train`, `examples/snake.py --gif` | BFS-teacher accuracy on held-out moves and one recorded game | `runs/snake/eval.json`, `docs/snake.gif` |
| verified bases | `any2jev train` + `any2jev eval` per base | accuracy, ECE, train time, trainable parameters | `runs/<base>/{train_report,eval}.json` |

Ground rules:

* The comparison rows share the base weights, the GPU, the dtype and the request. Architectural differences are
  the only variable.
* "Format failure" means the reply was not a JSON object with the requested key, or the value was outside the
  allowed set (an unknown option key, a non-boolean, an out-of-range level). any2jev reports 0% because its
  output is assembled in code from a probability vector over the allowed options; it cannot produce anything else.
* GPU-hours per million requests is `latency_ms × 1e6 / 3.6e6`, serial, single request. Batching lowers it for
  every method; the ratio between methods is what matters.
* The cloud row is intentionally empty. Publishing a latency for a service we did not measure would be a guess.

## Inference optimization check, 2026-09-21

The existing Qwen3-0.6B public checkpoint, fp32, RTX 2060 SUPER, 116 tokens and 3 questions.
`examples/bench_latency.py` warms up 5 times, then measures 60 requests per variant:

| variant | p50 ms | p95 ms | peak allocated MiB |
|---|---:|---:|---:|
| default (unused KV cache disabled) | 40.4 | 46.2 | 2918.4 |
| `--merge` | 31.3 | 32.3 | 2291.5 |

Reproduce with the following commands; the recorded Windows runs used 4 CPU threads
(`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`):

```bash
python examples/bench_latency.py runs/qwen3-0.6b-public --n 60
python examples/bench_latency.py runs/qwen3-0.6b-public --n 60 --merge
```

Merging changed no highest-probability option on the 1,000-question public test set; maximum probability
difference was 7.09e-6. Accuracy remained 0.796 and ECE 0.027014. The checkpoint files were unchanged.
[Measurement summary and checksums](experiments/2026-09-21.json) and
[the investigation notes](exploration-2026-09-21.md) record the scope and other correctness checks.

The original comparison table retains its historical reports. Adaptive ECE and selective metrics now
handle confidence ties correctly; rerunning older baselines can therefore change those metrics even
when the model's predictions are unchanged.
