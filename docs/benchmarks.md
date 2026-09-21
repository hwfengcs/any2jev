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
