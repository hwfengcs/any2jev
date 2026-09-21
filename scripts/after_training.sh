#!/usr/bin/env bash
# Runs after the public-data training finishes: baseline, latency, Snake, README numbers.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=runs/pipeline.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

until grep -q "^EXIT" runs/public_train.log 2>/dev/null; do sleep 30; done
log "public training finished: $(grep '^EXIT' runs/public_train.log)"

log "zero-shot baseline"
python examples/baseline_zeroshot.py Qwen/Qwen3-0.6B --data data/public/test.jsonl --val data/public/val.jsonl \
  --val-n 200 --out runs/qwen3-0.6b-public/baseline_zeroshot.json >> "$LOG" 2>&1 || log "baseline FAILED"

log "prompted-JSON baseline (generate)"
python examples/baseline_generate_json.py Qwen/Qwen3-0.6B --data data/public/test.jsonl   --out runs/qwen3-0.6b-public/baseline_generate_json.json >> "$LOG" 2>&1 || log "generate baseline FAILED"

log "split-screen recording + GIF"
python examples/record_vs.py --base Qwen/Qwen3-0.6B --checkpoint runs/qwen3-0.6b-public >> "$LOG" 2>&1 || log "record_vs FAILED"
python examples/render_vs.py --recording runs/vs_recording.json --out docs/vs.gif --slow 4 >> "$LOG" 2>&1 || log "render_vs FAILED"

log "latency"
python - >> "$LOG" 2>&1 <<'PY' || log "latency FAILED"
import json, subprocess, sys
rows = []
for dtype in ("fp32", "bf16"):
    out = subprocess.run([sys.executable, "examples/bench_latency.py", "runs/qwen3-0.6b-public", "--dtype", dtype, "--n", "40"],
                         capture_output=True, text=True, encoding="utf-8")
    line = out.stdout[out.stdout.find("{"):]
    rows.append(json.loads(line))
    print(rows[-1])
json.dump(rows, open("runs/latency.json", "w"), indent=2)
PY

log "snake: train"
any2jev train --base Qwen/Qwen3-0.6B --data data/snake/train.jsonl --val data/snake/val.jsonl --out runs/snake \
  --epochs 1 --batch-size 8 --max-state 256 >> "$LOG" 2>&1 || log "snake train FAILED"
log "snake: eval"
any2jev eval runs/snake --data data/snake/test.jsonl --out runs/snake/eval.json --n-perm 0 >> "$LOG" 2>&1 || log "snake eval FAILED"
log "snake: play + gif"
python examples/snake.py runs/snake --steps 300 --delay 0 --gif docs/snake.gif --seed 7 > runs/snake/play.log 2>&1 || log "snake play FAILED"
tail -2 runs/snake/play.log | tee -a "$LOG"

log "fill README"
python scripts/fill_readme.py --eval runs/qwen3-0.6b-public/eval.json \
  --baseline runs/qwen3-0.6b-public/baseline_zeroshot.json \
  --generate runs/qwen3-0.6b-public/baseline_generate_json.json \
  --vs runs/vs_recording.json --latency runs/latency.json \
  --snake-eval runs/snake/eval.json --snake-play runs/snake/play.log \
  --bases runs/qwen3-0.6b-public runs/qwen3.5-0.8b-synthetic >> "$LOG" 2>&1 || log "fill_readme FAILED"
log "PIPELINE DONE"
