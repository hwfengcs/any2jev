# Contributing

Thanks for helping make open decision models better. The bar is simple: every claim in the README
must be backed by a command in this repo that reproduces it.

## Dev setup

```bash
git clone https://github.com/hwfengcs/any2jev && cd any2jev
pip install -e ".[dev,compat]"
pytest -q          # ~1 minute on CPU: a 2-layer random Qwen3 stands in for a real base
ruff check src tests examples scripts
```

## Where things live

| path | what |
|---|---|
| `src/any2jev/schema.py` | TypeSafe wire contract, question → option mapping, answer building |
| `src/any2jev/packing.py` | delimiters, packed layout, block-causal mask, rows fallback |
| `src/any2jev/model.py` | backbone extraction, LoRA wrap, pointer head, save/load |
| `src/any2jev/train.py` | loss, training loop, temperature scaling |
| `src/any2jev/calibration.py` | ECE / Brier / AURC / temperature fitting (numpy only) |
| `src/any2jev/evaluate.py` | per-type metrics, permutation test, isolation check |
| `src/any2jev/serve.py` | FastAPI `/v1/systemone` |
| `src/any2jev/data.py` | JSONL format, synthetic generator, public dataset builders |

## Good first contributions

* A new public dataset builder in `data.py` (one function, add it to `PUBLIC_SOURCES`).
* Delimiter reuse sets for more tokenizer families in `packing.py::REUSABLE_DELIM_SETS`.
* Verified results for another base model (open a PR with the `eval.json` and the exact commands).
* A benchmark adapter for [jevbench](https://github.com/fstandhartinger/jevbench).

## Rules of thumb

* Keep train and serve byte-identical: anything that changes how a request becomes tokens must go
  through `schema.py` / `packing.py`, never a prompt string in a script.
* Any change to the mask or the packing needs the isolation test to stay green
  (`tests/test_model.py::test_packed_equals_rows_equals_alone`).
* Report calibration, not just accuracy.
