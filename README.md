<div align="center">

# any2jev

**Turn any open model into a Jev-style System One decision model.**
One forward pass. Typed answers. Calibrated probabilities. No text generation.

[![CI](https://img.shields.io/github/actions/workflow/status/any2jev/any2jev/ci.yml?branch=main&label=ci)](https://github.com/any2jev/any2jev/actions)
[![PyPI](https://img.shields.io/pypi/v/any2jev)](https://pypi.org/project/any2jev/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[中文文档](README_zh.md) · [How it works](docs/architecture.md) · [Data format](docs/data-format.md)

</div>

```
any2jev train --base Qwen/Qwen3-0.6B --data train.jsonl --val val.jsonl --out runs/my-jev
any2jev serve runs/my-jev            #  POST /v1/systemone, drop-in for the TypeSafe SDK
```

TypeSafe's [Jev](https://typesafe.ai) showed that a model which **decides** instead of **writes** is
40 to 200× faster and cheaper for routing, moderation, triage, ranking, guardrails and game agents.
Jev is closed. `any2jev` is the converter: point it at a Hugging Face checkpoint (Qwen, Llama, Gemma,
SmolLM, Phi, Mistral, ...) plus a few thousand labelled decisions, and get a model that answers
**Choice / Score / Noul** questions in one pass, with probabilities you can threshold.

## Why not just prompt an LLM for JSON?

| | LLM + JSON schema | zero-shot logit reading | **any2jev** |
|---|---|---|---|
| Latency | seconds (autoregressive) | one pass | **one pass, all questions at once** |
| Output | text to parse and validate | probabilities of label tokens | **typed answers, always valid** |
| Probabilities calibrated? | no ("90% sure" is just text) | no (base-model logits) | **trained + temperature scaled, ECE reported** |
| Many questions per state | N× the cost | N× the prefill | **state encoded once, questions isolated** |
| Works with any base model | yes | mostly | **yes, LoRA + a 0.5M-param head** |

## What you get

```
                 ┌─────────────────────── one forward pass ───────────────────────┐
  state ────────►│ <state> …  │ <q> which team? <opt>billing</opt><opt>shipping</opt> <decide> │──► {billing: 0.91, shipping: 0.09}, confidence 0.86
                 │            │ <q> urgent?     <opt>no</opt><opt>yes</opt>          <decide> │──► noul 0.97
                 │            │ <q> how angry?  <opt>calm</opt> … <opt>furious</opt> <decide> │──► score 1.4, {0: .05, 1: .5, 2: .45}
                 └─────────────── questions see the state, never each other ─────────────────┘
```

* **Model surgery** — the vocabulary head is dropped; a pointer head scores each option's hidden state
  against a decision token. Works on any `AutoModelForCausalLM`, including hybrid backbones (Qwen3.5).
* **Block-causal packing** — the state is encoded once; every question attends to it and to itself
  only, so answers never depend on sibling questions (verified by a test, not a promise).
* **Training** — LoRA + head + delimiter embeddings, cross-entropy plus optional Brier / ordinal terms,
  option shuffling for order robustness, then **temperature scaling** on held-out data.
* **Evaluation that matches the claims** — accuracy, NLL, Brier, ECE, AURC and coverage-at-risk per
  question type, an option-order sensitivity test and a packed-vs-separate isolation check.
* **Jev-compatible server** — `POST /v1/systemone` and `GET /v1/models` with TypeSafe's exact wire
  format. The official `typesafe-sdk` runs against it with only `base_url` changed.

## Quick start

```bash
pip install any2jev[serve]              # + [data] for public datasets, [compat] for the TypeSafe SDK

# 1. data: 2,000 synthetic support tickets (no download), or convert public datasets
any2jev data synthetic --out data/synthetic --n 2000
any2jev data build --sources boolq,ag_news,banking77,sst5 --out data/public --n-per-source 1500

# 2. train: surgery + LoRA + head + temperature scaling in one command (~7 min on an 8 GB GPU for 0.6B)
any2jev train --base Qwen/Qwen3-0.6B --data data/public/train.jsonl --val data/public/val.jsonl --out runs/qwen3-0.6b

# 3. evaluate: calibration, order sensitivity, isolation
any2jev eval runs/qwen3-0.6b --data data/public/test.jsonl

# 4. ask, or serve
any2jev ask runs/qwen3-0.6b --state "My payouts have failed 3 days in a row, fix this ASAP" \
    --choice "Which team? | billing, technical, sales" --noul "Is this urgent?" \
    --score "How frustrated? | calm, frustrated, furious"
any2jev serve runs/qwen3-0.6b --port 8009
```

Then, with the official SDK:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="any2jev-latest")
r = client.system_one(
    state="Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    questions={
        "department": Choice(instructions="Which team should handle this?",
                             criteria={"returns": "Exchanges, refunds, wrong or damaged items",
                                       "shipping": "Delivery status, delays, lost packages",
                                       "billing": "Charges, invoices, payment problems"}),
        "escalate": Noul(instructions="Does this need urgent human attention?"),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"]),
    })
r.choices["department"].probabilities   # {'returns': 0.47, 'shipping': 0.28, 'billing': 0.25}
r.nouls["escalate"].noul                # 0.93
r.scores["frustration"].score           # 1.44
```

Or raw HTTP, identical to Jev's contract:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @examples/request.json
```

## Results

Qwen3-0.6B, LoRA r=16, one epoch, one RTX 2060 SUPER (8 GB). Held-out test splits, temperature fitted
on validation. `any2jev eval` prints these tables; the JSON reports are in `runs/`.

<!-- RESULTS:PUBLIC -->
_(public-data run in progress; numbers land here)_
<!-- /RESULTS:PUBLIC -->

Synthetic support tickets (2,000 records, 4 question types, rule-based labels) are solved to
accuracy 1.000 / ECE 0.001 in 7 minutes; the option-order test is 100% stable and the isolation check
shows packed and separate answers agree to 1e-6. That run is the smoke test, not the benchmark.

Latency (steady state, single request, 3 questions, `examples/bench_latency.py`):

<!-- RESULTS:LATENCY -->
_(measured after the public run finishes)_
<!-- /RESULTS:LATENCY -->

## How it works

See [docs/architecture.md](docs/architecture.md). In short:

1. `extract_backbone()` keeps the decoder stack of a `*ForCausalLM` and discards `lm_head`.
2. Five delimiter tokens (`<state> <q> <opt> </opt> <decide>`) are reused from the tokenizer's spare
   specials or added. User text is sanitised so it can never forge one.
3. State + questions are packed into one sequence under a **block-causal mask**; branch positions
   restart after the state. Hybrid (linear-attention) backbones fall back to one causal row per question.
4. A **pointer head** scores `h(</opt>_k) · h(<decide>)`; softmax with a fitted temperature.
5. Noul, Choice and Score are all the same primitive with different option lists; `confidence` uses
   TypeSafe's published formulas.

## Which base model?

Anything `AutoModelForCausalLM` loads. Tested: Qwen3-0.6B (attention-only, packed mode) and the
Qwen3.5 family (hybrid, rows mode). Delimiter reuse is built in for Qwen, Llama 3 and Gemma tokenizers;
other tokenizers get five new tokens. Rough VRAM for training in fp32 with LoRA: 0.6B → 8 GB,
1.7B → 16 GB, 4B → 24 GB (use `--dtype bf16` to halve it).

## Roadmap

- [ ] RLCR-style RL stage (`correct − (confidence − correct)²` reward) on top of the supervised recipe
- [ ] State-prefix KV cache in the server (exact, thanks to the block-causal mask)
- [ ] vLLM / SGLang backend for large bases; ONNX export for the small ones
- [ ] Encoder backbones (ModernBERT) through the same interface
- [ ] Pretrained adapters on the Hub for Qwen3 0.6B / 1.7B / 4B

## Related work

Jev is [TypeSafe AI](https://typesafe.ai)'s model; this project is independent and not affiliated.
The packed-question architecture follows Archer Hume's
[*Jev's Architecture Unmasked*](https://archerhume.com/posts/jevs-architecture-unmasked) and the
[kev](https://github.com/jaredpalmer/kev) reproduction (Apache-2.0), which trains a fixed Qwen family;
`any2jev` generalises the recipe into a converter for arbitrary bases with an evaluation and calibration
toolkit. [rlcd-lite](https://github.com/arnabgho/rlcd-lite) informed the calibration objective.
[awesome-jev](https://github.com/OmniJev/awesome-jev) lists the wider ecosystem.

## License

Apache-2.0.
