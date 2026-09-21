# How any2jev works

any2jev turns a causal language model into a **decision model**: state in, typed probability
distributions out, in one forward pass. The design follows the community reconstruction of TypeSafe's
Jev (Archer Hume's *Jev's Architecture Unmasked* and the `kev` reproduction), with one change: any2jev
is a converter that works on any Hugging Face causal LM rather than one trained model family.

## 1. Surgery

```
AutoModelForCausalLM ──► extract_backbone() ──► decoder stack only  (lm_head is dropped)
                                                    │
                                                    ├── LoRA on attention + MLP projections (peft)
                                                    ├── 5 delimiter embeddings made trainable
                                                    └── PointerHead(d_model → d_ptr)  (new, ~0.5M params)
```

* `extract_backbone` uses `get_decoder()` when the model exposes it, otherwise the usual `.model` /
  `.transformer` attribute. The vocabulary projection never loads into the decision model.
* Delimiters: five special tokens mark `<state>`, `<q>`, `<opt>`, `</opt>`, `<decide>`. If the tokenizer
  already has rarely used specials (Qwen `<|fim_*|>`/`<|box_*|>`, Llama `<|reserved_special_token_N|>`,
  Gemma `<unusedN>`), they are reused; otherwise five new tokens are added and the embedding matrix is
  resized. Caller text is sanitised so it can never produce a delimiter (option boundaries are unforgeable).

## 2. Packing and the block-causal mask

```
<state> s1 … sS │ <q> instr <opt> o1 </opt> <opt> o2 </opt> <decide> │ <q> instr <opt> … <decide> │ …
  seg 0          │ seg 1                                               │ seg 2                      │
  pos 0..S-1     │ pos S..                                             │ pos S..  (restart)         │
```

Attention rule: token *i* may attend token *j* iff `j <= i` and `seg[j] ∈ {0, seg[i]}`.

* Every question sees the whole state and itself, never a sibling question. Answers therefore do not
  depend on which other questions are in the request (`tests/test_model.py::test_packed_equals_rows_equals_alone`).
* Branch position ids restart right after the state, so a question's representation is the same whether
  it is first or fifth in the request.
* The state never attends to any branch, so the state's activations and KV cache are identical with or
  without questions. That is what makes prefix caching exact (planned for the server).
* Hybrid backbones with linear-attention / recurrent layers (Qwen3.5, Mamba-style) cannot honour a custom
  4D mask. any2jev detects them and falls back to **rows mode**: one causal row per question (state +
  branch), batched. The two modes agree to float noise on attention-only models.

## 3. Readout

`PointerHead` compares the hidden state at the question's `<decide>` token with the hidden state at
each option's `</opt>` token:

```
logit_k = ( W_k · h(</opt>_k) ) · ( W_q · h(<decide>) ) / sqrt(d_ptr)
p = softmax(logits / T)
```

The head is option-count agnostic (1 to 255 options), and because `<decide>` attends over the full
option list, options can interact ("none of the above" works). All three primitives map onto it:

| primitive | options fed to the model | answer built in code |
|---|---|---|
| Noul | `["no", "yes"]` (+ criteria descriptions) | `noul = p(yes)` |
| Choice | `"key: description"` per criterion | `choice = argmax`, `probabilities`, `confidence` |
| Score | ordered level descriptions | `score = Σ i·p_i`, `legend`, `probabilities`, `confidence` |

`confidence` uses TypeSafe's published statistics (Choice: `(p_max − 1/K)/(1 − 1/K)`; Score: one minus
the expected distance from the modal level, normalised by the uniform distribution's).

## 4. Training

* Loss per question: cross-entropy over option logits, optionally plus a multiclass Brier term
  (`--brier-weight`) and, for Score questions, a ranked-probability-score term (`--ordinal-weight`).
* Choice options are re-shuffled every epoch so the model cannot learn positional shortcuts.
* Trainable parameters: LoRA adapters, the five delimiter embedding rows, and the pointer head.
  On Qwen3-0.6B with rank 16 that is about 10M of 600M parameters.
* After training, a scalar temperature is fitted on the validation set by minimising NLL
  (golden-section search). It is stored in the checkpoint and applied at inference.

RLCD, TypeSafe's reinforcement-learning recipe, is unpublished. The supervised proper-scoring objective
plus temperature scaling gets most of the calibration benefit for a fraction of the compute; an RL stage
with an RLCR-style reward (`correct − (confidence − correct)²`) is on the roadmap.

## 5. Evaluation

`any2jev eval` reports, per question type: accuracy, NLL, multiclass Brier, ECE (equal-width and
adaptive bins), AURC (area under the risk-coverage curve) and coverage at 5% risk. It also runs

* an **option-order test** (each Choice question under several permutations: argmax stability and the
  largest probability spread), the failure mode Archer Hume measured on Jev itself;
* an **isolation check** (packed answers vs. asking each question alone).

Adaptive ECE retains constant-confidence samples in a single bin. Selective metrics accept all
questions with the same confidence together: a probability threshold cannot separate tied samples.
AURC uses the right-step area at these attainable thresholds, and coverage at 5% risk chooses the
largest qualifying threshold set. Neither metric depends on the input order of tied predictions.

## 6. Serving

`any2jev serve` exposes `POST /v1/systemone` and `GET /v1/models` with TypeSafe's exact request and
response shapes, so the official `typesafe-sdk` and any Jev integration work with a `base_url` change.
