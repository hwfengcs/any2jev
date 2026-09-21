import json

import pytest
import torch

from any2jev.model import DecisionModel
from any2jev.schema import SystemOneRequest, render, to_specs


def _max_diff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def test_backbone_has_no_vocab_head(tiny_model):
    assert not hasattr(tiny_model.backbone, "lm_head")
    names = [n for n, _ in tiny_model.backbone.named_parameters()]
    assert not any("lm_head" in n for n in names)
    assert tiny_model.mode == "packed" and not tiny_model.hybrid


def test_packed_equals_rows_equals_alone(tiny_model, example_request):
    req = SystemOneRequest.model_validate(example_request)
    specs, state = to_specs(req), render(req.state)
    packed = tiny_model.encode(state, specs)
    p_packed = tiny_model.probs([packed])[0]
    tiny_model.mode = "rows"
    try:
        p_rows = tiny_model.probs([packed])[0]
    finally:
        tiny_model.mode = "packed"
    p_alone = [tiny_model.probs([tiny_model.encode(state, [s])])[0][0] for s in specs]
    assert _max_diff(p_packed, p_rows) < 1e-4
    assert _max_diff(p_packed, p_alone) < 1e-4
    for p in p_packed:
        assert abs(float(p.sum()) - 1) < 1e-5


def test_batch_padding_invariance(tiny_model, example_request):
    req = SystemOneRequest.model_validate(example_request)
    long = tiny_model.encode(render(req.state), to_specs(req))
    short_req = SystemOneRequest.model_validate({"state": "Short.", "questions": {"z": {"type": "noul", "instructions": "Is it short?"}}})
    short = tiny_model.encode(render(short_req.state), to_specs(short_req))
    batched = tiny_model.probs([long, short])
    assert _max_diff(batched[0], tiny_model.probs([long])[0]) < 1e-4
    assert _max_diff(batched[1], tiny_model.probs([short])[0]) < 1e-4


@pytest.mark.parametrize("mode", ["packed", "rows"])
def test_forward_does_not_allocate_unused_kv_cache(tiny_model, example_request, monkeypatch, mode):
    caches = []
    hook = tiny_model.backbone.register_forward_hook(lambda _m, _args, output: caches.append(output.past_key_values))
    monkeypatch.setattr(tiny_model, "mode", mode)
    try:
        tiny_model.decide(example_request)
    finally:
        hook.remove()
    assert caches == [None]


def test_decide_returns_wire_answers(tiny_model, example_request):
    answers, n_in = tiny_model.decide(example_request)
    assert set(answers) == {"nice", "season", "crowd"} and n_in > 0
    assert answers["nice"]["type"] == "noul" and 0 <= answers["nice"]["noul"] <= 1
    assert answers["season"]["choice"] in {"summer", "winter", "unknown"}
    assert abs(sum(answers["season"]["probabilities"].values()) - 1) < 1e-3
    assert answers["crowd"]["legend"] == {"0": "empty", "1": "some people", "2": "packed"}
    json.dumps(answers)  # serialisable


def test_gradients_reach_every_trainable_tensor(tiny_model, example_request):
    req = SystemOneRequest.model_validate(example_request)
    packed = tiny_model.encode(render(req.state), to_specs(req))
    tiny_model.zero_grad(set_to_none=True)
    loss = sum(z.logsumexp(-1) for z in tiny_model.logits([packed])[0])
    loss.backward()
    trainable = [(n, p) for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for _, p in trainable)
    assert any("head." in n for n, _ in trainable) and any("lora_" in n for n, _ in trainable)
    tiny_model.zero_grad(set_to_none=True)


def test_save_load_round_trip(tiny_model, example_request, tmp_path):
    req = SystemOneRequest.model_validate(example_request)
    specs, state = to_specs(req), render(req.state)
    before = tiny_model.probs([tiny_model.encode(state, specs)], temperature=1.0)[0]
    tiny_model.temperature = 1.7
    out = tiny_model.save(tmp_path / "ckpt")
    tiny_model.temperature = 1.0
    assert DecisionModel.is_checkpoint(out)
    cfg = json.loads((out / "any2jev.json").read_text())
    assert cfg["temperature"] == 1.7 and cfg["delimiters"] == tiny_model.delims.tokens
    m2 = DecisionModel.load(out, device="cpu")
    after = m2.probs([m2.encode(state, specs)], temperature=1.0)[0]
    assert _max_diff(before, after) < 1e-5
    assert m2.temperature == 1.7
    # the stored temperature is applied by default
    default = m2.probs([m2.encode(state, specs)])[0]
    explicit = m2.probs([m2.encode(state, specs)], temperature=1.7)[0]
    assert _max_diff(default, explicit) < 1e-6
    merged = DecisionModel.load(out, device="cpu", merge=True)
    assert _max_diff(default, merged.probs([merged.encode(state, specs)])[0]) < 1e-5


def test_state_truncation_and_branch_limit(tiny_model):
    req = SystemOneRequest.model_validate({"state": "word " * 500, "questions": {"q": {"type": "noul", "instructions": "x?"}}})
    p = tiny_model.encode(render(req.state), to_specs(req), max_state=32)
    assert p.n_state == 32 and p.truncated
    import pytest

    with pytest.raises(ValueError):
        tiny_model.encode("s", to_specs(req), max_branch=4)


@pytest.mark.parametrize("architecture", ["llama", "gpt2", "gemma2", "mistral"])
def test_additional_backbone_modes_preserve_native_attention(tiny_model, example_request, architecture):
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )

    from any2jev.model import extract_backbone

    common = dict(vocab_size=len(tiny_model.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    constructors = {
        "llama": lambda: LlamaForCausalLM(LlamaConfig(**common)),
        "gpt2": lambda: GPT2LMHeadModel(GPT2Config(vocab_size=len(tiny_model.tok), n_embd=32,
                                                n_layer=2, n_head=2, n_positions=4096)),
        "gemma2": lambda: Gemma2ForCausalLM(Gemma2Config(**common, head_dim=16, sliding_window=16)),
        "mistral": lambda: MistralForCausalLM(MistralConfig(**common, sliding_window=16)),
    }
    torch.manual_seed(7)
    backbone = extract_backbone(constructors[architecture]())
    model = DecisionModel(backbone, tiny_model.tok, tiny_model.delims, head_dim=16).eval()
    expected_mode = "packed" if architecture == "llama" else "rows"
    assert model.mode == expected_mode
    req = SystemOneRequest.model_validate(example_request)
    state, specs = render(req.state), to_specs(req)
    packed = model.encode(state, specs)
    automatic = model.probs([packed])[0]
    model.mode = "rows"
    rows = model.probs([packed])[0]
    alone = [model.probs([model.encode(state, [s])])[0][0] for s in specs]
    assert _max_diff(automatic, rows) < 1e-5
    assert _max_diff(rows, alone) < 1e-5
    if architecture in ("gemma2", "mistral"):
        with pytest.raises(ValueError, match="sliding-window"):
            DecisionModel(backbone, tiny_model.tok, tiny_model.delims, mode="packed")
