import os

import pytest
import torch

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# A real tokenizer (small download, cached) on top of a 2-layer random Qwen3: every test runs on CPU in seconds.
TOKENIZER_ID = os.environ.get("ANY2JEV_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory) -> str:
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {TOKENIZER_ID} unavailable: {e}")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    path = tmp_path_factory.mktemp("tiny") / "base"
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_model(tiny_base):
    from any2jev.model import DecisionModel

    torch.manual_seed(0)
    return DecisionModel.from_base(tiny_base, lora_r=4, head_dim=16, device="cpu").eval()


@pytest.fixture
def example_request() -> dict:
    return {
        "state": "The weather is nice today and the park is full of people.",
        "questions": {
            "nice": {"type": "noul", "instructions": "Is the weather described as nice?"},
            "season": {"type": "choice", "instructions": "Which season is it most likely?",
                       "criteria": {"summer": None, "winter": None, "unknown": "Cannot tell from the text"}},
            "crowd": {"type": "score", "instructions": "How crowded is the park?",
                      "criteria": ["empty", "some people", "packed"]},
        },
    }
