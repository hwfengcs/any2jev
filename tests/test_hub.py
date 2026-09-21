from any2jev.model import DecisionModel
from any2jev.train import TrainConfig, load_or_create


def test_model_load_accepts_hub_revision(tiny_model, tmp_path, monkeypatch, example_request):
    checkpoint = tiny_model.save(tmp_path / "checkpoint")
    downloads = []

    def download(repo, revision=None):
        downloads.append((repo, revision))
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    model = DecisionModel.load("hf://user/decision-model@v1", device="cpu")
    assert downloads == [("user/decision-model", "v1")]
    assert model.decide(example_request) == tiny_model.decide(example_request)


def test_training_resolves_hub_checkpoint(tiny_model, tmp_path, monkeypatch):
    checkpoint = tiny_model.save(tmp_path / "checkpoint")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(checkpoint))
    calls = []

    def load(path, **kwargs):
        calls.append((str(path), kwargs["trainable"]))
        return tiny_model

    monkeypatch.setattr(DecisionModel, "load", load)
    cfg = TrainConfig(base="hf://user/decision-model", data="unused", out="unused")
    assert load_or_create(cfg) is tiny_model
    assert calls == [(str(checkpoint), True)]
