import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.prosodynet.data import causal_features, load_sample, load_splits, normalize, split_leakage, training_statistics
from src.prosodynet.losses import prosody_loss
from src.prosodynet.model import ProsodyNetS


def test_model_is_causal_for_prefix():
    torch.manual_seed(1)
    model = ProsodyNetS(8, channels=32, hidden=40).eval()
    x = torch.randn(1, 24, 8)
    altered = x.clone()
    altered[:, 14:] += 10.0
    first, _ = model(x)
    second, _ = model(altered)
    assert torch.allclose(first[:, :14], second[:, :14], atol=1e-6)


def test_streaming_step_matches_full_forward_first_frame():
    torch.manual_seed(2)
    model = ProsodyNetS(8, channels=32, hidden=40).eval()
    x = torch.randn(1, 8)
    full, _ = model(x.unsqueeze(0))
    state = model.initial_state()
    streamed, _ = model.step(x, state)
    assert torch.allclose(full[:, 0], streamed, atol=1e-6)


def test_state_reset_reproduces_initial_output():
    torch.manual_seed(3)
    model = ProsodyNetS(8, channels=32, hidden=40).eval()
    x = torch.randn(8)
    state = model.initial_state()
    first, state = model.step(x, state)
    model.step(x, state)
    model.reset_state(state)
    reset, _ = model.step(x, state)
    assert torch.allclose(first, reset, atol=1e-6)


def test_output_bounds():
    model = ProsodyNetS(8, channels=32, hidden=40).eval()
    output, _ = model(torch.randn(2, 12, 8) * 100)
    assert torch.isfinite(output).all()
    assert float(output[..., 0].min()) >= -1.0
    assert float(output[..., 0].max()) <= 1.4
    assert float(output[..., 1].min()) >= -1.5
    assert float(output[..., 1].max()) <= 1.5


def test_loss_respects_padding_and_voiced_mask():
    pred = torch.zeros(1, 4, 2, requires_grad=True)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor([[1., 1., 0., 0.]])
    voiced = torch.tensor([[1., 0., 1., 1.]])
    loss, values = prosody_loss(pred, target, mask, voiced)
    loss.backward()
    assert torch.isfinite(loss)
    assert values["total"] > 0
    assert pred.grad[0, 2:].abs().sum() == 0


def test_dataset_split_has_no_leakage():
    root = __import__("pathlib").Path(os.environ.get("PROSODYNET_DATASET", "voice_sample/prosody_dataset_v2"))
    splits = load_splits(root)
    assert split_leakage(splits) == []
    assert len(splits["test"]) == 49


def test_normalization_is_train_only_and_finite():
    root = __import__("pathlib").Path(os.environ.get("PROSODYNET_DATASET", "voice_sample/prosody_dataset_v2"))
    splits = load_splits(root)
    info = training_statistics(root, splits["train"])
    sample = load_sample(root, splits["test"][0])
    features = normalize(causal_features(sample), info)
    assert features.shape[1] == 8
    assert np.isfinite(features).all()
    assert info["source_split"] == "train only"


def test_serialization_round_trip(tmp_path):
    model = ProsodyNetS(8, channels=32, hidden=40).eval()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = ProsodyNetS(8, channels=32, hidden=40).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(1, 3, 8)
    assert torch.allclose(model(x)[0], restored(x)[0])
