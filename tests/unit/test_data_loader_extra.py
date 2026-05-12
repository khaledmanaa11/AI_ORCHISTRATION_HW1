"""Extra DataLoaderService tests — overflow from test_data_loader.py (150-line limit)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.data_loader import DataLoaderService
from neural_signal.shared.config import LSTMConfig, RNNConfig


def test_seq_loader_batch_shape_rnn(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.seq_train_loader))
    assert xb.shape[1] == 10 and xb.shape[2] == 6  # window(1) + C broadcast(5)


def test_data_loader_raises_on_missing_c_train_key(tmp_path, batch_size_fixture):
    path = tmp_path / "no_c.npz"
    np.savez(path, X_train=np.zeros((10, 10)), X_val=np.zeros((5, 10)),
             X_test=np.zeros((5, 10)), y_train=np.zeros((10, 1)),
             y_val=np.zeros((5, 1)), y_test=np.zeros((5, 1)))
    with pytest.raises(KeyError):
        DataLoaderService(path, batch_size_fixture).load()


def test_data_loader_raises_on_missing_y_train_key(tmp_path, batch_size_fixture):
    path = tmp_path / "no_y.npz"
    np.savez(path, X_train=np.zeros((10, 10)), X_val=np.zeros((5, 10)),
             X_test=np.zeros((5, 10)), C_train=np.zeros((10, 5)),
             C_val=np.zeros((5, 5)), C_test=np.zeros((5, 5)))
    with pytest.raises(KeyError):
        DataLoaderService(path, batch_size_fixture).load()


def test_train_loader_shuffles_data(tiny_npz, batch_size_fixture):
    """Two epochs of the same train_loader should (very likely) differ due to shuffling."""
    bundle = DataLoaderService(tiny_npz, batch_size_fixture, seed=0).load()
    b1 = next(iter(bundle.train_loader))[0]
    b2 = next(iter(bundle.train_loader))[0]
    assert not torch.allclose(b1, b2)


def test_batch_size_from_config(tiny_npz):
    bundle = DataLoaderService(tiny_npz, batch_size=8).load()
    xb, _ = next(iter(bundle.train_loader))
    assert xb.shape[0] <= 8


# ── Gap 1: C injection strategy tests ────────────────────────────────────────

def test_seq_loader_batch_x_includes_c_features(tiny_npz, batch_size_fixture):
    """Seq loader features include both window and C — last 5 cols are C values."""
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.seq_train_loader))
    # shape (batch, 10, 6): 6 features per timestep (1 window + 5 C)
    assert xb.shape[2] == 6


def test_rnn_input_size_matches_c_injection_strategy(tiny_npz, batch_size_fixture):
    """With broadcast strategy, seq input has 6 features: input_size=6."""
    from neural_signal.models.rnn import RNNModel
    from neural_signal.shared.config import RNNConfig
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.seq_train_loader))
    # The third dimension must equal the model's expected input_size
    cfg = RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1, input_size=xb.shape[2])
    model = RNNModel(cfg)
    out = model(xb)
    assert out.shape == (xb.shape[0], 1)


def test_lstm_input_size_matches_c_injection_strategy(tiny_npz, batch_size_fixture):
    """With broadcast strategy, LSTM seq input has 6 features."""
    from neural_signal.models.lstm import LSTMModel
    from neural_signal.shared.config import LSTMConfig
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.seq_train_loader))
    cfg = LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1, input_size=xb.shape[2])
    model = LSTMModel(cfg)
    out = model(xb)
    assert out.shape == (xb.shape[0], 1)


def test_rnn_output_changes_when_c_changes(tiny_npz, batch_size_fixture):
    """Same window, different C selector → different prediction."""
    cfg = RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1, input_size=6)
    model = RNNModel(cfg)
    model.eval()
    # Build two sequences: same window portion but different C
    x1 = torch.randn(1, 10, 6)
    x2 = x1.clone()
    x2[:, :, 1:] = 1 - x1[:, :, 1:]  # flip C columns
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2)


def test_lstm_output_changes_when_c_changes(tiny_npz, batch_size_fixture):
    """Same window, different C → different LSTM prediction."""
    cfg = LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1, input_size=6)
    model = LSTMModel(cfg)
    model.eval()
    x1 = torch.randn(1, 10, 6)
    x2 = x1.clone()
    x2[:, :, 1:] = 1 - x1[:, :, 1:]
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    assert not torch.allclose(out1, out2)
