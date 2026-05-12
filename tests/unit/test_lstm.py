"""Tests for models/lstm.py — LSTMModel architecture and forward pass."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_signal.models.lstm import LSTMModel
from neural_signal.shared.config import LSTMConfig


@pytest.fixture()
def lstm_cfg() -> LSTMConfig:
    return LSTMConfig(hidden_size=64, num_layers=1, dense_hidden_size=32, output_size=1)


@pytest.fixture()
def model(lstm_cfg) -> LSTMModel:
    return LSTMModel(lstm_cfg)


def test_lstm_instantiates(model):
    assert model is not None


def test_lstm_output_shape_is_batch_by_1(model):
    assert model.eval()(torch.randn(8, 10, 1)).shape == (8, 1)


def test_lstm_output_shape_single_sample(model):
    assert model.eval()(torch.randn(1, 10, 1)).shape == (1, 1)


def test_lstm_output_shape_large_batch(model):
    assert model.eval()(torch.randn(64, 10, 1)).shape == (64, 1)


def test_lstm_forward_no_nan(model):
    assert not torch.isnan(model.eval()(torch.randn(8, 10, 1))).any()


def test_lstm_forward_no_inf(model):
    assert not torch.isinf(model.eval()(torch.randn(8, 10, 1))).any()


def test_lstm_has_lstm_layer(model):
    assert hasattr(model, "lstm")
    assert isinstance(model.lstm, nn.LSTM)


def test_lstm_has_head(model):
    assert hasattr(model, "head")
    assert isinstance(model.head, nn.Sequential)


def test_lstm_hidden_size_64(model):
    assert model.lstm.hidden_size == 64


def test_lstm_batch_first_true(model):
    assert model.lstm.batch_first is True


def test_lstm_head_has_two_linears(model):
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2


def test_lstm_head_first_linear_in_64(model):
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    assert linears[0].in_features == 64


def test_lstm_head_first_linear_out_32(model):
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    assert linears[0].out_features == 32


def test_lstm_head_second_linear_out_1(model):
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    assert linears[1].out_features == 1


def test_lstm_parameters_require_grad(model):
    assert all(p.requires_grad for p in model.parameters())


def test_lstm_output_deterministic_in_eval(model):
    x = torch.randn(4, 10, 1)
    model.eval()
    assert torch.allclose(model(x), model(x))


def test_lstm_output_is_float(model):
    assert model.eval()(torch.randn(4, 10, 1)).dtype == torch.float32


def test_lstm_second_dense_in_features_equals_32(model):
    linears = [m for m in model.head.modules() if isinstance(m, nn.Linear)]
    assert linears[1].in_features == 32


def test_lstm_relu_between_dense_layers(model):
    relus = [m for m in model.head.modules() if isinstance(m, nn.ReLU)]
    assert len(relus) >= 1


def test_lstm_no_activation_on_output(model):
    """Last module in head must be Linear, not an activation."""
    head_children = list(model.head.children())
    assert isinstance(head_children[-1], nn.Linear)


def test_lstm_many_to_one_last_output_used(model):
    """Verify that changing only the last time step changes the output."""
    x1 = torch.randn(1, 10, 1)
    x2 = x1.clone()
    x2[:, -1, :] += 10.0
    model.eval()
    assert not torch.allclose(model(x1), model(x2))


def test_lstm_input_size_equals_1(model):
    assert model.lstm.input_size == 1


def test_lstm_hidden_size_not_hardcoded():
    cfg_small = LSTMConfig(hidden_size=8, num_layers=1, dense_hidden_size=4, output_size=1)
    m = LSTMModel(cfg_small)
    assert m.lstm.hidden_size == 8


def test_lstm_dense_hidden_size_not_hardcoded():
    cfg_small = LSTMConfig(hidden_size=8, num_layers=1, dense_hidden_size=4, output_size=1)
    m = LSTMModel(cfg_small)
    linears = [mod for mod in m.head.modules() if isinstance(mod, nn.Linear)]
    assert linears[0].out_features == 4
