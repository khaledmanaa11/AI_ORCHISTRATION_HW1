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
