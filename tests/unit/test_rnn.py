"""Tests for models/rnn.py — RNNModel architecture and forward pass."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_signal.models.rnn import RNNModel
from neural_signal.shared.config import RNNConfig


@pytest.fixture()
def rnn_cfg() -> RNNConfig:
    return RNNConfig(hidden_size=64, nonlinearity="tanh", num_layers=1, output_size=1)


@pytest.fixture()
def model(rnn_cfg) -> RNNModel:
    return RNNModel(rnn_cfg)


def test_rnn_instantiates(model):
    assert model is not None


def test_rnn_output_shape_is_batch_by_1(model):
    x = torch.randn(8, 10, 1)
    assert model.eval()(x).shape == (8, 1)


def test_rnn_output_shape_single_sample(model):
    assert model.eval()(torch.randn(1, 10, 1)).shape == (1, 1)


def test_rnn_output_shape_large_batch(model):
    assert model.eval()(torch.randn(64, 10, 1)).shape == (64, 1)


def test_rnn_forward_no_nan(model):
    assert not torch.isnan(model.eval()(torch.randn(8, 10, 1))).any()


def test_rnn_forward_no_inf(model):
    assert not torch.isinf(model.eval()(torch.randn(8, 10, 1))).any()


def test_rnn_has_rnn_layer(model):
    assert hasattr(model, "rnn")
    assert isinstance(model.rnn, nn.RNN)


def test_rnn_has_linear_head(model):
    assert hasattr(model, "head")
    assert isinstance(model.head, nn.Linear)


def test_rnn_nonlinearity_is_tanh(model):
    assert model.rnn.nonlinearity == "tanh"


def test_rnn_hidden_size_64(model):
    assert model.rnn.hidden_size == 64


def test_rnn_batch_first_true(model):
    assert model.rnn.batch_first is True


def test_rnn_parameters_require_grad(model):
    assert all(p.requires_grad for p in model.parameters())


def test_rnn_output_deterministic_in_eval(model):
    x = torch.randn(4, 10, 1)
    model.eval()
    assert torch.allclose(model(x), model(x))


def test_rnn_output_is_float(model):
    assert model.eval()(torch.randn(4, 10, 1)).dtype == torch.float32


def test_rnn_head_in_features_equals_hidden_size(model, rnn_cfg):
    assert model.head.in_features == rnn_cfg.hidden_size


def test_rnn_head_out_features_equals_output_size(model, rnn_cfg):
    assert model.head.out_features == rnn_cfg.output_size
