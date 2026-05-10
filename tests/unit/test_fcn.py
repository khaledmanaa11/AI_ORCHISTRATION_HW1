"""Tests for models/fcn.py — FCNModel architecture and forward pass."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_signal.models.fcn import FCNModel
from neural_signal.shared.config import FCNConfig


@pytest.fixture()
def fcn_cfg() -> FCNConfig:
    return FCNConfig(hidden_sizes=[128, 64], dropout_rate=0.1, output_size=1)


@pytest.fixture()
def model(fcn_cfg) -> FCNModel:
    return FCNModel(fcn_cfg)


def test_fcn_instantiates(model):
    assert model is not None


def test_fcn_output_shape_is_batch_by_1(model):
    x = torch.randn(8, 15)
    assert model.eval()(x).shape == (8, 1)


def test_fcn_output_shape_single_sample(model):
    assert model.eval()(torch.randn(1, 15)).shape == (1, 1)


def test_fcn_output_shape_large_batch(model):
    assert model.eval()(torch.randn(64, 15)).shape == (64, 1)


def test_fcn_forward_no_nan_on_random_input(model):
    out = model.eval()(torch.randn(16, 15))
    assert not torch.isnan(out).any()


def test_fcn_forward_no_inf_on_random_input(model):
    out = model.eval()(torch.randn(16, 15))
    assert not torch.isinf(out).any()


def test_fcn_has_three_linear_layers(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 3


def test_fcn_first_linear_in_features_equals_15(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[0].in_features == 15


def test_fcn_first_linear_out_features_equals_128(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[0].out_features == 128


def test_fcn_second_linear_in_features_equals_128(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[1].in_features == 128


def test_fcn_second_linear_out_features_equals_64(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[1].out_features == 64


def test_fcn_third_linear_out_features_equals_1(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[2].out_features == 1


def test_fcn_has_dropout_layers(model):
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) >= 1


def test_fcn_dropout_rate_matches_config(model, fcn_cfg):
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert all(abs(d.p - fcn_cfg.dropout_rate) < 1e-6 for d in dropouts)


def test_fcn_parameters_require_grad(model):
    assert all(p.requires_grad for p in model.parameters())


def test_fcn_weight_shapes_correct_layer1(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[0].weight.shape == (128, 15)


def test_fcn_weight_shapes_correct_layer2(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[1].weight.shape == (64, 128)


def test_fcn_weight_shapes_correct_layer3(model):
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears[2].weight.shape == (1, 64)


def test_fcn_output_deterministic_in_eval_mode(model):
    x = torch.randn(4, 15)
    model.eval()
    o1 = model(x)
    o2 = model(x)
    assert torch.allclose(o1, o2)


def test_fcn_built_from_config_hidden_sizes():
    cfg = FCNConfig(hidden_sizes=[32, 16], dropout_rate=0.0, output_size=1)
    m = FCNModel(cfg)
    linears = [lyr for lyr in m.modules() if isinstance(lyr, nn.Linear)]
    assert linears[0].out_features == 32
    assert linears[1].out_features == 16


def test_fcn_forward_output_is_float_tensor(model):
    assert model.eval()(torch.randn(4, 15)).dtype == torch.float32


def test_fcn_forward_with_zero_input_no_error(model):
    model.eval()(torch.zeros(4, 15))


def test_fcn_forward_with_negative_input_no_error(model):
    model.eval()(torch.full((4, 15), -1.0))
