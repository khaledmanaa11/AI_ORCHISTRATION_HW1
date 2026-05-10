"""Tests for services/data_loader.py — DataLoaderService."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_signal.services.data_loader import DataBundle, DataLoaderService


def test_data_loader_instantiates(tiny_npz, batch_size_fixture):
    svc = DataLoaderService(tiny_npz, batch_size_fixture)
    assert svc is not None


def test_load_returns_data_bundle(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert isinstance(bundle, DataBundle)


def test_data_bundle_has_train_loader(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.train_loader is not None


def test_data_bundle_has_val_loader(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.val_loader is not None


def test_data_bundle_has_test_loader(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.test_loader is not None


def test_train_loader_batch_shape_fcn(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, yb = next(iter(bundle.train_loader))
    assert xb.shape[1] == 15


def test_train_loader_y_batch_shape(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    _, yb = next(iter(bundle.train_loader))
    assert yb.shape[1] == 1


def test_data_bundle_has_raw_x_train(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.X_train is not None and bundle.X_train.shape[1] == 10


def test_data_bundle_has_raw_c_train(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.C_train is not None and bundle.C_train.shape[1] == 5


def test_data_bundle_has_raw_y_train(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    assert bundle.y_train is not None


def test_fcn_input_is_concat_of_x_and_c(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.train_loader))
    assert xb.shape[1] == 15  # 10 + 5


def test_rnn_input_reshape_shape(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.seq_train_loader))
    assert xb.shape[1] == 10
    assert xb.shape[2] == 1


def test_data_loader_raises_on_missing_npz(tmp_path, batch_size_fixture):
    with pytest.raises(FileNotFoundError):
        DataLoaderService(tmp_path / "missing.npz", batch_size_fixture).load()


def test_data_loader_raises_on_missing_x_train_key(tmp_path, batch_size_fixture):
    path = tmp_path / "bad.npz"
    np.savez(path, X_val=np.zeros((10, 10)))
    with pytest.raises(KeyError):
        DataLoaderService(path, batch_size_fixture).load()


def test_val_loader_does_not_shuffle(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    b1 = next(iter(bundle.val_loader))[0]
    b2 = next(iter(bundle.val_loader))[0]
    assert torch.allclose(b1, b2)


def test_test_loader_does_not_shuffle(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    b1 = next(iter(bundle.test_loader))[0]
    b2 = next(iter(bundle.test_loader))[0]
    assert torch.allclose(b1, b2)


def test_no_nan_in_train_batch_x(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.train_loader))
    assert not torch.isnan(xb).any()


def test_no_nan_in_train_batch_y(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    _, yb = next(iter(bundle.train_loader))
    assert not torch.isnan(yb).any()


def test_no_inf_in_train_batch_x(tiny_npz, batch_size_fixture):
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    xb, _ = next(iter(bundle.train_loader))
    assert not torch.isinf(xb).any()
