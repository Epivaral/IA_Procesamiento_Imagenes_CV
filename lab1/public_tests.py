"""
Public tests for the MLP lab.

Usage inside the notebook:
    import public_tests as public_tests
    public_tests.test_tarea1(globals())
    public_tests.test_tarea2(globals())
    public_tests.test_tarea3(globals())

These tests are intentionally lightweight. Passing public tests does not guarantee
full credit in the private grader.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_DIR = Path("data")


def _require(namespace, name):
    assert name in namespace, f"Missing required object: {name}"
    return namespace[name]


def _preprocess_features(train_df, dev_df, feature_cols=("x1", "x2")):
    X_train = train_df[list(feature_cols)].to_numpy(dtype=np.float32)
    X_dev = dev_df[list(feature_cols)].to_numpy(dtype=np.float32)
    all_X = np.vstack([X_train, X_dev])
    normalizer = MinMaxScaler()
    normalizer.fit(all_X)
    X_train_norm = normalizer.transform(X_train)
    X_dev_norm = normalizer.transform(X_dev)
    standardizer = StandardScaler()
    standardizer.fit(X_train_norm)
    X_train_std = standardizer.transform(X_train_norm).astype(np.float32)
    X_dev_std = standardizer.transform(X_dev_norm).astype(np.float32)
    return X_train_std, X_dev_std


def _loader_regression(batch_size=64):
    train_df = pd.read_csv(DATA_DIR / "dataset_B_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dataset_B_dev.csv")
    X_train, X_dev = _preprocess_features(train_df, dev_df)
    y_train = train_df["y"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_dev = dev_df["y"].to_numpy(dtype=np.float32).reshape(-1, 1)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_ds = TensorDataset(torch.tensor(X_dev), torch.tensor(y_dev))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(dev_ds, batch_size=256)


def _loader_multiclass(batch_size=64):
    train_df = pd.read_csv(DATA_DIR / "dataset_C_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dataset_C_dev.csv")
    X_train, X_dev = _preprocess_features(train_df, dev_df)
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    y_dev = dev_df["y"].to_numpy(dtype=np.int64)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_ds = TensorDataset(torch.tensor(X_dev), torch.tensor(y_dev))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(dev_ds, batch_size=256)


def _loader_multilabel(batch_size=64):
    train_df = pd.read_csv(DATA_DIR / "dataset_D_train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dataset_D_dev.csv")
    X_train, X_dev = _preprocess_features(train_df, dev_df)
    y_cols = ["y0", "y1", "y2", "y3"]
    y_train = train_df[y_cols].to_numpy(dtype=np.float32)
    y_dev = dev_df[y_cols].to_numpy(dtype=np.float32)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_ds = TensorDataset(torch.tensor(X_dev), torch.tensor(y_dev))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(dev_ds, batch_size=256)


def _count_linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _has_relu(model):
    return any(isinstance(m, nn.ReLU) for m in model.modules())


def _assert_three_layer_mlp(model, output_dim):
    linears = _count_linears(model)
    assert len(linears) == 3, "The model must contain exactly 3 Linear layers."
    assert linears[0].in_features == 2, "The first Linear layer must receive 2 input features."
    assert linears[0].out_features == 8, "The first hidden layer must have 8 units."
    assert linears[1].in_features == 8 and linears[1].out_features == 8, "The second hidden layer must be 8 -> 8."
    assert linears[2].in_features == 8 and linears[2].out_features == output_dim, f"The output layer must have {output_dim} units."
    assert _has_relu(model), "The model should include ReLU activations."


def test_tarea1(namespace):
    torch.manual_seed(123)
    RegressionMLP = _require(namespace, "RegressionMLP")
    train_fn = _require(namespace, "train_regression_model")
    eval_fn = _require(namespace, "evaluate_regression_model")
    model = RegressionMLP()
    assert isinstance(model, nn.Module), "RegressionMLP must be a torch.nn.Module."
    _assert_three_layer_mlp(model, output_dim=1)
    x = torch.randn(5, 2)
    out = model(x)
    assert tuple(out.shape) == (5, 1), "RegressionMLP forward output must have shape (batch, 1)."
    train_loader, dev_loader = _loader_regression()
    _ = train_fn(model, train_loader, epochs=5, lr=1e-2)
    metrics = eval_fn(model, dev_loader)
    assert isinstance(metrics, dict), "evaluate_regression_model must return a dictionary."
    for key in ["mse", "rmse", "mae", "r2"]:
        assert key in metrics, f"Missing regression metric: {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} must be finite."
    print("Tarea 1 public tests passed.")


def test_tarea2(namespace):
    torch.manual_seed(123)
    MulticlassMLP = _require(namespace, "MulticlassMLP")
    train_fn = _require(namespace, "train_multiclass_model")
    eval_fn = _require(namespace, "evaluate_multiclass_model")
    model = MulticlassMLP()
    assert isinstance(model, nn.Module), "MulticlassMLP must be a torch.nn.Module."
    _assert_three_layer_mlp(model, output_dim=4)
    x = torch.randn(5, 2)
    out = model(x)
    assert tuple(out.shape) == (5, 4), "MulticlassMLP forward output must have shape (batch, 4)."
    train_loader, dev_loader = _loader_multiclass()
    _ = train_fn(model, train_loader, epochs=5, lr=1e-2)
    metrics = eval_fn(model, dev_loader)
    assert isinstance(metrics, dict), "evaluate_multiclass_model must return a dictionary."
    for key in ["accuracy", "macro_f1", "confusion_matrix"]:
        assert key in metrics, f"Missing multiclass metric: {key}"
    assert np.isfinite(metrics["accuracy"]), "accuracy must be finite."
    assert np.isfinite(metrics["macro_f1"]), "macro_f1 must be finite."
    print("Tarea 2 public tests passed.")


def test_tarea3(namespace):
    torch.manual_seed(123)
    MultilabelMLP = _require(namespace, "MultilabelMLP")
    train_fn = _require(namespace, "train_multilabel_model")
    eval_fn = _require(namespace, "evaluate_multilabel_model")
    model = MultilabelMLP()
    assert isinstance(model, nn.Module), "MultilabelMLP must be a torch.nn.Module."
    _assert_three_layer_mlp(model, output_dim=4)
    x = torch.randn(5, 2)
    out = model(x)
    assert tuple(out.shape) == (5, 4), "MultilabelMLP forward output must have shape (batch, 4)."
    train_loader, dev_loader = _loader_multilabel()
    _ = train_fn(model, train_loader, epochs=5, lr=1e-2)
    metrics = eval_fn(model, dev_loader)
    assert isinstance(metrics, dict), "evaluate_multilabel_model must return a dictionary."
    for key in ["subset_accuracy", "micro_f1", "macro_f1"]:
        assert key in metrics, f"Missing multilabel metric: {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} must be finite."
    print("Tarea 3 public tests passed.")
