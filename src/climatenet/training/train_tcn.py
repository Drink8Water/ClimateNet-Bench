"""Train a TCN and XGBoost baseline under temporal holdout validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from climatenet.utils.paths import resolve_project_path

os.environ.setdefault("MPLCONFIGDIR", str(resolve_project_path("outputs/matplotlib-cache")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from climatenet.data.sequence_dataset import build_sequence_arrays, metadata_to_dataframe
from climatenet.models.tcn import TCNRegressor
from climatenet.utils.random import set_random_seed

SEQUENCE_FEATURES = [
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
    "wind_speed",
    "dryness_proxy",
    "saturation_vapor_pressure",
]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def temporal_holdout_indices(
    metadata: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train/test indices based on the target month year."""
    train_mask = (metadata["year"] >= train_start_year) & (metadata["year"] <= train_end_year)
    test_mask = metadata["year"] == test_year
    train_indices = metadata.index[train_mask].to_numpy()
    test_indices = metadata.index[test_mask].to_numpy()

    if len(train_indices) == 0:
        raise ValueError(
            f"No TCN train samples found for {train_start_year}-{train_end_year}. "
            "Check your feature table years."
        )
    if len(test_indices) == 0:
        raise ValueError(
            f"No TCN test samples found for test_year={test_year}. "
            "For current synthetic data, try --train-start-year 2010 --train-end-year 2019 --test-year 2020. "
            "For ERA5 Phase 2 data, use the default 2019-2022 -> 2023 split."
        )
    return train_indices, test_indices


def standardize_sequences(
    sequences: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize sequence features using train samples only to avoid leakage."""
    scaler = StandardScaler()
    train_sequences = sequences[train_indices]
    test_sequences = sequences[test_indices]

    num_features = train_sequences.shape[-1]
    scaler.fit(train_sequences.reshape(-1, num_features))

    train_scaled = scaler.transform(train_sequences.reshape(-1, num_features)).reshape(train_sequences.shape)
    test_scaled = scaler.transform(test_sequences.reshape(-1, num_features)).reshape(test_sequences.shape)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32), scaler


def train_tcn_model(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    num_features: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    channels: list[int],
    dropout: float,
) -> tuple[TCNRegressor, list[float]]:
    """Train the TCN model with MSE loss."""
    set_random_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNRegressor(num_features=num_features, channels=channels, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(train_sequences, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32).view(-1, 1),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for batch_sequences, batch_targets in loader:
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_sequences)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        losses.append(float(np.mean(epoch_losses)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | train_mse={losses[-1]:.5f}")

    return model, losses


def predict_tcn(model: TCNRegressor, sequences: np.ndarray) -> np.ndarray:
    """Run TCN predictions."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(sequences, dtype=torch.float32).to(device)
        predictions = model(inputs).cpu().numpy().reshape(-1)
    return predictions


def train_xgboost_baseline(
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    test_sequences: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train XGBoost on flattened sequence windows for a fair temporal split baseline."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required for the baseline comparison.") from exc

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_sequences.reshape(len(train_sequences), -1), train_targets)
    return model.predict(test_sequences.reshape(len(test_sequences), -1))


def save_training_loss(losses: list[float], output_path: Path) -> None:
    """Save training loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("TCN Training Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_tcn_experiment(
    features_path: Path,
    output_dir: Path,
    feature_columns: list[str] | None = None,
    sequence_length: int = 6,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    test_year: int = 2023,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    channels: list[int] | None = None,
    dropout: float = 0.2,
) -> dict[str, Any]:
    """Run TCN plus XGBoost temporal holdout experiment."""
    set_random_seed(seed)
    channels = channels or [32, 32, 32]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = feature_columns or SEQUENCE_FEATURES

    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature table: {features_path}")

    data = pd.read_csv(features_path)
    sequences, targets, metadata = build_sequence_arrays(
        data=data,
        feature_columns=feature_columns,
        target_column="evaporation_anomaly",
        sequence_length=sequence_length,
    )
    metadata_df = metadata_to_dataframe(metadata)
    train_indices, test_indices = temporal_holdout_indices(
        metadata_df,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_year=test_year,
    )

    train_sequences, test_sequences, _ = standardize_sequences(sequences, train_indices, test_indices)
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]
    test_metadata = metadata_df.iloc[test_indices].reset_index(drop=True)

    print(f"Sequence shape: {sequences.shape}")
    print(f"Train samples: {len(train_indices):,}")
    print(f"Test samples: {len(test_indices):,}")

    tcn_model, losses = train_tcn_model(
        train_sequences=train_sequences,
        train_targets=train_targets,
        num_features=len(feature_columns),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        channels=channels,
        dropout=dropout,
    )
    tcn_predictions = predict_tcn(tcn_model, test_sequences)
    xgboost_predictions = train_xgboost_baseline(train_sequences, train_targets, test_sequences, seed)

    metrics = {
        "temporal_holdout": {
            "train_years": f"{train_start_year}-{train_end_year}",
            "test_year": test_year,
            "sequence_length": sequence_length,
            "num_features": len(feature_columns),
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
        },
        "TCN": regression_metrics(test_targets, tcn_predictions),
        "XGBoost_sequence_baseline": regression_metrics(test_targets, xgboost_predictions),
    }

    predictions = test_metadata.copy()
    predictions["actual"] = test_targets
    predictions["tcn_prediction"] = tcn_predictions
    predictions["xgboost_prediction"] = xgboost_predictions
    predictions.to_csv(output_dir / "tcn_predictions.csv", index=False)

    with (output_dir / "tcn_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_training_loss(losses, output_dir / "training_loss.png")

    print(f"Saved TCN metrics to {output_dir / 'tcn_metrics.json'}")
    print(f"Saved TCN predictions to {output_dir / 'tcn_predictions.csv'}")
    print(f"Saved training loss plot to {output_dir / 'training_loss.png'}")
    return metrics


def run_tcn_arrays_for_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    channels: list[int],
    dropout: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train TCN for a temporal holdout split and return metrics/predictions."""
    sequences, targets, metadata = build_sequence_arrays(
        data=data,
        feature_columns=feature_columns,
        target_column="evaporation_anomaly",
        sequence_length=sequence_length,
    )
    metadata_df = metadata_to_dataframe(metadata)
    train_indices, test_indices = temporal_holdout_indices(
        metadata_df,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_year=test_year,
    )
    train_sequences, test_sequences, _ = standardize_sequences(sequences, train_indices, test_indices)
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    model, _ = train_tcn_model(
        train_sequences=train_sequences,
        train_targets=train_targets,
        num_features=len(feature_columns),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        channels=channels,
        dropout=dropout,
    )
    predictions = predict_tcn(model, test_sequences)
    metrics = regression_metrics(test_targets, predictions)

    prediction_frame = metadata_df.iloc[test_indices].reset_index(drop=True)
    prediction_frame["actual"] = test_targets
    prediction_frame["prediction"] = predictions
    return metrics, prediction_frame
