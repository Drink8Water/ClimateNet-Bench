"""Temporal Convolutional Network for climate anomaly regression."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right-side padding to keep convolutions causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trim future-looking padded values."""
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One causal dilated convolution block with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply block and residual connection."""
        output = self.network(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(output + residual)


class TemporalConvNet(nn.Module):
    """Stack of causal dilated temporal convolution blocks."""

    def __init__(
        self,
        num_inputs: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        for layer_index, out_channels in enumerate(channels):
            dilation = 2**layer_index
            in_channels = num_inputs if layer_index == 0 else channels[layer_index - 1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal convolution stack."""
        return self.network(x)


class TCNRegressor(nn.Module):
    """TCN regression model for next-month evaporation anomaly."""

    def __init__(
        self,
        num_features: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = channels or [32, 32, 32]
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.regression_head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next-month anomaly.

        Input shape: [batch, sequence_length, num_features]
        Conv1d shape: [batch, num_features, sequence_length]
        """
        x = x.transpose(1, 2)
        features = self.tcn(x)
        last_step = features[:, :, -1]
        return self.regression_head(last_step)
