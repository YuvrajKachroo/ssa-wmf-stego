"""Phase 12: CNN-based steganalysis.

Contains two models:

1. SimpleBaselineCNN
   A plain CNN used as a sanity-check baseline.

2. XuNetStyle
   An approximation of the Xu-Net steganalysis architecture.

Important:
This is NOT claimed to be an exact reproduction of every layer
width/hyperparameter from the original Xu-Net paper.

The XuNetStyle model uses:
- fixed 5x5 KV high-pass filter
- non-trainable first layer
- abs() activation after the high-pass filter
- batch normalization
- average pooling
"""

from __future__ import annotations

import torch
import torch.nn as nn


_KV_KERNEL = (
    torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    )
    / 12.0
)


class SimpleBaselineCNN(nn.Module):
    """Plain CNN baseline.

    This model has no steganalysis-specific preprocessing.
    """

    def __init__(
        self,
        in_size: int = 64,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 4 * 4,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                2,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.features(x))


class XuNetStyle(nn.Module):
    """Approximate Xu-Net-style steganalysis CNN."""

    def __init__(self) -> None:
        super().__init__()

        # -----------------------------------------------------
        # Fixed high-pass preprocessing
        # -----------------------------------------------------
        self.hpf = nn.Conv2d(
            1,
            1,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        with torch.no_grad():
            self.hpf.weight.copy_(
                _KV_KERNEL.view(
                    1,
                    1,
                    5,
                    5,
                )
            )

        # Critical Xu-Net-style property:
        # the filter is fixed, not trainable.
        self.hpf.weight.requires_grad_(False)

        def make_block(
            in_channels: int,
            out_channels: int,
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    padding=2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.AvgPool2d(
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
            )

        self.group1 = make_block(
            1,
            8,
        )

        self.group2 = make_block(
            8,
            16,
        )

        self.group3 = make_block(
            16,
            32,
        )

        self.group4 = make_block(
            32,
            64,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            64,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.hpf(x)

        # Symmetric steganographic residuals:
        # remove sign information.
        x = torch.abs(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)

        x = self.global_pool(x)

        x = x.flatten(1)

        return self.classifier(x)
