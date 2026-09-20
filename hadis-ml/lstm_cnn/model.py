"""
HADIS — RF Signal Classification Model (LSTM + CNN Hybrid)

High Altitude Drone Intelligence System
Author: Prakash Tiwari | Chandigarh Engineering College (IKGPTU)

This module implements a hybrid 1D-CNN + Bidirectional LSTM architecture for
classifying drone RF signals from the DroneRF dataset. The CNN extracts local
spectral features while the BiLSTM captures temporal dependencies across the
signal sequence.

Architecture:
    Input (batch, 1, 256)
    -> Conv1D blocks: 1->64->128->256 channels
    -> Reshape for LSTM
    -> Bidirectional 2-layer LSTM (hidden=128, dropout=0.3)
    -> Global Average Pooling
    -> FC: 256 -> 128 -> num_classes
"""

import torch
import torch.nn as nn


class HADISRFClassifier(nn.Module):
    """Hybrid 1D-CNN + Bidirectional LSTM for drone RF signal classification.

    Processes raw RF signal windows through convolutional feature extraction
    followed by recurrent sequence modelling. Designed for the DroneRF dataset
    with configurable output classes.

    Args:
        num_classes (int): Number of output classes (default: 6).
        seq_len (int): Length of input RF signal window (default: 256).
    """

    def __init__(self, num_classes: int = 6, seq_len: int = 256):
        super(HADISRFClassifier, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len

        # --- 1D Convolutional Feature Extractor ---
        # Block 1: 1 -> 64 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 2: 64 -> 128 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 3: 128 -> 256 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # After 3x MaxPool(2): seq_len / 8
        self._lstm_seq_len = seq_len // 8

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # --- Classification Head ---
        # BiLSTM output: 128 * 2 = 256
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch, num_classes).
        """
        # CNN feature extraction
        x = self.conv_block1(x)   # (batch, 64, seq_len/2)
        x = self.conv_block2(x)   # (batch, 128, seq_len/4)
        x = self.conv_block3(x)   # (batch, 256, seq_len/8)

        # Reshape for LSTM: (batch, seq_steps, features)
        x = x.permute(0, 2, 1)    # (batch, seq_len/8, 256)

        # Bidirectional LSTM
        x, _ = self.lstm(x)        # (batch, seq_len/8, 256)

        # Global average pooling over the temporal dimension
        x = x.mean(dim=1)          # (batch, 256)

        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the classification head.

        Used by the fusion transformer to obtain RF feature vectors.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, seq_len).

        Returns:
            torch.Tensor: Feature vector of shape (batch, 256).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)  # (batch, 256)
        return x


def get_model(num_classes: int = 6, seq_len: int = 256) -> HADISRFClassifier:
    """Factory function to create an HADISRFClassifier instance.

    Args:
        num_classes (int): Number of output classes.
        seq_len (int): Length of input RF signal window.

    Returns:
        HADISRFClassifier: Initialized model.
    """
    model = HADISRFClassifier(num_classes=num_classes, seq_len=seq_len)
    return model


if __name__ == "__main__":
    # --- Smoke Test ---
    print("[HADIS] Running HADISRFClassifier smoke test...")

    batch_size = 4
    num_classes = 6
    seq_len = 256

    model = get_model(num_classes=num_classes, seq_len=seq_len)
    dummy_input = torch.randn(batch_size, 1, seq_len)

    # Forward pass
    logits = model(dummy_input)
    print(f"[HADIS] Input shape:   {dummy_input.shape}")
    print(f"[HADIS] Output shape:  {logits.shape}")
    assert logits.shape == (batch_size, num_classes), (
        f"Expected ({batch_size}, {num_classes}), got {logits.shape}"
    )

    # Feature extraction
    features = model.extract_features(dummy_input)
    print(f"[HADIS] Feature shape: {features.shape}")
    assert features.shape == (batch_size, 256), (
        f"Expected ({batch_size}, 256), got {features.shape}"
    )

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HADIS] Total parameters:     {total_params:,}")
    print(f"[HADIS] Trainable parameters: {trainable_params:,}")
    print("[HADIS] Smoke test PASSED.")
