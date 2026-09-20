"""
HADIS — Multi-Modal Transformer Fusion Model

High Altitude Drone Intelligence System
Author: Prakash Tiwari | Chandigarh Engineering College (IKGPTU)

This module implements the multi-modal fusion architecture that combines:
    1. Visual features from frozen YOLOv8 backbone (512-d)
    2. RF signal features from frozen LSTM/CNN model (256-d)
    3. Atmospheric features from ISA data via an MLP (32-d)

All three feature vectors are projected into a shared embedding space and
processed by a Transformer Encoder for cross-modal attention. Two output
heads provide drone classification and threat level regression.

Architecture:
    Input: yolo_feat(512), rf_feat(256), atmos_feat(32)
    -> Project each to d_model=256 via linear layers
    -> Stack to sequence (batch, 3, 256)
    -> TransformerEncoder (nhead=8, num_layers=4, dim_ff=512, dropout=0.1)
    -> Mean pool over sequence dim
    -> Classification head: 256 -> 128 -> 6
    -> Threat regression head: 256 -> 64 -> 1
"""

import torch
import torch.nn as nn


class AtmosphericMLP(nn.Module):
    """MLP for encoding atmospheric / ISA data into a feature vector.

    Takes 5 ISA parameters (altitude_m, pressure_hPa, temperature_K,
    density, speed_of_sound) and produces a 32-dimensional embedding.

    Args:
        input_dim (int): Number of atmospheric input features (default: 5).
        output_dim (int): Dimension of output embedding (default: 32).
    """

    def __init__(self, input_dim: int = 5, output_dim: int = 32):
        super(AtmosphericMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Atmospheric features of shape (batch, 5).

        Returns:
            torch.Tensor: Embedding of shape (batch, 32).
        """
        return self.net(x)


class HADISFusionTransformer(nn.Module):
    """Multi-modal Transformer Fusion model for HADIS.

    Fuses visual, RF, and atmospheric features through cross-modal
    Transformer attention, producing drone class logits and a threat score.

    Args:
        yolo_feat_dim (int): Dimension of YOLOv8 visual features (default: 512).
        rf_feat_dim (int): Dimension of LSTM/CNN RF features (default: 256).
        atmos_feat_dim (int): Dimension of atmospheric MLP output (default: 32).
        d_model (int): Internal Transformer dimension (default: 256).
        nhead (int): Number of attention heads (default: 8).
        num_layers (int): Number of Transformer encoder layers (default: 4).
        dim_feedforward (int): Feedforward dimension in Transformer (default: 512).
        dropout (float): Dropout rate (default: 0.1).
        num_classes (int): Number of output classes (default: 6).
    """

    def __init__(
        self,
        yolo_feat_dim: int = 512,
        rf_feat_dim: int = 256,
        atmos_feat_dim: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 6,
    ):
        super(HADISFusionTransformer, self).__init__()
        self.d_model = d_model

        # --- Projection layers: map each modality to d_model ---
        self.proj_yolo = nn.Linear(yolo_feat_dim, d_model)
        self.proj_rf = nn.Linear(rf_feat_dim, d_model)
        self.proj_atmos = nn.Linear(atmos_feat_dim, d_model)

        # --- Learnable modality embeddings ---
        self.modality_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- Layer norm after transformer ---
        self.layer_norm = nn.LayerNorm(d_model)

        # --- Classification head ---
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # --- Threat regression head ---
        self.threat_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        yolo_feat: torch.Tensor,
        rf_feat: torch.Tensor,
        atmos_feat: torch.Tensor,
    ) -> tuple:
        """Forward pass through the fusion model.

        Args:
            yolo_feat (torch.Tensor): Visual features from YOLOv8, shape (batch, 512).
            rf_feat (torch.Tensor): RF features from LSTM/CNN, shape (batch, 256).
            atmos_feat (torch.Tensor): Atmospheric features, shape (batch, 32).

        Returns:
            tuple: (class_logits (batch, 6), threat_score (batch, 1)).
        """
        # Project each modality to d_model
        v = self.proj_yolo(yolo_feat)     # (batch, 256)
        r = self.proj_rf(rf_feat)          # (batch, 256)
        a = self.proj_atmos(atmos_feat)    # (batch, 256)

        # Stack into sequence: (batch, 3, d_model)
        seq = torch.stack([v, r, a], dim=1)

        # Add learnable modality embeddings
        seq = seq + self.modality_embedding.unsqueeze(0)

        # Transformer encoder
        encoded = self.transformer_encoder(seq)   # (batch, 3, d_model)
        encoded = self.layer_norm(encoded)

        # Mean pool over the sequence (modality) dimension
        pooled = encoded.mean(dim=1)               # (batch, d_model)

        # Output heads
        class_logits = self.classification_head(pooled)   # (batch, num_classes)
        threat_score = self.threat_head(pooled)            # (batch, 1)

        return class_logits, threat_score


def get_fusion_model(
    yolo_feat_dim: int = 512,
    rf_feat_dim: int = 256,
    atmos_feat_dim: int = 32,
    num_classes: int = 6,
) -> tuple:
    """Factory function to create the fusion model and atmospheric MLP.

    Args:
        yolo_feat_dim: Dimension of YOLOv8 features.
        rf_feat_dim: Dimension of RF features.
        atmos_feat_dim: Dimension of atmospheric MLP output.
        num_classes: Number of output classes.

    Returns:
        tuple: (HADISFusionTransformer, AtmosphericMLP) pair.
    """
    atmos_mlp = AtmosphericMLP(input_dim=5, output_dim=atmos_feat_dim)
    fusion = HADISFusionTransformer(
        yolo_feat_dim=yolo_feat_dim,
        rf_feat_dim=rf_feat_dim,
        atmos_feat_dim=atmos_feat_dim,
        num_classes=num_classes,
    )
    return fusion, atmos_mlp


if __name__ == "__main__":
    # --- Smoke Test ---
    print("[HADIS] Running HADISFusionTransformer smoke test...")

    batch_size = 4
    num_classes = 6

    # Create models
    fusion_model, atmos_mlp = get_fusion_model(num_classes=num_classes)

    # Create dummy inputs
    yolo_feat = torch.randn(batch_size, 512)
    rf_feat = torch.randn(batch_size, 256)
    raw_atmos = torch.randn(batch_size, 5)

    # Forward pass through atmospheric MLP
    atmos_feat = atmos_mlp(raw_atmos)
    print(f"[HADIS] AtmosphericMLP input:  {raw_atmos.shape}")
    print(f"[HADIS] AtmosphericMLP output: {atmos_feat.shape}")
    assert atmos_feat.shape == (batch_size, 32), (
        f"Expected ({batch_size}, 32), got {atmos_feat.shape}"
    )

    # Forward pass through fusion transformer
    class_logits, threat_score = fusion_model(yolo_feat, rf_feat, atmos_feat)
    print(f"[HADIS] Fusion input shapes: yolo={yolo_feat.shape}, rf={rf_feat.shape}, atmos={atmos_feat.shape}")
    print(f"[HADIS] Class logits shape:  {class_logits.shape}")
    print(f"[HADIS] Threat score shape:  {threat_score.shape}")
    assert class_logits.shape == (batch_size, num_classes), (
        f"Expected ({batch_size}, {num_classes}), got {class_logits.shape}"
    )
    assert threat_score.shape == (batch_size, 1), (
        f"Expected ({batch_size}, 1), got {threat_score.shape}"
    )

    # Parameter counts
    atmos_params = sum(p.numel() for p in atmos_mlp.parameters())
    fusion_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"[HADIS] AtmosphericMLP parameters:        {atmos_params:,}")
    print(f"[HADIS] HADISFusionTransformer parameters: {fusion_params:,}")
    print(f"[HADIS] Total parameters:                  {atmos_params + fusion_params:,}")
    print("[HADIS] Smoke test PASSED.")
