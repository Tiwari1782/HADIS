"""
HADIS — Central Configuration

High Altitude Drone Intelligence System
Author: Prakash Tiwari | Chandigarh Engineering College (IKGPTU)

This module centralises every path, hyperparameter, and constant used
across the HADIS ML pipeline. All notebooks and modules import from
here — nothing is hardcoded elsewhere.

Usage:
    import sys
    sys.path.append('/content/HADIS')
    from config import PATHS, HYPERPARAMS, DRONE_CLASSES, THREAT_LEVELS, NUM_CLASSES
"""

import os

# ============================================================================
# Google Drive root for all HADIS research data
# ============================================================================
DRIVE_ROOT = "/content/drive/MyDrive/hadis-research"

# ============================================================================
# DRONE_CLASSES — the 6 UAV categories the system detects
# ============================================================================
DRONE_CLASSES = [
    "consumer_quadcopter",   # Level 1 — DJI Phantom, Mavic
    "commercial_uav",        # Level 2 — Delivery drones
    "tactical_uav",          # Level 3 — Bayraktar TB2
    "military_fixed_wing",   # Level 4 — MQ-9 Reaper, RQ-4
    "loitering_munition",    # Level 4 — Shahed-136, Harop
    "unknown",               # Level 3 — Unclassified target
]

NUM_CLASSES = len(DRONE_CLASSES)  # 6

# ============================================================================
# THREAT_LEVELS — maps each drone class to its threat severity (1–4)
# ============================================================================
THREAT_LEVELS = {
    "consumer_quadcopter": 1,
    "commercial_uav":      2,
    "tactical_uav":        3,
    "military_fixed_wing": 4,
    "loitering_munition":  4,
    "unknown":             3,
}

# ============================================================================
# PATHS — every dataset, weight, and log path used in the pipeline
# ============================================================================
PATHS = {
    # --- Datasets (read-only, already downloaded to Drive) ---
    "dronerf":           os.path.join(DRIVE_ROOT, "dronerf"),
    "drone_dataset":     os.path.join(DRIVE_ROOT, "processed-data"),
    "military_dataset":  os.path.join(DRIVE_ROOT, "military-aircraft"),
    "isa_data":          os.path.join(DRIVE_ROOT, "isa_atmosphere.csv"),

    # --- Model weights (saved during training) ---
    "weights_yolo":      os.path.join(DRIVE_ROOT, "weights", "yolov8"),
    "weights_lstm":      os.path.join(DRIVE_ROOT, "weights", "lstm_cnn"),
    "weights_fusion":    os.path.join(DRIVE_ROOT, "weights", "transformer_fusion"),
    "weights_rl":        os.path.join(DRIVE_ROOT, "weights", "rl_agent"),

    # --- Training logs, plots, and reports ---
    "logs":              os.path.join(DRIVE_ROOT, "logs"),
}

# ============================================================================
# HYPERPARAMS — all training hyperparameters
# ============================================================================
HYPERPARAMS = {
    # --- YOLOv8 visual detection ---
    "yolo_epochs":       100,
    "yolo_img_size":     640,
    "yolo_batch":        16,
    "yolo_patience":     20,

    # --- LSTM/CNN RF signal classifier ---
    "lstm_seq_len":      256,
    "lstm_epochs":       50,
    "lstm_batch":        32,

    # --- Transformer fusion ---
    "transformer_heads":  8,
    "transformer_layers": 4,
    "transformer_epochs": 40,

    # --- RL countermeasure agent (PPO) ---
    "rl_timesteps":      1_000_000,
    "rl_batch":          64,
    "rl_learning_rate":  3e-4,
}


# ============================================================================
# Quick sanity print when imported
# ============================================================================
if __name__ == "__main__":
    print("[HADIS] === Configuration Dump ===")
    print(f"[HADIS] Drive root:   {DRIVE_ROOT}")
    print(f"[HADIS] Num classes:  {NUM_CLASSES}")
    print(f"[HADIS] Drone classes: {DRONE_CLASSES}")
    print(f"[HADIS] Threat levels: {THREAT_LEVELS}")
    print()
    print("[HADIS] PATHS:")
    for key, val in PATHS.items():
        print(f"  {key:20s} -> {val}")
    print()
    print("[HADIS] HYPERPARAMS:")
    for key, val in HYPERPARAMS.items():
        print(f"  {key:22s} = {val}")
    print("[HADIS] === End ===")
