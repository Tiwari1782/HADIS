# HADIS — High Altitude Drone Intelligence System

> Simulation-Based High Altitude Performance Optimization and Robust Design
> of Anti-Drone System

---

## What Is This

HADIS is a multi-modal AI system that detects, classifies, and recommends
countermeasures for UAVs operating at high altitudes (0–10,000m) where
conventional ground-based detection systems fail.

The system fuses four sensor modalities simultaneously:
- RF Signal Analysis
- Radar Return Processing
- Visual Detection (Camera)
- Acoustic Signature Analysis

All models are trained entirely on simulated and publicly available data.
Real-world deployment testing is left as future work.

---

## Models

| Model | Purpose | Framework |
|---|---|---|
| YOLOv8 | Visual UAV detection | PyTorch |
| 1D-CNN / LSTM | RF signal classification | PyTorch |
| Transformer Fusion | Multi-modal sensor fusion | PyTorch |
| RL Agent (PPO/DQN) | Countermeasure optimization | Stable-Baselines3 |

---

## UAV Categories Detected

| Level | Category | Examples |
|---|---|---|
| 1 | Consumer Quadcopter | DJI Phantom, Mavic |
| 2 | Commercial UAV | Delivery drones |
| 3 | Tactical UAV | Bayraktar TB2 |
| 4 | Military Fixed-Wing | MQ-9 Reaper, RQ-4 |
| 4 | Loitering Munition | Shahed-136, Harop |

---

## Project Structure
```
HADIS/
├── config.py
├── requirements.txt
├── data/
│ └── synthetic_gen/
│ ├── gen_rf.py
│ ├── gen_radar.py
│ ├── gen_atmospheric.py
│ └── gen_rl_env.py
├── hadis-ml/
│ ├── yolov8/
│ ├── lstm_cnn/
│ ├── transformer_fusion/
│ └── rl_agent/
├── hadis-web/
│ ├── backend/
│ └── frontend/
└── hadis-paper/
```

---

## Storage

All datasets and model weights are stored on Google Drive.
Training is performed on Google Colab (T4 GPU).
No data is stored locally.

---

## Team

| Name | Role |
|---|---|
| Prakash Tiwari (Jack) | Lead Developer |
| Piyush Kumar | ML Engineer |
| Harnoor Kaur | Data Engineer |
| Prince Sagwal | Backend Engineer |

---

## Disclaimer

HADIS is a simulation-based research project. All results are scoped
to the defined simulation environment. No real-world operational
performance claims are made. Real-world validation is identified
as future work.

---

## License

MIT License
