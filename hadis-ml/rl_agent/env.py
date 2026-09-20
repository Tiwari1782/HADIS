"""
HADIS — Countermeasure Decision RL Environment

High Altitude Drone Intelligence System
Author: Prakash Tiwari | Chandigarh Engineering College (IKGPTU)

Custom Gymnasium environment for training a PPO agent to select optimal
countermeasure actions against detected drones. The agent observes a
7-dimensional state vector encoding altitude, threat level, drone class,
atmospheric conditions, and sensor confidences, then selects from 5
countermeasure actions.

State Space:
    Box(7,) — [altitude_m/10000, threat_level/4, drone_class_idx/5,
               pressure_norm, temp_norm, rf_confidence, visual_confidence]

Action Space:
    Discrete(5):
        0 = Monitor only
        1 = RF jam 2.4GHz
        2 = RF jam 5.8GHz
        3 = Alert ground forces
        4 = Full spectrum jamming

Reward Structure:
    +10  correct action
    +25  bonus for neutralising threat Level 4
    -3   overkill (excessive response for low threat)
    -8   under-response (insufficient response for high threat)
    -0.1 per step (time penalty)
    -10  timeout (episode exceeds 50 steps)
"""

import os
import sys

# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# Optimal action mapping: drone class -> best countermeasure action
OPTIMAL_ACTION = {
    0: 0,  # consumer_quadcopter -> Monitor only
    1: 1,  # commercial_uav     -> RF jam 2.4GHz
    2: 2,  # tactical_uav       -> RF jam 5.8GHz
    3: 3,  # military_fixed_wing -> Alert ground forces
    4: 4,  # loitering_munition -> Full spectrum jamming
    5: 1,  # unknown            -> RF jam 2.4GHz (conservative default)
}

# Action names for logging
ACTION_NAMES = {
    0: "Monitor only",
    1: "RF jam 2.4GHz",
    2: "RF jam 5.8GHz",
    3: "Alert ground forces",
    4: "Full spectrum jamming",
}


class HADISCountermeasureEnv(gym.Env):
    """Custom Gymnasium environment for HADIS countermeasure decision-making.

    The agent must select the optimal countermeasure action given information
    about a detected drone: its class, threat level, altitude, atmospheric
    conditions, and sensor confidence scores.

    An episode ends when the agent selects the correct action or after
    50 steps (timeout).

    Args:
        isa_csv_path (str): Path to ISA atmosphere CSV file.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, isa_csv_path: str = None):
        """Initialise the environment.

        Args:
            isa_csv_path: Path to ISA atmosphere CSV. If None, attempts
                          to load from config.
        """
        super(HADISCountermeasureEnv, self).__init__()

        # Load config if available
        try:
            sys.path.append('/content/HADIS')
            from config import PATHS, DRONE_CLASSES, THREAT_LEVELS, NUM_CLASSES
            self.drone_classes = DRONE_CLASSES
            self.threat_levels = THREAT_LEVELS
            self.num_classes = NUM_CLASSES
            if isa_csv_path is None:
                isa_csv_path = PATHS['isa_data']
        except (ImportError, ModuleNotFoundError):
            # Fallback defaults for standalone testing
            self.drone_classes = [
                "consumer_quadcopter", "commercial_uav", "tactical_uav",
                "military_fixed_wing", "loitering_munition", "unknown"
            ]
            self.threat_levels = {
                "consumer_quadcopter": 1, "commercial_uav": 2,
                "tactical_uav": 3, "military_fixed_wing": 4,
                "loitering_munition": 4, "unknown": 3,
            }
            self.num_classes = 6

        # Load ISA atmospheric data
        self.isa_data = None
        if isa_csv_path is not None and os.path.exists(isa_csv_path):
            try:
                self.isa_df = pd.read_csv(isa_csv_path)
                self.isa_data = self.isa_df[
                    ['altitude_m', 'pressure_hPa', 'temperature_K', 'density', 'speed_of_sound']
                ].values.astype(np.float32)
                print(f'[HADIS] ISA data loaded: {len(self.isa_data)} rows from {isa_csv_path}')
            except Exception as e:
                print(f'[HADIS] WARNING: Failed to load ISA data: {e}')
                self.isa_data = None

        # If ISA data is unavailable, generate placeholder values
        if self.isa_data is None:
            print('[HADIS] Using synthetic ISA data for environment.')
            self.isa_data = np.array([
                [0, 1013.25, 288.15, 1.225, 340.3],
                [1000, 898.76, 281.65, 1.112, 336.4],
                [5000, 540.48, 255.65, 0.736, 320.5],
                [10000, 264.99, 223.15, 0.414, 299.5],
                [15000, 121.11, 216.65, 0.195, 295.1],
            ], dtype=np.float32)

        # Normalisation constants for ISA
        self.isa_pressure_range = (100.0, 1100.0)
        self.isa_temp_range = (200.0, 320.0)

        # --- Spaces ---
        # State: [alt_norm, threat_norm, class_norm, pressure_norm, temp_norm, rf_conf, visual_conf]
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

        # Action: 5 discrete countermeasure options
        self.action_space = spaces.Discrete(5)

        # Episode parameters
        self.max_steps = 50
        self.current_step = 0
        self.current_state = None
        self.current_drone_class = None
        self.current_threat_level = None

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector.

        Returns:
            np.ndarray: Observation of shape (7,).
        """
        return self.current_state.copy()

    def _sample_scenario(self):
        """Sample a random drone encounter scenario."""
        # Random drone class
        self.current_drone_class = np.random.randint(0, self.num_classes)
        class_name = self.drone_classes[self.current_drone_class]
        self.current_threat_level = self.threat_levels.get(class_name, 3)

        # Random ISA atmospheric conditions
        isa_row = self.isa_data[np.random.randint(0, len(self.isa_data))]
        altitude = isa_row[0]
        pressure = isa_row[1]
        temperature = isa_row[2]

        # Normalise components
        alt_norm = np.clip(altitude / 10000.0, 0.0, 1.0)
        threat_norm = np.clip(self.current_threat_level / 4.0, 0.0, 1.0)
        class_norm = np.clip(self.current_drone_class / 5.0, 0.0, 1.0)
        pressure_norm = np.clip(
            (pressure - self.isa_pressure_range[0]) /
            (self.isa_pressure_range[1] - self.isa_pressure_range[0]),
            0.0, 1.0,
        )
        temp_norm = np.clip(
            (temperature - self.isa_temp_range[0]) /
            (self.isa_temp_range[1] - self.isa_temp_range[0]),
            0.0, 1.0,
        )

        # Random sensor confidences
        rf_confidence = np.clip(np.random.beta(5, 2), 0.0, 1.0).astype(np.float32)
        visual_confidence = np.clip(np.random.beta(5, 2), 0.0, 1.0).astype(np.float32)

        self.current_state = np.array([
            alt_norm, threat_norm, class_norm,
            pressure_norm, temp_norm,
            rf_confidence, visual_confidence,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            tuple: (observation, info_dict).
        """
        super().reset(seed=seed)
        self.current_step = 0
        self._sample_scenario()

        info = {
            "drone_class": self.drone_classes[self.current_drone_class],
            "drone_class_idx": self.current_drone_class,
            "threat_level": self.current_threat_level,
            "optimal_action": OPTIMAL_ACTION[self.current_drone_class],
        }

        return self._get_obs(), info

    def step(self, action: int):
        """Execute one environment step.

        Args:
            action (int): Countermeasure action index (0-4).

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        self.current_step += 1

        optimal = OPTIMAL_ACTION[self.current_drone_class]
        reward = -0.1  # Per-step time penalty

        terminated = False
        truncated = False

        if action == optimal:
            # Correct action
            reward += 10.0
            terminated = True

            # Bonus for neutralising Level 4 threats
            if self.current_threat_level == 4:
                reward += 25.0

        elif action > optimal:
            # Overkill — excessive response
            reward -= 3.0

        else:
            # Under-response — insufficient action
            reward -= 8.0

        # Timeout check
        if self.current_step >= self.max_steps:
            reward -= 10.0
            truncated = True

        info = {
            "drone_class": self.drone_classes[self.current_drone_class],
            "threat_level": self.current_threat_level,
            "action_taken": ACTION_NAMES.get(action, f"Unknown({action})"),
            "optimal_action": ACTION_NAMES.get(optimal, f"Unknown({optimal})"),
            "is_correct": action == optimal,
            "step": self.current_step,
        }

        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    # --- Smoke Test: 3 random episodes ---
    print("[HADIS] Running HADISCountermeasureEnv smoke test (3 episodes)...")
    print("=" * 60)

    env = HADISCountermeasureEnv()

    for ep in range(3):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        steps = 0

        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Drone: {info['drone_class']} (class {info['drone_class_idx']})")
        print(f"  Threat level: {info['threat_level']}")
        print(f"  Optimal action: {ACTION_NAMES[info['optimal_action']]}")
        print(f"  Initial obs: {obs}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if done or steps <= 3:  # Print first few steps and final
                print(f"  Step {steps}: action={info['action_taken']}, "
                      f"reward={reward:.1f}, correct={info['is_correct']}")

        print(f"  Episode ended: steps={steps}, total_reward={total_reward:.1f}")

    env.close()
    print("\n" + "=" * 60)
    print("[HADIS] Smoke test PASSED.")
