from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Grid ---
    grid_size: int = 12
    num_channels: int = 4  # head, body, food, direction

    # --- DQN ---
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    target_update_freq: int = 1000
    num_actions: int = 4

    # --- Exploration ---
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 100_000

    # --- Replay buffer ---
    buffer_size: int = 100_000
    min_buffer: int = 10_000

    # --- Rewards ---
    food_reward: float = 1.0
    death_penalty: float = -1.0
    step_penalty: float = -0.01

    # --- Training ---
    total_steps: int = 500_000
    checkpoint_every: int = 50_000
    log_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"
    tb_log_dir: str = "runs"

    # --- Episode ---
    max_steps_per_episode: int = field(init=False)

    def __post_init__(self):
        self.max_steps_per_episode = self.grid_size * self.grid_size * 4
