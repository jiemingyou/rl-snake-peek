from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42

    grid_size: int = 32
    num_channels: int = 4  # head, body, food, direction

    lr: float = 1e-4
    lr_end: float = 1e-5  # final LR after full linear decay over total_steps
    gamma: float = 0.99
    batch_size: int = 64
    target_update_freq: int = 1000
    num_actions: int = 4

    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 500_000  # half of total_steps — larger grid needs longer exploration

    buffer_size: int = 50_000  # each 64x64 state is ~64KB; 50K keeps RAM around 6GB
    min_buffer: int = 10_000

    food_reward: float = 1.0
    death_penalty: float = -1.0
    step_penalty: float = -0.01

    num_envs: int = 32

    total_steps: int = 1_000_000
    checkpoint_every: int = 100_000
    log_every: int = 5_000
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"
    tb_log_dir: str = "runs"

    # Computed after init to stay in sync with grid_size
    max_steps_per_episode: int = field(init=False)

    def __post_init__(self):
        self.max_steps_per_episode = self.grid_size * self.grid_size * 4
