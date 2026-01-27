from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1
    epochs: int = 10
    lr: float = 1e-4
