from dataclasses import dataclass

@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = True
