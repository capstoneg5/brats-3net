from dataclasses import dataclass

@dataclass(frozen=True)
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
