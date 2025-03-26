from dataclasses import dataclass


@dataclass
class SwapConfig:
    max_packed_model: int = 1
    max_cpu_model: int = 4
