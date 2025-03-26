from typing import Optional
from dataclasses import dataclass

uss = []


@dataclass
class UpstreamServer:
    url: str
    weight: Optional[float]
