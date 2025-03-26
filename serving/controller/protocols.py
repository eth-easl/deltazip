from pydantic import BaseModel
from typing import Optional


class UpstreamRegistrationRequest(BaseModel):
    ip_address: str
    # None == uniform weight distribution
    weight: Optional[int] = None
