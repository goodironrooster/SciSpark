from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum

class StreamValidationType(Enum):
    ENCODING = "encoding"
    INTEGRITY = "integrity"
    BOUNDARY = "boundary"
    CORRUPTION = "corruption"
    FULL = "full"

@dataclass
class StreamStatus:
    success: bool
    validation_type: Optional[StreamValidationType] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    position: Optional[int] = None
    is_valid: Optional[bool] = None
