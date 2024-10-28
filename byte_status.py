from dataclasses import dataclass
from typing import Optional

@dataclass
class ByteStatus:
    success: bool
    value: Optional[int] = None
    error: Optional[str] = None
