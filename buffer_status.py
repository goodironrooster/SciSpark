from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class BufferStatus:
    success: bool
    buffer_size: int
    current_usage: int
    message: Optional[str] = None
    last_update: float = time.time()
