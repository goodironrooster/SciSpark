# buffer_manager.py
import threading
from typing import Optional, List
from collections import deque
from byte_retrieval import ByteRetrieval
from buffer_status import BufferStatus

class BufferManager:
    def __init__(self, initial_size: int = 8192, max_size: int = 32768):
        self.initial_size = initial_size
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._byte_retriever = ByteRetrieval(buffer_size=initial_size)
        self._current_position = 0
        self._is_buffer_full = False
        
    def allocate(self, size: int) -> BufferStatus:
        try:
            with self._lock:
                if size > self.max_size:
                    return BufferStatus(
                        success=False,
                        buffer_size=0,
                        current_usage=0
                    )
                    
                new_buffer = deque(maxlen=size)
                self._buffer = new_buffer
                self._current_position = 0
                return BufferStatus(
                    success=True,
                    buffer_size=size,
                    current_usage=0
                )
        except Exception:
            return BufferStatus(
                success=False,
                buffer_size=0,
                current_usage=0
            )
            
    def write(self, data: bytes) -> BufferStatus:
        try:
            with self._lock:
                if len(data) + len(self._buffer) > self._buffer.maxlen:
                    return BufferStatus(
                        success=False,
                        buffer_size=self._buffer.maxlen,
                        current_usage=len(self._buffer)
                    )
                    
                for byte in data:
                    self._buffer.append(byte)
                    status = self._byte_retriever.write_byte(self._current_position, byte)
                    if not status.success:
                        return BufferStatus(
                            success=False,
                            buffer_size=self._buffer.maxlen,
                            current_usage=len(self._buffer)
                        )
                    self._current_position += 1
                    
                return BufferStatus(
                    success=True,
                    buffer_size=self._buffer.maxlen,
                    current_usage=len(self._buffer)
                )
        except Exception:
            return BufferStatus(
                success=False,
                buffer_size=self._buffer.maxlen,
                current_usage=len(self._buffer)
            )
            
    def read(self, size: Optional[int] = None) -> tuple[BufferStatus, Optional[bytes]]:
        try:
            with self._lock:
                if not self._buffer:
                    return BufferStatus(
                        success=False,
                        buffer_size=self._buffer.maxlen,
                        current_usage=0
                    ), None
                    
                read_size = size if size is not None else len(self._buffer)
                read_size = min(read_size, len(self._buffer))
                
                data = []
                for _ in range(read_size):
                    if self._buffer:
                        data.append(self._buffer.popleft())
                        
                return BufferStatus(
                    success=True,
                    buffer_size=self._buffer.maxlen,
                    current_usage=len(self._buffer)
                ), bytes(data)
        except Exception:
            return BufferStatus(
                success=False,
                buffer_size=self._buffer.maxlen,
                current_usage=len(self._buffer)
            ), None
            
    def peek(self, size: Optional[int] = None) -> tuple[BufferStatus, Optional[bytes]]:
        try:
            with self._lock:
                if not self._buffer:
                    return BufferStatus(
                        success=False,
                        buffer_size=self._buffer.maxlen,
                        current_usage=0
                    ), None
                    
                peek_size = size if size is not None else len(self._buffer)
                peek_size = min(peek_size, len(self._buffer))
                
                data = list(self._buffer)[:peek_size]
                return BufferStatus(
                    success=True,
                    buffer_size=self._buffer.maxlen,
                    current_usage=len(self._buffer)
                ), bytes(data)
        except Exception:
            return BufferStatus(
                success=False,
                buffer_size=self._buffer.maxlen,
                current_usage=len(self._buffer)
            ), None
            
    def clear(self) -> BufferStatus:
        try:
            with self._lock:
                self._buffer.clear()
                self._current_position = 0
                return BufferStatus(
                    success=True,
                    buffer_size=self._buffer.maxlen,
                    current_usage=0
                )
        except Exception:
            return BufferStatus(
                success=False,
                buffer_size=self._buffer.maxlen,
                current_usage=len(self._buffer)
            )
