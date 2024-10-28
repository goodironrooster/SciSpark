import threading
import mmap
import time
from byte_status import ByteStatus

class ByteRetrieval:
    def __init__(self, buffer_size: int = 4096):
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._memory = mmap.mmap(-1, buffer_size)
        self._position = 0
        self.operation_times = []

    def fetch_byte(self, address: int) -> ByteStatus:
        start_time = time.perf_counter_ns()
        try:
            with self._lock:
                if not (0 <= address < self.buffer_size):
                    return ByteStatus(False, error="Address out of bounds")
                
                value = self._memory[address]
                return ByteStatus(True, value=value)
                
        except Exception as e:
            return ByteStatus(False, error=str(e))
        finally:
            end_time = time.perf_counter_ns()
            self.operation_times.append(end_time - start_time)
            
    def write_byte(self, address: int, value: int) -> ByteStatus:
        start_time = time.perf_counter_ns()
        try:
            with self._lock:
                if not (0 <= address < self.buffer_size):
                    return ByteStatus(False, error="Address out of bounds")
                if not (0 <= value <= 255):
                    return ByteStatus(False, error="Invalid byte value")
                
                self._memory[address] = value
                return ByteStatus(True)
                
        except Exception as e:
            return ByteStatus(False, error=str(e))
        finally:
            end_time = time.perf_counter_ns()
            self.operation_times.append(end_time - start_time)

    def simulate_error(self, error_type: str) -> None:
        if error_type == "memory_full":
            self._memory = None
        elif error_type == "corruption":
            self._memory[0] = 0xFF

    def __del__(self):
        if self._memory is not None:
            self._memory.close()
