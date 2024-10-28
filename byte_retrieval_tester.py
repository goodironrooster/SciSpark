import threading
from byte_retrieval import ByteRetrieval

class ByteRetrievalTester:
    def __init__(self):
        self.retriever = ByteRetrieval()
        self.test_results = []
        
    def run_test(self, name: str, test_func) -> bool:
        try:
            result = test_func()
            self.test_results.append((name, result, None))
            return result
        except Exception as e:
            self.test_results.append((name, False, str(e)))
            return False
            
    def test_basic_write_read(self) -> bool:
        write_status = self.retriever.write_byte(0, 42)
        if not write_status.success:
            return False
            
        read_status = self.retriever.fetch_byte(0)
        return read_status.success and read_status.value == 42
        
    def test_boundary_conditions(self) -> bool:
        write_status = self.retriever.write_byte(-1, 0)
        if write_status.success:
            return False
            
        write_status = self.retriever.write_byte(self.retriever.buffer_size, 0)
        if write_status.success:
            return False
            
        return True
        
    def test_concurrent_access(self) -> bool:
        def worker(address: int, value: int):
            self.retriever.write_byte(address, value)
            
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i, i))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        for i in range(10):
            status = self.retriever.fetch_byte(i)
            if not status.success or status.value != i:
                return False
                
        return True

    def test_sequential_access(self) -> bool:
        for i in range(0, 255):
            self.retriever.write_byte(i, i)
        
        for i in range(0, 255):
            status = self.retriever.fetch_byte(i)
            if not status.success or status.value != i:
                return False
        return True

    def test_memory_patterns(self) -> bool:
        patterns = [(0x00, 0xFF), (0xAA, 0x55), (0xFF, 0x00)]
        for addr, pattern in enumerate(patterns):
            self.retriever.write_byte(addr, pattern[0])
            status = self.retriever.fetch_byte(addr)
            if not status.success or status.value != pattern[0]:
                return False
        return True

    def test_error_handling(self) -> bool:
        new_retriever = ByteRetrieval()
        new_retriever.simulate_error("memory_full")
        status = new_retriever.fetch_byte(0)
        if status.success:
            return False
        return True
        
    def run_all_tests(self):
        tests = [
            ("Basic Write/Read", self.test_basic_write_read),
            ("Boundary Conditions", self.test_boundary_conditions),
            ("Concurrent Access", self.test_concurrent_access),
            ("Sequential Access", self.test_sequential_access),
            ("Memory Patterns", self.test_memory_patterns),
            ("Error Handling", self.test_error_handling)
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
            
    def print_results(self):
        print("\nTest Results:")
        print("-" * 50)
        for name, result, error in self.test_results:
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status}")
            if error:
                print(f"  Error: {error}")
            if result and hasattr(self.retriever, 'operation_times') and self.retriever.operation_times:
                avg_time = sum(self.retriever.operation_times) / len(self.retriever.operation_times)
                print(f"  Average operation time: {avg_time:.2f} ns")
        print("-" * 50)
