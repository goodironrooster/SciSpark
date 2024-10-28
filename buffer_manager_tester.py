import threading
from buffer_manager import BufferManager

class BufferManagerTester:
    def __init__(self):
        self.buffer_manager = BufferManager()
        self.test_results = []
        
    def run_test(self, name: str, test_func) -> bool:
        try:
            result = test_func()
            self.test_results.append((name, result, None))
            return result
        except Exception as e:
            self.test_results.append((name, False, str(e)))
            return False
            
    def test_basic_operations(self) -> bool:
        # Test write and read
        data = b"Hello, World!"
        write_status = self.buffer_manager.write(data)
        if not write_status.success:
            return False
            
        read_status = self.buffer_manager.read(len(data))
        if not read_status.success or read_status.data != data:
            return False
            
        return True
        
    def test_buffer_overflow(self) -> bool:
        # Test writing beyond buffer size
        large_data = bytes([i % 256 for i in range(40000)])
        status = self.buffer_manager.write(large_data)
        return not status.success  # Should fail
        
    def test_peek_operations(self) -> bool:
        data = b"Test data"
        self.buffer_manager.write(data)
        
        peek_status = self.buffer_manager.peek(4)
        if not peek_status.success or peek_status.data != b"Test":
            return False
            
        # Verify data still in buffer
        read_status = self.buffer_manager.read(4)
        return read_status.success and read_status.data == b"Test"
        
    def test_concurrent_access(self) -> bool:
        def writer():
            for i in range(100):
                self.buffer_manager.write(bytes([i % 256]))
                
        def reader():
            for i in range(100):
                self.buffer_manager.read(1)
                
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            threads.extend([t1, t2])
            t1.start()
            t2.start()
            
        for t in threads:
            t.join()
            
        return True
        
    def run_all_tests(self):
        self.buffer_manager.clear()  # Start fresh
        tests = [
            ("Basic Operations", self.test_basic_operations),
            ("Buffer Overflow", self.test_buffer_overflow),
            ("Peek Operations", self.test_peek_operations),
            ("Concurrent Access", self.test_concurrent_access)
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
            
    def print_results(self):
        print("\nBuffer Manager Test Results:")
        print("-" * 50)
        for name, result, error in self.test_results:
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status}")
            if error:
                print(f"  Error: {error}")
        print("-" * 50)