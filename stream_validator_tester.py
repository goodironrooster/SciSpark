from buffer_manager import BufferManager
from stream_validator import StreamValidator
from test_utils import timeout
import time
import json
from typing import List, Tuple, Optional, Dict, Any

class StreamValidatorTester:
    def __init__(self):
        self.buffer_manager = BufferManager()
        self.stream_validator = StreamValidator(self.buffer_manager)
        self.test_results: List[Tuple[str, bool, Optional[str], float]] = []
        self.debug_mode = True
        self.performance_log: List[Dict[str, Any]] = []
        
    def debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG Tester: {message}")

    def log_performance(self, test_name: str, start_time: float, end_time: float) -> None:
        """Log performance metrics for a test"""
        duration = end_time - start_time
        self.performance_log.append({
            'test_name': test_name,
            'duration': duration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'start_time': start_time,
            'end_time': end_time
        })

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test with performance logging"""
        self.debug_print(f"Starting test: {name}")
        start_time = time.perf_counter()
        try:
            result = test_func()
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.log_performance(name, start_time, end_time)
            self.debug_print(f"Test {name} completed in {duration:.2f} seconds")
            self.test_results.append((name, result, None, duration))
            return result
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            error_msg = f"Error in {duration:.2f} seconds: {str(e)}"
            self.debug_print(error_msg)
            self.test_results.append((name, False, error_msg, duration))
            return False

    @timeout(5)
    def test_encoding_validation(self) -> bool:
        """Test encoding validation functionality"""
        self.debug_print("Testing encoding validation...")
        try:
            # Test valid ASCII data
            data = "Hello, World!".encode('utf-8')
            status = self.stream_validator.validate_encoding(data)
            self.debug_print(f"Encoding validation status (ASCII): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("ASCII validation failed")
                return False

            # Test valid UTF-8 data with non-ASCII characters
            data = "Hello, 世界!".encode('utf-8')
            status = self.stream_validator.validate_encoding(data)
            self.debug_print(f"Encoding validation status (UTF-8): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("UTF-8 validation failed")
                return False

            # Test invalid data
            data = b'\xFF\xFE\xFF\xFE'  # Invalid UTF-8 sequence
            status = self.stream_validator.validate_encoding(data)
            self.debug_print(f"Encoding validation status (Invalid): {status}")
            # This should either fail validation or detect as UTF-16
            if not status.success:
                return True  # It's okay if invalid data fails
            if status.is_valid and status.data.get('encoding') not in {'utf-16', 'utf-16le', 'utf-16be'}:
                self.debug_print(f"Invalid data was incorrectly validated as {status.data.get('encoding')}")
                return False

            self.debug_print("All encoding validation tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Encoding validation error: {str(e)}")
            return False

    @timeout(5)
    def test_integrity_validation(self) -> bool:
        """Test data integrity validation"""
        self.debug_print("Testing integrity validation...")
        try:
            # Test valid data
            data = b"Valid data stream"
            status = self.stream_validator.validate_integrity(data)
            self.debug_print(f"Integrity validation status (Valid): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("Valid data integrity check failed")
                return False

            # Test empty data
            status = self.stream_validator.validate_integrity(b"")
            self.debug_print(f"Integrity validation status (Empty): {status}")
            if status.success:
                self.debug_print("Empty data was incorrectly validated")
                return False

            # Test data with null bytes
            data = b"Test\x00data"
            status = self.stream_validator.validate_integrity(data)
            self.debug_print(f"Integrity validation status (Null bytes): {status}")
            if not status.success:
                self.debug_print("Null byte handling failed")
                return False

            self.debug_print("All integrity validation tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Integrity validation error: {str(e)}")
            return False

    @timeout(5)
    def test_boundary_validation(self) -> bool:
        """Test boundary validation functionality"""
        self.debug_print("Testing boundary validation...")
        try:
            # Test valid UTF-8 data
            data = "Hello, 世界!".encode('utf-8')
            status = self.stream_validator.validate_boundaries(data)
            self.debug_print(f"Boundary validation status (Valid UTF-8): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("UTF-8 boundary validation failed")
                return False

            # Test valid ASCII data
            data = "Hello, World!".encode('ascii')
            status = self.stream_validator.validate_boundaries(data)
            self.debug_print(f"Boundary validation status (Valid ASCII): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("ASCII boundary validation failed")
                return False

            # Test invalid boundaries
            data = b"Hello\xFF\xFE World"
            status = self.stream_validator.validate_boundaries(data)
            self.debug_print(f"Boundary validation status (Invalid): {status}")
            if status.success and status.is_valid:
                self.debug_print("Invalid boundary was incorrectly validated")
                return False

            self.debug_print("All boundary validation tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Boundary validation error: {str(e)}")
            return False

    @timeout(5)
    def test_corruption_detection(self) -> bool:
        """Test corruption detection functionality"""
        self.debug_print("Testing corruption detection...")
        try:
            # Test with clean data
            clean_data = "Clean data".encode('utf-8')
            status = self.stream_validator.check_corruption(clean_data)
            self.debug_print(f"Corruption detection status (Clean): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("Clean data was incorrectly marked as corrupted")
                return False

            # Test with corrupted data
            corrupted_data = b"Clean\xFF\xFEdata"
            status = self.stream_validator.check_corruption(corrupted_data)
            self.debug_print(f"Corruption detection status (Corrupted): {status}")
            if not status.success or status.is_valid:
                self.debug_print("Corrupted data was not detected")
                return False

            # Test with edge cases
            edge_data = b'\x80\x80'
            status = self.stream_validator.check_corruption(edge_data)
            self.debug_print(f"Corruption detection status (Edge): {status}")
            if not status.success:
                self.debug_print("Edge case handling failed")
                return False

            self.debug_print("All corruption detection tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Corruption detection error: {str(e)}")
            return False

    @timeout(5)
    def test_full_validation(self) -> bool:
        """Test full stream validation functionality"""
        self.debug_print("Testing full stream validation...")
        try:
            # Test with valid data
            self.debug_print("Testing valid data case")
            valid_data = "Hello, World!".encode('utf-8')
            status = self.stream_validator.validate_stream(valid_data)
            self.debug_print(f"Full validation status (Valid): {status}")
            if not status.success or not status.is_valid:
                self.debug_print("Valid data full validation failed")
                return False

            # Test with invalid data
            self.debug_print("Testing invalid data case")
            invalid_data = b"Invalid\xFF\xFEdata\x00stream"
            status = self.stream_validator.validate_stream(invalid_data)
            self.debug_print(f"Full validation status (Invalid): {status}")
            if status.success and status.is_valid:
                self.debug_print("Invalid data was incorrectly validated")
                return False

            self.debug_print("Full validation test passed")
            return True
        except Exception as e:
            self.debug_print(f"Full validation error: {str(e)}")
            return False

    def run_all_tests(self) -> None:
        """Run all validation tests"""
        self.debug_print("Starting all Stream Validator tests...")
        tests = [
            ("Encoding Validation", self.test_encoding_validation),
            ("Integrity Validation", self.test_integrity_validation),
            ("Boundary Validation", self.test_boundary_validation),
            ("Corruption Detection", self.test_corruption_detection),
            ("Full Stream Validation", self.test_full_validation)
        ]
        
        for name, test_func in tests:
            self.debug_print(f"\nExecuting test: {name}")
            self.run_test(name, test_func)

    def print_results(self) -> None:
        """Print test results and performance metrics"""
        print("\nStream Validator Test Results:")
        print("-" * 50)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, result, _, _ in self.test_results if result)

        # Print individual test results
        for name, result, error, elapsed in self.test_results:
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status} ({elapsed:.2f}s)")
            if error:
                print(f"  Error: {error}")

        # Print summary
        print("-" * 50)
        print(f"Tests Summary: {passed_tests}/{total_tests} passed")
        
        # Get and print validator performance metrics
        metrics = self.stream_validator.get_performance_metrics()
        print("\nPerformance Metrics:")
        print("-" * 50)
        for metric_name, metric_data in metrics.items():
            print(f"{metric_name}:")
            print(f"  Average: {metric_data['avg']:.6f}s")
            print(f"  Min: {metric_data['min']:.6f}s")
            print(f"  Max: {metric_data['max']:.6f}s")
            print(f"  Count: {metric_data['count']}")

    def save_performance_log(self, filename: str = "performance_log.json") -> None:
        """Save performance metrics to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
            self.debug_print(f"Performance log saved to {filename}")
        except Exception as e:
            self.debug_print(f"Error saving performance log: {str(e)}")
