from pattern_analyzer import PatternAnalyzer, PatternType, PatternStatus
from buffer_manager import BufferManager
from stream_validator import StreamValidator
from test_utils import timeout
import time
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class PatternAnalyzerTester:
    def __init__(self):
        self.buffer_manager = BufferManager()
        self.stream_validator = StreamValidator(self.buffer_manager)
        self.pattern_analyzer = PatternAnalyzer(self.buffer_manager, self.stream_validator)
        self.test_results: List[Tuple[str, bool, Optional[str], float]] = []
        self.debug_mode = True
        self.performance_log: List[Dict[str, Any]] = []
        
    def debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG PatternTester: {message}")

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
    def test_repeating_patterns(self) -> bool:
        """Test repeating pattern detection"""
        self.debug_print("Testing repeating pattern detection...")
        try:
            # Test with simple repeating pattern
            data = b"ABCABCABCABC"  # Simple repeating pattern
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.REPEATING])
            self.debug_print(f"Simple repeating pattern status: {status.patterns}")
            if not status.success or not status.patterns.get('repeating', {}).get('total_patterns', 0) > 0:
                self.debug_print("Failed to detect simple repeating pattern")
                return False

            # Test with no repeating pattern
            data = bytes(range(10))  # Sequential bytes without repetition
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.REPEATING])
            self.debug_print(f"Non-repeating pattern status: {status.patterns}")
            if not status.success or status.patterns.get('repeating', {}).get('total_patterns', 0) != 0:
                self.debug_print("Incorrectly detected pattern in non-repeating sequence")
                return False

            # Test with complex repeating pattern
            data = b"ABC123ABC123"  # Complex repeating pattern
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.REPEATING])
            self.debug_print(f"Complex repeating pattern status: {status.patterns}")
            if not status.success or not status.patterns.get('repeating', {}).get('total_patterns', 0) > 0:
                self.debug_print("Failed to detect complex repeating pattern")
                return False

            self.debug_print("All repeating pattern tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Repeating pattern test error: {str(e)}")
            return False

    @timeout(5)
    def test_sequential_patterns(self) -> bool:
        """Test sequential pattern detection"""
        self.debug_print("Testing sequential pattern detection...")
        try:
            # Test with simple sequence
            data = bytes(range(10))  # 0,1,2,3,4,5,6,7,8,9
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.SEQUENTIAL])
            self.debug_print(f"Simple sequence status: {status.patterns}")
            if not status.success or not status.patterns.get('sequential', {}).get('sequences'):
                self.debug_print("Failed to detect simple sequence")
                return False

            # Test with alternating sequence
            data = bytes([0, 2, 0, 2, 0, 2])
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.SEQUENTIAL])
            self.debug_print(f"Alternating sequence status: {status.patterns}")
            if not status.success or not status.patterns.get('sequential', {}).get('sequences'):
                self.debug_print("Failed to detect alternating sequence")
                return False

            # Test with random sequence
            data = bytes([x % 256 for x in range(0, 50, 7)])  # Non-trivial sequence
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.SEQUENTIAL])
            self.debug_print(f"Complex sequence status: {status.patterns}")
            if not status.success:
                self.debug_print("Failed to analyze complex sequence")
                return False

            self.debug_print("All sequential pattern tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Sequential pattern test error: {str(e)}")
            return False

    @timeout(5)
    def test_structural_patterns(self) -> bool:
        """Test structural pattern detection"""
        self.debug_print("Testing structural pattern detection...")
        try:
            # Test with structured data
            data = b"Header\x00Data\x00Footer"
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STRUCTURAL])
            self.debug_print(f"Structured data status: {status.patterns}")
            if not status.success or not status.patterns.get('structural', {}).get('boundary_markers'):
                self.debug_print("Failed to detect basic structure")
                return False

            # Test with boundary markers
            data = b"\xFF" + b"Section1" + b"\x00" + b"Section2" + b"\xFF"
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STRUCTURAL])
            self.debug_print(f"Boundary marker status: {status.patterns}")
            if not status.success or not status.patterns.get('structural', {}).get('boundary_markers'):
                self.debug_print("Failed to detect boundary markers")
                return False

            # Test with mixed content
            data = bytes([65, 66, 0, 67, 255, 68, 0, 69])  # Mixed content with boundaries
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STRUCTURAL])
            self.debug_print(f"Mixed content status: {status.patterns}")
            if not status.success:
                self.debug_print("Failed to analyze mixed content")
                return False

            self.debug_print("All structural pattern tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Structural pattern test error: {str(e)}")
            return False

    @timeout(5)
    def test_statistical_patterns(self) -> bool:
        """Test statistical pattern detection"""
        self.debug_print("Testing statistical pattern detection...")
        try:
            # Test with normal distribution
            mu, sigma = 128, 20
            data = bytes([min(max(int(x), 0), 255) for x in np.random.normal(mu, sigma, 100)])
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STATISTICAL])
            self.debug_print(f"Normal distribution status: {status.patterns}")
            if not status.success or 'statistical' not in status.patterns:
                self.debug_print("Failed to analyze normal distribution")
                return False

            # Test with uniform distribution
            data = bytes([x % 256 for x in range(100)])
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STATISTICAL])
            self.debug_print(f"Uniform distribution status: {status.patterns}")
            if not status.success or 'statistical' not in status.patterns:
                self.debug_print("Failed to analyze uniform distribution")
                return False

            # Test with constant data
            data = bytes([128] * 100)
            status = self.pattern_analyzer.analyze_sequence(data, [PatternType.STATISTICAL])
            self.debug_print(f"Constant data status: {status.patterns}")
            if not status.success or 'statistical' not in status.patterns:
                self.debug_print("Failed to analyze constant data")
                return False

            self.debug_print("All statistical pattern tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Statistical pattern test error: {str(e)}")
            return False

    @timeout(5)
    def test_anomaly_detection(self) -> bool:
        """Test anomaly detection"""
        self.debug_print("Testing anomaly detection...")
        try:
            # Test with clear anomaly
            base_data = bytes([100] * 50)
            anomaly_pos = 25
            data = bytearray(base_data)
            data[anomaly_pos] = 200
            anomalies = self.pattern_analyzer.detect_anomalies(bytes(data))
            self.debug_print(f"Clear anomaly detection results: {anomalies}")
            if not any(a['position'] == anomaly_pos for a in anomalies):
                self.debug_print("Failed to detect clear anomaly")
                return False

            # Test with no anomalies
            normal_data = bytes([100] * 50)
            anomalies = self.pattern_analyzer.detect_anomalies(normal_data)
            self.debug_print(f"No anomaly results: {anomalies}")
            if anomalies:
                self.debug_print("False positive in anomaly detection")
                return False

            # Test with multiple anomalies
            data = bytearray([100] * 50)
            data[20] = 200
            data[35] = 50
            anomalies = self.pattern_analyzer.detect_anomalies(bytes(data))
            self.debug_print(f"Multiple anomaly results: {anomalies}")
            if len(anomalies) < 2:
                self.debug_print("Failed to detect multiple anomalies")
                return False

            self.debug_print("All anomaly detection tests passed")
            return True
        except Exception as e:
            self.debug_print(f"Anomaly detection test error: {str(e)}")
            return False

    def run_all_tests(self) -> None:
        """Run all pattern analyzer tests"""
        self.debug_print("Starting all Pattern Analyzer tests...")
        tests = [
            ("Repeating Patterns", self.test_repeating_patterns),
            ("Sequential Patterns", self.test_sequential_patterns),
            ("Structural Patterns", self.test_structural_patterns),
            ("Statistical Patterns", self.test_statistical_patterns),
            ("Anomaly Detection", self.test_anomaly_detection)
        ]
        
        for name, test_func in tests:
            self.debug_print(f"\nExecuting test: {name}")
            self.run_test(name, test_func)

    def print_results(self) -> None:
        """Print test results and performance metrics"""
        print("\nPattern Analyzer Test Results:")
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
        
        # Get and print analyzer performance metrics
        metrics = self.pattern_analyzer.get_performance_metrics()
        print("\nPerformance Metrics:")
        print("-" * 50)
        for metric_name, metric_data in metrics.items():
            print(f"{metric_name}:")
            print(f"  Average: {metric_data['avg']:.6f}s")
            print(f"  Min: {metric_data['min']:.6f}s")
            print(f"  Max: {metric_data['max']:.6f}s")
            print(f"  Count: {metric_data['count']}")

    def save_performance_log(self, filename: str = "pattern_analyzer_performance.json") -> None:
        """Save performance metrics to a file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.performance_log, f, indent=2)
            self.debug_print(f"Performance log saved to {filename}")
        except Exception as e:
            self.debug_print(f"Error saving performance log: {str(e)}")
