import unittest
import time
import logging
from typing import List, Dict, Optional, Any
import threading
import numpy as np
from dataclasses import dataclass
import traceback
from benchmark_system import BenchmarkSystem, BenchmarkMetricType, BenchmarkResult

@dataclass
class TestResult:
    name: str
    success: bool
    error: Optional[str]
    execution_time: float
    details: Dict[str, Any]
    stack_trace: Optional[str] = None

class BenchmarkSystemTester:
    def __init__(self):
        self.benchmark_system = BenchmarkSystem()
        self.test_results: List[TestResult] = []
        self.debug_mode = True
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('BenchmarkSystemTester')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def run_test_with_monitoring(self, test_name: str, test_func) -> TestResult:
        """Run a test with comprehensive monitoring"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting test: {test_name}")
            result = test_func()
            
            execution_time = time.time() - start_time
            
            details = {
                'execution_time': execution_time,
                'success_rate': 1.0 if result else 0.0,
                'memory_usage': self._get_memory_usage()
            }
            
            if not result:
                raise Exception(f"Test {test_name} failed")
                
            self.logger.info(f"Test {test_name} completed successfully in {execution_time:.4f}s")
            return TestResult(test_name, True, None, execution_time, details)
            
        except Exception as e:
            execution_time = time.time() - start_time
            stack_trace = traceback.format_exc()
            error_msg = str(e)
            
            self.logger.error(f"Error in {test_name}: {error_msg}")
            
            details = {
                'execution_time': execution_time,
                'success_rate': 0.0,
                'error_type': type(e).__name__,
                'error_message': error_msg,
                'memory_usage': self._get_memory_usage()
            }
            
            return TestResult(test_name, False, error_msg, execution_time, details, stack_trace)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def test_metric_recording(self) -> bool:
        """Test basic metric recording functionality"""
        try:
            self.benchmark_system.reset_metrics()
            
            # Test accuracy metric
            self.benchmark_system.record_metric(
                name="test_accuracy",
                value=0.85,
                metric_type=BenchmarkMetricType.ACCURACY,
                metadata={"test": "accuracy"}
            )
            
            # Test performance metric
            self.benchmark_system.record_metric(
                name="test_latency",
                value=0.1,
                metric_type=BenchmarkMetricType.PERFORMANCE,
                metadata={"test": "performance"}
            )
            
            # Verify metrics were recorded
            stats_accuracy = self.benchmark_system.get_metric_statistics("test_accuracy")
            stats_latency = self.benchmark_system.get_metric_statistics("test_latency")
            
            if not stats_accuracy or not stats_latency:
                self.logger.error("Failed to retrieve metric statistics")
                return False
                
            if abs(stats_accuracy['mean'] - 0.85) > 0.001:
                self.logger.error(f"Accuracy metric mismatch: {stats_accuracy['mean']} != 0.85")
                return False
                
            if abs(stats_latency['mean'] - 0.1) > 0.001:
                self.logger.error(f"Latency metric mismatch: {stats_latency['mean']} != 0.1")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in metric recording test: {str(e)}")
            return False

    def test_performance_evaluation(self) -> bool:
        """Test performance evaluation functionality"""
        try:
            self.benchmark_system.reset_metrics()
            
            # Record exactly 100 pairs of metrics (200 total operations)
            for i in range(100):
                self.benchmark_system.record_metric(
                    name="test_accuracy",
                    value=0.9,
                    metric_type=BenchmarkMetricType.ACCURACY,
                    metadata={"iteration": i}
                )
                
                self.benchmark_system.record_metric(
                    name="test_latency",
                    value=0.05,
                    metric_type=BenchmarkMetricType.PERFORMANCE,
                    metadata={"iteration": i}
                )
            
            # Evaluate performance
            meets_thresholds, details = self.benchmark_system.evaluate_performance()
            
            # Verify evaluation results
            if not meets_thresholds:
                self.logger.error("Performance evaluation failed to meet thresholds")
                return False
                
            operations_count = details['performance']['total_operations']
            if operations_count != 200:
                self.logger.error(f"Incorrect operation count: {operations_count} != 200")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in performance evaluation test: {str(e)}")
            return False

    def test_report_generation(self) -> bool:
        """Test report generation functionality"""
        try:
            self.benchmark_system.reset_metrics()
            
            # Generate some test data
            self.benchmark_system.record_metric(
                name="test_metric",
                value=0.95,
                metric_type=BenchmarkMetricType.ACCURACY
            )
            
            # Generate report
            report = self.benchmark_system.generate_report()
            
            # Verify report structure
            required_fields = [
                'timestamp', 
                'meets_thresholds', 
                'system_performance', 
                'evaluations', 
                'thresholds', 
                'metrics',
                'deviations'
            ]
            
            for field in required_fields:
                if field not in report:
                    self.logger.error(f"Missing required field in report: {field}")
                    return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in report generation test: {str(e)}")
            return False

    def test_concurrent_access(self) -> bool:
        """Test thread safety of the benchmark system"""
        try:
            self.benchmark_system.reset_metrics()
            
            def record_metrics():
                for _ in range(100):
                    self.benchmark_system.record_metric(
                        name="concurrent_test",
                        value=np.random.random(),
                        metric_type=BenchmarkMetricType.ACCURACY
                    )
            
            # Create and run multiple threads
            threads = [threading.Thread(target=record_metrics) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Verify results
            stats = self.benchmark_system.get_metric_statistics("concurrent_test")
            if stats['count'] != 500:  # 5 threads * 100 metrics each
                self.logger.error(f"Expected 500 metrics, got {stats['count']}")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in concurrent access test: {str(e)}")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        try:
            self.benchmark_system.reset_metrics()
            
            # Test invalid metric type
            try:
                self.benchmark_system.record_metric(
                    name="invalid_metric",
                    value=-1.0,
                    metric_type="invalid_type"  # Should be BenchmarkMetricType enum
                )
                self.logger.error("Failed to catch invalid metric type")
                return False
            except (ValueError, TypeError, AttributeError):
                # Successfully caught the error
                pass
            
            # Test invalid metric name
            try:
                self.benchmark_system.record_metric(
                    name="",
                    value=0.5,
                    metric_type=BenchmarkMetricType.ACCURACY
                )
                self.logger.error("Failed to catch empty metric name")
                return False
            except ValueError:
                # Successfully caught the error
                pass
            
            # Test invalid metric value
            try:
                self.benchmark_system.record_metric(
                    name="test_metric",
                    value="invalid_value",  # Should be float
                    metric_type=BenchmarkMetricType.ACCURACY
                )
                self.logger.error("Failed to catch invalid metric value")
                return False
            except (ValueError, TypeError):
                # Successfully caught the error
                pass
            
            return True

        except Exception as e:
            self.logger.error(f"Error in error handling test: {str(e)}")
            return False

    def test_resource_monitoring(self) -> bool:
        """Test resource monitoring capabilities"""
        try:
            self.benchmark_system.reset_metrics()
            
            # Record resource metrics
            self.benchmark_system.record_metric(
                name="memory_usage",
                value=100.0,  # MB
                metric_type=BenchmarkMetricType.RESOURCE,
                metadata={"type": "memory"}
            )
            
            # Verify resource tracking
            performance = self.benchmark_system.get_system_performance()
            if 'peak_memory_usage' not in performance:
                self.logger.error("Missing peak memory usage metric")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Error in resource monitoring test: {str(e)}")
            return False

    def run_all_tests(self) -> None:
        """Run all benchmark system tests"""
        tests = [
            ("Metric Recording", self.test_metric_recording),
            ("Performance Evaluation", self.test_performance_evaluation),
            ("Report Generation", self.test_report_generation),
            ("Concurrent Access", self.test_concurrent_access),
            ("Error Handling", self.test_error_handling),
            ("Resource Monitoring", self.test_resource_monitoring)
        ]

        for test_name, test_func in tests:
            result = self.run_test_with_monitoring(test_name, test_func)
            self.test_results.append(result)

    def print_results(self) -> None:
        """Print detailed test results"""
        print("\nBenchmark System Test Results:")
        print("-" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        
        for result in self.test_results:
            status = "PASS" if result.success else "FAIL"
            print(f"{result.name}: {status} ({result.execution_time:.3f}s)")
            
            if not result.success:
                print(f"  Error: {result.error}")
                if result.stack_trace:
                    print("  Stack trace:")
                    print("    " + "\n    ".join(result.stack_trace.split("\n")))
            
            print("  Details:")
            for key, value in result.details.items():
                print(f"    {key}: {value}")
            print()

        print("-" * 50)
        print(f"Tests Summary: {passed_tests}/{total_tests} passed\n")

        if passed_tests < total_tests:
            self.logger.warning(f"Some tests failed: {passed_tests}/{total_tests} passed")
        else:
            self.logger.info(f"All tests passed: {passed_tests}/{total_tests}")

if __name__ == "__main__":
    tester = BenchmarkSystemTester()
    tester.run_all_tests()
    tester.print_results()
