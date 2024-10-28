import unittest
from typing import List, Dict, Optional, Tuple
import time
import traceback
import psutil
import os
from dataclasses import dataclass
import numpy as np
from failure_detection import FailureDetector, FailureVector, FailureType

@dataclass
class TestResult:
    name: str
    success: bool
    error: Optional[str]
    execution_time: float
    details: Dict
    stack_trace: Optional[str] = None

class FailureDetectionTester:
    def __init__(self):
        self.detector = FailureDetector(debug_mode=True)
        self.test_results: List[TestResult] = []
        self.debug_mode = True
        self.verbose = True
        self.error_logs: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Dict] = []
        self.tolerance = 0.2  # Acceptable deviation in scores

    def debug_print(self, message: str, level: str = "INFO") -> None:
        """Enhanced debug printing with timestamp and level"""
        if self.debug_mode:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] DEBUG FailureDetectionTester [{level}]: {message}")

    def run_test_with_monitoring(self, test_name: str, test_func) -> TestResult:
        """Run a test with comprehensive monitoring"""
        start_time = time.perf_counter()
        self._take_memory_snapshot(f"{test_name}_start")
        
        try:
            self.debug_print(f"Starting test: {test_name}")
            result = test_func()
            
            execution_time = time.perf_counter() - start_time
            self._take_memory_snapshot(f"{test_name}_end")
            
            details = {
                'memory_usage': self._get_memory_usage(),
                'execution_time': execution_time,
                'success_rate': 1.0 if result else 0.0
            }
            
            if not result:
                raise Exception(f"Test {test_name} failed")
                
            self.debug_print(f"Test {test_name} completed successfully in {execution_time:.4f}s")
            return TestResult(test_name, True, None, execution_time, details)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            stack_trace = traceback.format_exc()
            error_msg = str(e)
            
            self.log_error(type(e).__name__, error_msg, stack_trace)
            
            details = {
                'memory_usage': self._get_memory_usage(),
                'execution_time': execution_time,
                'success_rate': 0.0,
                'error_type': type(e).__name__,
                'error_message': error_msg
            }
            
            return TestResult(test_name, False, error_msg, execution_time, details, stack_trace)

    def test_factual_accuracy(self) -> bool:
        """Test factual accuracy detection"""
        try:
            test_cases = [
                {
                    'content': "The Earth is round and it orbits the Sun.",
                    'context': None,
                    'expected': 0.9,
                    'description': "True astronomical fact"
                },
                {
                    'content': "The Earth is flat and the sky is green.",
                    'context': None,
                    'expected': 0.1,
                    'description': "False facts"
                },
                {
                    'content': "2 + 2 = 4 is a basic mathematical fact.",
                    'context': None,
                    'expected': 0.9,
                    'description': "Mathematical fact"
                },
                {
                    'content': "Today is a nice day.",
                    'context': None,
                    'expected': 0.5,
                    'description': "Subjective statement"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'], case['context'])
                if abs(vector.factual_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Factual accuracy test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.factual_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("FactualAccuracyError", str(e), traceback.format_exc())
            return False

    def test_logical_consistency(self) -> bool:
        """Test logical consistency detection"""
        try:
            test_cases = [
                {
                    'content': "If it rains, the ground gets wet. It is raining now. Therefore, the ground is wet.",
                    'expected': 0.9,
                    'description': "Valid logical sequence"
                },
                {
                    'content': "All humans need water. Plants need water too.",
                    'expected': 0.8,
                    'description': "Related statements"
                },
                {
                    'content': "A is true. A is false.",
                    'expected': 0.1,
                    'description': "Direct contradiction"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'])
                if abs(vector.logical_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Logical consistency test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.logical_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("LogicalConsistencyError", str(e), traceback.format_exc())
            return False

    def test_toxicity_detection(self) -> bool:
        """Test toxicity detection"""
        try:
            test_cases = [
                {
                    'content': "You are amazing and helpful!",
                    'expected': 0.0,
                    'description': "Positive content"
                },
                {
                    'content': "I hate you, you stupid idiot!",
                    'expected': 0.8,
                    'description': "Highly toxic content"
                },
                {
                    'content': "This is terrible and awful work.",
                    'expected': 0.6,
                    'description': "Moderately toxic content"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'])
                if abs(vector.toxicity_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Toxicity detection test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.toxicity_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("ToxicityDetectionError", str(e), traceback.format_exc())
            return False

    def test_hallucination_detection(self) -> bool:
        """Test hallucination detection"""
        try:
            context = {"known_facts": ["Earth is a planet", "Sun is a star", "Mars is red"]}
            test_cases = [
                {
                    'content': "The Earth is a planet that orbits the Sun.",
                    'context': context,
                    'expected': 0.1,
                    'description': "Known fact"
                },
                {
                    'content': "Dragons live on Mars and breathe fire.",
                    'context': context,
                    'expected': 0.9,
                    'description': "Clear hallucination"
                },
                {
                    'content': "The sky sometimes appears blue.",
                    'context': context,
                    'expected': 0.5,
                    'description': "Common knowledge"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'], case['context'])
                if abs(vector.hallucination_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Hallucination detection test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.hallucination_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("HallucinationDetectionError", str(e), traceback.format_exc())
            return False

    def test_topic_relevance(self) -> bool:
        """Test topic relevance detection"""
        try:
            context = {"topic": "space exploration"}
            test_cases = [
                {
                    'content': "NASA launched a new rocket to explore Mars.",
                    'context': context,
                    'expected': 0.1,
                    'description': "Highly relevant content"
                },
                {
                    'content': "I enjoy baking chocolate cake.",
                    'context': context,
                    'expected': 0.9,
                    'description': "Off-topic content"
                },
                {
                    'content': "Some missions involve space travel.",
                    'context': context,
                    'expected': 0.5,
                    'description': "Partially relevant content"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'], case['context'])
                if abs(vector.off_topic_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Topic relevance test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.off_topic_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("TopicRelevanceError", str(e), traceback.format_exc())
            return False

    def test_novelty_detection(self) -> bool:
        """Test novelty detection"""
        try:
            test_cases = [
                {
                    'content': "This is a standard response template.",
                    'expected': 0.1,
                    'description': "Common phrase"
                },
                {
                    'content': "The quantum entanglement of consciousness could potentially explain parallel universe perception.",
                    'expected': 0.8,
                    'description': "Novel concept"
                },
                {
                    'content': "Hello world, this is a basic concept.",
                    'expected': 0.2,
                    'description': "Basic content"
                }
            ]

            for case in test_cases:
                vector = self.detector.detect_failures(case['content'])
                if abs(vector.novelty_score - case['expected']) > self.tolerance:
                    self.debug_print(
                        f"Novelty detection test failed for: '{case['description']}'\n"
                        f"Content: '{case['content']}'\n"
                        f"Expected: {case['expected']}, Got: {vector.novelty_score}"
                    )
                    return False

            return True

        except Exception as e:
            self.log_error("NoveltyDetectionError", str(e), traceback.format_exc())
            return False

    def test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        try:
            # Test with invalid inputs
            test_cases = [
                (None, "NoneType input", ValueError),
                ("", "Empty input", None),
                ("A" * 1000000, "Very long input", None),
                ("Mixed\0with\0nulls", "Null characters", None),
                ("🌟✨🌙", "Unicode characters", None)
            ]

            for content, case_name, expected_error in test_cases:
                try:
                    vector = self.detector.detect_failures(content)
                    if expected_error:
                        self.debug_print(f"Expected {expected_error.__name__} for '{case_name}' but no error was raised")
                        return False
                    self.debug_print(f"Error handling test case '{case_name}' completed successfully")
                except Exception as e:
                    if expected_error and not isinstance(e, expected_error):
                        self.debug_print(f"Expected {expected_error.__name__} for '{case_name}' but got {type(e).__name__}")
                        return False
                    elif expected_error:
                        self.debug_print(f"Got expected error for '{case_name}': {str(e)}")
                    else:
                        self.debug_print(f"Unexpected error for '{case_name}': {str(e)}")
                        return False

            return True

        except Exception as e:
            self.log_error("ErrorHandlingError", str(e), traceback.format_exc())
            return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            self.debug_print(f"Error getting memory usage: {str(e)}", "ERROR")
            return 0.0

    def _take_memory_snapshot(self, label: str) -> None:
        """Take a snapshot of current memory usage"""
        try:
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'memory_usage': self._get_memory_usage()
            }
            self.memory_snapshots.append(snapshot)
        except Exception as e:
            self.debug_print(f"Error taking memory snapshot: {str(e)}", "ERROR")

    def log_error(self, error_type: str, message: str, stack_trace: Optional[str] = None) -> None:
        """Log detailed error information"""
        error_log = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'type': error_type,
            'message': message,
            'stack_trace': stack_trace,
            'memory_usage': self._get_memory_usage()
        }
        self.error_logs.append(error_log)
        self.debug_print(f"Error logged: {error_type} - {message}", "ERROR")

    def run_all_tests(self) -> None:
        """Run all failure detection tests"""
        tests = [
            ("Factual Accuracy", self.test_factual_accuracy),
            ("Logical Consistency", self.test_logical_consistency),
            ("Toxicity Detection", self.test_toxicity_detection),
            ("Hallucination Detection", self.test_hallucination_detection),
            ("Topic Relevance", self.test_topic_relevance),
            ("Novelty Detection", self.test_novelty_detection),
            ("Error Handling", self.test_error_handling)
        ]

        for test_name, test_func in tests:
            result = self.run_test_with_monitoring(test_name, test_func)
            self.test_results.append(result)

    def print_results(self) -> None:
        """Print detailed test results"""
        print("\nFailure Detection Test Results:")
        print("-" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        
        for result in self.test_results:
            status = "PASS" if result.success else "FAIL"
            print(f"{result.name}: {status} ({result.execution_time:.3f}s)")
            
            if not result.success:
                print(f"  Error: {result.error}")
                if result.stack_trace and self.verbose:
                    print("  Stack trace:")
                    print("    " + "\n    ".join(result.stack_trace.split("\n")))
            
            if self.verbose:
                print("  Details:")
                for key, value in result.details.items():
                    print(f"    {key}: {value}")

        print("-" * 50)
        print(f"Tests Summary: {passed_tests}/{total_tests} passed")
        
        if self.error_logs:
            print("\nError Log Summary:")
            print("-" * 50)
            for error in self.error_logs:
                print(f"[{error['timestamp']}] {error['type']}: {error['message']}")
                print(f"Memory Usage: {error['memory_usage']:.2f} MB")

        self._print_performance_metrics()
        self._print_memory_analysis()

    def _print_performance_metrics(self) -> None:
        """Print performance metrics"""
        print("\nPerformance Metrics:")
        print("-" * 50)
        
        if self.performance_metrics:
            for metric_name, values in self.performance_metrics.items():
                print(f"{metric_name}:")
                print(f"  Average: {np.mean(values):.6f}s")
                print(f"  Min: {min(values):.6f}s")
                print(f"  Max: {max(values):.6f}s")
                print(f"  Count: {len(values)}")

    def _print_memory_analysis(self) -> None:
        """Print memory usage analysis"""
        if self.memory_snapshots:
            print("\nMemory Usage Analysis:")
            print("-" * 50)
            
            initial = self.memory_snapshots[0]['memory_usage']
            peak = max(s['memory_usage'] for s in self.memory_snapshots)
            final = self.memory_snapshots[-1]['memory_usage']
            
            print(f"Initial Memory: {initial:.2f} MB")
            print(f"Peak Memory: {peak:.2f} MB")
            print(f"Final Memory: {final:.2f} MB")
            print(f"Memory Growth: {final - initial:.2f} MB")

if __name__ == "__main__":
    tester = FailureDetectionTester()
    tester.run_all_tests()
    tester.print_results()
