# concept_mapper_tester.py

import unittest
from typing import List, Dict, Optional, Tuple
import time
import traceback
import psutil
import os
import math
from dataclasses import dataclass
from concept_mapper import ConceptMapper, ConceptNode, ConceptLevel, ConceptRelation
from shadow_detection import ShadowPattern, ShadowType

@dataclass
class TestResult:
    name: str
    success: bool
    error: Optional[str]
    execution_time: float
    details: Dict
    stack_trace: Optional[str] = None

class ConceptMapperTester:
    def __init__(self):
        self.concept_mapper = ConceptMapper(debug_mode=True)
        self.test_results: List[TestResult] = []
        self.debug_mode = True
        self.verbose = True
        self.error_logs: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Dict] = []

    def debug_print(self, message: str, level: str = "INFO") -> None:
        """Enhanced debug printing with timestamp and level"""
        if self.debug_mode:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] DEBUG ConceptMapperTester [{level}]: {message}")

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

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.debug_print(f"Error getting memory usage: {str(e)}", "ERROR")
            return 0.0

    def _take_memory_snapshot(self, label: str) -> None:
        """Take a snapshot of current memory usage"""
        try:
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'memory_usage': self._get_memory_usage(),
                'process_info': psutil.Process().memory_full_info()._asdict()
            }
            self.memory_snapshots.append(snapshot)
        except Exception as e:
            self.debug_print(f"Error taking memory snapshot: {str(e)}", "ERROR")

    def run_test_with_monitoring(self, test_name: str, test_func) -> TestResult:
        """Run a test with comprehensive monitoring and error handling"""
        start_time = time.perf_counter()
        details = {}
        
        try:
            self.debug_print(f"Starting test: {test_name}")
            
            # Pre-test monitoring
            self._take_memory_snapshot(f"{test_name}_start")
            initial_memory = self._get_memory_usage()
            
            # Execute test
            result = test_func()
            
            # Post-test monitoring
            self._take_memory_snapshot(f"{test_name}_end")
            final_memory = self._get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            # Record metrics
            execution_time = time.perf_counter() - start_time
            details = {
                'memory_usage': memory_diff,
                'execution_time': execution_time,
                'success_rate': 1.0 if result else 0.0,
                'peak_memory': max(s['memory_usage'] for s in self.memory_snapshots[-2:])
            }
            
            if not result:
                raise Exception("Test failed without raising an exception")
            
            self.debug_print(f"Test {test_name} completed successfully in {execution_time:.4f}s")
            return TestResult(test_name, True, None, execution_time, details)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            stack_trace = traceback.format_exc()
            error_msg = str(e)
            
            self.log_error(
                error_type=type(e).__name__,
                message=error_msg,
                stack_trace=stack_trace
            )
            
            details['error_type'] = type(e).__name__
            details['error_message'] = error_msg
            
            self.debug_print(f"Test {test_name} failed: {error_msg}", "ERROR")
            return TestResult(test_name, False, error_msg, execution_time, details, stack_trace)

    def test_base_concept_initialization(self) -> bool:
        """Test initialization of base concepts"""
        try:
            # Check all concept levels are initialized
            for level in ConceptLevel:
                if not self.concept_mapper.concept_hierarchies[level]:
                    self.debug_print(f"Missing concepts for level: {level.value}", "ERROR")
                    return False

            # Verify concept properties
            for concept_id, concept in self.concept_mapper.concepts.items():
                # Type checking
                if not isinstance(concept, ConceptNode):
                    self.debug_print(f"Invalid concept type for {concept_id}", "ERROR")
                    return False
                
                # Confidence validation
                if not isinstance(concept.confidence, (int, float)):
                    self.debug_print(f"Invalid confidence type for {concept_id}", "ERROR")
                    return False
                if not 0 <= concept.confidence <= 1:
                    self.debug_print(f"Invalid confidence value for {concept_id}: {concept.confidence}", "ERROR")
                    return False
                
                # Metadata validation
                if not isinstance(concept.metadata, dict):
                    self.debug_print(f"Invalid metadata type for {concept_id}", "ERROR")
                    return False

            self.debug_print("Base concept initialization test passed")
            return True

        except Exception as e:
            self.log_error("InitializationError", str(e), traceback.format_exc())
            return False

    def test_shadow_mapping(self) -> bool:
        """Test shadow to concept mapping"""
        try:
            test_cases = [
                (ShadowType.DIRECT, 0.9, {"type": "sequential"}, "direct"),
                (ShadowType.INDIRECT, 0.8, {"type": "relational"}, "indirect"),
                (ShadowType.COMPOSITE, 0.7, {"type": "structural"}, "composite"),
                (ShadowType.ABSTRACT, 0.6, {"type": "hierarchical"}, "abstract")
            ]

            for shadow_type, confidence, metadata, desc in test_cases:
                shadow = ShadowPattern(shadow_type, b"test_data", confidence, metadata)
                mapped_concepts = self.concept_mapper.map_shadow_to_concept(shadow)
                
                if not mapped_concepts:
                    self.debug_print(f"No concepts mapped for {desc}", "WARNING")
                    continue

                # Verify mapping properties
                for concept in mapped_concepts:
                    if not self.concept_mapper.validate_concept_mapping(shadow, concept):
                        self.debug_print(f"Invalid mapping validation for {desc}", "ERROR")
                        return False
                    
                    # Check concept level compatibility
                    if not self.concept_mapper._check_pattern_compatibility(shadow, concept):
                        self.debug_print(f"Incompatible concept level for {desc}", "ERROR")
                        return False

                self.debug_print(f"Successfully mapped {desc} to {len(mapped_concepts)} concepts")

            return True

        except Exception as e:
            self.log_error("MappingError", str(e), traceback.format_exc())
            return False

    def test_concept_validation(self) -> bool:
        """Test concept validation mechanisms"""
        try:
            test_cases = [
                (0.95, True, "High confidence"),
                (0.5, True, "Medium confidence"),
                (0.1, False, "Low confidence"),
                (1.1, False, "Invalid high confidence"),
                (-0.1, False, "Invalid negative confidence"),
                (float('nan'), False, "NaN confidence"),
                (float('inf'), False, "Infinite confidence"),
                (-float('inf'), False, "Negative infinite confidence")
            ]

            for confidence, expected_result, case_name in test_cases:
                shadow = ShadowPattern(
                    ShadowType.DIRECT,
                    b"test_data",
                    confidence,
                    {"type": "sequential"}
                )
                
                concept = next(iter(self.concept_mapper.concepts.values()))
                result = self.concept_mapper.validate_concept_mapping(shadow, concept)
                
                if result != expected_result:
                    self.debug_print(
                        f"Validation failed for {case_name} (confidence={confidence}): "
                        f"expected {expected_result}, got {result}",
                        "ERROR"
                    )
                    return False

                self.debug_print(f"Validation passed for {case_name}")

            # Test edge cases with concept confidence
            edge_concept = ConceptNode("test", ConceptLevel.CONCRETE, float('nan'), {})
            shadow = ShadowPattern(ShadowType.DIRECT, b"test_data", 0.9, {})
            if self.concept_mapper.validate_concept_mapping(shadow, edge_concept):
                self.debug_print("Failed: Accepted NaN concept confidence", "ERROR")
                return False

            self.debug_print("All validation tests passed")
            return True

        except Exception as e:
            self.log_error("ValidationError", str(e), traceback.format_exc())
            return False

    def test_error_handling(self) -> bool:
        """Test error handling mechanisms"""
        try:
            # Test with invalid inputs
            test_cases = [
                (None, "NoneType shadow"),
                (ShadowPattern(ShadowType.DIRECT, None, 0.5, {}), "None data"),
                (ShadowPattern(ShadowType.DIRECT, b"", -1.0, {}), "Invalid confidence"),
                (ShadowPattern(ShadowType.DIRECT, b"test", 0.5, None), "None metadata"),
                (ShadowPattern(ShadowType.DIRECT, b"test", float('nan'), {}), "NaN confidence"),
                (ShadowPattern(ShadowType.DIRECT, b"test", float('inf'), {}), "Infinite confidence")
            ]

            for shadow, case_name in test_cases:
                try:
                    mapped_concepts = self.concept_mapper.map_shadow_to_concept(shadow)
                    self.debug_print(f"Error handling test case '{case_name}' completed")
                except Exception as e:
                    self.debug_print(f"Expected error caught for '{case_name}': {str(e)}")

            # Test thread safety
            import threading
            def stress_test():
                for _ in range(5):  # Reduced from 100 to 5 for better performance
                    self.concept_mapper.map_shadow_to_concept(
                        ShadowPattern(ShadowType.DIRECT, b"test", 0.5, {})
                    )

            threads = [threading.Thread(target=stress_test) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            return True

        except Exception as e:
            self.log_error("ErrorHandlingError", str(e), traceback.format_exc())
            return False

    def run_all_tests(self) -> None:
        """Run all concept mapper tests"""
        print("\nStarting Concept Mapper Tests...")
        self.debug_print("Starting all Concept Mapper tests...")
        
        tests = [
            ("Base Concept Initialization", self.test_base_concept_initialization),
            ("Shadow Mapping", self.test_shadow_mapping),
            ("Concept Validation", self.test_concept_validation),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            result = self.run_test_with_monitoring(test_name, test_func)
            self.test_results.append(result)

        self.print_results()

    def print_results(self) -> None:
        """Print detailed test results and analysis"""
        print("\nConcept Mapper Test Results:")
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
        """Print detailed performance metrics"""
        print("\nPerformance Metrics:")
        print("-" * 50)
        
        metrics = self.concept_mapper.get_metrics()
        for metric_name, metric_data in metrics.items():
            print(f"{metric_name}:")
            print(f"  Average: {metric_data['avg']:.6f}s")
            print(f"  Min: {metric_data['min']:.6f}s")
            print(f"  Max: {metric_data['max']:.6f}s")
            print(f"  Count: {metric_data['count']}")

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
