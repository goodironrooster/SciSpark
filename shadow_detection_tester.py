import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from shadow_detection import ShadowDetectionSystem, ShadowPattern, ShadowType

class ShadowDetectionTester:
    def __init__(self):
        self.shadow_detector = ShadowDetectionSystem(debug_mode=True)
        self.test_results: List[Tuple[str, bool, Optional[str], float]] = []
        self.debug_mode = True

    def debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG ShadowTester: {message}")

    def run_test(self, name: str, test_func) -> bool:
        """Run a single test with performance logging"""
        self.debug_print(f"Starting test: {name}")
        start_time = time.perf_counter()
        try:
            result = test_func()
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.test_results.append((name, result, None, duration))
            self.debug_print(f"Test {name} completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            error_msg = str(e)
            self.debug_print(f"Test error: {error_msg}")
            self.test_results.append((name, False, error_msg, duration))
            return False

    def test_direct_shadow_detection(self) -> bool:
        """Test direct shadow pattern detection"""
        self.debug_print("Testing direct shadow detection...")
        try:
            # Test cases with expected results
            test_cases = [
                (b"ABCABCABC", "Simple repeating pattern", True),
                (b"AAAAABBBBB", "Consecutive repeating pattern", True),
                (b"A" * 20, "Single character repetition", True),
                (b"ABCDEFGHIJK", "No repetition pattern", False)
            ]
            
            for data, desc, should_have_shadows in test_cases:
                shadows = self.shadow_detector.detect_shadows(data)
                self.debug_print(f"Testing {desc}: Found {len(shadows)} shadows")
                
                # Find direct shadows
                direct_shadows = [s for s in shadows if s.pattern_type == ShadowType.DIRECT]
                
                # Validate all shadow confidence values
                for shadow in direct_shadows:
                    if not isinstance(shadow.confidence, float):
                        self.debug_print(f"Failed: Confidence not float type: {type(shadow.confidence)}")
                        return False
                    if not 0 <= shadow.confidence <= 1:
                        self.debug_print(f"Failed: Invalid confidence value: {shadow.confidence}")
                        return False
                
                # Validate shadow presence/absence
                if should_have_shadows and not direct_shadows:
                    self.debug_print(f"Failed: No direct shadows found in {desc}")
                    return False
                if not should_have_shadows and direct_shadows:
                    self.debug_print(f"Failed: Found unexpected direct shadows in {desc}")
                    return False
                
                # Validate shadow metadata
                for shadow in direct_shadows:
                    if not shadow.metadata:
                        self.debug_print("Failed: Shadow missing metadata")
                        return False
                    if 'pattern' not in shadow.metadata:
                        self.debug_print("Failed: Shadow missing pattern in metadata")
                        return False
            
            return True
            
        except Exception as e:
            self.debug_print(f"Direct shadow detection test error: {str(e)}")
            return False

    def test_indirect_shadow_detection(self) -> bool:
        """Test indirect shadow pattern detection"""
        self.debug_print("Testing indirect shadow detection...")
        try:
            # Test sequential patterns
            data = bytes(range(10))  # 0,1,2,3,4,5,6,7,8,9
            shadows = self.shadow_detector.detect_shadows(data)
            
            indirect_shadows = [s for s in shadows if s.pattern_type == ShadowType.INDIRECT]
            
            if not indirect_shadows:
                self.debug_print("Failed: No indirect shadows found")
                return False
                
            for shadow in indirect_shadows:
                if not isinstance(shadow.confidence, float):
                    self.debug_print(f"Failed: Invalid confidence type: {type(shadow.confidence)}")
                    return False
                if not 0 <= shadow.confidence <= 1:
                    self.debug_print(f"Failed: Invalid confidence value: {shadow.confidence}")
                    return False
                
            self.debug_print(f"Found {len(shadows)} shadows, indirect shadows: {bool(indirect_shadows)}")
            return True
            
        except Exception as e:
            self.debug_print(f"Indirect shadow detection test error: {str(e)}")
            return False

    def test_composite_shadow_detection(self) -> bool:
        """Test composite shadow pattern detection"""
        self.debug_print("Testing composite shadow detection...")
        try:
            # Test data with clear segments
            data = b"Header\x00Content\x00Footer"
            shadows = self.shadow_detector.detect_shadows(data)
            
            composite_shadows = [s for s in shadows if s.pattern_type == ShadowType.COMPOSITE]
            
            if not composite_shadows:
                self.debug_print("Failed: No composite shadows found")
                return False
                
            for shadow in composite_shadows:
                if not isinstance(shadow.confidence, float):
                    self.debug_print(f"Failed: Invalid confidence type: {type(shadow.confidence)}")
                    return False
                if not 0 <= shadow.confidence <= 1:
                    self.debug_print(f"Failed: Invalid confidence value: {shadow.confidence}")
                    return False
                if 'segments' not in shadow.metadata:
                    self.debug_print("Failed: Missing segment information")
                    return False
                    
            self.debug_print(f"Found {len(shadows)} shadows, composite shadows: {bool(composite_shadows)}")
            return True
            
        except Exception as e:
            self.debug_print(f"Composite shadow detection test error: {str(e)}")
            return False

    def test_abstract_shadow_detection(self) -> bool:
        """Test abstract shadow pattern detection"""
        self.debug_print("Testing abstract shadow detection...")
        try:
            # Test data with statistical patterns
            data = bytes([i % 256 for i in range(100)])  # Linear pattern
            shadows = self.shadow_detector.detect_shadows(data)
            
            abstract_shadows = [s for s in shadows if s.pattern_type == ShadowType.ABSTRACT]
            
            if not abstract_shadows:
                self.debug_print("Failed: No abstract shadows found")
                return False
                
            for shadow in abstract_shadows:
                if not isinstance(shadow.confidence, float):
                    self.debug_print(f"Failed: Invalid confidence type: {type(shadow.confidence)}")
                    return False
                if not 0 <= shadow.confidence <= 1:
                    self.debug_print(f"Failed: Invalid confidence value: {shadow.confidence}")
                    return False
                if 'stats' not in shadow.metadata:
                    self.debug_print("Failed: Missing statistical information")
                    return False
                    
            self.debug_print(f"Found {len(shadows)} shadows, abstract shadows: {bool(abstract_shadows)}")
            return True
            
        except Exception as e:
            self.debug_print(f"Abstract shadow detection test error: {str(e)}")
            return False

    def test_dynamic_shadow_detection(self) -> bool:
        """Test dynamic shadow pattern detection"""
        self.debug_print("Testing dynamic shadow detection...")
        try:
            # Test with changing patterns
            data1 = bytes([i % 256 for i in range(100)])
            data2 = bytes([(i + 1) % 256 for i in range(100)])
            
            # First detection establishes baseline
            shadows1 = self.shadow_detector.detect_shadows(data1)
            
            # Second detection should find dynamic changes
            shadows2 = self.shadow_detector.detect_shadows(data2)
            
            dynamic_shadows = [s for s in shadows2 if s.pattern_type == ShadowType.DYNAMIC]
            
            if not dynamic_shadows:
                self.debug_print("Failed: No dynamic shadows found")
                return False
                
            for shadow in dynamic_shadows:
                if not isinstance(shadow.confidence, float):
                    self.debug_print(f"Failed: Invalid confidence type: {type(shadow.confidence)}")
                    return False
                if not 0 <= shadow.confidence <= 1:
                    self.debug_print(f"Failed: Invalid confidence value: {shadow.confidence}")
                    return False
                if 'change_ratio' not in shadow.metadata:
                    self.debug_print("Failed: Missing change ratio information")
                    return False
                    
            self.debug_print(f"Found {len(shadows2)} shadows, dynamic shadows: {bool(dynamic_shadows)}")
            return True
            
        except Exception as e:
            self.debug_print(f"Dynamic shadow detection test error: {str(e)}")
            return False

    def run_all_tests(self) -> None:
        """Run all shadow detection tests"""
        self.debug_print("Starting all Shadow Detection tests...")
        tests = [
            ("Direct Shadow Detection", self.test_direct_shadow_detection),
            ("Indirect Shadow Detection", self.test_indirect_shadow_detection),
            ("Composite Shadow Detection", self.test_composite_shadow_detection),
            ("Abstract Shadow Detection", self.test_abstract_shadow_detection),
            ("Dynamic Shadow Detection", self.test_dynamic_shadow_detection)
        ]
        
        for name, test_func in tests:
            self.debug_print(f"\nExecuting test: {name}")
            self.run_test(name, test_func)

    def print_results(self) -> None:
        """Print test results and performance metrics"""
        print("\nShadow Detection Test Results:")
        print("-" * 50)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, result, _, _ in self.test_results if result)

        for name, result, error, elapsed in self.test_results:
            status = "PASS" if result else "FAIL"
            print(f"{name}: {status} ({elapsed:.2f}s)")
            if error:
                print(f"  Error: {error}")

        print("-" * 50)
        print(f"Tests Summary: {passed_tests}/{total_tests} passed")

        print("\nPerformance Metrics:")
        print("-" * 50)
        metrics = self.shadow_detector.get_performance_metrics()
        for metric_name, metric_data in metrics.items():
            print(f"{metric_name}:")
            print(f"  Average: {metric_data['avg']:.6f}s")
            print(f"  Min: {metric_data['min']:.6f}s")
            print(f"  Max: {metric_data['max']:.6f}s")
            print(f"  Count: {metric_data['count']}")
