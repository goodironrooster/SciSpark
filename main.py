import logging
import time
from typing import Dict, Optional
import numpy as np
from failure_detection_tester import FailureDetectionTester
from shadow_concept_tester import ShadowConceptTester
from benchmark_system_tester import BenchmarkSystemTester
from failure_detection import FailureDetector
from shadow_concept import ShadowConceptSystem, ConceptType
from benchmark_system import BenchmarkSystem, BenchmarkMetricType

class IntegratedSystemTester:
    def __init__(self):
        self.logger = self._setup_logger()
        self.failure_detector = FailureDetector()
        self.shadow_system = ShadowConceptSystem()
        self.benchmark_system = BenchmarkSystem()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('IntegratedSystemTester')
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

    def run_integrated_tests(self):
        """Run comprehensive tests of the integrated system"""
        self.logger.info("Starting Integrated System Tests...")
        print("\n" + "="*50)
        print("Starting Integrated System Tests...")
        print("="*50 + "\n")

        try:
            # 1. Test Failure Detection System
            self.logger.info("Starting Failure Detection System Tests...")
            print("\nStarting Failure Detection System Tests...")
            print("-" * 50)
            
            failure_tester = FailureDetectionTester()
            failure_tester.run_all_tests()
            failure_tester.print_results()

            # 2. Test Shadow Concept System
            self.logger.info("Starting Shadow Concept System Tests...")
            print("\nStarting Shadow Concept System Tests...")
            print("-" * 50)
            
            shadow_tester = ShadowConceptTester()
            shadow_tester.run_all_tests()
            shadow_tester.print_results()

            # 3. Test Benchmark System
            self.logger.info("Starting Benchmark System Tests...")
            print("\nStarting Benchmark System Tests...")
            print("-" * 50)
            
            benchmark_tester = BenchmarkSystemTester()
            benchmark_tester.run_all_tests()
            benchmark_tester.print_results()

            # 4. Test Integration
            self.logger.info("Starting Integration Tests...")
            print("\nStarting Integration Tests...")
            print("-" * 50)
            self._run_integration_tests()

        except Exception as e:
            self.logger.error(f"Error in test execution: {str(e)}")
            raise

    def _run_integration_tests(self):
        """Run tests that verify the integration between systems"""
        test_cases = [
            {
                'name': 'Basic Integration Test',
                'content': "The Earth is flat and the sky is green.",
                'expected_failure_types': ['factual_inaccuracy'],
                'concept_type': ConceptType.FACTUAL,
                'expected_scores': {
                    'factual': 0.1,
                    'logical': 0.7,
                    'toxicity': 0.0,
                    'hallucination': 0.9
                },
                'benchmark_expectations': {
                    'latency': 0.1,
                    'accuracy': 0.9
                }
            },
            {
                'name': 'Performance Test',
                'content': "Testing system performance under load.",
                'expected_failure_types': [],
                'concept_type': ConceptType.PROCEDURAL,
                'expected_scores': {
                    'factual': 0.5,
                    'logical': 0.7,
                    'toxicity': 0.0,
                    'hallucination': 0.5
                },
                'benchmark_expectations': {
                    'latency': 0.05,
                    'throughput': 100
                }
            }
        ]

        print("\nRunning Integration Tests:")
        print("=" * 50)
        
        total_tests = len(test_cases)
        passed_tests = 0
        failed_tests = []
        
        for test_case in test_cases:
            try:
                success = self._run_single_integration_test(test_case)
                if success:
                    passed_tests += 1
                else:
                    failed_tests.append(test_case['name'])
            except Exception as e:
                self.logger.error(f"Test case {test_case['name']} failed: {str(e)}")
                failed_tests.append(test_case['name'])
        
        print("\nIntegration Tests Summary:")
        print(f"Passed: {passed_tests}/{total_tests}")
        if failed_tests:
            print("Failed tests:")
            for test_name in failed_tests:
                print(f"  - {test_name}")
        print("=" * 50)

    def _run_single_integration_test(self, test_case: Dict) -> bool:
        """Run a single integration test case"""
        try:
            self.logger.info(f"Running integration test: {test_case['name']}")
            print(f"\nRunning: {test_case['name']}")
            print(f"Content: {test_case['content']}")
            print(f"Expected failure types: {test_case['expected_failure_types']}")
            print(f"Expected scores: {test_case['expected_scores']}")
            print("-" * 50)

            start_time = time.time()

            # 1. Get failure detection results
            failure_vector = self.failure_detector.detect_failures(test_case['content'])

            # 2. Create/update shadow concept
            embedding = self._create_embedding_from_content(test_case['content'])
            concept_name = f"concept_{test_case['name'].lower().replace(' ', '_')}"

            try:
                concept = self.shadow_system.create_concept(
                    name=concept_name,
                    initial_embedding=embedding,
                    concept_type=test_case['concept_type']
                )
                creation_status = "Created new concept"
            except ValueError:
                feedback = self._create_feedback_from_failure_vector(failure_vector)
                concept = self.shadow_system.refine_concept(
                    name=concept_name,
                    new_embedding=embedding,
                    feedback=feedback
                )
                creation_status = "Refined existing concept"

            # 3. Record benchmark metrics
            execution_time = time.time() - start_time
            self.benchmark_system.record_metric(
                name=f"test_{test_case['name']}_latency",
                value=execution_time,
                metric_type=BenchmarkMetricType.PERFORMANCE,
                metadata={"test_case": test_case['name']}
            )

            self.benchmark_system.record_metric(
                name=f"test_{test_case['name']}_accuracy",
                value=1.0 - failure_vector.factual_score,
                metric_type=BenchmarkMetricType.ACCURACY,
                metadata={"test_case": test_case['name']}
            )

            # 4. Verify results
            success = self._verify_integration_results(test_case, failure_vector, concept)

            # 5. Print results
            print(f"Status: {'PASS' if success else 'FAIL'}")
            print(f"Execution Time: {execution_time:.3f}s")
            print(f"Operation: {creation_status}")
            print(f"Failure Vector Scores:")
            print(f"  - Factual Score: {failure_vector.factual_score:.3f}")
            print(f"  - Logical Score: {failure_vector.logical_score:.3f}")
            print(f"  - Toxicity Score: {failure_vector.toxicity_score:.3f}")
            print(f"  - Hallucination Score: {failure_vector.hallucination_score:.3f}")
            print(f"  - Off-topic Score: {failure_vector.off_topic_score:.3f}")
            print(f"Shadow Concept:")
            print(f"  - Name: {concept.name}")
            print(f"  - Version: {concept.version}")
            print(f"  - Type: {concept.concept_type.value}")
            print(f"  - Confidence: {concept.confidence:.3f}")
            print(f"Benchmark Metrics:")
            print(f"  - Latency: {execution_time:.3f}s")
            print(f"  - Accuracy: {(1.0 - failure_vector.factual_score):.3f}")

            return success

        except Exception as e:
            self.logger.error(f"Error in integration test {test_case['name']}: {str(e)}")
            print(f"Error: {str(e)}")
            return False

    def _create_embedding_from_content(self, content: str, dim: int = 768) -> np.ndarray:
        """Create a simple embedding from content (placeholder implementation)"""
        # Convert hash to a valid seed value (between 0 and 2**32 - 1)
        hash_value = abs(hash(content)) % (2**32 - 1)
        np.random.seed(hash_value)
        embedding = np.random.randn(dim)
        return embedding / np.linalg.norm(embedding)

    def _create_feedback_from_failure_vector(self, failure_vector) -> Dict[str, float]:
        """Create feedback dictionary from failure vector"""
        return {
            "accuracy": 1.0 - failure_vector.factual_score,
            "consistency": 1.0 - failure_vector.logical_score,
            "relevance": 1.0 - failure_vector.off_topic_score
        }

    def _verify_integration_results(self, test_case: Dict, failure_vector, concept) -> bool:
        """Verify that integration results match expected outcomes"""
        success = True
        failures = []
        
        # Check failure types
        for expected_failure in test_case['expected_failure_types']:
            if expected_failure == 'factual_inaccuracy':
                if failure_vector.factual_score > 0.7:
                    failures.append(f"Expected factual inaccuracy but got score {failure_vector.factual_score}")
                    success = False
            elif expected_failure == 'logical_inconsistency':
                if failure_vector.logical_score > 0.7:
                    failures.append(f"Expected logical inconsistency but got score {failure_vector.logical_score}")
                    success = False
            elif expected_failure == 'hallucination':
                if failure_vector.hallucination_score < 0.7:
                    failures.append(f"Expected hallucination but got score {failure_vector.hallucination_score}")
                    success = False

        # Verify concept properties
        if concept.concept_type != test_case['concept_type']:
            failures.append(f"Expected concept type {test_case['concept_type']} but got {concept.concept_type}")
            success = False
        if concept.confidence < 0 or concept.confidence > 1:
            failures.append(f"Confidence score {concept.confidence} out of valid range [0,1]")
            success = False

        # Check expected scores if provided
        if 'expected_scores' in test_case:
            expected = test_case['expected_scores']
            tolerance = 0.2  # Allow some deviation in scores
            
            for score_type, expected_value in expected.items():
                actual_value = getattr(failure_vector, f"{score_type}_score")
                if abs(actual_value - expected_value) > tolerance:
                    failures.append(
                        f"{score_type.capitalize()} score {actual_value:.3f} differs from "
                        f"expected {expected_value:.3f} (tolerance: {tolerance})"
                    )
                    success = False

        # Check benchmark expectations if provided
        if 'benchmark_expectations' in test_case:
            for metric_name, expected_value in test_case['benchmark_expectations'].items():
                # Get actual value from benchmark system
                metric_stats = self.benchmark_system.get_metric_statistics(
                    f"test_{test_case['name']}_{metric_name}"
                )
                if metric_stats:
                    actual_value = metric_stats['mean']
                    if abs(actual_value - expected_value) > tolerance:
                        failures.append(
                            f"Benchmark {metric_name} {actual_value:.3f} differs from "
                            f"expected {expected_value:.3f} (tolerance: {tolerance})"
                        )
                        success = False

        # Print detailed failure information if any
        if not success:
            print("\nTest Failures:")
            for failure in failures:
                print(f"  - {failure}")
            print()

        return success

def main():
    """Main entry point"""
    print("\nStarting Complete System Integration Tests...")
    print("=" * 70)
    
    try:
        tester = IntegratedSystemTester()
        tester.run_integrated_tests()
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise
    
    print("\nTest execution completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
