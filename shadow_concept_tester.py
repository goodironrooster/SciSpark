import unittest
import numpy as np
import time
import logging
from typing import Dict, List, Optional
import threading
from dataclasses import dataclass
import traceback
from shadow_concept import ShadowConceptSystem, ConceptType, ShadowConcept

@dataclass
class TestResult:
    name: str
    success: bool
    error: Optional[str]
    execution_time: float
    details: Dict
    stack_trace: Optional[str] = None

class ShadowConceptTester:
    def __init__(self):
        self.system = ShadowConceptSystem(embedding_dim=128)  # Smaller dimension for testing
        self.test_results: List[TestResult] = []
        self.debug_mode = True
        self.embedding_dim = 128
        
        # Configure logging
        self.logger = logging.getLogger('ShadowConceptTester')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_test_embedding(self) -> np.ndarray:
        """Create a normalized test embedding"""
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)

    def run_test_with_monitoring(self, test_name: str, test_func) -> TestResult:
        """Run a test with comprehensive monitoring"""
        start_time = time.perf_counter()
        
        try:
            self.logger.info(f"Starting test: {test_name}")
            result = test_func()
            
            execution_time = time.perf_counter() - start_time
            
            details = {
                'execution_time': execution_time,
                'success_rate': 1.0 if result else 0.0
            }
            
            if not result:
                raise Exception(f"Test {test_name} failed")
                
            self.logger.info(f"Test {test_name} completed successfully in {execution_time:.4f}s")
            return TestResult(test_name, True, None, execution_time, details)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            stack_trace = traceback.format_exc()
            error_msg = str(e)
            
            self.logger.error(f"Error in {test_name}: {error_msg}")
            
            details = {
                'execution_time': execution_time,
                'success_rate': 0.0,
                'error_type': type(e).__name__,
                'error_message': error_msg
            }
            
            return TestResult(test_name, False, error_msg, execution_time, details, stack_trace)

    def test_concept_creation(self) -> bool:
        """Test creation of shadow concepts"""
        try:
            # Test basic concept creation
            embedding = self._create_test_embedding()
            concept = self.system.create_concept(
                name="test_concept",
                initial_embedding=embedding,
                concept_type=ConceptType.FACTUAL
            )
            
            if not isinstance(concept, ShadowConcept):
                self.logger.error("Created concept is not an instance of ShadowConcept")
                return False
                
            # Test duplicate creation
            try:
                self.system.create_concept(
                    name="test_concept",
                    initial_embedding=embedding,
                    concept_type=ConceptType.FACTUAL
                )
                self.logger.error("Failed to catch duplicate concept creation")
                return False
            except ValueError:
                pass  # Expected behavior
                
            # Test invalid embedding
            try:
                invalid_embedding = np.random.randn(64)  # Wrong dimension
                self.system.create_concept(
                    name="invalid_concept",
                    initial_embedding=invalid_embedding,
                    concept_type=ConceptType.FACTUAL
                )
                self.logger.error("Failed to catch invalid embedding dimensions")
                return False
            except ValueError:
                pass  # Expected behavior
            
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in concept creation test: {str(e)}")
            return False

    def test_concept_refinement(self) -> bool:
        """Test refinement of shadow concepts"""
        try:
            # Create initial concept
            initial_embedding = self._create_test_embedding()
            concept = self.system.create_concept(
                name="refinement_test",
                initial_embedding=initial_embedding,
                concept_type=ConceptType.FACTUAL
            )
            
            # Test refinement with valid feedback
            new_embedding = self._create_test_embedding()
            feedback = {
                "accuracy": 0.8,
                "consistency": 0.7,
                "relevance": 0.9
            }
            
            refined = self.system.refine_concept(
                name="refinement_test",
                new_embedding=new_embedding,
                feedback=feedback
            )
            
            if not isinstance(refined, ShadowConcept):
                self.logger.error("Refined concept is not an instance of ShadowConcept")
                return False
                
            if refined.version <= concept.version:
                self.logger.error("Refined concept version not incremented")
                return False
                
            # Test refinement with invalid concept name
            try:
                self.system.refine_concept(
                    name="nonexistent_concept",
                    new_embedding=new_embedding,
                    feedback=feedback
                )
                self.logger.error("Failed to catch refinement of nonexistent concept")
                return False
            except ValueError:
                pass  # Expected behavior
            
            # Test confidence bounds
            extreme_feedback = {
                "accuracy": 1.0,
                "consistency": 1.0,
                "relevance": 1.0
            }
            
            refined = self.system.refine_concept(
                name="refinement_test",
                new_embedding=new_embedding,
                feedback=extreme_feedback
            )
            
            if refined.confidence > self.system.max_confidence_threshold:
                self.logger.error("Confidence exceeded maximum threshold")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in concept refinement test: {str(e)}")
            return False

    def test_concept_history(self) -> bool:
        """Test concept history tracking"""
        try:
            # Create and refine a concept multiple times
            name = "history_test"
            initial_embedding = self._create_test_embedding()
            
            concept = self.system.create_concept(
                name=name,
                initial_embedding=initial_embedding,
                concept_type=ConceptType.FACTUAL
            )
            
            refinements = 5
            for _ in range(refinements):
                self.system.refine_concept(
                    name=name,
                    new_embedding=self._create_test_embedding(),
                    feedback={
                        "accuracy": 0.8,
                        "consistency": 0.7,
                        "relevance": 0.9
                    }
                )
            
            # Get history
            history = self.system.get_concept_history(name)
            
            if len(history) != refinements + 1:  # Initial version + refinements
                self.logger.error(f"Expected {refinements + 1} versions in history, got {len(history)}")
                return False
            
            # Check version progression
            versions = [entry["version"] for entry in history]
            if not all(versions[i] < versions[i+1] for i in range(len(versions)-1)):
                self.logger.error("Version numbers not strictly increasing")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in concept history test: {str(e)}")
            return False

    def test_concept_drift(self) -> bool:
        """Test concept drift analysis"""
        try:
            # Create and refine a concept with controlled drift
            name = "drift_test"
            initial_embedding = self._create_test_embedding()
            
            self.system.create_concept(
                name=name,
                initial_embedding=initial_embedding,
                concept_type=ConceptType.FACTUAL
            )
            
            # Create increasingly different embeddings
            base_embedding = initial_embedding
            for i in range(5):
                noise = self._create_test_embedding() * (i * 0.1)  # Increasing noise
                new_embedding = base_embedding + noise
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
                
                self.system.refine_concept(
                    name=name,
                    new_embedding=new_embedding,
                    feedback={
                        "accuracy": 0.8 - (i * 0.1),
                        "consistency": 0.7 - (i * 0.1),
                        "relevance": 0.9 - (i * 0.1)
                    }
                )
            
            # Analyze drift
            drift_analysis = self.system.analyze_concept_drift(name)
            
            if 'drift' not in drift_analysis or 'confidence_change' not in drift_analysis:
                self.logger.error("Missing drift analysis metrics")
                return False
            
            if drift_analysis['drift'] < 0 or drift_analysis['drift'] > 1:
                self.logger.error("Drift value outside valid range [0,1]")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in concept drift test: {str(e)}")
            return False

    def test_thread_safety(self) -> bool:
        """Test thread safety of the system"""
        try:
            # Create a concept to be refined by multiple threads
            name = "thread_test"
            initial_embedding = self._create_test_embedding()
            
            self.system.create_concept(
                name=name,
                initial_embedding=initial_embedding,
                concept_type=ConceptType.FACTUAL
            )
            
            def refine_concept():
                try:
                    for _ in range(5):
                        self.system.refine_concept(
                            name=name,
                            new_embedding=self._create_test_embedding(),
                            feedback={
                                "accuracy": 0.8,
                                "consistency": 0.7,
                                "relevance": 0.9
                            }
                        )
                except Exception as e:
                    self.logger.error(f"Thread error: {str(e)}")
                    return False

            # Create and run multiple threads
            threads = [threading.Thread(target=refine_concept) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Verify history integrity
            history = self.system.get_concept_history(name)
            versions = [entry["version"] for entry in history]
            
            if not all(versions[i] < versions[i+1] for i in range(len(versions)-1)):
                self.logger.error("Version numbers not strictly increasing under concurrent refinement")
                return False
            
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error in thread safety test: {str(e)}")
            return False

    def run_all_tests(self) -> None:
        """Run all shadow concept system tests"""
        tests = [
            ("Concept Creation", self.test_concept_creation),
            ("Concept Refinement", self.test_concept_refinement),
            ("Concept History", self.test_concept_history),
            ("Concept Drift", self.test_concept_drift),
            ("Thread Safety", self.test_thread_safety)
        ]

        for test_name, test_func in tests:
            result = self.run_test_with_monitoring(test_name, test_func)
            self.test_results.append(result)

    def print_results(self) -> None:
        """Print detailed test results"""
        print("\nShadow Concept System Test Results:")
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
        print(f"Tests Summary: {passed_tests}/{total_tests} passed")

if __name__ == "__main__":
    tester = ShadowConceptTester()
    tester.run_all_tests()
    tester.print_results()