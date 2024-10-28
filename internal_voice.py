from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass
import numpy as np
import threading
from llm_interface import LLMInterface
from shadow_detection import ShadowDetectionSystem
from concept_mapper import ConceptMapper
from failure_detection import FailureDetector, FailureVector
from benchmark_system import BenchmarkSystem, BenchmarkMetricType
from neural_shadow_optimizer import CreativeOptimizer
import logging

@dataclass
class InternalDialogue:
    iteration: int
    original_response: str
    internal_reaction: str
    improved_response: str
    confidence: float
    shadows_detected: int
    concepts_mapped: int
    timestamp: float = time.time()

class InternalVoiceSystem:
    def __init__(
        self,
        llm: LLMInterface,
        shadow_detector: ShadowDetectionSystem,
        concept_mapper: ConceptMapper,
        failure_detector: FailureDetector,
        benchmark_system: BenchmarkSystem,
        debug_mode: bool = True
    ):
        self.llm = llm
        self.shadow_detector = shadow_detector
        self.concept_mapper = concept_mapper
        self.failure_detector = failure_detector
        self.benchmark_system = benchmark_system
        self.debug_mode = debug_mode
        self.logger = logging.getLogger("InternalVoice")

    def process_thought(self, prompt: str, max_iterations: int = 3) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Initial response
            initial_response = self.llm.generate(prompt)
            current_response = initial_response.content
            
            dialogue_iterations = []
            
            for iteration in range(max_iterations):
                # Generate internal reaction
                reaction = self._generate_internal_reaction(current_response)
                
                # Analyze current state
                analysis = self._analyze_state(current_response)
                
                # Generate improvement
                improved_response = self._generate_improvement(
                    current_response,
                    reaction,
                    analysis
                )
                
                # Create dialogue entry
                dialogue = InternalDialogue(
                    iteration=iteration + 1,
                    original_response=current_response,
                    internal_reaction=reaction,
                    improved_response=improved_response,
                    confidence=analysis['confidence'],
                    shadows_detected=analysis['shadow_count'],
                    concepts_mapped=analysis['concept_count']
                )
                
                dialogue_iterations.append(dialogue)
                
                if self._is_significant_improvement(
                    current_response,
                    improved_response,
                    analysis
                ):
                    current_response = improved_response
                else:
                    break
                    
            processing_time = time.time() - start_time
            
            # Record benchmark metrics silently
            self.benchmark_system.record_metric(
                name="internal_voice_processing",
                value=processing_time,
                metric_type=BenchmarkMetricType.PERFORMANCE,
                metadata={
                    'iterations': len(dialogue_iterations),
                    'final_confidence': dialogue_iterations[-1].confidence
                }
            )
            
            return {
                'final_response': current_response,
                'dialogue_iterations': dialogue_iterations,
                'metrics': {
                    'iterations': len(dialogue_iterations),
                    'processing_time': processing_time,
                    'final_confidence': dialogue_iterations[-1].confidence
                }
            }
            
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"Error in thought processing: {str(e)}")
            raise

    def _generate_internal_reaction(self, response: str) -> str:
        reaction_prompt = (
            "As an internal voice analyzing this response, provide a detailed "
            "critique considering:\n"
            "1. Logical coherence and accuracy\n"
            "2. Clarity and understandability\n"
            "3. Creative potential and uniqueness\n"
            "4. Potential improvements\n\n"
            f"Response to analyze: {response}\n\n"
            "Internal voice critique:"
        )
        
        reaction = self.llm.generate(
            prompt=reaction_prompt,
            max_tokens=256,
            temperature=0.7
        )
        
        return reaction.content

    def _analyze_state(self, response: str) -> Dict[str, Any]:
        # Detect shadows
        shadows = self.shadow_detector.detect_shadows(response.encode())
        
        # Map concepts
        concepts = []
        for shadow in shadows:
            mapped = self.concept_mapper.map_shadow_to_concept(shadow)
            concepts.extend(mapped)
            
        # Detect failures
        failures = self.failure_detector.detect_failures(response)
        
        # Calculate confidence
        confidence = self._calculate_confidence(shadows, concepts, failures)
        
        return {
            'shadow_count': len(shadows),
            'concept_count': len(concepts),
            'confidence': confidence,
            'shadows': shadows,
            'concepts': concepts,
            'failures': failures
        }

    def _generate_improvement(
        self,
        original_response: str,
        reaction: str,
        analysis: Dict[str, Any]
    ) -> str:
        improvement_prompt = (
            "Improve this response based on the internal critique and analysis:\n\n"
            f"Original response: {original_response}\n\n"
            f"Internal critique: {reaction}\n\n"
            "Requirements:\n"
            "1. Address all concerns raised in the critique\n"
            "2. Maintain or improve logical coherence\n"
            "3. Enhance clarity while preserving complexity\n"
            "4. Explore creative opportunities identified\n\n"
            "Improved response:"
        )
        
        improved = self.llm.generate(
            prompt=improvement_prompt,
            max_tokens=512,
            temperature=0.8
        )
        
        return improved.content

    def _calculate_confidence(
        self,
        shadows: List[Any],
        concepts: List[Any],
        failures: FailureVector
    ) -> float:
        if not shadows or not concepts:
            return 0.0

        # Calculate confidence scores for each component
        shadow_confidence = sum(s.confidence for s in shadows) / len(shadows) if shadows else 0.0
        concept_confidence = sum(c.confidence for c in concepts) / len(concepts) if concepts else 0.0
        
        # Calculate failure impact using FailureVector scores
        failure_impact = (
            (1 - failures.factual_score) * 0.3 +    # Higher factual score is better
            (1 - failures.logical_score) * 0.3 +     # Higher logical score is better
            failures.toxicity_score * 0.1 +          # Lower toxicity is better
            failures.hallucination_score * 0.1 +     # Lower hallucination is better
            failures.off_topic_score * 0.1 +         # Lower off-topic is better
            (1 - failures.novelty_score) * 0.1       # Higher novelty is better
        )

        # Weights for different components
        weights = (0.3, 0.4, 0.3)  # shadow, concept, failure weights

        # Weighted average with adjusted failure impact
        confidence = (
            weights[0] * shadow_confidence +
            weights[1] * concept_confidence +
            weights[2] * (1 - failure_impact)
        )
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def _is_significant_improvement(
        self,
        original: str,
        improved: str,
        analysis: Dict[str, Any]
    ) -> bool:
        original_analysis = self._analyze_state(original)
        improved_analysis = self._analyze_state(improved)
        
        return (improved_analysis['confidence'] - 
                original_analysis['confidence']) > 0.1

    def debug_print(self, message: str, level: str = "INFO") -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG InternalVoiceSystem [{level}]: {message}")
