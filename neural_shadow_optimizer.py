# neural_shadow_optimizer.py
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from buffer_manager import BufferManager
from pattern_analyzer import PatternAnalyzer
from shadow_detection import ShadowDetectionSystem, ShadowPattern
from concept_mapper import ConceptMapper

@dataclass
class Hypothesis:
    shadow_pattern: ShadowPattern
    novelty_score: float
    implications: List[str]
    confidence: float = 0.0
    
    def __str__(self):
        return f"Hypothesis(novelty={self.novelty_score:.2f}, confidence={self.confidence:.2f})"

@dataclass
class CreativeIdea:
    base_hypothesis: Hypothesis
    novelty: float
    potential_impact: float
    description: str = ""
    
    def __str__(self):
        return f"CreativeIdea(novelty={self.novelty:.2f}, impact={self.potential_impact:.2f})"

@dataclass
class ValidationResult:
    is_novel: bool
    connected_concepts: List[str]
    confidence: float
    
    def __str__(self):
        return f"ValidationResult(novel={self.is_novel}, confidence={self.confidence:.2f})"

class HypothesisGenerator:
    def __init__(self):
        self.min_confidence = 0.8
        self.novelty_weight = 0.7
        self.coherence_weight = 0.3

    def generate(self, shadows: List[ShadowPattern]) -> List[Hypothesis]:
        hypotheses = []
        for shadow in shadows:
            if shadow.confidence > self.min_confidence:
                hypothesis = self._create_hypothesis(shadow)
                hypotheses.append(hypothesis)
        return hypotheses

    def _create_hypothesis(self, shadow: ShadowPattern) -> Hypothesis:
        novelty_score = self._calculate_novelty(shadow)
        implications = self._derive_implications(shadow)
        confidence = shadow.confidence * self.novelty_weight + self._calculate_coherence(shadow) * self.coherence_weight
        
        return Hypothesis(
            shadow_pattern=shadow,
            novelty_score=novelty_score,
            implications=implications,
            confidence=confidence
        )

    def _calculate_novelty(self, shadow: ShadowPattern) -> float:
        # Calculate novelty based on shadow pattern characteristics
        pattern_novelty = len(shadow.metadata.get('unique_patterns', [])) / 100
        return min(pattern_novelty + shadow.confidence * 0.3, 1.0)

    def _calculate_coherence(self, shadow: ShadowPattern) -> float:
        # Calculate coherence based on pattern structure
        return shadow.confidence * 0.8 + 0.2  # Base coherence on confidence with a minimum threshold

    def _derive_implications(self, shadow: ShadowPattern) -> List[str]:
        implications = []
        if 'patterns' in shadow.metadata:
            for pattern in shadow.metadata['patterns']:
                implications.append(f"Pattern implication: {pattern}")
        return implications

class IdeaEvaluator:
    def __init__(self):
        self.creativity_metrics = {
            'novelty': self._evaluate_novelty,
            'usefulness': self._evaluate_usefulness,
            'coherence': self._evaluate_coherence
        }
        self.impact_threshold = 0.6

    def explore_implications(self, hypothesis: Hypothesis) -> List[CreativeIdea]:
        ideas = []
        base_impact = self._evaluate_impact(hypothesis)
        
        if base_impact > self.impact_threshold:
            for i in range(3):  # Generate three creative variations
                novelty = self._evaluate_novelty(hypothesis) * (1.0 + i * 0.1)  # Increase novelty for each variation
                impact = base_impact * (1.0 - i * 0.05)  # Slightly decrease impact for more novel ideas
                
                idea = CreativeIdea(
                    base_hypothesis=hypothesis,
                    novelty=novelty,
                    potential_impact=impact,
                    description=self._generate_description(hypothesis, i)
                )
                ideas.append(idea)
        
        return ideas

    def _evaluate_novelty(self, hypothesis: Hypothesis) -> float:
        return hypothesis.novelty_score * 0.8 + np.random.random() * 0.2

    def _evaluate_usefulness(self, hypothesis: Hypothesis) -> float:
        return hypothesis.confidence * 0.7 + 0.3

    def _evaluate_coherence(self, hypothesis: Hypothesis) -> float:
        return hypothesis.confidence * 0.9 + 0.1

    def _evaluate_impact(self, hypothesis: Hypothesis) -> float:
        return (hypothesis.novelty_score * 0.6 + hypothesis.confidence * 0.4)

    def _generate_description(self, hypothesis: Hypothesis, variation: int) -> str:
        base_desc = f"Creative direction {variation + 1} based on {hypothesis.shadow_pattern.pattern_type.value}"
        implications = "\n".join(hypothesis.implications[:2])  # Include top 2 implications
        return f"{base_desc}\nImplications:\n{implications}"

class DynamicKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        self.novelty_threshold = 0.7
        self.confidence_threshold = 0.6

    def validate_hypothesis(self, hypothesis: Hypothesis) -> ValidationResult:
        connected_concepts = self._find_connected_concepts(hypothesis)
        novelty_score = self._calculate_novelty_in_graph(hypothesis)
        confidence = self._calculate_confidence(hypothesis)
        
        if novelty_score > self.novelty_threshold and confidence > self.confidence_threshold:
            self._update_graph(hypothesis)
        
        return ValidationResult(
            is_novel=novelty_score > self.novelty_threshold,
            connected_concepts=connected_concepts,
            confidence=confidence
        )

    def _find_connected_concepts(self, hypothesis: Hypothesis) -> List[str]:
        connected = []
        pattern_type = hypothesis.shadow_pattern.pattern_type.value
        
        for node in self.graph.nodes():
            if pattern_type in node:
                connected.append(node)
        
        return connected[:5]  # Return top 5 connected concepts

    def _calculate_novelty_in_graph(self, hypothesis: Hypothesis) -> float:
        if not self.graph.nodes:
            return 1.0  # Completely novel if graph is empty
            
        similar_patterns = sum(1 for node in self.graph.nodes() 
                             if hypothesis.shadow_pattern.pattern_type.value in node)
        
        return 1.0 - (similar_patterns / max(len(self.graph.nodes), 1))

    def _calculate_confidence(self, hypothesis: Hypothesis) -> float:
        base_confidence = hypothesis.confidence
        graph_support = len(self._find_connected_concepts(hypothesis)) / 10  # Scale by maximum expected connections
        return (base_confidence * 0.7 + graph_support * 0.3)

    def _update_graph(self, hypothesis: Hypothesis):
        node_id = f"{hypothesis.shadow_pattern.pattern_type.value}_{len(self.graph.nodes)}"
        self.graph.add_node(node_id, 
                           confidence=hypothesis.confidence,
                           novelty=hypothesis.novelty_score)
        
        # Connect to similar concepts
        for concept in self._find_connected_concepts(hypothesis):
            self.graph.add_edge(node_id, concept, weight=hypothesis.confidence)

class ResearchMetrics:
    def __init__(self):
        self.metrics = {
            'novelty': [],
            'coherence': [],
            'impact': []
        }
        self.weights = {
            'novelty': 0.4,
            'coherence': 0.3,
            'impact': 0.3
        }

    def update_metrics(self, response: str, hypothesis: Hypothesis):
        self.metrics['novelty'].append(self._calculate_novelty(response))
        self.metrics['coherence'].append(self._calculate_coherence(response))
        self.metrics['impact'].append(self._calculate_potential_impact(hypothesis))

    def get_research_score(self) -> float:
        if not all(self.metrics.values()):
            return 0.0
            
        weighted_scores = [
            np.mean(self.metrics[metric]) * self.weights[metric]
            for metric in self.metrics
        ]
        return sum(weighted_scores)

    def _calculate_novelty(self, response: str) -> float:
        # Simple novelty calculation based on response length and unique words
        words = set(response.split())
        return min(len(words) / 100, 1.0)

    def _calculate_coherence(self, response: str) -> float:
        # Simple coherence check based on sentence structure
        sentences = response.split('.')
        return min(len(sentences) / 10, 1.0)

    def _calculate_potential_impact(self, hypothesis: Hypothesis) -> float:
        return hypothesis.novelty_score * 0.7 + hypothesis.confidence * 0.3

class CreativeOptimizer:
    def __init__(self, base_llm, initial_buffer_size: int = 8192):
        self.llm = base_llm
        self.buffer_manager = BufferManager(initial_size=initial_buffer_size)
        self.pattern_analyzer = PatternAnalyzer(self.buffer_manager)
        self.shadow_detector = ShadowDetectionSystem(self.pattern_analyzer)
        self.concept_mapper = ConceptMapper(debug_mode=True)
        
        # Initialize creative optimization components
        self.hypothesis_generator = HypothesisGenerator()
        self.idea_evaluator = IdeaEvaluator()
        self.knowledge_graph = DynamicKnowledgeGraph()
        self.research_metrics = ResearchMetrics()

    def generate_creative_output(self, prompt: str, research_mode: bool = True) -> Dict[str, Any]:
        # Generate initial response
        base_response = self.llm.generate(prompt)
        
        # Analyze patterns and shadows
        patterns = self.pattern_analyzer.analyze_sequence(base_response.content.encode())
        shadows = self.shadow_detector.detect_shadows(base_response.content.encode())
        
        # Generate and evaluate hypotheses
        hypotheses = self.hypothesis_generator.generate(shadows)
        
        # Refine output
        refined_response = self.refine_output(base_response.content, hypotheses)
        
        # Update research metrics
        if hypotheses:
            self.research_metrics.update_metrics(refined_response, hypotheses[0])
        
        return {
            'original_response': base_response.content,
            'refined_response': refined_response,
            'research_score': self.research_metrics.get_research_score(),
            'shadows_detected': len(shadows),
            'hypotheses_generated': len(hypotheses)
        }

    def refine_output(self, base_response: str, hypotheses: List[Hypothesis]) -> str:
        refinements = []
        
        for hypothesis in hypotheses:
            # Validate hypothesis
            validation = self.knowledge_graph.validate_hypothesis(hypothesis)
            
            if validation.is_novel:
                # Generate creative ideas based on hypothesis
                new_ideas = self.idea_evaluator.explore_implications(hypothesis)
                refinements.extend([idea.description for idea in new_ideas])
        
        # Combine original response with refinements
        if refinements:
            combined = f"{base_response}\n\nCreative Extensions:\n"
            combined += "\n".join(f"- {ref}" for ref in refinements[:3])  # Include top 3 refinements
            return combined
        
        return base_response

    def get_performance_metrics(self) -> Dict[str, float]:
        return {
            'research_score': self.research_metrics.get_research_score(),
            'knowledge_graph_size': len(self.knowledge_graph.graph.nodes),
            'average_novelty': np.mean(self.research_metrics.metrics['novelty']) if self.research_metrics.metrics['novelty'] else 0.0
        }