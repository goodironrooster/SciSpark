from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field
import numpy as np
from internal_voice import InternalVoiceSystem, InternalDialogue
from benchmark_system import BenchmarkMetricType  # Add this import

@dataclass
class PerspectiveAnalysis:
    perspective_type: str
    analysis: str
    confidence: float
    key_points: List[str]
    suggestions: List[str]

@dataclass
class EnhancedDialogue:
    iteration: int
    original_response: str
    internal_reaction: str
    improved_response: str
    confidence: float
    shadows_detected: int
    concepts_mapped: int
    perspective_analyses: List[PerspectiveAnalysis]
    synthesis: str
    improvement_priority: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

class AdvancedInternalVoiceSystem(InternalVoiceSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thought_history = []
        self.concept_evolution = {}

    def process_thought(self, prompt: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Process thought through internal dialogue with multiple perspectives"""
        start_time = time.time()
        
        try:
            # Initial response
            initial_response = self.llm.generate(prompt)
            current_response = initial_response.content
            
            dialogue_iterations = []
            
            for iteration in range(max_iterations):
                # Generate internal reaction with multiple perspectives
                reaction_result = self._generate_internal_reaction(current_response)
                
                # Analyze current state
                analysis = self._analyze_state(current_response)
                
                # Generate improvement
                improved_response = self._generate_improvement(
                    current_response,
                    reaction_result,
                    analysis
                )
                
                # Create enhanced dialogue entry
                dialogue = EnhancedDialogue(
                    iteration=iteration + 1,
                    original_response=current_response,
                    internal_reaction=reaction_result['synthesis'],
                    improved_response=improved_response,
                    confidence=analysis['confidence'],
                    shadows_detected=analysis['shadow_count'],
                    concepts_mapped=analysis['concept_count'],
                    perspective_analyses=list(reaction_result['analyses'].values()),
                    synthesis=reaction_result['synthesis'],
                    improvement_priority=reaction_result['improvement_priority']
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
            
            # Record benchmark metrics with correct enum type
            self.benchmark_system.record_metric(
                name="internal_voice_processing",
                value=processing_time,
                metric_type=BenchmarkMetricType.PERFORMANCE,  # Use enum type
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

        
    def _generate_internal_reaction(self, response: str) -> Dict[str, Any]:
        """Generate reactions from multiple perspectives"""
        perspectives = {
            "logical_analysis": {
                "prompt": (
                    "Analyze the logical structure, coherence, and validity of this response:\n"
                    "1. Evaluate the argument structure\n"
                    "2. Identify any logical fallacies\n"
                    "3. Check factual accuracy\n"
                    "4. Assess the flow of ideas\n\n"
                    f"Response: {response}\n\n"
                    "Logical Analysis:"
                ),
                "weight": 0.3
            },
            "creative_expansion": {
                "prompt": (
                    "Explore creative possibilities and novel connections in this response:\n"
                    "1. Identify unique metaphors or analogies\n"
                    "2. Suggest creative expansions\n"
                    "3. Find unexpected connections\n"
                    "4. Propose innovative angles\n\n"
                    f"Response: {response}\n\n"
                    "Creative Analysis:"
                ),
                "weight": 0.25
            },
            "critical_evaluation": {
                "prompt": (
                    "Critically evaluate this response for improvements:\n"
                    "1. Identify gaps in explanation\n"
                    "2. Find potential misunderstandings\n"
                    "3. Suggest clarifications\n"
                    "4. Assess completeness\n\n"
                    f"Response: {response}\n\n"
                    "Critical Analysis:"
                ),
                "weight": 0.25
            },
            "practical_application": {
                "prompt": (
                    "Consider practical implications and applications:\n"
                    "1. Evaluate real-world relevance\n"
                    "2. Identify practical examples\n"
                    "3. Suggest applications\n"
                    "4. Consider audience understanding\n\n"
                    f"Response: {response}\n\n"
                    "Practical Analysis:"
                ),
                "weight": 0.2
            }
        }

        # Generate analyses from each perspective
        analyses = {}
        for perspective, config in perspectives.items():
            reaction = self.llm.generate(
                config["prompt"],
                max_tokens=256,
                temperature=0.7
            )
            
            # Extract key points and suggestions using the LLM
            key_points_prompt = f"Extract key points from this analysis:\n{reaction.content}"
            key_points = self.llm.generate(key_points_prompt, max_tokens=128).content.split('\n')
            
            suggestions_prompt = f"Suggest specific improvements based on this analysis:\n{reaction.content}"
            suggestions = self.llm.generate(suggestions_prompt, max_tokens=128).content.split('\n')
            
            analyses[perspective] = PerspectiveAnalysis(
                perspective_type=perspective,
                analysis=reaction.content,
                confidence=self._calculate_perspective_confidence(reaction.content),
                key_points=key_points,
                suggestions=suggestions
            )

        # Synthesize analyses
        synthesis = self._synthesize_analyses(analyses, perspectives)
        
        return {
            'analyses': analyses,
            'synthesis': synthesis,
            'improvement_priority': self._calculate_improvement_priorities(analyses)
        }

    def _calculate_perspective_confidence(self, analysis: str) -> float:
        """Calculate confidence score for a perspective analysis"""
        # Use shadow detection to analyze the confidence
        shadows = self.shadow_detector.detect_shadows(analysis.encode())
        concepts = []
        for shadow in shadows:
            mapped = self.concept_mapper.map_shadow_to_concept(shadow)
            concepts.extend(mapped)
            
        # Calculate confidence based on shadows and concepts
        if not shadows or not concepts:
            return 0.5
            
        shadow_confidence = np.mean([s.confidence for s in shadows])
        concept_confidence = np.mean([c.confidence for c in concepts])
        
        return np.mean([shadow_confidence, concept_confidence])

    def _synthesize_analyses(
        self,
        analyses: Dict[str, PerspectiveAnalysis],
        perspectives: Dict[str, Dict]
    ) -> str:
        """Synthesize multiple perspective analyses into a coherent response"""
        # Create synthesis prompt
        synthesis_prompt = "Synthesize these analytical perspectives into a coherent evaluation:\n\n"
        
        for perspective, analysis in analyses.items():
            synthesis_prompt += f"{perspective.replace('_', ' ').title()}:\n"
            synthesis_prompt += f"Confidence: {analysis.confidence:.2f}\n"
            synthesis_prompt += f"Key Points:\n"
            for point in analysis.key_points[:3]:  # Top 3 key points
                synthesis_prompt += f"- {point}\n"
            synthesis_prompt += "\n"
        
        synthesis_prompt += "\nProvide a synthesized evaluation that integrates these perspectives:"
        
        # Generate synthesis
        synthesis = self.llm.generate(
            synthesis_prompt,
            max_tokens=384,
            temperature=0.7
        )
        
        return synthesis.content

    def _calculate_improvement_priorities(
        self,
        analyses: Dict[str, PerspectiveAnalysis]
    ) -> Dict[str, float]:
        """Calculate improvement priorities based on analyses"""
        priorities = {}
        
        for perspective, analysis in analyses.items():
            # Calculate priority based on confidence and number of suggestions
            base_priority = 1 - analysis.confidence
            suggestion_weight = len(analysis.suggestions) / 5  # Normalize by max expected suggestions
            
            priorities[perspective] = (base_priority + suggestion_weight) / 2
            
        # Normalize priorities
        total = sum(priorities.values())
        if total > 0:
            priorities = {k: v/total for k, v in priorities.items()}
            
        return priorities

    def _generate_improvement(
        self,
        original_response: str,
        reaction: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        improvement_prompt = (
            "Create a complete, creative explanation of why the sky is blue, following these guidelines:\n\n"
            "Required Structure:\n"
            "1. Opening Hook: Start with an engaging metaphor or question\n"
            "2. Main Explanation: Use creative analogies while maintaining scientific accuracy\n"
            "3. Visual Details: Include vivid descriptions of the sky's appearance\n"
            "4. Interactive Element: Add a relatable observation or experience\n"
            "5. Conclusion: End with a memorable takeaway\n\n"
            "Previous Response:\n"
            f"{original_response}\n\n"
            "Analysis Synthesis:\n"
            f"{reaction['synthesis']}\n\n"
            "Key Requirements:\n"
            "- Balance creativity with scientific accuracy\n"
            "- Use clear, engaging language\n"
            "- Include 2-3 memorable metaphors\n"
            "- Keep explanation complete and self-contained\n\n"
            "Complete explanation (minimum 300 words):"
        )
        
        improved = self.llm.generate(
            prompt=improvement_prompt,
            max_tokens=800,  # Increased token limit
            temperature=0.85,
            top_p=0.92,
            stop=["</s>", "<|im_end|>"]  # Explicit stop tokens
        )
        
        return improved.content

    def _synthesize_analyses(
        self,
        analyses: Dict[str, PerspectiveAnalysis],
        perspectives: Dict[str, Dict]
    ) -> str:
        """Synthesize multiple perspective analyses into a coherent evaluation"""
        synthesis_prompt = (
            "Create a cohesive synthesis of these analytical perspectives for explaining why the sky is blue:\n\n"
        )
        
        for perspective, analysis in analyses.items():
            synthesis_prompt += f"{perspective.replace('_', ' ').title()}:\n"
            synthesis_prompt += f"Confidence: {analysis.confidence:.2f}\n"
            synthesis_prompt += "Key Points:\n"
            for point in analysis.key_points[:3]:
                synthesis_prompt += f"- {point}\n"
            synthesis_prompt += "\n"
        
        synthesis_prompt += (
            "\nProvide a synthesized evaluation that:\n"
            "1. Integrates insights from all perspectives\n"
            "2. Identifies key strengths and opportunities\n"
            "3. Suggests specific improvements\n"
            "4. Maintains balance between creativity and accuracy\n\n"
            "Synthesis:"
        )
        
        synthesis = self.llm.generate(
            prompt=synthesis_prompt,
            max_tokens=384,
            temperature=0.7
        )
        
        return synthesis.content

