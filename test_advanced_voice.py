import logging
import sys
from advanced_internal_voice import AdvancedInternalVoiceSystem
from llm_interface import LLMInterface
from shadow_detection import ShadowDetectionSystem
from concept_mapper import ConceptMapper
from failure_detection import FailureDetector
from benchmark_system import BenchmarkSystem
import traceback
import os

# Suppress CUDA and other initialization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('llama_cpp').setLevel(logging.ERROR)

def configure_logging():
    # Suppress all loggers
    logging.getLogger().setLevel(logging.ERROR)
    
    # Configure specific loggers
    components = [
        'llm_interface',
        'ConceptMapper',
        'ShadowDetector',
        'PatternAnalyzer',
        'FailureDetector',
        'BenchmarkSystem',
        'InternalVoice'
    ]
    
    for component in components:
        logging.getLogger(component).setLevel(logging.ERROR)

def format_perspective_analysis(perspective):
    """Format a single perspective analysis for display"""
    formatted = f"\n{perspective.perspective_type.replace('_', ' ').title()}:"
    formatted += f"\nConfidence: {perspective.confidence:.2f}"
    
    if perspective.key_points:
        formatted += "\nKey Insights:"
        for point in perspective.key_points:
            if point and point.strip():
                point = point.strip()
                # Cleanup and format the point
                point = point.replace('\n', ' ').strip()
                if len(point) > 100:
                    point = point[:97] + "..."
                formatted += f"\n• {point}"
    
    if perspective.suggestions:
        formatted += "\nSuggestions:"
        for suggestion in perspective.suggestions:
            if suggestion and suggestion.strip():
                suggestion = suggestion.strip()
                # Cleanup and format the suggestion
                suggestion = suggestion.replace('\n', ' ').strip()
                if len(suggestion) > 100:
                    suggestion = suggestion[:97] + "..."
                formatted += f"\n→ {suggestion}"
    
    return formatted

def print_analysis_results(result):
    print("\nAnalysis Results:")
    print("=" * 60)
    
    for iteration in result['dialogue_iterations']:
        print(f"\nIteration {iteration.iteration}")
        print("-" * 40)
        
        print("\nPerspective Analyses:")
        for perspective in iteration.perspective_analyses:
            print(format_perspective_analysis(perspective))
        
        print("\nSynthesized Insights:")
        print("-" * 30)
        synthesis = iteration.synthesis
        # Format synthesis for better readability
        synthesis = synthesis.replace('\n\n', '\n').strip()
        print(synthesis[:500] + "..." if len(synthesis) > 500 else synthesis)
        
        print("\nImprovement Priorities:")
        print("-" * 30)
        sorted_priorities = sorted(
            iteration.improvement_priority.items(),
            key=lambda x: x[1],
            reverse=True
        )
        max_priority = max(p[1] for p in sorted_priorities)
        for perspective, priority in sorted_priorities:
            bars = "█" * int((priority / max_priority) * 20)
            print(f"{perspective.replace('_', ' ').title():20} [{bars:<20}] {priority:.2f}")
        
        print("\nImproved Response:")
        print("-" * 30)
        improved = iteration.improved_response.strip()
        # Add paragraph breaks for readability
        improved = "\n".join(p.strip() for p in improved.split('\n') if p.strip())
        print(improved)
        print(f"\nConfidence: {iteration.confidence:.2f}")

    print("\nFinal Results:")
    print("=" * 60)
    print("\nFinal Response:")
    final_response = result['final_response'].strip()
    final_response = "\n".join(p.strip() for p in final_response.split('\n') if p.strip())
    print(final_response)
    
    print("\nPerformance Metrics:")
    print("-" * 30)
    print(f"Total iterations: {result['metrics']['iterations']}")
    print(f"Processing time: {result['metrics']['processing_time']:.1f} seconds")
    print(f"Final confidence: {result['metrics']['final_confidence']:.2f}")

def save_results_to_file(result, filename):
    """Optional: Save results to a file for later analysis"""
    import json
    from datetime import datetime
    
    # Convert result to serializable format
    serializable_result = {
        'timestamp': datetime.now().isoformat(),
        'final_response': result['final_response'],
        'metrics': result['metrics'],
        'iterations': [
            {
                'iteration': d.iteration,
                'confidence': d.confidence,
                'shadows_detected': d.shadows_detected,
                'concepts_mapped': d.concepts_mapped,
                'perspective_analyses': [
                    {
                        'type': p.perspective_type,
                        'confidence': p.confidence,
                        'key_points': p.key_points,
                        'suggestions': p.suggestions
                    }
                    for p in d.perspective_analyses
                ],
                'synthesis': d.synthesis,
                'improvement_priority': d.improvement_priority
            }
            for d in result['dialogue_iterations']
        ]
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)

def test_advanced_voice():
    configure_logging()
    
    try:
        print("\nInitializing Advanced Internal Voice System...")
        
        # Initialize components
        llm = LLMInterface(
            model_path="D:/LLM models/phi-2-orange.Q4_0.gguf",
            n_ctx=2048,
            n_gpu_layers=32,
            quiet=True
        )
        
        shadow_detector = ShadowDetectionSystem(debug_mode=False)
        concept_mapper = ConceptMapper(debug_mode=False)
        failure_detector = FailureDetector(debug_mode=False)
        benchmark_system = BenchmarkSystem()
        
        # Create advanced system
        advanced_voice = AdvancedInternalVoiceSystem(
            llm=llm,
            shadow_detector=shadow_detector,
            concept_mapper=concept_mapper,
            failure_detector=failure_detector,
            benchmark_system=benchmark_system,
            debug_mode=False
        )
        
        print("System initialized successfully")
        
        # Test prompts
        test_prompts = [
            {
                "prompt": "Explain why the sky is blue in a creative way",
                "description": "Creative Science Explanation"
            },
            # Add more test prompts here as needed
        ]
        
        # Process each prompt
        for test_case in test_prompts:
            print(f"\nProcessing: {test_case['description']}")
            print(f"Prompt: {test_case['prompt']}")
            print("-" * 50)
            
            print("Generating initial response...")
            result = advanced_voice.process_thought(
                prompt=test_case['prompt'],
                max_iterations=3
            )
            
            print("\nAnalyzing perspectives...")
            print_analysis_results(result)
            
            # Optional: Save results to file
            # save_results_to_file(result, f"results_{test_case['description'].lower().replace(' ', '_')}.json")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    print("Advanced Internal Voice System Test")
    print("=" * 50)
    try:
        test_advanced_voice()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
