from internal_voice import InternalVoiceSystem
from llm_interface import LLMInterface
from shadow_detection import ShadowDetectionSystem
from concept_mapper import ConceptMapper
from failure_detection import FailureDetector
from benchmark_system import BenchmarkSystem

def main():
    # Initialize components
    llm = LLMInterface(
        model_path="path/to/your/model",
        n_gpu_layers=32
    )
    
    shadow_detector = ShadowDetectionSystem(debug_mode=True)
    concept_mapper = ConceptMapper(debug_mode=True)
    failure_detector = FailureDetector(debug_mode=True)
    benchmark_system = BenchmarkSystem()
    
    # Create internal voice system
    internal_voice = InternalVoiceSystem(
        llm=llm,
        shadow_detector=shadow_detector,
        concept_mapper=concept_mapper,
        failure_detector=failure_detector,
        benchmark_system=benchmark_system,
        debug_mode=True
    )
    
    # Test prompts
    test_prompts = [
        "Explain how quantum computers work",
        "Why do we dream?",
        "Describe a new color that doesn't exist"
    ]
    
    # Process each prompt
    for prompt in test_prompts:
        print(f"\nProcessing prompt: {prompt}")
        print("-" * 50)
        
        result = internal_voice.process_thought(prompt)
        
        print("\nInternal Dialogue:")
        for iteration in result['dialogue_iterations']:
            print(f"\nIteration {iteration.iteration}:")
            print(f"Original: {iteration.original_response[:100]}...")
            print(f"Internal Voice: {iteration.internal_reaction[:100]}...")
            print(f"Improved: {iteration.improved_response[:100]}...")
            print(f"Confidence: {iteration.confidence:.2f}")
        
        print("\nFinal Response:")
        print(result['final_response'])
        
        print("\nMetrics:")
        print(f"Total iterations: {result['metrics']['iterations']}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        print(f"Final confidence: {result['metrics']['final_confidence']:.2f}")
        
if __name__ == "__main__":
    main()