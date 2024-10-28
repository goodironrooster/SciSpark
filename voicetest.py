import logging
import sys
from internal_voice import InternalVoiceSystem
from llm_interface import LLMInterface
from shadow_detection import ShadowDetectionSystem
from concept_mapper import ConceptMapper
from failure_detection import FailureDetector
from benchmark_system import BenchmarkSystem
import traceback
import os

# Suppress CUDA initialization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def configure_logging():
    # Suppress most logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure specific loggers
    components = [
        'llm_interface',
        'ConceptMapper',
        'ShadowDetector',
        'PatternAnalyzer',
        'FailureDetector',
        'BenchmarkSystem',
        'InternalVoice',
        'transformers',
        'numpy'
    ]
    
    for component in components:
        logger = logging.getLogger(component)
        logger.setLevel(logging.ERROR)
        logger.handlers = [console_handler]

def test_internal_voice():
    configure_logging()
    
    try:
        print("\nInitializing LLM system...")
        
        # Initialize components silently
        llm = LLMInterface(
            model_path="D:/LLM models/Phi-3.5-mini-instruct_Uncensored-Q4_0_4_8.gguf",
            n_ctx=2048,
            n_gpu_layers=32,
            quiet=True
        )
        
        shadow_detector = ShadowDetectionSystem(debug_mode=False)
        concept_mapper = ConceptMapper(debug_mode=False)
        failure_detector = FailureDetector(debug_mode=False)
        benchmark_system = BenchmarkSystem()
        
        internal_voice = InternalVoiceSystem(
            llm=llm,
            shadow_detector=shadow_detector,
            concept_mapper=concept_mapper,
            failure_detector=failure_detector,
            benchmark_system=benchmark_system,
            debug_mode=False
        )
        
        print("System initialized successfully")
        
        # Test prompt
        test_prompt = "Explain why the sky is blue in a creative way"
        print(f"\nProcessing prompt: {test_prompt}")
        print("-" * 50)
        
        # Process thought
        result = internal_voice.process_thought(
            prompt=test_prompt,
            max_iterations=3
        )
        
        # Print results
        if result and 'dialogue_iterations' in result:
            for dialogue in result['dialogue_iterations']:
                print(f"\nIteration {dialogue.iteration}:")
                print("\nOriginal Response:")
                print(dialogue.original_response[:200] + "..." if len(dialogue.original_response) > 200 else dialogue.original_response)
                print("\nImproved Response:")
                print(dialogue.improved_response[:200] + "..." if len(dialogue.improved_response) > 200 else dialogue.improved_response)
                print(f"\nConfidence: {dialogue.confidence:.2f}")
            
            print("\nFinal Response:")
            print("-" * 50)
            print(result['final_response'])
            
            print("\nMetrics:")
            print("-" * 50)
            print(f"Total iterations: {result['metrics']['iterations']}")
            print(f"Processing time: {result['metrics']['processing_time']:.2f} seconds")
            print(f"Final confidence: {result['metrics']['final_confidence']:.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            print("\nTraceback:")
            print(traceback.format_exc())

if __name__ == "__main__":
    print("Internal Voice System Test")
    print("=" * 50)
    test_internal_voice()
