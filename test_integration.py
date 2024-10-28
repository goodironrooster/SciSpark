# test_integration.py
from llm_integration import LLMIntegration
from stream_status import StreamValidationType
import logging
import sys
import time
from typing import Dict, Any

def setup_logger():
    logger = logging.getLogger('TestIntegration')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def format_pattern_summary(patterns: dict) -> str:
    """Format pattern analysis results in a concise way"""
    if not patterns:
        return "No patterns detected"
        
    summary = []
    if 'repeating' in patterns:
        total = patterns['repeating'].get('total_patterns', 0)
        summary.append(f"Repeating patterns: {total}")
        
    if 'sequential' in patterns:
        total = len(patterns['sequential'].get('sequences', {}))
        summary.append(f"Sequential patterns: {total}")
        
    if 'structural' in patterns:
        dist = len(patterns['structural'].get('byte_distribution', {}))
        segments = patterns['structural'].get('segment_analysis', {}).get('total_segments', 0)
        summary.append(f"Structural: {dist} distributions, {segments} segments")
        
    return "\n".join(summary)

def format_shadow_summary(shadow_count: int, metrics: Dict[str, float]) -> str:
    """Format shadow detection results in a concise way"""
    return (
        f"Total shadows: {shadow_count}\n"
        f"Detection time: {metrics.get('pattern_analysis_time', 0):.3f}s"
    )

def format_concept_summary(updates: Dict[str, Any]) -> str:
    """Format concept mapping results in a concise way"""
    if not updates:
        return "No concept updates"
        
    return (
        f"Concept: {updates.get('name', 'N/A')}\n"
        f"Confidence: {updates.get('confidence', 0):.2f}\n"
        f"Shadow type: {updates.get('shadow_type', 'N/A')}\n"
        f"Mapped concepts: {len(updates.get('mapped_concepts', []))}"
    )

def print_section(title: str, content: str = None, separator: str = "-"):
    """Print a formatted section with title"""
    print(f"\n{title}")
    print(separator * 50)
    if content:
        print(content)

def main():
    logger = setup_logger()
    
    try:
        logger.info("Initializing LLM Integration...")
        
        # Create integration
        integration = LLMIntegration(
            model_path="D:/LLM models/phi-2-orange.Q4_0.gguf",
            n_gpu_layers=32,
            initial_buffer_size=8192,
            max_buffer_size=32768
        )

        # System Status
        status = integration.get_system_status()
        print_section("System Status Summary")
        print(f"Buffer: {status['buffer']['size']} bytes ({status['buffer']['usage']} used)")
        print(f"Pattern analyzers: {status['patterns']['analyzers']}")
        print(f"Shadow detectors: {status['shadows']['detectors']}")
        print(f"Concept mappings: {status['concepts']['total_mappings']}")

        # Test Cases
        test_prompts = [
            {
                "prompt": "What is gravity?",
                "concept_name": "gravity",
                "validation_type": StreamValidationType.FULL
            },
            {
                "prompt": "Explain how a computer works in one sentence.",
                "concept_name": "computer_function",
                "validation_type": StreamValidationType.INTEGRITY
            }
        ]

        # Process test cases
        for i, test_case in enumerate(test_prompts, 1):
            print_section(f"Test Case {i}")
            print(f"Prompt: {test_case['prompt']}")
            
            try:
                response = integration.process_prompt(
                    prompt=test_case['prompt'],
                    concept_name=test_case['concept_name']
                )

                # Display results
                print_section("Response", response.llm_response.content)
                
                print_section("Analysis Results")
                print("Patterns:", format_pattern_summary(response.patterns.patterns))
                print("\nShadow Detection:", format_shadow_summary(
                    response.performance_metrics['shadow_count'],
                    response.performance_metrics
                ))
                print("\nConcept Mapping:", format_concept_summary(response.concept_updates))
                
                print_section("Performance")
                print(f"Total time: {response.performance_metrics['total_time']:.3f}s")
                print(f"LLM latency: {response.performance_metrics['llm_latency']:.3f}s")
                print(f"Tokens/sec: {response.performance_metrics['tokens_per_second']:.2f}")

            except Exception as e:
                logger.error(f"Error processing test case {i}: {str(e)}")
                continue

        # Quick benchmark
        logger.info("Running benchmark...")
        print_section("Benchmark Results")
        
        benchmark_prompt = "What is the capital of France?"
        start_time = time.time()
        
        benchmark_response = integration.process_prompt(
            prompt=benchmark_prompt,
            max_tokens=32,
            temperature=0.7
        )
        
        print(f"Total time: {time.time() - start_time:.3f}s")
        print(f"Response time: {benchmark_response.performance_metrics['llm_latency']:.3f}s")
        print(f"Tokens/sec: {benchmark_response.performance_metrics['tokens_per_second']:.2f}")
        print(f"Shadows detected: {benchmark_response.performance_metrics['shadow_count']}")
        print(f"Analysis time: {benchmark_response.performance_metrics['pattern_analysis_time']:.3f}s")

    except Exception as e:
        logger.error(f"Error in integration test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    logger.info("Integration test completed successfully")

if __name__ == "__main__":
    main()
