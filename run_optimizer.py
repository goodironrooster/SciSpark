# run_optimizer.py
from neural_shadow_optimizer import CreativeOptimizer
from your_llm_implementation import LLM  # Replace with your LLM implementation
import logging
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    try:
        # Initialize your LLM
        llm = LLM(model_path="D:\LLM models\phi-2-orange.Q4_0.gguf")
        
        # Initialize the Creative Optimizer
        optimizer = CreativeOptimizer(base_llm=llm)
        
        # Test prompts
        test_prompts = [
            "Explore the relationship between quantum entanglement and consciousness",
            "Propose a new theory about dark matter's interaction with ordinary matter",
            "Investigate potential mechanisms for faster-than-light communication"
        ]
        
        # Process each prompt
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\nProcessing prompt {i}: {prompt}")
            
            start_time = time.time()
            result = optimizer.generate_creative_output(prompt, research_mode=True)
            
            # Log results
            logger.info(f"\nOriginal Response:\n{result['original_response'][:200]}...")
            logger.info(f"\nRefined Response:\n{result['refined_response'][:200]}...")
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"Research Score: {result['research_score']:.2f}")
            logger.info(f"Shadows Detected: {result['shadows_detected']}")
            logger.info(f"Hypotheses Generated: {result['hypotheses_generated']}")
            logger.info(f"Processing Time: {time.time() - start_time:.2f}s")
            
            # Get additional metrics
            metrics = optimizer.get_performance_metrics()
            logger.info("\nSystem Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}")
            
            logger.info("-" * 80)
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
