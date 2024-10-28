# test_llm.py

from llm_interface import LLMInterface

def main():
    # Initialize the interface with your phi-2 model
    llm = LLMInterface(
        model_path="D:/LLM models/phi-2-orange.Q4_0.gguf",
        n_ctx=2048,
        n_threads=4,  # You can adjust this based on your CPU
        n_gpu_layers=0  # Keep 0 for CPU-only, increase if you want to use GPU
    )

    # Test single response generation
    response = llm.generate(
        prompt="What is the capital of France?",
        max_tokens=128,
        temperature=0.7
    )

    print(f"Response: {response.content}")
    print(f"Latency: {response.latency:.3f}s")
    print(f"Tokens: {response.token_count}")

    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_results = llm.benchmark_inference(
        prompt="Explain the theory of relativity",
        n_runs=5
    )

    print("\nBenchmark Results:")
    for metric, value in benchmark_results.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()