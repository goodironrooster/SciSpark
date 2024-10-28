from typing import Dict, Any, Optional, List
import time
import logging
from dataclasses import dataclass
from llama_cpp import Llama
import numpy as np
import os
import sys

@dataclass
class LLMResponse:
    content: str
    metadata: Dict[str, Any]
    raw_response: Any
    latency: float
    token_count: int
    model_name: str

class LLMInterface:
    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        quiet: bool = False
    ):
        self.logger = self._setup_logger()
        self.model_path = model_path
        self.quiet = quiet
        
        try:
            if self.quiet:
                # Suppress stdout temporarily
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers
                    )
                    sys.stdout = old_stdout
            else:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers
                )
            
            self.model_name = model_path.split('/')[-1]
            self.logger.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('LLMInterface')
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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                **kwargs
            )
            
            latency = time.time() - start_time
            
            # Get token counts
            prompt_tokens = len(self.llm.tokenize(prompt.encode()))
            completion_tokens = len(self.llm.tokenize(response['choices'][0]['text'].encode()))
            
            # Create response object
            llm_response = LLMResponse(
                content=response['choices'][0]['text'],
                metadata={
                    'model': self.model_name,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'repeat_penalty': repeat_penalty,
                    **kwargs
                },
                raw_response=response,
                latency=latency,
                token_count=prompt_tokens + completion_tokens,
                model_name=self.model_name
            )
            
            return llm_response

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'context_size': self.llm.n_ctx(),
            'vocab_size': self.llm.n_vocab(),
            'embedding_size': self.llm.n_embd()
        }

    def benchmark_inference(
        self,
        prompt: str,
        n_runs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        latencies = []
        token_counts = []
        
        try:
            for _ in range(n_runs):
                response = self.generate(prompt, **kwargs)
                latencies.append(response.latency)
                token_counts.append(response.token_count)
            
            return {
                'mean_latency': float(np.mean(latencies)),
                'std_latency': float(np.std(latencies)),
                'min_latency': float(np.min(latencies)),
                'max_latency': float(np.max(latencies)),
                'mean_tokens': float(np.mean(token_counts)),
                'tokens_per_second': float(np.mean(token_counts) / np.mean(latencies))
            }

        except Exception as e:
            self.logger.error(f"Error in benchmark: {str(e)}")
            raise
