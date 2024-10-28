# llm_integration.py
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import time

from llm_interface import LLMInterface, LLMResponse
from failure_detection import FailureDetector, FailureVector
from shadow_concept import ShadowConceptSystem, ConceptType
from benchmark_system import BenchmarkSystem, BenchmarkMetricType
from pattern_analyzer import PatternAnalyzer, PatternType, PatternStatus
from shadow_detection import ShadowDetectionSystem, ShadowPattern, ShadowType
from concept_mapper import ConceptMapper, ConceptLevel
from buffer_manager import BufferManager
from stream_status import StreamStatus, StreamValidationType

@dataclass
class IntegratedResponse:
    llm_response: LLMResponse
    failures: FailureVector
    concept_updates: Dict[str, Any]
    performance_metrics: Dict[str, float]
    patterns: PatternStatus
    stream_valid: bool

class LLMIntegration:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        initial_buffer_size: int = 8192,
        max_buffer_size: int = 32768
    ):
        # Initialize logger
        self.logger = self._setup_logger()
        
        try:
            # Initialize buffer manager
            self.buffer_manager = BufferManager(
                initial_size=initial_buffer_size,
                max_size=max_buffer_size
            )
            
            # Initialize StreamStatus
            self.stream_status = StreamStatus(
                success=True,
                validation_type=StreamValidationType.FULL,
                is_valid=True
            )
            
            # Initialize pattern analyzer
            self.pattern_analyzer = PatternAnalyzer(
                buffer_manager=self.buffer_manager
            )
            
            # Initialize shadow detection system
            self.shadow_detector = ShadowDetectionSystem(
                pattern_analyzer=self.pattern_analyzer,
                debug_mode=True
            )
            
            # Initialize all main systems
            self.llm = LLMInterface(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers
            )
            
            self.failure_detector = FailureDetector()
            self.shadow_system = ShadowConceptSystem()
            self.benchmark_system = BenchmarkSystem()
            self.concept_mapper = ConceptMapper(debug_mode=True)
            
            # Allocate initial buffer
            buffer_status = self.buffer_manager.allocate(initial_buffer_size)
            if not buffer_status.success:
                raise RuntimeError("Buffer allocation failed")
            
            self.logger.info("All systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing systems: {str(e)}")
            raise

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('LLMIntegration')
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

    def process_prompt(
        self,
        prompt: str,
        concept_name: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> IntegratedResponse:
        """Process a prompt through all systems"""
        start_time = time.time()
        try:
            # 1. Generate LLM Response
            llm_response = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # 2. Write response to buffer
            response_bytes = llm_response.content.encode()
            buffer_status = self.buffer_manager.write(response_bytes)
            if not buffer_status.success:
                self.logger.warning("Buffer write failed")
            
            # 3. Detect Failures
            failures = self.failure_detector.detect_failures(llm_response.content)
            
            # 4. Analyze Patterns
            pattern_status = self.pattern_analyzer.analyze_sequence(
                response_bytes,
                pattern_types=[PatternType.REPEATING, PatternType.SEQUENTIAL, PatternType.STRUCTURAL]
            )
            
            # 5. Detect Shadows
            shadows = self.shadow_detector.detect_shadows(response_bytes)
            
            # 6. Map Shadows to Concepts
            concept_updates = {}
            if shadows and concept_name:
                # Use the most confident shadow for mapping
                best_shadow = max(shadows, key=lambda x: x.confidence)
                mapped_concepts = self.concept_mapper.map_shadow_to_concept(best_shadow)
                
                if mapped_concepts:
                    concept_updates = {
                        'name': concept_name,
                        'mapped_concepts': [c.concept_id for c in mapped_concepts],
                        'confidence': sum(c.confidence for c in mapped_concepts) / len(mapped_concepts),
                        'shadow_type': best_shadow.pattern_type.value,
                        'shadow_confidence': best_shadow.confidence,
                        'patterns': pattern_status.patterns if pattern_status.success else {}
                    }
            
            # 7. Calculate Performance Metrics
            performance_metrics = {
                'total_time': time.time() - start_time,
                'llm_latency': llm_response.latency,
                'tokens_per_second': llm_response.token_count / llm_response.latency if llm_response.latency > 0 else 0,
                'failure_count': len(failures) if hasattr(failures, '__len__') else 0,
                'buffer_usage': buffer_status.current_usage if buffer_status.success else 0,
                'pattern_analysis_time': pattern_status.metrics.get('duration', 0) if pattern_status.metrics else 0,
                'shadow_count': len(shadows)
            }
            
            return IntegratedResponse(
                llm_response=llm_response,
                failures=failures,
                concept_updates=concept_updates,
                performance_metrics=performance_metrics,
                patterns=pattern_status,
                stream_valid=buffer_status.success
            )

        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of all systems"""
        try:
            buffer_status, _ = self.buffer_manager.peek()
            pattern_metrics = self.pattern_analyzer.get_performance_metrics()
            shadow_metrics = self.shadow_detector.get_performance_metrics()
            concept_metrics = self.concept_mapper.get_metrics()
            
            return {
                'buffer': {
                    'success': buffer_status.success,
                    'size': buffer_status.buffer_size,
                    'usage': buffer_status.current_usage
                },
                'patterns': {
                    'metrics': pattern_metrics,
                    'analyzers': len(self.pattern_analyzer._analyzers)
                },
                'shadows': {
                    'metrics': shadow_metrics,
                    'detectors': len(self.shadow_detector._detectors)
                },
                'concepts': {
                    'metrics': concept_metrics,
                    'total_mappings': len(self.concept_mapper.shadow_to_concept)
                },
                'llm': self.llm.get_model_info(),
                'benchmark': self.benchmark_system.get_system_performance()
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {
                'error': str(e),
                'status': 'error',
                'timestamp': time.time()
            }
