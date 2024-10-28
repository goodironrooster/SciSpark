import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import threading
import time
from collections import defaultdict
from pattern_analyzer import PatternAnalyzer, PatternStatus
from stream_validator import StreamValidator
from buffer_manager import BufferManager

class ShadowType(Enum):
    DIRECT = "direct"          # Clear pattern shadows
    INDIRECT = "indirect"      # Implied pattern shadows
    COMPOSITE = "composite"    # Combined pattern shadows
    ABSTRACT = "abstract"      # Higher-level concept shadows
    DYNAMIC = "dynamic"        # Time-varying shadows

class ShadowPattern:
    def __init__(self, 
                 pattern_type: ShadowType,
                 data: bytes,
                 confidence: float,
                 metadata: Optional[Dict] = None):
        self.pattern_type = pattern_type
        self.data = data
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.validation_status = None

    def __str__(self):
        return f"ShadowPattern(type={self.pattern_type.value}, confidence={self.confidence:.2f})"

class ShadowDetectionSystem:
    def __init__(self, 
                 pattern_analyzer: Optional[PatternAnalyzer] = None,
                 debug_mode: bool = True):
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer()
        self.shadow_patterns: Dict[str, ShadowPattern] = {}
        self._lock = threading.Lock()
        self.debug_mode = debug_mode
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.sequence_history: List[bytes] = []
        self._initialize_shadow_detectors()

    def debug_print(self, message: str, level: str = "INFO") -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG ShadowDetector [{level}]: {message}")

    def _initialize_shadow_detectors(self) -> None:
        """Initialize the shadow detection methods"""
        self._detectors = {
            ShadowType.DIRECT: self._detect_direct_shadows,
            ShadowType.INDIRECT: self._detect_indirect_shadows,
            ShadowType.COMPOSITE: self._detect_composite_shadows,
            ShadowType.ABSTRACT: self._detect_abstract_shadows,
            ShadowType.DYNAMIC: self._detect_dynamic_shadows
        }

    def detect_shadows(self, data: bytes) -> List[ShadowPattern]:
        """Main method to detect shadows in data"""
        start_time = time.perf_counter()
        self.debug_print(f"Starting shadow detection for {len(data)} bytes")
        
        try:
            results = []
            # Store sequence in history
            self.sequence_history.append(data)
            if len(self.sequence_history) > 100:  # Limit history
                self.sequence_history.pop(0)
            
            # Analyze patterns first
            pattern_results = self.pattern_analyzer.analyze_sequence(data)

            # Detect shadows using each detector
            for shadow_type, detector in self._detectors.items():
                try:
                    detector_start = time.perf_counter()
                    shadows = detector(data, pattern_results.patterns)
                    detector_time = time.perf_counter() - detector_start
                    
                    self.metrics[f"{shadow_type.value}_detection_time"].append(detector_time)
                    results.extend(shadows)
                    
                    self.debug_print(f"{shadow_type.value} detection found {len(shadows)} shadows")
                except Exception as e:
                    self.debug_print(f"Error in {shadow_type.value} detection: {str(e)}", "ERROR")

            # Record total time
            total_time = time.perf_counter() - start_time
            self.metrics["total_detection_time"].append(total_time)
            
            self.debug_print(f"Shadow detection completed in {total_time:.4f} seconds")
            return results

        except Exception as e:
            self.debug_print(f"Shadow detection failed: {str(e)}", "ERROR")
            return []

    def _detect_direct_shadows(self, data: bytes, pattern_results: Dict) -> List[ShadowPattern]:
        """Detect direct shadows from clear patterns"""
        shadows = []
        try:
            # Look for repeating patterns first
            if 'repeating' in pattern_results:
                pattern_data = pattern_results['repeating']
                if 'patterns' in pattern_data:
                    for length, patterns in pattern_data['patterns'].items():
                        for i, (pattern_hex, count) in enumerate(patterns.items()):
                            try:
                                confidence = self._calculate_confidence(
                                    (pattern_hex, count), i, len(data)
                                )
                                shadow = ShadowPattern(
                                    ShadowType.DIRECT,
                                    data,
                                    confidence,
                                    {'pattern': pattern_hex, 'count': count, 'length': length}
                                )
                                shadows.append(shadow)
                            except Exception as e:
                                self.debug_print(f"Error processing pattern: {str(e)}", "ERROR")
                    
            self.debug_print(f"Detected {len(shadows)} direct shadows")
            return shadows
        except Exception as e:
            self.debug_print(f"Direct shadow detection failed: {str(e)}", "ERROR")
            return []

    def _detect_indirect_shadows(self, data: bytes, pattern_results: Dict) -> List[ShadowPattern]:
        """Detect indirect shadows from implied patterns"""
        try:
            shadows = []
            if 'sequential' in pattern_results:
                sequences = pattern_results['sequential'].get('sequences', {})
                for seq_type, positions in sequences.items():
                    if len(positions) > 2:  # Minimum sequence length
                        confidence = len(positions) / len(data)
                        shadow = ShadowPattern(
                            ShadowType.INDIRECT,
                            data,
                            confidence,
                            {'sequence_type': seq_type, 'positions': positions}
                        )
                        shadows.append(shadow)
            return shadows
        except Exception as e:
            self.debug_print(f"Indirect shadow detection failed: {str(e)}", "ERROR")
            return []

    def _detect_composite_shadows(self, data: bytes, pattern_results: Dict) -> List[ShadowPattern]:
        """Detect composite shadows from combined patterns"""
        try:
            shadows = []
            if 'structural' in pattern_results:
                structure = pattern_results['structural']
                if 'segment_analysis' in structure:
                    segments = structure['segment_analysis'].get('total_segments', 0)
                    if segments > 0:
                        avg_length = structure['segment_analysis'].get('average_segment_length', 0)
                        confidence = min(1.0, segments * avg_length / len(data))
                        shadow = ShadowPattern(
                            ShadowType.COMPOSITE,
                            data,
                            confidence,
                            {'segments': segments, 'avg_length': avg_length}
                        )
                        shadows.append(shadow)
            return shadows
        except Exception as e:
            self.debug_print(f"Composite shadow detection failed: {str(e)}", "ERROR")
            return []

    def _detect_abstract_shadows(self, data: bytes, pattern_results: Dict) -> List[ShadowPattern]:
        """Detect abstract shadows from higher-level patterns"""
        try:
            shadows = []
            if 'statistical' in pattern_results:
                stats = pattern_results['statistical']
                if 'basic_stats' in stats:
                    basic_stats = stats['basic_stats']
                    entropy = stats.get('entropy', 0)
                    
                    # Create abstract shadow based on statistical properties
                    confidence = 1.0 - (entropy / 8.0)  # 8 bits max entropy
                    shadow = ShadowPattern(
                        ShadowType.ABSTRACT,
                        data,
                        confidence,
                        {'stats': basic_stats, 'entropy': entropy}
                    )
                    shadows.append(shadow)
            return shadows
        except Exception as e:
            self.debug_print(f"Abstract shadow detection failed: {str(e)}", "ERROR")
            return []

    def _detect_dynamic_shadows(self, data: bytes, pattern_results: Dict) -> List[ShadowPattern]:
        """Detect dynamic shadows from time-varying patterns"""
        try:
            shadows = []
            # Compare with historical patterns if available
            if len(self.sequence_history) >1:
                prev_data = self.sequence_history[-2]  # Get previous data
                if len(prev_data) == len(data):
                    changes = sum(1 for a, b in zip(prev_data, data) if a != b)
                    change_ratio = changes / len(data)
                    confidence = 1.0 - change_ratio

                    shadow = ShadowPattern(
                        ShadowType.DYNAMIC,
                        data,
                        confidence,
                        {'changes': changes, 'change_ratio': change_ratio}
                    )
                    shadows.append(shadow)
            return shadows
        except Exception as e:
            self.debug_print(f"Dynamic shadow detection failed: {str(e)}", "ERROR")
            return []

    def _calculate_confidence(self, pattern: Tuple, position: int, text_length: int) -> float:
        """Calculate confidence score for a shadow pattern"""
        try:
            pattern_hex, frequency = pattern
            pattern_bytes = bytes.fromhex(pattern_hex)
            length = len(pattern_bytes)

            # Calculate complexity (example)
            complexity = len(set(pattern_bytes)) / length

            # Calculate position weight
            position_weight = 1 - (position / text_length)

            # Weighted confidence calculation
            raw_confidence = (frequency * length * (complexity ** 1.5) * position_weight) / 100  # Increased impact of complexity
            confidence = max(0.0, min(1.0, raw_confidence))
            return float(confidence)

        except Exception as e:
            self.debug_print(f"Confidence calculation failed: {str(e)}", "ERROR")
            return 0.0

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get the current performance metrics"""
        with self._lock:
            metrics = {}
            for metric_name, times in self.metrics.items():
                if times:
                    metrics[metric_name] = {
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
                else:
                    metrics[metric_name] = {
                        'avg': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0
                    }
            return metrics
