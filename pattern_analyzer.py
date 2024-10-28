import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import threading
import time
from collections import defaultdict
from stream_validator import StreamValidator
from buffer_manager import BufferManager

class PatternType(Enum):
    REPEATING = "repeating"
    SEQUENTIAL = "sequential"
    STRUCTURAL = "structural"
    STATISTICAL = "statistical"
    CUSTOM = "custom"

class PatternStatus:
    def __init__(self, 
                 success: bool, 
                 pattern_type: Optional[PatternType] = None,
                 patterns: Optional[Dict] = None,
                 metrics: Optional[Dict] = None,
                 error: Optional[str] = None):
        self.success = success
        self.pattern_type = pattern_type
        self.patterns = patterns or {}
        self.metrics = metrics or {}
        self.error = error
        self.timestamp = time.time()

    def __str__(self):
        return f"PatternStatus(success={self.success}, patterns={self.patterns}, error={self.error})"

class PatternAnalyzer:
    def __init__(self, buffer_manager: Optional[BufferManager] = None,
                 stream_validator: Optional[StreamValidator] = None):
        self.buffer_manager = buffer_manager
        self.stream_validator = stream_validator
        self._lock = threading.Lock()
        self.patterns: Dict[str, Dict] = {}
        self.sequence_history: List[bytes] = []
        self._analyzers: Dict[PatternType, callable] = {
            PatternType.REPEATING: self._analyze_repeating_patterns,
            PatternType.SEQUENTIAL: self._analyze_sequential_patterns,
            PatternType.STRUCTURAL: self._analyze_structural_patterns,
            PatternType.STATISTICAL: self._analyze_statistical_patterns
        }
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.debug_mode = True

    def debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG PatternAnalyzer: {message}")

    def _record_metric(self, metric_name: str, duration: float) -> None:
        """Record a performance metric"""
        with self._lock:
            self.metrics[metric_name].append(duration)

    def analyze_sequence(self, data: bytes, pattern_types: Optional[List[PatternType]] = None) -> PatternStatus:
        """Analyze a byte sequence for specified pattern types"""
        start_time = time.perf_counter()
        self.debug_print(f"Starting sequence analysis for {len(data)} bytes")
        
        try:
            pattern_types = pattern_types or list(PatternType)
            results = {}
            
            for pattern_type in pattern_types:
                if pattern_type in self._analyzers:
                    analyzer_start = time.perf_counter()
                    self.debug_print(f"Analyzing {pattern_type.value} patterns")
                    
                    pattern_result = self._analyzers[pattern_type](data)
                    analyzer_duration = time.perf_counter() - analyzer_start
                    self._record_metric(f"{pattern_type.value}_analysis_time", analyzer_duration)
                    
                    results[pattern_type.value] = pattern_result

            # Store sequence in history
            with self._lock:
                self.sequence_history.append(data)
                if len(self.sequence_history) > 1000:  # Limit history size
                    self.sequence_history.pop(0)

            duration = time.perf_counter() - start_time
            self._record_metric('total_analysis_time', duration)
            
            self.debug_print(f"Analysis completed in {duration:.4f} seconds")
            
            return PatternStatus(
                success=True,
                patterns=results,
                metrics={'duration': duration}
            )

        except Exception as e:
            self.debug_print(f"Analysis failed: {str(e)}")
            return PatternStatus(False, error=str(e))

    def _analyze_repeating_patterns(self, data: bytes) -> Dict:
        """Analyze repeating patterns in the data"""
        try:
            patterns = {}
            min_pattern_length = 2
            max_pattern_length = min(32, len(data) // 2)
            
            for pattern_length in range(min_pattern_length, max_pattern_length + 1):
                pattern_counts = defaultdict(int)
                
                for i in range(0, len(data) - pattern_length + 1):
                    pattern = data[i:i + pattern_length]
                    pattern_counts[pattern] += 1
                
                # Filter significant patterns (occurring more than once)
                significant_patterns = {
                    pattern.hex(): count 
                    for pattern, count in pattern_counts.items() 
                    if count > 1
                }
                
                if significant_patterns:
                    patterns[pattern_length] = significant_patterns

            return {
                'patterns': patterns,
                'total_patterns': sum(len(p) for p in patterns.values())
            }

        except Exception as e:
            self.debug_print(f"Repeating pattern analysis failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_sequential_patterns(self, data: bytes) -> Dict:
        """Analyze sequential patterns in the data"""
        try:
            sequences = defaultdict(list)
            byte_array = list(data)
            
            for i in range(len(byte_array) - 1):
                diff = (byte_array[i + 1] - byte_array[i]) % 256  # Handle wrap-around
                seq_type = f"diff_{diff}"
                sequences[seq_type].append(i)

            # Filter significant sequences (appearing more than twice)
            significant_sequences = {
                k: v for k, v in sequences.items()
                if len(v) > 2
            }

            return {
                'sequences': dict(significant_sequences),
                'total_sequences': len(significant_sequences)
            }

        except Exception as e:
            self.debug_print(f"Sequential pattern analysis failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_structural_patterns(self, data: bytes) -> Dict:
        """Analyze structural patterns in the data"""
        try:
            structure = {
                'byte_distribution': {},
                'segment_analysis': {},
                'boundary_markers': []
            }

            # Analyze byte distribution without validation
            byte_counts = defaultdict(int)
            for byte in data:
                byte_counts[byte] += 1
            structure['byte_distribution'] = dict(byte_counts)

            # Analyze potential segments and boundaries
            for i in range(len(data)):
                if data[i] in {0x00, 0xFF}:  # Look for common boundary markers
                    structure['boundary_markers'].append(i)

            # Analyze segment patterns (including last segment)
            segments = []
            last_boundary = 0
            markers = sorted(structure['boundary_markers'] + [len(data)])
            
            for marker in markers:
                if marker - last_boundary > 0:  # Include all valid segments
                    segments.append(data[last_boundary:marker])
                last_boundary = marker + 1

            # Calculate segment statistics
            if segments:
                avg_length = sum(len(s) for s in segments) / len(segments)
            else:
                avg_length = 0

            structure['segment_analysis'] = {
                'total_segments': len(segments),
                'average_segment_length': float(avg_length),
                'segment_boundaries': structure['boundary_markers']
            }

            return structure

        except Exception as e:
            self.debug_print(f"Structural pattern analysis failed: {str(e)}")
            return {'error': str(e)}

    def _analyze_statistical_patterns(self, data: bytes) -> Dict:
        """Analyze statistical patterns in the data"""
        try:
            stats = {
                'basic_stats': {},
                'distribution': {},
                'entropy': 0.0
            }

            # Convert bytes to numpy array without validation
            byte_array = np.array([int(b) for b in data], dtype=np.uint8)
            
            # Basic statistics
            stats['basic_stats'] = {
                'mean': float(np.mean(byte_array)),
                'std': float(np.std(byte_array)),
                'min': int(np.min(byte_array)),
                'max': int(np.max(byte_array)),
                'median': float(np.median(byte_array))
            }

            # Byte distribution
            unique, counts = np.unique(byte_array, return_counts=True)
            total_bytes = len(byte_array)
            
            # Calculate entropy (with protection against zero probabilities)
            probabilities = counts / total_bytes
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            stats['entropy'] = float(entropy)

            # Store distribution
            stats['distribution'] = {
                int(value): int(count) 
                for value, count in zip(unique, counts)
            }

            return stats

        except Exception as e:
            self.debug_print(f"Statistical pattern analysis failed: {str(e)}")
            return {'error': str(e)}

    def detect_anomalies(self, data: bytes, threshold: float = 2.0) -> List[Dict]:
        """Detect anomalies in byte sequences using statistical analysis"""
        try:
            if len(data) < 3:  # Need at least 3 points for meaningful analysis
                return []

            # Convert bytes to numpy array
            byte_array = np.array([int(b) for b in data], dtype=np.float64)
            
            # Calculate rolling statistics with smaller window for better sensitivity
            window_size = min(10, len(byte_array) // 2)
            if window_size < 3:
                window_size = 3
                
            anomalies = []
            
            # Use centered window for better anomaly detection
            for i in range(window_size, len(byte_array) - window_size + 1):
                window = byte_array[i-window_size:i+window_size]
                current_value = byte_array[i]
                
                # Calculate local statistics
                mean = np.mean(window)
                std = np.std(window)
                
                # Avoid division by zero
                if std < 1e-6:
                    std = 1e-6
                
                # Calculate z-score
                z_score = abs(current_value - mean) / std
                
                # Check for anomaly
                if z_score > threshold:
                    anomalies.append({
                        'position': i,
                        'value': int(byte_array[i]),
                        'z_score': float(z_score),
                        'local_mean': float(mean),
                        'local_std': float(std)
                    })
            
            # Check edges separately
            edge_window_size = min(5, len(byte_array) // 4)
            if edge_window_size >= 2:
                # Check start
                start_mean = np.mean(byte_array[:edge_window_size*2])
                start_std = np.std(byte_array[:edge_window_size*2])
                if start_std < 1e-6:
                    start_std = 1e-6
                    
                for i in range(edge_window_size):
                    z_score = abs(byte_array[i] - start_mean) / start_std
                    if z_score > threshold:
                        anomalies.append({
                            'position': i,
                            'value': int(byte_array[i]),
                            'z_score': float(z_score),
                            'local_mean': float(start_mean),
                            'local_std': float(start_std)
                        })
                
                # Check end
                end_mean = np.mean(byte_array[-edge_window_size*2:])
                end_std = np.std(byte_array[-edge_window_size*2:])
                if end_std < 1e-6:
                    end_std = 1e-6
                    
                for i in range(len(byte_array) - edge_window_size, len(byte_array)):
                    z_score = abs(byte_array[i] - end_mean) / end_std
                    if z_score > threshold:
                        anomalies.append({
                            'position': i,
                            'value': int(byte_array[i]),
                            'z_score': float(z_score),
                            'local_mean': float(end_mean),
                            'local_std': float(end_std)
                        })
            
            # Sort anomalies by position
            anomalies.sort(key=lambda x: x['position'])
            
            # Remove duplicates while preserving order
            seen_positions = set()
            unique_anomalies = []
            for anomaly in anomalies:
                if anomaly['position'] not in seen_positions:
                    seen_positions.add(anomaly['position'])
                    unique_anomalies.append(anomaly)
            
            return unique_anomalies

        except Exception as e:
            self.debug_print(f"Anomaly detection failed: {str(e)}")
            return []

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
