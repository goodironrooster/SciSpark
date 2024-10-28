from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import json
import os

class BenchmarkMetricType(Enum):
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"

@dataclass
class BenchmarkResult:
    metric_name: str
    metric_type: BenchmarkMetricType
    value: float
    timestamp: float
    metadata: Dict[str, Any]
    confidence: float

class BenchmarkSystem:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logger()
        self.metrics: Dict[str, List[BenchmarkResult]] = {}
        self.thresholds = self._load_config(config_path)
        self._lock = threading.Lock()
        
        # Performance tracking
        self.start_time = time.time()
        self.total_operations = 0
        self.failed_operations = 0
        
        # Resource tracking
        self.peak_memory = 0
        self.total_execution_time = 0
        
        # Reliability metrics
        self.uptime_start = time.time()
        self.error_count = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('BenchmarkSystem')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _load_config(self, config_path: Optional[str]) -> Dict[str, float]:
        """Load benchmark thresholds from config file"""
        default_thresholds = {
            'min_accuracy': 0.8,
            'max_latency': 1.0,  # seconds
            'min_reliability': 0.99,
            'max_memory_usage': 1024,  # MB
            'min_throughput': 100,  # operations/second
            'accuracy_tolerance': 0.1,
            'latency_tolerance': 0.2,
            'memory_tolerance': 0.15,
            'throughput_tolerance': 0.1
        }
        
        if not config_path:
            return default_thresholds
            
        try:
            with open(config_path, 'r') as f:
                custom_thresholds = json.load(f)
                return {**default_thresholds, **custom_thresholds}
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return default_thresholds

    def record_metric(self, name: str, value: float, metric_type: BenchmarkMetricType, 
                     metadata: Optional[Dict] = None) -> None:
        """Record a benchmark metric"""
        try:
            if not isinstance(metric_type, BenchmarkMetricType):
                raise TypeError(f"metric_type must be BenchmarkMetricType, got {type(metric_type)}")
            if not isinstance(value, (int, float)):
                raise TypeError(f"value must be numeric, got {type(value)}")
            if not name:
                raise ValueError("Metric name cannot be empty")

            with self._lock:
                result = BenchmarkResult(
                    metric_name=name,
                    metric_type=metric_type,
                    value=float(value),
                    timestamp=time.time(),
                    metadata=metadata or {},
                    confidence=self._calculate_confidence(value, metric_type)
                )
                
                if name not in self.metrics:
                    self.metrics[name] = []
                    
                self.metrics[name].append(result)
                self._update_system_metrics(result)
                
                self.logger.debug(f"Recorded metric {name}: {value}")
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error recording metric: {str(e)}")
            raise

    def _calculate_confidence(self, value: float, metric_type: BenchmarkMetricType) -> float:
        """Calculate confidence score for a metric value"""
        try:
            if metric_type == BenchmarkMetricType.ACCURACY:
                return min(1.0, max(0.0, value))
            elif metric_type == BenchmarkMetricType.PERFORMANCE:
                # Lower is better for latency/performance metrics
                max_latency = self.thresholds['max_latency']
                return max(0.0, min(1.0, 1.0 - (value / max_latency)))
            elif metric_type == BenchmarkMetricType.RELIABILITY:
                return max(0.0, min(1.0, value))
            elif metric_type == BenchmarkMetricType.RESOURCE:
                # Lower is better for resource usage
                max_memory = self.thresholds['max_memory_usage']
                return max(0.0, min(1.0, 1.0 - (value / max_memory)))
            elif metric_type == BenchmarkMetricType.SCALABILITY:
                # Higher is better for throughput/scalability
                min_throughput = self.thresholds['min_throughput']
                return max(0.0, min(1.0, value / min_throughput))
            else:
                self.logger.warning(f"Unknown metric type: {metric_type}")
                return 0.5
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _update_system_metrics(self, result: BenchmarkResult) -> None:
        """Update system-wide metrics based on new result"""
        try:
            self.total_operations += 1
            
            # Update failure count based on thresholds
            if result.metric_type == BenchmarkMetricType.ACCURACY:
                if result.value < self.thresholds['min_accuracy']:
                    self.failed_operations += 1
            elif result.metric_type == BenchmarkMetricType.PERFORMANCE:
                if result.value > self.thresholds['max_latency']:
                    self.failed_operations += 1
            
            # Update resource metrics
            if 'memory_usage' in result.metadata:
                self.peak_memory = max(self.peak_memory, result.metadata['memory_usage'])
            if 'execution_time' in result.metadata:
                self.total_execution_time += result.metadata['execution_time']

        except Exception as e:
            self.logger.error(f"Error updating system metrics: {str(e)}")

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical analysis of a metric"""
        try:
            if metric_name not in self.metrics:
                return {}
                
            values = [result.value for result in self.metrics[metric_name]]
            if not values:
                return {}
                
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
                'latest': float(values[-1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating statistics for {metric_name}: {str(e)}")
            return {}

    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.uptime_start
            
            metrics = {
                'uptime': uptime,
                'total_operations': self.total_operations,
                'operations_per_second': self.total_operations / uptime if uptime > 0 else 0,
                'error_rate': self.error_count / self.total_operations if self.total_operations > 0 else 0,
                'success_rate': (self.total_operations - self.failed_operations) / self.total_operations 
                              if self.total_operations > 0 else 0,
                'peak_memory_usage': self.peak_memory,
                'average_latency': self.total_execution_time / self.total_operations 
                                 if self.total_operations > 0 else 0
            }
            
            metrics['reliability_score'] = 1.0 - metrics['error_rate']
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system performance: {str(e)}")
            return {}

    def evaluate_performance(self) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if system meets performance thresholds"""
        try:
            performance = self.get_system_performance()
            
            evaluations = {
                'accuracy': performance['success_rate'] >= self.thresholds['min_accuracy'],
                'latency': performance['average_latency'] <= self.thresholds['max_latency'],
                'reliability': performance['reliability_score'] >= self.thresholds['min_reliability'],
                'throughput': performance['operations_per_second'] >= self.thresholds['min_throughput'],
                'memory': performance['peak_memory_usage'] <= self.thresholds['max_memory_usage']
            }
            
            meets_thresholds = all(evaluations.values())
            
            details = {
                'evaluations': evaluations,
                'performance': performance,
                'thresholds': self.thresholds,
                'deviations': self._calculate_deviations(performance)
            }
            
            return meets_thresholds, details
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance: {str(e)}")
            return False, {}

    def _calculate_deviations(self, performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate deviations from thresholds"""
        try:
            deviations = {}
            
            if 'success_rate' in performance:
                deviations['accuracy'] = performance['success_rate'] - self.thresholds['min_accuracy']
                
            if 'average_latency' in performance:
                deviations['latency'] = self.thresholds['max_latency'] - performance['average_latency']
                
            if 'reliability_score' in performance:
                deviations['reliability'] = performance['reliability_score'] - self.thresholds['min_reliability']
                
            if 'operations_per_second' in performance:
                deviations['throughput'] = (performance['operations_per_second'] - 
                                          self.thresholds['min_throughput'])
                
            if 'peak_memory_usage' in performance:
                deviations['memory'] = (self.thresholds['max_memory_usage'] - 
                                      performance['peak_memory_usage'])
                
            return deviations
            
        except Exception as e:
            self.logger.error(f"Error calculating deviations: {str(e)}")
            return {}

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        try:
            meets_thresholds, evaluation_details = self.evaluate_performance()
            
            report = {
                'timestamp': time.time(),
                'meets_thresholds': meets_thresholds,
                'system_performance': evaluation_details['performance'],
                'evaluations': evaluation_details['evaluations'],
                'thresholds': evaluation_details['thresholds'],
                'deviations': evaluation_details['deviations'],
                'metrics': {}
            }
            
            # Add individual metric statistics
            for metric_name in self.metrics:
                report['metrics'][metric_name] = self.get_metric_statistics(metric_name)
                
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

    def save_report(self, filepath: str) -> None:
        """Save benchmark report to file"""
        try:
            report = self.generate_report()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Benchmark report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            raise

    def reset_metrics(self) -> None:
        """Reset all benchmark metrics"""
        try:
            with self._lock:
                self.metrics.clear()
                self.total_operations = 0
                self.failed_operations = 0
                self.peak_memory = 0
                self.total_execution_time = 0
                self.error_count = 0
                self.uptime_start = time.time()
                self.logger.info("Benchmark metrics reset")
                
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {str(e)}")
            raise
