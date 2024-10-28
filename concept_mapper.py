from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import time
import threading
import math
from shadow_detection import ShadowPattern, ShadowType

class ConceptLevel(Enum):
    CONCRETE = "concrete"      # Direct physical patterns
    ABSTRACT = "abstract"      # Higher-level abstractions
    UNIVERSAL = "universal"    # Universal concepts
    METAPHYSICAL = "metaphysical"  # Highest level abstractions

@dataclass
class ConceptRelation:
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    metadata: Dict[str, Any]

class ConceptNode:
    def __init__(self, 
                 concept_id: str,
                 level: ConceptLevel,
                 confidence: float,
                 metadata: Optional[Dict] = None):
        self.concept_id = concept_id
        self.level = level
        self.confidence = confidence
        self.metadata = metadata or {}
        self.connections: Dict[str, float] = {}
        self.shadow_patterns: List[str] = []
        self.creation_time = time.time()
        self.last_updated = self.creation_time
        self.validation_status = None

    def __str__(self):
        return f"ConceptNode(id={self.concept_id}, level={self.level.value}, confidence={self.confidence:.2f})"

    def add_connection(self, target_id: str, weight: float) -> None:
        """Add or update a connection to another concept"""
        self.connections[target_id] = max(0.0, min(1.0, weight))  # Clamp weight between 0 and 1
        self.last_updated = time.time()

    def add_shadow_pattern(self, pattern_id: str) -> None:
        """Associate a shadow pattern with this concept"""
        if pattern_id not in self.shadow_patterns:
            self.shadow_patterns.append(pattern_id)
            self.last_updated = time.time()

class ConceptMapper:
    def __init__(self, debug_mode: bool = True):
        self.concepts: Dict[str, ConceptNode] = {}
        self.shadow_to_concept: Dict[str, Set[str]] = {}
        self.concept_hierarchies: Dict[ConceptLevel, Set[str]] = {
            level: set() for level in ConceptLevel
        }
        self.relations: List[ConceptRelation] = []
        self.debug_mode = debug_mode
        self._lock = threading.Lock()
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._initialize_base_concepts()

    def debug_print(self, message: str, level: str = "INFO") -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG ConceptMapper [{level}]: {message}")

    def _initialize_base_concepts(self) -> None:
        """Initialize fundamental concept structures"""
        try:
            # Initialize fundamental concepts for each level
            base_concepts = {
                ConceptLevel.CONCRETE: [
                    ("pattern_sequence", 1.0, {"type": "sequential"}),
                    ("pattern_repetition", 1.0, {"type": "repetitive"}),
                    ("pattern_structure", 1.0, {"type": "structural"})
                ],
                ConceptLevel.ABSTRACT: [
                    ("pattern_group", 0.9, {"type": "grouping"}),
                    ("pattern_relation", 0.9, {"type": "relational"}),
                    ("pattern_hierarchy", 0.9, {"type": "hierarchical"})
                ],
                ConceptLevel.UNIVERSAL: [
                    ("pattern_archetype", 0.8, {"type": "archetypal"}),
                    ("pattern_principle", 0.8, {"type": "principal"}),
                    ("pattern_essence", 0.8, {"type": "essential"})
                ],
                ConceptLevel.METAPHYSICAL: [
                    ("pattern_truth", 0.7, {"type": "truth"}),
                    ("pattern_form", 0.7, {"type": "form"}),
                    ("pattern_reality", 0.7, {"type": "reality"})
                ]
            }

            # Create concept nodes for each base concept
            for level, concepts in base_concepts.items():
                for concept_id, confidence, metadata in concepts:
                    node = ConceptNode(concept_id, level, confidence, metadata)
                    self.concepts[concept_id] = node
                    self.concept_hierarchies[level].add(concept_id)

            self.debug_print(f"Initialized {len(self.concepts)} base concepts")

        except Exception as e:
            self.debug_print(f"Error initializing base concepts: {str(e)}", "ERROR")
            raise

    def map_shadow_to_concept(self, shadow: ShadowPattern) -> List[ConceptNode]:
        """Map a shadow pattern to relevant concepts"""
        try:
            start_time = time.perf_counter()
            mapped_concepts = []

            if shadow is None:
                raise ValueError("Shadow pattern cannot be None")

            # Map based on pattern type
            if shadow.pattern_type == ShadowType.DIRECT:
                mapped_concepts.extend(self._map_direct_pattern(shadow))
            elif shadow.pattern_type == ShadowType.INDIRECT:
                mapped_concepts.extend(self._map_indirect_pattern(shadow))
            elif shadow.pattern_type == ShadowType.COMPOSITE:
                mapped_concepts.extend(self._map_composite_pattern(shadow))
            elif shadow.pattern_type == ShadowType.ABSTRACT:
                mapped_concepts.extend(self._map_abstract_pattern(shadow))

            # Record mapping metrics
            elapsed_time = time.perf_counter() - start_time
            self.metrics['mapping_time'].append(elapsed_time)

            if mapped_concepts:
                avg_confidence = sum(c.confidence for c in mapped_concepts) / len(mapped_concepts)
                self.metrics['mapping_confidence'].append(avg_confidence)

            self.debug_print(f"Mapped shadow to {len(mapped_concepts)} concepts in {elapsed_time:.6f}s")
            return mapped_concepts

        except Exception as e:
            self.debug_print(f"Error mapping shadow to concept: {str(e)}", "ERROR")
            return []

    def validate_concept_mapping(self, shadow: ShadowPattern, concept: ConceptNode) -> bool:
        """Validate the mapping between shadow and concept"""
        try:
            # Strict confidence validation
            if not isinstance(shadow.confidence, (int, float)):
                return False
                
            if math.isnan(shadow.confidence) or math.isinf(shadow.confidence):
                return False
                
            if shadow.confidence < 0 or shadow.confidence > 1:
                return False

            if not isinstance(concept.confidence, (int, float)):
                return False
                
            if math.isnan(concept.confidence) or math.isinf(concept.confidence):
                return False
                
            if concept.confidence < 0 or concept.confidence > 1:
                return False

            # Check pattern type compatibility
            if not self._check_pattern_compatibility(shadow, concept):
                return False

            # Check combined confidence threshold
            combined_confidence = shadow.confidence * concept.confidence
            if combined_confidence < 0.3:
                return False

            # Validate metadata matching
            if not self._validate_metadata_matching(shadow, concept):
                return False

            return True

        except Exception as e:
            self.debug_print(f"Error validating concept mapping: {str(e)}", "ERROR")
            return False

    def _map_direct_pattern(self, shadow: ShadowPattern) -> List[ConceptNode]:
        """Map direct shadow patterns to concepts"""
        concepts = []
        try:
            # Map to concrete concepts first
            for concept_id in self.concept_hierarchies[ConceptLevel.CONCRETE]:
                concept = self.concepts[concept_id]
                if self.validate_concept_mapping(shadow, concept):
                    concepts.append(concept)
                    self._update_mapping_statistics(shadow, concept)

            # If strong confidence, consider abstract concepts
            if shadow.confidence > 0.8:
                for concept_id in self.concept_hierarchies[ConceptLevel.ABSTRACT]:
                    concept = self.concepts[concept_id]
                    if self.validate_concept_mapping(shadow, concept):
                        concepts.append(concept)
                        self._update_mapping_statistics(shadow, concept)

        except Exception as e:
            self.debug_print(f"Error in direct pattern mapping: {str(e)}", "ERROR")

        return concepts

    def _map_indirect_pattern(self, shadow: ShadowPattern) -> List[ConceptNode]:
        """Map indirect shadow patterns to concepts"""
        concepts = []
        try:
            # Map to abstract concepts primarily
            for concept_id in self.concept_hierarchies[ConceptLevel.ABSTRACT]:
                concept = self.concepts[concept_id]
                if self.validate_concept_mapping(shadow, concept):
                    concepts.append(concept)
                    self._update_mapping_statistics(shadow, concept)

            # If strong confidence, consider universal concepts
            if shadow.confidence > 0.8:
                for concept_id in self.concept_hierarchies[ConceptLevel.UNIVERSAL]:
                    concept = self.concepts[concept_id]
                    if self.validate_concept_mapping(shadow, concept):
                        concepts.append(concept)
                        self._update_mapping_statistics(shadow, concept)

        except Exception as e:
            self.debug_print(f"Error in indirect pattern mapping: {str(e)}", "ERROR")

        return concepts

    def _map_composite_pattern(self, shadow: ShadowPattern) -> List[ConceptNode]:
        """Map composite shadow patterns to concepts"""
        concepts = []
        try:
            # Map to abstract and universal concepts
            for level in [ConceptLevel.ABSTRACT, ConceptLevel.UNIVERSAL]:
                for concept_id in self.concept_hierarchies[level]:
                    concept = self.concepts[concept_id]
                    if self.validate_concept_mapping(shadow, concept):
                        concepts.append(concept)
                        self._update_mapping_statistics(shadow, concept)

        except Exception as e:
            self.debug_print(f"Error in composite pattern mapping: {str(e)}", "ERROR")

        return concepts

    def _map_abstract_pattern(self, shadow: ShadowPattern) -> List[ConceptNode]:
        """Map abstract shadow patterns to concepts"""
        concepts = []
        try:
            # Map to universal and metaphysical concepts
            for level in [ConceptLevel.UNIVERSAL, ConceptLevel.METAPHYSICAL]:
                for concept_id in self.concept_hierarchies[level]:
                    concept = self.concepts[concept_id]
                    if self.validate_concept_mapping(shadow, concept):
                        concepts.append(concept)
                        self._update_mapping_statistics(shadow, concept)

        except Exception as e:
            self.debug_print(f"Error in abstract pattern mapping: {str(e)}", "ERROR")

        return concepts

    def _update_mapping_statistics(self, shadow: ShadowPattern, concept: ConceptNode) -> None:
        """Update statistical information about the mapping"""
        try:
            with self._lock:
                # Update shadow to concept mapping
                if shadow.pattern_type not in self.shadow_to_concept:
                    self.shadow_to_concept[shadow.pattern_type] = set()
                self.shadow_to_concept[shadow.pattern_type].add(concept.concept_id)

                # Update concept node
                concept.add_shadow_pattern(str(shadow.pattern_type))
                concept.last_updated = time.time()

                # Update metrics
                self.metrics['mapping_confidence'].append(shadow.confidence * concept.confidence)

        except Exception as e:
            self.debug_print(f"Error updating mapping statistics: {str(e)}", "ERROR")

    def _validate_metadata_matching(self, shadow: ShadowPattern, concept: ConceptNode) -> bool:
        """Validate metadata compatibility between shadow and concept"""
        try:
            # Must have metadata to compare
            if not shadow.metadata or not concept.metadata:
                return True  # No metadata to compare, assume compatible

            # Check for type compatibility
            shadow_type = shadow.metadata.get('type')
            concept_type = concept.metadata.get('type')
            
            if shadow_type and concept_type:
                return self._check_type_compatibility(shadow_type, concept_type)

            return True

        except Exception as e:
            self.debug_print(f"Error validating metadata: {str(e)}", "ERROR")
            return False

    def _check_pattern_compatibility(self, shadow: ShadowPattern, concept: ConceptNode) -> bool:
        """Check if shadow pattern is compatible with concept"""
        try:
            # Pattern type compatibility rules
            pattern_concept_compatibility = {
                ShadowType.DIRECT: [ConceptLevel.CONCRETE, ConceptLevel.ABSTRACT],
                ShadowType.INDIRECT: [ConceptLevel.ABSTRACT, ConceptLevel.UNIVERSAL],
                ShadowType.COMPOSITE: [ConceptLevel.ABSTRACT, ConceptLevel.UNIVERSAL],
                ShadowType.ABSTRACT: [ConceptLevel.UNIVERSAL, ConceptLevel.METAPHYSICAL]
            }

            return concept.level in pattern_concept_compatibility.get(shadow.pattern_type, [])

        except Exception as e:
            self.debug_print(f"Error checking pattern compatibility: {str(e)}", "ERROR")
            return False

    def _check_type_compatibility(self, shadow_type: str, concept_type: str) -> bool:
        """Check compatibility between shadow and concept types"""
        # Type compatibility rules
        compatibility_rules = {
            "sequential": ["sequential", "relational", "hierarchical"],
            "repetitive": ["repetitive", "grouping", "archetypal"],
            "structural": ["structural", "hierarchical", "form"],
            "relational": ["relational", "principle", "truth"],
            "hierarchical": ["hierarchical", "essence", "reality"]
        }

        return concept_type in compatibility_rules.get(shadow_type, [])

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics"""
        try:
            metrics = {}
            for metric_name, values in self.metrics.items():
                if values:
                    metrics[metric_name] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            return metrics
        except Exception as e:
            self.debug_print(f"Error getting metrics: {str(e)}", "ERROR")
            return {}
