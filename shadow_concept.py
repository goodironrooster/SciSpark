from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
from dataclasses import dataclass
import threading
from enum import Enum
import logging

class ConceptType(Enum):
    FACTUAL = "factual"         # Represents verifiable facts
    LOGICAL = "logical"         # Represents logical relationships
    SEMANTIC = "semantic"       # Represents meaning and context
    PROCEDURAL = "procedural"   # Represents processes or methods
    ABSTRACT = "abstract"       # Represents abstract ideas
    RELATIONAL = "relational"   # Represents relationships between concepts

@dataclass
class ShadowConcept:
    """Represents a single version of a shadow concept"""
    name: str
    version: float
    embedding: np.ndarray
    concept_type: ConceptType
    confidence: float
    last_updated: float
    parent_concepts: List[str]
    related_concepts: Dict[str, float]
    metadata: Dict[str, Any]

    def __lt__(self, other):
        if not isinstance(other, ShadowConcept):
            return NotImplemented
        return self.version < other.version

    def __le__(self, other):
        if not isinstance(other, ShadowConcept):
            return NotImplemented
        return self.version <= other.version

    def __gt__(self, other):
        if not isinstance(other, ShadowConcept):
            return NotImplemented
        return self.version > other.version

    def __ge__(self, other):
        if not isinstance(other, ShadowConcept):
            return NotImplemented
        return self.version >= other.version

    def __eq__(self, other):
        if not isinstance(other, ShadowConcept):
            return NotImplemented
        return self.version == other.version

class ConceptHistory:
    """Tracks the evolution of a concept over time"""
    def __init__(self, concept_name: str):
        self.concept_name = concept_name
        self.versions: List[ShadowConcept] = []
        self.performance_metrics: List[Dict[str, float]] = []
        self.refinement_history: List[Dict[str, Any]] = []

    def add_version(self, concept: ShadowConcept, metrics: Dict[str, float], 
                   refinement_info: Dict[str, Any]) -> None:
        """Add a new version with associated metrics and refinement information"""
        self.versions.append(concept)
        self.performance_metrics.append(metrics)
        self.refinement_history.append(refinement_info)

    def get_version(self, version: Optional[float] = None) -> ShadowConcept:
        """Get a specific version of the concept"""
        if not self.versions:
            raise ValueError(f"No versions exist for concept {self.concept_name}")
            
        if version is None:
            return self.versions[-1]  # Latest version
        
        # Find closest version
        closest_idx = min(range(len(self.versions)), 
                         key=lambda i: abs(self.versions[i].version - version))
        return self.versions[closest_idx]

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get complete version history with metrics and refinement info"""
        history = []
        for i in range(len(self.versions)):
            history.append({
                'version': self.versions[i],
                'metrics': self.performance_metrics[i],
                'refinement_info': self.refinement_history[i]
            })
        return history

class ShadowConceptSystem:
    """Main system for managing and refining shadow concepts"""
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.concepts: Dict[str, ConceptHistory] = {}
        self._lock = threading.Lock()
        self.logger = self._setup_logger()
        
        # Refinement parameters
        self.min_confidence_threshold = 0.3
        self.max_confidence_threshold = 0.95
        self.learning_rate = 0.1
        self.version_increment = 0.1

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the shadow concept system"""
        logger = logging.getLogger('ShadowConceptSystem')
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

    def create_concept(self, name: str, initial_embedding: np.ndarray, 
                      concept_type: ConceptType, parent_concepts: List[str] = None) -> ShadowConcept:
        """Create a new shadow concept"""
        if not self._validate_embedding(initial_embedding):
            raise ValueError(f"Invalid embedding dimensions. Expected {self.embedding_dim}")

        with self._lock:
            if name in self.concepts:
                raise ValueError(f"Concept '{name}' already exists")

            concept = ShadowConcept(
                name=name,
                version=1.0,
                embedding=initial_embedding,
                concept_type=concept_type,
                confidence=0.5,  # Initial confidence
                last_updated=time.time(),
                parent_concepts=parent_concepts or [],
                related_concepts={},
                metadata={
                    "creation_time": time.time(),
                    "refinement_count": 0,
                    "source": "initial_creation"
                }
            )

            history = ConceptHistory(name)
            history.add_version(
                concept=concept,
                metrics={"initial_confidence": 0.5},
                refinement_info={"type": "creation", "timestamp": time.time()}
            )
            
            self.concepts[name] = history
            self.logger.info(f"Created new concept: {name} (v{concept.version})")
            return concept

    def refine_concept(self, name: str, new_embedding: np.ndarray, 
                      feedback: Dict[str, float], context: Optional[Dict] = None) -> ShadowConcept:
        """Refine an existing concept based on feedback and context"""
        if not self._validate_embedding(new_embedding):
            raise ValueError(f"Invalid embedding dimensions. Expected {self.embedding_dim}")

        with self._lock:
            if name not in self.concepts:
                raise ValueError(f"Concept '{name}' does not exist")

            current = self.concepts[name].get_version()
            
            # Calculate confidence adjustment
            confidence_delta = self._calculate_confidence_adjustment(feedback)
            new_confidence = self._adjust_confidence(current.confidence, confidence_delta)
            
            # Calculate refined embedding
            refined_embedding = self._calculate_refined_embedding(
                current.embedding, new_embedding, feedback)
            
            # Update related concepts based on context
            related_concepts = self._update_related_concepts(
                current.related_concepts, context)
            
            # Create new version
            new_version = current.version + self.version_increment
            
            refined = ShadowConcept(
                name=name,
                version=new_version,
                embedding=refined_embedding,
                concept_type=current.concept_type,
                confidence=new_confidence,
                last_updated=time.time(),
                parent_concepts=current.parent_concepts,
                related_concepts=related_concepts,
                metadata={
                    "previous_version": current.version,
                    "refinement_count": current.metadata["refinement_count"] + 1,
                    "feedback": feedback,
                    "context": context
                }
            )
            
            # Update history
            self.concepts[name].add_version(
                concept=refined,
                metrics={"confidence": new_confidence, "feedback_scores": feedback},
                refinement_info={
                    "type": "refinement",
                    "timestamp": time.time(),
                    "context": context
                }
            )
            
            self.logger.info(f"Refined concept: {name} (v{new_version})")
            return refined

    def get_concept(self, name: str, version: Optional[float] = None) -> ShadowConcept:
        """Get a specific version of a concept"""
        if name not in self.concepts:
            raise ValueError(f"Concept '{name}' does not exist")
            
        return self.concepts[name].get_version(version)

    def get_concept_history(self, name: str) -> List[Dict[str, Any]]:
        """Get the complete history of a concept's evolution"""
        if name not in self.concepts:
            raise ValueError(f"Concept '{name}' does not exist")
            
        return self.concepts[name].get_version_history()

    def analyze_concept_drift(self, name: str, window_size: int = 5) -> Dict[str, float]:
        """Analyze how a concept has drifted over recent versions"""
        if name not in self.concepts:
            raise ValueError(f"Concept '{name}' does not exist")
            
        history = self.concepts[name]
        if len(history.versions) < 2:
            return {
                "drift": 0.0,
                "confidence_change": 0.0,
                "versions_analyzed": 1
            }
            
        recent_versions = history.versions[-window_size:]
        
        # Calculate embedding drift
        embedding_drifts = []
        for i in range(1, len(recent_versions)):
            prev_embedding = recent_versions[i-1].embedding
            curr_embedding = recent_versions[i].embedding
            drift = 1 - np.dot(prev_embedding, curr_embedding) / \
                   (np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding))
            embedding_drifts.append(drift)
            
        # Calculate confidence changes
        confidence_changes = [v.confidence - recent_versions[i-1].confidence 
                            for i, v in enumerate(recent_versions[1:], 1)]
        
        return {
            "drift": np.mean(embedding_drifts) if embedding_drifts else 0.0,
            "confidence_change": np.mean(confidence_changes) if confidence_changes else 0.0,
            "versions_analyzed": len(recent_versions)
        }

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding dimensions"""
        return isinstance(embedding, np.ndarray) and embedding.shape == (self.embedding_dim,)

    def _calculate_confidence_adjustment(self, feedback: Dict[str, float]) -> float:
        """Calculate confidence adjustment based on feedback"""
        weights = {
            "accuracy": 0.4,
            "consistency": 0.3,
            "relevance": 0.3
        }
        
        adjustment = 0.0
        for metric, weight in weights.items():
            if metric in feedback:
                adjustment += feedback[metric] * weight
        
        return adjustment

    def _adjust_confidence(self, current_confidence: float, adjustment: float) -> float:
        """Adjust confidence score while respecting bounds"""
        new_confidence = current_confidence + (adjustment * self.learning_rate)
        return max(self.min_confidence_threshold, 
                  min(self.max_confidence_threshold, new_confidence))

    def _calculate_refined_embedding(self, current_embedding: np.ndarray, 
                                   new_embedding: np.ndarray, 
                                   feedback: Dict[str, float]) -> np.ndarray:
        """Calculate refined embedding based on current embedding and feedback"""
        # Calculate adaptive learning rate based on feedback
        feedback_strength = np.mean(list(feedback.values()))
        adaptive_lr = self.learning_rate * feedback_strength
        
        # Weighted average of current and new embeddings
        refined = (1 - adaptive_lr) * current_embedding + adaptive_lr * new_embedding
        
        # Normalize the embedding
        return refined / np.linalg.norm(refined)

    def _update_related_concepts(self, current_relations: Dict[str, float], 
                               context: Optional[Dict]) -> Dict[str, float]:
        """Update related concepts based on context"""
        if not context or 'related_concepts' not in context:
            return current_relations.copy()
        
        updated_relations = current_relations.copy()
        
        # Update relationship strengths
        for concept, strength in context['related_concepts'].items():
            if concept in updated_relations:
                # Smooth update of existing relationships
                updated_relations[concept] = (updated_relations[concept] + strength) / 2
            else:
                # Add new relationships
                updated_relations[concept] = strength
        
        return updated_relations

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_concepts': len(self.concepts),
            'total_versions': sum(len(h.versions) for h in self.concepts.values()),
            'average_confidence': np.mean([h.get_version().confidence 
                                         for h in self.concepts.values()]),
            'concept_types': {ct.value: sum(1 for h in self.concepts.values() 
                                          if h.get_version().concept_type == ct)
                            for ct in ConceptType}
        }
