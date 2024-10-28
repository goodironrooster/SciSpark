from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import time
import threading

class FailureType(Enum):
    FACTUAL_INACCURACY = "factual_inaccuracy"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    OFF_TOPIC = "off_topic"
    NOVEL_OUTPUT = "novel_output"

@dataclass
class FailureVector:
    factual_score: float  # 0-1, higher means more accurate
    logical_score: float  # 0-1, higher means more consistent
    toxicity_score: float  # 0-1, higher means more toxic
    hallucination_score: float  # 0-1, higher means more hallucinated
    off_topic_score: float  # 0-1, higher means more off-topic
    novelty_score: float  # 0-1, higher means more novel
    timestamp: float
    metadata: Dict[str, Any]

class FactChecker:
    def __init__(self):
        # Define known facts with their truth values
        self.knowledge_base = {
            # Mathematics
            "2 + 2 = 4": 0.9,
            "basic arithmetic": 0.9,
            "mathematics": 0.9,
            # Earth and Space
            "earth is round": 0.9,
            "earth is flat": 0.1,
            "earth is a planet": 0.9,
            "earth orbits sun": 0.9,
            "sun is a star": 0.9,
            # General Science
            "sky is blue": 0.9,
            "sky is green": 0.1,
            "water is wet": 0.9,
            "gravity exists": 0.9
        }

    def extract_claims(self, text: str) -> List[str]:
        if not text:
            return []
            
        # Split by sentence endings and conjunctions
        separators = ['. ', '! ', '? ', '; ', ' because ', ' but ', ' however ']
        claims = [text.lower()]
        for sep in separators:
            new_claims = []
            for claim in claims:
                new_claims.extend([c.strip() for c in claim.split(sep) if c.strip()])
            claims = new_claims
        return claims

    def verify_claim(self, claim: str, context: Optional[Dict] = None) -> float:
        if not claim:
            return 0.5
        
        claim = claim.lower()
        scores = []
        
        # Check against knowledge base
        for fact, score in self.knowledge_base.items():
            if fact in claim:
                scores.append(score)
        
        # Check against context if provided
        if context and 'known_facts' in context:
            for fact in context['known_facts']:
                fact = fact.lower()
                if fact in claim:
                    scores.append(0.9)
        
        return np.mean(scores) if scores else 0.5

class LogicAnalyzer:
    def __init__(self):
        self.contradiction_patterns = [
            ("is true", "is false"),
            ("all", "none"),
            ("always", "never"),
            ("can", "cannot"),
            ("must", "must not"),
            ("is", "is not")
        ]
        
        self.logical_patterns = [
            ("if", "then"),
            ("because", "therefore"),
            ("since", "thus"),
            ("given", "hence"),
            ("implies", "follows")
        ]

    def extract_statements(self, text: str) -> List[str]:
        if not text:
            return []
        
        connectors = ['. ', ', ', ' therefore ', ' thus ', ' hence ', ' so ']
        statements = [text.lower()]
        for conn in connectors:
            new_statements = []
            for stmt in statements:
                new_statements.extend([s.strip() for s in stmt.split(conn) if s.strip()])
            statements = new_statements
        return statements

    def check_consistency(self, stmt1: str, stmt2: str) -> float:
        if not stmt1 or not stmt2:
            return 0.5
            
        stmt1, stmt2 = stmt1.lower(), stmt2.lower()
        
        # Check for contradictions
        for pattern in self.contradiction_patterns:
            if (pattern[0] in stmt1 and pattern[1] in stmt2) or \
               (pattern[1] in stmt1 and pattern[0] in stmt2):
                return 0.1
        
        # Check for logical connections
        for pattern in self.logical_patterns:
            if pattern[0] in stmt1 and pattern[1] in stmt2:
                return 0.9
        
        # Check for repeated elements (potential consistency)
        words1 = set(stmt1.split())
        words2 = set(stmt2.split())
        if words1.intersection(words2):
            return 0.8
            
        return 0.7

class ToxicityAnalyzer:
    def __init__(self):
        self.toxic_words = {
            'hate': 0.8,
            'idiot': 0.7,
            'stupid': 0.7,
            'terrible': 0.6,
            'worst': 0.6,
            'awful': 0.6,
            'horrible': 0.6,
            'useless': 0.5,
            'pathetic': 0.7,
            'destroy': 0.5
        }

    def analyze(self, text: str) -> float:
        if not text:
            return 0.0
            
        text = text.lower()
        scores = []
        
        for word, score in self.toxic_words.items():
            if word in text:
                scores.append(score)
        
        if scores:
            return min(1.0, sum(scores) / len(scores) + 0.1 * (len(scores) - 1))
        return 0.0

class TopicAnalyzer:
    def __init__(self):
        self.topic_terms = {
            'space exploration': {
                'space': 0.4, 'rocket': 0.4, 'nasa': 0.4, 'mars': 0.4,
                'astronaut': 0.4, 'mission': 0.3, 'satellite': 0.3,
                'orbit': 0.3, 'launch': 0.3, 'galaxy': 0.3
            }
        }

    def analyze_relevance(self, text: str, context: Optional[Dict]) -> float:
        if not text or not context or 'topic' not in context:
            return 0.5
            
        topic = context['topic'].lower()
        text = text.lower()
        
        if topic in self.topic_terms:
            terms = self.topic_terms[topic]
            scores = []
            
            for term, weight in terms.items():
                if term in text:
                    scores.append(weight)
            
            if scores:
                relevance = min(1.0, sum(scores))
                return 1.0 - relevance  # Convert to off-topic score
        
        # Check basic word overlap
        topic_words = set(topic.split())
        text_words = set(text.split())
        if topic_words.intersection(text_words):
            return 0.5
            
        return 0.9  # Highly off-topic

class NoveltyAnalyzer:
    def __init__(self):
        self.common_phrases = {
            'standard response': 0.1,
            'hello world': 0.1,
            'common knowledge': 0.2,
            'basic concept': 0.2,
            'typical example': 0.2,
            'regular pattern': 0.2,
            'normal behavior': 0.2
        }

    def analyze(self, text: str) -> float:
        if not text:
            return 0.0
            
        text = text.lower()
        
        # Check for common phrases
        for phrase, score in self.common_phrases.items():
            if phrase in text:
                return score
        
        # Calculate novelty based on text properties
        words = text.split()
        if not words:
            return 0.0
            
        unique_words = len(set(words))
        total_words = len(words)
        
        # Base score from vocabulary diversity
        diversity = unique_words / total_words
        
        # Length bonus
        length_bonus = min(0.3, total_words / 50)
        
        # Complexity bonus
        avg_word_length = sum(len(word) for word in words) / total_words
        complexity_bonus = min(0.2, (avg_word_length - 4) / 10)
        
        return min(1.0, diversity + length_bonus + complexity_bonus)

class FailureDetector:
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self._lock = threading.Lock()
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.failure_thresholds = {
            FailureType.FACTUAL_INACCURACY: 0.7,
            FailureType.LOGICAL_INCONSISTENCY: 0.6,
            FailureType.TOXICITY: 0.3,
            FailureType.HALLUCINATION: 0.5,
            FailureType.OFF_TOPIC: 0.6,
            FailureType.NOVEL_OUTPUT: 0.8
        }
        self._initialize_detection_system()

    def _initialize_detection_system(self) -> None:
        try:
            self.fact_checker = FactChecker()
            self.logic_analyzer = LogicAnalyzer()
            self.toxicity_analyzer = ToxicityAnalyzer()
            self.topic_analyzer = TopicAnalyzer()
            self.novelty_analyzer = NoveltyAnalyzer()
            self.debug_print("Failure Detection System initialized")
        except Exception as e:
            self.debug_print(f"Error initializing detection system: {str(e)}", "ERROR")
            raise

    def detect_failures(self, content: str, context: Optional[Dict] = None) -> FailureVector:
        try:
            if content is None:
                raise ValueError("Content cannot be None")
                
            start_time = time.perf_counter()

            with self._lock:
                # Analyze all aspects
                factual_score = self._analyze_factual_accuracy(content, context)
                logical_score = self._analyze_logical_consistency(content)
                toxicity_score = self._analyze_toxicity(content)
                hallucination_score = self._analyze_hallucination(content, context)
                off_topic_score = self._analyze_topic_relevance(content, context)
                novelty_score = self._analyze_novelty(content)

                vector = FailureVector(
                    factual_score=factual_score,
                    logical_score=logical_score,
                    toxicity_score=toxicity_score,
                    hallucination_score=hallucination_score,
                    off_topic_score=off_topic_score,
                    novelty_score=novelty_score,
                    timestamp=time.time(),
                    metadata=self._generate_metadata(content, context)
                )

            elapsed_time = time.perf_counter() - start_time
            self.metrics['detection_time'].append(elapsed_time)
            self.debug_print(f"Failure detection completed in {elapsed_time:.4f}s")
            
            return vector

        except Exception as e:
            self.debug_print(f"Error in failure detection: {str(e)}", "ERROR")
            raise

    def _analyze_factual_accuracy(self, content: str, context: Optional[Dict]) -> float:
        try:
            claims = self.fact_checker.extract_claims(content)
            accuracy_scores = []
            for claim in claims:
                score = self.fact_checker.verify_claim(claim, context)
                accuracy_scores.append(score)
            return np.mean(accuracy_scores) if accuracy_scores else 0.5
        except Exception as e:
            self.debug_print(f"Error in factual analysis: {str(e)}", "ERROR")
            return 0.5

    def _analyze_logical_consistency(self, content: str) -> float:
        try:
            statements = self.logic_analyzer.extract_statements(content)
            consistency_scores = []
            for i in range(len(statements)):
                for j in range(i + 1, len(statements)):
                    score = self.logic_analyzer.check_consistency(
                        statements[i], statements[j]
                    )
                    consistency_scores.append(score)
            return np.mean(consistency_scores) if consistency_scores else 0.7
        except Exception as e:
            self.debug_print(f"Error in logical analysis: {str(e)}", "ERROR")
            return 0.5

    def _analyze_toxicity(self, content: str) -> float:
        try:
            return self.toxicity_analyzer.analyze(content)
        except Exception as e:
            self.debug_print(f"Error in toxicity analysis: {str(e)}", "ERROR")
            return 0.0

    def _analyze_hallucination(self, content: str, context: Optional[Dict]) -> float:
        try:
            claims = self.fact_checker.extract_claims(content)
            hallucination_scores = []
            
            # Build known facts set
            known_facts = set()
            if context and 'known_facts' in context:
                known_facts.update(fact.lower() for fact in context['known_facts'])
            
            for claim in claims:
                claim = claim.lower()
                factual_score = self.fact_checker.verify_claim(claim, context)
                
                # Increase hallucination score for claims with fantasy elements
                fantasy_terms = {'dragon', 'magic', 'unicorn', 'fairy', 'wizard', 'supernatural'}
                if any(term in claim for term in fantasy_terms):
                    hallucination_scores.append(0.9)
                else:
                    # Invert and adjust the factual score
                    hallucination_score = 1.0 - factual_score
                    if hallucination_score > 0.5:  # For claims that are likely false
                        hallucination_score = min(0.9, hallucination_score * 1.2)
                    hallucination_scores.append(hallucination_score)
            
            return np.mean(hallucination_scores) if hallucination_scores else 0.5
            
        except Exception as e:
            self.debug_print(f"Error in hallucination analysis: {str(e)}", "ERROR")
            return 0.5

    def _analyze_topic_relevance(self, content: str, context: Optional[Dict]) -> float:
        try:
            return self.topic_analyzer.analyze_relevance(content, context)
        except Exception as e:
            self.debug_print(f"Error in topic analysis: {str(e)}", "ERROR")
            return 0.5

    def _analyze_novelty(self, content: str) -> float:
        try:
            return self.novelty_analyzer.analyze(content)
        except Exception as e:
            self.debug_print(f"Error in novelty analysis: {str(e)}", "ERROR")
            return 0.5

    def _generate_metadata(self, content: str, context: Optional[Dict]) -> Dict:
        return {
            'content_length': len(content),
            'context_provided': context is not None,
            'timestamp': time.time(),
            'analysis_version': '1.0'
        }

    def debug_print(self, message: str, level: str = "INFO") -> None:
        if self.debug_mode:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] DEBUG FailureDetector [{level}]: {message}")
