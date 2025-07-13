"""
Intelligent Domain Router - Ultimate MoE Solution
Advanced router with multiple routing strategies and intelligent expert selection
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass, field

@dataclass
class RoutingResult:
    """Result from domain routing"""
    expert_weights: Dict[str, float]
    primary_domain: str
    confidence: float
    routing_strategy: str
    metadata: Dict[str, Any]
    domain_weights: Dict[str, float] = field(default_factory=dict)

class KeywordBasedRouter:
    """Keyword-based routing strategy"""
    
    def __init__(self):
        self.domain_keywords = {
            "ecommerce": ["product", "price", "shipping", "payment", "order", "cart", "inventory", "customer", "store"],
            "banking": ["account", "balance", "transaction", "loan", "credit", "debit", "interest", "bank", "financial"],
            "insurance": ["policy", "coverage", "claim", "premium", "deductible", "risk", "liability", "insurance"],
            "healthcare": ["patient", "diagnosis", "treatment", "medication", "symptoms", "doctor", "hospital", "medical"],
            "legal": ["contract", "law", "legal", "court", "judgment", "attorney", "clause", "legal", "regulation"],
            "finance": ["investment", "portfolio", "market", "stock", "bond", "dividend", "capital", "trading"],
            "technology": ["software", "hardware", "algorithm", "database", "api", "cloud", "security", "tech"],
            "education": ["student", "teacher", "course", "curriculum", "learning", "assessment", "grade", "education"],
            "government": ["policy", "regulation", "government", "agency", "compliance", "legislation", "official"],
            "media": ["content", "publishing", "broadcast", "journalism", "editorial", "coverage", "story", "media"]
        }
        
        # Calculate keyword weights based on domain specificity
        self.keyword_weights = {}
        for domain, keywords in self.domain_keywords.items():
            self.keyword_weights[domain] = {kw: 1.0 for kw in keywords}
    
    def route(self, text: str) -> Dict[str, float]:
        """Route text based on keyword matching"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += self.keyword_weights[domain].get(keyword, 1.0)
            
            # Normalize by text length and keyword count
            domain_scores[domain] = score / (len(text.split()) + 1)
        
        # Normalize scores to sum to 1
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v / total_score for k, v in domain_scores.items()}
        else:
            # Equal distribution if no keywords found
            domain_scores = {k: 1.0 / len(domain_scores) for k in domain_scores.keys()}
        
        return domain_scores

class SemanticBasedRouter:
    """Semantic-based routing using embeddings"""
    
    def __init__(self):
        self.domain_embeddings = {
            "ecommerce": "online shopping retail product purchase",
            "banking": "financial account money transaction banking",
            "insurance": "coverage policy claim protection risk",
            "healthcare": "medical patient treatment health care",
            "legal": "law legal contract court attorney",
            "finance": "investment market portfolio financial trading",
            "technology": "software hardware tech computer digital",
            "education": "learning student teacher course education",
            "government": "policy regulation government official agency",
            "media": "content publishing journalism news media"
        }
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """Fit the vectorizer with domain descriptions"""
        domain_texts = list(self.domain_embeddings.values())
        self.vectorizer.fit(domain_texts)
    
    def route(self, text: str) -> Dict[str, float]:
        """Route text based on semantic similarity"""
        # Transform text and domain descriptions
        text_vector = self.vectorizer.transform([text])
        domain_vectors = self.vectorizer.transform(list(self.domain_embeddings.values()))
        
        # Calculate similarities
        similarities = cosine_similarity(text_vector, domain_vectors)[0]
        
        # Create domain scores
        domain_scores = {}
        for i, domain in enumerate(self.domain_embeddings.keys()):
            domain_scores[domain] = max(0, similarities[i])  # Ensure non-negative
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v / total_score for k, v in domain_scores.items()}
        else:
            domain_scores = {k: 1.0 / len(domain_scores) for k in domain_scores.keys()}
        
        return domain_scores

class MachineLearningRouter:
    """Machine learning-based routing"""
    
    def __init__(self):
        # Simple rule-based ML router (can be enhanced with actual ML models)
        self.domain_patterns = {
            "ecommerce": [
                r"\$\d+",  # Price patterns
                r"buy|purchase|order|cart|checkout",
                r"product|item|goods|merchandise"
            ],
            "banking": [
                r"account|balance|transaction",
                r"loan|credit|debit|interest",
                r"bank|financial|money"
            ],
            "insurance": [
                r"policy|coverage|claim",
                r"premium|deductible|risk",
                r"insurance|liability|protection"
            ],
            "healthcare": [
                r"patient|diagnosis|treatment",
                r"medication|symptoms|doctor",
                r"hospital|medical|health"
            ],
            "legal": [
                r"contract|law|legal",
                r"court|judgment|attorney",
                r"clause|regulation|compliance"
            ],
            "finance": [
                r"investment|portfolio|market",
                r"stock|bond|dividend|capital",
                r"trading|financial|returns"
            ],
            "technology": [
                r"software|hardware|algorithm",
                r"database|api|cloud|security",
                r"tech|digital|computer"
            ],
            "education": [
                r"student|teacher|course",
                r"curriculum|learning|assessment",
                r"grade|education|academic"
            ],
            "government": [
                r"policy|regulation|government",
                r"agency|compliance|legislation",
                r"official|public|administrative"
            ],
            "media": [
                r"content|publishing|broadcast",
                r"journalism|editorial|coverage",
                r"story|news|media"
            ]
        }
    
    def route(self, text: str) -> Dict[str, float]:
        """Route text using ML patterns"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            domain_scores[domain] = score
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v / total_score for k, v in domain_scores.items()}
        else:
            domain_scores = {k: 1.0 / len(domain_scores) for k in domain_scores.keys()}
        
        return domain_scores

class HybridRouter:
    """Hybrid router combining multiple strategies"""
    
    def __init__(self):
        self.keyword_router = KeywordBasedRouter()
        self.semantic_router = SemanticBasedRouter()
        self.ml_router = MachineLearningRouter()
        
        # Strategy weights (can be learned/optimized)
        self.strategy_weights = {
            "keyword": 0.4,
            "semantic": 0.4,
            "ml": 0.2
        }
    
    def combine_weights(self, weight_lists: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine weights from multiple strategies"""
        combined_weights = {}
        
        for domain in weight_lists[0].keys():
            weighted_sum = 0
            for i, weights in enumerate(weight_lists):
                strategy_name = list(self.strategy_weights.keys())[i]
                weighted_sum += weights.get(domain, 0) * self.strategy_weights[strategy_name]
            
            combined_weights[domain] = weighted_sum
        
        # Normalize
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}
        
        return combined_weights

class AdvancedDomainDetector:
    """Advanced domain detection with context analysis"""
    
    def __init__(self):
        self.domain_indicators = {
            "ecommerce": {
                "strong": ["shopping cart", "checkout", "add to cart", "product page"],
                "medium": ["price", "product", "customer", "order"],
                "weak": ["buy", "purchase", "store"]
            },
            "banking": {
                "strong": ["bank account", "transaction history", "loan application"],
                "medium": ["balance", "credit", "debit", "interest"],
                "weak": ["money", "financial", "account"]
            }
            # Add more domains as needed
        }
    
    def detect_domain(self, text: str, context: str = "") -> Dict[str, float]:
        """Detect domain with confidence levels"""
        text_lower = text.lower()
        context_lower = context.lower()
        combined_text = f"{text_lower} {context_lower}"
        
        domain_scores = {}
        
        for domain, indicators in self.domain_indicators.items():
            score = 0
            
            # Strong indicators (weight: 3)
            for indicator in indicators["strong"]:
                if indicator in combined_text:
                    score += 3
            
            # Medium indicators (weight: 2)
            for indicator in indicators["medium"]:
                if indicator in combined_text:
                    score += 2
            
            # Weak indicators (weight: 1)
            for indicator in indicators["weak"]:
                if indicator in combined_text:
                    score += 1
            
            domain_scores[domain] = score
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v / total_score for k, v in domain_scores.items()}
        else:
            domain_scores = {k: 1.0 / len(domain_scores) for k in domain_scores.keys()}
        
        return domain_scores

class ContentAnalyzer:
    """Content analysis for routing decisions"""
    
    def __init__(self):
        self.content_types = {
            "technical": ["algorithm", "implementation", "code", "architecture", "system"],
            "business": ["strategy", "market", "revenue", "customer", "business"],
            "academic": ["research", "study", "analysis", "methodology", "findings"],
            "casual": ["chat", "conversation", "discussion", "opinion", "personal"]
        }
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content characteristics"""
        text_lower = text.lower()
        
        analysis = {
            "content_type": {},
            "complexity": self._calculate_complexity(text),
            "length": len(text.split()),
            "technical_terms": 0,
            "domain_specific_terms": 0
        }
        
        # Determine content type
        for content_type, keywords in self.content_types.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            analysis["content_type"][content_type] = score
        
        # Count technical terms
        technical_terms = ["algorithm", "implementation", "system", "architecture", "framework"]
        analysis["technical_terms"] = sum(text_lower.count(term) for term in technical_terms)
        
        return analysis
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity"""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity based on word length and sentence structure
        avg_word_length = np.mean([len(word) for word in words])
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        complexity = (avg_word_length * 0.3) + (avg_sentence_length * 0.7)
        return min(1.0, complexity / 20.0)  # Normalize to 0-1

class EntityExtractor:
    """Entity extraction for routing"""
    
    def __init__(self):
        self.entity_patterns = {
            "person": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "organization": r"\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Organization)\b",
            "location": r"\b[A-Z][a-z]+, [A-Z]{2}\b",
            "date": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "money": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "percentage": r"\d+(?:\.\d+)?%"
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = matches
        
        return entities

class IntelligentExpertSelector:
    """Intelligent expert selection based on multiple factors"""
    
    def __init__(self):
        self.expert_performance_history = {}
        self.expert_specializations = {
            "EcommerceExpert": ["ecommerce", "retail", "shopping"],
            "BankingExpert": ["banking", "finance", "financial"],
            "InsuranceExpert": ["insurance", "risk", "coverage"],
            "HealthcareExpert": ["healthcare", "medical", "health"],
            "LegalExpert": ["legal", "law", "contract"],
            "FinanceExpert": ["finance", "investment", "trading"],
            "TechnologyExpert": ["technology", "tech", "software"],
            "EducationExpert": ["education", "learning", "academic"],
            "GovernmentExpert": ["government", "policy", "regulation"],
            "MediaExpert": ["media", "content", "publishing"]
        }
    
    def select_experts(self, domain_weights: Dict[str, float], 
                      content_analysis: Dict[str, Any],
                      performance_threshold: float = 0.8) -> Dict[str, float]:
        """Select experts based on domain weights and content analysis"""
        
        expert_weights = {}
        
        for domain, weight in domain_weights.items():
            # Find experts for this domain
            domain_experts = [expert for expert, specializations in self.expert_specializations.items()
                            if domain in specializations]
            
            # Distribute weight among domain experts
            for expert in domain_experts:
                performance_score = self.expert_performance_history.get(expert, 0.8)
                
                if performance_score >= performance_threshold:
                    expert_weights[expert] = weight / len(domain_experts) * performance_score
        
        # Normalize expert weights
        total_weight = sum(expert_weights.values())
        if total_weight > 0:
            expert_weights = {k: v / total_weight for k, v in expert_weights.items()}
        
        return expert_weights

class AdaptiveLoadBalancer:
    """Adaptive load balancing for expert selection"""
    
    def __init__(self):
        self.expert_loads = {}
        self.expert_response_times = {}
        self.max_load_threshold = 0.9
    
    def balance_load(self, expert_weights: Dict[str, float]) -> Dict[str, float]:
        """Balance load across experts"""
        balanced_weights = {}
        
        for expert, weight in expert_weights.items():
            current_load = self.expert_loads.get(expert, 0.0)
            
            # Reduce weight if expert is overloaded
            if current_load > self.max_load_threshold:
                reduction_factor = 1.0 - (current_load - self.max_load_threshold)
                weight *= max(0.1, reduction_factor)
            
            balanced_weights[expert] = weight
        
        # Normalize weights
        total_weight = sum(balanced_weights.values())
        if total_weight > 0:
            balanced_weights = {k: v / total_weight for k, v in balanced_weights.items()}
        
        return balanced_weights
    
    def update_load(self, expert: str, load: float):
        """Update expert load"""
        self.expert_loads[expert] = load
    
    def update_response_time(self, expert: str, response_time: float):
        """Update expert response time"""
        self.expert_response_times[expert] = response_time

class PerformanceOptimizer:
    """Performance optimization for routing decisions"""
    
    def __init__(self):
        self.performance_history = {}
        self.optimization_strategies = {
            "latency": self._optimize_for_latency,
            "accuracy": self._optimize_for_accuracy,
            "throughput": self._optimize_for_throughput
        }
    
    def optimize(self, weights: Dict[str, float], 
                optimization_target: str = "accuracy") -> Dict[str, float]:
        """Optimize weights based on performance target"""
        
        if optimization_target in self.optimization_strategies:
            return self.optimization_strategies[optimization_target](weights)
        
        return weights
    
    def _optimize_for_latency(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Optimize for low latency"""
        # Prefer experts with lower response times
        optimized_weights = {}
        
        for expert, weight in weights.items():
            avg_response_time = self.performance_history.get(f"{expert}_response_time", 100)
            latency_factor = max(0.1, 1.0 - (avg_response_time / 1000))  # Normalize to 0-1
            optimized_weights[expert] = weight * latency_factor
        
        # Normalize
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        
        return optimized_weights
    
    def _optimize_for_accuracy(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Optimize for high accuracy"""
        # Prefer experts with higher accuracy
        optimized_weights = {}
        
        for expert, weight in weights.items():
            accuracy = self.performance_history.get(f"{expert}_accuracy", 0.8)
            optimized_weights[expert] = weight * accuracy
        
        # Normalize
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        
        return optimized_weights
    
    def _optimize_for_throughput(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Optimize for high throughput"""
        # Prefer experts with higher throughput
        optimized_weights = {}
        
        for expert, weight in weights.items():
            throughput = self.performance_history.get(f"{expert}_throughput", 100)
            throughput_factor = min(2.0, throughput / 100)  # Cap at 2x
            optimized_weights[expert] = weight * throughput_factor
        
        # Normalize
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        
        return optimized_weights

class IntelligentDomainRouter:
    """Advanced router with multiple routing strategies"""
    
    def __init__(self):
        # Routing Strategies
        self.keyword_router = KeywordBasedRouter()
        self.semantic_router = SemanticBasedRouter()
        self.ml_router = MachineLearningRouter()
        self.hybrid_router = HybridRouter()
        
        # Domain Detection
        self.domain_detector = AdvancedDomainDetector()
        self.content_analyzer = ContentAnalyzer()
        self.entity_extractor = EntityExtractor()
        
        # Expert Selection
        self.expert_selector = IntelligentExpertSelector()
        self.load_balancer = AdaptiveLoadBalancer()
        self.performance_optimizer = PerformanceOptimizer()
    
    async def route_to_experts(self, text: str, context: str = "") -> RoutingResult:
        """Advanced routing with multiple strategies"""
        
        # Multiple routing approaches
        keyword_weights = self.keyword_router.route(text)
        semantic_weights = self.semantic_router.route(text)
        ml_weights = self.ml_router.route(text)
        
        # Hybrid decision
        final_weights = self.hybrid_router.combine_weights([
            keyword_weights, semantic_weights, ml_weights
        ])
        
        # Content analysis
        content_analysis = self.content_analyzer.analyze_content(text)
        entity_analysis = self.entity_extractor.extract_entities(text)
        
        # Expert selection
        expert_weights = self.expert_selector.select_experts(
            final_weights, content_analysis
        )
        
        # Load balancing
        balanced_weights = self.load_balancer.balance_load(expert_weights)
        
        # Performance optimization
        optimized_weights = self.performance_optimizer.optimize(
            balanced_weights, optimization_target="accuracy"
        )
        
        # Determine primary domain
        primary_domain = max(final_weights.items(), key=lambda x: x[1])[0]
        
        # Calculate overall confidence
        confidence = np.mean(list(final_weights.values()))
        
        return RoutingResult(
            expert_weights=optimized_weights,
            primary_domain=primary_domain,
            confidence=confidence,
            routing_strategy="hybrid",
            metadata={
                "keyword_weights": keyword_weights,
                "semantic_weights": semantic_weights,
                "ml_weights": ml_weights,
                "content_analysis": content_analysis,
                "entity_analysis": entity_analysis,
                "expert_weights": expert_weights,
                "balanced_weights": balanced_weights
            },
            domain_weights=final_weights
        ) 