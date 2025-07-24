"""
Enhanced Dataset Profiler - Ultimate MoE Solution
Advanced dataset profiling with quality metrics and domain analysis
"""

import asyncio
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import logging
from datetime import datetime

@dataclass
class TextProfile:
    """Comprehensive text profile with all metrics"""
    quality_score: float
    complexity_analysis: Dict[str, Any]
    domain_indicators: Dict[str, float]
    data_quality_metrics: Dict[str, Any]
    readability_scores: Dict[str, float]
    language_analysis: Dict[str, Any]
    metadata: Dict[str, Any]

class EnhancedDatasetProfiler:
    """Enhanced dataset profiling with advanced features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Domain-specific keywords for analysis
        self.domain_keywords = {
            "ecommerce": ["product", "price", "shipping", "payment", "order", "cart", "customer", "store", "inventory"],
            "banking": ["account", "balance", "transaction", "loan", "credit", "debit", "interest", "bank", "financial"],
            "insurance": ["policy", "coverage", "claim", "premium", "deductible", "risk", "liability", "insurance"],
            "healthcare": ["patient", "diagnosis", "treatment", "medication", "symptoms", "doctor", "hospital", "medical"],
            "legal": ["contract", "law", "legal", "court", "judgment", "attorney", "clause", "regulation"],
            "finance": ["investment", "portfolio", "market", "stock", "bond", "dividend", "capital", "trading"],
            "technology": ["software", "hardware", "algorithm", "database", "api", "cloud", "security", "tech"],
            "education": ["student", "teacher", "course", "curriculum", "learning", "assessment", "grade", "education"],
            "government": ["policy", "regulation", "government", "agency", "compliance", "legislation", "official"],
            "media": ["content", "publishing", "broadcast", "journalism", "editorial", "coverage", "story", "media"]
        }
        
        # Quality indicators
        self.quality_indicators = {
            "positive": ["accurate", "verified", "confirmed", "reliable", "trusted", "valid", "correct"],
            "negative": ["uncertain", "doubtful", "speculative", "unverified", "questionable", "suspicious"],
            "formal": ["therefore", "consequently", "furthermore", "moreover", "additionally"],
            "informal": ["like", "you know", "basically", "actually", "literally"]
        }
        
        # Readability formulas
        self.readability_formulas = {
            "flesch_reading_ease": self._calculate_flesch_reading_ease,
            "gunning_fog": self._calculate_gunning_fog,
            "smog": self._calculate_smog_index,
            "coleman_liau": self._calculate_coleman_liau
        }
    
    async def profile_text(self, text: str) -> TextProfile:
        """Advanced text profiling with quality metrics"""
        
        try:
            # Parallel processing of different analysis components
            tasks = [
                self._analyze_complexity(text),
                self._detect_domain_indicators(text),
                self._calculate_data_quality_metrics(text),
                self._analyze_readability(text),
                self._analyze_language(text)
            ]
            
            results = await asyncio.gather(*tasks)
            
            complexity_analysis, domain_indicators, data_quality_metrics, readability_scores, language_analysis = results
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(
                complexity_analysis, domain_indicators, data_quality_metrics, readability_scores
            )
            
            return TextProfile(
                quality_score=quality_score,
                complexity_analysis=complexity_analysis,
                domain_indicators=domain_indicators,
                data_quality_metrics=data_quality_metrics,
                readability_scores=readability_scores,
                language_analysis=language_analysis,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "sentence_count": len(text.split('.')),
                    "analysis_version": "1.0"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error profiling text: {e}")
            return self._create_fallback_profile(text)
    
    async def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return {"complexity_score": 0.0, "word_complexity": 0.0, "sentence_complexity": 0.0}
        
        # Word-level complexity
        avg_word_length = np.mean([len(word) for word in words])
        long_words = len([word for word in words if len(word) > 6])
        word_complexity = long_words / len(words)
        
        # Sentence-level complexity
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        sentence_complexity = min(1.0, avg_sentence_length / 20.0)  # Normalize
        
        # Overall complexity score
        complexity_score = (word_complexity * 0.4) + (sentence_complexity * 0.6)
        
        return {
            "complexity_score": complexity_score,
            "word_complexity": word_complexity,
            "sentence_complexity": sentence_complexity,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "long_word_ratio": word_complexity,
            "sentence_count": len(sentences),
            "word_count": len(words)
        }
    
    async def _detect_domain_indicators(self, text: str) -> Dict[str, float]:
        """Detect domain indicators in text"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            # Normalize by number of keywords and text length
            normalized_score = score / (len(keywords) * (len(text.split()) / 100 + 1))
            domain_scores[domain] = min(1.0, normalized_score)
        
        return domain_scores
    
    async def _calculate_data_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        text_lower = text.lower()
        
        # Quality indicators analysis
        positive_count = sum(text_lower.count(indicator) for indicator in self.quality_indicators["positive"])
        negative_count = sum(text_lower.count(indicator) for indicator in self.quality_indicators["negative"])
        formal_count = sum(text_lower.count(indicator) for indicator in self.quality_indicators["formal"])
        informal_count = sum(text_lower.count(indicator) for indicator in self.quality_indicators["informal"])
        
        # Calculate quality ratios
        total_indicators = positive_count + negative_count + formal_count + informal_count
        if total_indicators > 0:
            positive_ratio = positive_count / total_indicators
            negative_ratio = negative_count / total_indicators
            formal_ratio = formal_count / total_indicators
            informal_ratio = informal_count / total_indicators
        else:
            positive_ratio = negative_ratio = formal_ratio = informal_ratio = 0.0
        
        # Consistency analysis
        word_frequency = Counter(text.split())
        unique_words = len(word_frequency)
        total_words = len(text.split())
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Repetition analysis
        repeated_words = sum(1 for count in word_frequency.values() if count > 2)
        repetition_ratio = repeated_words / unique_words if unique_words > 0 else 0
        
        return {
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "formal_indicators": formal_count,
            "informal_indicators": informal_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "formal_ratio": formal_ratio,
            "informal_ratio": informal_ratio,
            "vocabulary_richness": vocabulary_richness,
            "repetition_ratio": repetition_ratio,
            "unique_words": unique_words,
            "total_words": total_words,
            "quality_score": positive_ratio - negative_ratio + formal_ratio - informal_ratio
        }
    
    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability using multiple formulas"""
        readability_scores = {}
        
        for formula_name, formula_func in self.readability_formulas.items():
            try:
                score = formula_func(text)
                readability_scores[formula_name] = score
            except Exception as e:
                self.logger.warning(f"Error calculating {formula_name}: {e}")
                readability_scores[formula_name] = 0.0
        
        return readability_scores
    
    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        syllables = self._count_syllables(text)
        
        if not sentences or not words:
            return 0.0
        
        # Flesch formula: 206.835 - 1.015 × (total words ÷ total sentences) - 84.6 × (total syllables ÷ total words)
        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        return max(0.0, min(100.0, score))
    
    def _calculate_gunning_fog(self, text: str) -> float:
        """Calculate Gunning Fog Index"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        complex_words = len([word for word in words if len(word) > 6])
        
        if not sentences or not words:
            return 0.0
        
        # Gunning Fog formula: 0.4 × ((words ÷ sentences) + 100 × (complex words ÷ words))
        score = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_words / len(words)))
        return max(0.0, score)
    
    def _calculate_smog_index(self, text: str) -> float:
        """Calculate SMOG Index"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        complex_words = len([word for word in text.split() if len(word) > 2])
        
        if not sentences:
            return 0.0
        
        # SMOG formula: 1.043 × √(complex words × (30 ÷ sentences)) + 3.1291
        score = 1.043 * np.sqrt(complex_words * (30 / len(sentences))) + 3.1291
        return max(0.0, score)
    
    def _calculate_coleman_liau(self, text: str) -> float:
        """Calculate Coleman-Liau Index"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        letters = sum(len(word) for word in words)
        
        if not sentences or not words:
            return 0.0
        
        # Coleman-Liau formula: 0.0588 × (letters ÷ words × 100) - 0.296 × (sentences ÷ words × 100) - 15.8
        L = letters / len(words) * 100
        S = len(sentences) / len(words) * 100
        score = 0.0588 * L - 0.296 * S - 15.8
        return max(0.0, score)
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified approach)"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    async def _analyze_language(self, text: str) -> Dict[str, Any]:
        """Analyze language characteristics"""
        words = text.split()
        
        # Language patterns
        has_numbers = bool(re.search(r'\d', text))
        has_special_chars = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', text))
        has_urls = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        has_emails = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        # Capitalization analysis
        capitalized_words = len([word for word in words if word and word[0].isupper()])
        all_caps_words = len([word for word in words if word.isupper() and len(word) > 1])
        
        return {
            "has_numbers": has_numbers,
            "has_special_chars": has_special_chars,
            "has_urls": has_urls,
            "has_emails": has_emails,
            "capitalized_words": capitalized_words,
            "all_caps_words": all_caps_words,
            "capitalization_ratio": capitalized_words / len(words) if words else 0,
            "all_caps_ratio": all_caps_words / len(words) if words else 0
        }
    
    def _calculate_overall_quality_score(self, complexity_analysis: Dict[str, Any],
                                       domain_indicators: Dict[str, float],
                                       data_quality_metrics: Dict[str, Any],
                                       readability_scores: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        
        # Component scores
        complexity_score = 1.0 - complexity_analysis.get("complexity_score", 0.0)  # Lower complexity = higher quality
        domain_score = max(domain_indicators.values()) if domain_indicators else 0.0
        quality_score = data_quality_metrics.get("quality_score", 0.0)
        
        # Readability score (average of all formulas)
        readability_avg = np.mean(list(readability_scores.values())) if readability_scores else 0.0
        normalized_readability = readability_avg / 100.0  # Normalize to 0-1
        
        # Weighted combination
        weights = {
            "complexity": 0.2,
            "domain": 0.3,
            "quality": 0.3,
            "readability": 0.2
        }
        
        overall_score = (
            complexity_score * weights["complexity"] +
            domain_score * weights["domain"] +
            quality_score * weights["quality"] +
            normalized_readability * weights["readability"]
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _create_fallback_profile(self, text: str) -> TextProfile:
        """Create fallback profile when analysis fails"""
        return TextProfile(
            quality_score=0.5,
            complexity_analysis={"complexity_score": 0.5, "error": "fallback"},
            domain_indicators={},
            data_quality_metrics={"quality_score": 0.5, "error": "fallback"},
            readability_scores={"flesch_reading_ease": 50.0},
            language_analysis={"error": "fallback"},
            metadata={"error": "fallback", "timestamp": datetime.now().isoformat()}
        )
    
    async def profile_dataset(self, texts: List[str]) -> Dict[str, Any]:
        """Profile entire dataset"""
        profiles = []
        
        for text in texts:
            profile = await self.profile_text(text)
            profiles.append(profile)
        
        # Aggregate statistics
        quality_scores = [p.quality_score for p in profiles]
        complexity_scores = [p.complexity_analysis.get("complexity_score", 0) for p in profiles]
        
        # Domain distribution
        domain_distribution = {}
        for profile in profiles:
            for domain, score in profile.domain_indicators.items():
                if domain not in domain_distribution:
                    domain_distribution[domain] = []
                domain_distribution[domain].append(score)
        
        # Calculate averages
        avg_domain_scores = {}
        for domain, scores in domain_distribution.items():
            avg_domain_scores[domain] = np.mean(scores)
        
        return {
            "dataset_size": len(texts),
            "average_quality_score": np.mean(quality_scores),
            "average_complexity_score": np.mean(complexity_scores),
            "quality_score_std": np.std(quality_scores),
            "domain_distribution": avg_domain_scores,
            "profiles": profiles,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0"
            }
        } 