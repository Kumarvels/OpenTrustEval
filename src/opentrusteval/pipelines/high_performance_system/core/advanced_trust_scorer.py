"""
Advanced Trust Scoring System for Ultimate MoE Solution

This module provides comprehensive trust scoring with multiple factors,
confidence interval calculation, and risk assessment for text verification.
"""

import asyncio
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import re
import json


class TrustLevel(Enum):
    """Trust levels for content"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskLevel(Enum):
    """Risk levels for content"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrustFactor:
    """Individual trust factor analysis"""
    factor_name: str
    score: float
    weight: float
    confidence: float
    evidence: List[str]
    impact: str


@dataclass
class TrustAnalysisResult:
    """Result of comprehensive trust analysis"""
    trust_score: float
    trust_level: TrustLevel
    confidence_interval: Tuple[float, float]
    risk_level: RiskLevel
    trust_factors: List[TrustFactor]
    risk_factors: List[str]
    recommendations: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class CredibilityAnalyzer:
    """Analyze content credibility based on various factors"""
    
    def __init__(self):
        self.credibility_indicators = {
            'authoritative_sources': [
                'research', 'study', 'analysis', 'report', 'survey',
                'peer-reviewed', 'published', 'journal', 'university',
                'institution', 'organization', 'government', 'official'
            ],
            'citation_indicators': [
                'according to', 'cited by', 'reference', 'source',
                'study shows', 'research indicates', 'data from',
                'statistics show', 'evidence suggests'
            ],
            'expert_indicators': [
                'expert', 'specialist', 'professor', 'doctor', 'researcher',
                'analyst', 'consultant', 'authority', 'professional'
            ]
        }
        
        self.credibility_penalties = {
            'unsubstantiated_claims': [
                'proven', 'definitely', 'certainly', 'absolutely',
                'guaranteed', '100%', 'without doubt'
            ],
            'emotional_language': [
                'amazing', 'incredible', 'shocking', 'outrageous',
                'terrible', 'horrible', 'fantastic', 'perfect'
            ],
            'conspiracy_indicators': [
                'conspiracy', 'cover-up', 'secret', 'hidden truth',
                'they don\'t want you to know', 'mainstream media lies'
            ]
        }
    
    async def analyze_credibility(self, text: str) -> TrustFactor:
        """Analyze text credibility"""
        text_lower = text.lower()
        
        # Calculate positive credibility indicators
        positive_score = 0
        positive_evidence = []
        
        for category, indicators in self.credibility_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                positive_score += min(matches * 0.1, 0.3)  # Cap at 0.3 per category
                positive_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Calculate negative credibility indicators
        negative_score = 0
        negative_evidence = []
        
        for category, indicators in self.credibility_penalties.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                negative_score += min(matches * 0.15, 0.4)  # Cap at 0.4 per category
                negative_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Calculate final credibility score
        credibility_score = max(0.0, min(1.0, positive_score - negative_score + 0.5))
        
        # Determine confidence based on evidence strength
        total_evidence = len(positive_evidence) + len(negative_evidence)
        confidence = min(0.9, 0.5 + (total_evidence * 0.1))
        
        return TrustFactor(
            factor_name="credibility",
            score=credibility_score,
            weight=0.25,
            confidence=confidence,
            evidence=positive_evidence + negative_evidence,
            impact="high" if abs(credibility_score - 0.5) > 0.2 else "medium"
        )


class ConsistencyAnalyzer:
    """Analyze content consistency and coherence"""
    
    def __init__(self):
        self.consistency_indicators = {
            'logical_flow': ['therefore', 'thus', 'consequently', 'as a result', 'because'],
            'transition_words': ['however', 'moreover', 'furthermore', 'additionally', 'similarly'],
            'structured_content': ['first', 'second', 'third', 'finally', 'in conclusion']
        }
        
        self.inconsistency_indicators = {
            'contradictions': ['but', 'however', 'nevertheless', 'on the other hand'],
            'uncertainty': ['maybe', 'perhaps', 'possibly', 'might', 'could'],
            'vague_language': ['some', 'many', 'several', 'various', 'certain']
        }
    
    async def analyze_consistency(self, text: str) -> TrustFactor:
        """Analyze text consistency"""
        text_lower = text.lower()
        sentences = text.split('.')
        
        # Calculate positive consistency indicators
        positive_score = 0
        positive_evidence = []
        
        for category, indicators in self.consistency_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                positive_score += min(matches * 0.05, 0.2)
                positive_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Calculate negative consistency indicators
        negative_score = 0
        negative_evidence = []
        
        for category, indicators in self.inconsistency_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                negative_score += min(matches * 0.03, 0.15)
                negative_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Analyze sentence structure consistency
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
            structure_consistency = max(0, 1 - (length_variance / 100))
            positive_score += structure_consistency * 0.1
            positive_evidence.append("consistent sentence structure")
        
        # Calculate final consistency score
        consistency_score = max(0.0, min(1.0, positive_score - negative_score + 0.5))
        
        # Determine confidence
        total_evidence = len(positive_evidence) + len(negative_evidence)
        confidence = min(0.9, 0.5 + (total_evidence * 0.1))
        
        return TrustFactor(
            factor_name="consistency",
            score=consistency_score,
            weight=0.20,
            confidence=confidence,
            evidence=positive_evidence + negative_evidence,
            impact="medium"
        )


class SourceQualityAnalyzer:
    """Analyze source quality and reliability"""
    
    def __init__(self):
        self.high_quality_sources = {
            'academic': ['edu', 'university', 'college', 'institute'],
            'government': ['gov', 'government', 'official'],
            'reputable_media': ['bbc', 'reuters', 'ap', 'npr', 'pbs'],
            'research': ['research', 'study', 'analysis', 'report']
        }
        
        self.low_quality_sources = {
            'social_media': ['facebook', 'twitter', 'instagram', 'tiktok'],
            'blog': ['blog', 'blogspot', 'wordpress'],
            'forum': ['forum', 'reddit', '4chan'],
            'unreliable': ['conspiracy', 'fake', 'hoax', 'rumor']
        }
    
    async def analyze_source_quality(self, text: str, context: str = "") -> TrustFactor:
        """Analyze source quality"""
        combined_text = (text + " " + context).lower()
        
        # Calculate high-quality source indicators
        high_quality_score = 0
        high_quality_evidence = []
        
        for category, indicators in self.high_quality_sources.items():
            matches = sum(1 for indicator in indicators if indicator in combined_text)
            if matches > 0:
                high_quality_score += min(matches * 0.15, 0.3)
                high_quality_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Calculate low-quality source indicators
        low_quality_score = 0
        low_quality_evidence = []
        
        for category, indicators in self.low_quality_sources.items():
            matches = sum(1 for indicator in indicators if indicator in combined_text)
            if matches > 0:
                low_quality_score += min(matches * 0.2, 0.4)
                low_quality_evidence.append(f"{matches} {category.replace('_', ' ')} indicators")
        
        # Calculate final source quality score
        source_quality_score = max(0.0, min(1.0, high_quality_score - low_quality_score + 0.5))
        
        # Determine confidence
        total_evidence = len(high_quality_evidence) + len(low_quality_evidence)
        confidence = min(0.9, 0.5 + (total_evidence * 0.1))
        
        return TrustFactor(
            factor_name="source_quality",
            score=source_quality_score,
            weight=0.25,
            confidence=confidence,
            evidence=high_quality_evidence + low_quality_evidence,
            impact="high" if abs(source_quality_score - 0.5) > 0.2 else "medium"
        )


class TemporalFreshnessAnalyzer:
    """Analyze temporal freshness and relevance"""
    
    def __init__(self):
        self.temporal_indicators = {
            'recent_dates': [
                r'\b20[2-9][0-9]\b',  # Years 2020-2099
                r'\b(202[3-9]|20[3-9][0-9])\b',  # Recent years
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+20[2-9][0-9]\b'
            ],
            'current_events': [
                'covid', 'pandemic', 'vaccine', 'lockdown',
                'climate change', 'global warming', 'renewable energy',
                'artificial intelligence', 'machine learning', 'blockchain'
            ]
        }
        
        self.outdated_indicators = [
            'old', 'outdated', 'obsolete', 'deprecated', 'legacy',
            'previous version', 'old version', 'no longer valid'
        ]
    
    async def analyze_temporal_freshness(self, text: str) -> TrustFactor:
        """Analyze temporal freshness"""
        text_lower = text.lower()
        
        # Calculate recent date indicators
        recent_score = 0
        recent_evidence = []
        
        for pattern in self.temporal_indicators['recent_dates']:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                recent_score += min(matches * 0.1, 0.2)
                recent_evidence.append(f"{matches} recent date references")
        
        # Calculate current event indicators
        for event in self.temporal_indicators['current_events']:
            if event in text_lower:
                recent_score += 0.05
                recent_evidence.append(f"current event: {event}")
        
        # Calculate outdated indicators
        outdated_score = 0
        outdated_evidence = []
        
        for indicator in self.outdated_indicators:
            if indicator in text_lower:
                outdated_score += 0.1
                outdated_evidence.append(f"outdated indicator: {indicator}")
        
        # Calculate final temporal freshness score
        temporal_score = max(0.0, min(1.0, recent_score - outdated_score + 0.5))
        
        # Determine confidence
        total_evidence = len(recent_evidence) + len(outdated_evidence)
        confidence = min(0.9, 0.5 + (total_evidence * 0.1))
        
        return TrustFactor(
            factor_name="temporal_freshness",
            score=temporal_score,
            weight=0.15,
            confidence=confidence,
            evidence=recent_evidence + outdated_evidence,
            impact="medium"
        )


class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for trust scores"""
    
    def __init__(self):
        self.confidence_levels = {
            0.90: 1.645,  # 90% confidence
            0.95: 1.96,   # 95% confidence
            0.99: 2.576   # 99% confidence
        }
    
    def calculate_confidence_interval(self, trust_factors: List[TrustFactor], 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for trust score"""
        if not trust_factors:
            return (0.0, 1.0)
        
        # Calculate weighted mean
        weighted_sum = sum(factor.score * factor.weight for factor in trust_factors)
        total_weight = sum(factor.weight for factor in trust_factors)
        mean_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate weighted standard error
        weighted_variance = sum(
            factor.weight * (factor.score - mean_score) ** 2 
            for factor in trust_factors
        ) / total_weight if total_weight > 0 else 0.25
        
        standard_error = math.sqrt(weighted_variance / len(trust_factors))
        
        # Calculate confidence interval
        z_score = self.confidence_levels.get(confidence_level, 1.96)
        margin_of_error = z_score * standard_error
        
        lower_bound = max(0.0, mean_score - margin_of_error)
        upper_bound = min(1.0, mean_score + margin_of_error)
        
        return (lower_bound, upper_bound)


class RiskAssessmentSystem:
    """Assess risk factors in content"""
    
    def __init__(self):
        self.risk_indicators = {
            'misinformation': [
                'fake news', 'false information', 'misleading', 'inaccurate',
                'wrong', 'incorrect', 'false claim', 'debunked'
            ],
            'bias': [
                'biased', 'one-sided', 'partisan', 'political agenda',
                'propaganda', 'manipulation', 'spin'
            ],
            'manipulation': [
                'clickbait', 'sensational', 'shocking', 'you won\'t believe',
                'secret', 'hidden', 'exposed', 'revealed'
            ],
            'uncertainty': [
                'maybe', 'perhaps', 'possibly', 'might', 'could',
                'uncertain', 'unclear', 'unknown', 'unverified'
            ]
        }
    
    async def assess_risk(self, text: str, trust_factors: List[TrustFactor]) -> Tuple[RiskLevel, List[str]]:
        """Assess overall risk level"""
        text_lower = text.lower()
        
        # Calculate risk indicators
        risk_scores = {}
        risk_evidence = []
        
        for risk_type, indicators in self.risk_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            if matches > 0:
                risk_scores[risk_type] = min(matches * 0.1, 0.5)
                risk_evidence.append(f"{matches} {risk_type} indicators")
        
        # Calculate overall risk score
        if risk_scores:
            overall_risk_score = sum(risk_scores.values()) / len(risk_scores)
        else:
            overall_risk_score = 0.0
        
        # Determine risk level
        if overall_risk_score >= 0.4:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk_score >= 0.3:
            risk_level = RiskLevel.HIGH
        elif overall_risk_score >= 0.2:
            risk_level = RiskLevel.MEDIUM
        elif overall_risk_score >= 0.1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
        
        # Add trust factor-based risks
        low_trust_factors = [f for f in trust_factors if f.score < 0.4]
        if low_trust_factors:
            risk_evidence.append(f"{len(low_trust_factors)} low-trust factors")
        
        return risk_level, risk_evidence


class AdvancedTrustScorer:
    """Advanced trust scoring with multiple factors and comprehensive analysis"""
    
    def __init__(self):
        self.credibility_analyzer = CredibilityAnalyzer()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.source_quality_analyzer = SourceQualityAnalyzer()
        self.temporal_freshness_analyzer = TemporalFreshnessAnalyzer()
        self.confidence_calculator = ConfidenceIntervalCalculator()
        self.risk_assessor = RiskAssessmentSystem()
        
        # Performance tracking
        self.scoring_stats = {
            'total_analyses': 0,
            'average_processing_time': 0.0,
            'trust_score_distribution': {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        }
    
    async def score_trust(self, text: str, context: str = "") -> TrustAnalysisResult:
        """Comprehensive trust scoring"""
        start_time = datetime.now()
        
        # Step 1: Analyze individual trust factors
        credibility_factor = await self.credibility_analyzer.analyze_credibility(text)
        consistency_factor = await self.consistency_analyzer.analyze_consistency(text)
        source_quality_factor = await self.source_quality_analyzer.analyze_source_quality(text, context)
        temporal_freshness_factor = await self.temporal_freshness_analyzer.analyze_temporal_freshness(text)
        
        trust_factors = [
            credibility_factor,
            consistency_factor,
            source_quality_factor,
            temporal_freshness_factor
        ]
        
        # Step 2: Calculate overall trust score
        trust_score = self._calculate_overall_trust_score(trust_factors)
        trust_level = self._determine_trust_level(trust_score)
        
        # Step 3: Calculate confidence interval
        confidence_interval = self.confidence_calculator.calculate_confidence_interval(trust_factors)
        
        # Step 4: Assess risk
        risk_level, risk_factors = await self.risk_assessor.assess_risk(text, trust_factors)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(trust_factors, risk_level, trust_score)
        
        # Step 6: Calculate processing time and update stats
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(trust_level, processing_time)
        
        # Step 7: Prepare metadata
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'context_provided': bool(context),
            'factor_count': len(trust_factors),
            'confidence_level': 0.95
        }
        
        return TrustAnalysisResult(
            trust_score=trust_score,
            trust_level=trust_level,
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            trust_factors=trust_factors,
            risk_factors=risk_factors,
            recommendations=recommendations,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _calculate_overall_trust_score(self, trust_factors: List[TrustFactor]) -> float:
        """Calculate weighted overall trust score"""
        if not trust_factors:
            return 0.5
        
        weighted_sum = sum(factor.score * factor.weight for factor in trust_factors)
        total_weight = sum(factor.weight for factor in trust_factors)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Determine trust level based on score"""
        if trust_score >= 0.8:
            return TrustLevel.VERY_HIGH
        elif trust_score >= 0.6:
            return TrustLevel.HIGH
        elif trust_score >= 0.4:
            return TrustLevel.MEDIUM
        elif trust_score >= 0.2:
            return TrustLevel.LOW
        else:
            return TrustLevel.VERY_LOW
    
    def _generate_recommendations(self, trust_factors: List[TrustFactor], 
                                risk_level: RiskLevel, trust_score: float) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Trust score-based recommendations
        if trust_score < 0.3:
            recommendations.append("VERY LOW TRUST: Content requires extensive verification")
            recommendations.append("Consider multiple independent sources for validation")
        elif trust_score < 0.5:
            recommendations.append("LOW TRUST: Additional verification recommended")
            recommendations.append("Cross-reference with authoritative sources")
        elif trust_score < 0.7:
            recommendations.append("MEDIUM TRUST: Moderate confidence in content")
            recommendations.append("Verify key claims with reliable sources")
        elif trust_score < 0.9:
            recommendations.append("HIGH TRUST: Good confidence in content")
            recommendations.append("Minor verification may be beneficial")
        else:
            recommendations.append("VERY HIGH TRUST: High confidence in content")
            recommendations.append("Content appears reliable and trustworthy")
        
        # Risk-based recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL RISK: Content poses significant risk")
            recommendations.append("Immediate review and verification required")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("HIGH RISK: Content requires careful review")
            recommendations.append("Verify all claims and check for bias")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("MEDIUM RISK: Exercise caution with content")
            recommendations.append("Verify important claims independently")
        
        # Factor-specific recommendations
        low_factors = [f for f in trust_factors if f.score < 0.4]
        for factor in low_factors:
            recommendations.append(f"Low {factor.factor_name}: {factor.evidence}")
        
        return recommendations
    
    def _update_stats(self, trust_level: TrustLevel, processing_time: float):
        """Update performance statistics"""
        self.scoring_stats['total_analyses'] += 1
        self.scoring_stats['trust_score_distribution'][trust_level.value] += 1
        
        # Update average processing time
        current_avg = self.scoring_stats['average_processing_time']
        total_analyses = self.scoring_stats['total_analyses']
        self.scoring_stats['average_processing_time'] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_analyses = self.scoring_stats['total_analyses']
        if total_analyses == 0:
            return self.scoring_stats
        
        # Calculate distribution percentages
        distribution = self.scoring_stats['trust_score_distribution'].copy()
        for level in distribution:
            distribution[level] = (distribution[level] / total_analyses) * 100
        
        return {
            'total_analyses': total_analyses,
            'average_processing_time': self.scoring_stats['average_processing_time'],
            'trust_score_distribution': distribution,
            'most_common_trust_level': max(
                self.scoring_stats['trust_score_distribution'].items(),
                key=lambda x: x[1]
            )[0]
        }


# Example usage and testing
async def test_advanced_trust_scoring():
    """Test the advanced trust scoring system"""
    scorer = AdvancedTrustScorer()
    
    # Test cases with different trust levels
    test_cases = [
        {
            'text': "According to a peer-reviewed study published in Nature in 2023, climate change has accelerated significantly. The research, conducted by leading scientists at MIT, analyzed data from over 1000 weather stations worldwide.",
            'context': "Academic research publication",
            'expected_level': TrustLevel.HIGH
        },
        {
            'text': "This is AMAZING! You won't BELIEVE what they found! The secret cure for everything is revealed in this shocking video!",
            'context': "Social media post",
            'expected_level': TrustLevel.VERY_LOW
        },
        {
            'text': "The weather forecast predicts rain tomorrow with a 60% probability. Temperatures will range from 15-20°C.",
            'context': "Weather service",
            'expected_level': TrustLevel.MEDIUM
        }
    ]
    
    print("=== Advanced Trust Scoring Test Results ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {test_case['text'][:100]}...")
        print(f"Context: {test_case['context']}")
        
        result = await scorer.score_trust(test_case['text'], test_case['context'])
        
        print(f"Trust Score: {result.trust_score:.3f}")
        print(f"Trust Level: {result.trust_level.value}")
        print(f"Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"Risk Level: {result.risk_level.value}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        
        print("\nTrust Factors:")
        for factor in result.trust_factors:
            print(f"- {factor.factor_name}: {factor.score:.3f} (weight: {factor.weight}, confidence: {factor.confidence:.3f})")
        
        print("\nRisk Factors:")
        for risk in result.risk_factors:
            print(f"- {risk}")
        
        print("\nRecommendations:")
        for rec in result.recommendations[:3]:  # Show first 3 recommendations
            print(f"- {rec}")
        
        # Check if result matches expected
        if result.trust_level == test_case['expected_level']:
            print("✅ Expected trust level matched!")
        else:
            print(f"❌ Expected {test_case['expected_level'].value}, got {result.trust_level.value}")
    
    print("\n=== Performance Statistics ===")
    stats = scorer.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_advanced_trust_scoring()) 