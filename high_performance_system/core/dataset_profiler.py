"""
Advanced Dataset Profiler
Cleanlab Replacement Component

Features:
- Duplicate detection via embeddings + cosine similarity
- Outlier detection via isolation forest
- Metadata extraction and tagging
- Language quality filters
- PII detection and anonymization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetProfile:
    """Comprehensive dataset profile"""
    total_samples: int
    duplicate_pairs: List[Tuple[int, int, float]]
    outliers: List[int]
    metadata: List[Dict[str, Any]]
    quality_scores: List[float]
    pii_detections: List[Dict[str, Any]]
    recommendations: List[str]

@dataclass
class TextMetadata:
    """Metadata extracted from text"""
    length: int
    word_count: int
    language: str
    sentiment: float
    readability: float
    timestamp: Optional[datetime]
    has_pii: bool
    quality_score: float

class AdvancedDatasetProfiler:
    """Advanced dataset profiling for Cleanlab replacement"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 duplicate_threshold: float = 0.95,
                 outlier_contamination: float = 0.1):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.duplicate_threshold = duplicate_threshold
        self.outlier_contamination = outlier_contamination
        self.outlier_detector = IsolationForest(
            contamination=outlier_contamination,
            random_state=42
        )
        
        # Language detection patterns
        self.language_patterns = {
            'english': r'[a-zA-Z]',
            'spanish': r'[áéíóúñ]',
            'french': r'[àâäéèêëïîôöùûüÿç]',
            'german': r'[äöüß]',
            'chinese': r'[\u4e00-\u9fff]',
            'japanese': r'[\u3040-\u309f\u30a0-\u30ff]',
            'korean': r'[\uac00-\ud7af]'
        }

    def profile_dataset(self, texts: List[str]) -> DatasetProfile:
        """Comprehensive dataset profiling"""
        logger.info(f"Profiling dataset with {len(texts)} samples")
        
        # Detect duplicates
        duplicate_pairs = self.detect_duplicates(texts)
        
        # Detect outliers
        outliers = self.detect_outliers(texts)
        
        # Extract metadata
        metadata = self.extract_metadata(texts)
        
        # Calculate quality scores
        quality_scores = self.calculate_quality_scores(texts)
        
        # Detect PII
        pii_detections = self.detect_pii_batch(texts)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            texts, duplicate_pairs, outliers, quality_scores, pii_detections
        )
        
        return DatasetProfile(
            total_samples=len(texts),
            duplicate_pairs=duplicate_pairs,
            outliers=outliers,
            metadata=metadata,
            quality_scores=quality_scores,
            pii_detections=pii_detections,
            recommendations=recommendations
        )

    def detect_duplicates(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """Detect duplicate texts using embeddings and cosine similarity"""
        logger.info("Detecting duplicates...")
        
        if len(texts) < 2:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Find duplicate pairs
        duplicates = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                similarity = similarities[i][j]
                if similarity > self.duplicate_threshold:
                    duplicates.append((i, j, similarity))
        
        # Sort by similarity score
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates

    def detect_outliers(self, texts: List[str]) -> List[int]:
        """Detect outlier texts using isolation forest"""
        logger.info("Detecting outliers...")
        
        if len(texts) < 10:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Fit isolation forest
        outlier_labels = self.outlier_detector.fit_predict(embeddings)
        
        # Get outlier indices
        outliers = [i for i, label in enumerate(outlier_labels) if label == -1]
        
        logger.info(f"Found {len(outliers)} outliers")
        return outliers

    def extract_metadata(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract comprehensive metadata from texts"""
        logger.info("Extracting metadata...")
        
        metadata = []
        for text in texts:
            meta = {
                'length': len(text),
                'word_count': len(text.split()),
                'language': self.detect_language(text),
                'sentiment': self.analyze_sentiment(text),
                'readability': self.calculate_readability(text),
                'timestamp': self.extract_timestamp(text),
                'has_numbers': bool(re.search(r'\d', text)),
                'has_urls': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
                'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
                'sentence_count': len(re.split(r'[.!?]+', text)),
                'avg_sentence_length': len(text.split()) / max(1, len(re.split(r'[.!?]+', text)))
            }
            metadata.append(meta)
        
        return metadata

    def detect_language(self, text: str) -> str:
        """Detect language using pattern matching"""
        text_lower = text.lower()
        
        # Count matches for each language
        language_scores = {}
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            language_scores[lang] = matches
        
        # Return language with most matches
        if language_scores:
            return max(language_scores.items(), key=lambda x: x[1])[0]
        return 'unknown'

    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        # Simple rule-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'dislike', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)

    def calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        syllables = self._count_syllables(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0.0, min(100.0, flesch_score))

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text"""
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

    def extract_timestamp(self, text: str) -> Optional[datetime]:
        """Extract timestamp from text"""
        # Common date patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # DD MMM YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime.strptime(match.group(), '%Y-%m-%d')
                except ValueError:
                    try:
                        return datetime.strptime(match.group(), '%m/%d/%Y')
                    except ValueError:
                        try:
                            return datetime.strptime(match.group(), '%m-%d-%Y')
                        except ValueError:
                            continue
        
        return None

    def calculate_quality_scores(self, texts: List[str]) -> List[float]:
        """Calculate quality scores for texts"""
        quality_scores = []
        
        for text in texts:
            score = 0.0
            
            # Length score (prefer medium length)
            length = len(text)
            if 50 <= length <= 500:
                score += 0.3
            elif 20 <= length <= 1000:
                score += 0.2
            else:
                score += 0.1
            
            # Readability score
            readability = self.calculate_readability(text)
            if readability >= 60:
                score += 0.3
            elif readability >= 30:
                score += 0.2
            else:
                score += 0.1
            
            # Completeness score (has numbers, proper punctuation)
            if re.search(r'\d', text):
                score += 0.1
            if re.search(r'[.!?]', text):
                score += 0.1
            if len(text.split()) >= 5:
                score += 0.1
            
            # Language consistency
            if self.detect_language(text) == 'english':
                score += 0.1
            
            quality_scores.append(min(1.0, score))
        
        return quality_scores

    def detect_pii_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect PII in batch of texts"""
        pii_detections = []
        
        for i, text in enumerate(texts):
            pii_info = self.detect_pii_single(text)
            pii_info['text_index'] = i
            pii_detections.append(pii_info)
        
        return pii_detections

    def detect_pii_single(self, text: str) -> Dict[str, Any]:
        """Detect PII in single text"""
        pii_types = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        detected_pii = {}
        for pii_type, pattern in pii_types.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii[pii_type] = {
                    'count': len(matches),
                    'examples': matches[:3]  # Limit to first 3 examples
                }
        
        return {
            'has_pii': len(detected_pii) > 0,
            'pii_types': detected_pii,
            'risk_level': self._calculate_pii_risk(detected_pii)
        }

    def _calculate_pii_risk(self, detected_pii: Dict[str, Any]) -> str:
        """Calculate PII risk level"""
        high_risk_types = ['ssn', 'credit_card']
        medium_risk_types = ['email', 'phone']
        
        if any(pii_type in detected_pii for pii_type in high_risk_types):
            return 'high'
        elif any(pii_type in detected_pii for pii_type in medium_risk_types):
            return 'medium'
        elif detected_pii:
            return 'low'
        else:
            return 'none'

    def generate_recommendations(self, 
                               texts: List[str],
                               duplicates: List[Tuple[int, int, float]],
                               outliers: List[int],
                               quality_scores: List[float],
                               pii_detections: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on profiling results"""
        recommendations = []
        
        # Duplicate recommendations
        if duplicates:
            recommendations.append(f"Found {len(duplicates)} duplicate pairs. Consider removing duplicates to improve dataset quality.")
        
        # Outlier recommendations
        if outliers:
            recommendations.append(f"Found {len(outliers)} outliers. Review these samples for data quality issues.")
        
        # Quality recommendations
        low_quality_count = sum(1 for score in quality_scores if score < 0.5)
        if low_quality_count > 0:
            recommendations.append(f"Found {low_quality_count} low-quality samples. Consider improving or removing these.")
        
        # PII recommendations
        high_risk_pii = sum(1 for pii in pii_detections if pii['risk_level'] == 'high')
        if high_risk_pii > 0:
            recommendations.append(f"Found {high_risk_pii} samples with high-risk PII. Anonymize or remove sensitive data.")
        
        # General recommendations
        if len(texts) < 100:
            recommendations.append("Dataset is small. Consider collecting more data for better model performance.")
        
        avg_quality = np.mean(quality_scores)
        if avg_quality < 0.6:
            recommendations.append("Overall dataset quality is low. Consider data cleaning and preprocessing.")
        
        return recommendations

    def create_profile_report(self, profile: DatasetProfile) -> Dict[str, Any]:
        """Create comprehensive profile report"""
        return {
            'summary': {
                'total_samples': profile.total_samples,
                'duplicate_pairs': len(profile.duplicate_pairs),
                'outliers': len(profile.outliers),
                'avg_quality_score': np.mean(profile.quality_scores),
                'pii_risk_samples': sum(1 for pii in profile.pii_detections if pii['risk_level'] != 'none')
            },
            'duplicates': [
                {
                    'index1': pair[0],
                    'index2': pair[1],
                    'similarity': pair[2]
                }
                for pair in profile.duplicate_pairs[:10]  # Top 10 duplicates
            ],
            'outliers': profile.outliers,
            'quality_distribution': {
                'excellent': sum(1 for score in profile.quality_scores if score >= 0.8),
                'good': sum(1 for score in profile.quality_scores if 0.6 <= score < 0.8),
                'fair': sum(1 for score in profile.quality_scores if 0.4 <= score < 0.6),
                'poor': sum(1 for score in profile.quality_scores if score < 0.4)
            },
            'pii_summary': {
                'high_risk': sum(1 for pii in profile.pii_detections if pii['risk_level'] == 'high'),
                'medium_risk': sum(1 for pii in profile.pii_detections if pii['risk_level'] == 'medium'),
                'low_risk': sum(1 for pii in profile.pii_detections if pii['risk_level'] == 'low'),
                'no_risk': sum(1 for pii in profile.pii_detections if pii['risk_level'] == 'none')
            },
            'recommendations': profile.recommendations
        }

    def anonymize_text(self, text: str, pii_detection: Dict[str, Any]) -> str:
        """Anonymize PII in text"""
        anonymized_text = text
        
        if not pii_detection['has_pii']:
            return anonymized_text
        
        # Replace emails
        if 'email' in pii_detection['pii_types']:
            anonymized_text = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '[EMAIL]',
                anonymized_text
            )
        
        # Replace phone numbers
        if 'phone' in pii_detection['pii_types']:
            anonymized_text = re.sub(
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                '[PHONE]',
                anonymized_text
            )
        
        # Replace SSNs
        if 'ssn' in pii_detection['pii_types']:
            anonymized_text = re.sub(
                r'\b\d{3}-\d{2}-\d{4}\b',
                '[SSN]',
                anonymized_text
            )
        
        # Replace credit cards
        if 'credit_card' in pii_detection['pii_types']:
            anonymized_text = re.sub(
                r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                '[CREDIT_CARD]',
                anonymized_text
            )
        
        return anonymized_text

# Example usage
if __name__ == "__main__":
    # Sample texts for testing
    sample_texts = [
        "This is a great product that I love using.",
        "This is a great product that I love using.",  # Duplicate
        "Terrible service, worst experience ever.",
        "The quick brown fox jumps over the lazy dog.",
        "Contact me at john.doe@email.com or call 555-123-4567",
        "My SSN is 123-45-6789 and my credit card is 1234-5678-9012-3456"
    ]
    
    profiler = AdvancedDatasetProfiler()
    profile = profiler.profile_dataset(sample_texts)
    
    print("Dataset Profile:")
    print(f"Total samples: {profile.total_samples}")
    print(f"Duplicate pairs: {len(profile.duplicate_pairs)}")
    print(f"Outliers: {len(profile.outliers)}")
    print(f"Average quality score: {np.mean(profile.quality_scores):.3f}")
    print(f"PII risk samples: {sum(1 for pii in profile.pii_detections if pii['risk_level'] != 'none')}")
    
    print("\nRecommendations:")
    for rec in profile.recommendations:
        print(f"- {rec}") 