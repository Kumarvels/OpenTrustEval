"""
PII Detection and Anonymization
Cleanlab Replacement Component

Features:
- Comprehensive PII detection using Presidio
- Custom pattern-based detection
- Anonymization capabilities
- Risk assessment and scoring
- Batch processing support
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Try to import Presidio, fallback to custom implementation
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available, using custom PII detection")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PIIEntity:
    """PII entity detection result"""
    entity_type: str
    start: int
    end: int
    score: float
    text: str
    risk_level: str

@dataclass
class PIIAnalysisResult:
    """Complete PII analysis result"""
    has_pii: bool
    entities: List[PIIEntity]
    risk_level: str
    risk_score: float
    anonymized_text: str
    recommendations: List[str]

class PIIDetector:
    """Comprehensive PII detection and anonymization"""
    
    def __init__(self, use_presidio: bool = True):
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        
        if self.use_presidio:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        
        # Custom PII patterns
        self.pii_patterns = {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'risk_level': 'medium',
                'replacement': '[EMAIL]'
            },
            'phone': {
                'pattern': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
                'risk_level': 'medium',
                'replacement': '[PHONE]'
            },
            'ssn': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
                'risk_level': 'high',
                'replacement': '[SSN]'
            },
            'credit_card': {
                'pattern': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                'risk_level': 'high',
                'replacement': '[CREDIT_CARD]'
            },
            'ip_address': {
                'pattern': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                'risk_level': 'medium',
                'replacement': '[IP_ADDRESS]'
            },
            'url': {
                'pattern': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'risk_level': 'low',
                'replacement': '[URL]'
            },
            'date_of_birth': {
                'pattern': r'\b(?:birth|born|DOB|date of birth)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                'risk_level': 'high',
                'replacement': '[DATE_OF_BIRTH]'
            },
            'address': {
                'pattern': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                'risk_level': 'high',
                'replacement': '[ADDRESS]'
            },
            'passport': {
                'pattern': r'\b[A-Z]{1,2}\d{6,9}\b',
                'risk_level': 'high',
                'replacement': '[PASSPORT]'
            },
            'driver_license': {
                'pattern': r'\b[A-Z]{1,2}\d{6,8}\b',
                'risk_level': 'high',
                'replacement': '[DRIVER_LICENSE]'
            }
        }
        
        # Risk level weights
        self.risk_weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }

    def detect_pii(self, text: str) -> PIIAnalysisResult:
        """Detect PII in text using both Presidio and custom patterns"""
        logger.info(f"Detecting PII in text of length {len(text)}")
        
        entities = []
        
        # Use Presidio if available
        if self.use_presidio:
            presidio_entities = self._detect_with_presidio(text)
            entities.extend(presidio_entities)
        
        # Use custom patterns
        custom_entities = self._detect_with_custom_patterns(text)
        entities.extend(custom_entities)
        
        # Remove duplicates (entities that overlap)
        entities = self._remove_duplicate_entities(entities)
        
        # Calculate risk metrics
        risk_level, risk_score = self._calculate_risk(entities)
        
        # Anonymize text
        anonymized_text = self._anonymize_text(text, entities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(entities, risk_level)
        
        return PIIAnalysisResult(
            has_pii=len(entities) > 0,
            entities=entities,
            risk_level=risk_level,
            risk_score=risk_score,
            anonymized_text=anonymized_text,
            recommendations=recommendations
        )

    def _detect_with_presidio(self, text: str) -> List[PIIEntity]:
        """Detect PII using Presidio"""
        try:
            results = self.analyzer.analyze(text=text, language='en')
            
            entities = []
            for result in results:
                entity = PIIEntity(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text=text[result.start:result.end],
                    risk_level=self._map_presidio_risk(result.entity_type)
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.warning(f"Presidio detection failed: {e}")
            return []

    def _detect_with_custom_patterns(self, text: str) -> List[PIIEntity]:
        """Detect PII using custom patterns"""
        entities = []
        
        for entity_type, config in self.pii_patterns.items():
            pattern = config['pattern']
            risk_level = config['risk_level']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = PIIEntity(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.9,  # High confidence for pattern matches
                    text=match.group(),
                    risk_level=risk_level
                )
                entities.append(entity)
        
        return entities

    def _map_presidio_risk(self, entity_type: str) -> str:
        """Map Presidio entity types to risk levels"""
        high_risk_types = ['CREDIT_CARD', 'SSN', 'PASSPORT', 'DRIVER_LICENSE']
        medium_risk_types = ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'IP_ADDRESS']
        
        if entity_type in high_risk_types:
            return 'high'
        elif entity_type in medium_risk_types:
            return 'medium'
        else:
            return 'low'

    def _remove_duplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove overlapping entities, keeping the one with higher score"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        filtered_entities = []
        for entity in entities:
            # Check if this entity overlaps with any existing entity
            overlaps = False
            for existing in filtered_entities:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Overlap detected, keep the one with higher score
                    if entity.score > existing.score:
                        filtered_entities.remove(existing)
                        filtered_entities.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities

    def _calculate_risk(self, entities: List[PIIEntity]) -> Tuple[str, float]:
        """Calculate overall risk level and score"""
        if not entities:
            return 'none', 0.0
        
        # Calculate weighted risk score
        total_score = 0.0
        for entity in entities:
            weight = self.risk_weights.get(entity.risk_level, 0.3)
            total_score += entity.score * weight
        
        # Normalize score
        risk_score = min(1.0, total_score / len(entities))
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return risk_level, risk_score

    def _anonymize_text(self, text: str, entities: List[PIIEntity]) -> str:
        """Anonymize PII entities in text"""
        if not entities:
            return text
        
        # Sort entities by start position (reverse order to avoid index issues)
        entities.sort(key=lambda x: x.start, reverse=True)
        
        anonymized_text = text
        for entity in entities:
            # Find replacement based on entity type
            replacement = self._get_replacement_for_entity(entity)
            
            # Replace the entity
            anonymized_text = (
                anonymized_text[:entity.start] + 
                replacement + 
                anonymized_text[entity.end:]
            )
        
        return anonymized_text

    def _get_replacement_for_entity(self, entity: PIIEntity) -> str:
        """Get appropriate replacement for entity type"""
        # Check custom patterns first
        if entity.entity_type in self.pii_patterns:
            return self.pii_patterns[entity.entity_type]['replacement']
        
        # Default replacements for Presidio entities
        presidio_replacements = {
            'CREDIT_CARD': '[CREDIT_CARD]',
            'SSN': '[SSN]',
            'EMAIL_ADDRESS': '[EMAIL]',
            'PHONE_NUMBER': '[PHONE]',
            'IP_ADDRESS': '[IP_ADDRESS]',
            'PASSPORT': '[PASSPORT]',
            'DRIVER_LICENSE': '[DRIVER_LICENSE]',
            'PERSON': '[PERSON]',
            'ORGANIZATION': '[ORGANIZATION]',
            'LOCATION': '[LOCATION]'
        }
        
        return presidio_replacements.get(entity.entity_type, '[PII]')

    def _generate_recommendations(self, entities: List[PIIEntity], risk_level: str) -> List[str]:
        """Generate recommendations based on PII detection results"""
        recommendations = []
        
        if not entities:
            recommendations.append("No PII detected. Text is safe for processing.")
            return recommendations
        
        # Count entities by type
        entity_counts = {}
        for entity in entities:
            entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
        
        # High-risk recommendations
        if risk_level == 'high':
            recommendations.append("High-risk PII detected. Immediate anonymization required.")
            recommendations.append("Consider implementing stricter data handling policies.")
        
        # Entity-specific recommendations
        if 'ssn' in entity_counts or 'CREDIT_CARD' in entity_counts:
            recommendations.append("Sensitive financial information detected. Use encryption for storage.")
        
        if 'email' in entity_counts or 'EMAIL_ADDRESS' in entity_counts:
            recommendations.append("Email addresses detected. Consider email masking for privacy.")
        
        if 'phone' in entity_counts or 'PHONE_NUMBER' in entity_counts:
            recommendations.append("Phone numbers detected. Use phone number masking.")
        
        # General recommendations
        recommendations.append(f"Found {len(entities)} PII entities across {len(entity_counts)} types.")
        recommendations.append("Review data collection practices to minimize PII exposure.")
        
        return recommendations

    def batch_detect_pii(self, texts: List[str]) -> List[PIIAnalysisResult]:
        """Detect PII in a batch of texts"""
        logger.info(f"Batch PII detection for {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.detect_pii(text)
                results.append(result)
            except Exception as e:
                logger.error(f"PII detection failed for text {i}: {e}")
                # Create empty result for failed detection
                results.append(PIIAnalysisResult(
                    has_pii=False,
                    entities=[],
                    risk_level='none',
                    risk_score=0.0,
                    anonymized_text=text,
                    recommendations=[f"PII detection failed: {str(e)}"]
                ))
        
        return results

    def create_pii_report(self, results: List[PIIAnalysisResult]) -> Dict[str, Any]:
        """Create comprehensive PII analysis report"""
        total_texts = len(results)
        texts_with_pii = sum(1 for r in results if r.has_pii)
        
        # Aggregate entity types
        entity_counts = {}
        risk_level_counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        
        for result in results:
            risk_level_counts[result.risk_level] += 1
            for entity in result.entities:
                entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
        
        # Calculate average risk score
        avg_risk_score = sum(r.risk_score for r in results) / total_texts if total_texts > 0 else 0
        
        return {
            'summary': {
                'total_texts': total_texts,
                'texts_with_pii': texts_with_pii,
                'pii_percentage': (texts_with_pii / total_texts * 100) if total_texts > 0 else 0,
                'average_risk_score': avg_risk_score
            },
            'risk_distribution': risk_level_counts,
            'entity_types': entity_counts,
            'recommendations': self._generate_batch_recommendations(results),
            'high_risk_texts': [
                i for i, r in enumerate(results) 
                if r.risk_level == 'high'
            ]
        }

    def _generate_batch_recommendations(self, results: List[PIIAnalysisResult]) -> List[str]:
        """Generate recommendations for batch analysis"""
        recommendations = []
        
        total_texts = len(results)
        high_risk_count = sum(1 for r in results if r.risk_level == 'high')
        medium_risk_count = sum(1 for r in results if r.risk_level == 'medium')
        
        if high_risk_count > 0:
            recommendations.append(f"Found {high_risk_count} high-risk texts. Immediate action required.")
        
        if medium_risk_count > 0:
            recommendations.append(f"Found {medium_risk_count} medium-risk texts. Review and anonymize.")
        
        pii_percentage = sum(1 for r in results if r.has_pii) / total_texts * 100
        if pii_percentage > 50:
            recommendations.append("High PII prevalence detected. Consider data anonymization pipeline.")
        
        recommendations.append("Implement PII detection in data preprocessing workflow.")
        recommendations.append("Train team on PII handling best practices.")
        
        return recommendations

    def validate_anonymization(self, original_text: str, anonymized_text: str) -> Dict[str, Any]:
        """Validate that anonymization was successful"""
        # Check if any PII patterns still exist in anonymized text
        remaining_pii = []
        
        for entity_type, config in self.pii_patterns.items():
            pattern = config['pattern']
            matches = re.findall(pattern, anonymized_text, re.IGNORECASE)
            if matches:
                remaining_pii.append({
                    'entity_type': entity_type,
                    'count': len(matches),
                    'examples': matches[:3]
                })
        
        # Calculate anonymization effectiveness
        original_pii = self.detect_pii(original_text)
        anonymized_pii = self.detect_pii(anonymized_text)
        
        effectiveness = 1.0 - (len(anonymized_pii.entities) / max(1, len(original_pii.entities)))
        
        return {
            'successful': len(remaining_pii) == 0,
            'effectiveness': effectiveness,
            'remaining_pii': remaining_pii,
            'original_entities': len(original_pii.entities),
            'anonymized_entities': len(anonymized_pii.entities)
        }

# Example usage
if __name__ == "__main__":
    # Sample texts with PII
    sample_texts = [
        "Contact me at john.doe@email.com or call 555-123-4567",
        "My SSN is 123-45-6789 and my credit card is 1234-5678-9012-3456",
        "This is a safe text without any PII",
        "Visit our website at https://example.com for more information",
        "My address is 123 Main Street, Anytown, USA 12345"
    ]
    
    detector = PIIDetector()
    
    # Single text detection
    result = detector.detect_pii(sample_texts[0])
    print(f"PII Detection Result:")
    print(f"Has PII: {result.has_pii}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.risk_score:.3f}")
    print(f"Anonymized: {result.anonymized_text}")
    
    # Batch detection
    batch_results = detector.batch_detect_pii(sample_texts)
    report = detector.create_pii_report(batch_results)
    
    print(f"\nBatch Report:")
    print(f"Total texts: {report['summary']['total_texts']}")
    print(f"Texts with PII: {report['summary']['texts_with_pii']}")
    print(f"PII percentage: {report['summary']['pii_percentage']:.1f}%")
    print(f"Average risk score: {report['summary']['average_risk_score']:.3f}")
    
    print(f"\nRisk distribution: {report['risk_distribution']}")
    print(f"Entity types: {report['entity_types']}") 