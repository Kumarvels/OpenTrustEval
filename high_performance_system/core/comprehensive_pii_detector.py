"""
Comprehensive PII Detection System for Ultimate MoE Solution

This module provides advanced PII (Personally Identifiable Information) detection
with multiple detection methods, privacy risk assessment, and compliance checking.
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from datetime import datetime


class PIIRiskLevel(Enum):
    """PII risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    UNKNOWN = "unknown"


@dataclass
class DetectedPII:
    """Represents detected PII information"""
    pii_type: str
    value: str
    confidence: float
    risk_level: PIIRiskLevel
    position: Tuple[int, int]
    context: str
    compliance_impact: List[str]


@dataclass
class PIIAnalysisResult:
    """Result of PII analysis"""
    pii_score: float
    detected_pii: List[DetectedPII]
    privacy_risk: PIIRiskLevel
    compliance_status: ComplianceStatus
    risk_factors: List[str]
    recommendations: List[str]
    processing_time: float


class PatternBasedDetector:
    """Pattern-based PII detection using regex patterns"""
    
    def __init__(self):
        self.patterns = {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'risk_level': PIIRiskLevel.MEDIUM,
                'compliance_impact': ['GDPR', 'CCPA']
            },
            'phone': {
                'pattern': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                'risk_level': PIIRiskLevel.MEDIUM,
                'compliance_impact': ['GDPR', 'CCPA', 'HIPAA']
            },
            'ssn': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
                'risk_level': PIIRiskLevel.CRITICAL,
                'compliance_impact': ['GDPR', 'CCPA', 'HIPAA', 'SOX']
            },
            'credit_card': {
                'pattern': r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
                'risk_level': PIIRiskLevel.CRITICAL,
                'compliance_impact': ['PCI-DSS', 'GDPR', 'CCPA']
            },
            'ip_address': {
                'pattern': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                'risk_level': PIIRiskLevel.LOW,
                'compliance_impact': ['GDPR']
            },
            'mac_address': {
                'pattern': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
                'risk_level': PIIRiskLevel.LOW,
                'compliance_impact': ['GDPR']
            },
            'date_of_birth': {
                'pattern': r'\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-]\d{4}\b',
                'risk_level': PIIRiskLevel.HIGH,
                'compliance_impact': ['GDPR', 'CCPA', 'HIPAA']
            },
            'address': {
                'pattern': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                'risk_level': PIIRiskLevel.MEDIUM,
                'compliance_impact': ['GDPR', 'CCPA']
            }
        }
    
    async def detect_patterns(self, text: str) -> List[DetectedPII]:
        """Detect PII using pattern matching"""
        detected = []
        
        for pii_type, config in self.patterns.items():
            matches = re.finditer(config['pattern'], text, re.IGNORECASE)
            
            for match in matches:
                # Calculate confidence based on pattern strength
                confidence = self._calculate_pattern_confidence(pii_type, match.group())
                
                # Get context around the match
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                detected.append(DetectedPII(
                    pii_type=pii_type,
                    value=match.group(),
                    confidence=confidence,
                    risk_level=config['risk_level'],
                    position=(match.start(), match.end()),
                    context=context,
                    compliance_impact=config['compliance_impact']
                ))
        
        return detected
    
    def _calculate_pattern_confidence(self, pii_type: str, value: str) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.8
        
        # Adjust confidence based on PII type and value characteristics
        if pii_type == 'email':
            if '@' in value and '.' in value.split('@')[1]:
                return 0.95
            return 0.85
        
        elif pii_type == 'phone':
            digits = sum(c.isdigit() for c in value)
            if 10 <= digits <= 15:
                return 0.9
            return 0.7
        
        elif pii_type == 'ssn':
            if len(value.replace('-', '')) == 9:
                return 0.95
            return 0.8
        
        elif pii_type == 'credit_card':
            digits = value.replace(' ', '').replace('-', '').replace('.', '')
            if len(digits) == 16 and self._luhn_check(digits):
                return 0.95
            return 0.8
        
        return base_confidence
    
    def _luhn_check(self, digits: str) -> bool:
        """Luhn algorithm for credit card validation"""
        if not digits.isdigit():
            return False
        
        total = 0
        for i, digit in enumerate(reversed(digits)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        
        return total % 10 == 0


class ContextBasedDetector:
    """Context-based PII detection using surrounding text analysis"""
    
    def __init__(self):
        self.context_indicators = {
            'name': [
                'name', 'full name', 'first name', 'last name', 'given name',
                'surname', 'family name', 'user', 'customer', 'client', 'patient'
            ],
            'email': [
                'email', 'e-mail', 'email address', 'contact', 'send to',
                'notify', 'reach me at'
            ],
            'phone': [
                'phone', 'telephone', 'mobile', 'cell', 'contact number',
                'call', 'reach me', 'contact me'
            ],
            'address': [
                'address', 'location', 'residence', 'home', 'street',
                'city', 'state', 'zip', 'postal code'
            ],
            'ssn': [
                'ssn', 'social security', 'social security number',
                'tax id', 'identification number'
            ],
            'credit_card': [
                'credit card', 'card number', 'cc', 'visa', 'mastercard',
                'amex', 'payment', 'billing'
            ]
        }
    
    async def detect_context(self, text: str, pattern_detections: List[DetectedPII]) -> List[DetectedPII]:
        """Enhance pattern detections with context analysis"""
        enhanced_detections = []
        
        for detection in pattern_detections:
            # Get surrounding context
            start = max(0, detection.position[0] - 100)
            end = min(len(text), detection.position[1] + 100)
            context = text[start:end].lower()
            
            # Check for context indicators
            context_score = self._calculate_context_score(detection.pii_type, context)
            
            # Adjust confidence based on context
            adjusted_confidence = min(1.0, detection.confidence + context_score * 0.2)
            
            enhanced_detection = DetectedPII(
                pii_type=detection.pii_type,
                value=detection.value,
                confidence=adjusted_confidence,
                risk_level=detection.risk_level,
                position=detection.position,
                context=detection.context,
                compliance_impact=detection.compliance_impact
            )
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _calculate_context_score(self, pii_type: str, context: str) -> float:
        """Calculate context relevance score"""
        if pii_type not in self.context_indicators:
            return 0.0
        
        indicators = self.context_indicators[pii_type]
        matches = sum(1 for indicator in indicators if indicator in context)
        
        return min(1.0, matches / len(indicators))


class HeuristicDetector:
    """Heuristic-based PII detection using statistical analysis"""
    
    def __init__(self):
        self.heuristics = {
            'likely_name': self._is_likely_name,
            'likely_address': self._is_likely_address,
            'likely_phone': self._is_likely_phone,
            'likely_email': self._is_likely_email
        }
    
    async def detect_heuristics(self, text: str) -> List[DetectedPII]:
        """Detect PII using heuristic analysis"""
        detected = []
        words = text.split()
        
        for i, word in enumerate(words):
            for heuristic_name, heuristic_func in self.heuristics.items():
                if heuristic_func(word):
                    confidence = 0.6  # Lower confidence for heuristic detection
                    
                    # Get context
                    start = max(0, i - 3)
                    end = min(len(words), i + 4)
                    context = ' '.join(words[start:end])
                    
                    detected.append(DetectedPII(
                        pii_type=f'potential_{heuristic_name}',
                        value=word,
                        confidence=confidence,
                        risk_level=PIIRiskLevel.LOW,
                        position=(text.find(word), text.find(word) + len(word)),
                        context=context,
                        compliance_impact=['GDPR']
                    ))
        
        return detected
    
    def _is_likely_name(self, word: str) -> bool:
        """Check if word is likely a name"""
        # Simple heuristic: capitalized word with reasonable length
        return (word[0].isupper() and 
                len(word) >= 2 and 
                len(word) <= 20 and
                word.isalpha())
    
    def _is_likely_address(self, word: str) -> bool:
        """Check if word is likely part of an address"""
        # Check for common address components
        address_indicators = ['street', 'avenue', 'road', 'drive', 'lane', 'blvd']
        return any(indicator in word.lower() for indicator in address_indicators)
    
    def _is_likely_phone(self, word: str) -> bool:
        """Check if word is likely a phone number"""
        digits = sum(c.isdigit() for c in word)
        return 7 <= digits <= 15 and len(word) >= 10
    
    def _is_likely_email(self, word: str) -> bool:
        """Check if word is likely an email"""
        return '@' in word and '.' in word.split('@')[1]


class RiskAssessor:
    """Assess privacy risk and compliance status"""
    
    def __init__(self):
        self.risk_weights = {
            PIIRiskLevel.LOW: 1,
            PIIRiskLevel.MEDIUM: 2,
            PIIRiskLevel.HIGH: 3,
            PIIRiskLevel.CRITICAL: 4
        }
        
        self.compliance_frameworks = {
            'GDPR': {
                'high_risk_pii': ['ssn', 'credit_card', 'date_of_birth'],
                'medium_risk_pii': ['email', 'phone', 'address'],
                'low_risk_pii': ['ip_address', 'mac_address']
            },
            'CCPA': {
                'high_risk_pii': ['ssn', 'credit_card'],
                'medium_risk_pii': ['email', 'phone', 'address', 'date_of_birth'],
                'low_risk_pii': ['ip_address']
            },
            'HIPAA': {
                'high_risk_pii': ['ssn', 'date_of_birth'],
                'medium_risk_pii': ['phone', 'address'],
                'low_risk_pii': ['email']
            }
        }
    
    async def assess_risk(self, detections: List[DetectedPII]) -> Tuple[PIIRiskLevel, List[str]]:
        """Assess overall privacy risk"""
        if not detections:
            return PIIRiskLevel.LOW, []
        
        # Calculate weighted risk score
        total_score = 0
        max_possible_score = 0
        
        for detection in detections:
            weight = self.risk_weights[detection.risk_level]
            score = weight * detection.confidence
            total_score += score
            max_possible_score += weight
        
        risk_ratio = total_score / max_possible_score if max_possible_score > 0 else 0
        
        # Determine risk level
        if risk_ratio >= 0.75:
            risk_level = PIIRiskLevel.CRITICAL
        elif risk_ratio >= 0.5:
            risk_level = PIIRiskLevel.HIGH
        elif risk_ratio >= 0.25:
            risk_level = PIIRiskLevel.MEDIUM
        else:
            risk_level = PIIRiskLevel.LOW
        
        # Generate risk factors
        risk_factors = self._generate_risk_factors(detections)
        
        return risk_level, risk_factors
    
    async def assess_compliance(self, detections: List[DetectedPII]) -> ComplianceStatus:
        """Assess compliance status"""
        if not detections:
            return ComplianceStatus.COMPLIANT
        
        # Check for critical PII types
        critical_types = ['ssn', 'credit_card']
        has_critical = any(d.pii_type in critical_types for d in detections)
        
        if has_critical:
            return ComplianceStatus.NON_COMPLIANT
        
        # Check for high-risk PII types
        high_risk_types = ['date_of_birth', 'address']
        has_high_risk = any(d.pii_type in high_risk_types for d in detections)
        
        if has_high_risk:
            return ComplianceStatus.REQUIRES_REVIEW
        
        return ComplianceStatus.COMPLIANT
    
    def _generate_risk_factors(self, detections: List[DetectedPII]) -> List[str]:
        """Generate list of risk factors"""
        risk_factors = []
        
        # Count by risk level
        risk_counts = {}
        for detection in detections:
            risk_level = detection.risk_level.value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        # Generate risk factor descriptions
        for risk_level, count in risk_counts.items():
            if count > 0:
                risk_factors.append(f"{count} {risk_level}-risk PII items detected")
        
        # Check for specific high-risk types
        critical_types = [d.pii_type for d in detections if d.risk_level == PIIRiskLevel.CRITICAL]
        if critical_types:
            risk_factors.append(f"Critical PII types found: {', '.join(critical_types)}")
        
        return risk_factors


class ComprehensivePIIDetector:
    """Comprehensive PII detection with multiple detection methods"""
    
    def __init__(self):
        self.pattern_detector = PatternBasedDetector()
        self.context_detector = ContextBasedDetector()
        self.heuristic_detector = HeuristicDetector()
        self.risk_assessor = RiskAssessor()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'pattern_detections': 0,
            'context_enhanced': 0,
            'heuristic_detections': 0,
            'processing_times': []
        }
    
    async def detect_pii(self, text: str) -> PIIAnalysisResult:
        """Comprehensive PII detection with all methods"""
        start_time = datetime.now()
        
        # Step 1: Pattern-based detection
        pattern_detections = await self.pattern_detector.detect_patterns(text)
        
        # Step 2: Context-based enhancement
        context_enhanced = await self.context_detector.detect_context(text, pattern_detections)
        
        # Step 3: Heuristic detection
        heuristic_detections = await self.heuristic_detector.detect_heuristics(text)
        
        # Step 4: Combine all detections
        all_detections = context_enhanced + heuristic_detections
        
        # Step 5: Risk assessment
        privacy_risk, risk_factors = await self.risk_assessor.assess_risk(all_detections)
        compliance_status = await self.risk_assessor.assess_compliance(all_detections)
        
        # Step 6: Calculate overall PII score
        pii_score = self._calculate_pii_score(all_detections)
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(all_detections, privacy_risk, compliance_status)
        
        # Step 8: Update statistics
        self._update_stats(len(pattern_detections), len(context_enhanced), 
                          len(heuristic_detections), start_time)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PIIAnalysisResult(
            pii_score=pii_score,
            detected_pii=all_detections,
            privacy_risk=privacy_risk,
            compliance_status=compliance_status,
            risk_factors=risk_factors,
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    def _calculate_pii_score(self, detections: List[DetectedPII]) -> float:
        """Calculate overall PII score (0-1)"""
        if not detections:
            return 0.0
        
        # Weighted average based on confidence and risk level
        total_weighted_score = 0
        total_weight = 0
        
        for detection in detections:
            weight = self._get_risk_weight(detection.risk_level)
            score = detection.confidence * weight
            total_weighted_score += score
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _get_risk_weight(self, risk_level: PIIRiskLevel) -> float:
        """Get weight for risk level"""
        weights = {
            PIIRiskLevel.LOW: 1.0,
            PIIRiskLevel.MEDIUM: 2.0,
            PIIRiskLevel.HIGH: 3.0,
            PIIRiskLevel.CRITICAL: 4.0
        }
        return weights.get(risk_level, 1.0)
    
    def _generate_recommendations(self, detections: List[DetectedPII], 
                                privacy_risk: PIIRiskLevel, 
                                compliance_status: ComplianceStatus) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if privacy_risk == PIIRiskLevel.CRITICAL:
            recommendations.append("CRITICAL: Immediate action required - remove or encrypt all PII")
            recommendations.append("Implement strict access controls and audit logging")
        
        elif privacy_risk == PIIRiskLevel.HIGH:
            recommendations.append("HIGH RISK: Review and sanitize PII data")
            recommendations.append("Consider data anonymization or pseudonymization")
        
        elif privacy_risk == PIIRiskLevel.MEDIUM:
            recommendations.append("MEDIUM RISK: Monitor PII usage and implement safeguards")
            recommendations.append("Ensure proper consent mechanisms are in place")
        
        if compliance_status == ComplianceStatus.NON_COMPLIANT:
            recommendations.append("COMPLIANCE: Data processing violates regulatory requirements")
            recommendations.append("Immediate remediation required to meet compliance standards")
        
        elif compliance_status == ComplianceStatus.REQUIRES_REVIEW:
            recommendations.append("COMPLIANCE: Review data processing practices")
            recommendations.append("Consult legal team for compliance assessment")
        
        # Specific recommendations based on PII types
        pii_types = set(d.pii_type for d in detections)
        if 'credit_card' in pii_types:
            recommendations.append("PCI-DSS: Ensure PCI-DSS compliance for credit card data")
        if 'ssn' in pii_types:
            recommendations.append("SSN: Implement strict controls for SSN handling")
        if 'date_of_birth' in pii_types:
            recommendations.append("DOB: Consider age verification requirements")
        
        return recommendations
    
    def _update_stats(self, pattern_count: int, context_count: int, 
                     heuristic_count: int, start_time: datetime):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += pattern_count + heuristic_count
        self.detection_stats['pattern_detections'] += pattern_count
        self.detection_stats['context_enhanced'] += context_count
        self.detection_stats['heuristic_detections'] += heuristic_count
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.detection_stats['processing_times'].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.detection_stats['processing_times']) > 100:
            self.detection_stats['processing_times'] = self.detection_stats['processing_times'][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_processing_time = (sum(self.detection_stats['processing_times']) / 
                              len(self.detection_stats['processing_times']) 
                              if self.detection_stats['processing_times'] else 0)
        
        return {
            'total_detections': self.detection_stats['total_detections'],
            'pattern_detections': self.detection_stats['pattern_detections'],
            'context_enhanced': self.detection_stats['context_enhanced'],
            'heuristic_detections': self.detection_stats['heuristic_detections'],
            'average_processing_time': avg_processing_time,
            'detection_efficiency': (self.detection_stats['pattern_detections'] / 
                                   max(1, self.detection_stats['total_detections']))
        }


# Example usage and testing
async def test_comprehensive_pii_detection():
    """Test the comprehensive PII detection system"""
    detector = ComprehensivePIIDetector()
    
    # Test text with various PII types
    test_text = """
    Customer Information:
    Name: John Smith
    Email: john.smith@example.com
    Phone: (555) 123-4567
    Address: 123 Main Street, Anytown, CA 90210
    SSN: 123-45-6789
    Credit Card: 4111-1111-1111-1111
    Date of Birth: 01/15/1985
    IP Address: 192.168.1.1
    """
    
    result = await detector.detect_pii(test_text)
    
    print("=== Comprehensive PII Detection Results ===")
    print(f"PII Score: {result.pii_score:.3f}")
    print(f"Privacy Risk: {result.privacy_risk.value}")
    print(f"Compliance Status: {result.compliance_status.value}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Detected PII Items: {len(result.detected_pii)}")
    
    print("\nDetected PII:")
    for pii in result.detected_pii:
        print(f"- {pii.pii_type}: {pii.value} (confidence: {pii.confidence:.2f}, risk: {pii.risk_level.value})")
    
    print("\nRisk Factors:")
    for factor in result.risk_factors:
        print(f"- {factor}")
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
    
    print("\nPerformance Stats:")
    stats = detector.get_performance_stats()
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_comprehensive_pii_detection()) 