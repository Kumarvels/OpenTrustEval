"""
Mixture of Experts (MoE) Domain-Specific Verification System
Cleanlab Replacement Component

Features:
- Specialized expert models for each domain (Ecommerce, Banking, Insurance, etc.)
- Dynamic expert selection based on content analysis
- Domain-specific datasets for training and validation
- Ensemble decision making with confidence weighting
- Continuous learning and expert adaptation
- DeepSeek-style approach with multiple specialized models
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Supported domain types"""
    ECOMMERCE = "ecommerce"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    GENERAL = "general"

@dataclass
class ExpertPrediction:
    """Prediction from a single expert"""
    expert_id: str
    domain: DomainType
    confidence: float
    prediction: Dict[str, Any]
    reasoning: str
    metadata: Dict[str, Any]
    sources_used: list = None  # NEW FIELD

@dataclass
class MoEVerificationResult:
    """Complete MoE verification result"""
    verified: bool
    confidence: float
    expert_predictions: List[ExpertPrediction]
    ensemble_decision: Dict[str, Any]
    domain_detected: DomainType
    quality_score: float
    hallucination_risk: float
    recommendations: List[str]
    sources_used: list = None  # NEW FIELD

class DomainExpert(nn.Module):
    """Base class for domain-specific experts"""
    
    def __init__(self, domain: DomainType, model_config: Dict[str, Any]):
        super().__init__()
        self.domain = domain
        self.expert_id = f"{domain.value}_expert_{hashlib.md5(str(model_config).encode()).hexdigest()[:8]}"
        self.config = model_config
        
        # Initialize domain-specific layers
        self.embedding_dim = model_config.get('embedding_dim', 768)
        self.hidden_dim = model_config.get('hidden_dim', 512)
        self.num_classes = model_config.get('num_classes', 3)  # verified, uncertain, hallucination
        
        # Domain-specific architecture
        self.embedding_layer = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.domain_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.classification_layer = nn.Linear(self.hidden_dim, self.num_classes)
        self.confidence_layer = nn.Linear(self.hidden_dim, 1)
        
        # Domain-specific knowledge base
        self.domain_knowledge = self._load_domain_knowledge()
        
        # Performance tracking
        self.accuracy_history = []
        self.confidence_history = []
        
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain-specific knowledge base"""
        knowledge_path = Path(f"high_performance_system/data/domain_knowledge/{self.domain.value}.json")
        
        if knowledge_path.exists():
            with open(knowledge_path, 'r') as f:
                return json.load(f)
        else:
            # Return default knowledge structure
            return {
                'key_entities': [],
                'validation_rules': [],
                'common_patterns': [],
                'quality_metrics': []
            }
    
    def forward(self, embeddings: torch.Tensor, metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the expert"""
        # Domain-specific processing
        x = F.relu(self.embedding_layer(embeddings))
        x = F.relu(self.domain_layer(x))
        
        # Classification output
        classification = self.classification_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(x))
        
        return classification, confidence
    
    def predict(self, text: str, context: str = "") -> ExpertPrediction:
        """Make domain-specific prediction"""
        # This would be implemented with actual model inference
        # For now, return a structured prediction
        # Use domain knowledge as sources if available
        sources = [f"{self.domain.value}_knowledge_base"]
        return ExpertPrediction(
            expert_id=self.expert_id,
            domain=self.domain,
            confidence=0.85,  # Placeholder
            prediction={
                'verified': True,
                'quality_score': 0.9,
                'hallucination_risk': 0.1
            },
            reasoning=f"Domain-specific analysis for {self.domain.value}",
            metadata={},
            sources_used=sources
        )

class EcommerceExpert(DomainExpert):
    """Specialized expert for ecommerce domain"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(DomainType.ECOMMERCE, model_config)
        
        # Ecommerce-specific layers
        self.product_verifier = nn.Linear(self.hidden_dim, 256)
        self.pricing_verifier = nn.Linear(self.hidden_dim, 256)
        self.inventory_verifier = nn.Linear(self.hidden_dim, 256)
        
    def predict(self, text: str, context: str = "") -> ExpertPrediction:
        """Ecommerce-specific prediction"""
        # Extract product information
        products = self._extract_products(text)
        pricing_info = self._extract_pricing(text)
        inventory_info = self._extract_inventory(text)
        # Verify against ecommerce knowledge base
        product_verified = self._verify_products(products)
        pricing_verified = self._verify_pricing(pricing_info)
        inventory_verified = self._verify_inventory(inventory_info)
        # Calculate confidence based on verification results
        verification_scores = [product_verified, pricing_verified, inventory_verified]
        confidence = np.mean([score for score in verification_scores if score > 0])
        sources = ["ecommerce_knowledge_base"]
        return ExpertPrediction(
            expert_id=self.expert_id,
            domain=self.domain,
            confidence=confidence,
            prediction={
                'verified': confidence > 0.7,
                'quality_score': confidence,
                'hallucination_risk': 1.0 - confidence,
                'product_verification': product_verified,
                'pricing_verification': pricing_verified,
                'inventory_verification': inventory_verified
            },
            reasoning=f"Ecommerce verification: Products({product_verified:.2f}), Pricing({pricing_verified:.2f}), Inventory({inventory_verified:.2f})",
            metadata={
                'domain': self.domain.value,
                'products_found': len(products),
                'pricing_found': len(pricing_info),
                'inventory_found': len(inventory_info)
            },
            sources_used=sources
        )
    
    def _extract_products(self, text: str) -> List[str]:
        """Extract product information from text"""
        # Simple extraction - in production, use NER models
        products = []
        # Add logic to extract product names, SKUs, etc.
        return products
    
    def _extract_pricing(self, text: str) -> List[Dict[str, Any]]:
        """Extract pricing information from text"""
        pricing = []
        # Add logic to extract prices, currencies, discounts, etc.
        return pricing
    
    def _extract_inventory(self, text: str) -> List[Dict[str, Any]]:
        """Extract inventory information from text"""
        inventory = []
        # Add logic to extract stock levels, availability, etc.
        return inventory
    
    def _verify_products(self, products: List[str]) -> float:
        """Verify product information"""
        if not products:
            return 0.5  # Neutral if no products found
        
        # Verify against product database
        verified_count = 0
        for product in products:
            # Check against product knowledge base
            if self._check_product_exists(product):
                verified_count += 1
        
        return verified_count / len(products) if products else 0.5
    
    def _verify_pricing(self, pricing: List[Dict[str, Any]]) -> float:
        """Verify pricing information"""
        if not pricing:
            return 0.5
        
        # Verify pricing against market data
        verified_count = 0
        for price_info in pricing:
            if self._check_price_reasonable(price_info):
                verified_count += 1
        
        return verified_count / len(pricing) if pricing else 0.5
    
    def _verify_inventory(self, inventory: List[Dict[str, Any]]) -> float:
        """Verify inventory information"""
        if not inventory:
            return 0.5
        
        # Verify inventory against real-time data
        verified_count = 0
        for inv_info in inventory:
            if self._check_inventory_accurate(inv_info):
                verified_count += 1
        
        return verified_count / len(inventory) if inventory else 0.5
    
    def _check_product_exists(self, product: str) -> bool:
        """Check if product exists in knowledge base"""
        # This would query a product database
        return True  # Placeholder
    
    def _check_price_reasonable(self, price_info: Dict[str, Any]) -> bool:
        """Check if price is reasonable"""
        # This would compare against market prices
        return True  # Placeholder
    
    def _check_inventory_accurate(self, inv_info: Dict[str, Any]) -> bool:
        """Check if inventory information is accurate"""
        # This would query real-time inventory
        return True  # Placeholder

class BankingExpert(DomainExpert):
    """Specialized expert for banking domain"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(DomainType.BANKING, model_config)
        
        # Banking-specific layers
        self.account_verifier = nn.Linear(self.hidden_dim, 256)
        self.transaction_verifier = nn.Linear(self.hidden_dim, 256)
        self.regulatory_verifier = nn.Linear(self.hidden_dim, 256)
        
    def predict(self, text: str, context: str = "") -> ExpertPrediction:
        """Banking-specific prediction"""
        # Extract banking information
        accounts = self._extract_accounts(text)
        transactions = self._extract_transactions(text)
        regulatory_info = self._extract_regulatory(text)
        
        # Verify against banking knowledge base
        account_verified = self._verify_accounts(accounts)
        transaction_verified = self._verify_transactions(transactions)
        regulatory_verified = self._verify_regulatory(regulatory_info)
        
        # Calculate confidence
        verification_scores = [account_verified, transaction_verified, regulatory_verified]
        confidence = np.mean([score for score in verification_scores if score > 0])
        
        return ExpertPrediction(
            expert_id=self.expert_id,
            domain=self.domain,
            confidence=confidence,
            prediction={
                'verified': confidence > 0.7,
                'quality_score': confidence,
                'hallucination_risk': 1.0 - confidence,
                'account_verification': account_verified,
                'transaction_verification': transaction_verified,
                'regulatory_verification': regulatory_verified
            },
            reasoning=f"Banking verification: Accounts({account_verified:.2f}), Transactions({transaction_verified:.2f}), Regulatory({regulatory_verified:.2f})",
            metadata={
                'domain': self.domain.value,
                'accounts_found': len(accounts),
                'transactions_found': len(transactions),
                'regulatory_found': len(regulatory_info)
            },
            sources_used=["banking_knowledge_base"]
        )
    
    def _extract_accounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract account information"""
        accounts = []
        # Add logic to extract account numbers, types, balances, etc.
        return accounts
    
    def _extract_transactions(self, text: str) -> List[Dict[str, Any]]:
        """Extract transaction information"""
        transactions = []
        # Add logic to extract transaction details
        return transactions
    
    def _extract_regulatory(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulatory information"""
        regulatory = []
        # Add logic to extract compliance information
        return regulatory
    
    def _verify_accounts(self, accounts: List[Dict[str, Any]]) -> float:
        """Verify account information"""
        if not accounts:
            return 0.5
        
        verified_count = 0
        for account in accounts:
            if self._check_account_valid(account):
                verified_count += 1
        
        return verified_count / len(accounts) if accounts else 0.5
    
    def _verify_transactions(self, transactions: List[Dict[str, Any]]) -> float:
        """Verify transaction information"""
        if not transactions:
            return 0.5
        
        verified_count = 0
        for transaction in transactions:
            if self._check_transaction_valid(transaction):
                verified_count += 1
        
        return verified_count / len(transactions) if transactions else 0.5
    
    def _verify_regulatory(self, regulatory: List[Dict[str, Any]]) -> float:
        """Verify regulatory compliance"""
        if not regulatory:
            return 0.5
        
        verified_count = 0
        for reg_info in regulatory:
            if self._check_regulatory_compliant(reg_info):
                verified_count += 1
        
        return verified_count / len(regulatory) if regulatory else 0.5
    
    def _check_account_valid(self, account: Dict[str, Any]) -> bool:
        """Check if account information is valid"""
        return True  # Placeholder
    
    def _check_transaction_valid(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction is valid"""
        return True  # Placeholder
    
    def _check_regulatory_compliant(self, reg_info: Dict[str, Any]) -> bool:
        """Check regulatory compliance"""
        return True  # Placeholder

class InsuranceExpert(DomainExpert):
    """Specialized expert for insurance domain"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(DomainType.INSURANCE, model_config)
        
        # Insurance-specific layers
        self.policy_verifier = nn.Linear(self.hidden_dim, 256)
        self.coverage_verifier = nn.Linear(self.hidden_dim, 256)
        self.claim_verifier = nn.Linear(self.hidden_dim, 256)
        
    def predict(self, text: str, context: str = "") -> ExpertPrediction:
        """Insurance-specific prediction"""
        # Extract insurance information
        policies = self._extract_policies(text)
        coverage = self._extract_coverage(text)
        claims = self._extract_claims(text)
        
        # Verify against insurance knowledge base
        policy_verified = self._verify_policies(policies)
        coverage_verified = self._verify_coverage(coverage)
        claim_verified = self._verify_claims(claims)
        
        # Calculate confidence
        verification_scores = [policy_verified, coverage_verified, claim_verified]
        confidence = np.mean([score for score in verification_scores if score > 0])
        
        return ExpertPrediction(
            expert_id=self.expert_id,
            domain=self.domain,
            confidence=confidence,
            prediction={
                'verified': confidence > 0.7,
                'quality_score': confidence,
                'hallucination_risk': 1.0 - confidence,
                'policy_verification': policy_verified,
                'coverage_verification': coverage_verified,
                'claim_verification': claim_verified
            },
            reasoning=f"Insurance verification: Policies({policy_verified:.2f}), Coverage({coverage_verified:.2f}), Claims({claim_verified:.2f})",
            metadata={
                'domain': self.domain.value,
                'policies_found': len(policies),
                'coverage_found': len(coverage),
                'claims_found': len(claims)
            },
            sources_used=["insurance_knowledge_base"]
        )
    
    def _extract_policies(self, text: str) -> List[Dict[str, Any]]:
        """Extract policy information"""
        policies = []
        # Add logic to extract policy details
        return policies
    
    def _extract_coverage(self, text: str) -> List[Dict[str, Any]]:
        """Extract coverage information"""
        coverage = []
        # Add logic to extract coverage details
        return coverage
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claim information"""
        claims = []
        # Add logic to extract claim details
        return claims
    
    def _verify_policies(self, policies: List[Dict[str, Any]]) -> float:
        """Verify policy information"""
        if not policies:
            return 0.5
        
        verified_count = 0
        for policy in policies:
            if self._check_policy_valid(policy):
                verified_count += 1
        
        return verified_count / len(policies) if policies else 0.5
    
    def _verify_coverage(self, coverage: List[Dict[str, Any]]) -> float:
        """Verify coverage information"""
        if not coverage:
            return 0.5
        
        verified_count = 0
        for cov in coverage:
            if self._check_coverage_valid(cov):
                verified_count += 1
        
        return verified_count / len(coverage) if coverage else 0.5
    
    def _verify_claims(self, claims: List[Dict[str, Any]]) -> float:
        """Verify claim information"""
        if not claims:
            return 0.5
        
        verified_count = 0
        for claim in claims:
            if self._check_claim_valid(claim):
                verified_count += 1
        
        return verified_count / len(claims) if claims else 0.5
    
    def _check_policy_valid(self, policy: Dict[str, Any]) -> bool:
        """Check if policy is valid"""
        return True  # Placeholder
    
    def _check_coverage_valid(self, coverage: Dict[str, Any]) -> bool:
        """Check if coverage is valid"""
        return True  # Placeholder
    
    def _check_claim_valid(self, claim: Dict[str, Any]) -> bool:
        """Check if claim is valid"""
        return True  # Placeholder

class DomainRouter(nn.Module):
    """Router for selecting appropriate experts"""
    
    def __init__(self, num_experts: int, embedding_dim: int = 768):
        super().__init__()
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(DomainType)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route to experts and classify domain"""
        expert_weights = self.router(embeddings)
        domain_probs = self.domain_classifier(embeddings)
        
        return expert_weights, domain_probs

class MoEDomainVerifier:
    """Mixture of Experts for domain-specific verification"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize experts
        self.experts = self._initialize_experts()
        
        # Initialize router
        self.router = DomainRouter(len(self.experts), self.config['embedding_dim'])
        
        # Expert selection history
        self.expert_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'expert_usage': {expert_id: 0 for expert_id in self.experts.keys()},
            'domain_accuracy': {domain.value: [] for domain in DomainType}
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'embedding_dim': 768,
            'expert_configs': {
                DomainType.ECOMMERCE: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.BANKING: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.INSURANCE: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.HEALTHCARE: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.LEGAL: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.FINANCE: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.TECHNOLOGY: {'hidden_dim': 512, 'num_classes': 3},
                DomainType.GENERAL: {'hidden_dim': 512, 'num_classes': 3}
            },
            'confidence_threshold': 0.7,
            'ensemble_method': 'weighted_average'
        }
    
    def _initialize_experts(self) -> Dict[str, DomainExpert]:
        """Initialize domain-specific experts"""
        experts = {}
        
        # Create specialized experts
        experts['ecommerce'] = EcommerceExpert(self.config['expert_configs'][DomainType.ECOMMERCE])
        experts['banking'] = BankingExpert(self.config['expert_configs'][DomainType.BANKING])
        experts['insurance'] = InsuranceExpert(self.config['expert_configs'][DomainType.INSURANCE])
        
        # Add more experts as needed
        for domain in [DomainType.HEALTHCARE, DomainType.LEGAL, DomainType.FINANCE, DomainType.TECHNOLOGY]:
            experts[domain.value] = DomainExpert(domain, self.config['expert_configs'][domain])
        
        return experts
    
    async def verify_text(self, text: str, context: str = "") -> MoEVerificationResult:
        """Verify text using mixture of experts"""
        logger.info(f"MoE verification for text: {text[:100]}...")
        
        try:
            # Step 1: Domain detection and expert routing
            domain_detected, expert_weights = await self._detect_domain_and_route(text)
            
            # Step 2: Get predictions from all experts
            expert_predictions = await self._get_expert_predictions(text, context, expert_weights)
            
            # Step 3: Ensemble decision making
            ensemble_decision = self._make_ensemble_decision(expert_predictions, expert_weights)
            
            # Step 4: Calculate final metrics
            verified = ensemble_decision['verified']
            confidence = ensemble_decision['confidence']
            quality_score = ensemble_decision['quality_score']
            hallucination_risk = ensemble_decision['hallucination_risk']
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(expert_predictions, ensemble_decision)
            
            # Step 6: Update performance metrics
            self._update_performance_metrics(domain_detected, verified, expert_predictions)
            
            # Create result
            all_sources = []
            for pred in expert_predictions:
                if pred.sources_used:
                    all_sources.extend(pred.sources_used)
            result = MoEVerificationResult(
                verified=verified,
                confidence=confidence,
                expert_predictions=expert_predictions,
                ensemble_decision=ensemble_decision,
                domain_detected=domain_detected,
                quality_score=quality_score,
                hallucination_risk=hallucination_risk,
                recommendations=recommendations,
                sources_used=all_sources
            )
            
            logger.info(f"MoE verification completed: {verified} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"MoE verification failed: {e}")
            # Return fallback result
            return self._create_fallback_result(text, str(e))
    
    async def _detect_domain_and_route(self, text: str) -> Tuple[DomainType, Dict[str, float]]:
        """Detect domain and route to appropriate experts"""
        # This would use the router to determine expert weights
        # For now, use simple keyword-based routing
        
        text_lower = text.lower()
        
        # Domain keywords
        domain_keywords = {
            DomainType.ECOMMERCE: ['product', 'price', 'shipping', 'inventory', 'store', 'shop', 'buy', 'sell'],
            DomainType.BANKING: ['account', 'balance', 'transaction', 'transfer', 'loan', 'credit', 'debit'],
            DomainType.INSURANCE: ['policy', 'coverage', 'claim', 'premium', 'deductible', 'insurance'],
            DomainType.HEALTHCARE: ['medical', 'health', 'doctor', 'patient', 'treatment', 'diagnosis'],
            DomainType.LEGAL: ['law', 'legal', 'contract', 'court', 'attorney', 'case'],
            DomainType.FINANCE: ['investment', 'stock', 'market', 'portfolio', 'fund'],
            DomainType.TECHNOLOGY: ['software', 'hardware', 'code', 'system', 'technology', 'app']
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Find primary domain
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_domain = DomainType.GENERAL
        
        # Calculate expert weights
        expert_weights = {}
        total_score = sum(domain_scores.values()) + 1  # Add 1 for general
        
        for domain in DomainType:
            expert_id = domain.value
            if expert_id in self.experts:
                weight = (domain_scores.get(domain, 0) + 0.1) / total_score
                expert_weights[expert_id] = weight
        
        # Normalize weights
        total_weight = sum(expert_weights.values())
        if total_weight > 0:
            expert_weights = {k: v / total_weight for k, v in expert_weights.items()}
        
        return primary_domain, expert_weights
    
    async def _get_expert_predictions(self, 
                                    text: str, 
                                    context: str, 
                                    expert_weights: Dict[str, float]) -> List[ExpertPrediction]:
        """Get predictions from all experts"""
        predictions = []
        
        for expert_id, weight in expert_weights.items():
            if weight > 0.1:  # Only use experts with significant weight
                expert = self.experts[expert_id]
                try:
                    prediction = expert.predict(text, context)
                    # Adjust confidence based on expert weight
                    prediction.confidence *= weight
                    predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Expert {expert_id} prediction failed: {e}")
                    # Add fallback prediction
                    predictions.append(ExpertPrediction(
                        expert_id=expert_id,
                        domain=expert.domain,
                        confidence=0.1,
                        prediction={'verified': False, 'quality_score': 0.1, 'hallucination_risk': 0.9},
                        reasoning=f"Expert failed: {str(e)}",
                        metadata={'error': str(e)}
                    ))
        
        return predictions
    
    def _make_ensemble_decision(self, 
                               predictions: List[ExpertPrediction], 
                               expert_weights: Dict[str, float]) -> Dict[str, Any]:
        """Make ensemble decision from expert predictions"""
        if not predictions:
            return {
                'verified': False,
                'confidence': 0.0,
                'quality_score': 0.0,
                'hallucination_risk': 1.0,
                'method': 'no_experts'
            }
        
        # Weighted ensemble
        total_verified_score = 0.0
        total_quality_score = 0.0
        total_hallucination_risk = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        
        for prediction in predictions:
            weight = expert_weights.get(prediction.expert_id, 0.1)
            total_weight += weight
            
            # Weighted scores
            verified_score = (1.0 if prediction.prediction['verified'] else 0.0) * weight
            quality_score = prediction.prediction['quality_score'] * weight
            hallucination_risk = prediction.prediction['hallucination_risk'] * weight
            confidence = prediction.confidence * weight
            
            total_verified_score += verified_score
            total_quality_score += quality_score
            total_hallucination_risk += hallucination_risk
            total_confidence += confidence
        
        # Normalize by total weight
        if total_weight > 0:
            verified_score = total_verified_score / total_weight
            quality_score = total_quality_score / total_weight
            hallucination_risk = total_hallucination_risk / total_weight
            confidence = total_confidence / total_weight
        else:
            verified_score = 0.0
            quality_score = 0.0
            hallucination_risk = 1.0
            confidence = 0.0
        
        # Final decision
        verified = verified_score > self.config['confidence_threshold']
        
        return {
            'verified': verified,
            'confidence': confidence,
            'quality_score': quality_score,
            'hallucination_risk': hallucination_risk,
            'verified_score': verified_score,
            'method': 'weighted_ensemble',
            'expert_weights': expert_weights
        }
    
    def _generate_recommendations(self, 
                                predictions: List[ExpertPrediction], 
                                ensemble_decision: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on expert predictions"""
        recommendations = []
        
        # Overall quality recommendations
        if ensemble_decision['quality_score'] < 0.5:
            recommendations.append("Low quality score detected. Consider data cleaning and validation.")
        
        if ensemble_decision['hallucination_risk'] > 0.7:
            recommendations.append("High hallucination risk detected. Verify information with authoritative sources.")
        
        # Expert-specific recommendations
        for prediction in predictions:
            if prediction.confidence < 0.5:
                recommendations.append(f"Low confidence from {prediction.expert_id}. Consider domain-specific validation.")
            
            if 'error' in prediction.metadata:
                recommendations.append(f"Expert {prediction.expert_id} encountered errors. Check system health.")
        
        # Domain-specific recommendations
        if ensemble_decision['verified']:
            recommendations.append("Information verified across multiple expert domains.")
        else:
            recommendations.append("Information could not be verified. Manual review recommended.")
        
        return recommendations
    
    def _update_performance_metrics(self, 
                                  domain: DomainType, 
                                  verified: bool, 
                                  predictions: List[ExpertPrediction]):
        """Update performance metrics"""
        self.performance_metrics['total_verifications'] += 1
        
        if verified:
            self.performance_metrics['successful_verifications'] += 1
        
        # Update expert usage
        for prediction in predictions:
            expert_id = prediction.expert_id
            if expert_id in self.performance_metrics['expert_usage']:
                self.performance_metrics['expert_usage'][expert_id] += 1
        
        # Update domain accuracy
        if domain.value in self.performance_metrics['domain_accuracy']:
            self.performance_metrics['domain_accuracy'][domain.value].append(verified)
    
    def _create_fallback_result(self, text: str, error: str) -> MoEVerificationResult:
        """Create fallback result when verification fails"""
        return MoEVerificationResult(
            verified=False,
            confidence=0.0,
            expert_predictions=[],
            ensemble_decision={
                'verified': False,
                'confidence': 0.0,
                'quality_score': 0.0,
                'hallucination_risk': 1.0,
                'method': 'fallback',
                'error': error
            },
            domain_detected=DomainType.GENERAL,
            quality_score=0.0,
            hallucination_risk=1.0,
            recommendations=[f"Verification failed: {error}. Manual review required."],
            sources_used=[]
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate success rate
        if metrics['total_verifications'] > 0:
            metrics['success_rate'] = metrics['successful_verifications'] / metrics['total_verifications']
        else:
            metrics['success_rate'] = 0.0
        
        # Calculate domain accuracy
        for domain, results in metrics['domain_accuracy'].items():
            if results:
                metrics['domain_accuracy'][domain] = sum(results) / len(results)
            else:
                metrics['domain_accuracy'][domain] = 0.0
        
        return metrics
    
    async def train_expert(self, expert_id: str, training_data: List[Dict[str, Any]]):
        """Train a specific expert with new data"""
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not found")
        
        expert = self.experts[expert_id]
        logger.info(f"Training expert {expert_id} with {len(training_data)} samples")
        
        # This would implement actual training logic
        # For now, just update the expert's performance history
        for data in training_data:
            if 'accuracy' in data:
                expert.accuracy_history.append(data['accuracy'])
            if 'confidence' in data:
                expert.confidence_history.append(data['confidence'])
        
        logger.info(f"Expert {expert_id} training completed")

# Example usage
async def main():
    """Example usage of MoE Domain Verifier"""
    
    # Initialize MoE verifier
    moe_verifier = MoEDomainVerifier()
    
    # Sample texts for different domains
    sample_texts = [
        "The iPhone 15 costs $999 and is available in all Apple stores.",
        "Your account balance is $5,000 and all transactions are up to date.",
        "Your comprehensive policy covers all damages up to $50,000.",
        "The patient's diagnosis shows early-stage diabetes requiring immediate treatment.",
        "The contract terms specify a 30-day payment period with late fees."
    ]
    
    # Verify each text
    for i, text in enumerate(sample_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        result = await moe_verifier.verify_text(text)
        
        print(f"Domain: {result.domain_detected.value}")
        print(f"Verified: {result.verified}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Hallucination Risk: {result.hallucination_risk:.3f}")
        print(f"Selected Experts: {result.expert_predictions}")
        print(f"Recommendations: {result.recommendations}")
        print(f"Sources Used: {result.sources_used}")
    
    # Get performance metrics
    metrics = moe_verifier.get_performance_metrics()
    print(f"\n--- Performance Metrics ---")
    print(f"Total Verifications: {metrics['total_verifications']}")
    print(f"Success Rate: {metrics['success_rate']:.3f}")
    print(f"Expert Usage: {metrics['expert_usage']}")

if __name__ == "__main__":
    asyncio.run(main()) 