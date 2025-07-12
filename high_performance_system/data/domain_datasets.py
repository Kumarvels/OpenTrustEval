"""
Domain-Specific Datasets for MoE Training
DeepSeek-style approach with specialized datasets for each domain

Features:
- Domain-specific training datasets
- Quality validation datasets
- Hallucination detection datasets
- Continuous learning datasets
- Multi-domain cross-validation
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DomainDataset:
    """Domain-specific dataset"""
    domain: str
    dataset_type: str  # 'training', 'validation', 'testing', 'quality', 'hallucination'
    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    version: str

class DomainDatasetManager:
    """Manager for domain-specific datasets"""
    
    def __init__(self, data_dir: str = "high_performance_system/data/domain_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain-specific dataset configurations
        self.domain_configs = {
            'ecommerce': {
                'entities': ['products', 'prices', 'inventory', 'shipping', 'reviews'],
                'validation_sources': ['product_catalogs', 'price_apis', 'inventory_systems'],
                'quality_metrics': ['accuracy', 'completeness', 'consistency', 'timeliness']
            },
            'banking': {
                'entities': ['accounts', 'transactions', 'balances', 'loans', 'cards'],
                'validation_sources': ['core_banking_systems', 'transaction_apis', 'regulatory_dbs'],
                'quality_metrics': ['security', 'compliance', 'accuracy', 'privacy']
            },
            'insurance': {
                'entities': ['policies', 'claims', 'coverage', 'premiums', 'risks'],
                'validation_sources': ['policy_systems', 'claims_databases', 'actuarial_data'],
                'quality_metrics': ['coverage_accuracy', 'risk_assessment', 'compliance', 'fraud_detection']
            },
            'healthcare': {
                'entities': ['patients', 'diagnoses', 'treatments', 'medications', 'procedures'],
                'validation_sources': ['medical_records', 'drug_databases', 'clinical_guidelines'],
                'quality_metrics': ['medical_accuracy', 'safety', 'compliance', 'privacy']
            },
            'legal': {
                'entities': ['cases', 'contracts', 'regulations', 'precedents', 'clients'],
                'validation_sources': ['legal_databases', 'court_records', 'regulatory_sources'],
                'quality_metrics': ['legal_accuracy', 'compliance', 'precedent_alignment', 'jurisdiction']
            },
            'finance': {
                'entities': ['investments', 'portfolios', 'markets', 'securities', 'analytics'],
                'validation_sources': ['market_data', 'financial_apis', 'regulatory_reports'],
                'quality_metrics': ['market_accuracy', 'risk_assessment', 'compliance', 'performance']
            },
            'technology': {
                'entities': ['software', 'hardware', 'systems', 'algorithms', 'protocols'],
                'validation_sources': ['technical_docs', 'api_documentation', 'code_repositories'],
                'quality_metrics': ['technical_accuracy', 'functionality', 'security', 'performance']
            }
        }

    def create_ecommerce_dataset(self) -> DomainDataset:
        """Create ecommerce domain dataset"""
        samples = [
            {
                'text': 'The iPhone 15 Pro costs $999 and is available in all Apple stores.',
                'entities': {
                    'products': ['iPhone 15 Pro'],
                    'prices': ['$999'],
                    'availability': ['available in all Apple stores']
                },
                'verified': True,
                'confidence': 0.95,
                'quality_score': 0.9,
                'hallucination_risk': 0.05,
                'validation_sources': ['apple.com', 'product_catalog'],
                'metadata': {
                    'product_category': 'electronics',
                    'brand': 'Apple',
                    'price_range': 'high'
                }
            },
            {
                'text': 'Free shipping on orders over $50 with 2-day delivery.',
                'entities': {
                    'shipping': ['free shipping'],
                    'prices': ['$50'],
                    'delivery': ['2-day delivery']
                },
                'verified': True,
                'confidence': 0.88,
                'quality_score': 0.85,
                'hallucination_risk': 0.12,
                'validation_sources': ['shipping_policy', 'delivery_api'],
                'metadata': {
                    'shipping_type': 'free',
                    'threshold': 50,
                    'delivery_speed': 'fast'
                }
            },
            {
                'text': 'This product has 4.5 stars from 1,234 customer reviews.',
                'entities': {
                    'reviews': ['4.5 stars'],
                    'customers': ['1,234 customer reviews']
                },
                'verified': True,
                'confidence': 0.92,
                'quality_score': 0.88,
                'hallucination_risk': 0.08,
                'validation_sources': ['review_system', 'customer_database'],
                'metadata': {
                    'rating': 4.5,
                    'review_count': 1234,
                    'sentiment': 'positive'
                }
            },
            {
                'text': 'The Samsung Galaxy S24 costs $1,199 and comes with 256GB storage.',
                'entities': {
                    'products': ['Samsung Galaxy S24'],
                    'prices': ['$1,199'],
                    'specifications': ['256GB storage']
                },
                'verified': True,
                'confidence': 0.94,
                'quality_score': 0.92,
                'hallucination_risk': 0.06,
                'validation_sources': ['samsung.com', 'product_specs'],
                'metadata': {
                    'product_category': 'electronics',
                    'brand': 'Samsung',
                    'storage': '256GB'
                }
            },
            {
                'text': 'Limited time offer: 30% off all clothing items.',
                'entities': {
                    'promotions': ['30% off'],
                    'categories': ['clothing items'],
                    'timing': ['limited time offer']
                },
                'verified': True,
                'confidence': 0.87,
                'quality_score': 0.83,
                'hallucination_risk': 0.13,
                'validation_sources': ['promotion_system', 'inventory_db'],
                'metadata': {
                    'discount_percentage': 30,
                    'category': 'clothing',
                    'promotion_type': 'percentage_off'
                }
            }
        ]
        
        return DomainDataset(
            domain='ecommerce',
            dataset_type='training',
            samples=samples,
            metadata={
                'total_samples': len(samples),
                'verified_samples': sum(1 for s in samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def create_banking_dataset(self) -> DomainDataset:
        """Create banking domain dataset"""
        samples = [
            {
                'text': 'Your checking account balance is $5,247.89 as of today.',
                'entities': {
                    'accounts': ['checking account'],
                    'balances': ['$5,247.89'],
                    'timing': ['today']
                },
                'verified': True,
                'confidence': 0.96,
                'quality_score': 0.94,
                'hallucination_risk': 0.04,
                'validation_sources': ['core_banking_system', 'account_api'],
                'metadata': {
                    'account_type': 'checking',
                    'balance_amount': 5247.89,
                    'currency': 'USD'
                }
            },
            {
                'text': 'Your credit card payment of $150.00 was processed successfully.',
                'entities': {
                    'cards': ['credit card'],
                    'transactions': ['payment of $150.00'],
                    'status': ['processed successfully']
                },
                'verified': True,
                'confidence': 0.93,
                'quality_score': 0.91,
                'hallucination_risk': 0.07,
                'validation_sources': ['payment_system', 'transaction_db'],
                'metadata': {
                    'transaction_type': 'payment',
                    'amount': 150.00,
                    'status': 'successful'
                }
            },
            {
                'text': 'Your loan application has been approved with a 3.5% interest rate.',
                'entities': {
                    'loans': ['loan application'],
                    'status': ['approved'],
                    'rates': ['3.5% interest rate']
                },
                'verified': True,
                'confidence': 0.89,
                'quality_score': 0.87,
                'hallucination_risk': 0.11,
                'validation_sources': ['loan_system', 'approval_workflow'],
                'metadata': {
                    'application_status': 'approved',
                    'interest_rate': 3.5,
                    'rate_type': 'annual'
                }
            },
            {
                'text': 'Your savings account earned $12.45 in interest this month.',
                'entities': {
                    'accounts': ['savings account'],
                    'earnings': ['$12.45 in interest'],
                    'timing': ['this month']
                },
                'verified': True,
                'confidence': 0.91,
                'quality_score': 0.89,
                'hallucination_risk': 0.09,
                'validation_sources': ['interest_calculation', 'account_statement'],
                'metadata': {
                    'account_type': 'savings',
                    'interest_earned': 12.45,
                    'period': 'monthly'
                }
            },
            {
                'text': 'Your account has been temporarily frozen due to suspicious activity.',
                'entities': {
                    'accounts': ['account'],
                    'status': ['temporarily frozen'],
                    'reason': ['suspicious activity']
                },
                'verified': True,
                'confidence': 0.88,
                'quality_score': 0.86,
                'hallucination_risk': 0.12,
                'validation_sources': ['fraud_detection', 'security_system'],
                'metadata': {
                    'account_status': 'frozen',
                    'freeze_reason': 'suspicious_activity',
                    'duration': 'temporary'
                }
            }
        ]
        
        return DomainDataset(
            domain='banking',
            dataset_type='training',
            samples=samples,
            metadata={
                'total_samples': len(samples),
                'verified_samples': sum(1 for s in samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def create_insurance_dataset(self) -> DomainDataset:
        """Create insurance domain dataset"""
        samples = [
            {
                'text': 'Your comprehensive auto policy covers up to $50,000 in damages.',
                'entities': {
                    'policies': ['comprehensive auto policy'],
                    'coverage': ['$50,000 in damages']
                },
                'verified': True,
                'confidence': 0.94,
                'quality_score': 0.92,
                'hallucination_risk': 0.06,
                'validation_sources': ['policy_system', 'coverage_database'],
                'metadata': {
                    'policy_type': 'comprehensive_auto',
                    'coverage_limit': 50000,
                    'coverage_type': 'damages'
                }
            },
            {
                'text': 'Your claim for $2,500 has been approved and payment will be processed within 5 business days.',
                'entities': {
                    'claims': ['claim for $2,500'],
                    'status': ['approved'],
                    'timing': ['within 5 business days']
                },
                'verified': True,
                'confidence': 0.91,
                'quality_score': 0.89,
                'hallucination_risk': 0.09,
                'validation_sources': ['claims_system', 'approval_workflow'],
                'metadata': {
                    'claim_amount': 2500,
                    'claim_status': 'approved',
                    'payment_timing': '5_business_days'
                }
            },
            {
                'text': 'Your monthly premium is $125.00 and covers medical, dental, and vision.',
                'entities': {
                    'premiums': ['$125.00'],
                    'coverage': ['medical, dental, and vision']
                },
                'verified': True,
                'confidence': 0.93,
                'quality_score': 0.91,
                'hallucination_risk': 0.07,
                'validation_sources': ['premium_calculation', 'coverage_system'],
                'metadata': {
                    'premium_amount': 125.00,
                    'premium_frequency': 'monthly',
                    'coverage_types': ['medical', 'dental', 'vision']
                }
            },
            {
                'text': 'Your home insurance policy includes coverage for natural disasters up to $300,000.',
                'entities': {
                    'policies': ['home insurance policy'],
                    'coverage': ['natural disasters up to $300,000']
                },
                'verified': True,
                'confidence': 0.92,
                'quality_score': 0.90,
                'hallucination_risk': 0.08,
                'validation_sources': ['policy_database', 'coverage_calculator'],
                'metadata': {
                    'policy_type': 'home_insurance',
                    'coverage_limit': 300000,
                    'coverage_type': 'natural_disasters'
                }
            },
            {
                'text': 'Your life insurance policy has a death benefit of $500,000 and is active.',
                'entities': {
                    'policies': ['life insurance policy'],
                    'benefits': ['$500,000'],
                    'status': ['active']
                },
                'verified': True,
                'confidence': 0.95,
                'quality_score': 0.93,
                'hallucination_risk': 0.05,
                'validation_sources': ['policy_system', 'status_checker'],
                'metadata': {
                    'policy_type': 'life_insurance',
                    'death_benefit': 500000,
                    'policy_status': 'active'
                }
            }
        ]
        
        return DomainDataset(
            domain='insurance',
            dataset_type='training',
            samples=samples,
            metadata={
                'total_samples': len(samples),
                'verified_samples': sum(1 for s in samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def create_hallucination_dataset(self, domain: str) -> DomainDataset:
        """Create hallucination detection dataset for a domain"""
        hallucination_samples = [
            {
                'text': f'The {domain} product costs $999,999 and is available everywhere.',
                'entities': {},
                'verified': False,
                'confidence': 0.15,
                'quality_score': 0.1,
                'hallucination_risk': 0.95,
                'validation_sources': ['fact_check'],
                'metadata': {
                    'hallucination_type': 'exaggerated_pricing',
                    'domain': domain,
                    'severity': 'high'
                }
            },
            {
                'text': f'All {domain} services are completely free and have no limitations.',
                'entities': {},
                'verified': False,
                'confidence': 0.12,
                'quality_score': 0.08,
                'hallucination_risk': 0.98,
                'validation_sources': ['service_terms'],
                'metadata': {
                    'hallucination_type': 'unrealistic_claims',
                    'domain': domain,
                    'severity': 'high'
                }
            },
            {
                'text': f'The {domain} system has 100% uptime and never experiences any issues.',
                'entities': {},
                'verified': False,
                'confidence': 0.18,
                'quality_score': 0.12,
                'hallucination_risk': 0.92,
                'validation_sources': ['system_monitoring'],
                'metadata': {
                    'hallucination_type': 'perfection_claim',
                    'domain': domain,
                    'severity': 'medium'
                }
            }
        ]
        
        return DomainDataset(
            domain=domain,
            dataset_type='hallucination',
            samples=hallucination_samples,
            metadata={
                'total_samples': len(hallucination_samples),
                'verified_samples': sum(1 for s in hallucination_samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in hallucination_samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in hallucination_samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in hallucination_samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def save_dataset(self, dataset: DomainDataset) -> str:
        """Save dataset to file"""
        filename = f"{dataset.domain}_{dataset.dataset_type}_{dataset.version}.json"
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            'domain': dataset.domain,
            'dataset_type': dataset.dataset_type,
            'samples': dataset.samples,
            'metadata': dataset.metadata,
            'created_at': dataset.created_at.isoformat(),
            'version': dataset.version
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Dataset saved to {filepath}")
        return str(filepath)

    def load_dataset(self, filename: str) -> DomainDataset:
        """Load dataset from file"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return DomainDataset(
            domain=data['domain'],
            dataset_type=data['dataset_type'],
            samples=data['samples'],
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at']),
            version=data['version']
        )

    def create_quality_validation_dataset(self, domain: str) -> DomainDataset:
        """Create quality validation dataset"""
        quality_samples = [
            {
                'text': f'High-quality {domain} information with verified sources.',
                'entities': {},
                'verified': True,
                'confidence': 0.95,
                'quality_score': 0.95,
                'hallucination_risk': 0.05,
                'validation_sources': ['verified_source'],
                'metadata': {
                    'quality_type': 'high',
                    'domain': domain,
                    'verification_method': 'source_check'
                }
            },
            {
                'text': f'Medium-quality {domain} information with some uncertainty.',
                'entities': {},
                'verified': True,
                'confidence': 0.75,
                'quality_score': 0.75,
                'hallucination_risk': 0.25,
                'validation_sources': ['partial_verification'],
                'metadata': {
                    'quality_type': 'medium',
                    'domain': domain,
                    'verification_method': 'partial_check'
                }
            },
            {
                'text': f'Low-quality {domain} information requiring verification.',
                'entities': {},
                'verified': False,
                'confidence': 0.45,
                'quality_score': 0.45,
                'hallucination_risk': 0.55,
                'validation_sources': ['manual_review'],
                'metadata': {
                    'quality_type': 'low',
                    'domain': domain,
                    'verification_method': 'manual_review'
                }
            }
        ]
        
        return DomainDataset(
            domain=domain,
            dataset_type='quality_validation',
            samples=quality_samples,
            metadata={
                'total_samples': len(quality_samples),
                'verified_samples': sum(1 for s in quality_samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in quality_samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in quality_samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in quality_samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def create_cross_domain_dataset(self) -> DomainDataset:
        """Create cross-domain validation dataset"""
        cross_domain_samples = [
            {
                'text': 'The iPhone costs $999 and your bank account has $5,000 balance.',
                'entities': {
                    'ecommerce': ['iPhone', '$999'],
                    'banking': ['bank account', '$5,000 balance']
                },
                'verified': True,
                'confidence': 0.88,
                'quality_score': 0.85,
                'hallucination_risk': 0.12,
                'validation_sources': ['product_catalog', 'banking_system'],
                'metadata': {
                    'domains': ['ecommerce', 'banking'],
                    'cross_domain': True,
                    'complexity': 'medium'
                }
            },
            {
                'text': 'Your insurance policy covers $50,000 and your investment portfolio is worth $100,000.',
                'entities': {
                    'insurance': ['insurance policy', '$50,000'],
                    'finance': ['investment portfolio', '$100,000']
                },
                'verified': True,
                'confidence': 0.90,
                'quality_score': 0.87,
                'hallucination_risk': 0.10,
                'validation_sources': ['policy_system', 'portfolio_tracker'],
                'metadata': {
                    'domains': ['insurance', 'finance'],
                    'cross_domain': True,
                    'complexity': 'high'
                }
            }
        ]
        
        return DomainDataset(
            domain='cross_domain',
            dataset_type='validation',
            samples=cross_domain_samples,
            metadata={
                'total_samples': len(cross_domain_samples),
                'verified_samples': sum(1 for s in cross_domain_samples if s['verified']),
                'avg_confidence': np.mean([s['confidence'] for s in cross_domain_samples]),
                'avg_quality_score': np.mean([s['quality_score'] for s in cross_domain_samples]),
                'avg_hallucination_risk': np.mean([s['hallucination_risk'] for s in cross_domain_samples])
            },
            created_at=datetime.now(),
            version='1.0.0'
        )

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics for all datasets"""
        datasets = list(self.data_dir.glob("*.json"))
        
        stats = {
            'total_datasets': len(datasets),
            'domains': {},
            'dataset_types': {},
            'total_samples': 0,
            'verified_samples': 0
        }
        
        for dataset_file in datasets:
            try:
                dataset = self.load_dataset(dataset_file.name)
                
                # Domain statistics
                if dataset.domain not in stats['domains']:
                    stats['domains'][dataset.domain] = {
                        'datasets': 0,
                        'samples': 0,
                        'verified_samples': 0
                    }
                
                stats['domains'][dataset.domain]['datasets'] += 1
                stats['domains'][dataset.domain]['samples'] += len(dataset.samples)
                stats['domains'][dataset.domain]['verified_samples'] += sum(1 for s in dataset.samples if s['verified'])
                
                # Dataset type statistics
                if dataset.dataset_type not in stats['dataset_types']:
                    stats['dataset_types'][dataset.dataset_type] = 0
                stats['dataset_types'][dataset.dataset_type] += 1
                
                # Overall statistics
                stats['total_samples'] += len(dataset.samples)
                stats['verified_samples'] += sum(1 for s in dataset.samples if s['verified'])
                
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_file.name}: {e}")
        
        # Calculate verification rates
        if stats['total_samples'] > 0:
            stats['overall_verification_rate'] = stats['verified_samples'] / stats['total_samples']
        else:
            stats['overall_verification_rate'] = 0.0
        
        for domain in stats['domains']:
            if stats['domains'][domain]['samples'] > 0:
                stats['domains'][domain]['verification_rate'] = (
                    stats['domains'][domain]['verified_samples'] / stats['domains'][domain]['samples']
                )
            else:
                stats['domains'][domain]['verification_rate'] = 0.0
        
        return stats

# Example usage
def main():
    """Example usage of domain dataset manager"""
    
    # Initialize dataset manager
    manager = DomainDatasetManager()
    
    # Create datasets for different domains
    domains = ['ecommerce', 'banking', 'insurance']
    
    for domain in domains:
        print(f"\n--- Creating {domain} dataset ---")
        
        # Create training dataset
        if domain == 'ecommerce':
            dataset = manager.create_ecommerce_dataset()
        elif domain == 'banking':
            dataset = manager.create_banking_dataset()
        elif domain == 'insurance':
            dataset = manager.create_insurance_dataset()
        
        # Save dataset
        filepath = manager.save_dataset(dataset)
        print(f"Saved {domain} dataset: {filepath}")
        
        # Create hallucination dataset
        hallucination_dataset = manager.create_hallucination_dataset(domain)
        hallucination_filepath = manager.save_dataset(hallucination_dataset)
        print(f"Saved {domain} hallucination dataset: {hallucination_filepath}")
        
        # Create quality validation dataset
        quality_dataset = manager.create_quality_validation_dataset(domain)
        quality_filepath = manager.save_dataset(quality_dataset)
        print(f"Saved {domain} quality dataset: {quality_filepath}")
    
    # Create cross-domain dataset
    print("\n--- Creating cross-domain dataset ---")
    cross_domain_dataset = manager.create_cross_domain_dataset()
    cross_domain_filepath = manager.save_dataset(cross_domain_dataset)
    print(f"Saved cross-domain dataset: {cross_domain_filepath}")
    
    # Get statistics
    print("\n--- Dataset Statistics ---")
    stats = manager.get_dataset_statistics()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main() 