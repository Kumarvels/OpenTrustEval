"""
Comprehensive Test Suite for Ultimate MoE Solution - Phase 1 Complete

This test suite validates all Phase 1 components:
1. Advanced Expert Ensemble (10+ domains)
2. Intelligent Domain Router (multiple strategies)
3. Ultimate MoE System Integration
4. Enhanced Dataset Profiler
5. Comprehensive PII Detection
6. Advanced Trust Scoring

Tests performance, accuracy, and integration of all components.
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# Import all Phase 1 components
from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
from high_performance_system.core.intelligent_domain_router import IntelligentDomainRouter
from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
from high_performance_system.core.enhanced_dataset_profiler import EnhancedDatasetProfiler
from high_performance_system.core.comprehensive_pii_detector import ComprehensivePIIDetector
from high_performance_system.core.advanced_trust_scorer import AdvancedTrustScorer


class UltimateMoEPhase1Tester:
    """Comprehensive tester for Phase 1 completion"""
    
    def __init__(self):
        self.test_results = {
            'expert_ensemble': {},
            'intelligent_routing': {},
            'ultimate_system': {},
            'dataset_profiler': {},
            'pii_detector': {},
            'trust_scorer': {},
            'integration': {},
            'performance': {},
            'overall': {}
        }
        
        # Test data for different domains
        self.test_data = {
            'ecommerce': [
                "The new iPhone 15 Pro features a titanium design and A17 Pro chip. Available for pre-order starting at $999.",
                "Amazon Prime Day sale offers up to 50% off on electronics, books, and home goods.",
                "Customer reviews show 4.5/5 stars for the wireless headphones with noise cancellation."
            ],
            'banking': [
                "The Federal Reserve raised interest rates by 0.25% to combat inflation.",
                "Online banking transactions are secured with 256-bit encryption and multi-factor authentication.",
                "Credit card fraud detection systems prevented $2.3 billion in fraudulent transactions last year."
            ],
            'healthcare': [
                "Clinical trials show the new vaccine is 95% effective against the virus.",
                "Telemedicine appointments increased by 300% during the pandemic.",
                "The FDA approved a new treatment for diabetes with fewer side effects."
            ],
            'legal': [
                "The Supreme Court ruled 6-3 in favor of the plaintiff in the landmark case.",
                "New privacy regulations require explicit consent for data collection.",
                "Contract law requires consideration for an agreement to be legally binding."
            ],
            'technology': [
                "Artificial intelligence models achieved 98% accuracy in image recognition tasks.",
                "Blockchain technology provides decentralized and immutable transaction records.",
                "Cloud computing reduces infrastructure costs by up to 60% for businesses."
            ],
            'mixed_domain': [
                "The tech company's quarterly earnings report shows 25% revenue growth, with strong performance in e-commerce and cloud services.",
                "Healthcare providers are adopting AI-powered diagnostic tools while ensuring HIPAA compliance.",
                "Financial institutions are implementing blockchain solutions for secure cross-border transactions."
            ]
        }
        
        # PII test data
        self.pii_test_data = [
            "Contact John Smith at john.smith@example.com or call (555) 123-4567.",
            "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111, DOB: 01/15/1985",
            "Address: 123 Main Street, Anytown, CA 90210, IP: 192.168.1.1"
        ]
        
        # Trust test data
        self.trust_test_data = [
            {
                'text': "According to peer-reviewed research published in Nature in 2023, climate change has accelerated significantly.",
                'context': "Academic research publication",
                'expected_level': 'high'
            },
            {
                'text': "AMAZING! You won't BELIEVE this SECRET cure! Click here for the shocking truth!",
                'context': "Social media post",
                'expected_level': 'very_low'
            },
            {
                'text': "The weather forecast predicts rain tomorrow with 60% probability based on meteorological data.",
                'context': "Weather service",
                'expected_level': 'medium'
            }
        ]
    
    async def test_enhanced_dataset_profiler(self) -> Dict[str, Any]:
        """Test the Enhanced Dataset Profiler"""
        print("\n=== Testing Enhanced Dataset Profiler ===")
        
        profiler = EnhancedDatasetProfiler()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'accuracy': []
        }
        
        for domain, texts in self.test_data.items():
            for i, text in enumerate(texts):
                start_time = time.time()
                
                try:
                    profile = await profiler.profile_text(text)
                    processing_time = time.time() - start_time
                    
                    # Validate profile structure
                    required_fields = ['quality_score', 'complexity_analysis', 'domain_indicators', 
                                     'data_quality_metrics', 'readability_analysis', 'language_analysis']
                    
                    all_fields_present = all(hasattr(profile, field) for field in required_fields)
                    
                    if all_fields_present and 0 <= profile.quality_score <= 1:
                        results['tests_passed'] += 1
                        results['performance'].append(processing_time)
                        results['accuracy'].append(profile.quality_score)
                        
                        print(f"✅ {domain} text {i+1}: Quality={profile.quality_score:.3f}, Time={processing_time:.3f}s")
                    else:
                        results['tests_failed'] += 1
                        print(f"❌ {domain} text {i+1}: Invalid profile structure")
                
                except Exception as e:
                    results['tests_failed'] += 1
                    print(f"❌ {domain} text {i+1}: Error - {str(e)}")
        
        # Calculate performance metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_quality_score'] = sum(results['accuracy']) / len(results['accuracy'])
        
        print(f"Dataset Profiler Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_comprehensive_pii_detection(self) -> Dict[str, Any]:
        """Test the Comprehensive PII Detection system"""
        print("\n=== Testing Comprehensive PII Detection ===")
        
        detector = ComprehensivePIIDetector()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'detection_accuracy': []
        }
        
        for i, text in enumerate(self.pii_test_data):
            start_time = time.time()
            
            try:
                pii_result = await detector.detect_pii(text)
                processing_time = time.time() - start_time
                
                # Validate result structure
                required_fields = ['pii_score', 'detected_pii', 'privacy_risk', 'compliance_status']
                all_fields_present = all(hasattr(pii_result, field) for field in required_fields)
                
                if all_fields_present and 0 <= pii_result.pii_score <= 1:
                    results['tests_passed'] += 1
                    results['performance'].append(processing_time)
                    results['detection_accuracy'].append(pii_result.pii_score)
                    
                    print(f"✅ PII Test {i+1}: Score={pii_result.pii_score:.3f}, Risk={pii_result.privacy_risk.value}, Time={processing_time:.3f}s")
                    print(f"   Detected {len(pii_result.detected_pii)} PII items")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ PII Test {i+1}: Invalid result structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ PII Test {i+1}: Error - {str(e)}")
        
        # Calculate performance metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_detection_accuracy'] = sum(results['detection_accuracy']) / len(results['detection_accuracy'])
        
        print(f"PII Detection Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_advanced_trust_scoring(self) -> Dict[str, Any]:
        """Test the Advanced Trust Scoring system"""
        print("\n=== Testing Advanced Trust Scoring ===")
        
        scorer = AdvancedTrustScorer()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'trust_accuracy': []
        }
        
        for i, test_case in enumerate(self.trust_test_data):
            start_time = time.time()
            
            try:
                trust_result = await scorer.score_trust(test_case['text'], test_case['context'])
                processing_time = time.time() - start_time
                
                # Validate result structure
                required_fields = ['trust_score', 'trust_level', 'confidence_interval', 'risk_level']
                all_fields_present = all(hasattr(trust_result, field) for field in required_fields)
                
                if all_fields_present and 0 <= trust_result.trust_score <= 1:
                    results['tests_passed'] += 1
                    results['performance'].append(processing_time)
                    results['trust_accuracy'].append(trust_result.trust_score)
                    
                    # Check if trust level matches expected
                    expected_matched = trust_result.trust_level.value == test_case['expected_level']
                    
                    print(f"✅ Trust Test {i+1}: Score={trust_result.trust_score:.3f}, Level={trust_result.trust_level.value}, Time={processing_time:.3f}s")
                    if expected_matched:
                        print(f"   ✅ Expected level matched: {test_case['expected_level']}")
                    else:
                        print(f"   ⚠️ Expected {test_case['expected_level']}, got {trust_result.trust_level.value}")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Trust Test {i+1}: Invalid result structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Trust Test {i+1}: Error - {str(e)}")
        
        # Calculate performance metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_trust_accuracy'] = sum(results['trust_accuracy']) / len(results['trust_accuracy'])
        
        print(f"Trust Scoring Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_expert_ensemble(self) -> Dict[str, Any]:
        """Test the Advanced Expert Ensemble"""
        print("\n=== Testing Advanced Expert Ensemble ===")
        
        ensemble = AdvancedExpertEnsemble()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'domain_accuracy': {}
        }
        
        for domain, texts in self.test_data.items():
            domain_results = []
            
            for i, text in enumerate(texts):
                start_time = time.time()
                
                try:
                    expert_results = await ensemble.verify_with_all_experts(text)
                    ensemble_result = await ensemble.get_ensemble_verification(text)
                    processing_time = time.time() - start_time
                    
                    # Validate results
                    if (isinstance(expert_results, dict) and 
                        isinstance(ensemble_result, dict) and
                        'verification_score' in ensemble_result):
                        
                        results['tests_passed'] += 1
                        results['performance'].append(processing_time)
                        domain_results.append(ensemble_result['verification_score'])
                        
                        print(f"✅ {domain} text {i+1}: Score={ensemble_result['verification_score']:.3f}, Time={processing_time:.3f}s")
                    else:
                        results['tests_failed'] += 1
                        print(f"❌ {domain} text {i+1}: Invalid result structure")
                
                except Exception as e:
                    results['tests_failed'] += 1
                    print(f"❌ {domain} text {i+1}: Error - {str(e)}")
            
            # Calculate domain accuracy
            if domain_results:
                results['domain_accuracy'][domain] = sum(domain_results) / len(domain_results)
        
        # Calculate overall performance
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
        
        print(f"Expert Ensemble Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_intelligent_routing(self) -> Dict[str, Any]:
        """Test the Intelligent Domain Router"""
        print("\n=== Testing Intelligent Domain Router ===")
        
        router = IntelligentDomainRouter()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'routing_accuracy': {}
        }
        
        for domain, texts in self.test_data.items():
            domain_results = []
            
            for i, text in enumerate(texts):
                start_time = time.time()
                
                try:
                    routing_result = await router.route_to_experts(text)
                    processing_time = time.time() - start_time
                    
                    # Validate routing result
                    if (isinstance(routing_result, dict) and 
                        'primary_domain' in routing_result and
                        'expert_weights' in routing_result):
                        
                        results['tests_passed'] += 1
                        results['performance'].append(processing_time)
                        
                        # Check if primary domain matches expected
                        primary_domain = routing_result['primary_domain']
                        domain_results.append(primary_domain == domain)
                        
                        print(f"✅ {domain} text {i+1}: Primary={primary_domain}, Time={processing_time:.3f}s")
                    else:
                        results['tests_failed'] += 1
                        print(f"❌ {domain} text {i+1}: Invalid routing result")
                
                except Exception as e:
                    results['tests_failed'] += 1
                    print(f"❌ {domain} text {i+1}: Error - {str(e)}")
            
            # Calculate domain routing accuracy
            if domain_results:
                results['routing_accuracy'][domain] = sum(domain_results) / len(domain_results)
        
        # Calculate overall performance
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
        
        print(f"Intelligent Routing Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_ultimate_system_integration(self) -> Dict[str, Any]:
        """Test the complete Ultimate MoE System integration"""
        print("\n=== Testing Ultimate MoE System Integration ===")
        
        system = UltimateMoESystem()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'integration_scores': []
        }
        
        # Test with a subset of data to avoid overwhelming output
        test_subset = {
            'ecommerce': self.test_data['ecommerce'][:1],
            'healthcare': self.test_data['healthcare'][:1],
            'mixed_domain': self.test_data['mixed_domain'][:1]
        }
        
        for domain, texts in test_subset.items():
            for i, text in enumerate(texts):
                start_time = time.time()
                
                try:
                    verification_result = await system.verify_text(text, enable_advanced_features=False)
                    processing_time = time.time() - start_time
                    
                    # Validate complete result structure
                    required_fields = ['verification_score', 'confidence', 'primary_domain', 
                                     'expert_results', 'ensemble_verification', 'routing_result']
                    all_fields_present = all(hasattr(verification_result, field) for field in required_fields)
                    
                    if all_fields_present and 0 <= verification_result.verification_score <= 1:
                        results['tests_passed'] += 1
                        results['performance'].append(processing_time)
                        results['integration_scores'].append(verification_result.verification_score)
                        
                        print(f"✅ {domain} integration test: Score={verification_result.verification_score:.3f}, Time={processing_time:.3f}s")
                    else:
                        results['tests_failed'] += 1
                        print(f"❌ {domain} integration test: Invalid result structure")
                
                except Exception as e:
                    results['tests_failed'] += 1
                    print(f"❌ {domain} integration test: Error - {str(e)}")
        
        # Calculate performance metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_integration_score'] = sum(results['integration_scores']) / len(results['integration_scores'])
        
        print(f"System Integration Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks against targets"""
        print("\n=== Testing Performance Benchmarks ===")
        
        results = {
            'targets_met': 0,
            'targets_missed': 0,
            'performance_metrics': {}
        }
        
        # Test individual component performance
        components = {
            'dataset_profiler': EnhancedDatasetProfiler(),
            'pii_detector': ComprehensivePIIDetector(),
            'trust_scorer': AdvancedTrustScorer(),
            'expert_ensemble': AdvancedExpertEnsemble(),
            'intelligent_router': IntelligentDomainRouter()
        }
        
        test_text = "This is a test text for performance benchmarking."
        
        for component_name, component in components.items():
            start_time = time.time()
            
            try:
                if component_name == 'dataset_profiler':
                    await component.profile_text(test_text)
                elif component_name == 'pii_detector':
                    await component.detect_pii(test_text)
                elif component_name == 'trust_scorer':
                    await component.score_trust(test_text)
                elif component_name == 'expert_ensemble':
                    await component.get_ensemble_verification(test_text)
                elif component_name == 'intelligent_router':
                    await component.route_to_experts(test_text)
                
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                results['performance_metrics'][component_name] = processing_time
                
                # Check against target (15ms for individual components)
                if processing_time <= 15:
                    results['targets_met'] += 1
                    print(f"✅ {component_name}: {processing_time:.2f}ms (target: ≤15ms)")
                else:
                    results['targets_missed'] += 1
                    print(f"❌ {component_name}: {processing_time:.2f}ms (target: ≤15ms)")
            
            except Exception as e:
                results['targets_missed'] += 1
                print(f"❌ {component_name}: Error - {str(e)}")
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests for Phase 1"""
        print("🏆 Ultimate MoE Solution - Phase 1 Comprehensive Testing")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all component tests
        self.test_results['dataset_profiler'] = await self.test_enhanced_dataset_profiler()
        self.test_results['pii_detector'] = await self.test_comprehensive_pii_detection()
        self.test_results['trust_scorer'] = await self.test_advanced_trust_scoring()
        self.test_results['expert_ensemble'] = await self.test_expert_ensemble()
        self.test_results['intelligent_routing'] = await self.test_intelligent_routing()
        self.test_results['ultimate_system'] = await self.test_ultimate_system_integration()
        self.test_results['performance'] = await self.test_performance_benchmarks()
        
        # Calculate overall results
        total_tests = 0
        total_passed = 0
        
        for component, results in self.test_results.items():
            if 'tests_passed' in results and 'tests_failed' in results:
                total_tests += results['tests_passed'] + results['tests_failed']
                total_passed += results['tests_passed']
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results['overall'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_tests - total_passed,
            'success_rate': overall_success_rate,
            'total_testing_time': time.time() - start_time
        }
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("🏆 PHASE 1 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        for component, results in self.test_results.items():
            if component != 'overall':
                if 'tests_passed' in results and 'tests_failed' in results:
                    success_rate = (results['tests_passed'] / (results['tests_passed'] + results['tests_failed']) * 100) if (results['tests_passed'] + results['tests_failed']) > 0 else 0
                    print(f"{component.replace('_', ' ').title()}: {results['tests_passed']} passed, {results['tests_failed']} failed ({success_rate:.1f}%)")
                    
                    if 'avg_processing_time' in results:
                        print(f"  Average Processing Time: {results['avg_processing_time']:.3f}s")
        
        print(f"\nOverall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Testing Time: {self.test_results['overall']['total_testing_time']:.2f}s")
        
        # Performance summary
        print("\nPerformance Summary:")
        for component, metrics in self.test_results['performance']['performance_metrics'].items():
            print(f"  {component}: {metrics:.2f}ms")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase1_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {filename}")
        
        return self.test_results


async def main():
    """Main test execution"""
    tester = UltimateMoEPhase1Tester()
    results = await tester.run_comprehensive_tests()
    
    # Return results for potential further analysis
    return results


if __name__ == "__main__":
    asyncio.run(main()) 