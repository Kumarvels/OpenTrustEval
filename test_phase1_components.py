"""
Simplified Test Suite for Phase 1 Components

Tests only the components that are actually implemented and working:
1. Enhanced Dataset Profiler
2. Comprehensive PII Detection
3. Advanced Trust Scoring
4. Advanced Expert Ensemble
5. Intelligent Domain Router
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# Import only implemented components
from high_performance_system.core.enhanced_dataset_profiler import EnhancedDatasetProfiler
from high_performance_system.core.comprehensive_pii_detector import ComprehensivePIIDetector
from high_performance_system.core.advanced_trust_scorer import AdvancedTrustScorer
from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
from high_performance_system.core.intelligent_domain_router import IntelligentDomainRouter


class Phase1ComponentTester:
    """Tester for Phase 1 components"""
    
    def __init__(self):
        self.test_results = {
            'dataset_profiler': {},
            'pii_detector': {},
            'trust_scorer': {},
            'expert_ensemble': {},
            'intelligent_router': {},
            'overall': {}
        }
        
        # Test data
        self.test_texts = [
            "The new iPhone 15 Pro features a titanium design and A17 Pro chip. Available for pre-order starting at $999.",
            "According to peer-reviewed research published in Nature in 2023, climate change has accelerated significantly.",
            "Contact John Smith at john.smith@example.com or call (555) 123-4567 for more information.",
            "AMAZING! You won't BELIEVE this SECRET cure! Click here for the shocking truth!",
            "The Federal Reserve raised interest rates by 0.25% to combat inflation, according to official reports."
        ]
    
    async def test_enhanced_dataset_profiler(self) -> Dict[str, Any]:
        """Test the Enhanced Dataset Profiler"""
        print("\n=== Testing Enhanced Dataset Profiler ===")
        
        profiler = EnhancedDatasetProfiler()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'quality_scores': []
        }
        
        for i, text in enumerate(self.test_texts):
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
                    results['quality_scores'].append(profile.quality_score)
                    
                    print(f"✅ Test {i+1}: Quality={profile.quality_score:.3f}, Time={processing_time:.3f}s")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Test {i+1}: Invalid profile structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Test {i+1}: Error - {str(e)}")
        
        # Calculate metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_quality_score'] = sum(results['quality_scores']) / len(results['quality_scores'])
        
        print(f"Dataset Profiler: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_comprehensive_pii_detection(self) -> Dict[str, Any]:
        """Test the Comprehensive PII Detection"""
        print("\n=== Testing Comprehensive PII Detection ===")
        
        detector = ComprehensivePIIDetector()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'pii_scores': []
        }
        
        for i, text in enumerate(self.test_texts):
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
                    results['pii_scores'].append(pii_result.pii_score)
                    
                    print(f"✅ Test {i+1}: PII Score={pii_result.pii_score:.3f}, Risk={pii_result.privacy_risk.value}, Time={processing_time:.3f}s")
                    print(f"   Detected {len(pii_result.detected_pii)} PII items")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Test {i+1}: Invalid result structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Test {i+1}: Error - {str(e)}")
        
        # Calculate metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_pii_score'] = sum(results['pii_scores']) / len(results['pii_scores'])
        
        print(f"PII Detection: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_advanced_trust_scoring(self) -> Dict[str, Any]:
        """Test the Advanced Trust Scoring"""
        print("\n=== Testing Advanced Trust Scoring ===")
        
        scorer = AdvancedTrustScorer()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'trust_scores': []
        }
        
        for i, text in enumerate(self.test_texts):
            start_time = time.time()
            
            try:
                trust_result = await scorer.score_trust(text)
                processing_time = time.time() - start_time
                
                # Validate result structure
                required_fields = ['trust_score', 'trust_level', 'confidence_interval', 'risk_level']
                all_fields_present = all(hasattr(trust_result, field) for field in required_fields)
                
                if all_fields_present and 0 <= trust_result.trust_score <= 1:
                    results['tests_passed'] += 1
                    results['performance'].append(processing_time)
                    results['trust_scores'].append(trust_result.trust_score)
                    
                    print(f"✅ Test {i+1}: Trust Score={trust_result.trust_score:.3f}, Level={trust_result.trust_level.value}, Time={processing_time:.3f}s")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Test {i+1}: Invalid result structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Test {i+1}: Error - {str(e)}")
        
        # Calculate metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_trust_score'] = sum(results['trust_scores']) / len(results['trust_scores'])
        
        print(f"Trust Scoring: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_advanced_expert_ensemble(self) -> Dict[str, Any]:
        """Test the Advanced Expert Ensemble"""
        print("\n=== Testing Advanced Expert Ensemble ===")
        
        ensemble = AdvancedExpertEnsemble()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'verification_scores': []
        }
        
        for i, text in enumerate(self.test_texts):
            start_time = time.time()
            
            try:
                ensemble_result = await ensemble.get_ensemble_verification(text)
                processing_time = time.time() - start_time
                
                # Validate result structure
                if isinstance(ensemble_result, dict) and 'verification_score' in ensemble_result:
                    results['tests_passed'] += 1
                    results['performance'].append(processing_time)
                    results['verification_scores'].append(ensemble_result['verification_score'])
                    
                    print(f"✅ Test {i+1}: Score={ensemble_result['verification_score']:.3f}, Time={processing_time:.3f}s")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Test {i+1}: Invalid result structure")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Test {i+1}: Error - {str(e)}")
        
        # Calculate metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['avg_verification_score'] = sum(results['verification_scores']) / len(results['verification_scores'])
        
        print(f"Expert Ensemble: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def test_intelligent_domain_router(self) -> Dict[str, Any]:
        """Test the Intelligent Domain Router"""
        print("\n=== Testing Intelligent Domain Router ===")
        
        router = IntelligentDomainRouter()
        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance': [],
            'routing_success': []
        }
        
        for i, text in enumerate(self.test_texts):
            start_time = time.time()
            
            try:
                routing_result = await router.route_to_experts(text)
                processing_time = time.time() - start_time
                
                # Validate routing result
                if isinstance(routing_result, dict) and 'primary_domain' in routing_result:
                    results['tests_passed'] += 1
                    results['performance'].append(processing_time)
                    results['routing_success'].append(True)
                    
                    print(f"✅ Test {i+1}: Primary Domain={routing_result['primary_domain']}, Time={processing_time:.3f}s")
                else:
                    results['tests_failed'] += 1
                    print(f"❌ Test {i+1}: Invalid routing result")
            
            except Exception as e:
                results['tests_failed'] += 1
                print(f"❌ Test {i+1}: Error - {str(e)}")
        
        # Calculate metrics
        if results['performance']:
            results['avg_processing_time'] = sum(results['performance']) / len(results['performance'])
            results['routing_accuracy'] = sum(results['routing_success']) / len(results['routing_success'])
        
        print(f"Intelligent Router: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 1 component tests"""
        print("🏆 Phase 1 Component Testing")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all component tests
        self.test_results['dataset_profiler'] = await self.test_enhanced_dataset_profiler()
        self.test_results['pii_detector'] = await self.test_comprehensive_pii_detection()
        self.test_results['trust_scorer'] = await self.test_advanced_trust_scoring()
        self.test_results['expert_ensemble'] = await self.test_advanced_expert_ensemble()
        self.test_results['intelligent_router'] = await self.test_intelligent_domain_router()
        
        # Calculate overall results
        total_tests = 0
        total_passed = 0
        
        for component, results in self.test_results.items():
            if component != 'overall' and 'tests_passed' in results and 'tests_failed' in results:
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
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏆 PHASE 1 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for component, results in self.test_results.items():
            if component != 'overall' and 'tests_passed' in results and 'tests_failed' in results:
                success_rate = (results['tests_passed'] / (results['tests_passed'] + results['tests_failed']) * 100) if (results['tests_passed'] + results['tests_failed']) > 0 else 0
                print(f"{component.replace('_', ' ').title()}: {results['tests_passed']} passed, {results['tests_failed']} failed ({success_rate:.1f}%)")
                
                if 'avg_processing_time' in results:
                    print(f"  Average Processing Time: {results['avg_processing_time']:.3f}s")
        
        print(f"\nOverall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Testing Time: {self.test_results['overall']['total_testing_time']:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase1_simple_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
        
        return self.test_results


async def main():
    """Main test execution"""
    tester = Phase1ComponentTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    asyncio.run(main()) 