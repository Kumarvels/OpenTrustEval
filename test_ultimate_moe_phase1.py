"""
Test Ultimate MoE System - Phase 1 Implementation
Comprehensive testing of the enhanced MoE core and intelligent routing
"""

import asyncio
import time
import json
from typing import Dict, List, Any
import sys
import os

# Add the high_performance_system to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'high_performance_system'))

from core.ultimate_moe_system import UltimateMoESystem
from core.advanced_expert_ensemble import AdvancedExpertEnsemble
from core.intelligent_domain_router import IntelligentDomainRouter

class UltimateMoETester:
    """Comprehensive tester for Ultimate MoE System"""
    
    def __init__(self):
        self.system = UltimateMoESystem()
        self.expert_ensemble = AdvancedExpertEnsemble()
        self.router = IntelligentDomainRouter()
        
        # Test data for different domains
        self.test_data = {
            "ecommerce": [
                "The product price is $299.99 with free shipping included.",
                "Add this item to your shopping cart and proceed to checkout.",
                "Customer reviews show 4.5 stars for this product.",
                "Inventory is running low, only 5 items remaining in stock."
            ],
            "banking": [
                "Your account balance is $2,450.67 as of today.",
                "The loan application has been approved with 3.5% interest rate.",
                "Please verify your transaction history for the past month.",
                "Your credit card payment is due on the 15th of this month."
            ],
            "insurance": [
                "Your insurance policy covers up to $500,000 in liability.",
                "The claim has been processed and approved for payment.",
                "Your premium will increase by 5% due to recent claims.",
                "The deductible amount is $1,000 for this coverage."
            ],
            "healthcare": [
                "The patient was diagnosed with diabetes type 2.",
                "Treatment plan includes medication and lifestyle changes.",
                "Symptoms include frequent urination and increased thirst.",
                "The doctor recommended regular blood sugar monitoring."
            ],
            "legal": [
                "The contract terms are legally binding for both parties.",
                "The court judgment was in favor of the plaintiff.",
                "Your attorney will review the legal documents.",
                "This clause protects your rights under the law."
            ],
            "finance": [
                "The investment portfolio shows 12% annual returns.",
                "Stock market analysis indicates bullish trends.",
                "Your dividend payment will be processed next week.",
                "Capital gains tax applies to this transaction."
            ],
            "technology": [
                "The software algorithm processes data efficiently.",
                "Hardware requirements include 16GB RAM minimum.",
                "API integration is complete and functional.",
                "Cloud security measures protect user data."
            ],
            "education": [
                "The student achieved excellent grades in mathematics.",
                "Teacher evaluations show strong performance.",
                "Course curriculum covers advanced topics.",
                "Learning assessment results are available online."
            ],
            "government": [
                "Government policy affects all citizens equally.",
                "Regulation compliance is mandatory for businesses.",
                "Agency officials will review your application.",
                "Legislation passed with bipartisan support."
            ],
            "media": [
                "Content publishing follows editorial guidelines.",
                "Journalism standards ensure accurate reporting.",
                "Broadcast coverage reaches millions of viewers.",
                "Media coverage of the event was comprehensive."
            ]
        }
    
    async def test_expert_ensemble(self) -> Dict[str, Any]:
        """Test the advanced expert ensemble"""
        print("🧪 Testing Advanced Expert Ensemble...")
        
        results = {
            "total_tests": 0,
            "passed_tests": 0,
            "expert_performance": {},
            "domain_accuracy": {},
            "ensemble_scores": []
        }
        
        for domain, texts in self.test_data.items():
            domain_scores = []
            
            for text in texts:
                results["total_tests"] += 1
                
                try:
                    # Test individual expert verification
                    expert_results = await self.expert_ensemble.verify_with_all_experts(text)
                    ensemble_result = await self.expert_ensemble.get_ensemble_verification(text)
                    
                    # Check if the primary domain expert has high confidence
                    primary_expert = f"{domain.capitalize()}Expert"
                    if primary_expert in expert_results:
                        expert_score = expert_results[primary_expert].verification_score
                        domain_scores.append(expert_score)
                        
                        if expert_score > 0.7:  # Threshold for success
                            results["passed_tests"] += 1
                    
                    results["ensemble_scores"].append(ensemble_result["weighted_verification_score"])
                    
                except Exception as e:
                    print(f"❌ Error testing {domain}: {e}")
            
            # Calculate domain accuracy
            if domain_scores:
                results["domain_accuracy"][domain] = {
                    "average_score": sum(domain_scores) / len(domain_scores),
                    "max_score": max(domain_scores),
                    "min_score": min(domain_scores)
                }
        
        # Calculate overall performance
        if results["ensemble_scores"]:
            results["overall_accuracy"] = sum(results["ensemble_scores"]) / len(results["ensemble_scores"])
        
        print(f"✅ Expert Ensemble Tests: {results['passed_tests']}/{results['total_tests']} passed")
        print(f"📊 Overall Accuracy: {results.get('overall_accuracy', 0):.3f}")
        
        return results
    
    async def test_intelligent_routing(self) -> Dict[str, Any]:
        """Test the intelligent domain router"""
        print("🧪 Testing Intelligent Domain Router...")
        
        results = {
            "total_tests": 0,
            "passed_tests": 0,
            "routing_accuracy": {},
            "strategy_performance": {}
        }
        
        for domain, texts in self.test_data.items():
            domain_correct = 0
            
            for text in texts:
                results["total_tests"] += 1
                
                try:
                    # Test routing
                    routing_result = await self.router.route_to_experts(text)
                    
                    # Check if primary domain matches expected
                    if routing_result.primary_domain == domain:
                        domain_correct += 1
                        results["passed_tests"] += 1
                    
                    # Store strategy performance
                    for strategy, weights in routing_result.metadata.items():
                        if strategy not in results["strategy_performance"]:
                            results["strategy_performance"][strategy] = []
                        results["strategy_performance"][strategy].append(weights.get(domain, 0))
                    
                except Exception as e:
                    print(f"❌ Error testing routing for {domain}: {e}")
            
            # Calculate domain routing accuracy
            if texts:
                results["routing_accuracy"][domain] = domain_correct / len(texts)
        
        print(f"✅ Routing Tests: {results['passed_tests']}/{results['total_tests']} passed")
        
        # Calculate average strategy performance
        for strategy, scores in results["strategy_performance"].items():
            results["strategy_performance"][strategy] = sum(scores) / len(scores)
        
        return results
    
    async def test_ultimate_system(self) -> Dict[str, Any]:
        """Test the complete Ultimate MoE System"""
        print("🧪 Testing Ultimate MoE System...")
        
        results = {
            "total_tests": 0,
            "passed_tests": 0,
            "performance_metrics": {},
            "verification_scores": [],
            "latency_measurements": []
        }
        
        # Test with a subset of data for performance
        test_subset = []
        for domain, texts in self.test_data.items():
            test_subset.extend(texts[:2])  # Test first 2 texts from each domain
        
        for text in test_subset:
            results["total_tests"] += 1
            
            try:
                start_time = time.time()
                
                # Test complete verification
                verification_result = await self.system.verify_text(
                    text, enable_advanced_features=False  # Phase 1 only
                )
                
                latency = time.time() - start_time
                results["latency_measurements"].append(latency * 1000)  # Convert to ms
                
                # Check verification quality
                if verification_result.verification_score > 0.7:
                    results["passed_tests"] += 1
                
                results["verification_scores"].append(verification_result.verification_score)
                
            except Exception as e:
                print(f"❌ Error testing ultimate system: {e}")
        
        # Calculate performance metrics
        if results["latency_measurements"]:
            results["performance_metrics"] = {
                "average_latency_ms": sum(results["latency_measurements"]) / len(results["latency_measurements"]),
                "min_latency_ms": min(results["latency_measurements"]),
                "max_latency_ms": max(results["latency_measurements"]),
                "throughput_req_s": len(results["latency_measurements"]) / (sum(results["latency_measurements"]) / 1000)
            }
        
        if results["verification_scores"]:
            results["performance_metrics"]["average_verification_score"] = sum(results["verification_scores"]) / len(results["verification_scores"])
        
        print(f"✅ Ultimate System Tests: {results['passed_tests']}/{results['total_tests']} passed")
        if results["performance_metrics"]:
            print(f"⚡ Average Latency: {results['performance_metrics']['average_latency_ms']:.2f}ms")
            print(f"📊 Average Verification Score: {results['performance_metrics'].get('average_verification_score', 0):.3f}")
        
        return results
    
    async def test_cross_domain_accuracy(self) -> Dict[str, Any]:
        """Test cross-domain verification accuracy"""
        print("🧪 Testing Cross-Domain Accuracy...")
        
        # Create cross-domain test cases
        cross_domain_tests = [
            "The financial software helps manage banking transactions and investment portfolios.",
            "Healthcare insurance policies cover medical treatments and patient care.",
            "Legal technology solutions improve court efficiency and contract management.",
            "Educational media content enhances student learning and teacher effectiveness.",
            "Government e-commerce platforms provide citizen services and regulatory compliance."
        ]
        
        results = {
            "total_tests": len(cross_domain_tests),
            "passed_tests": 0,
            "cross_domain_scores": []
        }
        
        for text in cross_domain_tests:
            try:
                verification_result = await self.system.verify_text(text, enable_advanced_features=False)
                
                if verification_result.verification_score > 0.6:  # Lower threshold for cross-domain
                    results["passed_tests"] += 1
                
                results["cross_domain_scores"].append(verification_result.verification_score)
                
            except Exception as e:
                print(f"❌ Error testing cross-domain: {e}")
        
        if results["cross_domain_scores"]:
            results["average_cross_domain_score"] = sum(results["cross_domain_scores"]) / len(results["cross_domain_scores"])
        
        print(f"✅ Cross-Domain Tests: {results['passed_tests']}/{results['total_tests']} passed")
        print(f"📊 Average Cross-Domain Score: {results.get('average_cross_domain_score', 0):.3f}")
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🚀 Starting Ultimate MoE System - Phase 1 Comprehensive Test")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        test_results = {
            "expert_ensemble": await self.test_expert_ensemble(),
            "intelligent_routing": await self.test_intelligent_routing(),
            "ultimate_system": await self.test_ultimate_system(),
            "cross_domain": await self.test_cross_domain_accuracy()
        }
        
        # Calculate overall metrics
        total_tests = sum(result.get("total_tests", 0) for result in test_results.values())
        total_passed = sum(result.get("passed_tests", 0) for result in test_results.values())
        
        overall_results = {
            "test_suites": test_results,
            "overall_metrics": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0,
                "test_duration_seconds": time.time() - start_time
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {total_passed}")
        print(f"Success Rate: {overall_results['overall_metrics']['success_rate']:.2%}")
        print(f"Test Duration: {overall_results['overall_metrics']['test_duration_seconds']:.2f} seconds")
        
        # Performance comparison with targets
        if "ultimate_system" in test_results:
            perf_metrics = test_results["ultimate_system"].get("performance_metrics", {})
            avg_latency = perf_metrics.get("average_latency_ms", 0)
            target_latency = 15  # Target from ULTIMATE_MOE_SOLUTION
            
            print(f"\n🎯 PERFORMANCE TARGETS")
            print(f"Target Latency: {target_latency}ms")
            print(f"Actual Latency: {avg_latency:.2f}ms")
            print(f"Latency Performance: {'✅' if avg_latency <= target_latency else '❌'}")
        
        return overall_results
    
    def save_test_results(self, results: Dict[str, Any], filename: str = "ultimate_moe_phase1_test_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Test results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving test results: {e}")

async def main():
    """Main test execution"""
    tester = UltimateMoETester()
    
    try:
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Save results
        tester.save_test_results(results)
        
        # Print detailed results
        print("\n📊 DETAILED RESULTS")
        print("=" * 60)
        
        for suite_name, suite_results in results["test_suites"].items():
            print(f"\n{suite_name.upper()}:")
            if "domain_accuracy" in suite_results:
                print("  Domain Accuracy:")
                for domain, accuracy in suite_results["domain_accuracy"].items():
                    print(f"    {domain}: {accuracy['average_score']:.3f}")
            
            if "routing_accuracy" in suite_results:
                print("  Routing Accuracy:")
                for domain, accuracy in suite_results["routing_accuracy"].items():
                    print(f"    {domain}: {accuracy:.3f}")
            
            if "performance_metrics" in suite_results:
                print("  Performance Metrics:")
                for metric, value in suite_results["performance_metrics"].items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    {metric}: {value}")
        
        print(f"\n🎉 Ultimate MoE System - Phase 1 Testing Complete!")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 