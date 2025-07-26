#!/usr/bin/env python3
"""
Real-time Operation Sindoor News Hallucination Detection Test
Tests misinformation and disinformation scenarios for geopolitical news verification
"""

import asyncio
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any
import random

class OperationSindoorRealtimeTester:
    """Real-time tester for Operation Sindoor news hallucination detection"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.test_results = []
        self.start_time = time.time()
        
    def log_test(self, test_name: str, result: Dict[str, Any], expected_sources: List[str]):
        """Log test results with analysis"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Extract key metrics
        verified = result.get('verified', False)
        confidence = result.get('confidence', 0.0)
        hallucination_risk = result.get('hallucination_risk', 1.0)
        domain = result.get('domain', 'unknown')
        sources_used = result.get('sources_used', [])
        latency = result.get('latency', 0.0)
        
        # Analyze source accuracy
        source_accuracy = self._analyze_source_accuracy(sources_used, expected_sources)
        
        # Determine if hallucination was correctly detected
        is_misinformation = "misinformation" in test_name.lower() or "disinformation" in test_name.lower()
        correct_detection = (is_misinformation and hallucination_risk > 0.5) or (not is_misinformation and hallucination_risk < 0.5)
        
        test_result = {
            "timestamp": timestamp,
            "test_name": test_name,
            "verified": verified,
            "confidence": confidence,
            "hallucination_risk": hallucination_risk,
            "domain": domain,
            "latency_ms": latency * 1000,
            "sources_used": sources_used,
            "expected_sources": expected_sources,
            "source_accuracy": source_accuracy,
            "correct_detection": correct_detection,
            "is_misinformation": is_misinformation
        }
        
        self.test_results.append(test_result)
        
        # Print real-time results
        status_icon = "✅" if correct_detection else "❌"
        risk_level = "HIGH" if hallucination_risk > 0.7 else "MEDIUM" if hallucination_risk > 0.3 else "LOW"
        
        print(f"{status_icon} [{timestamp}] {test_name}")
        print(f"   Risk: {risk_level} ({hallucination_risk:.3f}) | Confidence: {confidence:.3f} | Latency: {latency*1000:.1f}ms")
        print(f"   Sources: {len(sources_used)} found | Accuracy: {source_accuracy:.1%}")
        print(f"   Domain: {domain} | Detection: {'CORRECT' if correct_detection else 'INCORRECT'}")
        print()

    def _analyze_source_accuracy(self, actual_sources: List[str], expected_sources: List[str]) -> float:
        """Analyze how well the sources match expected domain-specific sources"""
        if not expected_sources:
            return 1.0
        
        matches = sum(1 for source in actual_sources if any(expected in source.lower() for expected in expected_sources))
        return matches / len(expected_sources) if expected_sources else 0.0

    def test_operation_sindoor_scenarios(self):
        """Run comprehensive Operation Sindoor testing scenarios"""
        
        print("🔍 OPERATION SIND OOR REAL-TIME HALLUCINATION DETECTION TEST")
        print("=" * 70)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing server: {self.base_url}")
        print("=" * 70)
        print()

        # Test scenarios with different levels of misinformation
        scenarios = [
            # LEGITIMATE NEWS (Should have low hallucination risk)
            {
                "name": "Legitimate Operation Sindoor News",
                "query": "What is Operation Sindoor?",
                "response": "Operation Sindoor is a military operation conducted by Indian security forces in Kashmir region to counter terrorism and maintain security.",
                "domain": "government",
                "expected_sources": ["government", "policy", "security"],
                "should_be_verified": True
            },
            
            # MISINFORMATION SCENARIOS (Should have high hallucination risk)
            {
                "name": "Misinformation - False Casualty Numbers",
                "query": "How many casualties in Operation Sindoor?",
                "response": "Operation Sindoor resulted in over 500 civilian casualties and 200 military deaths, making it the deadliest operation in Kashmir history.",
                "domain": "media",
                "expected_sources": ["media", "news", "journalism"],
                "should_be_verified": False
            },
            
            {
                "name": "Misinformation - False Timeline",
                "query": "When did Operation Sindoor start?",
                "response": "Operation Sindoor began in 2020 and has been ongoing for 5 years, making it the longest military operation in Indian history.",
                "domain": "media",
                "expected_sources": ["media", "news", "journalism"],
                "should_be_verified": False
            },
            
            # DISINFORMATION SCENARIOS (Should have very high hallucination risk)
            {
                "name": "Disinformation - False International Involvement",
                "query": "Which countries are involved in Operation Sindoor?",
                "response": "Operation Sindoor is a joint operation between India, Pakistan, China, and Russia, with NATO providing logistical support and weapons.",
                "domain": "government",
                "expected_sources": ["government", "policy", "international"],
                "should_be_verified": False
            },
            
            {
                "name": "Disinformation - False Nuclear Threat",
                "query": "Are there nuclear weapons involved in Operation Sindoor?",
                "response": "Operation Sindoor involves tactical nuclear weapons and has escalated to nuclear threats between India and Pakistan, with both countries on high alert.",
                "domain": "government",
                "expected_sources": ["government", "security", "nuclear"],
                "should_be_verified": False
            },
            
            # MIXED TRUTH SCENARIOS (Should have medium hallucination risk)
            {
                "name": "Mixed Truth - Exaggerated Claims",
                "query": "What is the scale of Operation Sindoor?",
                "response": "Operation Sindoor involves 50,000 troops, 1000 tanks, and 500 aircraft, making it the largest military operation since World War II.",
                "domain": "media",
                "expected_sources": ["media", "news", "military"],
                "should_be_verified": False
            },
            
            {
                "name": "Mixed Truth - Speculative Analysis",
                "query": "What are the long-term effects of Operation Sindoor?",
                "response": "Operation Sindoor will likely lead to complete independence of Kashmir, economic collapse of the region, and potential World War III between nuclear powers.",
                "domain": "media",
                "expected_sources": ["media", "analysis", "speculation"],
                "should_be_verified": False
            },
            
            # FACTUAL NEWS (Should have low hallucination risk)
            {
                "name": "Factual Military Operation Details",
                "query": "What type of operation is Operation Sindoor?",
                "response": "Operation Sindoor is a counter-terrorism operation conducted by Indian security forces in the Kashmir region to neutralize terrorist threats and maintain law and order.",
                "domain": "government",
                "expected_sources": ["government", "security", "military"],
                "should_be_verified": True
            },
            
            # CONSPIRACY THEORY (Should have very high hallucination risk)
            {
                "name": "Conspiracy Theory - Hidden Agenda",
                "query": "What is the real purpose of Operation Sindoor?",
                "response": "Operation Sindoor is actually a cover for India's secret plan to annex Pakistan, with hidden nuclear facilities and underground bases being constructed for the final invasion.",
                "domain": "media",
                "expected_sources": ["media", "conspiracy", "speculation"],
                "should_be_verified": False
            },
            
            # TECHNICAL MISINFORMATION (Should have high hallucination risk)
            {
                "name": "Technical Misinformation - False Equipment",
                "query": "What military equipment is used in Operation Sindoor?",
                "response": "Operation Sindoor uses advanced AI-powered drones, invisible tanks, and laser weapons that can destroy entire cities with a single shot.",
                "domain": "technology",
                "expected_sources": ["technology", "military", "equipment"],
                "should_be_verified": False
            }
        ]

        # Run all scenarios
        for i, scenario in enumerate(scenarios, 1):
            print(f"🧪 Test {i}/{len(scenarios)}: {scenario['name']}")
            
            try:
                # Make API request
                payload = {
                    "query": scenario["query"],
                    "response": scenario["response"],
                    "domain": scenario["domain"]
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/detect",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    result['latency'] = end_time - start_time
                    
                    self.log_test(
                        scenario['name'],
                        result,
                        scenario['expected_sources']
                    )
                else:
                    print(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
            
            # Small delay between tests
            time.sleep(0.5)

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 OPERATION SIND OOR TEST REPORT")
        print("=" * 70)
        
        if not self.test_results:
            print("No test results to report.")
            return
        
        # Calculate statistics
        total_tests = len(self.test_results)
        correct_detections = sum(1 for r in self.test_results if r['correct_detection'])
        misinformation_tests = sum(1 for r in self.test_results if r['is_misinformation'])
        legitimate_tests = total_tests - misinformation_tests
        
        avg_latency = sum(r['latency_ms'] for r in self.test_results) / total_tests
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total_tests
        avg_hallucination_risk = sum(r['hallucination_risk'] for r in self.test_results) / total_tests
        
        # Misinformation detection accuracy
        misinformation_detected = sum(1 for r in self.test_results 
                                    if r['is_misinformation'] and r['hallucination_risk'] > 0.5)
        legitimate_correctly_identified = sum(1 for r in self.test_results 
                                            if not r['is_misinformation'] and r['hallucination_risk'] < 0.5)
        
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Correct Detections: {correct_detections}/{total_tests} ({correct_detections/total_tests:.1%})")
        print(f"   Average Latency: {avg_latency:.1f}ms")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Hallucination Risk: {avg_hallucination_risk:.3f}")
        
        print(f"\n🎯 MISINFORMATION DETECTION:")
        print(f"   Misinformation Tests: {misinformation_tests}")
        print(f"   Legitimate Tests: {legitimate_tests}")
        print(f"   Misinformation Detected: {misinformation_detected}/{misinformation_tests} ({misinformation_detected/misinformation_tests:.1%})")
        print(f"   Legitimate Correctly Identified: {legitimate_correctly_identified}/{legitimate_tests} ({legitimate_correctly_identified/legitimate_tests:.1%})")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Fastest Response: {min(r['latency_ms'] for r in self.test_results):.1f}ms")
        print(f"   Slowest Response: {max(r['latency_ms'] for r in self.test_results):.1f}ms")
        print(f"   Total Test Time: {time.time() - self.start_time:.2f}s")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result['correct_detection'] else "❌"
            risk_level = "HIGH" if result['hallucination_risk'] > 0.7 else "MEDIUM" if result['hallucination_risk'] > 0.3 else "LOW"
            print(f"   {status} {result['test_name']} - Risk: {risk_level} ({result['hallucination_risk']:.3f})")
        
        # Save detailed results to file
        report_filename = f"operation_sindoor_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump({
                "test_summary": {
                    "total_tests": total_tests,
                    "correct_detections": correct_detections,
                    "accuracy": correct_detections/total_tests,
                    "avg_latency_ms": avg_latency,
                    "avg_confidence": avg_confidence,
                    "avg_hallucination_risk": avg_hallucination_risk,
                    "misinformation_detection_rate": misinformation_detected/misinformation_tests if misinformation_tests > 0 else 0,
                    "legitimate_identification_rate": legitimate_correctly_identified/legitimate_tests if legitimate_tests > 0 else 0
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_filename}")
        print("=" * 70)

def main():
    """Main function to run the Operation Sindoor real-time test"""
    tester = OperationSindoorRealtimeTester()
    
    try:
        # Run the comprehensive test
        tester.test_operation_sindoor_scenarios()
        
        # Generate and display report
        tester.generate_report()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        tester.generate_report()
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        tester.generate_report()

if __name__ == "__main__":
    main() 