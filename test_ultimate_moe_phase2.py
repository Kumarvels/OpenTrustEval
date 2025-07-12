#!/usr/bin/env python3
"""
Ultimate MoE Solution - Phase 2 Testing Suite
Tests all Phase 2 components: Enhanced RAG Pipeline, Multi-Agent System, 
Uncertainty-Aware System, and Performance Optimizer
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime

# Import Phase 2 components
from high_performance_system.core.enhanced_rag_pipeline import EnhancedRAGPipeline
from high_performance_system.core.advanced_multi_agent_system import AdvancedMultiAgentSystem
from high_performance_system.core.uncertainty_aware_system import UncertaintyAwareSystem
from high_performance_system.core.performance_optimizer import AdvancedPerformanceOptimizer

class UltimateMoEPhase2Tester:
    """Comprehensive tester for Phase 2 components"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
        # Initialize Phase 2 components
        self.rag_pipeline = EnhancedRAGPipeline()
        self.multi_agent = AdvancedMultiAgentSystem()
        self.uncertainty_system = UncertaintyAwareSystem()
        self.performance_optimizer = AdvancedPerformanceOptimizer()
        
        # Test data
        self.test_queries = [
            "What are the latest developments in quantum computing?",
            "How does blockchain technology work in banking?",
            "What are the health benefits of Mediterranean diet?",
            "Explain the legal implications of AI in healthcare",
            "What are the environmental impacts of renewable energy?"
        ]
        
        self.test_documents = [
            "Quantum computing represents a paradigm shift in computational power. Recent developments include superconducting qubits achieving 99.9% fidelity and quantum supremacy demonstrations by Google and IBM.",
            "Blockchain technology in banking enables secure, transparent, and immutable transaction records. It reduces fraud, lowers costs, and enables real-time settlement across borders.",
            "The Mediterranean diet emphasizes fruits, vegetables, whole grains, and healthy fats. Research shows it reduces heart disease risk by 30% and improves cognitive function.",
            "AI in healthcare raises legal concerns about liability, privacy, and regulatory compliance. The FDA has approved over 500 AI medical devices, but legal frameworks are evolving.",
            "Renewable energy reduces greenhouse gas emissions by 80-90% compared to fossil fuels. Solar and wind power are now cost-competitive and create more jobs than fossil fuel industries."
        ]
        
        self.test_texts = [
            "The COVID-19 pandemic has accelerated digital transformation across industries, with remote work becoming the new normal for many organizations worldwide.",
            "Machine learning algorithms can achieve 95% accuracy in medical diagnosis, outperforming human doctors in certain specialized areas.",
            "Climate change is causing unprecedented global temperature increases, with the last decade being the warmest on record according to NASA data.",
            "The global economy is expected to grow by 3.5% in 2024, driven by technological innovation and sustainable development initiatives.",
            "Artificial intelligence will create more jobs than it eliminates, with estimates suggesting 2.3 million new jobs by 2030."
        ]
    
    async def test_enhanced_rag_pipeline(self) -> Dict[str, Any]:
        """Test the Enhanced RAG Pipeline"""
        print("🧠 Testing Enhanced RAG Pipeline...")
        
        results = {
            "component": "Enhanced RAG Pipeline",
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Test 1: Basic RAG functionality
            start_time = time.time()
            rag_result = await self.rag_pipeline.enhanced_rag_with_moe(
                query=self.test_queries[0],
                documents=self.test_documents
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Basic RAG with MoE",
                "status": "PASS" if rag_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(rag_result.keys()) if rag_result else []
            })
            
            # Test 2: Semantic chunking
            start_time = time.time()
            chunks = await self.rag_pipeline.semantic_chunker.advanced_chunking(self.test_documents)
            end_time = time.time()
            
            results["tests"].append({
                "test": "Semantic Chunking",
                "status": "PASS" if chunks else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "chunk_count": len(chunks) if chunks else 0
            })
            
            # Test 3: Embedding generation
            start_time = time.time()
            embeddings = await self.rag_pipeline.embedding_generator.generate_embeddings(chunks[:3])
            end_time = time.time()
            
            results["tests"].append({
                "test": "Embedding Generation",
                "status": "PASS" if embeddings else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "embedding_count": len(embeddings) if embeddings else 0
            })
            
            # Test 4: Hybrid search
            start_time = time.time()
            search_results = await self.rag_pipeline.hybrid_search.search(
                query=self.test_queries[1],
                embeddings=embeddings
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Hybrid Search",
                "status": "PASS" if search_results else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_count": len(search_results) if search_results else 0
            })
            
            # Test 5: Context re-ranking
            if search_results:
                start_time = time.time()
                reranked_results = await self.rag_pipeline.context_reranker.rerank(
                    query=self.test_queries[1],
                    search_results=search_results
                )
                end_time = time.time()
                
                results["tests"].append({
                    "test": "Context Re-ranking",
                    "status": "PASS" if reranked_results else "FAIL",
                    "latency": round((end_time - start_time) * 1000, 2),
                    "reranked_count": len(reranked_results) if reranked_results else 0
                })
            
            # Performance metrics
            total_latency = sum(test["latency"] for test in results["tests"])
            results["performance"] = {
                "total_latency_ms": total_latency,
                "average_latency_ms": total_latency / len(results["tests"]),
                "tests_passed": len([t for t in results["tests"] if t["status"] == "PASS"]),
                "tests_failed": len([t for t in results["tests"] if t["status"] == "FAIL"])
            }
            
        except Exception as e:
            results["errors"].append(f"RAG Pipeline Error: {str(e)}")
            print(f"❌ RAG Pipeline Error: {e}")
        
        return results
    
    async def test_advanced_multi_agent_system(self) -> Dict[str, Any]:
        """Test the Advanced Multi-Agent System"""
        print("🤖 Testing Advanced Multi-Agent System...")
        
        results = {
            "component": "Advanced Multi-Agent System",
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Test 1: Fact checking agent
            start_time = time.time()
            fact_result = await self.multi_agent.fact_checking_agent.evaluate(
                text=self.test_texts[0],
                context="Digital transformation trends"
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Fact Checking Agent",
                "status": "PASS" if fact_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(fact_result.keys()) if fact_result else []
            })
            
            # Test 2: QA validation agent
            start_time = time.time()
            qa_result = await self.multi_agent.qa_validation_agent.evaluate(
                text=self.test_texts[1],
                context="Medical AI applications"
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "QA Validation Agent",
                "status": "PASS" if qa_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(qa_result.keys()) if qa_result else []
            })
            
            # Test 3: Adversarial agent
            start_time = time.time()
            adv_result = await self.multi_agent.adversarial_agent.evaluate(
                text=self.test_texts[2],
                context="Climate change data"
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Adversarial Agent",
                "status": "PASS" if adv_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(adv_result.keys()) if adv_result else []
            })
            
            # Test 4: Consistency agent
            start_time = time.time()
            cons_result = await self.multi_agent.consistency_agent.evaluate(
                text=self.test_texts[3],
                context="Economic forecasting"
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Consistency Agent",
                "status": "PASS" if cons_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(cons_result.keys()) if cons_result else []
            })
            
            # Test 5: Comprehensive evaluation
            start_time = time.time()
            comprehensive_result = await self.multi_agent.comprehensive_evaluation(
                text=self.test_texts[4],
                context="AI job market impact"
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Comprehensive Evaluation",
                "status": "PASS" if comprehensive_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(comprehensive_result.keys()) if comprehensive_result else []
            })
            
            # Performance metrics
            total_latency = sum(test["latency"] for test in results["tests"])
            results["performance"] = {
                "total_latency_ms": total_latency,
                "average_latency_ms": total_latency / len(results["tests"]),
                "tests_passed": len([t for t in results["tests"] if t["status"] == "PASS"]),
                "tests_failed": len([t for t in results["tests"] if t["status"] == "FAIL"])
            }
            
        except Exception as e:
            results["errors"].append(f"Multi-Agent System Error: {str(e)}")
            print(f"❌ Multi-Agent System Error: {e}")
        
        return results
    
    async def test_uncertainty_aware_system(self) -> Dict[str, Any]:
        """Test the Uncertainty-Aware System"""
        print("🎲 Testing Uncertainty-Aware System...")
        
        results = {
            "component": "Uncertainty-Aware System",
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Test 1: Bayesian ensemble analysis
            start_time = time.time()
            bayesian_result = await self.uncertainty_system.bayesian_ensemble.analyze(
                text=self.test_texts[0]
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Bayesian Ensemble Analysis",
                "status": "PASS" if bayesian_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(bayesian_result.keys()) if bayesian_result else []
            })
            
            # Test 2: Monte Carlo simulation
            start_time = time.time()
            monte_carlo_result = await self.uncertainty_system.monte_carlo_simulator.simulate(
                text=self.test_texts[1]
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Monte Carlo Simulation",
                "status": "PASS" if monte_carlo_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(monte_carlo_result.keys()) if monte_carlo_result else []
            })
            
            # Test 3: Confidence calibration
            start_time = time.time()
            confidence_result = await self.uncertainty_system.confidence_calibrator.calibrate(
                text=self.test_texts[2]
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Confidence Calibration",
                "status": "PASS" if confidence_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(confidence_result.keys()) if confidence_result else []
            })
            
            # Test 4: Risk assessment
            start_time = time.time()
            risk_result = await self.uncertainty_system.risk_assessor.assess_risk(
                text=self.test_texts[3]
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Risk Assessment",
                "status": "PASS" if risk_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(risk_result.keys()) if risk_result else []
            })
            
            # Test 5: Comprehensive uncertainty analysis
            start_time = time.time()
            comprehensive_result = await self.uncertainty_system.comprehensive_uncertainty_analysis(
                text=self.test_texts[4]
            )
            end_time = time.time()
            
            results["tests"].append({
                "test": "Comprehensive Uncertainty Analysis",
                "status": "PASS" if comprehensive_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(comprehensive_result.keys()) if comprehensive_result else []
            })
            
            # Performance metrics
            total_latency = sum(test["latency"] for test in results["tests"])
            results["performance"] = {
                "total_latency_ms": total_latency,
                "average_latency_ms": total_latency / len(results["tests"]),
                "tests_passed": len([t for t in results["tests"] if t["status"] == "PASS"]),
                "tests_failed": len([t for t in results["tests"] if t["status"] == "FAIL"])
            }
            
        except Exception as e:
            results["errors"].append(f"Uncertainty System Error: {str(e)}")
            print(f"❌ Uncertainty System Error: {e}")
        
        return results
    
    async def test_performance_optimizer(self) -> Dict[str, Any]:
        """Test the Performance Optimizer"""
        print("⚡ Testing Performance Optimizer...")
        
        results = {
            "component": "Performance Optimizer",
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Test 1: Latency optimization
            start_time = time.time()
            latency_result = await self.performance_optimizer.latency_optimizer.optimize_latency()
            end_time = time.time()
            
            results["tests"].append({
                "test": "Latency Optimization",
                "status": "PASS" if latency_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(latency_result.keys()) if latency_result else []
            })
            
            # Test 2: Throughput optimization
            start_time = time.time()
            throughput_result = await self.performance_optimizer.throughput_optimizer.optimize_throughput()
            end_time = time.time()
            
            results["tests"].append({
                "test": "Throughput Optimization",
                "status": "PASS" if throughput_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(throughput_result.keys()) if throughput_result else []
            })
            
            # Test 3: Memory optimization
            start_time = time.time()
            memory_result = await self.performance_optimizer.memory_optimizer.optimize_memory()
            end_time = time.time()
            
            results["tests"].append({
                "test": "Memory Optimization",
                "status": "PASS" if memory_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(memory_result.keys()) if memory_result else []
            })
            
            # Test 4: Cache optimization
            start_time = time.time()
            cache_result = await self.performance_optimizer.cache_optimizer.optimize_cache()
            end_time = time.time()
            
            results["tests"].append({
                "test": "Cache Optimization",
                "status": "PASS" if cache_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(cache_result.keys()) if cache_result else []
            })
            
            # Test 5: Comprehensive optimization
            start_time = time.time()
            comprehensive_result = await self.performance_optimizer.optimize_system_performance()
            end_time = time.time()
            
            results["tests"].append({
                "test": "Comprehensive Performance Optimization",
                "status": "PASS" if comprehensive_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(comprehensive_result.keys()) if comprehensive_result else []
            })
            
            # Performance metrics
            total_latency = sum(test["latency"] for test in results["tests"])
            results["performance"] = {
                "total_latency_ms": total_latency,
                "average_latency_ms": total_latency / len(results["tests"]),
                "tests_passed": len([t for t in results["tests"] if t["status"] == "PASS"]),
                "tests_failed": len([t for t in results["tests"] if t["status"] == "FAIL"])
            }
            
        except Exception as e:
            results["errors"].append(f"Performance Optimizer Error: {str(e)}")
            print(f"❌ Performance Optimizer Error: {e}")
        
        return results
    
    async def test_phase2_integration(self) -> Dict[str, Any]:
        """Test Phase 2 components integration"""
        print("🔗 Testing Phase 2 Integration...")
        
        results = {
            "component": "Phase 2 Integration",
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Test 1: RAG + Multi-Agent integration
            start_time = time.time()
            
            # Get RAG results
            rag_result = await self.rag_pipeline.enhanced_rag_with_moe(
                query=self.test_queries[0],
                documents=self.test_documents
            )
            
            # Use RAG results in multi-agent evaluation
            if rag_result and "final_answer" in rag_result:
                agent_result = await self.multi_agent.comprehensive_evaluation(
                    text=rag_result["final_answer"],
                    context="RAG-generated content"
                )
            else:
                agent_result = None
            
            end_time = time.time()
            
            results["tests"].append({
                "test": "RAG + Multi-Agent Integration",
                "status": "PASS" if agent_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(agent_result.keys()) if agent_result else []
            })
            
            # Test 2: Multi-Agent + Uncertainty integration
            start_time = time.time()
            
            # Get multi-agent results
            agent_result = await self.multi_agent.comprehensive_evaluation(
                text=self.test_texts[0],
                context="Integration test"
            )
            
            # Use agent results in uncertainty analysis
            if agent_result and "final_decision" in agent_result:
                uncertainty_result = await self.uncertainty_system.comprehensive_uncertainty_analysis(
                    text=str(agent_result["final_decision"])
                )
            else:
                uncertainty_result = None
            
            end_time = time.time()
            
            results["tests"].append({
                "test": "Multi-Agent + Uncertainty Integration",
                "status": "PASS" if uncertainty_result else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(uncertainty_result.keys()) if uncertainty_result else []
            })
            
            # Test 3: Performance optimization integration
            start_time = time.time()
            
            # Run performance optimization
            perf_result = await self.performance_optimizer.optimize_system_performance()
            
            # Test optimized performance
            if perf_result:
                optimized_rag = await self.rag_pipeline.enhanced_rag_with_moe(
                    query=self.test_queries[1],
                    documents=self.test_documents[:2]
                )
            else:
                optimized_rag = None
            
            end_time = time.time()
            
            results["tests"].append({
                "test": "Performance Optimization Integration",
                "status": "PASS" if optimized_rag else "FAIL",
                "latency": round((end_time - start_time) * 1000, 2),
                "result_keys": list(optimized_rag.keys()) if optimized_rag else []
            })
            
            # Performance metrics
            total_latency = sum(test["latency"] for test in results["tests"])
            results["performance"] = {
                "total_latency_ms": total_latency,
                "average_latency_ms": total_latency / len(results["tests"]),
                "tests_passed": len([t for t in results["tests"] if t["status"] == "PASS"]),
                "tests_failed": len([t for t in results["tests"] if t["status"] == "FAIL"])
            }
            
        except Exception as e:
            results["errors"].append(f"Integration Error: {str(e)}")
            print(f"❌ Integration Error: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 2 tests"""
        print("🚀 Starting Ultimate MoE Phase 2 Testing Suite")
        print("=" * 60)
        
        # Run individual component tests
        rag_results = await self.test_enhanced_rag_pipeline()
        multi_agent_results = await self.test_advanced_multi_agent_system()
        uncertainty_results = await self.test_uncertainty_aware_system()
        performance_results = await self.test_performance_optimizer()
        integration_results = await self.test_phase2_integration()
        
        # Compile overall results
        all_results = {
            "test_suite": "Ultimate MoE Phase 2",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "enhanced_rag_pipeline": rag_results,
                "advanced_multi_agent_system": multi_agent_results,
                "uncertainty_aware_system": uncertainty_results,
                "performance_optimizer": performance_results,
                "phase2_integration": integration_results
            }
        }
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_latency = 0
        
        for component_name, component_results in all_results["components"].items():
            if "tests" in component_results:
                component_tests = len(component_results["tests"])
                component_passed = len([t for t in component_results["tests"] if t["status"] == "PASS"])
                component_failed = len([t for t in component_results["tests"] if t["status"] == "FAIL"])
                component_latency = sum(t["latency"] for t in component_results["tests"])
                
                total_tests += component_tests
                total_passed += component_passed
                total_failed += component_failed
                total_latency += component_latency
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": round((total_passed / total_tests) * 100, 2) if total_tests > 0 else 0,
            "total_latency_ms": total_latency,
            "average_latency_ms": round(total_latency / total_tests, 2) if total_tests > 0 else 0
        }
        
        return all_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way"""
        print("\n" + "=" * 60)
        print("🏆 ULTIMATE MOE PHASE 2 TEST RESULTS")
        print("=" * 60)
        
        # Print summary
        summary = results["summary"]
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['total_passed']} ✅")
        print(f"   Failed: {summary['total_failed']} ❌")
        print(f"   Success Rate: {summary['success_rate']}%")
        print(f"   Total Latency: {summary['total_latency_ms']}ms")
        print(f"   Average Latency: {summary['average_latency_ms']}ms")
        
        # Print component results
        print(f"\n🔧 COMPONENT RESULTS:")
        for component_name, component_results in results["components"].items():
            if "tests" in component_results:
                component_tests = len(component_results["tests"])
                component_passed = len([t for t in component_results["tests"] if t["status"] == "PASS"])
                component_failed = len([t for t in component_results["tests"] if t["status"] == "FAIL"])
                component_success_rate = round((component_passed / component_tests) * 100, 2) if component_tests > 0 else 0
                
                status_icon = "✅" if component_success_rate >= 80 else "⚠️" if component_success_rate >= 60 else "❌"
                print(f"   {status_icon} {component_name.replace('_', ' ').title()}: {component_success_rate}% ({component_passed}/{component_tests})")
                
                # Print errors if any
                if "errors" in component_results and component_results["errors"]:
                    for error in component_results["errors"]:
                        print(f"      ❌ Error: {error}")
        
        # Print detailed test results
        print(f"\n📋 DETAILED TEST RESULTS:")
        for component_name, component_results in results["components"].items():
            if "tests" in component_results:
                print(f"\n   {component_name.replace('_', ' ').title()}:")
                for test in component_results["tests"]:
                    status_icon = "✅" if test["status"] == "PASS" else "❌"
                    print(f"     {status_icon} {test['test']}: {test['latency']}ms")
        
        print("\n" + "=" * 60)
        print("🏆 PHASE 2 TESTING COMPLETE")
        print("=" * 60)

async def main():
    """Main test runner"""
    tester = UltimateMoEPhase2Tester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print results
        tester.print_results(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filename}")
        
        # Return success/failure
        success_rate = results["summary"]["success_rate"]
        if success_rate >= 80:
            print(f"\n🎉 Phase 2 Testing PASSED with {success_rate}% success rate!")
            return True
        else:
            print(f"\n⚠️ Phase 2 Testing needs improvement: {success_rate}% success rate")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 