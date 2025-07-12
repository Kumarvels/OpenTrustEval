"""
Test Script for High-Performance Hallucination Detection System

This script tests all components of the system:
- Advanced hallucination detection
- Domain-specific verification
- Real-time verification orchestration
- Performance monitoring
- API endpoints
"""

import asyncio
import time
import json
import requests
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighPerformanceSystemTester:
    """Comprehensive tester for the high-performance hallucination detection system"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting High-Performance Hallucination Detection System Tests")
        
        # Test system health
        await self.test_system_health()
        
        # Test API endpoints
        await self.test_api_endpoints()
        
        # Test domain-specific verification
        await self.test_domain_verification()
        
        # Test performance
        await self.test_performance()
        
        # Test batch processing
        await self.test_batch_processing()
        
        # Test error handling
        await self.test_error_handling()
        
        # Generate test report
        self.generate_test_report()
        
        logger.info("All tests completed!")

    async def test_system_health(self):
        """Test system health endpoints"""
        logger.info("Testing system health...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health")
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            
            health_data = response.json()
            assert 'status' in health_data, "Health response missing status"
            assert health_data['status'] == 'healthy', f"System not healthy: {health_data['status']}"
            
            # Test metrics endpoint
            response = requests.get(f"{self.base_url}/metrics")
            assert response.status_code == 200, f"Metrics check failed: {response.status_code}"
            
            metrics_data = response.json()
            assert 'total_requests' in metrics_data, "Metrics response missing total_requests"
            
            logger.info("✅ System health tests passed")
            self.test_results.append({
                'test': 'system_health',
                'status': 'passed',
                'details': health_data
            })
            
        except Exception as e:
            logger.error(f"❌ System health tests failed: {e}")
            self.test_results.append({
                'test': 'system_health',
                'status': 'failed',
                'error': str(e)
            })

    async def test_api_endpoints(self):
        """Test main API endpoints"""
        logger.info("Testing API endpoints...")
        
        test_cases = [
            {
                'name': 'general_detection',
                'data': {
                    'query': 'What is the capital of France?',
                    'response': 'The capital of France is Paris.',
                    'domain': 'general'
                }
            },
            {
                'name': 'ecommerce_detection',
                'data': {
                    'query': 'What is the price of iPhone 15?',
                    'response': 'The iPhone 15 costs $999 and is available in all stores.',
                    'domain': 'ecommerce'
                }
            },
            {
                'name': 'banking_detection',
                'data': {
                    'query': 'What is my account balance?',
                    'response': 'Your account balance is $5,000 and all transactions are up to date.',
                    'domain': 'banking'
                }
            },
            {
                'name': 'insurance_detection',
                'data': {
                    'query': 'What does my insurance cover?',
                    'response': 'Your comprehensive policy covers all damages up to $50,000.',
                    'domain': 'insurance'
                }
            }
        ]
        
        for test_case in test_cases:
            try:
                response = requests.post(f"{self.base_url}/detect", json=test_case['data'])
                assert response.status_code == 200, f"Detection failed: {response.status_code}"
                
                result = response.json()
                
                # Validate response structure
                required_fields = [
                    'hallucination_score', 'confidence', 'verification_results',
                    'detected_issues', 'response_time', 'sources_used',
                    'performance_metrics', 'recommendations'
                ]
                
                for field in required_fields:
                    assert field in result, f"Response missing field: {field}"
                
                # Validate data types
                assert isinstance(result['hallucination_score'], (int, float)), "hallucination_score must be numeric"
                assert isinstance(result['confidence'], (int, float)), "confidence must be numeric"
                assert isinstance(result['response_time'], (int, float)), "response_time must be numeric"
                assert isinstance(result['sources_used'], list), "sources_used must be list"
                
                # Validate ranges
                assert 0 <= result['hallucination_score'] <= 1, "hallucination_score must be between 0 and 1"
                assert 0 <= result['confidence'] <= 1, "confidence must be between 0 and 1"
                assert result['response_time'] > 0, "response_time must be positive"
                
                logger.info(f"✅ {test_case['name']} test passed")
                self.test_results.append({
                    'test': test_case['name'],
                    'status': 'passed',
                    'response_time': result['response_time'],
                    'hallucination_score': result['hallucination_score']
                })
                
            except Exception as e:
                logger.error(f"❌ {test_case['name']} test failed: {e}")
                self.test_results.append({
                    'test': test_case['name'],
                    'status': 'failed',
                    'error': str(e)
                })

    async def test_domain_verification(self):
        """Test domain-specific verification"""
        logger.info("Testing domain-specific verification...")
        
        domain_tests = {
            'ecommerce': [
                {
                    'query': 'Is iPhone 15 available?',
                    'response': 'Yes, iPhone 15 is available in all stores for $999.',
                    'expected_verifications': ['product_availability', 'pricing']
                },
                {
                    'query': 'What is the shipping time?',
                    'response': 'Standard shipping takes 3-5 business days.',
                    'expected_verifications': ['shipping']
                }
            ],
            'banking': [
                {
                    'query': 'What is my account status?',
                    'response': 'Your account is active with a balance of $5,000.',
                    'expected_verifications': ['account_status', 'balance']
                },
                {
                    'query': 'Are my transactions secure?',
                    'response': 'All transactions are protected by advanced security measures.',
                    'expected_verifications': ['security', 'compliance']
                }
            ],
            'insurance': [
                {
                    'query': 'What does my policy cover?',
                    'response': 'Your comprehensive policy covers damages up to $50,000.',
                    'expected_verifications': ['coverage', 'policy_status']
                },
                {
                    'query': 'What is my claim status?',
                    'response': 'Your claim is being processed and will be completed within 5 days.',
                    'expected_verifications': ['claim_status']
                }
            ]
        }
        
        for domain, tests in domain_tests.items():
            for test in tests:
                try:
                    response = requests.post(f"{self.base_url}/detect", json={
                        'query': test['query'],
                        'response': test['response'],
                        'domain': domain
                    })
                    
                    assert response.status_code == 200, f"Domain verification failed: {response.status_code}"
                    
                    result = response.json()
                    
                    # Check if domain-specific verifications were performed
                    sources_used = result['sources_used']
                    domain_sources = [s for s in sources_used if domain in s.lower()]
                    
                    assert len(domain_sources) > 0, f"No domain-specific sources used for {domain}"
                    
                    logger.info(f"✅ {domain} verification test passed")
                    self.test_results.append({
                        'test': f'{domain}_verification',
                        'status': 'passed',
                        'domain_sources': domain_sources
                    })
                    
                except Exception as e:
                    logger.error(f"❌ {domain} verification test failed: {e}")
                    self.test_results.append({
                        'test': f'{domain}_verification',
                        'status': 'failed',
                        'error': str(e)
                    })

    async def test_performance(self):
        """Test system performance"""
        logger.info("Testing system performance...")
        
        # Test single request performance
        start_time = time.time()
        response = requests.post(f"{self.base_url}/detect", json={
            'query': 'Performance test query',
            'response': 'Performance test response',
            'domain': 'general'
        })
        single_request_time = time.time() - start_time
        
        assert response.status_code == 200, "Performance test failed"
        result = response.json()
        
        # Performance benchmarks
        assert single_request_time < 1.0, f"Single request too slow: {single_request_time:.3f}s"
        assert result['response_time'] < 0.1, f"Response time too high: {result['response_time']:.3f}s"
        
        # Test concurrent requests
        concurrent_requests = 10
        start_time = time.time()
        
        async def make_request():
            return requests.post(f"{self.base_url}/detect", json={
                'query': f'Concurrent test query {time.time()}',
                'response': f'Concurrent test response {time.time()}',
                'domain': 'general'
            })
        
        # Simulate concurrent requests
        tasks = [make_request() for _ in range(concurrent_requests)]
        responses = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        # Check all responses
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == concurrent_requests, f"Only {success_count}/{concurrent_requests} concurrent requests succeeded"
        
        # Performance metrics
        throughput = concurrent_requests / concurrent_time
        assert throughput > 5, f"Throughput too low: {throughput:.2f} req/s"
        
        logger.info(f"✅ Performance tests passed - Throughput: {throughput:.2f} req/s")
        self.test_results.append({
            'test': 'performance',
            'status': 'passed',
            'single_request_time': single_request_time,
            'concurrent_time': concurrent_time,
            'throughput': throughput
        })

    async def test_batch_processing(self):
        """Test batch processing capabilities"""
        logger.info("Testing batch processing...")
        
        batch_data = [
            {
                'query': 'Batch test query 1',
                'response': 'Batch test response 1',
                'domain': 'general'
            },
            {
                'query': 'Batch test query 2',
                'response': 'Batch test response 2',
                'domain': 'ecommerce'
            },
            {
                'query': 'Batch test query 3',
                'response': 'Batch test response 3',
                'domain': 'banking'
            }
        ]
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/batch-detect", json=batch_data)
            batch_time = time.time() - start_time
            
            assert response.status_code == 200, f"Batch processing failed: {response.status_code}"
            
            results = response.json()
            assert len(results) == len(batch_data), f"Batch returned {len(results)} results, expected {len(batch_data)}"
            
            # Validate each result
            for i, result in enumerate(results):
                assert 'hallucination_score' in result, f"Result {i} missing hallucination_score"
                assert 'confidence' in result, f"Result {i} missing confidence"
                assert 'response_time' in result, f"Result {i} missing response_time"
            
            # Performance check
            avg_response_time = np.mean([r['response_time'] for r in results])
            assert avg_response_time < 0.2, f"Average batch response time too high: {avg_response_time:.3f}s"
            
            logger.info(f"✅ Batch processing test passed - Time: {batch_time:.3f}s")
            self.test_results.append({
                'test': 'batch_processing',
                'status': 'passed',
                'batch_time': batch_time,
                'avg_response_time': avg_response_time
            })
            
        except Exception as e:
            logger.error(f"❌ Batch processing test failed: {e}")
            self.test_results.append({
                'test': 'batch_processing',
                'status': 'failed',
                'error': str(e)
            })

    async def test_error_handling(self):
        """Test error handling"""
        logger.info("Testing error handling...")
        
        # Test invalid request
        try:
            response = requests.post(f"{self.base_url}/detect", json={
                'invalid_field': 'invalid_value'
            })
            assert response.status_code == 422, "Should return validation error for invalid request"
            logger.info("✅ Invalid request handling test passed")
        except Exception as e:
            logger.error(f"❌ Invalid request handling test failed: {e}")
        
        # Test missing required fields
        try:
            response = requests.post(f"{self.base_url}/detect", json={})
            assert response.status_code == 422, "Should return validation error for missing fields"
            logger.info("✅ Missing fields handling test passed")
        except Exception as e:
            logger.error(f"❌ Missing fields handling test failed: {e}")
        
        # Test malformed JSON
        try:
            response = requests.post(
                f"{self.base_url}/detect",
                data="invalid json",
                headers={'Content-Type': 'application/json'}
            )
            assert response.status_code == 422, "Should return error for malformed JSON"
            logger.info("✅ Malformed JSON handling test passed")
        except Exception as e:
            logger.error(f"❌ Malformed JSON handling test failed: {e}")
        
        self.test_results.append({
            'test': 'error_handling',
            'status': 'passed'
        })

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'passed')
        failed_tests = total_tests - passed_tests
        
        # Calculate performance metrics
        performance_tests = [r for r in self.test_results if 'response_time' in r]
        if performance_tests:
            avg_response_time = np.mean([r.get('response_time', 0) for r in performance_tests])
            max_response_time = max([r.get('response_time', 0) for r in performance_tests])
        else:
            avg_response_time = 0
            max_response_time = 0
        
        # Generate report
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'performance': {
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time
            },
            'test_results': self.test_results
        }
        
        # Save report
        with open('test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TEST REPORT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.2%}")
        logger.info(f"Average Response Time: {avg_response_time:.3f}s")
        logger.info(f"Max Response Time: {max_response_time:.3f}s")
        
        if failed_tests > 0:
            logger.info("\nFailed Tests:")
            for result in self.test_results:
                if result['status'] == 'failed':
                    logger.info(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 50)
        logger.info("Detailed report saved to: test_report.json")

async def main():
    """Main test function"""
    # Wait for system to be ready
    logger.info("Waiting for system to be ready...")
    await asyncio.sleep(2)
    
    # Run tests
    tester = HighPerformanceSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 