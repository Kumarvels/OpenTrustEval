#!/usr/bin/env python3
"""
Simple test script to verify MoE integration
"""

import requests
import json
import time

def test_moe_integration():
    """Test MoE integration"""
    base_url = "http://localhost:8003"
    
    print("🔍 Testing MoE Integration...")
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        health_data = response.json()
        print(f"   MoE Status: {health_data.get('moe_status', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: MoE status
    print("\n2. Testing MoE status...")
    try:
        response = requests.get(f"{base_url}/moe-status")
        print(f"✅ MoE status: {response.status_code}")
        moe_data = response.json()
        print(f"   Available: {moe_data.get('available', False)}")
        print(f"   Components: {moe_data.get('components', {})}")
    except Exception as e:
        print(f"❌ MoE status failed: {e}")
        return
    
    # Test 3: Simple detection
    print("\n3. Testing simple detection...")
    try:
        test_data = {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "domain": "general",
            "enable_advanced_features": True
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/detect", json=test_data)
        end_time = time.time()
        
        print(f"✅ Detection: {response.status_code}")
        print(f"   Response time: {end_time - start_time:.3f}s")
        
        result = response.json()
        print(f"   Is hallucination: {result.get('is_hallucination', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 'unknown')}")
        print(f"   Domain: {result.get('domain', 'unknown')}")
        print(f"   Reasoning: {result.get('reasoning', 'unknown')}")
        
        # Check if MoE components were used
        if result.get('expert_results'):
            print(f"   Expert results: {len(result['expert_results'])} experts used")
        else:
            print("   ⚠️  No expert results found - using fallback")
            
        if result.get('routing_info'):
            print(f"   Routing info: {result['routing_info']}")
        else:
            print("   ⚠️  No routing info found")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return
    
    # Test 4: Domain-specific detection
    print("\n4. Testing domain-specific detection...")
    try:
        test_data = {
            "query": "What is the price of iPhone 15?",
            "response": "The iPhone 15 costs $999 and is available in all stores.",
            "domain": "ecommerce",
            "enable_advanced_features": True
        }
        
        response = requests.post(f"{base_url}/detect", json=test_data)
        print(f"✅ Ecommerce detection: {response.status_code}")
        
        result = response.json()
        print(f"   Domain: {result.get('domain', 'unknown')}")
        print(f"   Sources used: {result.get('sources_used', [])}")
        
        # Check for domain-specific sources
        sources = result.get('sources_used', [])
        domain_sources = [s for s in sources if 'ecommerce' in s.lower()]
        if domain_sources:
            print(f"   ✅ Domain-specific sources: {domain_sources}")
        else:
            print("   ⚠️  No domain-specific sources found")
            
    except Exception as e:
        print(f"❌ Domain-specific detection failed: {e}")
        return
    
    # Test 5: Performance metrics
    print("\n5. Testing performance metrics...")
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"✅ Metrics: {response.status_code}")
        
        metrics = response.json()
        print(f"   Total requests: {metrics.get('total_requests', 0)}")
        print(f"   Avg response time: {metrics.get('avg_response_time', 0):.3f}s")
        print(f"   Requests per second: {metrics.get('requests_per_second', 0):.2f}")
        
        if metrics.get('moe_metrics'):
            print(f"   MoE metrics: {metrics['moe_metrics']}")
        else:
            print("   ⚠️  No MoE metrics found")
            
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
        return
    
    print("\n🎉 MoE Integration Test Complete!")

if __name__ == "__main__":
    test_moe_integration() 