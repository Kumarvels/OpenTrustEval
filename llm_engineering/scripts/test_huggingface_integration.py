#!/usr/bin/env python3
"""
Test Script for Hugging Face Integration with OpenTrustEval
Demonstrates ColBERT-v2 retrieval and other Hugging Face models
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from llm_engineering.llm_lifecycle import LLMLifecycleManager
    from llm_engineering.providers.huggingface_provider import HuggingFaceProvider
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"Warning: LLM engineering not available: {e}")

def test_colbert_retrieval():
    """Test ColBERT-v2 retrieval functionality"""
    print("🔍 Testing ColBERT-v2 Retrieval...")
    
    try:
        # Initialize provider
        config = {
            'model_name': 'LinWeizheDragon/ColBERT-v2',
            'model_type': 'retrieval',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        provider = HuggingFaceProvider(config)
        
        # Sample documents for retrieval
        documents = [
            "OpenTrustEval is a comprehensive AI evaluation platform for trust scoring and hallucination detection.",
            "ColBERT-v2 is a state-of-the-art retrieval model that uses late interaction for efficient document search.",
            "The platform includes modules for LLM management, data engineering, security, and research.",
            "High-performance systems can achieve 1000x speed improvements through optimized architectures.",
            "Trust scoring involves multiple factors including accuracy, consistency, and source verification."
        ]
        
        # Test queries
        queries = [
            "What is OpenTrustEval?",
            "How does ColBERT work?",
            "What are the main components?",
            "How fast is the system?",
            "What is trust scoring?"
        ]
        
        results = {}
        for query in queries:
            print(f"\n📝 Query: {query}")
            start_time = time.time()
            
            try:
                retrieval_results = provider._retrieve(query, documents, top_k=3)
                end_time = time.time()
                
                print(f"⏱️  Retrieval time: {end_time - start_time:.3f}s")
                print("📊 Top Results:")
                
                for i, result in enumerate(retrieval_results['results']):
                    print(f"  {i+1}. Score: {result['score']:.4f} | {result['document'][:80]}...")
                
                results[query] = retrieval_results
                
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
                results[query] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ ColBERT test failed: {e}")
        return {'error': str(e)}

def test_text_generation():
    """Test text generation models"""
    print("\n🤖 Testing Text Generation Models...")
    
    models_to_test = [
        {
            'name': 'gpt2',
            'type': 'causal',
            'prompt': 'The future of artificial intelligence is'
        },
        {
            'name': 'microsoft/DialoGPT-medium',
            'type': 'causal',
            'prompt': 'Hello, how are you today?'
        }
    ]
    
    results = {}
    
    for model_config in models_to_test:
        print(f"\n📝 Testing {model_config['name']}...")
        
        try:
            config = {
                'model_name': model_config['name'],
                'model_type': model_config['type'],
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            provider = HuggingFaceProvider(config)
            
            start_time = time.time()
            generated_text = provider.generate(
                model_config['prompt'], 
                max_length=50, 
                temperature=0.7
            )
            end_time = time.time()
            
            print(f"⏱️  Generation time: {end_time - start_time:.3f}s")
            print(f"📝 Prompt: {model_config['prompt']}")
            print(f"🤖 Generated: {generated_text}")
            
            results[model_config['name']] = {
                'prompt': model_config['prompt'],
                'generated': generated_text,
                'time': end_time - start_time
            }
            
        except Exception as e:
            print(f"❌ Error with {model_config['name']}: {e}")
            results[model_config['name']] = {'error': str(e)}
    
    return results

def test_seq2seq_models():
    """Test sequence-to-sequence models"""
    print("\n🔄 Testing Seq2Seq Models...")
    
    models_to_test = [
        {
            'name': 't5-small',
            'type': 'seq2seq',
            'prompt': 'translate English to French: Hello world'
        },
        {
            'name': 'facebook/bart-base',
            'type': 'seq2seq',
            'prompt': 'summarize: OpenTrustEval is a comprehensive AI evaluation platform that provides advanced trust scoring and hallucination detection capabilities for large language models.'
        }
    ]
    
    results = {}
    
    for model_config in models_to_test:
        print(f"\n📝 Testing {model_config['name']}...")
        
        try:
            config = {
                'model_name': model_config['name'],
                'model_type': model_config['type'],
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            provider = HuggingFaceProvider(config)
            
            start_time = time.time()
            generated_text = provider.generate(
                model_config['prompt'], 
                max_length=100, 
                temperature=0.7
            )
            end_time = time.time()
            
            print(f"⏱️  Generation time: {end_time - start_time:.3f}s")
            print(f"📝 Input: {model_config['prompt']}")
            print(f"🔄 Output: {generated_text}")
            
            results[model_config['name']] = {
                'input': model_config['prompt'],
                'output': generated_text,
                'time': end_time - start_time
            }
            
        except Exception as e:
            print(f"❌ Error with {model_config['name']}: {e}")
            results[model_config['name']] = {'error': str(e)}
    
    return results

def test_classification():
    """Test classification models"""
    print("\n🏷️ Testing Classification Models...")
    
    try:
        config = {
            'model_name': 'bert-base-uncased',
            'model_type': 'classification',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        provider = HuggingFaceProvider(config)
        
        # Test texts for classification
        texts = [
            "I love this product, it's amazing!",
            "This is terrible, I hate it.",
            "The weather is nice today.",
            "The movie was okay, nothing special."
        ]
        
        results = {}
        
        for text in texts:
            print(f"\n📝 Classifying: {text}")
            
            try:
                start_time = time.time()
                
                # For classification, we'll use the model directly
                inputs = provider.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(provider.device)
                
                with torch.no_grad():
                    outputs = provider.model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)
                
                end_time = time.time()
                
                print(f"⏱️  Classification time: {end_time - start_time:.3f}s")
                print(f"🏷️  Confidence: {predictions.max().item():.4f}")
                
                results[text] = {
                    'confidence': predictions.max().item(),
                    'time': end_time - start_time
                }
                
            except Exception as e:
                print(f"❌ Error classifying '{text}': {e}")
                results[text] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return {'error': str(e)}

def test_model_info():
    """Test model information retrieval"""
    print("\nℹ️ Testing Model Information...")
    
    models_to_test = [
        {'name': 'gpt2', 'type': 'causal'},
        {'name': 't5-small', 'type': 'seq2seq'},
        {'name': 'bert-base-uncased', 'type': 'classification'}
    ]
    
    results = {}
    
    for model_config in models_to_test:
        print(f"\n📊 Getting info for {model_config['name']}...")
        
        try:
            config = {
                'model_name': model_config['name'],
                'model_type': model_config['type'],
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            provider = HuggingFaceProvider(config)
            info = provider.get_model_info()
            
            print(f"📋 Model Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            results[model_config['name']] = info
            
        except Exception as e:
            print(f"❌ Error getting info for {model_config['name']}: {e}")
            results[model_config['name']] = {'error': str(e)}
    
    return results

def test_llm_lifecycle_integration():
    """Test integration with LLM Lifecycle Manager"""
    print("\n🔗 Testing LLM Lifecycle Integration...")
    
    try:
        manager = LLMLifecycleManager()
        
        # Check if Hugging Face providers are loaded
        hf_providers = [name for name in manager.list_models() if 'huggingface' in name.lower() or 'colbert' in name.lower()]
        
        print(f"📋 Found Hugging Face providers: {hf_providers}")
        
        results = {}
        
        for provider_name in hf_providers:
            print(f"\n🔍 Testing provider: {provider_name}")
            
            try:
                provider = manager.llm_providers.get(provider_name)
                if provider:
                    info = provider.get_model_info()
                    print(f"✅ Provider loaded: {info}")
                    results[provider_name] = info
                else:
                    print(f"❌ Provider not found: {provider_name}")
                    results[provider_name] = {'error': 'Provider not found'}
                    
            except Exception as e:
                print(f"❌ Error with provider {provider_name}: {e}")
                results[provider_name] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ LLM Lifecycle integration test failed: {e}")
        return {'error': str(e)}

def main():
    """Run all tests"""
    print("🚀 Starting Hugging Face Integration Tests")
    print("=" * 60)
    
    if not LLM_AVAILABLE:
        print("❌ LLM Engineering module not available")
        return
    
    # Import torch for device detection
    try:
        import torch
        print(f"🔧 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("⚠️ PyTorch not available")
        return
    
    all_results = {}
    
    # Run tests
    try:
        all_results['colbert_retrieval'] = test_colbert_retrieval()
    except Exception as e:
        all_results['colbert_retrieval'] = {'error': str(e)}
    
    try:
        all_results['text_generation'] = test_text_generation()
    except Exception as e:
        all_results['text_generation'] = {'error': str(e)}
    
    try:
        all_results['seq2seq'] = test_seq2seq_models()
    except Exception as e:
        all_results['seq2seq'] = {'error': str(e)}
    
    try:
        all_results['classification'] = test_classification()
    except Exception as e:
        all_results['classification'] = {'error': str(e)}
    
    try:
        all_results['model_info'] = test_model_info()
    except Exception as e:
        all_results['model_info'] = {'error': str(e)}
    
    try:
        all_results['lifecycle_integration'] = test_llm_lifecycle_integration()
    except Exception as e:
        all_results['lifecycle_integration'] = {'error': str(e)}
    
    # Save results
    output_file = 'huggingface_integration_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📊 Test results saved to: {output_file}")
    print("\n✅ Hugging Face Integration Tests Completed!")

if __name__ == "__main__":
    main() 