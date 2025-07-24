#!/usr/bin/env python3
"""
Integration Script: Hugging Face MCP Server with OpenTrustEval
Connects Hugging Face models with OpenTrustEval's MCP server for unified model management
"""

import asyncio
import json
import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mcp import ClientSession, StdioClientParameters
    from mcp.types import CallToolRequest, ListToolsRequest
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP library not available. Install with: pip install mcp")

try:
    from llm_engineering.llm_lifecycle import LLMLifecycleManager
    from llm_engineering.providers.huggingface_provider import HuggingFaceProvider
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"⚠️ LLM engineering not available: {e}")

class OpenTrustEvalHuggingFaceIntegration:
    """
    Integration class that connects Hugging Face models with OpenTrustEval
    Provides unified interface for model management and inference
    """
    
    def __init__(self):
        self.hf_mcp_client = None
        self.llm_manager = None
        self.integration_results = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("🔧 Initializing OpenTrustEval + Hugging Face Integration...")
        
        # Initialize LLM manager
        if LLM_AVAILABLE:
            try:
                self.llm_manager = LLMLifecycleManager()
                print("✅ LLM Manager initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM Manager: {e}")
        
        # Initialize Hugging Face MCP client
        if MCP_AVAILABLE:
            try:
                self.hf_mcp_client = HuggingFaceMCPClient()
                print("✅ Hugging Face MCP Client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Hugging Face MCP Client: {e}")
    
    async def connect_to_hf_mcp_server(self) -> bool:
        """Connect to Hugging Face MCP server"""
        if not self.hf_mcp_client:
            print("❌ Hugging Face MCP Client not available")
            return False
        
        try:
            success = await self.hf_mcp_client.connect()
            if success:
                print("✅ Connected to Hugging Face MCP Server")
                return True
            else:
                print("❌ Failed to connect to Hugging Face MCP Server")
                return False
        except Exception as e:
            print(f"❌ Error connecting to MCP server: {e}")
            return False
    
    async def test_huggingface_models(self):
        """Test Hugging Face models through MCP server"""
        print("\n🤗 Testing Hugging Face Models via MCP")
        print("=" * 50)
        
        if not self.hf_mcp_client:
            print("❌ Hugging Face MCP Client not available")
            return
        
        # Test basic functionality
        print("📋 Testing MCP Tools...")
        tools = await self.hf_mcp_client.list_tools()
        
        if tools:
            print(f"✅ Found {len(tools)} MCP tools")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
        else:
            print("❌ No MCP tools found")
            return
        
        # Test model loading
        print("\n📥 Testing Model Loading via MCP...")
        load_result = await self.hf_mcp_client.call_tool("load_model", {
            "model_name": "gpt2",
            "model_type": "causal"
        })
        print(f"Load Result: {load_result}")
        
        # Test text generation
        print("\n📝 Testing Text Generation via MCP...")
        gen_result = await self.hf_mcp_client.call_tool("generate_text", {
            "model_name": "gpt2",
            "prompt": "OpenTrustEval is a platform for",
            "max_length": 50,
            "temperature": 0.7
        })
        print(f"Generation Result: {gen_result}")
        
        self.integration_results['hf_mcp_tests'] = {
            'tools_found': len(tools),
            'load_result': load_result,
            'generation_result': gen_result
        }
    
    async def test_colbert_integration(self):
        """Test ColBERT-v2 integration specifically"""
        print("\n🔍 Testing ColBERT-v2 Integration")
        print("=" * 50)
        
        if not self.hf_mcp_client:
            print("❌ Hugging Face MCP Client not available")
            return
        
        try:
            # Load ColBERT-v2 via MCP
            print("📥 Loading ColBERT-v2 via MCP...")
            load_result = await self.hf_mcp_client.call_tool("load_model", {
                "model_name": "LinWeizheDragon/ColBERT-v2",
                "model_type": "retrieval"
            })
            print(f"ColBERT Load Result: {load_result}")
            
            # Test retrieval
            documents = [
                "OpenTrustEval provides comprehensive AI evaluation capabilities.",
                "ColBERT-v2 uses late interaction for efficient retrieval.",
                "The platform supports multiple model types and providers.",
                "Trust scoring involves accuracy, consistency, and verification.",
                "High-performance systems achieve 1000x speed improvements."
            ]
            
            print("\n🔍 Testing Document Retrieval via MCP...")
            retrieval_result = await self.hf_mcp_client.call_tool("retrieve_documents", {
                "model_name": "LinWeizheDragon/ColBERT-v2",
                "query": "What is OpenTrustEval?",
                "documents": documents,
                "top_k": 3
            })
            print(f"Retrieval Result: {retrieval_result}")
            
            self.integration_results['colbert_integration'] = {
                'load_result': load_result,
                'retrieval_result': retrieval_result
            }
            
        except Exception as e:
            print(f"❌ ColBERT integration test failed: {e}")
            self.integration_results['colbert_integration'] = {'error': str(e)}
    
    async def test_llm_lifecycle_integration(self):
        """Test integration with LLM Lifecycle Manager"""
        print("\n🔗 Testing LLM Lifecycle Integration")
        print("=" * 50)
        
        if not self.llm_manager:
            print("❌ LLM Manager not available")
            return
        
        try:
            # Check for Hugging Face providers in LLM manager
            models = self.llm_manager.list_models()
            hf_models = [m for m in models if 'huggingface' in m.lower() or 'colbert' in m.lower()]
            
            print(f"📋 Found {len(hf_models)} Hugging Face models in LLM Manager:")
            for model in hf_models:
                print(f"  - {model}")
            
            # Test direct provider access
            if 'colbert_v2_retrieval' in self.llm_manager.llm_providers:
                print("\n🔍 Testing Direct ColBERT Provider Access...")
                provider = self.llm_manager.llm_providers['colbert_v2_retrieval']
                
                # Get model info
                info = provider.get_model_info()
                print(f"Model Info: {info}")
                
                # Test retrieval if model is loaded
                if info.get('loaded', False):
                    documents = [
                        "OpenTrustEval is a comprehensive AI evaluation platform.",
                        "ColBERT-v2 provides efficient document retrieval.",
                        "The system supports multiple model types."
                    ]
                    
                    try:
                        results = provider._retrieve("What is OpenTrustEval?", documents, top_k=2)
                        print(f"Direct Retrieval Results: {results}")
                        
                        self.integration_results['llm_lifecycle'] = {
                            'hf_models_found': len(hf_models),
                            'colbert_loaded': info.get('loaded', False),
                            'direct_retrieval': results
                        }
                        
                    except Exception as e:
                        print(f"❌ Direct retrieval failed: {e}")
                        self.integration_results['llm_lifecycle'] = {
                            'hf_models_found': len(hf_models),
                            'colbert_loaded': info.get('loaded', False),
                            'error': str(e)
                        }
                else:
                    print("⚠️ ColBERT model not loaded in LLM Manager")
                    self.integration_results['llm_lifecycle'] = {
                        'hf_models_found': len(hf_models),
                        'colbert_loaded': False
                    }
            else:
                print("❌ ColBERT provider not found in LLM Manager")
                self.integration_results['llm_lifecycle'] = {
                    'hf_models_found': len(hf_models),
                    'colbert_found': False
                }
                
        except Exception as e:
            print(f"❌ LLM Lifecycle integration test failed: {e}")
            self.integration_results['llm_lifecycle'] = {'error': str(e)}
    
    async def test_unified_interface(self):
        """Test unified interface combining both approaches"""
        print("\n🔄 Testing Unified Interface")
        print("=" * 50)
        
        # Test both MCP and direct access
        unified_results = {}
        
        # Test via MCP
        if self.hf_mcp_client:
            try:
                print("📝 Testing Text Generation via MCP...")
                mcp_result = await self.hf_mcp_client.call_tool("generate_text", {
                    "model_name": "gpt2",
                    "prompt": "AI evaluation platforms provide",
                    "max_length": 30
                })
                unified_results['mcp_generation'] = mcp_result
                print(f"MCP Result: {mcp_result}")
            except Exception as e:
                unified_results['mcp_generation'] = {'error': str(e)}
        
        # Test via direct LLM manager
        if self.llm_manager and 'gpt2_generation' in self.llm_manager.llm_providers:
            try:
                print("📝 Testing Text Generation via LLM Manager...")
                provider = self.llm_manager.llm_providers['gpt2_generation']
                direct_result = provider.generate("AI evaluation platforms provide", max_length=30)
                unified_results['direct_generation'] = direct_result
                print(f"Direct Result: {direct_result}")
            except Exception as e:
                unified_results['direct_generation'] = {'error': str(e)}
        
        self.integration_results['unified_interface'] = unified_results
    
    async def run_comprehensive_test(self):
        """Run comprehensive integration test"""
        print("🚀 Starting Comprehensive OpenTrustEval + Hugging Face Integration Test")
        print("=" * 80)
        
        start_time = time.time()
        
        # Connect to MCP server
        mcp_connected = await self.connect_to_hf_mcp_server()
        
        try:
            # Run all tests
            await self.test_huggingface_models()
            await self.test_colbert_integration()
            await self.test_llm_lifecycle_integration()
            await self.test_unified_interface()
            
            # Calculate test duration
            end_time = time.time()
            test_duration = end_time - start_time
            
            # Generate summary
            self._generate_test_summary(test_duration, mcp_connected)
            
        except Exception as e:
            print(f"\n❌ Comprehensive test failed: {e}")
            self.integration_results['test_error'] = str(e)
        
        finally:
            # Disconnect from MCP server
            if self.hf_mcp_client:
                await self.hf_mcp_client.disconnect()
    
    def _generate_test_summary(self, duration: float, mcp_connected: bool):
        """Generate test summary"""
        print("\n📊 Integration Test Summary")
        print("=" * 50)
        print(f"⏱️  Test Duration: {duration:.2f} seconds")
        print(f"🔗 MCP Server Connected: {'✅ Yes' if mcp_connected else '❌ No'}")
        print(f"🤖 LLM Manager Available: {'✅ Yes' if self.llm_manager else '❌ No'}")
        
        # Save results
        output_file = 'opentrusteval_huggingface_integration_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.integration_results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {output_file}")
        
        # Print key results
        if 'colbert_integration' in self.integration_results:
            colbert_result = self.integration_results['colbert_integration']
            if 'error' not in colbert_result:
                print("✅ ColBERT-v2 Integration: Successful")
            else:
                print(f"❌ ColBERT-v2 Integration: Failed - {colbert_result['error']}")
        
        if 'llm_lifecycle' in self.integration_results:
            llm_result = self.integration_results['llm_lifecycle']
            if 'error' not in llm_result:
                print(f"✅ LLM Lifecycle Integration: {llm_result.get('hf_models_found', 0)} models found")
            else:
                print(f"❌ LLM Lifecycle Integration: Failed - {llm_result['error']}")
        
        print("\n🎉 Integration Test Completed!")

class HuggingFaceMCPClient:
    """Simple MCP client for testing"""
    
    def __init__(self):
        self.session = None
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        if not MCP_AVAILABLE:
            return False
        
        try:
            params = StdioClientParameters(
                command="python",
                args=["mcp_server/huggingface_mcp_server.py"]
            )
            self.session = ClientSession(params)
            await self.session.__aenter__()
            return True
        except Exception as e:
            print(f"MCP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.session:
            await self.session.__aexit__(None, None, None)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.session:
            return []
        
        try:
            request = ListToolsRequest()
            response = await self.session.list_tools(request)
            
            tools = []
            for tool in response.tools:
                tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema
                })
            return tools
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool"""
        if not self.session:
            return "Error: Not connected"
        
        try:
            request = CallToolRequest(name=name, arguments=arguments)
            response = await self.session.call_tool(request)
            
            if response.content:
                return response.content[0].text
            else:
                return "No response content"
        except Exception as e:
            return f"Error calling tool {name}: {e}"

async def main():
    """Main entry point"""
    integration = OpenTrustEvalHuggingFaceIntegration()
    await integration.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 