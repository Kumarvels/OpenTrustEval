#!/usr/bin/env python3
"""
Simple Unit Test for High Performance System Components
Tests core functionality without requiring external services
"""

import asyncio
import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test core system imports
        from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
        print("   ✅ UltimateMoESystem imported successfully")
        
        from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
        print("   ✅ AdvancedExpertEnsemble imported successfully")
        
        from high_performance_system.core.intelligent_domain_router import IntelligentDomainRouter
        print("   ✅ IntelligentDomainRouter imported successfully")
        
        from high_performance_system.core.enhanced_dataset_profiler import EnhancedDatasetProfiler
        print("   ✅ EnhancedDatasetProfiler imported successfully")
        
        from high_performance_system.core.comprehensive_pii_detector import ComprehensivePIIDetector
        print("   ✅ ComprehensivePIIDetector imported successfully")
        
        from high_performance_system.core.advanced_trust_scorer import AdvancedTrustScorer
        print("   ✅ AdvancedTrustScorer imported successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_system_initialization():
    """Test system initialization"""
    print("\n🧪 Testing system initialization...")
    
    try:
        from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
        
        # Initialize the system
        system = UltimateMoESystem()
        print("   ✅ UltimateMoESystem initialized successfully")
        
        # Check that core components are available
        assert hasattr(system, 'moe_verifier'), "Missing moe_verifier"
        assert hasattr(system, 'expert_ensemble'), "Missing expert_ensemble"
        assert hasattr(system, 'intelligent_router'), "Missing intelligent_router"
        assert hasattr(system, 'dataset_profiler'), "Missing dataset_profiler"
        assert hasattr(system, 'pii_detector'), "Missing pii_detector"
        assert hasattr(system, 'trust_scorer'), "Missing trust_scorer"
        
        print("   ✅ All core components verified")
        return True
    except Exception as e:
        print(f"   ❌ System initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_expert_ensemble():
    """Test expert ensemble functionality"""
    print("\n🧪 Testing expert ensemble...")
    
    try:
        from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
        
        ensemble = AdvancedExpertEnsemble()
        print("   ✅ Expert ensemble initialized")
        
        # Test domain experts
        assert hasattr(ensemble, 'ecommerce_expert'), "Missing ecommerce_expert"
        assert hasattr(ensemble, 'banking_expert'), "Missing banking_expert"
        assert hasattr(ensemble, 'insurance_expert'), "Missing insurance_expert"
        assert hasattr(ensemble, 'healthcare_expert'), "Missing healthcare_expert"
        assert hasattr(ensemble, 'legal_expert'), "Missing legal_expert"
        assert hasattr(ensemble, 'finance_expert'), "Missing finance_expert"
        assert hasattr(ensemble, 'technology_expert'), "Missing technology_expert"
        
        print("   ✅ All domain experts verified")
        return True
    except Exception as e:
        print(f"   ❌ Expert ensemble test failed: {e}")
        traceback.print_exc()
        return False

def test_domain_router():
    """Test domain router functionality"""
    print("\n🧪 Testing domain router...")
    
    try:
        from high_performance_system.core.intelligent_domain_router import IntelligentDomainRouter
        
        router = IntelligentDomainRouter()
        print("   ✅ Domain router initialized")
        
        # Test routing strategies
        assert hasattr(router, 'keyword_router'), "Missing keyword_router"
        assert hasattr(router, 'semantic_router'), "Missing semantic_router"
        assert hasattr(router, 'ml_router'), "Missing ml_router"
        assert hasattr(router, 'hybrid_router'), "Missing hybrid_router"
        
        print("   ✅ All routing strategies verified")
        return True
    except Exception as e:
        print(f"   ❌ Domain router test failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_profiler():
    """Test dataset profiler functionality"""
    print("\n🧪 Testing dataset profiler...")
    
    try:
        from high_performance_system.core.enhanced_dataset_profiler import EnhancedDatasetProfiler
        
        profiler = EnhancedDatasetProfiler()
        print("   ✅ Dataset profiler initialized")
        
        # Test profiling methods
        assert hasattr(profiler, 'profile_text'), "Missing profile_text method"
        assert hasattr(profiler, 'analyze_quality'), "Missing analyze_quality method"
        assert hasattr(profiler, 'detect_anomalies'), "Missing detect_anomalies method"
        
        print("   ✅ All profiling methods verified")
        return True
    except Exception as e:
        print(f"   ❌ Dataset profiler test failed: {e}")
        traceback.print_exc()
        return False

def test_pii_detector():
    """Test PII detector functionality"""
    print("\n🧪 Testing PII detector...")
    
    try:
        from high_performance_system.core.comprehensive_pii_detector import ComprehensivePIIDetector
        
        detector = ComprehensivePIIDetector()
        print("   ✅ PII detector initialized")
        
        # Test detection methods
        assert hasattr(detector, 'detect_pii'), "Missing detect_pii method"
        assert hasattr(detector, 'analyze_privacy_risk'), "Missing analyze_privacy_risk method"
        assert hasattr(detector, 'check_compliance'), "Missing check_compliance method"
        
        print("   ✅ All PII detection methods verified")
        return True
    except Exception as e:
        print(f"   ❌ PII detector test failed: {e}")
        traceback.print_exc()
        return False

def test_trust_scorer():
    """Test trust scorer functionality"""
    print("\n🧪 Testing trust scorer...")
    
    try:
        from high_performance_system.core.advanced_trust_scorer import AdvancedTrustScorer
        
        scorer = AdvancedTrustScorer()
        print("   ✅ Trust scorer initialized")
        
        # Test scoring methods
        assert hasattr(scorer, 'score_trust'), "Missing score_trust method"
        assert hasattr(scorer, 'analyze_trust_factors'), "Missing analyze_trust_factors method"
        assert hasattr(scorer, 'calculate_confidence'), "Missing calculate_confidence method"
        
        print("   ✅ All trust scoring methods verified")
        return True
    except Exception as e:
        print(f"   ❌ Trust scorer test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that essential files and directories exist"""
    print("\n🧪 Testing file structure...")
    
    essential_files = [
        "README.md",
        "README_HIGH_PERFORMANCE.md",
        "requirements_high_performance.txt",
        "setup.py",
        "Dockerfile",
        "LICENSE",
        ".gitignore"
    ]
    
    essential_dirs = [
        "high_performance_system/",
        "docs/",
        ".git/",
        ".github/"
    ]
    
    all_good = True
    
    # Check essential files
    for file_name in essential_files:
        if Path(file_name).exists():
            print(f"   ✅ Essential file present: {file_name}")
        else:
            print(f"   ❌ Missing essential file: {file_name}")
            all_good = False
    
    # Check essential directories
    for dir_name in essential_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ Essential directory present: {dir_name}")
        else:
            print(f"   ❌ Missing essential directory: {dir_name}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Unit Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_system_initialization,
        test_expert_ensemble,
        test_domain_router,
        test_dataset_profiler,
        test_pii_detector,
        test_trust_scorer,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 