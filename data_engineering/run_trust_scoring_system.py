#!/usr/bin/env python3
"""
Trust Scoring System Launcher
Easy launcher for batch testing and dashboard
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'scipy', 'streamlit', 'plotly', 'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def run_batch_test_suite():
    """Run the comprehensive batch test suite"""
    print("🚀 Starting Comprehensive Batch Trust Scoring Test Suite...")
    print("=" * 80)
    
    try:
        # Import and run the batch test suite
        from batch_trust_scoring_test_suite import BatchTrustScoringTestSuite
        
        test_suite = BatchTrustScoringTestSuite()
        results = test_suite.run_complete_test_suite()
        
        print("\n" + "=" * 80)
        print("✅ BATCH TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Print summary
        if 'report_generation' in results and 'summary' in results['report_generation']:
            summary = results['report_generation']['summary']
            print(f"📊 Total Tests: {summary['total_tests']}")
            print(f"✅ Passed: {summary['passed_tests']}")
            print(f"❌ Failed: {summary['failed_tests']}")
            print(f"📈 Success Rate: {summary['success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test suite failed: {e}")
        return False

def run_dashboard(port=8501):
    """Run the Streamlit dashboard"""
    print(f"🌐 Starting Trust Scoring Dashboard on port {port}...")
    print("=" * 80)
    
    try:
        # Check if dashboard file exists
        dashboard_file = Path(__file__).parent / "trust_scoring_dashboard.py"
        if not dashboard_file.exists():
            print(f"❌ Dashboard file not found: {dashboard_file}")
            return False
        
        # Run Streamlit dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file), 
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ]
        
        print(f"🔗 Dashboard will be available at: http://localhost:{port}")
        print("📊 Press Ctrl+C to stop the dashboard")
        print("=" * 80)
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify system functionality"""
    print("⚡ Running Quick System Test...")
    print("=" * 50)
    
    try:
        # Test basic imports
        from advanced_trust_scoring import AdvancedTrustScoringEngine
        from cleanlab_integration import FallbackDataQualityManager
        from dataset_integration import DatasetManager
        
        print("✅ All components imported successfully")
        
        # Test basic functionality
        engine = AdvancedTrustScoringEngine()
        print("✅ Advanced Trust Scoring Engine initialized")
        
        manager = FallbackDataQualityManager()
        print("✅ Fallback Quality Manager initialized")
        
        dataset_manager = DatasetManager()
        print("✅ Dataset Manager initialized")
        
        # Create a simple test dataset
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        # Test trust scoring
        result = engine.calculate_advanced_trust_score(test_data)
        print(f"✅ Trust scoring test completed: {result.get('trust_score', 'N/A'):.3f}")
        
        # Test quality assessment
        quality_result = manager.calculate_data_trust_score(test_data)
        print(f"✅ Quality assessment test completed: {quality_result.get('trust_score', 'N/A'):.3f}")
        
        print("\n" + "=" * 50)
        print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Trust Scoring System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trust_scoring_system.py --test                    # Run quick test
  python run_trust_scoring_system.py --batch                   # Run batch test suite
  python run_trust_scoring_system.py --dashboard               # Start dashboard
  python run_trust_scoring_system.py --dashboard --port 8502   # Start dashboard on port 8502
  python run_trust_scoring_system.py --all                     # Run all components
        """
    )
    
    parser.add_argument("--test", action="store_true", 
                       help="Run quick system test")
    parser.add_argument("--batch", action="store_true", 
                       help="Run comprehensive batch test suite")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Start Streamlit dashboard")
    parser.add_argument("--port", type=int, default=8501, 
                       help="Port for dashboard (default: 8501)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all components (test + batch + dashboard)")
    
    args = parser.parse_args()
    
    # Print banner
    print("🔍 OpenTrustEval - Trust Scoring System")
    print("=" * 50)
    print("Comprehensive data quality and trust assessment platform")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        sys.exit(1)
    
    # Determine what to run
    if args.all:
        print("\n🚀 Running all components...")
        
        # Quick test
        print("\n1️⃣ Running Quick Test...")
        if not run_quick_test():
            print("❌ Quick test failed, stopping")
            sys.exit(1)
        
        # Batch test suite
        print("\n2️⃣ Running Batch Test Suite...")
        if not run_batch_test_suite():
            print("❌ Batch test suite failed, stopping")
            sys.exit(1)
        
        # Dashboard
        print("\n3️⃣ Starting Dashboard...")
        run_dashboard(args.port)
        
    elif args.test:
        run_quick_test()
        
    elif args.batch:
        run_batch_test_suite()
        
    elif args.dashboard:
        run_dashboard(args.port)
        
    else:
        # Interactive mode
        print("\n🎯 What would you like to do?")
        print("1. Quick Test (verify system functionality)")
        print("2. Batch Test Suite (comprehensive testing)")
        print("3. Dashboard (monitoring and analytics)")
        print("4. All (test + batch + dashboard)")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    run_quick_test()
                    break
                elif choice == "2":
                    run_batch_test_suite()
                    break
                elif choice == "3":
                    port = input("Enter dashboard port (default: 8501): ").strip()
                    port = int(port) if port.isdigit() else 8501
                    run_dashboard(port)
                    break
                elif choice == "4":
                    print("\n🚀 Running all components...")
                    
                    print("\n1️⃣ Running Quick Test...")
                    if not run_quick_test():
                        print("❌ Quick test failed, stopping")
                        break
                    
                    print("\n2️⃣ Running Batch Test Suite...")
                    if not run_batch_test_suite():
                        print("❌ Batch test suite failed, stopping")
                        break
                    
                    print("\n3️⃣ Starting Dashboard...")
                    run_dashboard(args.port)
                    break
                elif choice == "5":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 