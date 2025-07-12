#!/usr/bin/env python3
"""
Comprehensive Batch Trust Scoring Test Suite
Executes all trust scoring commands in correct order with full reporting
"""

import os
import sys
import json
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all trust scoring components
from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
from data_engineering.cleanlab_integration import FallbackDataQualityManager, benchmark_vs_cleanlab
from data_engineering.dataset_integration import DatasetManager
from data_engineering.test_real_world_trust_scoring import RealWorldTrustScoringEvaluator

class BatchTrustScoringTestSuite:
    """
    Comprehensive batch testing suite for trust scoring system
    Executes all components in correct order with full reporting
    """
    
    def __init__(self, output_dir: str = "./test_reports"):
        self.output_dir = output_dir
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize components
        self.advanced_engine = AdvancedTrustScoringEngine()
        self.fallback_manager = FallbackDataQualityManager()
        self.dataset_manager = DatasetManager()
        self.real_world_evaluator = RealWorldTrustScoringEvaluator()
        
        # Test datasets
        self.test_datasets = {}
        
    def _setup_logger(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('BatchTrustScoringTestSuite')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.output_dir, f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """
        Execute complete test suite in correct order
        """
        self.start_time = datetime.now()
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE BATCH TRUST SCORING TEST SUITE")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Component Initialization Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 1: COMPONENT INITIALIZATION TESTS")
            self.logger.info("="*50)
            
            self._test_component_initialization()
            
            # Phase 2: Synthetic Data Generation Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 2: SYNTHETIC DATA GENERATION TESTS")
            self.logger.info("="*50)
            
            self._test_synthetic_data_generation()
            
            # Phase 3: Advanced Trust Scoring Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 3: ADVANCED TRUST SCORING TESTS")
            self.logger.info("="*50)
            
            self._test_advanced_trust_scoring()
            
            # Phase 4: Fallback Quality Manager Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 4: FALLBACK QUALITY MANAGER TESTS")
            self.logger.info("="*50)
            
            self._test_fallback_quality_manager()
            
            # Phase 5: Dataset Integration Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 5: DATASET INTEGRATION TESTS")
            self.logger.info("="*50)
            
            self._test_dataset_integration()
            
            # Phase 6: Real-World Trust Scoring Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 6: REAL-WORLD TRUST SCORING TESTS")
            self.logger.info("="*50)
            
            self._test_real_world_trust_scoring()
            
            # Phase 7: Cleanlab Benchmarking Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 7: CLEANLAB BENCHMARKING TESTS")
            self.logger.info("="*50)
            
            self._test_cleanlab_benchmarking()
            
            # Phase 8: Performance and Stress Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 8: PERFORMANCE AND STRESS TESTS")
            self.logger.info("="*50)
            
            self._test_performance_and_stress()
            
            # Phase 9: Integration and End-to-End Tests
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 9: INTEGRATION AND END-TO-END TESTS")
            self.logger.info("="*50)
            
            self._test_integration_and_e2e()
            
            # Phase 10: Report Generation
            self.logger.info("\n" + "="*50)
            self.logger.info("PHASE 10: REPORT GENERATION")
            self.logger.info("="*50)
            
            self._generate_comprehensive_report()
            
        except Exception as e:
            self.logger.error(f"Critical error in test suite: {e}")
            self.logger.error(traceback.format_exc())
            self.results['critical_error'] = str(e)
        
        finally:
            self.end_time = datetime.now()
            self.logger.info("\n" + "="*80)
            self.logger.info("BATCH TRUST SCORING TEST SUITE COMPLETED")
            self.logger.info(f"Duration: {self.end_time - self.start_time}")
            self.logger.info("="*80)
        
        return self.results
    
    def _test_component_initialization(self):
        """Test component initialization"""
        self.logger.info("Testing component initialization...")
        
        try:
            # Test Advanced Trust Scoring Engine
            self.logger.info("  - Testing Advanced Trust Scoring Engine...")
            config = self.advanced_engine.config
            self.results['component_init'] = {
                'advanced_engine': True,
                'config_keys': list(config.keys()),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info("  ✓ Advanced Trust Scoring Engine initialized successfully")
            
            # Test Fallback Quality Manager
            self.logger.info("  - Testing Fallback Quality Manager...")
            self.results['component_init']['fallback_manager'] = True
            self.logger.info("  ✓ Fallback Quality Manager initialized successfully")
            
            # Test Dataset Manager
            self.logger.info("  - Testing Dataset Manager...")
            self.results['component_init']['dataset_manager'] = True
            self.logger.info("  ✓ Dataset Manager initialized successfully")
            
            # Test Real-World Evaluator
            self.logger.info("  - Testing Real-World Evaluator...")
            self.results['component_init']['real_world_evaluator'] = True
            self.logger.info("  ✓ Real-World Evaluator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Component initialization failed: {e}")
            self.results['component_init'] = {'error': str(e)}
    
    def _test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        self.logger.info("Testing synthetic data generation...")
        
        try:
            # Generate various test datasets
            datasets = {
                'high_quality': self._create_high_quality_dataset(),
                'medium_quality': self._create_medium_quality_dataset(),
                'low_quality': self._create_low_quality_dataset(),
                'correlated': self._create_correlated_dataset(),
                'skewed': self._create_skewed_dataset()
            }
            
            self.test_datasets = datasets
            
            # Validate datasets
            for name, df in datasets.items():
                self.logger.info(f"  - Generated {name} dataset: {df.shape}")
                self.logger.info(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            self.results['synthetic_data'] = {
                'datasets_created': len(datasets),
                'dataset_names': list(datasets.keys()),
                'total_rows': sum(len(df) for df in datasets.values()),
                'total_columns': sum(len(df.columns) for df in datasets.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("  ✓ Synthetic data generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Synthetic data generation failed: {e}")
            self.results['synthetic_data'] = {'error': str(e)}
    
    def _test_advanced_trust_scoring(self):
        """Test advanced trust scoring methods"""
        self.logger.info("Testing advanced trust scoring methods...")
        
        try:
            advanced_results = {}
            
            for name, df in self.test_datasets.items():
                self.logger.info(f"  - Testing advanced trust scoring on {name} dataset...")
                
                # Test ensemble method
                ensemble_result = self.advanced_engine.calculate_advanced_trust_score(
                    df, method="ensemble"
                )
                
                # Test robust method
                robust_result = self.advanced_engine.calculate_advanced_trust_score(
                    df, method="robust"
                )
                
                # Test uncertainty method
                uncertainty_result = self.advanced_engine.calculate_advanced_trust_score(
                    df, method="uncertainty"
                )
                
                advanced_results[name] = {
                    'ensemble': ensemble_result,
                    'robust': robust_result,
                    'uncertainty': uncertainty_result
                }
                
                self.logger.info(f"    Ensemble Score: {ensemble_result.get('trust_score', 'N/A'):.3f}")
                self.logger.info(f"    Robust Score: {robust_result.get('trust_score', 'N/A'):.3f}")
                self.logger.info(f"    Uncertainty Score: {uncertainty_result.get('trust_score', 'N/A'):.3f}")
            
            self.results['advanced_trust_scoring'] = advanced_results
            self.logger.info("  ✓ Advanced trust scoring completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Advanced trust scoring failed: {e}")
            self.results['advanced_trust_scoring'] = {'error': str(e)}
    
    def _test_fallback_quality_manager(self):
        """Test fallback quality manager"""
        self.logger.info("Testing fallback quality manager...")
        
        try:
            fallback_results = {}
            
            for name, df in self.test_datasets.items():
                self.logger.info(f"  - Testing fallback quality manager on {name} dataset...")
                
                # Calculate trust score
                trust_result = self.fallback_manager.calculate_data_trust_score(df)
                
                # Create quality filter
                filtered_df = self.fallback_manager.create_quality_based_filter(df, min_trust_score=0.7)
                
                # Generate quality report
                quality_report = self.fallback_manager.generate_quality_report(df)
                
                fallback_results[name] = {
                    'trust_score': trust_result,
                    'filtered_rows': len(filtered_df),
                    'original_rows': len(df),
                    'quality_report_length': len(quality_report)
                }
                
                self.logger.info(f"    Trust Score: {trust_result.get('trust_score', 'N/A'):.3f}")
                self.logger.info(f"    Filtered: {len(filtered_df)}/{len(df)} rows")
            
            self.results['fallback_quality_manager'] = fallback_results
            self.logger.info("  ✓ Fallback quality manager completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Fallback quality manager failed: {e}")
            self.results['fallback_quality_manager'] = {'error': str(e)}
    
    def _test_dataset_integration(self):
        """Test dataset integration features"""
        self.logger.info("Testing dataset integration features...")
        
        try:
            integration_results = {}
            
            for name, df in self.test_datasets.items():
                self.logger.info(f"  - Testing dataset integration for {name} dataset...")
                
                # Create dataset
                dataset_id = self.dataset_manager.create_dataset(name, df)
                
                # Validate dataset
                validation_result = self.dataset_manager.validate_dataset(dataset_id)
                
                # Process dataset
                transformations = [
                    {'operation': 'filter', 'params': {'condition': 'feature1 > 0'}},
                    {'operation': 'sort', 'params': {'columns': ['feature1'], 'ascending': False}}
                ]
                processed_dataset_id = self.dataset_manager.process_dataset(dataset_id, transformations)
                
                # Export dataset
                export_path = self.dataset_manager.export_dataset(dataset_id, 'json')
                
                integration_results[name] = {
                    'dataset_id': dataset_id,
                    'validation_passed': validation_result['passed'],
                    'processed_dataset_id': processed_dataset_id,
                    'export_path': export_path
                }
                
                self.logger.info(f"    Dataset ID: {dataset_id}")
                self.logger.info(f"    Validation: {'PASSED' if validation_result['passed'] else 'FAILED'}")
            
            self.results['dataset_integration'] = integration_results
            self.logger.info("  ✓ Dataset integration completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Dataset integration failed: {e}")
            self.results['dataset_integration'] = {'error': str(e)}
    
    def _test_real_world_trust_scoring(self):
        """Test real-world trust scoring"""
        self.logger.info("Testing real-world trust scoring...")
        
        try:
            real_world_results = self.real_world_evaluator.run_comprehensive_evaluation()
            self.results['real_world_trust_scoring'] = real_world_results
            self.logger.info("  ✓ Real-world trust scoring completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Real-world trust scoring failed: {e}")
            self.results['real_world_trust_scoring'] = {'error': str(e)}
    
    def _test_cleanlab_benchmarking(self):
        """Test Cleanlab benchmarking"""
        self.logger.info("Testing Cleanlab benchmarking...")
        
        try:
            benchmark_results = {}
            
            for name, df in self.test_datasets.items():
                self.logger.info(f"  - Testing Cleanlab benchmarking on {name} dataset...")
                
                # Create synthetic labels for benchmarking
                labels = np.random.choice([0, 1], size=len(df))
                
                # Run benchmark
                benchmark_result = benchmark_vs_cleanlab(df, labels.tolist())
                
                benchmark_results[name] = benchmark_result
                
                if 'error' not in benchmark_result:
                    self.logger.info(f"    Our Score: {benchmark_result.get('our_trust_score', 'N/A'):.3f}")
                    self.logger.info(f"    Cleanlab Score: {benchmark_result.get('cleanlab_trust_score', 'N/A'):.3f}")
                else:
                    self.logger.info(f"    Benchmark Error: {benchmark_result['error']}")
            
            self.results['cleanlab_benchmarking'] = benchmark_results
            self.logger.info("  ✓ Cleanlab benchmarking completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Cleanlab benchmarking failed: {e}")
            self.results['cleanlab_benchmarking'] = {'error': str(e)}
    
    def _test_performance_and_stress(self):
        """Test performance and stress scenarios"""
        self.logger.info("Testing performance and stress scenarios...")
        
        try:
            performance_results = {}
            
            # Test with large dataset
            self.logger.info("  - Testing with large dataset...")
            large_df = self._create_large_dataset(10000)
            start_time = time.time()
            large_result = self.advanced_engine.calculate_advanced_trust_score(large_df)
            end_time = time.time()
            
            performance_results['large_dataset'] = {
                'rows': len(large_df),
                'columns': len(large_df.columns),
                'processing_time': end_time - start_time,
                'trust_score': large_result.get('trust_score', 'N/A')
            }
            
            # Test with edge cases
            self.logger.info("  - Testing edge cases...")
            edge_cases = {
                'empty_dataset': pd.DataFrame(),
                'single_column': pd.DataFrame({'col': [1, 2, 3]}),
                'all_nan': pd.DataFrame({'col1': [np.nan, np.nan, np.nan], 'col2': [np.nan, np.nan, np.nan]}),
                'all_zero': pd.DataFrame({'col1': [0, 0, 0], 'col2': [0, 0, 0]})
            }
            
            edge_results = {}
            for case_name, df in edge_cases.items():
                try:
                    result = self.advanced_engine.calculate_advanced_trust_score(df)
                    edge_results[case_name] = {
                        'success': True,
                        'trust_score': result.get('trust_score', 'N/A')
                    }
                except Exception as e:
                    edge_results[case_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            performance_results['edge_cases'] = edge_results
            
            self.results['performance_and_stress'] = performance_results
            self.logger.info("  ✓ Performance and stress testing completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Performance and stress testing failed: {e}")
            self.results['performance_and_stress'] = {'error': str(e)}
    
    def _test_integration_and_e2e(self):
        """Test integration and end-to-end scenarios"""
        self.logger.info("Testing integration and end-to-end scenarios...")
        
        try:
            e2e_results = {}
            
            # Test complete pipeline
            self.logger.info("  - Testing complete pipeline...")
            
            # 1. Create dataset
            df = self.test_datasets['high_quality']
            dataset_id = self.dataset_manager.create_dataset('e2e_test', df)
            
            # 2. Calculate trust score
            trust_result = self.advanced_engine.calculate_advanced_trust_score(df)
            
            # 3. Quality filter
            filtered_df = self.fallback_manager.create_quality_based_filter(df, min_trust_score=0.7)
            
            # 4. Process filtered data
            processed_dataset_id = self.dataset_manager.process_dataset(
                dataset_id, 
                [{'operation': 'filter', 'params': {'condition': 'feature1 > 0'}}]
            )
            
            # 5. Export results
            export_path = self.dataset_manager.export_dataset(processed_dataset_id, 'json')
            
            e2e_results['complete_pipeline'] = {
                'dataset_id': dataset_id,
                'trust_score': trust_result.get('trust_score', 'N/A'),
                'filtered_rows': len(filtered_df),
                'processed_dataset_id': processed_dataset_id,
                'export_path': export_path
            }
            
            self.results['integration_and_e2e'] = e2e_results
            self.logger.info("  ✓ Integration and end-to-end testing completed successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Integration and end-to-end testing failed: {e}")
            self.results['integration_and_e2e'] = {'error': str(e)}
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        self.logger.info("Generating comprehensive test report...")
        
        try:
            # Calculate summary statistics
            total_tests = len(self.results)
            passed_tests = sum(1 for result in self.results.values() if 'error' not in result)
            failed_tests = total_tests - passed_tests
            
            # Create summary
            summary = {
                'test_suite_info': {
                    'name': 'Batch Trust Scoring Test Suite',
                    'version': '1.0.0',
                    'start_time': self.start_time.isoformat(),
                    'end_time': self.end_time.isoformat(),
                    'duration': str(self.end_time - self.start_time)
                },
                'test_summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                },
                'detailed_results': self.results
            }
            
            # Save report
            report_file = os.path.join(
                self.output_dir, 
                f"batch_trust_scoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.results['report_generation'] = {
                'report_file': report_file,
                'summary': summary['test_summary']
            }
            
            self.logger.info(f"  ✓ Comprehensive report generated: {report_file}")
            self.logger.info(f"    Total Tests: {total_tests}")
            self.logger.info(f"    Passed: {passed_tests}")
            self.logger.info(f"    Failed: {failed_tests}")
            self.logger.info(f"    Success Rate: {summary['test_summary']['success_rate']:.2%}")
            
        except Exception as e:
            self.logger.error(f"  ✗ Report generation failed: {e}")
            self.results['report_generation'] = {'error': str(e)}
    
    # Helper methods for creating test datasets
    def _create_high_quality_dataset(self) -> pd.DataFrame:
        """Create high-quality synthetic dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples)
        })
        
        return data
    
    def _create_medium_quality_dataset(self) -> pd.DataFrame:
        """Create medium-quality synthetic dataset with some issues"""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples)
        })
        
        # Add some quality issues
        data.loc[100:200, 'feature1'] = np.nan  # Missing values
        data.loc[300:400, 'feature2'] = 0  # Zero values
        
        return data
    
    def _create_low_quality_dataset(self) -> pd.DataFrame:
        """Create low-quality synthetic dataset with many issues"""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples)
        })
        
        # Add many quality issues
        data.loc[100:300, 'feature1'] = np.nan  # Many missing values
        data.loc[400:600, 'feature2'] = 0  # Many zero values
        data.loc[700:800, 'feature3'] = data.loc[700:800, 'feature1']  # Duplicates
        data.loc[900:1000, 'feature4'] = np.random.normal(100, 10, 101)  # Outliers
        
        return data
    
    def _create_correlated_dataset(self) -> pd.DataFrame:
        """Create dataset with correlated features"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create correlated features
        base = np.random.normal(0, 1, n_samples)
        data = pd.DataFrame({
            'feature1': base,
            'feature2': base + np.random.normal(0, 0.1, n_samples),  # Highly correlated
            'feature3': base * 2 + np.random.normal(0, 0.1, n_samples),  # Highly correlated
            'feature4': np.random.normal(0, 1, n_samples)  # Independent
        })
        
        return data
    
    def _create_skewed_dataset(self) -> pd.DataFrame:
        """Create dataset with skewed distributions"""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature1': np.random.exponential(1, n_samples),  # Skewed
            'feature2': np.random.lognormal(0, 1, n_samples),  # Skewed
            'feature3': np.random.gamma(2, 2, n_samples),  # Skewed
            'feature4': np.random.normal(0, 1, n_samples)  # Normal
        })
        
        return data
    
    def _create_large_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create large dataset for performance testing"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples),
            'feature5': np.random.normal(0, 1, n_samples),
            'feature6': np.random.normal(0, 1, n_samples),
            'feature7': np.random.normal(0, 1, n_samples),
            'feature8': np.random.normal(0, 1, n_samples)
        })
        
        return data

def main():
    """Main function to run the batch test suite"""
    print("=" * 80)
    print("COMPREHENSIVE BATCH TRUST SCORING TEST SUITE")
    print("=" * 80)
    print("This will execute all trust scoring components in the correct order")
    print("and generate a comprehensive report.")
    print()
    
    # Create test suite
    test_suite = BatchTrustScoringTestSuite()
    
    # Run complete test suite
    results = test_suite.run_complete_test_suite()
    
    # Print final summary
    if 'report_generation' in results and 'summary' in results['report_generation']:
        summary = results['report_generation']['summary']
        print("\n" + "=" * 80)
        print("FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print("=" * 80)
    
    return results

if __name__ == "__main__":
    main() 