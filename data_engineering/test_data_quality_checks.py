"""
Unit tests for data quality checks
"""

import unittest
import pandas as pd
import numpy as np

from data_engineering.cleanlab_integration import FallbackDataQualityManager
from data_engineering.deepchecks_integration import DeepchecksDataQualityManager

class TestFallbackDataQualityManager(unittest.TestCase):
    """
    Tests for the FallbackDataQualityManager
    """

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.manager = FallbackDataQualityManager()
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 1, 6, 7, 8, 9],
            'feature2': [1, 2, 3, 4, 5, 1, 6, 7, 8, 100],
            'feature3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

    def test_detect_outliers_ratio_iqr(self):
        """Test outlier detection using the IQR method"""
        outlier_ratio = self.manager._detect_outliers_ratio(self.data, method='iqr')
        self.assertAlmostEqual(outlier_ratio, 0.1)

    def test_detect_outliers_ratio_isolation_forest(self):
        """Test outlier detection using the Isolation Forest method"""
        outlier_ratio = self.manager._detect_outliers_ratio(self.data, method='isolation_forest')
        self.assertGreaterEqual(outlier_ratio, 0.0)
        self.assertLessEqual(outlier_ratio, 1.0)

    def test_detect_exact_duplicates(self):
        """Test exact duplicate detection"""
        duplicate_ratio = self.manager._detect_exact_duplicates(self.data)
        self.assertAlmostEqual(duplicate_ratio, 0.1)

    def test_detect_approximate_duplicates(self):
        """Test approximate duplicate detection"""
        duplicate_ratio = self.manager._detect_approximate_duplicates(self.data)
        self.assertAlmostEqual(duplicate_ratio, 0.0)


try:
    from data_engineering.deepchecks_integration import DEEPCHECKS_AVAILABLE
except ImportError:
    DEEPCHECKS_AVAILABLE = False

@unittest.skipIf(not DEEPCHECKS_AVAILABLE, "Deepchecks not available")
class TestDeepchecksDataQualityManager(unittest.TestCase):
    """
    Tests for the DeepchecksDataQualityManager
    """

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.manager = DeepchecksDataQualityManager()
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        self.train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 0, 0, 1, 1]
        })
        self.test_data = pd.DataFrame({
            'feature1': [6, 7, 8, 9, 10],
            'target': [0, 0, 1, 1, 1]
        })

    def test_run_data_integrity_suite(self):
        """Test the data integrity suite"""
        if self.manager:
            result = self.manager.run_data_integrity_suite(self.data, target_col='target')
            self.assertIsNotNone(result)
            self.assertIn('suite_name', result)
            self.assertEqual(result['suite_name'], 'Data Integrity')

    def test_run_train_test_validation_suite(self):
        """Test the train-test validation suite"""
        if self.manager:
            result = self.manager.run_train_test_validation_suite(self.train_data, self.test_data, target_col='target')
            self.assertIsNotNone(result)
            self.assertIn('suite_name', result)
            self.assertEqual(result['suite_name'], 'Train-Test Validation')

from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine

class TestAdvancedTrustScoringEngine(unittest.TestCase):
    """
    Tests for the AdvancedTrustScoringEngine
    """

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.engine = AdvancedTrustScoringEngine()
        self.data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.labels = np.random.randint(0, 2, 100)

    def test_find_label_issues(self):
        """Test the find_label_issues method"""
        # Introduce a label error
        self.labels[50] = 1 - self.labels[50]

        label_issues = self.engine.find_label_issues(self.data, self.labels)
        self.assertIsNotNone(label_issues)
        self.assertIsInstance(label_issues, pd.DataFrame)

        if not label_issues.empty:
            self.assertIn('index', label_issues.columns)
            self.assertIn('true_label', label_issues.columns)
            self.assertIn('predicted_label', label_issues.columns)
            self.assertIn('confidence', label_issues.columns)

if __name__ == '__main__':
    unittest.main()
