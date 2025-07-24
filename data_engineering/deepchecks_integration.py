"""
Deepchecks Integration for OpenTrustEval
Implements data quality checks using the deepchecks library
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime

try:
    from deepchecks.tabular.suites import data_integrity, train_test_validation
    from deepchecks.tabular import Dataset
    DEEPCHECKS_AVAILABLE = True
except ImportError:
    DEEPCHECKS_AVAILABLE = False
    print("Warning: Deepchecks not available. Install with: pip install deepchecks")

class DeepchecksDataQualityManager:
    """
    Manages data quality assessment using Deepchecks
    """

    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for Deepchecks operations"""
        logger = logging.getLogger('DeepchecksDataQualityManager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def run_data_integrity_suite(self, dataset: pd.DataFrame, target_col: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run the Deepchecks data integrity suite
        """
        if not DEEPCHECKS_AVAILABLE:
            self.logger.warning("Deepchecks is not installed. Cannot run data integrity suite.")
            return None

        try:
            ds = Dataset(dataset, label=target_col)
            suite_result = data_integrity().run(ds)

            return {
                "suite_name": "Data Integrity",
                "results": suite_result.to_json(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error running Deepchecks data integrity suite: {e}")
            return {"error": str(e)}

    def run_train_test_validation_suite(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run the Deepchecks train-test validation suite
        """
        if not DEEPCHECKS_AVAILABLE:
            self.logger.warning("Deepchecks is not installed. Cannot run train-test validation suite.")
            return None

        try:
            train_ds = Dataset(train_dataset, label=target_col)
            test_ds = Dataset(test_dataset, label=target_col)
            suite_result = train_test_validation().run(train_ds, test_ds)

            return {
                "suite_name": "Train-Test Validation",
                "results": suite_result.to_json(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error running Deepchecks train-test validation suite: {e}")
            return {"error": str(e)}

    def generate_report(self, suite_result: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a report from a Deepchecks suite result
        """
        if not DEEPCHECKS_AVAILABLE:
            self.logger.warning("Deepchecks is not installed. Cannot generate report.")
            return None

        try:
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(suite_result['results'])
                return output_path
            else:
                return suite_result['results']
        except Exception as e:
            self.logger.error(f"Error generating Deepchecks report: {e}")
            return None
