import unittest
import subprocess
import sys

class TestRunTrustScoringSystem(unittest.TestCase):
    def test_run_quick_test(self):
        result = subprocess.run(
            [sys.executable, "data_engineering/run_trust_scoring_system.py", "--test"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("QUICK TEST COMPLETED SUCCESSFULLY", result.stdout)

    def test_run_batch_test_suite(self):
        result = subprocess.run(
            [sys.executable, "data_engineering/run_trust_scoring_system.py", "--batch"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("BATCH TEST SUITE COMPLETED SUCCESSFULLY", result.stdout)

    def test_run_dashboard(self):
        process = subprocess.Popen(
            [sys.executable, "data_engineering/run_trust_scoring_system.py", "--dashboard", "--port", "8502"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Give the process a moment to start
        try:
            output = process.stdout.read(1000)
            self.assertIn("Dashboard will be available at:", output)
        finally:
            # We need to kill the process to clean up.
            import psutil
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

    def test_run_all(self):
        process = subprocess.Popen(
            [sys.executable, "data_engineering/run_trust_scoring_system.py", "--all", "--port", "8503"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give the process a moment to start
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # The process is running, which is what we expect.
            # We need to kill it to clean up.
            import psutil
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        else:
            # The process terminated, which is not what we expect.
            self.fail("Dashboard process terminated unexpectedly.")

if __name__ == '__main__':
    unittest.main()
