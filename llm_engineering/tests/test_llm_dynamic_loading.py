import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager

def test_dynamic_loading_all_providers():
    manager = LLMLifecycleManager()
    provider_names = [
        'llama_factory',
        'phi_4_mini_flash_reasoning',
        'liquid_ai',
        'smollm3_3b',
    ]
    for name in provider_names:
        provider = manager.llm_providers.get(name)
        assert provider is not None, f"Provider {name} should be loaded dynamically."
        # Test stub methods (will raise ImportError if llamafactory is not installed)
        try:
            provider.generate('Test prompt')
        except ImportError:
            pass  # Acceptable if llamafactory is not installed
        try:
            provider.fine_tune(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/small_test_dataset.csv')))
        except ImportError:
            pass
        try:
            provider.evaluate(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/small_test_dataset.csv')))
        except ImportError:
            pass 

def test_phi4mini_finetune_and_evaluate():
    """Test the example fine-tuning and evaluation for phi-4-mini-flash-reasoning."""
    from llm_engineering.llm_lifecycle import example_fine_tune_phi4mini, example_evaluate_phi4mini
    try:
        example_fine_tune_phi4mini()
        example_evaluate_phi4mini()
    except ImportError:
        # Acceptable if llamafactory is not installed
        pass 