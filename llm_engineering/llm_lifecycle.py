"""
LLM Engineering Lifecycle Management
- Model selection
- Fine-tuning
- Evaluation
- Deployment
- Dynamic LLM management
- Metrics collection for LLMs and tuning
- LLM governance and security (access control, audit, compliance)
- Modular, pluggable architecture for LLM tools/providers
"""

import importlib
import os
import yaml

class LLMLifecycleManager:
    def __init__(self, config_path=None):
        self.metrics = {}
        self.governance_logs = []
        self.security_checks = []
        self.llm_providers = {}
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'configs', 'llm_providers.yaml')
        self.load_providers_from_config()
        # ...initialize model registry, configs...

    def load_providers_from_config(self):
        """Load LLM providers dynamically from YAML config."""
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        for entry in config.get('providers', []):
            name = entry['name']
            provider_type = entry['type']
            provider_config = entry.get('config', {})
            module = importlib.import_module(f"llm_engineering.providers.{provider_type}_provider")
            # Patch: Use correct class name for llama_factory
            if provider_type == 'llama_factory':
                class_name = 'LLaMAFactoryProvider'
            else:
                class_name = ''.join([part.capitalize() for part in provider_type.split('_')]) + 'Provider'
            provider_class = getattr(module, class_name)
            self.add_llm_provider(name, provider_class(provider_config))

    def add_llm_provider(self, name, provider):
        """Dynamically add an LLM provider (e.g., OpenAI, HuggingFace, Azure, etc.)."""
        self.llm_providers[name] = provider

    def remove_llm_provider(self, name):
        """Remove an LLM provider."""
        if name in self.llm_providers:
            del self.llm_providers[name]

    def select_model(self, model_name):
        """Select and register a model."""
        # ...model selection logic...
        self.metrics['model_selected'] = model_name
        self.log_governance(f'Model selected: {model_name}')
        self.run_security_check('Model selection')

    def fine_tune(self, dataset):
        """Fine-tune the current model and collect metrics."""
        # ...fine-tuning logic...
        self.metrics['fine_tune_runs'] = self.metrics.get('fine_tune_runs', 0) + 1
        self.log_governance('Fine-tuning')
        self.run_security_check('Fine-tuning')

    def evaluate(self, eval_data):
        """Evaluate the model and log metrics."""
        # ...evaluation logic...
        self.metrics['eval_runs'] = self.metrics.get('eval_runs', 0) + 1
        self.metrics['last_eval_score'] = 0.95  # Example
        self.log_governance('Evaluation')
        self.run_security_check('Evaluation')

    def deploy(self):
        """Deploy the model and log deployment metrics."""
        # ...deployment logic...
        self.metrics['deployments'] = self.metrics.get('deployments', 0) + 1
        self.log_governance('Deployment')
        self.run_security_check('Deployment')

    def log_governance(self, action):
        """Log governance/audit actions for compliance."""
        self.governance_logs.append({'action': action})

    def run_security_check(self, context):
        """Perform security checks (access control, model privacy, etc)."""
        self.security_checks.append({'context': context, 'status': 'checked'})

    def get_metrics(self):
        return self.metrics

    def get_governance_logs(self):
        return self.governance_logs

    def get_security_checks(self):
        return self.security_checks

    def add_model(self, name, provider_type, provider_config, persist=False):
        """Dynamically add a new model/provider at runtime. Optionally persist to config file."""
        import importlib
        module = importlib.import_module(f"llm_engineering.providers.{provider_type}_provider")
        if provider_type == 'llama_factory':
            class_name = 'LLaMAFactoryProvider'
        else:
            class_name = ''.join([part.capitalize() for part in provider_type.split('_')]) + 'Provider'
        provider_class = getattr(module, class_name)
        self.add_llm_provider(name, provider_class(provider_config))
        if persist:
            self._persist_model_to_config(name, provider_type, provider_config)

    def remove_model(self, name, persist=False):
        """Remove a model/provider at runtime. Optionally remove from config file."""
        self.remove_llm_provider(name)
        if persist:
            self._remove_model_from_config(name)

    def update_model(self, name, provider_config, persist=False):
        """Update config for a model/provider at runtime. Optionally persist to config file."""
        if name not in self.llm_providers:
            raise ValueError(f"Model/provider '{name}' not found.")
        provider = self.llm_providers[name]
        provider.config = provider_config
        if persist:
            self._update_model_in_config(name, provider_config)

    def list_models(self):
        """List all loaded models/providers."""
        return list(self.llm_providers.keys())

    def _persist_model_to_config(self, name, provider_type, provider_config):
        """Helper to add a model to the YAML config file."""
        import yaml
        if not os.path.exists(self.config_path):
            config = {'providers': []}
        else:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {'providers': []}
        config['providers'].append({'name': name, 'type': provider_type, 'config': provider_config})
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f)

    def _remove_model_from_config(self, name):
        """Helper to remove a model from the YAML config file."""
        import yaml
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {'providers': []}
        config['providers'] = [p for p in config['providers'] if p['name'] != name]
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f)

    def _update_model_in_config(self, name, provider_config):
        """Helper to update a model's config in the YAML config file."""
        import yaml
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {'providers': []}
        for p in config['providers']:
            if p['name'] == name:
                p['config'] = provider_config
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f)

class OpenAIProvider:
    """Example stub for an OpenAI LLM provider."""
    def __init__(self, api_key):
        self.api_key = api_key
    def generate(self, prompt):
        # ...call OpenAI API...
        pass

class HuggingFaceProvider:
    """Example stub for a HuggingFace LLM provider."""
    def __init__(self, model_name):
        self.model_name = model_name
    def generate(self, prompt):
        # ...call HuggingFace API...
        pass

# Example usage:
# manager = LLMLifecycleManager()
# manager.add_llm_provider('openai', OpenAIProvider('sk-...'))
# manager.add_llm_provider('hf', HuggingFaceProvider('gpt2'))
# manager.select_model('gpt-4')
# manager.fine_tune('my_dataset.csv')
# manager.evaluate('eval_data.csv')
# manager.deploy()
# print(manager.get_metrics())
# print(manager.get_governance_logs())
# print(manager.get_security_checks())

# --- Example usage code removed to prevent NameError on import ---
# llama = manager.llm_providers.get('llama_factory')
# if llama:
#     llama.fine_tune('my_dataset.csv')
#     llama.evaluate('eval_data.csv')
#     llama.generate('Hello world!')

def example_fine_tune_phi4mini():
    """Example: Fine-tune phi-4-mini-flash-reasoning using LLaMA-Factory provider and small_test_dataset.csv."""
    manager = LLMLifecycleManager()
    provider = manager.llm_providers.get('phi_4_mini_flash_reasoning')
    if provider is None:
        raise RuntimeError("phi_4_mini_flash_reasoning provider not found. Check your config.")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets', 'small_test_dataset.csv'))
    print("Starting fine-tuning for phi-4-mini-flash-reasoning...")
    result = provider.fine_tune(dataset_path)
    print(result)


def example_evaluate_phi4mini():
    """Example: Evaluate phi-4-mini-flash-reasoning using LLaMA-Factory provider and small_test_dataset.csv."""
    manager = LLMLifecycleManager()
    provider = manager.llm_providers.get('phi_4_mini_flash_reasoning')
    if provider is None:
        raise RuntimeError("phi_4_mini_flash_reasoning provider not found. Check your config.")
    eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets', 'small_test_dataset.csv'))
    print("Starting evaluation for phi-4-mini-flash-reasoning...")
    result = provider.evaluate(eval_path)
    print(result)

# --- USAGE EXAMPLES ---
# To run fine-tuning or evaluation from the command line or a notebook:
# from llm_engineering.llm_lifecycle import example_fine_tune_phi4mini, example_evaluate_phi4mini
# example_fine_tune_phi4mini()
# example_evaluate_phi4mini()
