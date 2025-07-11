from .base_provider import BaseLLMProvider

# Import LLaMA-Factory APIs (assuming llamafactory is installed)
try:
    from llamafactory import Trainer, Inferencer, Evaluator  # type: ignore
except ImportError:
    Trainer = Inferencer = Evaluator = None

class LLaMAFactoryProvider(BaseLLMProvider):
    """Provider for LLaMA-Factory LLMs. Integrates fine-tuning, inference, and evaluation using LLaMA-Factory APIs.
    All advanced options (e.g., LoRA, QLoRA, batch size, epochs, learning rate, etc.) can be passed as kwargs.
    """
    def __init__(self, config):
        self.config = config
        self.model_name = config.get('model_name', 'Llama-3-8B')
        self.model_path = config.get('model_path', None)
        self.trainer = Trainer if Trainer else None
        self.inferencer = Inferencer if Inferencer else None
        self.evaluator = Evaluator if Evaluator else None

    def generate(self, prompt, **kwargs):
        """Generate text using a LLaMA-Factory model. Passes all kwargs to Inferencer."""
        if not self.inferencer:
            raise ImportError("LLaMA-Factory Inferencer not available. Please install llamafactory.")
        inferencer = self.inferencer(model_name_or_path=self.model_path or self.model_name, **kwargs)
        return inferencer.infer(prompt, **kwargs)

    def fine_tune(self, dataset_path, **kwargs):
        """Fine-tune a LLaMA-Factory model on a dataset. Pass all advanced options as kwargs."""
        if not self.trainer:
            raise ImportError("LLaMA-Factory Trainer not available. Please install llamafactory.")
        trainer = self.trainer(model_name_or_path=self.model_path or self.model_name, train_data=dataset_path, **kwargs)
        trainer.train()
        return "Fine-tuning complete."

    def evaluate(self, eval_data, **kwargs):
        """Evaluate a LLaMA-Factory model. Pass all advanced options as kwargs."""
        if not self.evaluator:
            raise ImportError("LLaMA-Factory Evaluator not available. Please install llamafactory.")
        evaluator = self.evaluator(model_name_or_path=self.model_path or self.model_name, eval_data=eval_data, **kwargs)
        return evaluator.evaluate()

    def deploy(self, **kwargs):
        """Deploy a LLaMA-Factory model (optional)."""
        # Deployment logic can be added as needed
        return "Deployment not implemented." 