from .base_provider import BaseLLMProvider
# High-Performance System Integration
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
    
    # Initialize high-performance components
    moe_system = UltimateMoESystem()
    expert_ensemble = AdvancedExpertEnsemble()
    
    HIGH_PERFORMANCE_AVAILABLE = True
    print(f"✅ LLaMA Factory Provider integrated with high-performance system")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available for LLaMA Factory Provider: {e}")

def get_high_performance_llm_status():
    """Get high-performance LLM system status"""
    global moe_system, expert_ensemble
    return {
        'available': HIGH_PERFORMANCE_AVAILABLE,
        'moe_system': 'active' if HIGH_PERFORMANCE_AVAILABLE and 'moe_system' in globals() and moe_system else 'inactive',
        'expert_ensemble': 'active' if HIGH_PERFORMANCE_AVAILABLE and 'expert_ensemble' in globals() and expert_ensemble else 'inactive'
    }


# Import LLaMA-Factory APIs (assuming llamafactory is installed)
try:
    from llamafactory import Trainer, Inferencer, Evaluator  # type: ignore
except ImportError:
    Trainer = Inferencer = Evaluator = None

class LLaMAFactoryProvider(BaseLLMProvider):
    """
    Provider for LLaMA-Factory LLMs. Integrates fine-tuning, inference, and evaluation using LLaMA-Factory APIs.
    Exposes all advanced options, including:
      - LoRA, QLoRA, DoRA, GaLore, LoRA+, freeze-tuning
      - Quantization (AQLM, AWQ, GPTQ, LLM.int8, PTQ, QAT)
      - Mixed precision, activation checkpointing, flash attention, Unsloth
      - Distributed training (DeepSpeed, FSDP, NativeDDP)
      - Monitors (LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab)
      - Adapters, custom chat templates, sequence packing
      - RLHF, DPO, PPO, reward modeling
      - All Trainer/Evaluator/Inferencer kwargs
    See: https://llamafactory.readthedocs.io/en/latest/Arguments/
    """
    def __init__(self, config):
        self.config = config
        self.model_name = config.get('model_name', 'Llama-3-8B')
        self.model_path = config.get('model_path', None)
        self.trainer = Trainer if Trainer else None
        self.inferencer = Inferencer if Inferencer else None
        self.evaluator = Evaluator if Evaluator else None

    def generate(self, prompt, **kwargs):
        """
        Generate text using a LLaMA-Factory model.
        All advanced options (e.g., quantization, adapters, monitors, etc.) can be passed as kwargs.
        """
        if not self.inferencer:
            raise ImportError("LLaMA-Factory Inferencer not available. Please install llamafactory.")
        inferencer = self.inferencer(model_name_or_path=self.model_path or self.model_name, **kwargs)
        return inferencer.infer(prompt, **kwargs)

    def fine_tune(self, dataset_path, **kwargs):
        """
        Fine-tune a LLaMA-Factory model on a dataset.
        All advanced options (e.g., LoRA, QLoRA, DoRA, GaLore, quantization, distributed, monitors, etc.) can be passed as kwargs.
        Example kwargs:
            use_lora=True, use_qlora=True, use_galore=True, use_dora=True,
            quantization="awq", monitor="wandb", deepspeed_config="ds_config.json",
            per_device_train_batch_size=4, num_train_epochs=3, learning_rate=2e-5, ...
        """
        if not self.trainer:
            raise ImportError("LLaMA-Factory Trainer not available. Please install llamafactory.")
        trainer = self.trainer(model_name_or_path=self.model_path or self.model_name, train_data=dataset_path, **kwargs)
        # Optionally attach callbacks/monitors if provided
        if 'callbacks' in kwargs:
            for cb in kwargs['callbacks']:
                trainer.add_callback(cb)
        if 'monitor' in kwargs:
            trainer.set_monitor(kwargs['monitor'])
        trainer.train()
        return "Fine-tuning complete."

    def evaluate(self, eval_data, **kwargs):
        """
        Evaluate a LLaMA-Factory model.
        All advanced options (e.g., quantization, adapters, monitors, distributed, etc.) can be passed as kwargs.
        """
        if not self.evaluator:
            raise ImportError("LLaMA-Factory Evaluator not available. Please install llamafactory.")
        evaluator = self.evaluator(model_name_or_path=self.model_path or self.model_name, eval_data=eval_data, **kwargs)
        if 'monitor' in kwargs:
            evaluator.set_monitor(kwargs['monitor'])
        return evaluator.evaluate()

    def deploy(self, **kwargs):
        """
        Deploy a LLaMA-Factory model (optional).
        Deployment logic can be added as needed (e.g., vLLM, OpenAI-style API, etc.)
        """
        return "Deployment not implemented." 