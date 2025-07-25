import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager

def main():
    """
    Basic script to fine-tune and evaluate phi-4-mini-flash-reasoning using LLaMA-Factory provider.
    For advanced options (LoRA, QLoRA, quantization, distributed, monitors, etc.),
    use the CLI: phi4mini_finetune_eval_cli.py
    """
    manager = LLMLifecycleManager()
    provider = manager.llm_providers.get('phi_4_mini_flash_reasoning')
    if provider is None:
        print("phi_4_mini_flash_reasoning provider not found. Check your config.")
        return
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/small_test_dataset.csv'))
    print("Starting fine-tuning for phi-4-mini-flash-reasoning...")
    try:
        result = provider.fine_tune(dataset_path)
        print("Fine-tuning result:", result)
    except ImportError as e:
        print("LLaMA-Factory not installed:", e)
        return
    print("Starting evaluation for phi-4-mini-flash-reasoning...")
    try:
        result = provider.evaluate(dataset_path)
        print("Evaluation result:", result)
    except ImportError as e:
        print("LLaMA-Factory not installed:", e)
        return

if __name__ == "__main__":
    main()

# For advanced usage, see phi4mini_finetune_eval_cli.py
# Example: python phi4mini_finetune_eval_cli.py --lora --batch_size 4 --epochs 3 --monitor wandb 