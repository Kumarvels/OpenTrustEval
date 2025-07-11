import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Phi-4-mini-reasoning with LLaMA-Factory.")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--qlora', action='store_true', help='Enable QLoRA fine-tuning')
    parser.add_argument('--dataset', type=str, default=None, help='Path to training/eval dataset (CSV)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation, not fine-tuning')
    args = parser.parse_args()

    manager = LLMLifecycleManager()
    provider = manager.llm_providers.get('phi_4_mini_flash_reasoning')
    if provider is None:
        print("phi_4_mini_flash_reasoning provider not found. Check your config.")
        return
    dataset_path = args.dataset or os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/small_test_dataset.csv'))
    extra_kwargs = {
        'per_device_train_batch_size': args.batch_size,
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
    }
    if args.lora:
        extra_kwargs['use_lora'] = True
    if args.qlora:
        extra_kwargs['use_qlora'] = True

    if not args.eval_only:
        print("Starting fine-tuning for phi-4-mini-flash-reasoning...")
        try:
            result = provider.fine_tune(dataset_path, **extra_kwargs)
            print("Fine-tuning result:", result)
        except ImportError as e:
            print("LLaMA-Factory not installed:", e)
            return
    print("Starting evaluation for phi-4-mini-flash-reasoning...")
    try:
        result = provider.evaluate(dataset_path, **extra_kwargs)
        print("Evaluation result:", result)
    except ImportError as e:
        print("LLaMA-Factory not installed:", e)
        return

if __name__ == "__main__":
    main() 