import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate Phi-4-mini-reasoning with LLaMA-Factory.\n"
        "Supports all advanced options: LoRA, QLoRA, DoRA, GaLore, quantization, distributed, monitors, etc.\n"
        "See https://llamafactory.readthedocs.io/en/latest/Arguments/ for full list."
    )
    # Core training/eval args
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--dataset', type=str, default=None, help='Path to training/eval dataset (CSV)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation, not fine-tuning')
    # Advanced LLaMA-Factory options
    parser.add_argument('--lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--qlora', action='store_true', help='Enable QLoRA fine-tuning')
    parser.add_argument('--galore', action='store_true', help='Enable GaLore fine-tuning')
    parser.add_argument('--dora', action='store_true', help='Enable DoRA fine-tuning')
    parser.add_argument('--lora_plus', action='store_true', help='Enable LoRA+ fine-tuning')
    parser.add_argument('--freeze_tuning', action='store_true', help='Enable freeze-tuning')
    parser.add_argument('--quantization', type=str, default=None, help='Quantization method (e.g., awq, gptq, aqlm, ptq, qat, llm.int8)')
    parser.add_argument('--monitor', type=str, default=None, help='Experiment monitor (wandb, mlflow, tensorboard, llamaboard, swanlab)')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config JSON for distributed training')
    parser.add_argument('--fsdp', action='store_true', help='Enable FSDP distributed training')
    parser.add_argument('--native_ddp', action='store_true', help='Enable NativeDDP distributed training')
    parser.add_argument('--flash_attention', action='store_true', help='Enable FlashAttention')
    parser.add_argument('--unsloth', action='store_true', help='Enable Unsloth acceleration for LoRA/QLoRA')
    parser.add_argument('--adapter', type=str, default=None, help='Adapter type (e.g., lora, dora, galore, etc.)')
    parser.add_argument('--chat_template', type=str, default=None, help='Custom chat template name')
    parser.add_argument('--sequence_packing', action='store_true', help='Enable sequence packing')
    parser.add_argument('--rlhf', action='store_true', help='Enable RLHF training')
    parser.add_argument('--dpo', action='store_true', help='Enable DPO training')
    parser.add_argument('--ppo', action='store_true', help='Enable PPO training')
    parser.add_argument('--reward_modeling', action='store_true', help='Enable reward modeling')
    # Add more as needed
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
    # Map CLI args to LLaMA-Factory kwargs
    if args.lora:
        extra_kwargs['use_lora'] = True
    if args.qlora:
        extra_kwargs['use_qlora'] = True
    if args.galore:
        extra_kwargs['use_galore'] = True
    if args.dora:
        extra_kwargs['use_dora'] = True
    if args.lora_plus:
        extra_kwargs['use_lora_plus'] = True
    if args.freeze_tuning:
        extra_kwargs['use_freeze_tuning'] = True
    if args.quantization:
        extra_kwargs['quantization'] = args.quantization
    if args.monitor:
        extra_kwargs['monitor'] = args.monitor
    if args.deepspeed_config:
        extra_kwargs['deepspeed_config'] = args.deepspeed_config
    if args.fsdp:
        extra_kwargs['use_fsdp'] = True
    if args.native_ddp:
        extra_kwargs['use_native_ddp'] = True
    if args.flash_attention:
        extra_kwargs['use_flash_attention'] = True
    if args.unsloth:
        extra_kwargs['use_unsloth'] = True
    if args.adapter:
        extra_kwargs['adapter'] = args.adapter
    if args.chat_template:
        extra_kwargs['chat_template'] = args.chat_template
    if args.sequence_packing:
        extra_kwargs['use_sequence_packing'] = True
    if args.rlhf:
        extra_kwargs['use_rlhf'] = True
    if args.dpo:
        extra_kwargs['use_dpo'] = True
    if args.ppo:
        extra_kwargs['use_ppo'] = True
    if args.reward_modeling:
        extra_kwargs['use_reward_modeling'] = True

    # Print all options for transparency
    print("LLaMA-Factory fine-tune/eval options:", extra_kwargs)

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

# USAGE EXAMPLES:
# python phi4mini_finetune_eval_cli.py --lora --batch_size 4 --epochs 3 --monitor wandb
# python phi4mini_finetune_eval_cli.py --qlora --quantization awq --deepspeed_config ds_config.json
# python phi4mini_finetune_eval_cli.py --eval_only --monitor llamaboard
# For full argument reference, see: https://llamafactory.readthedocs.io/en/latest/Arguments/ 