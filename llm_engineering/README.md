## Advanced LLaMA-Factory CLI Usage

The script `llm_engineering/scripts/phi4mini_finetune_eval_cli.py` supports all advanced LLaMA-Factory options for fine-tuning and evaluation, including:
- LoRA, QLoRA, DoRA, GaLore, LoRA+, freeze-tuning
- Quantization (AQLM, AWQ, GPTQ, LLM.int8, PTQ, QAT)
- Mixed precision, activation checkpointing, flash attention, Unsloth
- Distributed training (DeepSpeed, FSDP, NativeDDP)
- Monitors (LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab)
- Adapters, custom chat templates, sequence packing
- RLHF, DPO, PPO, reward modeling
- All Trainer/Evaluator/Inferencer kwargs

### Example Usage

```bash
# Basic LoRA fine-tuning
python phi4mini_finetune_eval_cli.py --lora --batch_size 4 --epochs 3 --monitor wandb

# QLoRA with quantization and DeepSpeed
python phi4mini_finetune_eval_cli.py --qlora --quantization awq --deepspeed_config ds_config.json

# Evaluation only with experiment monitor
python phi4mini_finetune_eval_cli.py --eval_only --monitor llamaboard
```

For a full list of supported arguments and their effects, see the [LLaMA-Factory Arguments documentation](https://llamafactory.readthedocs.io/en/latest/Arguments/). 