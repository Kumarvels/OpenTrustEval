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

## Dynamic Model Management CLI

The script `llm_engineering/scripts/llm_model_manager.py` allows you to add, remove, update, and list models/providers at runtime. You can also persist changes to the config file with `--persist`.

### Example Usage

```bash
# Add a model/provider
python llm_model_manager.py add --name my_llama --type llama_factory --config '{"model_name": "Llama-3-8B", "model_path": "../models/llama-3-8b"}' --persist

# Remove a model/provider
python llm_model_manager.py remove --name my_llama --persist

# Update a model/provider config
python llm_model_manager.py update --name my_llama --config '{"model_name": "Llama-3-8B-v2", "model_path": "../models/llama-3-8b-v2"}' --persist

# List all loaded models/providers
python llm_model_manager.py list
```

This enables dynamic, scriptable management of your LLM providers without editing YAML files by hand. 

## WebUI for Model Management & Tuning

The script `llm_engineering/scripts/llm_webui.py` provides a Gradio-based WebUI for dynamic model management and tuning.

### Features
- List all loaded models/providers
- Add a new model/provider (with config as JSON)
- Remove a model/provider
- Update a model/provider config
- Fine-tune a selected model/provider (with dataset and extra kwargs)
- Evaluate a selected model/provider (with dataset and extra kwargs)

### Launch

```bash
python llm_webui.py
```

This will open a browser UI for interactive model management and LLM tuning. 