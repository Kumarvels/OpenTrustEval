import os
import sys
import argparse
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager


def parse_json_config(config_str):
    """Parse JSON config string with better error handling."""
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config: {e}")
        print("Make sure to use proper JSON format with double quotes.")
        print("Example: '{\"model_name\": \"test\", \"model_path\": \"path\"}'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Dynamic LLM Model Manager: add, remove, update, list models/providers at runtime.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Add model
    add_parser = subparsers.add_parser('add', help='Add a new model/provider')
    add_parser.add_argument('--name', required=True, help='Model/provider name')
    add_parser.add_argument('--type', required=True, help='Provider type (e.g., llama_factory)')
    add_parser.add_argument('--config', required=True, help='Provider config as JSON string')
    add_parser.add_argument('--persist', action='store_true', help='Persist to config file')

    # Remove model
    remove_parser = subparsers.add_parser('remove', help='Remove a model/provider')
    remove_parser.add_argument('--name', required=True, help='Model/provider name')
    remove_parser.add_argument('--persist', action='store_true', help='Remove from config file')

    # Update model
    update_parser = subparsers.add_parser('update', help='Update config for a model/provider')
    update_parser.add_argument('--name', required=True, help='Model/provider name')
    update_parser.add_argument('--config', required=True, help='New provider config as JSON string')
    update_parser.add_argument('--persist', action='store_true', help='Update config file')

    # List models
    list_parser = subparsers.add_parser('list', help='List all loaded models/providers')

    args = parser.parse_args()
    manager = LLMLifecycleManager()

    if args.command == 'add':
        provider_config = parse_json_config(args.config)
        manager.add_model(args.name, args.type, provider_config, persist=args.persist)
        print(f"Added model/provider '{args.name}' of type '{args.type}'.")
    elif args.command == 'remove':
        manager.remove_model(args.name, persist=args.persist)
        print(f"Removed model/provider '{args.name}'.")
    elif args.command == 'update':
        provider_config = parse_json_config(args.config)
        manager.update_model(args.name, provider_config, persist=args.persist)
        print(f"Updated model/provider '{args.name}'.")
    elif args.command == 'list':
        models = manager.list_models()
        print("Loaded models/providers:")
        for m in models:
            print(" -", m)

if __name__ == "__main__":
    main()

# USAGE EXAMPLES:
# Add a model:
# python llm_model_manager.py add --name my_llama --type llama_factory --config '{"model_name": "Llama-3-8B", "model_path": "../models/llama-3-8b"}' --persist
# Remove a model:
# python llm_model_manager.py remove --name my_llama --persist
# Update a model config:
# python llm_model_manager.py update --name my_llama --config '{"model_name": "Llama-3-8B-v2", "model_path": "../models/llama-3-8b-v2"}' --persist
# List models:
# python llm_model_manager.py list 