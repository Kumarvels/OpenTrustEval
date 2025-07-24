# Workflows Directory

This directory contains reusable, domain-specific workflow modules for multi-agent orchestration, LLM pipelines, and RAG tasks.

## Purpose
- Organize complex and advanced workflows (e.g., ecommerce support, finance, healthcare, etc.)
- Enable easy extension and maintenance as workflow complexity grows
- Support integration with test scripts, notebooks, and the WebUI

## How to Add a New Workflow
1. Create a new Python file (e.g., `finance_qa.py`).
2. Define input preparation and run functions for your workflow:

```python
def get_<workflow>_workflow_input(...):
    # Prepare input dict for the workflow
    ...

def run_<workflow>_workflow(orchestrator, ...):
    # Run the workflow using the orchestrator
    ...
```

3. Import and use your workflow in test scripts, notebooks, or the WebUI.

## Example: Ecommerce Support
See `ecommerce_support.py` for order status, returns, and escalation workflows.

## Integration
- **Test scripts:** Import and call workflow functions for automated testing.
- **WebUI:** Expose workflows as selectable options for interactive execution.
- **Notebooks:** Demonstrate workflows for prototyping and documentation.

## Best Practices
- Keep each workflow focused and modular.
- Use clear function names and docstrings.
- Add templates for new workflows to encourage consistency. 