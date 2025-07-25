from .base_provider import BaseLLMProvider

# Import LangGraph (install with: pip install langgraph)
try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

class LangGraphOrchestrator(BaseLLMProvider):
    """
    Adapter for LangGraph-based multi-agent workflow orchestration.
    Allows running graph-based, stateful agent workflows.
    """
    def __init__(self, config):
        self.config = config
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is not installed. Please run 'pip install langgraph'.")
        graph_config = config.get('graph_config')
        if not graph_config:
            raise ValueError("graph_config must be provided in config for LangGraphOrchestrator.")
        self.graph = StateGraph(graph_config)

    def run_workflow(self, input_data, **kwargs):
        return self.graph.run(input_data)

    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Direct text generation not supported via LangGraphOrchestrator.")

    def fine_tune(self, dataset_path, **kwargs):
        raise NotImplementedError("Fine-tuning not supported via LangGraphOrchestrator.")

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError("Evaluation not supported via LangGraphOrchestrator.")

    def deploy(self, **kwargs):
        return "Deployment not implemented for LangGraphOrchestrator." 