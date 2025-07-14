from .base_provider import BaseLLMProvider

# Import LlamaIndex (install with: pip install llama-index)
try:
    from llama_index import VectorStoreIndex
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

class LlamaIndexRAGProvider(BaseLLMProvider):
    """
    Adapter for LlamaIndex RAG pipelines.
    Supports answering queries using a LlamaIndex vector store.
    """
    def __init__(self, config):
        self.config = config
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex is not installed. Please run 'pip install llama-index'.")
        index_path = config.get('index_path')
        if not index_path:
            raise ValueError("index_path must be provided in config for LlamaIndexRAGProvider.")
        self.index = VectorStoreIndex.load(index_path)

    def generate(self, prompt, **kwargs):
        # For RAG, treat prompt as query
        return self.index.query(prompt)

    def fine_tune(self, dataset_path, **kwargs):
        raise NotImplementedError("Fine-tuning not supported via LlamaIndexRAGProvider.")

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError("Evaluation not supported via LlamaIndexRAGProvider.")

    def deploy(self, **kwargs):
        return "Deployment not implemented for LlamaIndexRAGProvider." 