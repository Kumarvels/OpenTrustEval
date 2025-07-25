from .base_provider import BaseLLMProvider

# Import LangChain LLMs (install with: pip install langchain openai)
try:
    from langchain.llms import OpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LangChainLLMProvider(BaseLLMProvider):
    """
    Adapter for LangChain LLMs (e.g., OpenAI, Azure, etc.).
    Supports text generation via LangChain's LLM interface.
    """
    def __init__(self, config):
        self.config = config
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Please run 'pip install langchain openai'.")
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.openai_api_key = config.get('openai_api_key')
        self.llm = OpenAI(model_name=self.model_name, openai_api_key=self.openai_api_key)

    def generate(self, prompt, **kwargs):
        return self.llm(prompt)

    def fine_tune(self, dataset_path, **kwargs):
        raise NotImplementedError("Fine-tuning not supported via LangChain LLMProvider.")

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError("Evaluation not supported via LangChain LLMProvider.")

    def deploy(self, **kwargs):
        return "Deployment not implemented for LangChain LLMProvider." 