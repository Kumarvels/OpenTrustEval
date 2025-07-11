class BaseLLMProvider:
    def generate(self, prompt, **kwargs):
        raise NotImplementedError
    def fine_tune(self, dataset_path, **kwargs):
        raise NotImplementedError
    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError
    def deploy(self, **kwargs):
        raise NotImplementedError 