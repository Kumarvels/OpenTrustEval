from .base_provider import BaseLLMProvider
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np

# High-Performance System Integration
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
    
    # Initialize high-performance components
    moe_system = UltimateMoESystem()
    expert_ensemble = AdvancedExpertEnsemble()
    
    HIGH_PERFORMANCE_AVAILABLE = True
    print(f"✅ Hugging Face Provider integrated with high-performance system")
except ImportError as e:
    HIGH_PERFORMANCE_AVAILABLE = False
    print(f"⚠️ High-performance system not available for Hugging Face Provider: {e}")

def get_high_performance_hf_status():
    """Get high-performance Hugging Face system status"""
    global moe_system, expert_ensemble
    return {
        'available': HIGH_PERFORMANCE_AVAILABLE,
        'moe_system': 'active' if HIGH_PERFORMANCE_AVAILABLE and 'moe_system' in globals() and moe_system else 'inactive',
        'expert_ensemble': 'active' if HIGH_PERFORMANCE_AVAILABLE and 'expert_ensemble' in globals() and expert_ensemble else 'inactive'
    }

# Try to import Hugging Face libraries
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        PreTrainedTokenizer, PreTrainedModel
    )
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from accelerate import Accelerator
    from huggingface_hub import login, HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ Transformers not available: {e}")

# Try to import FLMR for ColBERT-v2
try:
    from flmr import FLMRConfig, FLMRModelForRetrieval, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer
    FLMR_AVAILABLE = True
except ImportError as e:
    FLMR_AVAILABLE = False
    print(f"⚠️ FLMR not available for ColBERT-v2: {e}")

class HuggingFaceProvider(BaseLLMProvider):
    """
    Comprehensive Hugging Face provider for OpenTrustEval.
    
    Supports:
    - Direct model loading from Hugging Face Hub
    - Text generation (causal LM, seq2seq)
    - Retrieval models (ColBERT-v2, DPR, etc.)
    - Fine-tuning with LoRA, QLoRA, etc.
    - Evaluation and benchmarking
    - Model deployment and serving
    
    Models supported:
    - Text generation: GPT-2, Llama, Mistral, etc.
    - Retrieval: ColBERT-v2, DPR, BGE, etc.
    - Classification: BERT, RoBERTa, etc.
    - Translation: T5, mT5, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'gpt2')
        self.model_path = config.get('model_path', None)
        self.model_type = config.get('model_type', 'auto')  # auto, causal, seq2seq, retrieval, classification
        self.device = config.get('device', 'auto')
        self.use_auth_token = config.get('use_auth_token', None)
        self.trust_remote_code = config.get('trust_remote_code', True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.retrieval_model = None
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model if specified
        if self.model_name:
            self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer based on model type"""
        try:
            if self.model_type == 'retrieval' and 'colbert' in self.model_name.lower():
                self._load_colbert_model()
            elif self.model_type in ['causal', 'generation']:
                self._load_generation_model()
            elif self.model_type == 'seq2seq':
                self._load_seq2seq_model()
            elif self.model_type == 'classification':
                self._load_classification_model()
            else:
                self._load_auto_model()
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_colbert_model(self):
        """Load ColBERT-v2 retrieval model"""
        if not FLMR_AVAILABLE:
            raise ImportError("FLMR not available. Install with: pip install flmr")
        
        try:
            checkpoint_path = self.model_path or self.model_name
            
            # Load tokenizers
            self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                checkpoint_path, 
                subfolder="query_tokenizer",
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                checkpoint_path, 
                subfolder="context_tokenizer",
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            # Load model
            self.retrieval_model = FLMRModelForRetrieval.from_pretrained(
                checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            self.retrieval_model.to(self.device)
            print(f"✅ ColBERT-v2 model loaded: {checkpoint_path}")
            
        except Exception as e:
            logging.error(f"Failed to load ColBERT model: {e}")
            raise
    
    def _load_generation_model(self):
        """Load text generation model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")
        
        try:
            model_path = self.model_path or self.model_name
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.model.to(self.device)
            print(f"✅ Generation model loaded: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load generation model: {e}")
            raise
    
    def _load_seq2seq_model(self):
        """Load sequence-to-sequence model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")
        
        try:
            model_path = self.model_path or self.model_name
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.model.to(self.device)
            print(f"✅ Seq2Seq model loaded: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load seq2seq model: {e}")
            raise
    
    def _load_classification_model(self):
        """Load classification model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")
        
        try:
            model_path = self.model_path or self.model_name
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.model.to(self.device)
            print(f"✅ Classification model loaded: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load classification model: {e}")
            raise
    
    def _load_auto_model(self):
        """Auto-detect and load model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")
        
        try:
            model_path = self.model_path or self.model_name
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                use_auth_token=self.use_auth_token,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            self.model.to(self.device)
            print(f"✅ Auto model loaded: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load auto model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Generation parameters (max_length, temperature, etc.)
        
        Returns:
            Generated text
        """
        if self.model_type == 'retrieval':
            return self._retrieve(prompt, **kwargs)
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Set default generation parameters
            gen_kwargs = {
                'max_length': kwargs.get('max_length', 100),
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.9),
                'do_sample': kwargs.get('do_sample', True),
                'pad_token_id': self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise
    
    def _retrieve(self, query: str, documents: List[str] = None, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Perform retrieval using ColBERT-v2 or other retrieval models.
        
        Args:
            query: Search query
            documents: List of documents to search in (optional)
            top_k: Number of top results to return
            **kwargs: Additional parameters
        
        Returns:
            Retrieval results with scores
        """
        if not self.retrieval_model:
            raise ValueError("Retrieval model not loaded.")
        
        try:
            # Encode query
            Q_encoding = self.query_tokenizer([query])
            
            # Encode documents (if provided)
            if documents:
                D_encoding = self.context_tokenizer(documents)
            else:
                # Use default documents or raise error
                raise ValueError("Documents must be provided for retrieval.")
            
            # Prepare inputs
            inputs = {
                'query_input_ids': Q_encoding['input_ids'],
                'query_attention_mask': Q_encoding['attention_mask'],
                'context_input_ids': D_encoding['input_ids'],
                'context_attention_mask': D_encoding['attention_mask'],
                'use_in_batch_negatives': kwargs.get('use_in_batch_negatives', True),
            }
            
            # Get retrieval scores
            with torch.no_grad():
                results = self.retrieval_model.forward(**inputs)
            
            # Process results
            scores = results.scores.cpu().numpy()
            
            # Get top-k results
            top_indices = np.argsort(scores[0])[-top_k:][::-1]
            
            retrieval_results = {
                'query': query,
                'results': []
            }
            
            for i, idx in enumerate(top_indices):
                retrieval_results['results'].append({
                    'rank': i + 1,
                    'document': documents[idx],
                    'score': float(scores[0][idx])
                })
            
            return retrieval_results
            
        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            raise
    
    def fine_tune(self, dataset_path: str, **kwargs) -> str:
        """
        Fine-tune the model on a dataset.
        
        Args:
            dataset_path: Path to training dataset
            **kwargs: Training parameters
        
        Returns:
            Training result message
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available for fine-tuning.")
        
        try:
            # Load dataset
            if dataset_path.endswith('.csv'):
                dataset = Dataset.from_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                dataset = Dataset.from_json(dataset_path)
            else:
                dataset = load_dataset(dataset_path)
            
            # Prepare training arguments
            training_args = TrainingArguments(
                output_dir=kwargs.get('output_dir', './results'),
                num_train_epochs=kwargs.get('num_train_epochs', 3),
                per_device_train_batch_size=kwargs.get('per_device_train_batch_size', 8),
                per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 8),
                warmup_steps=kwargs.get('warmup_steps', 500),
                weight_decay=kwargs.get('weight_decay', 0.01),
                logging_dir=kwargs.get('logging_dir', './logs'),
                logging_steps=kwargs.get('logging_steps', 10),
                save_steps=kwargs.get('save_steps', 1000),
                eval_steps=kwargs.get('eval_steps', 1000),
                evaluation_strategy=kwargs.get('evaluation_strategy', 'steps'),
                load_best_model_at_end=kwargs.get('load_best_model_at_end', True),
                metric_for_best_model=kwargs.get('metric_for_best_model', 'eval_loss'),
                greater_is_better=kwargs.get('greater_is_better', False),
                **kwargs
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            trainer.train()
            
            # Save model
            output_path = kwargs.get('output_path', f'./fine_tuned_{self.model_name}')
            trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            return f"Fine-tuning completed. Model saved to {output_path}"
            
        except Exception as e:
            logging.error(f"Fine-tuning failed: {e}")
            raise
    
    def evaluate(self, eval_data: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on evaluation data.
        
        Args:
            eval_data: Path to evaluation dataset
            **kwargs: Evaluation parameters
        
        Returns:
            Evaluation results
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available for evaluation.")
        
        try:
            # Load evaluation dataset
            if eval_data.endswith('.csv'):
                dataset = Dataset.from_csv(eval_data)
            elif eval_data.endswith('.json'):
                dataset = Dataset.from_json(eval_data)
            else:
                dataset = load_dataset(eval_data)
            
            # Prepare evaluation arguments
            eval_args = TrainingArguments(
                output_dir=kwargs.get('output_dir', './eval_results'),
                per_device_eval_batch_size=kwargs.get('per_device_eval_batch_size', 8),
                **kwargs
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Initialize trainer for evaluation
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Run evaluation
            results = trainer.evaluate()
            
            return {
                'model_name': self.model_name,
                'evaluation_results': results,
                'timestamp': str(torch.datetime.now())
            }
            
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
    
    def deploy(self, **kwargs) -> str:
        """
        Deploy the model for serving.
        
        Args:
            **kwargs: Deployment parameters
        
        Returns:
            Deployment status
        """
        try:
            # Create pipeline for serving
            if self.model_type == 'retrieval':
                self.pipeline = pipeline(
                    "feature-extraction",
                    model=self.retrieval_model,
                    tokenizer=self.context_tokenizer,
                    device=self.device
                )
            else:
                task = kwargs.get('task', 'text-generation')
                self.pipeline = pipeline(
                    task,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device
                )
            
            return f"Model deployed successfully. Pipeline created for {self.model_name}"
            
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'loaded': bool(self.model or self.retrieval_model),
            'tokenizer_loaded': bool(self.tokenizer),
            'pipeline_loaded': bool(self.pipeline)
        }
        
        if self.model:
            info['model_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['model_trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return info
    
    def list_available_models(self, search_term: str = None) -> List[str]:
        """
        List available models from Hugging Face Hub.
        
        Args:
            search_term: Optional search term to filter models
        
        Returns:
            List of available model names
        """
        if not TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            api = HfApi()
            models = api.list_models(search=search_term, limit=50)
            return [model.modelId for model in models]
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []

# Example usage functions
def example_colbert_retrieval():
    """Example: Use ColBERT-v2 for document retrieval"""
    config = {
        'model_name': 'LinWeizheDragon/ColBERT-v2',
        'model_type': 'retrieval',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    provider = HuggingFaceProvider(config)
    
    # Example documents
    documents = [
        "Paris is the capital of France.",
        "Beijing is the capital of China.",
        "London is the capital of England.",
        "Tokyo is the capital of Japan.",
        "Berlin is the capital of Germany."
    ]
    
    # Perform retrieval
    query = "What is the capital of France?"
    results = provider._retrieve(query, documents, top_k=3)
    
    print("Retrieval Results:")
    for result in results['results']:
        print(f"Rank {result['rank']}: {result['document']} (Score: {result['score']:.4f})")
    
    return results

def example_text_generation():
    """Example: Use a text generation model"""
    config = {
        'model_name': 'gpt2',
        'model_type': 'causal',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    provider = HuggingFaceProvider(config)
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    generated_text = provider.generate(prompt, max_length=50, temperature=0.8)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    return generated_text

def example_fine_tuning():
    """Example: Fine-tune a model"""
    config = {
        'model_name': 'gpt2',
        'model_type': 'causal',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    provider = HuggingFaceProvider(config)
    
    # Fine-tune (example with small dataset)
    dataset_path = "path/to/your/dataset.csv"
    result = provider.fine_tune(dataset_path, num_train_epochs=1)
    
    print(f"Fine-tuning result: {result}")
    return result 