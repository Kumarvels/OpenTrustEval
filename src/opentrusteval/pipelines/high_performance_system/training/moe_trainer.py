"""
MoE Training Infrastructure
DeepSeek-style training for domain-specific experts

Features:
- Specialized training for each domain expert
- Cross-domain validation
- Continuous learning
- Performance evaluation
- Model adaptation and fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# Import our components
from src.opentrusteval.pipelines.high_performance_system.core.moe_domain_verifier import MoEDomainVerifier, DomainExpert, ExpertPrediction
from src.opentrusteval.pipelines.high_performance_system.data.domain_datasets import DomainDatasetManager, DomainDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoEDataset(Dataset):
    """PyTorch dataset for MoE training"""
    
    def __init__(self, domain_dataset: DomainDataset, tokenizer=None):
        self.samples = domain_dataset.samples
        self.domain = domain_dataset.domain
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create embeddings (placeholder - would use actual tokenizer)
        text = sample['text']
        embeddings = self._create_embeddings(text)
        
        # Create labels
        labels = {
            'verified': torch.tensor(1.0 if sample['verified'] else 0.0, dtype=torch.float),
            'confidence': torch.tensor(sample['confidence'], dtype=torch.float),
            'quality_score': torch.tensor(sample['quality_score'], dtype=torch.float),
            'hallucination_risk': torch.tensor(sample['hallucination_risk'], dtype=torch.float)
        }
        
        return {
            'text': text,
            'embeddings': embeddings,
            'labels': labels,
            'metadata': sample['metadata']
        }
    
    def _create_embeddings(self, text: str) -> torch.Tensor:
        """Create embeddings for text (placeholder implementation)"""
        # In production, this would use a proper tokenizer and embedding model
        # For now, create random embeddings
        embedding_dim = 768
        return torch.randn(embedding_dim)

class MoETrainer:
    """Trainer for Mixture of Experts system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.moe_verifier = MoEDomainVerifier(self.config)
        self.dataset_manager = DomainDatasetManager()
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'expert_performances': {},
            'domain_performances': {}
        }
        
        # Model checkpoints
        self.checkpoint_dir = Path("high_performance_system/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'embedding_dim': 768,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'validation_split': 0.2,
            'expert_configs': {
                'ecommerce': {'hidden_dim': 512, 'num_classes': 3},
                'banking': {'hidden_dim': 512, 'num_classes': 3},
                'insurance': {'hidden_dim': 512, 'num_classes': 3},
                'healthcare': {'hidden_dim': 512, 'num_classes': 3},
                'legal': {'hidden_dim': 512, 'num_classes': 3},
                'finance': {'hidden_dim': 512, 'num_classes': 3},
                'technology': {'hidden_dim': 512, 'num_classes': 3},
                'general': {'hidden_dim': 512, 'num_classes': 3}
            },
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 5,
            'gradient_clipping': 1.0
        }
    
    async def train_expert(self, domain: str, dataset: DomainDataset) -> Dict[str, Any]:
        """Train a specific domain expert"""
        logger.info(f"Training expert for domain: {domain}")
        
        # Create dataset
        moe_dataset = MoEDataset(dataset)
        
        # Split into train/validation
        train_size = int(0.8 * len(moe_dataset))
        val_size = len(moe_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(moe_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Get expert
        expert = self.moe_verifier.experts[domain]
        expert.to(self.device)
        
        # Setup optimizer
        optimizer = self._create_optimizer(expert)
        scheduler = self._create_scheduler(optimizer)
        criterion = self._create_loss_function()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss = await self._train_epoch(expert, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_metrics = await self._validate_epoch(expert, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save training history
            self.training_history['losses'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy']
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_expert_checkpoint(expert, domain, epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Final evaluation
        final_metrics = await self._evaluate_expert(expert, val_loader)
        
        # Update expert performance history
        if domain not in self.training_history['expert_performances']:
            self.training_history['expert_performances'][domain] = []
        
        self.training_history['expert_performances'][domain].append({
            'timestamp': datetime.now().isoformat(),
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': best_val_loss,
            'epochs_trained': epoch + 1
        })
        
        return {
            'domain': domain,
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'training_history': self.training_history['losses'][-self.config['num_epochs']:]
        }
    
    async def _train_epoch(self, expert: DomainExpert, train_loader: DataLoader, 
                          optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        expert.train()
        total_loss = 0.0
        
        for batch in train_loader:
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels']
            
            # Forward pass
            optimizer.zero_grad()
            classification, confidence = expert(embeddings, batch['metadata'])
            
            # Calculate loss
            loss = self._calculate_loss(classification, confidence, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(expert.parameters(), self.config['gradient_clipping'])
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    async def _validate_epoch(self, expert: DomainExpert, val_loader: DataLoader, 
                             criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        expert.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                classification, confidence = expert(embeddings, batch['metadata'])
                
                # Calculate loss
                loss = self._calculate_loss(classification, confidence, labels)
                total_loss += loss.item()
                
                # Store predictions for metrics
                pred_labels = torch.argmax(classification, dim=1)
                predictions.extend(pred_labels.cpu().numpy())
                true_labels.extend(labels['verified'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return total_loss / len(val_loader), metrics
    
    async def _evaluate_expert(self, expert: DomainExpert, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate expert performance"""
        expert.eval()
        predictions = []
        true_labels = []
        confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                classification, confidence = expert(embeddings, batch['metadata'])
                
                # Store results
                pred_labels = torch.argmax(classification, dim=1)
                predictions.extend(pred_labels.cpu().numpy())
                true_labels.extend(labels['verified'].cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        # Calculate confidence calibration
        confidence_accuracy = self._calculate_confidence_calibration(confidences, predictions, true_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confidence_calibration': confidence_accuracy,
            'avg_confidence': np.mean(confidences)
        }
    
    def _calculate_loss(self, classification: torch.Tensor, confidence: torch.Tensor, 
                       labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate combined loss"""
        # Classification loss
        ce_loss = nn.CrossEntropyLoss()(classification, labels['verified'].long())
        
        # Confidence loss (encourage high confidence for correct predictions)
        confidence_loss = nn.MSELoss()(confidence.squeeze(), labels['confidence'])
        
        # Quality score loss
        quality_loss = nn.MSELoss()(confidence.squeeze(), labels['quality_score'])
        
        # Combined loss
        total_loss = ce_loss + 0.1 * confidence_loss + 0.1 * quality_loss
        
        return total_loss
    
    def _calculate_confidence_calibration(self, confidences: List[float], 
                                        predictions: List[int], 
                                        true_labels: List[int]) -> float:
        """Calculate confidence calibration accuracy"""
        # Group by confidence bins
        confidence_bins = {}
        for conf, pred, true in zip(confidences, predictions, true_labels):
            bin_idx = int(conf * 10)  # 10 bins from 0 to 1
            if bin_idx not in confidence_bins:
                confidence_bins[bin_idx] = {'correct': 0, 'total': 0}
            
            confidence_bins[bin_idx]['total'] += 1
            if pred == true:
                confidence_bins[bin_idx]['correct'] += 1
        
        # Calculate calibration error
        calibration_errors = []
        for bin_idx, bin_data in confidence_bins.items():
            if bin_data['total'] > 0:
                expected_accuracy = bin_idx / 10.0
                actual_accuracy = bin_data['correct'] / bin_data['total']
                calibration_errors.append(abs(expected_accuracy - actual_accuracy))
        
        return 1.0 - np.mean(calibration_errors) if calibration_errors else 0.0
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
        else:
            return optim.SGD(model.parameters(), lr=self.config['learning_rate'])
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'])
        elif self.config['scheduler'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        else:
            return optim.lr_scheduler.StepLR(optimizer, step_size=5)
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        return nn.CrossEntropyLoss()
    
    def _save_expert_checkpoint(self, expert: DomainExpert, domain: str, 
                               epoch: int, val_loss: float):
        """Save expert checkpoint"""
        checkpoint = {
            'domain': domain,
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': expert.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"{domain}_expert_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    async def train_all_experts(self) -> Dict[str, Any]:
        """Train all domain experts"""
        logger.info("Training all domain experts")
        
        results = {}
        
        # Get all available datasets
        datasets = list(self.dataset_manager.data_dir.glob("*_training_*.json"))
        
        for dataset_file in datasets:
            try:
                # Load dataset
                dataset = self.dataset_manager.load_dataset(dataset_file.name)
                domain = dataset.domain
                
                # Skip if not a supported domain
                if domain not in self.moe_verifier.experts:
                    continue
                
                logger.info(f"Training expert for domain: {domain}")
                
                # Train expert
                result = await self.train_expert(domain, dataset)
                results[domain] = result
                
            except Exception as e:
                logger.error(f"Failed to train expert for {dataset_file.name}: {e}")
                results[dataset_file.name] = {'error': str(e)}
        
        # Save training history
        self._save_training_history()
        
        return results
    
    def _save_training_history(self):
        """Save training history"""
        history_path = self.checkpoint_dir / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"Saved training history: {history_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.training_history['losses']:
            logger.warning("No training history available")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        epochs = [entry['epoch'] for entry in self.training_history['losses']]
        train_losses = [entry['train_loss'] for entry in self.training_history['losses']]
        val_losses = [entry['val_loss'] for entry in self.training_history['losses']]
        val_accuracies = [entry['val_accuracy'] for entry in self.training_history['losses']]
        
        # Plot training and validation loss
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot validation accuracy
        axes[0, 1].plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot expert performances
        if self.training_history['expert_performances']:
            domains = list(self.training_history['expert_performances'].keys())
            accuracies = [perf[-1]['final_accuracy'] for perf in self.training_history['expert_performances'].values()]
            
            axes[1, 0].bar(domains, accuracies, color='skyblue')
            axes[1, 0].set_title('Expert Final Accuracies')
            axes[1, 0].set_xlabel('Domain')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)
        
        # Plot loss comparison
        axes[1, 1].scatter(train_losses, val_losses, alpha=0.6, color='purple')
        axes[1, 1].plot([0, max(train_losses)], [0, max(val_losses)], 'r--', alpha=0.5)
        axes[1, 1].set_title('Training vs Validation Loss')
        axes[1, 1].set_xlabel('Training Loss')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training plots: {save_path}")
        
        plt.show()
    
    async def cross_domain_evaluation(self) -> Dict[str, Any]:
        """Evaluate performance across domains"""
        logger.info("Performing cross-domain evaluation")
        
        # Load cross-domain dataset
        cross_domain_file = self.dataset_manager.data_dir / "cross_domain_validation_1.0.0.json"
        
        if not cross_domain_file.exists():
            logger.warning("Cross-domain dataset not found")
            return {}
        
        cross_domain_dataset = self.dataset_manager.load_dataset(cross_domain_file.name)
        moe_dataset = MoEDataset(cross_domain_dataset)
        
        # Create data loader
        test_loader = DataLoader(moe_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Evaluate each expert
        expert_results = {}
        
        for domain, expert in self.moe_verifier.experts.items():
            try:
                metrics = await self._evaluate_expert(expert, test_loader)
                expert_results[domain] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate expert {domain}: {e}")
                expert_results[domain] = {'error': str(e)}
        
        # Overall MoE evaluation
        overall_result = await self._evaluate_moe_system(cross_domain_dataset)
        
        return {
            'expert_results': expert_results,
            'overall_result': overall_result,
            'cross_domain_metrics': self._calculate_cross_domain_metrics(expert_results)
        }
    
    async def _evaluate_moe_system(self, dataset: DomainDataset) -> Dict[str, Any]:
        """Evaluate the complete MoE system"""
        logger.info("Evaluating complete MoE system")
        
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        
        for sample in dataset.samples:
            try:
                # Get MoE prediction
                result = await self.moe_verifier.verify_text(sample['text'])
                
                # Compare with ground truth
                if result.verified == sample['verified']:
                    correct_predictions += 1
                
                total_predictions += 1
                confidence_scores.append(result.confidence)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample: {e}")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        }
    
    def _calculate_cross_domain_metrics(self, expert_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cross-domain performance metrics"""
        valid_results = {k: v for k, v in expert_results.items() if 'error' not in v}
        
        if not valid_results:
            return {}
        
        accuracies = [result['accuracy'] for result in valid_results.values()]
        f1_scores = [result['f1'] for result in valid_results.values()]
        confidences = [result['avg_confidence'] for result in valid_results.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'best_expert': max(valid_results.items(), key=lambda x: x[1]['accuracy'])[0],
            'worst_expert': min(valid_results.items(), key=lambda x: x[1]['accuracy'])[0]
        }

# Example usage
async def main():
    """Example usage of MoE trainer"""
    
    # Initialize trainer
    trainer = MoETrainer()
    
    # Train all experts
    print("Training all domain experts...")
    results = await trainer.train_all_experts()
    
    print("\nTraining Results:")
    for domain, result in results.items():
        if 'error' not in result:
            print(f"{domain}: Accuracy={result['final_accuracy']:.4f}, "
                  f"Loss={result['final_loss']:.4f}, Epochs={result['epochs_trained']}")
        else:
            print(f"{domain}: Error={result['error']}")
    
    # Cross-domain evaluation
    print("\nPerforming cross-domain evaluation...")
    cross_domain_results = await trainer.cross_domain_evaluation()
    
    print("\nCross-Domain Results:")
    if 'overall_result' in cross_domain_results:
        overall = cross_domain_results['overall_result']
        print(f"Overall Accuracy: {overall['accuracy']:.4f}")
        print(f"Average Confidence: {overall['avg_confidence']:.4f}")
    
    if 'cross_domain_metrics' in cross_domain_results:
        metrics = cross_domain_results['cross_domain_metrics']
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        print(f"Best Expert: {metrics['best_expert']}")
        print(f"Worst Expert: {metrics['worst_expert']}")
    
    # Plot training history
    trainer.plot_training_history("high_performance_system/checkpoints/training_plots.png")

if __name__ == "__main__":
    asyncio.run(main()) 