"""
Model evaluation and validation with hardware optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

from ..system.optimizer import SystemOptimizer

class ModelEvaluator:
    """Evaluate and validate models with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup evaluation strategy
        self.setup_evaluation_strategy()
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        self.best_model = None
        
        # Output directory
        self.output_dir = Path("./data/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_evaluation_strategy(self):
        """Setup evaluation strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.evaluation_config = {
                'cross_validation_folds': 3,
                'learning_curve_points': 5,
                'validation_curve_points': 5,
                'feature_importance_top_n': 10,
                'generate_plots': False,  # Save memory
                'save_reports': True,
                'detailed_metrics': False,
                'parallel_evaluation': False
            }
        elif profile == 'mid_end':
            self.evaluation_config = {
                'cross_validation_folds': 5,
                'learning_curve_points': 10,
                'validation_curve_points': 10,
                'feature_importance_top_n': 20,
                'generate_plots': True,
                'save_reports': True,
                'detailed_metrics': True,
                'parallel_evaluation': True
            }
        else:  # high_end
            self.evaluation_config = {
                'cross_validation_folds': 10,
                'learning_curve_points': 20,
                'validation_curve_points': 20,
                'feature_importance_top_n': 30,
                'generate_plots': True,
                'save_reports': True,
                'detailed_metrics': True,
                'parallel_evaluation': True
            }
        
        self.logger.info(f"Model evaluation strategy: {self.evaluation_config}")
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = 'model') -> Dict:
        """Evaluate a single model"""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        # Update best model
        self._update_best_model(model_name, metrics)
        
        # Generate plots if enabled
        if self.evaluation_config['generate_plots']:
            self._generate_evaluation_plots(model, X_test, y_test, y_pred, model_name)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name: str) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC-ROC
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        if self.evaluation_config['detailed_metrics']:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        
        # Additional metrics for soccer prediction
        metrics['soccer_metrics'] = self._calculate_soccer_metrics(y_true, y_pred, y_pred_proba)
        
        return metrics
    
    def _calculate_soccer_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Calculate soccer-specific metrics"""
        metrics = {}
        
        # Convert to home/draw/away
        result_labels = {0: 'H', 1: 'D', 2: 'A'}
        y_true_labels = [result_labels.get(y, 'D') for y in y_true]
        y_pred_labels = [result_labels.get(y, 'D') for y in y_pred]
        
        # Win prediction accuracy
        win_correct = sum(1 for t, p in zip(y_true_labels, y_pred_labels) if t == p)
        metrics['win_accuracy'] = win_correct / len(y_true_labels) if len(y_true_labels) > 0 else 0
        
        # Home win accuracy
        home_true = [t for t in y_true_labels if t == 'H']
        home_pred = [p for p, t in zip(y_pred_labels, y_true_labels) if t == 'H']
        home_correct = sum(1 for t, p in zip(home_true, home_pred) if t == p)
        metrics['home_win_accuracy'] = home_correct / len(home_true) if len(home_true) > 0 else 0
        
        # Draw accuracy
        draw_true = [t for t in y_true_labels if t == 'D']
        draw_pred = [p for p, t in zip(y_pred_labels, y_true_labels) if t == 'D']
        draw_correct = sum(1 for t, p in zip(draw_true, draw_pred) if t == p)
        metrics['draw_accuracy'] = draw_correct / len(draw_true) if len(draw_true) > 0 else 0
        
        # Away win accuracy
        away_true = [t for t in y_true_labels if t == 'A']
        away_pred = [p for p, t in zip(y_pred_labels, y_true_labels) if t == 'A']
        away_correct = sum(1 for t, p in zip(away_true, away_pred) if t == p)
        metrics['away_win_accuracy'] = away_correct / len(away_true) if len(away_true) > 0 else 0
        
        # Profitability simulation (simplified)
        if y_pred_proba is not None:
            metrics['profitability'] = self._simulate_profitability(y_true_labels, y_pred_labels, y_pred_proba)
        
        return metrics
    
    def _simulate_profitability(self, y_true_labels, y_pred_labels, y_pred_proba) -> Dict:
        """Simulate betting profitability"""
        # Simplified profitability simulation
        # In reality, this would use actual odds
        
        # Assume fixed odds for simulation
        odds = {'H': 2.0, 'D': 3.0, 'A': 2.5}
        
        bankroll = 1000
        bet_size = 10
        bets_placed = 0
        bets_won = 0
        profit = 0
        
        for i, (true, pred, proba) in enumerate(zip(y_true_labels, y_pred_labels, y_pred_proba)):
            # Only bet if confidence is high
            confidence = np.max(proba)
            if confidence > 0.65:  # Confidence threshold
                bets_placed += 1
                
                if pred == true:
                    bets_won += 1
                    profit += (odds[pred] * bet_size) - bet_size
                else:
                    profit -= bet_size
        
        if bets_placed > 0:
            roi = (profit / (bets_placed * bet_size)) * 100
            win_rate = bets_won / bets_placed
        else:
            roi = 0
            win_rate = 0
        
        return {
            'bets_placed': bets_placed,
            'bets_won': bets_won,
            'win_rate': win_rate,
            'profit': profit,
            'roi_percent': roi,
            'final_bankroll': bankroll + profit
        }
    
    def _update_best_model(self, model_name: str, metrics: Dict):
        """Update the best model based on evaluation metrics"""
        if metrics.get('f1_score', 0) > 0:
            if self.best_model is None:
                self.best_model = (model_name, metrics)
            else:
                # Compare using F1 score (can be changed to other metrics)
                current_best_f1 = self.best_model[1].get('f1_score', 0)
                new_f1 = metrics.get('f1_score', 0)
                
                if new_f1 > current_best_f1:
                    self.best_model = (model_name, metrics)
    
    def compare_models(self, models: Dict[str, Any], X_test, y_test) -> Dict:
        """Compare multiple models"""
        self.logger.info(f"Comparing {len(models)} models...")
        
        comparison = {}
        
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                comparison[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison[model_name] = None
        
        # Store comparison results
        self.comparison_results = comparison
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison)
        
        # Save comparison results
        if self.evaluation_config['save_reports']:
            self._save_comparison_results(comparison, comparison_report)
        
        return comparison
    
    def _generate_comparison_report(self, comparison: Dict) -> Dict:
        """Generate comparison report"""
        report = {
            'comparison_date': datetime.now().isoformat(),
            'models_compared': list(comparison.keys()),
            'metrics_summary': {},
            'ranking': []
        }
        
        # Calculate summary statistics
        metrics_list = ['accuracy', 'f1_score', 'auc_roc']
        
        for metric in metrics_list:
            values = []
            for model_name, metrics in comparison.items():
                if metrics and metric in metrics:
                    values.append(metrics[metric])
            
            if values:
                report['metrics_summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Rank models by F1 score
        rankings = []
        for model_name, metrics in comparison.items():
            if metrics:
                rankings.append({
                    'model': model_name,
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'auc_roc': metrics.get('auc_roc', 0)
                })
        
        # Sort by F1 score (descending)
        rankings.sort(key=lambda x: x['f1_score'], reverse=True)
        report['ranking'] = rankings
        
        # Identify best model
        if rankings:
            report['best_model'] = rankings[0]['model']
            report['best_f1_score'] = rankings[0]['f1_score']
        
        return report
    
    def cross_validate_model(self, model, X, y, model_name: str = 'model') -> Dict:
        """Perform cross-validation"""
        self.logger.info(f"Cross-validating {model_name}...")
        
        try:
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.evaluation_config['cross_validation_folds'],
                scoring='accuracy',
                n_jobs=self.optimizer.optimization_config.max_parallel_processes
            )
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max(),
                'cv_folds': self.evaluation_config['cross_validation_folds']
            }
            
            # Store in evaluation results
            if model_name in self.evaluation_results:
                self.evaluation_results[model_name]['cross_validation'] = cv_results
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed for {model_name}: {e}")
            return {}
    
    def learning_curve_analysis(self, model, X, y, model_name: str = 'model'):
        """Generate learning curve"""
        if not self.evaluation_config['generate_plots']:
            return {}
        
        self.logger.info(f"Generating learning curve for {model_name}...")
        
        try:
            train_sizes = np.linspace(0.1, 1.0, self.evaluation_config['learning_curve_points'])
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=self.evaluation_config['cross_validation_folds'],
                scoring='accuracy',
                n_jobs=self.optimizer.optimization_config.max_parallel_processes,
                random_state=42
            )
            
            learning_curve_data = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist()
            }
            
            # Generate plot
            self._plot_learning_curve(learning_curve_data, model_name)
            
            return learning_curve_data
            
        except Exception as e:
            self.logger.error(f"Learning curve analysis failed for {model_name}: {e}")
            return {}
    
    def _plot_learning_curve(self, data: Dict, model_name: str):
        """Plot learning curve"""
        try:
            plt.figure(figsize=(10, 6))
            
            train_sizes = data['train_sizes']
            train_mean = data['train_scores_mean']
            train_std = data['train_scores_std']
            val_mean = data['val_scores_mean']
            val_std = data['val_scores_std']
            
            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
            plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
            
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
            
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy')
            plt.title(f'Learning Curve - {model_name}')
            plt.legend(loc='best')
            plt.grid(True)
            
            # Save plot
            plot_file = self.output_dir / f"learning_curve_{model_name}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=100)
            plt.close()
            
            self.logger.info(f"Learning curve saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot learning curve: {e}")
    
    def feature_importance_analysis(self, model, feature_names: List[str], 
                                   model_name: str = 'model') -> Dict:
        """Analyze feature importance"""
        self.logger.info(f"Analyzing feature importance for {model_name}...")
        
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                self.logger.warning(f"Feature importance not available for {model_name}")
                return {}
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Get top N features
            top_n = self.evaluation_config['feature_importance_top_n']
            top_features = sorted_importance[:top_n]
            
            importance_data = {
                'all_features': importance_dict,
                'top_features': dict(top_features),
                'feature_count': len(feature_names)
            }
            
            # Generate plot if enabled
            if self.evaluation_config['generate_plots']:
                self._plot_feature_importance(top_features, model_name)
            
            # Store in evaluation results
            if model_name in self.evaluation_results:
                self.evaluation_results[model_name]['feature_importance'] = importance_data
            
            return importance_data
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed for {model_name}: {e}")
            return {}
    
    def _plot_feature_importance(self, top_features: List[Tuple], model_name: str):
        """Plot feature importance"""
        try:
            features, importance = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            
            plt.barh(y_pos, importance, align='center', alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.grid(True, axis='x', alpha=0.3)
            
            # Save plot
            plot_file = self.output_dir / f"feature_importance_{model_name}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=100)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot feature importance: {e}")
    
    def _generate_evaluation_plots(self, model, X_test, y_test, y_pred, model_name: str):
        """Generate evaluation plots"""
        try:
            # Confusion matrix plot
            self._plot_confusion_matrix(y_test, y_pred, model_name)
            
            # ROC curve if probabilities available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                self._plot_roc_curve(y_test, y_pred_proba, model_name)
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation plots for {model_name}: {e}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_file = self.output_dir / f"confusion_matrix_{model_name}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=100)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot confusion matrix: {e}")
    
    def _plot_roc_curve(self, y_true, y_pred_proba, model_name: str):
        """Plot ROC curve"""
        try:
            from sklearn.metrics import roc_curve, auc
            
            # For multi-class, plot one-vs-rest
            n_classes = y_pred_proba.shape[1]
            
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = self.output_dir / f"roc_curve_{model_name}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=100)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to plot ROC curve: {e}")
    
    def _save_comparison_results(self, comparison: Dict, comparison_report: Dict):
        """Save comparison results to file"""
        try:
            # Save detailed comparison
            comparison_file = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(comparison_file, 'w') as f:
                json.dump({
                    'comparison': comparison,
                    'report': comparison_report,
                    'metadata': {
                        'evaluation_date': datetime.now().isoformat(),
                        'evaluation_config': self.evaluation_config,
                        'hardware_profile': self.optimizer.hardware_info.get('hardware_profile')
                    }
                }, f, indent=2, default=str)
            
            self.logger.info(f"Comparison results saved to {comparison_file}")
            
            # Save summary report
            summary_file = self.output_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d')}.txt"
            summary_text = self._generate_summary_text(comparison_report)
            
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            self.logger.info(f"Evaluation summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison results: {e}")
    
    def _generate_summary_text(self, comparison_report: Dict) -> str:
        """Generate summary text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL EVALUATION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Hardware Profile: {self.optimizer.hardware_info.get('hardware_profile', 'unknown')}")
        lines.append(f"Models Compared: {len(comparison_report.get('models_compared', []))}")
        lines.append("")
        
        # Best model
        best_model = comparison_report.get('best_model')
        if best_model:
            lines.append(f"BEST MODEL: {best_model}")
            lines.append(f"Best F1 Score: {comparison_report.get('best_f1_score', 0):.4f}")
            lines.append("")
        
        # Model rankings
        rankings = comparison_report.get('ranking', [])
        if rankings:
            lines.append("MODEL RANKINGS (by F1 Score):")
            lines.append("-" * 40)
            lines.append(f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'AUC-ROC':<10}")
            lines.append("-" * 40)
            
            for i, rank in enumerate(rankings[:10], 1):  # Top 10 only
                lines.append(
                    f"{i:<5} {rank['model']:<20} "
                    f"{rank['accuracy']:.4f}     "
                    f"{rank['f1_score']:.4f}     "
                    f"{rank['auc_roc']:.4f}"
                )
            lines.append("")
        
        # Metrics summary
        metrics_summary = comparison_report.get('metrics_summary', {})
        if metrics_summary:
            lines.append("METRICS SUMMARY ACROSS ALL MODELS:")
            lines.append("-" * 40)
            
            for metric, stats in metrics_summary.items():
                lines.append(
                    f"{metric.upper():<15} "
                    f"Mean: {stats['mean']:.4f} | "
                    f"Std: {stats['std']:.4f} | "
                    f"Min: {stats['min']:.4f} | "
                    f"Max: {stats['max']:.4f}"
                )
            lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)
        
        if rankings:
            best = rankings[0]
            lines.append(f"1. Use {best['model']} as primary model (F1: {best['f1_score']:.4f})")
            
            if len(rankings) > 1:
                second = rankings[1]
                lines.append(f"2. Consider {second['model']} as backup (F1: {second['f1_score']:.4f})")
            
            # Check for overfitting
            accuracy_diff = best['accuracy'] - best.get('cv_mean', best['accuracy'])
            if abs(accuracy_diff) > 0.1:
                lines.append(f"3. Warning: Potential overfitting detected (difference: {accuracy_diff:.4f})")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_config': self.evaluation_config,
            'evaluation_results': self.evaluation_results,
            'comparison_results': self.comparison_results,
            'best_model': self.best_model,
            'hardware_profile': self.optimizer.hardware_info.get('hardware_profile'),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Add summary statistics
        if self.comparison_results:
            report['summary'] = self._generate_comparison_report(self.comparison_results)
        
        return report
