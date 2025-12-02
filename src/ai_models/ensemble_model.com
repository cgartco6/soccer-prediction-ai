"""
Ensemble model combining multiple predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from ..system.optimizer import SystemOptimizer
from .model_architectures import BaseModel

class EnsembleModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self, optimizer: SystemOptimizer, models: Dict[str, BaseModel], 
                 weights: Dict[str, float] = None, config: Dict = None):
        self.optimizer = optimizer
        self.models = models
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup ensemble strategy
        self.setup_ensemble_strategy()
        
        # Set weights
        if weights is None:
            self.weights = self._calculate_default_weights()
        else:
            self.weights = weights
        
        # Normalize weights
        self._normalize_weights()
        
        # Ensemble metadata
        self.ensemble_metadata = {
            'models_included': list(models.keys()),
            'weights': self.weights,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        self.logger.info(f"Ensemble created with {len(models)} models, weights: {self.weights}")
    
    def setup_ensemble_strategy(self):
        """Setup ensemble strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.ensemble_config = {
                'combination_method': 'weighted_average',  # weighted_average, voting, stacking
                'confidence_weighting': True,
                'fallback_threshold': 0.5,
                'parallel_prediction': False,
                'calibration_enabled': False
            }
        elif profile == 'mid_end':
            self.ensemble_config = {
                'combination_method': 'weighted_average',
                'confidence_weighting': True,
                'fallback_threshold': 0.6,
                'parallel_prediction': True,
                'calibration_enabled': True
            }
        else:  # high_end
            self.ensemble_config = {
                'combination_method': 'stacking',
                'confidence_weighting': True,
                'fallback_threshold': 0.7,
                'parallel_prediction': True,
                'calibration_enabled': True
            }
        
        self.logger.info(f"Ensemble strategy: {self.ensemble_config}")
    
    def _calculate_default_weights(self) -> Dict[str, float]:
        """Calculate default weights based on model performance"""
        # Default equal weights
        weights = {model_name: 1.0 for model_name in self.models.keys()}
        
        # Adjust based on model type and hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            # Prefer faster, lighter models
            weight_adjustments = {
                'xgboost': 1.2,
                'catboost': 1.1,
                'lightgbm': 1.0,
                'gradient_boosting': 0.8,
                'random_forest': 0.7,
                'neural_network': 0.5  # Too heavy for low-end
            }
        elif profile == 'mid_end':
            # Balanced weights
            weight_adjustments = {
                'xgboost': 1.1,
                'catboost': 1.1,
                'lightgbm': 1.0,
                'gradient_boosting': 0.9,
                'random_forest': 0.8,
                'neural_network': 0.9
            }
        else:  # high_end
            # Favor more complex models
            weight_adjustments = {
                'xgboost': 1.0,
                'catboost': 1.0,
                'lightgbm': 1.0,
                'gradient_boosting': 0.9,
                'random_forest': 0.8,
                'neural_network': 1.2  # Can handle complex NN
            }
        
        # Apply adjustments
        for model_name in weights.keys():
            base_name = model_name.split('_')[0] if '_' in model_name else model_name
            adjustment = weight_adjustments.get(base_name, 1.0)
            weights[model_name] *= adjustment
        
        return weights
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make ensemble prediction"""
        if not self.models:
            self.logger.error("No models in ensemble")
            return self._create_fallback_prediction()
        
        # Get predictions from all models
        all_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                prediction = self._get_model_prediction(model, X)
                if prediction:
                    all_predictions[model_name] = prediction
            except Exception as e:
                self.logger.warning(f"Model {model_name} prediction failed: {e}")
        
        if not all_predictions:
            self.logger.error("All model predictions failed")
            return self._create_fallback_prediction()
        
        # Combine predictions
        combined_prediction = self._combine_predictions(all_predictions)
        
        # Add ensemble metadata
        combined_prediction['ensemble_metadata'] = {
            'models_used': list(all_predictions.keys()),
            'weights_used': {k: self.weights.get(k, 0) for k in all_predictions.keys()},
            'combination_method': self.ensemble_config['combination_method']
        }
        
        return combined_prediction
    
    def _get_model_prediction(self, model: BaseModel, X: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from a single model"""
        try:
            # Get probabilities
            proba = model.predict_proba(X)
            
            if proba is None or len(proba) == 0:
                return None
            
            # Get predicted class
            y_pred = model.predict(X)
            
            # Create prediction dictionary
            prediction = {
                'probabilities': proba[0].tolist(),
                'predicted_class': int(y_pred[0]),
                'confidence': float(np.max(proba[0]))
            }
            
            return prediction
            
        except Exception as e:
            self.logger.debug(f"Model prediction failed: {e}")
            return None
    
    def _combine_predictions(self, predictions: Dict[str, Dict]) -> Dict:
        """Combine predictions from multiple models"""
        combination_method = self.ensemble_config['combination_method']
        
        if combination_method == 'weighted_average':
            return self._weighted_average_combination(predictions)
        elif combination_method == 'voting':
            return self._voting_combination(predictions)
        elif combination_method == 'stacking':
            return self._stacking_combination(predictions)
        else:
            # Default to weighted average
            return self._weighted_average_combination(predictions)
    
    def _weighted_average_combination(self, predictions: Dict[str, Dict]) -> Dict:
        """Weighted average combination of probabilities"""
        # Initialize combined probabilities
        num_classes = len(predictions[list(predictions.keys())[0]]['probabilities'])
        combined_probs = np.zeros(num_classes)
        
        # Calculate weighted average
        total_weight = 0
        for model_name, prediction in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            
            # Apply confidence weighting if enabled
            if self.ensemble_config['confidence_weighting']:
                confidence = prediction.get('confidence', 0.5)
                weight *= confidence
            
            probs = np.array(prediction['probabilities'])
            combined_probs += probs * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            combined_probs /= total_weight
        
        # Get predicted class and confidence
        predicted_class = int(np.argmax(combined_probs))
        confidence = float(np.max(combined_probs))
        
        return {
            'probabilities': combined_probs.tolist(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'combination_method': 'weighted_average'
        }
    
    def _voting_combination(self, predictions: Dict[str, Dict]) -> Dict:
        """Majority voting combination"""
        # Collect votes
        votes = []
        confidences = []
        
        for model_name, prediction in predictions.items():
            votes.append(prediction['predicted_class'])
            confidences.append(prediction.get('confidence', 0.5))
        
        # Apply weights to votes
        weighted_votes = {}
        for i, (model_name, vote) in enumerate(zip(predictions.keys(), votes)):
            weight = self.weights.get(model_name, 1.0)
            
            if self.ensemble_config['confidence_weighting']:
                weight *= confidences[i]
            
            if vote not in weighted_votes:
                weighted_votes[vote] = 0
            weighted_votes[vote] += weight
        
        # Get majority vote
        predicted_class = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate average probabilities
        num_classes = len(predictions[list(predictions.keys())[0]]['probabilities'])
        avg_probs = np.zeros(num_classes)
        
        for model_name, prediction in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            if self.ensemble_config['confidence_weighting']:
                weight *= prediction.get('confidence', 0.5)
            
            avg_probs += np.array(prediction['probabilities']) * weight
        
        total_weight = sum(self.weights.get(name, 1.0) for name in predictions.keys())
        if total_weight > 0:
            avg_probs /= total_weight
        
        confidence = float(avg_probs[predicted_class])
        
        return {
            'probabilities': avg_probs.tolist(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'combination_method': 'voting',
            'vote_distribution': weighted_votes
        }
    
    def _stacking_combination(self, predictions: Dict[str, Dict]) -> Dict:
        """Stacking combination using meta-model"""
        # This would require training a meta-model on validation data
        # For now, fall back to weighted average
        
        self.logger.info("Stacking not implemented, using weighted average")
        return self._weighted_average_combination(predictions)
    
    def _create_fallback_prediction(self) -> Dict:
        """Create fallback prediction when ensemble fails"""
        self.logger.warning("Creating ensemble fallback prediction")
        
        # Simple equal probability prediction
        return {
            'probabilities': [0.33, 0.34, 0.33],  # Home, Draw, Away
            'predicted_class': 1,  # Draw
            'confidence': 0.34,
            'combination_method': 'fallback',
            'is_fallback': True
        }
    
    def predict_batch(self, X_batch: pd.DataFrame) -> List[Dict]:
        """Make batch predictions"""
        if X_batch.empty:
            return []
        
        predictions = []
        
        # Use parallel processing if enabled
        if self.ensemble_config['parallel_prediction'] and len(self.models) > 1:
            import concurrent.futures
            
            # Predict with each model in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                future_to_model = {}
                for model_name, model in self.models.items():
                    future = executor.submit(self._predict_batch_with_model, model, X_batch)
                    future_to_model[future] = model_name
                
                # Collect results
                model_predictions = {}
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        model_predictions[model_name] = future.result()
                    except Exception as e:
                        self.logger.warning(f"Batch prediction failed for {model_name}: {e}")
                
                # Combine predictions for each sample
                for i in range(len(X_batch)):
                    sample_predictions = {}
                    for model_name, batch_preds in model_predictions.items():
                        if i < len(batch_preds):
                            sample_predictions[model_name] = batch_preds[i]
                    
                    if sample_predictions:
                        combined = self._combine_predictions(sample_predictions)
                        predictions.append(combined)
                    else:
                        predictions.append(self._create_fallback_prediction())
        else:
            # Sequential processing
            for i in range(len(X_batch)):
                X_sample = X_batch.iloc[[i]]
                prediction = self.predict(X_sample)
                predictions.append(prediction)
        
        return predictions
    
    def _predict_batch_with_model(self, model: BaseModel, X_batch: pd.DataFrame) -> List[Dict]:
        """Make batch predictions with a single model"""
        predictions = []
        
        try:
            # Get probabilities for entire batch
            proba_batch = model.predict_proba(X_batch)
            y_pred_batch = model.predict(X_batch)
            
            for i in range(len(X_batch)):
                prediction = {
                    'probabilities': proba_batch[i].tolist(),
                    'predicted_class': int(y_pred_batch[i]),
                    'confidence': float(np.max(proba_batch[i]))
                }
                predictions.append(prediction)
                
        except Exception as e:
            self.logger.warning(f"Batch prediction failed for model: {e}")
            # Create fallback predictions
            for _ in range(len(X_batch)):
                predictions.append({
                    'probabilities': [0.33, 0.34, 0.33],
                    'predicted_class': 1,
                    'confidence': 0.34
                })
        
        return predictions
    
    def calibrate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate ensemble weights using validation data"""
        if not self.ensemble_config['calibration_enabled']:
            return
        
        self.logger.info("Calibrating ensemble weights...")
        
        # Get predictions from all models
        model_predictions = {}
        model_accuracies = {}
        
        for model_name, model in self.models.items():
            try:
                # Predict on validation data
                y_pred = model.predict(X_val)
                accuracy = np.mean(y_pred == y_val)
                
                model_predictions[model_name] = y_pred
                model_accuracies[model_name] = accuracy
                
                self.logger.info(f"Model {model_name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Calibration failed for {model_name}: {e}")
        
        if not model_accuracies:
            return
        
        # Update weights based on accuracy
        # Higher accuracy -> higher weight
        total_accuracy = sum(model_accuracies.values())
        if total_accuracy > 0:
            new_weights = {}
            for model_name, accuracy in model_accuracies.items():
                new_weights[model_name] = accuracy / total_accuracy
            
            # Smooth transition to new weights
            learning_rate = 0.1  # How quickly to adjust weights
            for model_name in self.weights.keys():
                if model_name in new_weights:
                    self.weights[model_name] = (
                        (1 - learning_rate) * self.weights.get(model_name, 0) +
                        learning_rate * new_weights[model_name]
                    )
            
            # Renormalize
            self._normalize_weights()
            
            self.logger.info(f"Updated weights: {self.weights}")
            self.ensemble_metadata['calibrated_weights'] = self.weights.copy()
            self.ensemble_metadata['calibration_date'] = pd.Timestamp.now().isoformat()
    
    def get_ensemble_report(self) -> Dict:
        """Generate ensemble report"""
        report = {
            'ensemble_config': self.ensemble_config,
            'ensemble_metadata': self.ensemble_metadata,
            'model_count': len(self.models),
            'weights': self.weights
        }
        
        # Add model information
        model_info = {}
        for model_name, model in self.models.items():
            model_info[model_name] = {
                'is_trained': model.is_trained,
                'weight': self.weights.get(model_name, 0)
            }
        
        report['model_info'] = model_info
        
        return report
