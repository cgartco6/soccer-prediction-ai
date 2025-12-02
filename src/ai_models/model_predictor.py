"""
Model prediction with hardware optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path

from ..system.optimizer import SystemOptimizer
from .model_architectures import BaseModel
from .ensemble_model import EnsembleModel

class ModelPredictor:
    """Make predictions using trained models with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup prediction strategy
        self.setup_prediction_strategy()
        
        # Model storage
        self.models = {}
        self.ensemble_model = None
        self.scalers = {}
        self.encoders = {}
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_dir = Path("./data/cache/predictions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.prediction_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'model_calls': {},
            'average_time': 0
        }
    
    def setup_prediction_strategy(self):
        """Setup prediction strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.prediction_config = {
                'use_ensemble': False,
                'max_models': 2,
                'batch_size': 16,
                'use_cache': True,
                'cache_ttl': 3600,  # 1 hour
                'confidence_threshold': 0.65,
                'parallel_prediction': False,
                'fallback_model': 'xgboost'
            }
        elif profile == 'mid_end':
            self.prediction_config = {
                'use_ensemble': True,
                'max_models': 4,
                'batch_size': 32,
                'use_cache': True,
                'cache_ttl': 1800,  # 30 minutes
                'confidence_threshold': 0.65,
                'parallel_prediction': True,
                'fallback_model': 'ensemble'
            }
        else:  # high_end
            self.prediction_config = {
                'use_ensemble': True,
                'max_models': 6,
                'batch_size': 64,
                'use_cache': True,
                'cache_ttl': 900,  # 15 minutes
                'confidence_threshold': 0.65,
                'parallel_prediction': True,
                'fallback_model': 'ensemble'
            }
        
        self.logger.info(f"Prediction strategy: {self.prediction_config}")
    
    def load_models(self, model_dir: str):
        """Load trained models"""
        from .model_trainer import ModelTrainer
        
        trainer = ModelTrainer(self.optimizer)
        success = trainer.load_models(model_dir)
        
        if success:
            self.models = trainer.models
            self.scalers = trainer.scalers
            self.encoders = trainer.encoders
            
            # Create ensemble if enabled
            if self.prediction_config['use_ensemble'] and len(self.models) > 1:
                self._create_ensemble()
            
            self.logger.info(f"Loaded {len(self.models)} models from {model_dir}")
            return True
        
        return False
    
    def _create_ensemble(self):
        """Create ensemble model"""
        # Get ensemble weights based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            weights = {'xgboost': 0.6, 'catboost': 0.4}
        elif profile == 'mid_end':
            weights = {'xgboost': 0.3, 'catboost': 0.3, 'lightgbm': 0.2, 'gradient_boosting': 0.2}
        else:  # high_end
            weights = {'xgboost': 0.25, 'catboost': 0.25, 'lightgbm': 0.2, 
                      'neural_network': 0.15, 'gradient_boosting': 0.1, 'random_forest': 0.05}
        
        # Filter to available models
        available_weights = {k: v for k, v in weights.items() if k in self.models}
        
        # Normalize weights
        total = sum(available_weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in available_weights.items()}
        else:
            normalized_weights = {k: 1/len(available_weights) for k in available_weights.keys()}
        
        self.ensemble_model = EnsembleModel(
            self.optimizer,
            models={k: self.models[k] for k in normalized_weights.keys()},
            weights=normalized_weights
        )
        
        self.logger.info(f"Ensemble created with weights: {normalized_weights}")
    
    def predict(self, features: Dict, use_cache: bool = None) -> Dict:
        """Make prediction for a single match"""
        import time
        start_time = time.time()
        
        if use_cache is None:
            use_cache = self.prediction_config['use_cache']
        
        # Generate cache key
        cache_key = self._get_prediction_cache_key(features)
        
        # Check cache
        if use_cache:
            cached_prediction = self._load_from_cache(cache_key)
            if cached_prediction is not None:
                self.prediction_stats['cache_hits'] += 1
                self.prediction_stats['total_predictions'] += 1
                return cached_prediction
        
        # Prepare features
        X = self._prepare_features(features)
        
        if X is None:
            self.logger.error("Failed to prepare features for prediction")
            return self._create_fallback_prediction(features)
        
        # Make predictions
        predictions = {}
        
        # Use ensemble if available
        if self.prediction_config['use_ensemble'] and self.ensemble_model is not None:
            ensemble_pred = self.ensemble_model.predict(X)
            predictions['ensemble'] = ensemble_pred
        
        # Make individual model predictions
        model_count = 0
        for model_name, model in self.models.items():
            if model_count >= self.prediction_config['max_models']:
                break
            
            try:
                pred = self._predict_with_model(model, X)
                if pred:
                    predictions[model_name] = pred
                    model_count += 1
                    
                    # Update stats
                    if model_name not in self.prediction_stats['model_calls']:
                        self.prediction_stats['model_calls'][model_name] = 0
                    self.prediction_stats['model_calls'][model_name] += 1
                    
            except Exception as e:
                self.logger.warning(f"Prediction failed for {model_name}: {e}")
        
        # Combine predictions
        final_prediction = self._combine_predictions(predictions, features)
        
        # Add metadata
        final_prediction['prediction_metadata'] = {
            'models_used': list(predictions.keys()),
            'prediction_time': time.time() - start_time,
            'cache_key': cache_key,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache prediction
        if use_cache:
            self._save_to_cache(cache_key, final_prediction)
        
        # Update stats
        self.prediction_stats['total_predictions'] += 1
        self.prediction_stats['average_time'] = (
            (self.prediction_stats['average_time'] * (self.prediction_stats['total_predictions'] - 1) +
             (time.time() - start_time)) / self.prediction_stats['total_predictions']
        )
        
        return final_prediction
    
    def _get_prediction_cache_key(self, features: Dict) -> str:
        """Generate cache key for prediction"""
        import hashlib
        
        # Extract key features for cache
        cache_features = {
            'home_team': features.get('home_team', ''),
            'away_team': features.get('away_team', ''),
            'date': features.get('date', ''),
            'league': features.get('league', ''),
            'feature_hash': self._get_features_hash(features)
        }
        
        cache_str = json.dumps(cache_features, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_features_hash(self, features: Dict) -> str:
        """Create hash of feature values"""
        import hashlib
        
        # Extract numerical features for hashing
        numerical_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numerical_features[key] = value
        
        feature_str = json.dumps(numerical_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load prediction from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age > self.prediction_config['cache_ttl']:
            try:
                cache_file.unlink()
            except:
                pass
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_key: str, prediction: Dict):
        """Save prediction to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(prediction, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save prediction to cache: {e}")
    
    def _prepare_features(self, features: Dict) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Apply scaling if available
            if 'standard' in self.scalers:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])
            
            # Handle missing values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _predict_with_model(self, model: BaseModel, X: pd.DataFrame) -> Optional[Dict]:
        """Make prediction with a single model"""
        try:
            # Get probabilities
            proba = model.predict_proba(X)
            
            if proba is None:
                return None
            
            # Get class labels
            if 'target' in self.encoders:
                classes = self.encoders['target'].classes_
            else:
                classes = [0, 1, 2]  # Home, Draw, Away
            
            # Create prediction dictionary
            prediction = {
                'probabilities': {str(cls): float(prob) for cls, prob in zip(classes, proba[0])},
                'predicted_class': int(model.predict(X)[0]),
                'confidence': float(np.max(proba[0]))
            }
            
            # Add predicted label
            if 'target' in self.encoders:
                prediction['predicted_label'] = self.encoders['target'].inverse_transform(
                    [prediction['predicted_class']]
                )[0]
            
            return prediction
            
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {e}")
            return None
    
    def _combine_predictions(self, predictions: Dict, original_features: Dict) -> Dict:
        """Combine predictions from multiple models"""
        if not predictions:
            return self._create_fallback_prediction(original_features)
        
        # Use ensemble prediction if available
        if 'ensemble' in predictions:
            main_prediction = predictions['ensemble']
        else:
            # Use the model with highest average confidence
            main_prediction = max(predictions.values(), key=lambda x: x.get('confidence', 0))
        
        # Calculate consensus
        consensus = self._calculate_consensus(predictions)
        
        # Create final prediction
        final_prediction = {
            'match_info': {
                'home_team': original_features.get('home_team', ''),
                'away_team': original_features.get('away_team', ''),
                'league': original_features.get('league', ''),
                'date': original_features.get('date', '')
            },
            'prediction': {
                'winner': self._get_winner_label(main_prediction),
                'probability': float(main_prediction.get('confidence', 0.5)),
                'confidence': float(main_prediction.get('confidence', 0.5)),
                'probabilities': main_prediction.get('probabilities', {}),
                'consensus': consensus
            },
            'model_predictions': predictions
        }
        
        # Add both teams to score prediction
        final_prediction['prediction']['both_teams_to_score'] = self._predict_btts(
            original_features, main_prediction
        )
        
        # Add predicted score
        final_prediction['prediction']['predicted_score'] = self._predict_score(
            original_features, main_prediction
        )
        
        return final_prediction
    
    def _calculate_consensus(self, predictions: Dict) -> Dict:
        """Calculate consensus among models"""
        if not predictions:
            return {}
        
        # Count predictions by class
        class_counts = {}
        confidences = {}
        
        for model_name, pred in predictions.items():
            pred_class = pred.get('predicted_class')
            confidence = pred.get('confidence', 0)
            
            if pred_class is not None:
                if pred_class not in class_counts:
                    class_counts[pred_class] = 0
                    confidences[pred_class] = []
                
                class_counts[pred_class] += 1
                confidences[pred_class].append(confidence)
        
        # Calculate consensus
        total_models = len(predictions)
        consensus = {}
        
        for pred_class, count in class_counts.items():
            consensus[pred_class] = {
                'count': count,
                'percentage': count / total_models,
                'avg_confidence': np.mean(confidences[pred_class]) if confidences[pred_class] else 0
            }
        
        return consensus
    
    def _get_winner_label(self, prediction: Dict) -> str:
        """Get winner label from prediction"""
        pred_class = prediction.get('predicted_class')
        
        if pred_class == 0:
            return 'home'
        elif pred_class == 1:
            return 'draw'
        elif pred_class == 2:
            return 'away'
        else:
            return 'unknown'
    
    def _predict_btts(self, features: Dict, prediction: Dict) -> Dict:
        """Predict both teams to score"""
        # Extract relevant features
        home_goals_avg = features.get('home_goals_scored_avg', 1.5)
        away_goals_avg = features.get('away_goals_scored_avg', 1.2)
        home_conceded_avg = features.get('home_goals_conceded_avg', 1.0)
        away_conceded_avg = features.get('away_goals_conceded_avg', 1.3)
        
        # Calculate BTTS probability
        home_score_prob = min(1.0, home_goals_avg / 5)  # Normalize
        away_score_prob = min(1.0, away_goals_avg / 5)
        
        btts_probability = home_score_prob * away_score_prob
        
        # Adjust based on defensive strength
        home_defense_factor = 1 - min(1.0, home_conceded_avg / 5)
        away_defense_factor = 1 - min(1.0, away_conceded_avg / 5)
        
        btts_probability = btts_probability * (home_defense_factor + away_defense_factor) / 2
        
        # Ensure reasonable range
        btts_probability = max(0.1, min(0.9, btts_probability))
        
        return {
            'prediction': btts_probability > 0.5,
            'probability': float(btts_probability),
            'confidence': float(abs(btts_probability - 0.5) * 2)  # Convert to 0-1 scale
        }
    
    def _predict_score(self, features: Dict, prediction: Dict) -> Dict:
        """Predict match score"""
        # Extract relevant features
        home_attack = features.get('home_goals_scored_avg', 1.5)
        away_attack = features.get('away_goals_scored_avg', 1.2)
        home_defense = features.get('home_goals_conceded_avg', 1.0)
        away_defense = features.get('away_goals_conceded_avg', 1.3)
        
        # Calculate expected goals (simplified)
        home_xg = (home_attack + away_defense) / 2
        away_xg = (away_attack + home_defense) / 2
        
        # Adjust based on prediction
        winner = prediction.get('predicted_class', 1)
        
        if winner == 0:  # Home win
            home_xg *= 1.2
            away_xg *= 0.8
        elif winner == 2:  # Away win
            home_xg *= 0.8
            away_xg *= 1.2
        
        # Convert to integers (Poisson distribution simulation)
        import math
        
        def poisson_goal(lam):
            # Simple rounding for now
            return max(0, int(round(lam)))
        
        home_goals = poisson_goal(home_xg)
        away_goals = poisson_goal(away_xg)
        
        # Ensure at least some goals
        if home_goals == 0 and away_goals == 0:
            home_goals = 1 if home_xg > away_xg else 0
            away_goals = 1 if away_xg > home_xg else 0
        
        return {
            'home': home_goals,
            'away': away_goals,
            'total': home_goals + away_goals,
            'difference': home_goals - away_goals
        }
    
    def _create_fallback_prediction(self, features: Dict) -> Dict:
        """Create fallback prediction when models fail"""
        self.logger.warning("Creating fallback prediction")
        
        # Simple prediction based on odds if available
        odds = features.get('odds', {})
        
        if odds and 'home' in odds and 'draw' in odds and 'away' in odds:
            # Use odds-based prediction
            home_odds = odds['home']
            draw_odds = odds['draw']
            away_odds = odds['away']
            
            # Calculate implied probabilities
            home_prob = 1 / home_odds
            draw_prob = 1 / draw_odds
            away_prob = 1 / away_odds
            
            total_prob = home_prob + draw_prob + away_prob
            home_prob /= total_prob
            draw_prob /= total_prob
            away_prob /= total_prob
            
            # Determine winner
            probs = [home_prob, draw_prob, away_prob]
            winner_idx = np.argmax(probs)
            confidence = max(probs)
            
            winner_map = {0: 'home', 1: 'draw', 2: 'away'}
            winner = winner_map.get(winner_idx, 'draw')
            
        else:
            # Default prediction (draw)
            winner = 'draw'
            confidence = 0.33
            home_prob, draw_prob, away_prob = 0.33, 0.34, 0.33
        
        return {
            'match_info': {
                'home_team': features.get('home_team', ''),
                'away_team': features.get('away_team', ''),
                'league': features.get('league', ''),
                'date': features.get('date', '')
            },
            'prediction': {
                'winner': winner,
                'probability': float(confidence),
                'confidence': float(confidence),
                'probabilities': {
                    'home': float(home_prob),
                    'draw': float(draw_prob),
                    'away': float(away_prob)
                },
                'consensus': {},
                'both_teams_to_score': {
                    'prediction': True,  # Default to yes
                    'probability': 0.6,
                    'confidence': 0.2
                },
                'predicted_score': {
                    'home': 1,
                    'away': 1,
                    'total': 2,
                    'difference': 0
                }
            },
            'model_predictions': {},
            'prediction_metadata': {
                'models_used': ['fallback'],
                'prediction_time': 0.0,
                'timestamp': datetime.now().isoformat(),
                'is_fallback': True
            }
        }
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """Make predictions for multiple matches"""
        if not features_list:
            return []
        
        self.logger.info(f"Making batch predictions for {len(features_list)} matches...")
        
        predictions = []
        
        # Use parallel processing if enabled
        if self.prediction_config['parallel_prediction']:
            import concurrent.futures
            
            max_workers = min(self.optimizer.optimization_config.max_parallel_processes, 
                            len(features_list))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_features = {
                    executor.submit(self.predict, features): features 
                    for features in features_list[:self.prediction_config['batch_size']]
                }
                
                for future in concurrent.futures.as_completed(future_to_features):
                    try:
                        prediction = future.result()
                        predictions.append(prediction)
                    except Exception as e:
                        self.logger.error(f"Batch prediction failed: {e}")
                        features = future_to_features[future]
                        predictions.append(self._create_fallback_prediction(features))
        else:
            # Sequential processing
            batch_size = self.prediction_config['batch_size']
            
            for i, features in enumerate(features_list[:batch_size]):
                try:
                    prediction = self.predict(features)
                    predictions.append(prediction)
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Processed {i + 1}/{min(len(features_list), batch_size)} matches")
                        
                except Exception as e:
                    self.logger.error(f"Prediction failed for match {i}: {e}")
                    predictions.append(self._create_fallback_prediction(features))
        
        self.logger.info(f"Batch predictions completed: {len(predictions)} predictions made")
        
        return predictions
    
    def filter_by_confidence(self, predictions: List[Dict], 
                           min_confidence: float = None) -> List[Dict]:
        """Filter predictions by confidence threshold"""
        if min_confidence is None:
            min_confidence = self.prediction_config['confidence_threshold']
        
        filtered = []
        
        for prediction in predictions:
            confidence = prediction['prediction'].get('confidence', 0)
            if confidence >= min_confidence:
                filtered.append(prediction)
        
        self.logger.info(f"Filtered {len(predictions)} to {len(filtered)} high-confidence predictions")
        
        return filtered
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old prediction cache"""
        try:
            current_time = datetime.now().timestamp()
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Prediction cache cleanup failed: {e}")
    
    def get_prediction_report(self) -> Dict:
        """Generate prediction report"""
        report = {
            'prediction_config': self.prediction_config,
            'prediction_stats': self.prediction_stats,
            'models_loaded': list(self.models.keys()),
            'ensemble_available': self.ensemble_model is not None,
            'cache_info': {
                'enabled': self.prediction_config['use_cache'],
                'ttl_seconds': self.prediction_config['cache_ttl'],
                'cache_dir': str(self.cache_dir)
            }
        }
        
        # Calculate cache hit rate
        if self.prediction_stats['total_predictions'] > 0:
            report['cache_hit_rate'] = (
                self.prediction_stats['cache_hits'] / self.prediction_stats['total_predictions']
            )
        
        # Model usage statistics
        if self.prediction_stats['model_calls']:
            total_calls = sum(self.prediction_stats['model_calls'].values())
            report['model_usage'] = {
                model: {
                    'calls': calls,
                    'percentage': calls / total_calls if total_calls > 0 else 0
                }
                for model, calls in self.prediction_stats['model_calls'].items()
            }
        
        return report
