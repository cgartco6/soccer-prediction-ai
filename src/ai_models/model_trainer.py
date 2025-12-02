"""
Model training and retraining with hardware optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
import joblib
import pickle
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from ..system.optimizer import SystemOptimizer
from .model_architectures import (
    XGBoostModel, CatBoostModel, LightGBMModel,
    NeuralNetworkModel, GradientBoostingModel, RandomForestModel
)

class ModelTrainer:
    """Train and manage machine learning models with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup training strategy
        self.setup_training_strategy()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Training metadata
        self.training_metadata = {
            'start_time': None,
            'end_time': None,
            'models_trained': [],
            'performance_metrics': {}
        }
        
        # Model directory
        self.models_dir = Path("./data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_training_strategy(self):
        """Setup training strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.training_config = {
                'train_models': ['xgboost', 'catboost'],  # Only essential models
                'cross_validation_folds': 3,
                'hyperparameter_tuning': False,
                'ensemble_training': False,
                'early_stopping_rounds': 20,
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.01,
                'batch_size': 16,
                'epochs': 50,
                'use_cache': True,
                'parallel_training': False
            }
        elif profile == 'mid_end':
            self.training_config = {
                'train_models': ['xgboost', 'catboost', 'lightgbm', 'gradient_boosting'],
                'cross_validation_folds': 5,
                'hyperparameter_tuning': True,
                'ensemble_training': True,
                'early_stopping_rounds': 30,
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'batch_size': 32,
                'epochs': 100,
                'use_cache': True,
                'parallel_training': True
            }
        else:  # high_end
            self.training_config = {
                'train_models': ['xgboost', 'catboost', 'lightgbm', 'neural_network', 
                                'gradient_boosting', 'random_forest'],
                'cross_validation_folds': 10,
                'hyperparameter_tuning': True,
                'ensemble_training': True,
                'early_stopping_rounds': 50,
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'batch_size': 64,
                'epochs': 200,
                'use_cache': True,
                'parallel_training': True
            }
        
        self.logger.info(f"Model training strategy: {self.training_config}")
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'result') -> Tuple:
        """Prepare data for training"""
        self.logger.info("Preparing data for training...")
        
        if data.empty:
            self.logger.error("No data provided for training")
            return None, None, None, None, None, None, []
        
        # Make a copy
        df = data.copy()
        
        # Separate features and target
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found")
            return None, None, None, None, None, None, []
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Remove non-numeric columns that aren't needed
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            self.logger.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode target if needed
        if y.dtype == 'object':
            self.encoders['target'] = LabelEncoder()
            y = pd.Series(self.encoders['target'].fit_transform(y), name=target_col)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scalers['standard'].fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data
        test_size = 0.2
        val_size = 0.1
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Second split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            random_state=42,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        self.logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, 
                    feature_names: List[str] = None, retrain: bool = False):
        """Train all models"""
        self.logger.info("Starting model training...")
        self.training_metadata['start_time'] = datetime.now().isoformat()
        
        models_to_train = self.training_config['train_models']
        
        # Train each model
        for model_name in models_to_train:
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Check if model already exists and retrain is False
                if not retrain and model_name in self.models:
                    self.logger.info(f"Model {model_name} already exists, skipping...")
                    continue
                
                # Train model
                model = self._train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                if model is not None:
                    self.models[model_name] = model
                    self.training_metadata['models_trained'].append(model_name)
                    
                    # Get feature importance
                    importance = model.get_feature_importance()
                    if importance:
                        self.feature_importance[model_name] = importance
                    
                    # Store training history
                    if hasattr(model, 'training_history') and model.training_history:
                        self.training_history[model_name] = model.training_history
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
        
        self.training_metadata['end_time'] = datetime.now().isoformat()
        self.logger.info(f"Model training completed: {len(self.models)} models trained")
    
    def _train_single_model(self, model_name: str, X_train, y_train, 
                           X_val=None, y_val=None) -> Optional[Any]:
        """Train a single model"""
        # Model configuration
        model_config = {
            'n_estimators': self.training_config['n_estimators'],
            'max_depth': self.training_config['max_depth'],
            'learning_rate': self.training_config['learning_rate'],
            'early_stopping_rounds': self.training_config['early_stopping_rounds']
        }
        
        # Create model instance
        if model_name == 'xgboost':
            if not XGB_AVAILABLE:
                self.logger.warning("XGBoost not available, skipping...")
                return None
            
            model = XGBoostModel(self.optimizer, model_config)
        
        elif model_name == 'catboost':
            if not CATBOOST_AVAILABLE:
                self.logger.warning("CatBoost not available, skipping...")
                return None
            
            model = CatBoostModel(self.optimizer, model_config)
        
        elif model_name == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                self.logger.warning("LightGBM not available, skipping...")
                return None
            
            model = LightGBMModel(self.optimizer, model_config)
        
        elif model_name == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("TensorFlow not available, skipping...")
                return None
            
            model_config['epochs'] = self.training_config['epochs']
            model_config['batch_size'] = self.training_config['batch_size']
            model = NeuralNetworkModel(self.optimizer, model_config)
        
        elif model_name == 'gradient_boosting':
            model = GradientBoostingModel(self.optimizer, model_config)
        
        elif model_name == 'random_forest':
            model = RandomForestModel(self.optimizer, model_config)
        
        else:
            self.logger.warning(f"Unknown model: {model_name}")
            return None
        
        # Train model
        trained_model = model.fit(X_train, y_train, X_val, y_val)
        
        return model if trained_model is not None else None
    
    def evaluate_models(self, X_test, y_test) -> Dict:
        """Evaluate all trained models"""
        self.logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                evaluation_results[model_name] = metrics
                self.training_metadata['performance_metrics'][model_name] = metrics
                
                self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                               f"F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = None
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC-ROC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['auc_roc'] = 0.0
        else:
            # Multi-class AUC-ROC (one-vs-rest)
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    def hyperparameter_tuning(self, X_train, y_train, model_name: str = 'xgboost') -> Dict:
        """Perform hyperparameter tuning"""
        if not self.training_config['hyperparameter_tuning']:
            self.logger.info("Hyperparameter tuning disabled")
            return {}
        
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define search space based on model
        param_grid = self._get_param_grid(model_name)
        
        if not param_grid:
            return {}
        
        # Create base model
        if model_name == 'xgboost':
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                random_state=42,
                n_jobs=self.optimizer.optimization_config.max_parallel_processes,
                verbosity=0
            )
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=self.optimizer.optimization_config.max_parallel_processes
            )
        else:
            self.logger.warning(f"Hyperparameter tuning not supported for {model_name}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.training_config['cross_validation_folds'],
            scoring='accuracy',
            n_jobs=self.optimizer.optimization_config.max_parallel_processes,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        if model_name in self.models:
            self.models[model_name].model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _get_param_grid(self, model_name: str) -> Dict:
        """Get parameter grid for hyperparameter tuning"""
        if model_name == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            return {}
    
    def cross_validate(self, X, y, model_name: str = 'xgboost') -> Dict:
        """Perform cross-validation"""
        self.logger.info(f"Performing cross-validation for {model_name}...")
        
        # Get model
        if model_name not in self.models:
            self.logger.warning(f"Model {model_name} not trained")
            return {}
        
        model = self.models[model_name].model
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=self.training_config['cross_validation_folds'],
            scoring='accuracy',
            n_jobs=self.optimizer.optimization_config.max_parallel_processes
        )
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': self.training_config['cross_validation_folds']
        }
    
    def save_models(self, model_dir: str = None):
        """Save all trained models"""
        if model_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = self.models_dir / f"models_{timestamp}"
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving models to {model_dir}...")
        
        # Save individual models
        for model_name, model in self.models.items():
            try:
                model_file = model_dir / f"{model_name}.model"
                model.save(str(model_file))
                self.logger.info(f"Saved {model_name} to {model_file}")
            except Exception as e:
                self.logger.error(f"Failed to save {model_name}: {e}")
        
        # Save scalers and encoders
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'training_metadata': self.training_metadata,
            'training_history': self.training_history,
            'training_config': self.training_config
        }
        
        artifacts_file = model_dir / "artifacts.pkl"
        try:
            with open(artifacts_file, 'wb') as f:
                pickle.dump(artifacts, f)
            self.logger.info(f"Saved artifacts to {artifacts_file}")
        except Exception as e:
            self.logger.error(f"Failed to save artifacts: {e}")
        
        # Save metadata as JSON
        metadata_file = model_dir / "metadata.json"
        try:
            # Convert numpy arrays to lists for JSON serialization
            metadata_json = json.dumps(self.training_metadata, default=self._json_serializer, indent=2)
            with open(metadata_file, 'w') as f:
                f.write(metadata_json)
            self.logger.info(f"Saved metadata to {metadata_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
        
        return str(model_dir)
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def load_models(self, model_dir: str):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return False
        
        self.logger.info(f"Loading models from {model_dir}...")
        
        # Load artifacts
        artifacts_file = model_dir / "artifacts.pkl"
        if artifacts_file.exists():
            try:
                with open(artifacts_file, 'rb') as f:
                    artifacts = pickle.load(f)
                
                self.scalers = artifacts.get('scalers', {})
                self.encoders = artifacts.get('encoders', {})
                self.feature_importance = artifacts.get('feature_importance', {})
                self.training_metadata = artifacts.get('training_metadata', {})
                self.training_history = artifacts.get('training_history', {})
                self.training_config = artifacts.get('training_config', self.training_config)
                
                self.logger.info("Artifacts loaded")
            except Exception as e:
                self.logger.error(f"Failed to load artifacts: {e}")
        
        # Load models
        model_files = list(model_dir.glob("*.model"))
        
        for model_file in model_files:
            model_name = model_file.stem
            
            try:
                # Create model instance
                if model_name == 'xgboost':
                    model = XGBoostModel(self.optimizer)
                elif model_name == 'catboost':
                    model = CatBoostModel(self.optimizer)
                elif model_name == 'lightgbm':
                    model = LightGBMModel(self.optimizer)
                elif model_name == 'neural_network':
                    model = NeuralNetworkModel(self.optimizer)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingModel(self.optimizer)
                elif model_name == 'random_forest':
                    model = RandomForestModel(self.optimizer)
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Load model
                model.load(str(model_file))
                self.models[model_name] = model
                
                self.logger.info(f"Loaded {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {e}")
        
        self.logger.info(f"Loaded {len(self.models)} models")
        return True
    
    def retrain(self, new_data: pd.DataFrame, incremental: bool = True):
        """Retrain models with new data"""
        self.logger.info("Starting model retraining...")
        
        if incremental and self.models:
            # Update existing models with new data
            for model_name, model in self.models.items():
                try:
                    self.logger.info(f"Retraining {model_name}...")
                    
                    # Get training data
                    X = new_data.drop(columns=['result'])
                    y = new_data['result']
                    
                    # Encode target if needed
                    if y.dtype == 'object' and 'target' in self.encoders:
                        y = self.encoders['target'].transform(y)
                    
                    # Scale features
                    if 'standard' in self.scalers:
                        X = self.scalers['standard'].transform(X)
                    
                    # Retrain model
                    model.fit(X, y)
                    
                    self.logger.info(f"{model_name} retraining completed")
                    
                except Exception as e:
                    self.logger.error(f"Failed to retrain {model_name}: {e}")
        else:
            # Full retraining
            self.models.clear()
            self.train_models_from_data(new_data, retrain=True)
        
        self.logger.info("Model retraining completed")
    
    def train_models_from_data(self, data: pd.DataFrame, retrain: bool = False):
        """Complete training pipeline from data"""
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.prepare_data(data)
        
        if X_train is None:
            return
        
        # Train models
        self.train_models(X_train, y_train, X_val, y_val, feature_names, retrain)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # Hyperparameter tuning if enabled
        if self.training_config['hyperparameter_tuning']:
            for model_name in self.models.keys():
                if model_name in ['xgboost', 'random_forest']:
                    self.hyperparameter_tuning(X_train, y_train, model_name)
        
        # Cross-validation
        cv_results = {}
        for model_name in self.models.keys():
            if model_name in ['xgboost', 'random_forest']:
                cv_results[model_name] = self.cross_validate(X_train, y_train, model_name)
        
        return evaluation_results, cv_results
    
    def get_training_report(self) -> Dict:
        """Generate training report"""
        report = {
            'training_config': self.training_config,
            'training_metadata': self.training_metadata,
            'models_trained': list(self.models.keys()),
            'model_count': len(self.models),
            'feature_importance': self._summarize_feature_importance(),
            'hardware_profile': self.optimizer.hardware_info.get('hardware_profile')
        }
        
        # Add performance metrics if available
        if self.training_metadata.get('performance_metrics'):
            report['performance_summary'] = self._summarize_performance()
        
        return report
    
    def _summarize_feature_importance(self) -> Dict:
        """Summarize feature importance across models"""
        if not self.feature_importance:
            return {}
        
        summary = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            if importance_dict:
                # Get top 10 features
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                summary[model_name] = {
                    'top_features': [feat for feat, _ in sorted_features],
                    'top_importance': [imp for _, imp in sorted_features]
                }
        
        return summary
    
    def _summarize_performance(self) -> Dict:
        """Summarize model performance"""
        performance = self.training_metadata.get('performance_metrics', {})
        
        if not performance:
            return {}
        
        summary = {}
        best_model = None
        best_accuracy = 0
        
        for model_name, metrics in performance.items():
            if metrics:
                summary[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'auc_roc': metrics.get('auc_roc', 0)
                }
                
                # Track best model
                if metrics.get('accuracy', 0) > best_accuracy:
                    best_accuracy = metrics.get('accuracy', 0)
                    best_model = model_name
        
        summary['best_model'] = best_model
        summary['best_accuracy'] = best_accuracy
        
        return summary

class AutomatedRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, trainer: ModelTrainer, config: Dict = None):
        self.trainer = trainer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Retraining configuration
        self.retrain_config = {
            'min_new_samples': 100,
            'performance_threshold': 0.65,
            'retrain_interval_days': 7,
            'max_models_to_keep': 10,
            'monitor_metrics': ['accuracy', 'f1_score'],
            'drift_detection': True,
            'drift_threshold': 0.05
        }
        
        # Update with user config
        self.retrain_config.update(self.config)
        
        # Monitoring data
        self.performance_history = []
        self.data_drift_history = []
        self.last_retrain_date = None
        
    def check_retrain_need(self, new_performance: Dict = None, 
                          new_data_count: int = 0) -> bool:
        """Check if retraining is needed"""
        reasons = []
        
        # Check performance degradation
        if new_performance and self.retrain_config['performance_threshold'] > 0:
            current_performance = self._calculate_current_performance()
            if current_performance < self.retrain_config['performance_threshold']:
                reasons.append(f"Performance below threshold: {current_performance:.3f}")
        
        # Check new data volume
        if new_data_count >= self.retrain_config['min_new_samples']:
            reasons.append(f"New data available: {new_data_count} samples")
        
        # Check time interval
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.retrain_config['retrain_interval_days']:
                reasons.append(f"Time interval reached: {days_since_retrain} days")
        else:
            # First time retraining
            reasons.append("Initial retraining")
        
        # Check data drift
        if self.retrain_config['drift_detection']:
            drift_detected = self._check_data_drift()
            if drift_detected:
                reasons.append("Data drift detected")
        
        if reasons:
            self.logger.info(f"Retraining needed: {', '.join(reasons)}")
            return True
        
        return False
    
    def _calculate_current_performance(self) -> float:
        """Calculate current model performance"""
        if not self.trainer.training_metadata.get('performance_metrics'):
            return 0.0
        
        performances = []
        for metrics in self.trainer.training_metadata['performance_metrics'].values():
            if metrics:
                # Use F1 score as primary metric
                f1 = metrics.get('f1_score', 0)
                performances.append(f1)
        
        return np.mean(performances) if performances else 0.0
    
    def _check_data_drift(self) -> bool:
        """Check for data drift"""
        # Implementation would require storing historical data distributions
        # For now, return False
        return False
    
    def monitor_performance(self, predictions: List[Dict], actual_results: List[Dict]) -> Dict:
        """Monitor model performance and collect data for retraining"""
        monitoring_data = {
            'total_predictions': len(predictions),
            'correct_predictions': 0,
            'accuracy': 0,
            'by_league': {},
            'by_confidence': {},
            'timestamp': datetime.now().isoformat()
        }
        
        training_samples = []
        
        for pred, actual in zip(predictions, actual_results):
            if actual.get('result'):
                is_correct = (pred['prediction']['winner'] == actual['result'])
                monitoring_data['correct_predictions'] += int(is_correct)
                
                # Collect data for retraining
                training_sample = self._create_training_sample(pred, actual)
                if training_sample:
                    training_samples.append(training_sample)
                
                # Track by league
                league = pred.get('league', 'unknown')
                if league not in monitoring_data['by_league']:
                    monitoring_data['by_league'][league] = {'total': 0, 'correct': 0}
                monitoring_data['by_league'][league]['total'] += 1
                monitoring_data['by_league'][league]['correct'] += int(is_correct)
                
                # Track by confidence
                confidence = pred['prediction'].get('confidence', 0.5)
                conf_level = int(confidence * 10) / 10  # Round to 0.1
                if conf_level not in monitoring_data['by_confidence']:
                    monitoring_data['by_confidence'][conf_level] = {'total': 0, 'correct': 0}
                monitoring_data['by_confidence'][conf_level]['total'] += 1
                monitoring_data['by_confidence'][conf_level]['correct'] += int(is_correct)
        
        # Calculate accuracy
        if monitoring_data['total_predictions'] > 0:
            monitoring_data['accuracy'] = (
                monitoring_data['correct_predictions'] / monitoring_data['total_predictions']
            )
        
        # Store in history
        self.performance_history.append(monitoring_data)
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Check if retraining is needed
        if self.check_retrain_need(
            new_performance={'accuracy': monitoring_data['accuracy']},
            new_data_count=len(training_samples)
        ):
            if training_samples:
                self._retrain_with_samples(training_samples)
        
        return monitoring_data
    
    def _create_training_sample(self, prediction: Dict, actual: Dict) -> Optional[Dict]:
        """Create training sample from prediction and actual result"""
        try:
            # Extract features from prediction
            features = prediction.get('features', {})
            
            # Get actual result
            result = actual.get('result', '')
            
            # Map result to numeric
            result_mapping = {'H': 0, 'D': 1, 'A': 2}
            result_numeric = result_mapping.get(result, 1)  # Default to draw
            
            return {
                'features': features,
                'result': result_numeric,
                'league': prediction.get('league', ''),
                'date': prediction.get('date', ''),
                'confidence': prediction['prediction'].get('confidence', 0.5)
            }
        except:
            return None
    
    def _retrain_with_samples(self, training_samples: List[Dict]):
        """Retrain models with new samples"""
        if not training_samples:
            return
        
        self.logger.info(f"Retraining with {len(training_samples)} new samples...")
        
        # Convert to DataFrame
        df = pd.DataFrame(training_samples)
        
        # Extract features
        feature_dfs = []
        for sample in training_samples:
            if 'features' in sample:
                feature_df = pd.DataFrame([sample['features']])
                feature_dfs.append(feature_df)
        
        if feature_dfs:
            features_df = pd.concat(feature_dfs, ignore_index=True)
            features_df['result'] = df['result'].values
            
            # Retrain models
            self.trainer.retrain(features_df, incremental=True)
            
            # Update last retrain date
            self.last_retrain_date = datetime.now()
            
            self.logger.info(f"Models retrained with {len(training_samples)} new samples")
    
    def cleanup_old_models(self, models_dir: str = "./data/models"):
        """Clean up old model versions"""
        models_path = Path(models_dir)
        
        if not models_path.exists():
            return
        
        # Find all model directories
        model_dirs = []
        for item in models_path.iterdir():
            if item.is_dir() and item.name.startswith('models_'):
                model_dirs.append(item)
        
        # Sort by creation time (oldest first)
        model_dirs.sort(key=lambda x: x.stat().st_ctime)
        
        # Keep only the most recent models
        max_models = self.retrain_config['max_models_to_keep']
        if len(model_dirs) > max_models:
            for old_dir in model_dirs[:-max_models]:
                try:
                    import shutil
                    shutil.rmtree(old_dir)
                    self.logger.info(f"Removed old model directory: {old_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {old_dir}: {e}")
    
    def get_monitoring_report(self) -> Dict:
        """Generate monitoring report"""
        report = {
            'retrain_config': self.retrain_config,
            'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            'performance_history_count': len(self.performance_history),
            'current_performance': self._calculate_current_performance(),
            'retraining_needed': self.check_retrain_need()
        }
        
        # Add recent performance if available
        if self.performance_history:
            recent = self.performance_history[-1]
            report['recent_performance'] = {
                'accuracy': recent.get('accuracy', 0),
                'total_predictions': recent.get('total_predictions', 0),
                'correct_predictions': recent.get('correct_predictions', 0)
            }
        
        return report
