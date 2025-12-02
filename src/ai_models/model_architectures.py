"""
Individual model architectures with hardware optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries (conditional based on hardware)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from ..system.optimizer import SystemOptimizer

class BaseModel:
    """Base class for all models with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, model_name: str, config: Dict = None):
        self.optimizer = optimizer
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.is_trained = False
        self.training_history = None
        
        # Setup model based on hardware
        self.setup_model()
    
    def setup_model(self):
        """Setup model configuration based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        model_config = self.optimizer.get_model_config(self.model_name)
        
        # Merge configs
        self.model_config = {**model_config, **self.config}
        
        # Adjust for hardware
        self._adjust_for_hardware()
    
    def _adjust_for_hardware(self):
        """Adjust model configuration for hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            # Reduce complexity for low-end hardware
            if 'n_estimators' in self.model_config:
                self.model_config['n_estimators'] = min(100, self.model_config.get('n_estimators', 100))
            if 'max_depth' in self.model_config:
                self.model_config['max_depth'] = min(4, self.model_config.get('max_depth', 4))
            if 'num_leaves' in self.model_config:
                self.model_config['num_leaves'] = min(31, self.model_config.get('num_leaves', 31))
        
        elif profile == 'high_end':
            # Increase complexity for high-end hardware
            if 'n_estimators' in self.model_config:
                self.model_config['n_estimators'] = max(500, self.model_config.get('n_estimators', 500))
            if 'max_depth' in self.model_config:
                self.model_config['max_depth'] = max(8, self.model_config.get('max_depth', 8))
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
    
    def predict_proba(self, X):
        """Predict probabilities"""
        raise NotImplementedError
    
    def save(self, filepath: str):
        """Save model to file"""
        raise NotImplementedError
    
    def load(self, filepath: str):
        """Load model from file"""
        raise NotImplementedError
    
    def get_feature_importance(self):
        """Get feature importance"""
        return {}

class XGBoostModel(BaseModel):
    """XGBoost model with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'xgboost', config)
        
        if not XGB_AVAILABLE:
            self.logger.error("XGBoost not available")
            return
        
        self._init_model()
    
    def _init_model(self):
        """Initialize XGBoost model"""
        # Get GPU settings
        use_gpu = self.optimizer.optimization_config.gpu_acceleration
        
        # Default parameters
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.01,
            'max_depth': self.model_config.get('max_depth', 6),
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'n_estimators': self.model_config.get('n_estimators', 300),
            'random_state': 42,
            'n_jobs': self.optimizer.optimization_config.max_parallel_processes,
            'verbosity': 0
        }
        
        # GPU optimization
        if use_gpu:
            default_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0
            })
        else:
            default_params['tree_method'] = 'hist'
        
        # Update with user config
        self.params = {**default_params, **self.model_config}
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        self.logger.info(f"XGBoost model initialized with params: {self.params}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train XGBoost model"""
        if self.model is None:
            self.logger.error("Model not initialized")
            return
        
        self.logger.info("Training XGBoost model...")
        
        eval_set = None
        early_stopping = kwargs.get('early_stopping_rounds', 50)
        verbose = kwargs.get('verbose', False)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose,
                early_stopping_rounds=early_stopping
            )
            
            self.is_trained = True
            self.logger.info("XGBoost training completed")
            
            # Store training history
            if eval_set:
                self.training_history = self.model.evals_result()
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        
        if feature_names is None:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"XGBoost model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"XGBoost model loaded from {filepath}")

class CatBoostModel(BaseModel):
    """CatBoost model with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'catboost', config)
        
        if not CATBOOST_AVAILABLE:
            self.logger.error("CatBoost not available")
            return
        
        self._init_model()
    
    def _init_model(self):
        """Initialize CatBoost model"""
        # Get GPU settings
        use_gpu = self.optimizer.optimization_config.gpu_acceleration
        
        # Default parameters
        default_params = {
            'iterations': self.model_config.get('n_estimators', 300),
            'learning_rate': 0.05,
            'depth': self.model_config.get('max_depth', 6),
            'l2_leaf_reg': 3,
            'loss_function': 'MultiClass',
            'verbose': False,
            'random_seed': 42,
            'task_type': 'GPU' if use_gpu else 'CPU',
            'devices': '0' if use_gpu else None,
            'early_stopping_rounds': 50,
            'use_best_model': True
        }
        
        # Update with user config
        self.params = {**default_params, **self.model_config}
        
        # Initialize model
        self.model = cb.CatBoostClassifier(**self.params)
        
        self.logger.info(f"CatBoost model initialized with params: {self.params}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train CatBoost model"""
        if self.model is None:
            self.logger.error("Model not initialized")
            return
        
        self.logger.info("Training CatBoost model...")
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
                plot=False
            )
            
            self.is_trained = True
            self.logger.info("CatBoost training completed")
            
            # Store training history
            self.training_history = self.model.get_evals_result()
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        if feature_names is None:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        self.model.save_model(filepath)
        self.logger.info(f"CatBoost model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = cb.CatBoostClassifier()
        self.model.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"CatBoost model loaded from {filepath}")

class LightGBMModel(BaseModel):
    """LightGBM model with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'lightgbm', config)
        
        if not LIGHTGBM_AVAILABLE:
            self.logger.error("LightGBM not available")
            return
        
        self._init_model()
    
    def _init_model(self):
        """Initialize LightGBM model"""
        # Get GPU settings
        use_gpu = self.optimizer.optimization_config.gpu_acceleration
        
        # Default parameters
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': self.model_config.get('max_depth', -1),
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'n_estimators': self.model_config.get('n_estimators', 300),
            'random_state': 42,
            'n_jobs': self.optimizer.optimization_config.max_parallel_processes,
            'verbose': -1
        }
        
        # GPU optimization
        if use_gpu:
            default_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        # Update with user config
        self.params = {**default_params, **self.model_config}
        
        # Initialize model
        self.model = lgb.LGBMClassifier(**self.params)
        
        self.logger.info(f"LightGBM model initialized with params: {self.params}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train LightGBM model"""
        if self.model is None:
            self.logger.error("Model not initialized")
            return
        
        self.logger.info("Training LightGBM model...")
        
        eval_set = None
        early_stopping = kwargs.get('early_stopping_rounds', 50)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping)]
            )
            
            self.is_trained = True
            self.logger.info("LightGBM training completed")
            
            # Store training history
            if eval_set:
                self.training_history = self.model.evals_result_
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.model.feature_name_
        
        if feature_names is None:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"LightGBM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"LightGBM model loaded from {filepath}")

class NeuralNetworkModel(BaseModel):
    """Neural Network model with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'neural_network', config)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return
        
        self._init_model()
    
    def _init_model(self):
        """Initialize Neural Network model"""
        # Get GPU settings
        use_gpu = self.optimizer.optimization_config.gpu_acceleration
        
        # Get layer configuration based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            layers = [64, 32]
            dropout_rate = 0.3
        elif profile == 'mid_end':
            layers = [128, 64, 32]
            dropout_rate = 0.3
        else:  # high_end
            layers = [256, 128, 64, 32]
            dropout_rate = 0.4
        
        # Override with config if provided
        layers = self.model_config.get('neural_network_layers', layers)
        dropout_rate = self.model_config.get('dropout_rate', dropout_rate)
        
        self.layers = layers
        self.dropout_rate = dropout_rate
        
        self.logger.info(f"Neural Network configured with layers: {layers}")
    
    def _build_model(self, input_shape: Tuple[int], num_classes: int = 3):
        """Build neural network architecture"""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Hidden layers
        for i, units in enumerate(self.layers):
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train Neural Network"""
        self.logger.info("Training Neural Network...")
        
        # Convert data to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
        if y_val is not None:
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
        
        # Build model
        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y_train))
        
        self.model = self._build_model(input_shape, num_classes)
        
        # Compile model
        learning_rate = self.model_config.get('learning_rate', 0.001)
        
        from tensorflow import keras
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=20,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        import tempfile
        checkpoint_file = tempfile.NamedTemporaryFile(delete=False).name
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_file,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            save_weights_only=True
        )
        callbacks.append(model_checkpoint)
        
        # Training parameters
        batch_size = self.optimizer.optimization_config.batch_size
        epochs = self.model_config.get('epochs', 100)
        
        try:
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1 if self.optimizer.hardware_info.get('hardware_profile') == 'high_end' else 0
            )
            
            self.is_trained = True
            self.training_history = history.history
            self.logger.info("Neural Network training completed")
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"Neural Network training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X, verbose=0)
    
    def get_feature_importance(self):
        """Get feature importance (not applicable for NN)"""
        return {}
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        self.model.save(filepath)
        self.logger.info(f"Neural Network model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        from tensorflow import keras
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"Neural Network model loaded from {filepath}")

class GradientBoostingModel(BaseModel):
    """Gradient Boosting model (scikit-learn)"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'gradient_boosting', config)
        self._init_model()
    
    def _init_model(self):
        """Initialize Gradient Boosting model"""
        # Default parameters
        default_params = {
            'n_estimators': self.model_config.get('n_estimators', 100),
            'learning_rate': 0.1,
            'max_depth': self.model_config.get('max_depth', 3),
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42,
            'verbose': 0
        }
        
        # Update with user config
        self.params = {**default_params, **self.model_config}
        
        # Initialize model
        self.model = GradientBoostingClassifier(**self.params)
        
        self.logger.info(f"Gradient Boosting model initialized with params: {self.params}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train Gradient Boosting model"""
        self.logger.info("Training Gradient Boosting model...")
        
        try:
            self.model.fit(X_train, y_train)
            
            self.is_trained = True
            self.logger.info("Gradient Boosting training completed")
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"Gradient Boosting training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else []
        
        if len(feature_names) == 0:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"Gradient Boosting model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"Gradient Boosting model loaded from {filepath}")

class RandomForestModel(BaseModel):
    """Random Forest model"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        super().__init__(optimizer, 'random_forest', config)
        self._init_model()
    
    def _init_model(self):
        """Initialize Random Forest model"""
        # Default parameters
        default_params = {
            'n_estimators': self.model_config.get('n_estimators', 100),
            'max_depth': self.model_config.get('max_depth', None),
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': self.optimizer.optimization_config.max_parallel_processes,
            'verbose': 0
        }
        
        # Update with user config
        self.params = {**default_params, **self.model_config}
        
        # Initialize model
        self.model = RandomForestClassifier(**self.params)
        
        self.logger.info(f"Random Forest model initialized with params: {self.params}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train Random Forest model"""
        self.logger.info("Training Random Forest model...")
        
        try:
            self.model.fit(X_train, y_train)
            
            self.is_trained = True
            self.logger.info("Random Forest training completed")
            
            return self.model
        
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            self.logger.error("Model not trained")
            return None
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else []
        
        if len(feature_names) == 0:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            return
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"Random Forest model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"Random Forest model loaded from {filepath}")
