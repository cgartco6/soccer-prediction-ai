"""
Model training and retraining module
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import pickle
from datetime import datetime
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and manage machine learning models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        # Separate features and target
        X = data.drop(['match_id', 'outcome', 'date', 'home_team', 'away_team'], axis=1, errors='ignore')
        y = data['outcome']
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['outcome'] = le
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        self.scalers['standard'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config['models']['training']['test_split'],
            stratify=y_encoded,
            random_state=42
        )
        
        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['models']['training']['validation_split'],
            stratify=y_train,
            random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, X.columns
    
    def train_xgboost(self, X_train, X_val, y_train, y_val, feature_names):
        """Train XGBoost model"""
        logging.info("Training XGBoost model...")
        
        # Define parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'n_estimators': self.config['models']['training']['n_estimators'],
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Early stopping callback
        eval_set = [(X_val, y_val)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=self.config['models']['training']['early_stopping_rounds']
        )
        
        # Get feature importance
        importance = model.feature_importances_
        self.feature_importance['xgboost'] = dict(zip(feature_names, importance))
        
        self.models['xgboost'] = model
        logging.info("XGBoost training completed")
        
        return model
    
    def train_catboost(self, X_train, X_val, y_train, y_val, feature_names):
        """Train CatBoost model"""
        logging.info("Training CatBoost model...")
        
        # Define parameters
        params = {
            'iterations': self.config['models']['training']['n_estimators'],
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'MultiClass',
            'verbose': False,
            'random_seed': 42,
            'task_type': 'CPU',
            'early_stopping_rounds': self.config['models']['training']['early_stopping_rounds']
        }
        
        # Train model
        model = cb.CatBoostClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # Get feature importance
        importance = model.get_feature_importance()
        self.feature_importance['catboost'] = dict(zip(feature_names, importance))
        
        self.models['catboost'] = model
        logging.info("CatBoost training completed")
        
        return model
    
    def train_neural_network(self, X_train, X_val, y_train, y_val, feature_names):
        """Train Neural Network model"""
        logging.info("Training Neural Network...")
        
        # Define model architecture
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            
            layers.Dense(3, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks_list,
            verbose=0
        )
        
        self.models['neural_network'] = model
        self.history = history
        
        logging.info("Neural Network training completed")
        
        return model
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val, feature_names):
        """Train LightGBM model"""
        logging.info("Training LightGBM model...")
        
        # Define parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'n_estimators': self.config['models']['training']['n_estimators'],
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(
                stopping_rounds=self.config['models']['training']['early_stopping_rounds']
            )]
        )
        
        # Get feature importance
        importance = model.feature_importances_
        self.feature_importance['lightgbm'] = dict(zip(feature_names, importance))
        
        self.models['lightgbm'] = model
        logging.info("LightGBM training completed")
        
        return model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # For binary classification metrics (Home Win vs Not Home Win)
                y_test_binary = (y_test == 0).astype(int)  # Assuming 0 is home win
                y_pred_binary = (y_pred == 0).astype(int)
                
                if len(np.unique(y_test_binary)) > 1:
                    auc_roc = roc_auc_score(y_test_binary, y_pred_prob[:, 0])
                else:
                    auc_roc = np.nan
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc
                }
                
                logging.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_roc:.4f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {name}: {e}")
                results[name] = None
        
        return results
    
    def train_ensemble(self, X_train, X_val, y_train, y_val):
        """Train ensemble model"""
        logging.info("Training ensemble model...")
        
        # Get predictions from base models for stacking
        base_predictions = {}
        
        for name, model in self.models.items():
            if name != 'ensemble':
                try:
                    if name == 'neural_network':
                        pred = model.predict(X_train)
                        val_pred = model.predict(X_val)
                    else:
                        pred = model.predict_proba(X_train)
                        val_pred = model.predict_proba(X_val)
                    
                    base_predictions[name] = {
                        'train': pred,
                        'val': val_pred
                    }
                except Exception as e:
                    logging.error(f"Error getting predictions from {name}: {e}")
        
        # Create stacked features
        if len(base_predictions) > 0:
            # Stack predictions
            X_train_stacked = np.hstack([
                preds['train'] for preds in base_predictions.values()
            ])
            X_val_stacked = np.hstack([
                preds['val'] for preds in base_predictions.values()
            ])
            
            # Train meta-model (Gradient Boosting)
            meta_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
            
            meta_model.fit(X_train_stacked, y_train)
            
            self.models['ensemble'] = meta_model
            self.base_predictors = base_predictions
            
            logging.info("Ensemble model training completed")
            
            return meta_model
        
        return None
    
    def save_models(self, save_dir: str = "./data/models"):
        """Save all trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for name, model in self.models.items():
            if name == 'neural_network':
                model_path = save_path / f"{name}_{timestamp}.h5"
                model.save(model_path)
            else:
                model_path = save_path / f"{name}_{timestamp}.pkl"
                joblib.dump(model, model_path)
        
        # Save scalers and encoders
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        artifacts_path = save_path / f"artifacts_{timestamp}.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logging.info(f"Models saved to {save_path}")
        
        return save_path
    
    def load_models(self, models_dir: str = "./data/models"):
        """Load trained models"""
        models_path = Path(models_dir)
        
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find latest models
        model_files = list(models_path.glob("*.pkl")) + list(models_path.glob("*.h5"))
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        # Load artifacts
        artifact_files = list(models_path.glob("artifacts_*.pkl"))
        if artifact_files:
            latest_artifact = max(artifact_files, key=lambda x: x.stat().st_mtime)
            with open(latest_artifact, 'rb') as f:
                artifacts = pickle.load(f)
                self.scalers = artifacts.get('scalers', {})
                self.encoders = artifacts.get('encoders', {})
                self.feature_importance = artifacts.get('feature_importance', {})
        
        # Load models
        for model_file in model_files:
            if model_file.name.startswith('artifacts_'):
                continue
                
            name = model_file.stem.split('_')[0]
            
            if model_file.suffix == '.h5':
                self.models[name] = keras.models.load_model(model_file)
            else:
                self.models[name] = joblib.load(model_file)
        
        logging.info(f"Loaded {len(self.models)} models from {models_dir}")
    
    def retrain(self, new_data: pd.DataFrame, incremental: bool = True):
        """Retrain models with new data"""
        logging.info("Starting model retraining...")
        
        if incremental:
            # Load existing models
            try:
                self.load_models()
            except:
                logging.warning("Could not load existing models, training from scratch")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.prepare_data(new_data)
        
        # Train or retrain models
        models_to_train = ['xgboost', 'catboost', 'neural_network', 'lightgbm']
        
        for model_name in models_to_train:
            try:
                if model_name == 'xgboost':
                    self.train_xgboost(X_train, X_val, y_train, y_val, feature_names)
                elif model_name == 'catboost':
                    self.train_catboost(X_train, X_val, y_train, y_val, feature_names)
                elif model_name == 'neural_network':
                    self.train_neural_network(X_train, X_val, y_train, y_val, feature_names)
                elif model_name == 'lightgbm':
                    self.train_lightgbm(X_train, X_val, y_train, y_val, feature_names)
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
        
        # Train ensemble
        self.train_ensemble(X_train, X_val, y_train, y_val)
        
        # Evaluate
        results = self.evaluate_models(X_test, y_test)
        
        # Save updated models
        self.save_models()
        
        return results

class AutomatedRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trainer = ModelTrainer(config)
        self.retrain_threshold = 100  # Minimum new samples to trigger retraining
        self.performance_threshold = 0.65  # Minimum accuracy to keep model
        
    def check_retrain_need(self, new_data_count: int, current_performance: float) -> bool:
        """Check if retraining is needed"""
        if new_data_count >= self.retrain_threshold:
            return True
        
        if current_performance < self.performance_threshold:
            return True
        
        # Weekly retraining
        last_retrain = self.get_last_retrain_date()
        if last_retrain and (datetime.now() - last_retrain).days >= 7:
            return True
        
        return False
    
    def get_last_retrain_date(self) -> Optional[datetime]:
        """Get date of last retraining"""
        models_dir = Path("./data/models")
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5"))
        
        if model_files:
            latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
            timestamp_str = latest_file.stem.split('_')[-1]
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        
        return None
    
    def monitor_performance(self, predictions: List[Dict], actual_results: List[Dict]) -> Dict:
        """Monitor model performance and collect data for retraining"""
        performance_metrics = {
            'total_predictions': len(predictions),
            'correct_predictions': 0,
            'accuracy': 0,
            'by_league': {},
            'by_confidence': {}
        }
        
        training_data = []
        
        for pred, actual in zip(predictions, actual_results):
            if actual.get('result'):
                is_correct = (pred['prediction']['winner'] == actual['result'])
                performance_metrics['correct_predictions'] += int(is_correct)
                
                # Collect data for retraining
                training_sample = self.create_training_sample(pred, actual)
                training_data.append(training_sample)
                
                # Track by league
                league = pred.get('league', 'unknown')
                if league not in performance_metrics['by_league']:
                    performance_metrics['by_league'][league] = {'total': 0, 'correct': 0}
                performance_metrics['by_league'][league]['total'] += 1
                performance_metrics['by_league'][league]['correct'] += int(is_correct)
                
                # Track by confidence level
                confidence = pred['prediction']['confidence']
                conf_level = int(confidence * 10) / 10  # Round to 0.1
                if conf_level not in performance_metrics['by_confidence']:
                    performance_metrics['by_confidence'][conf_level] = {'total': 0, 'correct': 0}
                performance_metrics['by_confidence'][conf_level]['total'] += 1
                performance_metrics['by_confidence'][conf_level]['correct'] += int(is_correct)
        
        if performance_metrics['total_predictions'] > 0:
            performance_metrics['accuracy'] = (
                performance_metrics['correct_predictions'] / performance_metrics['total_predictions']
            )
        
        # Check if retraining is needed
        if self.check_retrain_need(
            len(training_data),
            performance_metrics['accuracy']
        ):
            logging.info("Retraining triggered")
            if training_data:
                self.retrain_with_new_data(training_data)
        
        return performance_metrics
    
    def create_training_sample(self, prediction: Dict, actual: Dict) -> Dict:
        """Create training sample from prediction and actual result"""
        # This should extract features from the prediction context
        # and combine with the actual outcome
        return {
            'features': prediction.get('features', {}),
            'outcome': actual['result'],
            'league': prediction.get('league'),
            'date': prediction.get('date')
        }
    
    def retrain_with_new_data(self, new_samples: List[Dict]):
        """Retrain models with new samples"""
        # Convert to DataFrame
        df = pd.DataFrame(new_samples)
        
        # Retrain
        self.trainer.retrain(df)
        
        logging.info(f"Models retrained with {len(new_samples)} new samples")
