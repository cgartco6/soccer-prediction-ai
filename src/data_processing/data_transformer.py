"""
Data transformation and normalization with hardware optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
from pathlib import Path
from ..system.optimizer import SystemOptimizer

class DataTransformer:
    """Transform and normalize data for machine learning"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup transformation strategy
        self.setup_transformation_strategy()
        
        # Transformers storage
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.pca = None
        
        # Cache directory
        self.cache_dir = Path("./data/models/transformers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature metadata
        self.feature_metadata = {}
    
    def setup_transformation_strategy(self):
        """Setup transformation strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.transform_config = {
                'scaling_method': 'minmax',  # minmax, standard, robust
                'handle_categorical': 'label',  # label, onehot, target
                'feature_selection': True,
                'max_features': 30,
                'use_pca': False,
                'pca_components': 0,
                'handle_imbalance': 'undersample',  # undersample, oversample, smote, none
                'outlier_handling': 'clip',  # clip, remove, ignore
                'parallel_processing': False
            }
        elif profile == 'mid_end':
            self.transform_config = {
                'scaling_method': 'standard',
                'handle_categorical': 'onehot',
                'feature_selection': True,
                'max_features': 50,
                'use_pca': True,
                'pca_components': 20,
                'handle_imbalance': 'smote',
                'outlier_handling': 'clip',
                'parallel_processing': True
            }
        else:  # high_end
            self.transform_config = {
                'scaling_method': 'robust',
                'handle_categorical': 'onehot',
                'feature_selection': True,
                'max_features': 100,
                'use_pca': True,
                'pca_components': 50,
                'handle_imbalance': 'smote',
                'outlier_handling': 'remove',
                'parallel_processing': True
            }
        
        self.logger.info(f"Data transformation strategy: {self.transform_config}")
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'result') -> Tuple[pd.DataFrame, Dict]:
        """Fit transformers and transform data"""
        if df.empty:
            return df, {}
        
        self.logger.info(f"Transforming data: {len(df)} rows, {len(df.columns)} columns")
        
        # Create a copy
        X = df.copy()
        
        # Store original info
        original_shape = X.shape
        
        # Separate target if present
        y = None
        if target_col in X.columns:
            y = X[target_col].copy()
            X = X.drop(columns=[target_col])
        
        # 1. Handle missing values
        X = self._handle_missing_values(X)
        
        # 2. Handle outliers
        X = self._handle_outliers(X)
        
        # 3. Encode categorical variables
        X = self._encode_categorical(X)
        
        # 4. Scale numerical features
        X = self._scale_features(X)
        
        # 5. Feature selection
        if self.transform_config['feature_selection'] and y is not None:
            X = self._select_features(X, y)
        
        # 6. Dimensionality reduction
        if self.transform_config['use_pca']:
            X = self._apply_pca(X)
        
        # 7. Handle class imbalance
        if y is not None:
            X, y = self._handle_imbalance(X, y)
        
        # Add target back if it exists
        if y is not None:
            X[target_col] = y
        
        # Store transformation metadata
        self.feature_metadata = {
            'original_shape': original_shape,
            'transformed_shape': X.shape,
            'columns_removed': original_shape[1] - X.shape[1],
            'transform_config': self.transform_config,
            'scaler_type': self.transform_config['scaling_method'],
            'encoder_type': self.transform_config['handle_categorical']
        }
        
        self.logger.info(f"Transformation complete: {X.shape[1]} features")
        
        return X, self.feature_metadata
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers"""
        if df.empty:
            return df
        
        self.logger.info(f"Transforming new data: {len(df)} rows")
        
        X = df.copy()
        
        # 1. Handle missing values
        X = self._handle_missing_values(X, is_training=False)
        
        # 2. Handle outliers
        X = self._handle_outliers(X, is_training=False)
        
        # 3. Encode categorical variables
        X = self._encode_categorical(X, is_training=False)
        
        # 4. Scale numerical features
        X = self._scale_features(X, is_training=False)
        
        # 5. Apply feature selection
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # 6. Apply PCA
        if self.pca is not None:
            X = self.pca.transform(X)
        
        self.logger.info(f"Transformation complete: {X.shape[1]} features")
        
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle missing values"""
        if df.empty:
            return df
        
        X = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            if X[col].isna().any():
                if is_training:
                    # Use median for training
                    fill_value = X[col].median()
                    # Store for later use
                    if 'numeric_fill_values' not in self.feature_metadata:
                        self.feature_metadata['numeric_fill_values'] = {}
                    self.feature_metadata['numeric_fill_values'][col] = fill_value
                else:
                    # Use stored fill value
                    fill_value = self.feature_metadata.get('numeric_fill_values', {}).get(col, 0)
                
                X[col] = X[col].fillna(fill_value)
        
        # Handle categorical columns
        for col in categorical_cols:
            if X[col].isna().any():
                if is_training:
                    # Use mode for training
                    fill_value = X[col].mode()[0] if not X[col].mode().empty else 'missing'
                    # Store for later use
                    if 'categorical_fill_values' not in self.feature_metadata:
                        self.feature_metadata['categorical_fill_values'] = {}
                    self.feature_metadata['categorical_fill_values'][col] = fill_value
                else:
                    # Use stored fill value
                    fill_value = self.feature_metadata.get('categorical_fill_values', {}).get(col, 'missing')
                
                X[col] = X[col].fillna(fill_value)
        
        return X
    
    def _handle_outliers(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle outliers in numeric features"""
        if df.empty:
            return df
        
        X = df.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.transform_config['outlier_handling'] == 'ignore' or len(numeric_cols) == 0:
            return X
        
        for col in numeric_cols:
            if X[col].nunique() <= 1:
                continue  # Skip constant columns
            
            if is_training:
                # Calculate bounds
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Store bounds
                if 'outlier_bounds' not in self.feature_metadata:
                    self.feature_metadata['outlier_bounds'] = {}
                self.feature_metadata['outlier_bounds'][col] = (lower_bound, upper_bound)
            else:
                # Use stored bounds
                bounds = self.feature_metadata.get('outlier_bounds', {}).get(col, (None, None))
                lower_bound, upper_bound = bounds
            
            if lower_bound is not None and upper_bound is not None:
                if self.transform_config['outlier_handling'] == 'clip':
                    X[col] = X[col].clip(lower_bound, upper_bound)
                elif self.transform_config['outlier_handling'] == 'remove' and is_training:
                    # Only remove outliers during training
                    mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
                    X = X[mask].copy()
        
        return X
    
    def _encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        if df.empty:
            return df
        
        X = df.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return X
        
        if self.transform_config['handle_categorical'] == 'label':
            # Label encoding
            for col in categorical_cols:
                if is_training:
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    encoder = self.encoders.get(col)
                    if encoder:
                        # Handle unseen labels
                        unseen_mask = ~X[col].isin(encoder.classes_)
                        if unseen_mask.any():
                            X.loc[unseen_mask, col] = 'unknown'
                        X[col] = encoder.transform(X[col].astype(str))
        
        elif self.transform_config['handle_categorical'] == 'onehot':
            # One-hot encoding
            if is_training:
                # For training, fit one-hot encoder
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X[categorical_cols])
                
                # Get feature names
                feature_names = encoder.get_feature_names_out(categorical_cols)
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                
                # Drop original categorical columns and add encoded ones
                X = X.drop(columns=categorical_cols)
                X = pd.concat([X, encoded_df], axis=1)
                
                self.encoders['onehot'] = encoder
            else:
                # For testing/validation, transform using fitted encoder
                encoder = self.encoders.get('onehot')
                if encoder:
                    encoded = encoder.transform(X[categorical_cols])
                    feature_names = encoder.get_feature_names_out(categorical_cols)
                    
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    
                    X = X.drop(columns=categorical_cols)
                    X = pd.concat([X, encoded_df], axis=1)
        
        return X
    
    def _scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        if df.empty:
            return df
        
        X = df.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        scaling_method = self.transform_config['scaling_method']
        
        if is_training:
            # Fit scaler
            if scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['numeric'] = scaler
        else:
            # Transform using fitted scaler
            scaler = self.scalers.get('numeric')
            if scaler:
                X[numeric_cols] = scaler.transform(X[numeric_cols])
        
        return X
    
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select important features"""
        if df.empty or len(df.columns) <= self.transform_config['max_features']:
            return df
        
        X = df.copy()
        
        # Ensure target is encoded if categorical
        if target.dtype == 'object':
            le = LabelEncoder()
            target_encoded = le.fit_transform(target)
        else:
            target_encoded = target
        
        # Use appropriate scoring function
        if len(np.unique(target_encoded)) > 2:
            # Multi-class classification
            scorer = f_classif
        else:
            # Binary classification
            scorer = mutual_info_classif
        
        # Select top features
        k = min(self.transform_config['max_features'], X.shape[1])
        
        self.feature_selector = SelectKBest(score_func=scorer, k=k)
        X_selected = self.feature_selector.fit_transform(X, target_encoded)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        selected_features = X.columns[selected_mask]
        
        # Create new DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Store feature scores
        self.feature_metadata['feature_scores'] = dict(zip(
            X.columns, 
            self.feature_selector.scores_
        ))
        self.feature_metadata['selected_features'] = list(selected_features)
        
        self.logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
        
        return X_selected_df
    
    def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        if df.empty:
            return df
        
        X = df.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        n_components = self.transform_config['pca_components']
        if n_components <= 0:
            n_components = min(50, X.shape[1])
        
        # Only apply PCA if we have enough features
        if X.shape[1] > n_components:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
            
            # Create new column names
            pca_columns = [f'pca_{i+1}' for i in range(n_components)]
            X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
            
            # Store PCA info
            self.feature_metadata['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
            self.feature_metadata['pca_components'] = n_components
            
            self.logger.info(f"PCA applied: {n_components} components explain "
                           f"{self.feature_metadata['pca_explained_variance']:.2%} of variance")
            
            return X_pca_df
        
        return X
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance"""
        if len(np.unique(y)) <= 1:
            return X, y
        
        imbalance_method = self.transform_config['handle_imbalance']
        
        if imbalance_method == 'none':
            return X, y
        
        # Check if imbalance exists
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio < 2:  # Not significantly imbalanced
            return X, y
        
        self.logger.info(f"Class imbalance detected: ratio = {imbalance_ratio:.2f}")
        
        if imbalance_method == 'undersample':
            # Undersample majority class
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            self.logger.info(f"Undersampling applied: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
        
        elif imbalance_method == 'oversample':
            # Oversample minority class
            from imblearn.over_sampling import RandomOverSampler
            
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            self.logger.info(f"Oversampling applied: {len(X)} -> {len(X_resampled)} samples")
            
            return X_resampled, y_resampled
        
        elif imbalance_method == 'smote':
            # SMOTE (Synthetic Minority Over-sampling Technique)
            try:
                from imblearn.over_sampling import SMOTE
                
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                self.logger.info(f"SMOTE applied: {len(X)} -> {len(X_resampled)} samples")
                
                return X_resampled, y_resampled
            except ImportError:
                self.logger.warning("SMOTE not available, using random oversampling")
                return self._handle_imbalance(X, y)  # Fall back to oversample
        
        return X, y
    
    def save_transformers(self, filepath: str = None):
        """Save fitted transformers to disk"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.cache_dir / f"transformers_{timestamp}.joblib"
        
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'feature_metadata': self.feature_metadata,
            'transform_config': self.transform_config
        }
        
        try:
            joblib.dump(transformers, filepath)
            self.logger.info(f"Transformers saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save transformers: {e}")
    
    def load_transformers(self, filepath: str):
        """Load transformers from disk"""
        try:
            transformers = joblib.load(filepath)
            
            self.scalers = transformers.get('scalers', {})
            self.encoders = transformers.get('encoders', {})
            self.feature_selector = transformers.get('feature_selector')
            self.pca = transformers.get('pca')
            self.feature_metadata = transformers.get('feature_metadata', {})
            self.transform_config = transformers.get('transform_config', self.transform_config)
            
            self.logger.info(f"Transformers loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load transformers: {e}")
    
    def get_transformation_report(self) -> Dict:
        """Generate transformation report"""
        report = {
            'transformation_config': self.transform_config,
            'feature_metadata': self.feature_metadata,
            'transformers_available': {
                'scalers': list(self.scalers.keys()),
                'encoders': list(self.encoders.keys()),
                'has_feature_selector': self.feature_selector is not None,
                'has_pca': self.pca is not None
            }
        }
        
        if self.feature_metadata:
            report.update({
                'original_features': self.feature_metadata.get('original_shape', (0, 0))[1],
                'final_features': self.feature_metadata.get('transformed_shape', (0, 0))[1],
                'features_removed': self.feature_metadata.get('columns_removed', 0),
                'pca_variance_explained': self.feature_metadata.get('pca_explained_variance', 0)
            })
        
        return report
