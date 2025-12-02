"""
Data loading and batching with hardware optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Generator
import logging
import json
from pathlib import Path
import pickle
import lz4.frame
import gc
from ..system.optimizer import SystemOptimizer

class DataLoader:
    """Load and batch data with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup loading strategy
        self.setup_loading_strategy()
        
        # Data directories
        self.data_dir = Path("./data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        # Ensure directories exist
        for directory in [self.raw_dir, self.processed_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Data cache
        self.data_cache = {}
        self.cache_max_size = 100  # Maximum number of datasets to cache
        
    def setup_loading_strategy(self):
        """Setup data loading strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.loading_config = {
                'max_memory_mb': 1024,  # 1GB max memory usage
                'chunk_size': 1000,
                'use_compression': True,
                'cache_enabled': True,
                'cache_max_age_hours': 24,
                'parallel_loading': False,
                'prefetch_batches': 1
            }
        elif profile == 'mid_end':
            self.loading_config = {
                'max_memory_mb': 4096,  # 4GB max memory usage
                'chunk_size': 5000,
                'use_compression': True,
                'cache_enabled': True,
                'cache_max_age_hours': 12,
                'parallel_loading': True,
                'prefetch_batches': 2
            }
        else:  # high_end
            self.loading_config = {
                'max_memory_mb': 16384,  # 16GB max memory usage
                'chunk_size': 20000,
                'use_compression': False,
                'cache_enabled': True,
                'cache_max_age_hours': 6,
                'parallel_loading': True,
                'prefetch_batches': 4
            }
        
        self.logger.info(f"Data loading strategy: {self.loading_config}")
    
    def load_fixtures(self, date: str = None, source: str = 'all') -> List[Dict]:
        """Load fixtures from files"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        # Check cache first
        cache_key = f"fixtures_{date}_{source}"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            self.logger.info(f"Loaded {len(cached_data)} fixtures from cache")
            return cached_data
        
        fixtures = []
        
        # Look for fixture files
        if source == 'all' or source == 'raw':
            # Look for raw fixture files
            pattern = f"fixtures_{date}*.json"
            raw_files = list(self.raw_dir.glob(pattern))
            
            # Also check for compressed files
            pattern_lz4 = f"fixtures_{date}*.json.lz4"
            raw_files.extend(list(self.raw_dir.glob(pattern_lz4)))
            
            for filepath in raw_files:
                try:
                    file_fixtures = self._load_file(filepath)
                    fixtures.extend(file_fixtures)
                except Exception as e:
                    self.logger.warning(f"Failed to load {filepath}: {e}")
        
        if source == 'all' or source == 'processed':
            # Look for processed fixture files
            pattern = f"processed_fixtures_{date}*.pkl"
            processed_files = list(self.processed_dir.glob(pattern))
            
            pattern_lz4 = f"processed_fixtures_{date}*.pkl.lz4"
            processed_files.extend(list(self.processed_dir.glob(pattern_lz4)))
            
            for filepath in processed_files:
                try:
                    file_fixtures = self._load_file(filepath)
                    fixtures.extend(file_fixtures)
                except Exception as e:
                    self.logger.warning(f"Failed to load {filepath}: {e}")
        
        # Remove duplicates
        if fixtures:
            fixtures = self._deduplicate_fixtures(fixtures)
            
            # Cache results
            self._save_to_cache(cache_key, fixtures)
            
            self.logger.info(f"Loaded {len(fixtures)} fixtures from files")
        
        return fixtures
    
    def _load_file(self, filepath: Path) -> List[Dict]:
        """Load data from file"""
        if not filepath.exists():
            return []
        
        try:
            if filepath.suffix == '.lz4':
                with lz4.frame.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            elif filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                self.logger.warning(f"Unsupported file format: {filepath.suffix}")
                return []
            
            # Extract fixtures from data structure
            if isinstance(data, dict) and 'fixtures' in data:
                return data['fixtures']
            elif isinstance(data, list):
                return data
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to load file {filepath}: {e}")
            return []
    
    def _deduplicate_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures"""
        if not fixtures:
            return []
        
        seen = set()
        unique_fixtures = []
        
        for fixture in fixtures:
            # Create unique key
            home_team = fixture.get('home_team', '').lower().strip()
            away_team = fixture.get('away_team', '').lower().strip()
            date_str = fixture.get('date', '').split('T')[0] if fixture.get('date') else ''
            
            key = f"{home_team}_{away_team}_{date_str}"
            
            if key not in seen:
                seen.add(key)
                unique_fixtures.append(fixture)
        
        return unique_fixtures
    
    def load_historical_data(self, days_back: int = 365, leagues: List[str] = None) -> pd.DataFrame:
        """Load historical match data"""
        # Check cache
        cache_key = f"historical_{days_back}_{'_'.join(leagues or [])}"
        cached_data = self._load_from_cache(cache_key, as_dataframe=True)
        
        if cached_data is not None:
            self.logger.info(f"Loaded historical data from cache: {len(cached_data)} rows")
            return cached_data
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Look for historical data files
        historical_files = []
        
        # Check for pre-processed historical data
        historical_db = self.processed_dir / "historical_data.pkl"
        if historical_db.exists():
            historical_files.append(historical_db)
        
        # Also check daily files
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            pattern = f"processed_fixtures_{date_str}*.pkl"
            daily_files = list(self.processed_dir.glob(pattern))
            historical_files.extend(daily_files)
            
            pattern_lz4 = f"processed_fixtures_{date_str}*.pkl.lz4"
            daily_files_lz4 = list(self.processed_dir.glob(pattern_lz4))
            historical_files.extend(daily_files_lz4)
            
            current_date += timedelta(days=1)
        
        # Load and combine data
        all_data = []
        
        for filepath in historical_files:
            try:
                data = self._load_historical_file(filepath, leagues)
                if data is not None and not data.empty:
                    all_data.append(data)
                    
                    # Memory management for low-end hardware
                    if self.optimizer.hardware_info.get('hardware_profile') == 'low_end':
                        if len(all_data) % 10 == 0:
                            gc.collect()
                            
            except Exception as e:
                self.logger.warning(f"Failed to load historical file {filepath}: {e}")
        
        # Combine all data
        if all_data:
            historical_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by date
            if 'date' in historical_df.columns:
                historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
                historical_df = historical_df[
                    (historical_df['date'] >= start_date) & 
                    (historical_df['date'] <= end_date)
                ]
            
            # Cache results
            self._save_to_cache(cache_key, historical_df)
            
            self.logger.info(f"Loaded historical data: {len(historical_df)} rows")
            
            return historical_df
        
        self.logger.warning("No historical data found")
        return pd.DataFrame()
    
    def _load_historical_file(self, filepath: Path, leagues: List[str] = None) -> Optional[pd.DataFrame]:
        """Load historical data from file"""
        if not filepath.exists():
            return None
        
        try:
            # Load data
            if filepath.suffix == '.lz4':
                with lz4.frame.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'fixtures' in data:
                df = pd.DataFrame(data['fixtures'])
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return None
            
            # Filter by leagues if specified
            if leagues and 'league' in df.columns:
                df = df[df['league'].isin(leagues)]
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to load historical file {filepath}: {e}")
            return None
    
    def load_training_data(self, test_size: float = 0.2, 
                          val_size: float = 0.1,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split training data"""
        # Load historical data
        historical_df = self.load_historical_data(days_back=365)
        
        if historical_df.empty:
            self.logger.error("No training data available")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Prepare features and target
        X, y = self._prepare_training_data(historical_df)
        
        if X.empty or y.empty:
            self.logger.error("Failed to prepare training data")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: training + validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Second split: training vs validation
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        self.logger.info(f"Training data split: "
                        f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Make a copy
        data = df.copy()
        
        # Ensure required columns
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns in training data: {missing_cols}")
            
            # Try to create missing columns
            if 'home_score' not in data.columns and 'score' in data.columns:
                # Extract from score dict
                if isinstance(data['score'].iloc[0], dict):
                    data['home_score'] = data['score'].apply(lambda x: x.get('home', 0))
                    data['away_score'] = data['score'].apply(lambda x: x.get('away', 0))
        
        # Create target variable
        if all(col in data.columns for col in ['home_score', 'away_score']):
            # Result: 0=Home Win, 1=Draw, 2=Away Win
            conditions = [
                data['home_score'] > data['away_score'],  # Home win
                data['home_score'] == data['away_score'],  # Draw
                data['home_score'] < data['away_score']   # Away win
            ]
            choices = [0, 1, 2]
            
            y = pd.Series(np.select(conditions, choices, default=1), name='result')
        else:
            self.logger.error("Cannot create target variable: missing score columns")
            return pd.DataFrame(), pd.Series()
        
        # Prepare features
        feature_cols = [
            'home_team', 'away_team', 'league', 'date',
            'home_form', 'away_form', 'odds', 'weather',
            'venue', 'home_position', 'away_position'
        ]
        
        # Select available feature columns
        available_cols = [col for col in feature_cols if col in data.columns]
        X = data[available_cols].copy()
        
        # Add derived features
        X = self._add_training_features(X, data)
        
        return X, y
    
    def _add_training_features(self, X: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for training"""
        # Make a copy
        features = X.copy()
        
        # Date features
        if 'date' in features.columns:
            try:
                features['date'] = pd.to_datetime(features['date'], errors='coerce')
                features['match_day_of_week'] = features['date'].dt.dayofweek
                features['match_month'] = features['date'].dt.month
                features['match_year'] = features['date'].dt.year
                features = features.drop(columns=['date'])
            except:
                pass
        
        # Form features (simplified)
        if 'home_form' in features.columns:
            # Convert form string to points
            def form_to_points(form_str):
                if isinstance(form_str, str):
                    points = {'W': 3, 'D': 1, 'L': 0}
                    return sum(points.get(char, 0) for char in form_str[-5:])
                return 0
            
            features['home_form_points'] = features['home_form'].apply(form_to_points)
            features['away_form_points'] = features['away_form'].apply(form_to_points)
            features = features.drop(columns=['home_form', 'away_form'])
        
        # Odds features (simplified)
        if 'odds' in features.columns:
            # Extract home odds if available
            def extract_home_odds(odds):
                if isinstance(odds, dict):
                    return odds.get('home', 2.0)
                return 2.0
            
            features['home_odds'] = features['odds'].apply(extract_home_odds)
            features = features.drop(columns=['odds'])
        
        # Team encoding
        if 'home_team' in features.columns and 'away_team' in features.columns:
            # Simple team encoding (frequency based)
            team_freq = pd.concat([features['home_team'], features['away_team']]).value_counts()
            team_encoding = {team: i for i, (team, _) in enumerate(team_freq.items())}
            
            features['home_team_encoded'] = features['home_team'].map(team_encoding).fillna(-1)
            features['away_team_encoded'] = features['away_team'].map(team_encoding).fillna(-1)
            features = features.drop(columns=['home_team', 'away_team'])
        
        # League encoding
        if 'league' in features.columns:
            league_encoding = {league: i for i, league in enumerate(features['league'].unique())}
            features['league_encoded'] = features['league'].map(league_encoding).fillna(-1)
            features = features.drop(columns=['league'])
        
        return features
    
    def create_data_generator(self, X: pd.DataFrame, y: pd.Series, 
                            batch_size: int = None) -> Generator:
        """Create a data generator for batch processing"""
        if X.empty or y.empty:
            self.logger.error
