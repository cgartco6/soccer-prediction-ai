"""
Feature engineering with hardware optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import hashlib
from pathlib import Path
import pickle
import lz4.frame
from ..system.optimizer import SystemOptimizer

class FeatureEngineer:
    """Engineer features for match prediction with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup feature engineering strategy
        self.setup_feature_strategy()
        
        # Feature cache for performance
        self.feature_cache = {}
        self.cache_dir = Path("./data/cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature definitions
        self.feature_groups = self.define_feature_groups()
        
        # Historical data storage for rolling features
        self.historical_data = {}
        
    def setup_feature_strategy(self):
        """Setup feature engineering strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.feature_config = {
                'max_features': 30,
                'use_advanced_features': False,
                'use_interaction_features': False,
                'use_temporal_features': True,
                'temporal_window': 5,
                'use_rolling_features': True,
                'rolling_window': 3,
                'use_cache': True,
                'cache_compression': True,
                'parallel_processing': False
            }
        elif profile == 'mid_end':
            self.feature_config = {
                'max_features': 50,
                'use_advanced_features': True,
                'use_interaction_features': True,
                'use_temporal_features': True,
                'temporal_window': 10,
                'use_rolling_features': True,
                'rolling_window': 5,
                'use_cache': True,
                'cache_compression': True,
                'parallel_processing': True
            }
        else:  # high_end
            self.feature_config = {
                'max_features': 100,
                'use_advanced_features': True,
                'use_interaction_features': True,
                'use_temporal_features': True,
                'temporal_window': 20,
                'use_rolling_features': True,
                'rolling_window': 10,
                'use_cache': True,
                'cache_compression': False,
                'parallel_processing': True
            }
        
        self.logger.info(f"Feature engineering strategy: {self.feature_config}")
    
    def define_feature_groups(self) -> Dict[str, List[str]]:
        """Define feature groups for different hardware profiles"""
        return {
            'basic': [
                'home_team', 'away_team', 'league', 'match_date', 'is_weekend',
                'match_time_hour', 'season_month', 'is_derby'
            ],
            'form': [
                'home_form_last_5', 'away_form_last_5', 'home_form_last_10',
                'away_form_last_10', 'home_unbeaten_streak', 'away_unbeaten_streak',
                'home_win_streak', 'away_win_streak'
            ],
            'goals': [
                'home_goals_scored_avg', 'away_goals_scored_avg',
                'home_goals_conceded_avg', 'away_goals_conceded_avg',
                'home_goal_difference', 'away_goal_difference',
                'total_goals_avg', 'both_teams_to_score_rate'
            ],
            'h2h': [
                'h2h_total_matches', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
                'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
                'h2h_goals_avg', 'h2h_home_goals_avg', 'h2h_away_goals_avg'
            ],
            'odds': [
                'odds_home', 'odds_draw', 'odds_away',
                'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
                'bookmaker_margin', 'value_home', 'value_draw', 'value_away'
            ],
            'advanced': [
                'home_expected_goals', 'away_expected_goals',
                'home_possession_avg', 'away_possession_avg',
                'home_shots_avg', 'away_shots_avg',
                'home_shots_on_target_avg', 'away_shots_on_target_avg',
                'home_corners_avg', 'away_corners_avg',
                'home_fouls_avg', 'away_fouls_avg'
            ],
            'weather': [
                'temperature', 'humidity', 'wind_speed', 'precipitation',
                'is_rain', 'is_snow', 'is_extreme_weather'
            ],
            'venue': [
                'venue_capacity', 'is_home_stadium', 'travel_distance_km',
                'altitude_difference', 'timezone_difference'
            ],
            'team_strength': [
                'home_team_rating', 'away_team_rating', 'team_rating_difference',
                'home_league_position', 'away_league_position', 'position_difference',
                'home_points_per_game', 'away_points_per_game'
            ],
            'motivation': [
                'home_relegation_risk', 'away_relegation_risk',
                'home_european_qualification', 'away_european_qualification',
                'home_title_race', 'away_title_race',
                'is_cup_match', 'is_derby_match'
            ],
            'injuries': [
                'home_injured_players', 'away_injured_players',
                'home_suspended_players', 'away_suspended_players',
                'home_key_player_missing', 'away_key_player_missing',
                'injury_impact_home', 'injury_impact_away'
            ],
            'fatigue': [
                'home_days_since_last_match', 'away_days_since_last_match',
                'home_matches_last_30_days', 'away_matches_last_30_days',
                'home_travel_last_7_days', 'away_travel_last_7_days'
            ]
        }
    
    def create_features(self, fixture: Dict, historical_data: pd.DataFrame = None) -> Dict:
        """Create features for a single fixture"""
        try:
            # Check cache first
            cache_key = self._get_feature_cache_key(fixture)
            cached_features = self._load_from_cache(cache_key)
            
            if cached_features is not None:
                self.logger.debug(f"Using cached features for {fixture.get('home_team')} vs {fixture.get('away_team')}")
                return cached_features
            
            features = {}
            
            # 1. Basic features
            features.update(self._create_basic_features(fixture))
            
            # 2. Form features (requires historical data)
            if historical_data is not None:
                features.update(self._create_form_features(fixture, historical_data))
            
            # 3. H2H features (requires historical data)
            if historical_data is not None:
                features.update(self._create_h2h_features(fixture, historical_data))
            
            # 4. Odds features
            features.update(self._create_odds_features(fixture))
            
            # 5. Weather features
            features.update(self._create_weather_features(fixture))
            
            # 6. Venue features
            features.update(self._create_venue_features(fixture))
            
            # 7. Team strength features
            features.update(self._create_team_strength_features(fixture))
            
            # 8. Motivation features
            features.update(self._create_motivation_features(fixture))
            
            # 9. Injury features
            features.update(self._create_injury_features(fixture))
            
            # 10. Fatigue features
            features.update(self._create_fatigue_features(fixture))
            
            # 11. Advanced features (if enabled)
            if self.feature_config['use_advanced_features']:
                if historical_data is not None:
                    features.update(self._create_advanced_features(fixture, historical_data))
            
            # 12. Interaction features (if enabled)
            if self.feature_config['use_interaction_features']:
                features.update(self._create_interaction_features(features))
            
            # Add metadata
            features['feature_generation_time'] = datetime.now().isoformat()
            features['feature_count'] = len(features)
            
            # Cache features
            self._save_to_cache(cache_key, features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {e}")
            return {}
    
    def _get_feature_cache_key(self, fixture: Dict) -> str:
        """Generate cache key for features"""
        cache_data = {
            'home_team': fixture.get('home_team', ''),
            'away_team': fixture.get('away_team', ''),
            'date': fixture.get('date', ''),
            'feature_config': self.feature_config
        }
        
        cache_str = str(cache_data)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load features from cache"""
        if not self.feature_config['use_cache']:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.cache"
        
        if not cache_path.exists():
            return None
        
        # Check cache age
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        max_age = 3600  # 1 hour
        
        if cache_age > max_age:
            try:
                cache_path.unlink()
            except:
                pass
            return None
        
        try:
            if self.feature_config['cache_compression']:
                with lz4.frame.open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_key: str, features: Dict):
        """Save features to cache"""
        if not self.feature_config['use_cache']:
            return
        
        cache_path = self.cache_dir / f"{cache_key}.cache"
        
        try:
            if self.feature_config['cache_compression']:
                with lz4.frame.open(cache_path, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.warning(f"Failed to save features to cache: {e}")
    
    def _create_basic_features(self, fixture: Dict) -> Dict:
        """Create basic features"""
        features = {}
        
        # Team names (encoded)
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        
        if home_team and away_team:
            features['home_team_hash'] = self._hash_string(home_team)
            features['away_team_hash'] = self._hash_string(away_team)
        
        # League (encoded)
        league = fixture.get('league', '')
        if league:
            features['league_hash'] = self._hash_string(league)
        
        # Date features
        date_str = fixture.get('date', '')
        if date_str:
            try:
                match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                features['match_day_of_week'] = match_date.weekday()
                features['match_hour'] = match_date.hour
                features['match_month'] = match_date.month
                features['is_weekend'] = 1 if match_date.weekday() >= 5 else 0
                features['season_period'] = self._get_season_period(match_date.month)
                
                # Time until match
                now = datetime.now()
                if match_date > now:
                    features['hours_until_match'] = (match_date - now).total_seconds() / 3600
                else:
                    features['hours_until_match'] = 0
            except:
                pass
        
        # Derby match
        features['is_derby'] = self._is_derby_match(home_team, away_team, league)
        
        return features
    
    def _hash_string(self, text: str) -> int:
        """Hash string to integer"""
        if not text:
            return 0
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 10000
    
    def _get_season_period(self, month: int) -> int:
        """Get season period (0: early, 1: mid, 2: late)"""
        if month in [8, 9, 10]:  # Aug-Oct
            return 0  # Early season
        elif month in [11, 12, 1, 2]:  # Nov-Feb
            return 1  # Mid season
        else:  # Mar-Jul
            return 2  # Late season
    
    def _is_derby_match(self, home_team: str, away_team: str, league: str) -> int:
        """Check if match is a derby"""
        if not home_team or not away_team:
            return 0
        
        # Common derby patterns
        derby_patterns = [
            # City derbies
            ('Manchester', 'Manchester'),
            ('London', 'London'),
            ('Milan', 'Milan'),
            ('Madrid', 'Madrid'),
            ('Munich', 'Munich'),
            
            # Rivalries
            ('Barcelona', 'Real Madrid'),
            ('Liverpool', 'Manchester United'),
            ('Celtic', 'Rangers'),
            ('Boca Juniors', 'River Plate'),
            ('Fenerbahce', 'Galatasaray')
        ]
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        for pattern1, pattern2 in derby_patterns:
            if (pattern1.lower() in home_lower and pattern2.lower() in away_lower) or \
               (pattern2.lower() in home_lower and pattern1.lower() in away_lower):
                return 1
        
        return 0
    
    def _create_form_features(self, fixture: Dict, historical_data: pd.DataFrame) -> Dict:
        """Create form-based features"""
        features = {}
        
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        match_date = self._parse_date(fixture.get('date', ''))
        
        if not home_team or not away_team or not match_date:
            return features
        
        # Get historical matches for home team
        home_matches = historical_data[
            (historical_data['home_team'] == home_team) | 
            (historical_data['away_team'] == home_team)
        ].copy()
        
        # Get historical matches for away team
        away_matches = historical_data[
            (historical_data['home_team'] == away_team) | 
            (historical_data['away_team'] == away_team)
        ].copy()
        
        # Filter to matches before current match
        home_matches = home_matches[home_matches['date'] < match_date]
        away_matches = away_matches[away_matches['date'] < match_date]
        
        # Sort by date
        home_matches = home_matches.sort_values('date', ascending=False)
        away_matches = away_matches.sort_values('date', ascending=False)
        
        # Calculate form for last N matches
        window = self.feature_config['temporal_window']
        
        if len(home_matches) >= window:
            features.update(self._calculate_team_form(home_matches.head(window), home_team, 'home'))
        
        if len(away_matches) >= window:
            features.update(self._calculate_team_form(away_matches.head(window), away_team, 'away'))
        
        # Calculate rolling averages
        if self.feature_config['use_rolling_features']:
            rolling_window = self.feature_config['rolling_window']
            
            if len(home_matches) >= rolling_window:
                features.update(self._calculate_rolling_features(
                    home_matches.head(rolling_window), home_team, 'home'
                ))
            
            if len(away_matches) >= rolling_window:
                features.update(self._calculate_rolling_features(
                    away_matches.head(rolling_window), away_team, 'away'
                ))
        
        return features
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        try:
            # Handle Z timezone
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.fromisoformat(date_str)
        except:
            return None
    
    def _calculate_team_form(self, matches: pd.DataFrame, team_name: str, prefix: str) -> Dict:
        """Calculate team form from matches"""
        features = {}
        
        if matches.empty:
            return features
        
        # Initialize counters
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        points = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            is_away = match['away_team'] == team_name
            
            if not (is_home or is_away):
                continue
            
            if is_home:
                team_score = match.get('home_score', 0)
                opponent_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opponent_score = match.get('home_score', 0)
            
            goals_scored += team_score
            goals_conceded += opponent_score
            
            if team_score > opponent_score:
                wins += 1
                points += 3
            elif team_score == opponent_score:
                draws += 1
                points += 1
            else:
                losses += 1
        
        total_matches = wins + draws + losses
        
        if total_matches > 0:
            features[f'{prefix}_wins_last_{total_matches}'] = wins
            features[f'{prefix}_draws_last_{total_matches}'] = draws
            features[f'{prefix}_losses_last_{total_matches}'] = losses
            features[f'{prefix}_points_last_{total_matches}'] = points
            features[f'{prefix}_win_rate_last_{total_matches}'] = wins / total_matches
            features[f'{prefix}_unbeaten_rate_last_{total_matches}'] = (wins + draws) / total_matches
            features[f'{prefix}_goals_scored_avg_last_{total_matches}'] = goals_scored / total_matches
            features[f'{prefix}_goals_conceded_avg_last_{total_matches}'] = goals_conceded / total_matches
            features[f'{prefix}_goal_difference_last_{total_matches}'] = goals_scored - goals_conceded
        
        # Form string (e.g., "WWDLW")
        form_string = ''
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            is_away = match['away_team'] == team_name
            
            if not (is_home or is_away):
                continue
            
            if is_home:
                team_score = match.get('home_score', 0)
                opponent_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opponent_score = match.get('home_score', 0)
            
            if team_score > opponent_score:
                form_string += 'W'
            elif team_score == opponent_score:
                form_string += 'D'
            else:
                form_string += 'L'
        
        if form_string:
            features[f'{prefix}_form_string'] = form_string
            
            # Calculate form momentum
            momentum = 0
            for i, result in enumerate(form_string):
                weight = 1.0 / (i + 1)  # Recent matches weighted more
                if result == 'W':
                    momentum += weight
                elif result == 'D':
                    momentum += weight * 0.5
                # Losses add 0
            
            features[f'{prefix}_form_momentum'] = momentum
        
        return features
    
    def _calculate_rolling_features(self, matches: pd.DataFrame, team_name: str, prefix: str) -> Dict:
        """Calculate rolling features"""
        features = {}
        
        if matches.empty:
            return features
        
        # Calculate rolling averages
        goals_scored_list = []
        goals_conceded_list = []
        points_list = []
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team_name
            is_away = match['away_team'] == team_name
            
            if not (is_home or is_away):
                continue
            
            if is_home:
                team_score = match.get('home_score', 0)
                opponent_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opponent_score = match.get('home_score', 0)
            
            goals_scored_list.append(team_score)
            goals_conceded_list.append(opponent_score)
            
            if team_score > opponent_score:
                points_list.append(3)
            elif team_score == opponent_score:
                points_list.append(1)
            else:
                points_list.append(0)
        
        if goals_scored_list:
            features[f'{prefix}_goals_scored_rolling_avg'] = np.mean(goals_scored_list)
            features[f'{prefix}_goals_scored_rolling_std'] = np.std(goals_scored_list)
        
        if goals_conceded_list:
            features[f'{prefix}_goals_conceded_rolling_avg'] = np.mean(goals_conceded_list)
            features[f'{prefix}_goals_conceded_rolling_std'] = np.std(goals_conceded_list)
        
        if points_list:
            features[f'{prefix}_points_rolling_avg'] = np.mean(points_list)
        
        return features
    
    def _create_h2h_features(self, fixture: Dict, historical_data: pd.DataFrame) -> Dict:
        """Create head-to-head features"""
        features = {}
        
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        match_date = self._parse_date(fixture.get('date', ''))
        
        if not home_team or not away_team or not match_date:
            return features
        
        # Get H2H matches
        h2h_matches = historical_data[
            ((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
            ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))
        ].copy()
        
        # Filter to matches before current match
        h2h_matches = h2h_matches[h2h_matches['date'] < match_date]
        
        if h2h_matches.empty:
            return features
        
        # Sort by date
        h2h_matches = h2h_matches.sort_values('date', ascending=False)
        
        # Calculate H2H statistics
        total_matches = len(h2h_matches)
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0
        
        for _, match in h2h_matches.iterrows():
            is_home_at_home = match['home_team'] == home_team
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if is_home_at_home:
                home_goals += home_score
                away_goals += away_score
                
                if home_score > away_score:
                    home_wins += 1
                elif home_score < away_score:
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals += away_score  # Home team was away
                away_goals += home_score  # Away team was home
                
                if away_score > home_score:
                    home_wins += 1
                elif away_score < home_score:
                    away_wins += 1
                else:
                    draws += 1
        
        # Calculate features
        features['h2h_total_matches'] = total_matches
        features['h2h_home_wins'] = home_wins
        features['h2h_away_wins'] = away_wins
        features['h2h_draws'] = draws
        
        if total_matches > 0:
            features['h2h_home_win_rate'] = home_wins / total_matches
            features['h2h_away_win_rate'] = away_wins / total_matches
            features['h2h_draw_rate'] = draws / total_matches
            features['h2h_goals_avg'] = (home_goals + away_goals) / total_matches
            features['h2h_home_goals_avg'] = home_goals / total_matches
            features['h2h_away_goals_avg'] = away_goals / total_matches
            features['h2h_goal_difference'] = home_goals - away_goals
            
            # Recent form in H2H (last 5 matches)
            recent_matches = min(5, total_matches)
            recent_home_wins = 0
            recent_draws = 0
            
            for i in range(recent_matches):
                match = h2h_matches.iloc[i]
                is_home_at_home = match['home_team'] == home_team
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                if is_home_at_home:
                    if home_score > away_score:
                        recent_home_wins += 1
                    elif home_score == away_score:
                        recent_draws += 1
                else:
                    if away_score > home_score:
                        recent_home_wins += 1
                    elif away_score == home_score:
                        recent_draws += 1
            
            features['h2h_home_win_rate_last_5'] = recent_home_wins / recent_matches
            features['h2h_unbeaten_rate_last_5'] = (recent_home_wins + recent_draws) / recent_matches
        
        return features
    
    def _create_odds_features(self, fixture: Dict) -> Dict:
        """Create features from odds data"""
        features = {}
        
        # Get odds
        odds = fixture.get('odds', {})
        if not odds:
            return features
        
        # Extract 1X2 odds
        home_odds = odds.get('home', 0)
        draw_odds = odds.get('draw', 0)
        away_odds = odds.get('away', 0)
        
        if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
            return features
        
        # Basic odds features
        features['odds_home'] = home_odds
        features['odds_draw'] = draw_odds
        features['odds_away'] = away_odds
        
        # Calculate implied probabilities
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        
        total_prob = home_prob + draw_prob + away_prob
        
        if total_prob > 0:
            features['implied_prob_home'] = home_prob / total_prob
            features['implied_prob_draw'] = draw_prob / total_prob
            features['implied_prob_away'] = away_prob / total_prob
            
            # Bookmaker margin
            features['bookmaker_margin'] = (total_prob - 1) * 100
            
            # Favorite indicator
            max_prob = max(features['implied_prob_home'], 
                          features['implied_prob_draw'], 
                          features['implied_prob_away'])
            
            if max_prob == features['implied_prob_home']:
                features['favorite'] = 1  # Home favorite
            elif max_prob == features['implied_prob_away']:
                features['favorite'] = 2  # Away favorite
            else:
                features['favorite'] = 0  # Draw favorite
            
            # Odds difference
            features['odds_difference_home_away'] = home_odds - away_odds
            features['odds_ratio_home_away'] = home_odds / away_odds if away_odds > 0 else 0
        
        # Value bets (if we have predicted probabilities)
        predicted_probs = fixture.get('predicted_probabilities', {})
        if predicted_probs:
            pred_home = predicted_probs.get('home', 0)
            pred_draw = predicted_probs.get('draw', 0)
            pred_away = predicted_probs.get('away', 0)
            
            if pred_home > 0 and home_odds > 0:
                features['value_home'] = (pred_home * home_odds - 1) / (home_odds - 1) if home_odds > 1 else 0
            
            if pred_draw > 0 and draw_odds > 0:
                features['value_draw'] = (pred_draw * draw_odds - 1) / (draw_odds - 1) if draw_odds > 1 else 0
            
            if pred_away > 0 and away_odds > 0:
                features['value_away'] = (pred_away * away_odds - 1) / (away_odds - 1) if away_odds > 1 else 0
        
        return features
    
    def _create_weather_features(self, fixture: Dict) -> Dict:
        """Create features from weather data"""
        features = {}
        
        weather = fixture.get('weather', {})
        if not weather:
            return features
        
        # Temperature
        if 'temperature_c' in weather:
            temp = weather['temperature_c']
            features['temperature'] = temp
            features['is_hot'] = 1 if temp > 30 else 0
            features['is_cold'] = 1 if temp < 5 else 0
        
        # Humidity
        if 'humidity_percent' in weather:
            humidity = weather['humidity_percent']
            features['humidity'] = humidity
            features['is_humid'] = 1 if humidity > 80 else 0
            features['is_dry'] = 1 if humidity < 30 else 0
        
        # Wind
        if 'wind_speed_mps' in weather:
            wind_speed = weather['wind_speed_mps']
            features['wind_speed'] = wind_speed
            features['is_windy'] = 1 if wind_speed > 10 else 0
        
        # Precipitation
        if 'precipitation_mm' in weather:
            precipitation = weather['precipitation_mm']
            features['precipitation'] = precipitation
            features['is_rain'] = 1 if precipitation > 0 else 0
            features['is_heavy_rain'] = 1 if precipitation > 10 else 0
        
        # Conditions
        if 'conditions' in weather:
            conditions = str(weather['conditions']).lower()
            features['is_clear'] = 1 if 'clear' in conditions else 0
            features['is_cloudy'] = 1 if 'cloud' in conditions else 0
            features['is_fog'] = 1 if 'fog' in conditions else 0
            features['is_snow'] = 1 if 'snow' in conditions else 0
        
        # Extreme weather
        features['is_extreme_weather'] = int(
            features.get('is_heavy_rain', 0) == 1 or
            features.get('is_snow', 0) == 1 or
            (features.get('wind_speed', 0) > 20) or
            (features.get('temperature', 20) > 35) or
            (features.get('temperature', 20) < -5)
        )
        
        return features
    
    def _create_venue_features(self, fixture: Dict) -> Dict:
        """Create features from venue data"""
        features = {}
        
        venue = fixture.get('venue', {})
        if not venue:
            return features
        
        # Capacity
        if 'capacity' in venue:
            try:
                capacity = int(venue['capacity'])
                features['venue_capacity'] = capacity
                features['is_large_stadium'] = 1 if capacity > 50000 else 0
                features['is_small_stadium'] = 1 if capacity < 10000 else 0
            except:
                pass
        
        # City
        if 'city' in venue:
            city = venue['city']
            features['venue_city_hash'] = self._hash_string(city)
        
        # Surface
        if 'surface' in venue:
            surface = venue['surface'].lower()
            features['is_grass'] = 1 if 'grass' in surface else 0
            features['is_artificial'] = 1 if 'artificial' in surface or 'turf' in surface else 0
        
        # Home advantage
        home_team = fixture.get('home_team', '')
        if home_team and 'city' in venue:
            home_city = home_team.split()[-1]  # Simple extraction
            venue_city = venue['city']
            features['is_home_stadium'] = 1 if home_city.lower() in venue_city.lower() else 0
        
        return features
    
    def _create_team_strength_features(self, fixture: Dict) -> Dict:
        """Create team strength features"""
        features = {}
        
        # Team ratings (if available)
        home_rating = fixture.get('home_team_rating', 0)
        away_rating = fixture.get('away_team_rating', 0)
        
        if home_rating > 0 and away_rating > 0:
            features['home_team_rating'] = home_rating
            features['away_team_rating'] = away_rating
            features['team_rating_difference'] = home_rating - away_rating
            features['team_rating_ratio'] = home_rating / away_rating if away_rating > 0 else 0
        
        # League positions (if available)
        home_position = fixture.get('home_league_position', 0)
        away_position = fixture.get('away_league_position', 0)
        total_teams = fixture.get('total_teams_in_league', 20)
        
        if home_position > 0 and away_position > 0 and total_teams > 0:
            features['home_league_position'] = home_position
            features['away_league_position'] = away_position
            features['position_difference'] = home_position - away_position
            
            # Normalized positions (0-1, where 1 is top)
            features['home_position_normalized'] = 1 - (home_position / total_teams)
            features['away_position_normalized'] = 1 - (away_position / total_teams)
            
            # Position categories
            features['home_is_top_team'] = 1 if home_position <= 3 else 0
            features['away_is_top_team'] = 1 if away_position <= 3 else 0
            features['home_is_bottom_team'] = 1 if home_position >= total_teams - 3 else 0
            features['away_is_bottom_team'] = 1 if away_position >= total_teams - 3 else 0
        
        # Points per game (if available)
        home_ppg = fixture.get('home_points_per_game', 0)
        away_ppg = fixture.get('away_points_per_game', 0)
        
        if home_ppg > 0 and away_ppg > 0:
            features['home_points_per_game'] = home_ppg
            features['away_points_per_game'] = away_ppg
            features['ppg_difference'] = home_ppg - away_ppg
        
        return features
    
    def _create_motivation_features(self, fixture: Dict) -> Dict:
        """Create motivation features"""
        features = {}
        
        # League context
        league = fixture.get('league', '').lower()
        
        # Check if it's a cup match
        cup_keywords = ['cup', 'champions', 'europa', 'league', 'trophy', 'final']
        features['is_cup_match'] = 1 if any(keyword in league for keyword in cup_keywords) else 0
        
        # Check if it's a derby (already calculated)
        features['is_derby_match'] = fixture.get('is_derby', 0)
        
        # Relegation risk (if positions available)
        home_position = fixture.get('home_league_position', 0)
        away_position = fixture.get('away_league_position', 0)
        total_teams = fixture.get('total_teams_in_league', 20)
        
        if home_position > 0 and total_teams > 0:
            relegation_zone = total_teams - 3
            features['home_relegation_risk'] = 1 if home_position >= relegation_zone else 0
        
        if away_position > 0 and total_teams > 0:
            relegation_zone = total_teams - 3
            features['away_relegation_risk'] = 1 if away_position >= relegation_zone else 0
        
        # European qualification
        if home_position > 0:
            features['home_european_qualification'] = 1 if home_position <= 6 else 0
        
        if away_position > 0:
            features['away_european_qualification'] = 1 if away_position <= 6 else 0
        
        # Title race
        if home_position > 0:
            features['home_title_race'] = 1 if home_position <= 3 else 0
        
        if away_position > 0:
            features['away_title_race'] = 1 if away_position <= 3 else 0
        
        # Mid-table mediocrity
        if home_position > 0 and total_teams > 0:
            mid_table_start = total_teams // 3
            mid_table_end = total_teams * 2 // 3
            features['home_mid_table'] = 1 if mid_table_start <= home_position <= mid_table_end else 0
        
        if away_position > 0 and total_teams > 0:
            mid_table_start = total_teams // 3
            mid_table_end = total_teams * 2 // 3
            features['away_mid_table'] = 1 if mid_table_start <= away_position <= mid_table_end else 0
        
        return features
    
    def _create_injury_features(self, fixture: Dict) -> Dict:
        """Create injury features"""
        features = {}
        
        injuries = fixture.get('injuries', [])
        if not injuries:
            return features
        
        # Count injuries by team
        home_injuries = 0
        away_injuries = 0
        home_key_injuries = 0
        away_key_injuries = 0
        
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        
        for injury in injuries:
            player = injury.get('player', {})
            team = player.get('team', '')
            importance = player.get('importance', 0)
            
            if home_team.lower() in team.lower():
                home_injuries += 1
                if importance >= 8:  # Key player
                    home_key_injuries += 1
            elif away_team.lower() in team.lower():
                away_injuries += 1
                if importance >= 8:
                    away_key_injuries += 1
        
        features['home_injured_players'] = home_injuries
        features['away_injured_players'] = away_injuries
        features['home_key_player_missing'] = 1 if home_key_injuries > 0 else 0
        features['away_key_player_missing'] = 1 if away_key_injuries > 0 else 0
        
        # Injury impact scores
        features['injury_impact_home'] = min(1.0, home_injuries / 5)  # Max 1.0
        features['injury_impact_away'] = min(1.0, away_injuries / 5)
        
        # Total injuries
        features['total_injuries'] = home_injuries + away_injuries
        features['injury_difference'] = home_injuries - away_injuries
        
        return features
    
    def _create_fatigue_features(self, fixture: Dict) -> Dict:
        """Create fatigue features"""
        features = {}
        
        # Days since last match
        home_last_match = fixture.get('home_last_match_date', '')
        away_last_match = fixture.get('away_last_match_date', '')
        match_date = self._parse_date(fixture.get('date', ''))
        
        if home_last_match and match_date:
            home_last_date = self._parse_date(home_last_match)
            if home_last_date:
                days_diff = (match_date - home_last_date).days
                features['home_days_since_last_match'] = days_diff
                features['home_is_fresh'] = 1 if days_diff >= 7 else 0
                features['home_is_fatigued'] = 1 if days_diff <= 2 else 0
        
        if away_last_match and match_date:
            away_last_date = self._parse_date(away_last_match)
            if away_last_date:
                days_diff = (match_date - away_last_date).days
                features['away_days_since_last_match'] = days_diff
                features['away_is_fresh'] = 1 if days_diff >= 7 else 0
                features['away_is_fatigued'] = 1 if days_diff <= 2 else 0
        
        # Matches in last period
        home_matches_30 = fixture.get('home_matches_last_30_days', 0)
        away_matches_30 = fixture.get('away_matches_last_30_days', 0)
        
        if home_matches_30 > 0:
            features['home_matches_last_30_days'] = home_matches_30
            features['home_match_frequency'] = home_matches_30 / 30
        
        if away_matches_30 > 0:
            features['away_matches_last_30_days'] = away_matches_30
            features['away_match_frequency'] = away_matches_30 / 30
        
        # Travel
        home_travel = fixture.get('home_travel_last_7_days_km', 0)
        away_travel = fixture.get('away_travel_last_7_days_km', 0)
        
        if home_travel > 0:
            features['home_travel_last_7_days_km'] = home_travel
            features['home_is_heavy_travel'] = 1 if home_travel > 1000 else 0
        
        if away_travel > 0:
            features['away_travel_last_7_days_km'] = away_travel
            features['away_is_heavy_travel'] = 1 if away_travel > 1000 else 0
        
        return features
    
    def _create_advanced_features(self, fixture: Dict, historical_data: pd.DataFrame) -> Dict:
        """Create advanced statistical features"""
        features = {}
        
        # This would require advanced statistics like xG, possession, shots, etc.
        # For now, return empty dict - to be implemented with actual data
        
        return features
    
    def _create_interaction_features(self, existing_features: Dict) -> Dict:
        """Create interaction features between existing features"""
        features = {}
        
        # Only create interaction features if we have enough base features
        if len(existing_features) < 10:
            return features
        
        # List of features to create interactions for
        numeric_features = {}
        for key, value in existing_features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_features[key] = value
        
        if len(numeric_features) < 5:
            return features
        
        # Create some common interactions
        feature_pairs = [
            ('home_win_rate_last_5', 'away_win_rate_last_5'),
            ('home_goals_scored_avg', 'away_goals_conceded_avg'),
            ('away_goals_scored_avg', 'home_goals_conceded_avg'),
            ('home_team_rating', 'away_team_rating'),
            ('implied_prob_home', 'implied_prob_away')
        ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in numeric_features and feat2 in numeric_features:
                # Product interaction
                features[f'{feat1}_x_{feat2}'] = numeric_features[feat1] * numeric_features[feat2]
                
                # Ratio interaction (avoid division by zero)
                if numeric_features[feat2] != 0:
                    features[f'{feat1}_div_{feat2}'] = numeric_features[feat1] / numeric_features[feat2]
                
                # Difference interaction
                features[f'{feat1}_minus_{feat2}'] = numeric_features[feat1] - numeric_features[feat2]
        
        # Create polynomial features for key metrics
        key_metrics = ['home_win_rate_last_5', 'away_win_rate_last_5', 
                      'implied_prob_home', 'implied_prob_away']
        
        for metric in key_metrics:
            if metric in numeric_features:
                features[f'{metric}_squared'] = numeric_features[metric] ** 2
                features[f'{metric}_cubed'] = numeric_features[metric] ** 3
                features[f'{metric}_log'] = np.log(numeric_features[metric] + 1e-10)
        
        return features
    
    def create_features_batch(self, fixtures: List[Dict], historical_data: pd.DataFrame = None) -> List[Dict]:
        """Create features for multiple fixtures"""
        if not fixtures:
            return []
        
        self.logger.info(f"Creating features for {len(fixtures)} fixtures...")
        
        features_list = []
        
        # Process in parallel if enabled
        if self.feature_config['parallel_processing']:
            import concurrent.futures
            
            max_workers = min(self.optimizer.optimization_config.max_parallel_processes, len(fixtures))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for fixture in fixtures:
                    future = executor.submit(self.create_features, fixture, historical_data)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        features = future.result()
                        if features:
                            features_list.append(features)
                    except Exception as e:
                        self.logger.error(f"Feature creation failed: {e}")
        else:
            # Sequential processing
            for i, fixture in enumerate(fixtures):
                try:
                    features = self.create_features(fixture, historical_data)
                    if features:
                        features_list.append(features)
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Processed {i + 1}/{len(fixtures)} fixtures")
                        
                except Exception as e:
                    self.logger.error(f"Failed to create features for fixture {i}: {e}")
        
        self.logger.info(f"Feature creation complete: {len(features_list)} fixtures processed")
        
        return features_list
    
    def convert_to_dataframe(self, features_list: List[Dict]) -> pd.DataFrame:
        """Convert list of feature dicts to DataFrame"""
        if not features_list:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Select features based on hardware constraints
        df = self._select_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature DataFrame"""
        if df.empty:
            return df
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns
        for col in numeric_cols:
            if df[col].isna().any():
                # Use median for numeric columns
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('missing')
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features based on hardware constraints"""
        if df.empty:
            return df
        
        max_features = self.feature_config['max_features']
        
        # If we have more features than allowed, select the most important ones
        if len(df.columns) > max_features:
            self.logger.warning(f"Too many features ({len(df.columns)}), selecting top {max_features}")
            
            # Simple feature selection based on variance
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Calculate variance
                variances = {}
                for col in numeric_cols:
                    try:
                        variances[col] = df[col].var()
                    except:
                        variances[col] = 0
                
                # Sort by variance (descending)
                sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
                
                # Select top features
                selected_features = [feat for feat, _ in sorted_features[:max_features]]
                
                # Always keep key features
                key_features = ['home_team_hash', 'away_team_hash', 'league_hash', 
                               'match_date', 'result']
                
                for feat in key_features:
                    if feat in df.columns and feat not in selected_features:
                        selected_features.append(feat)
                
                # Select columns
                df = df[selected_features]
        
        return df
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old feature cache"""
        try:
            current_time = datetime.now().timestamp()
            for cache_file in self.cache_dir.glob("*.cache"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Feature cache cleanup failed: {e}")
    
    def get_feature_report(self, features_list: List[Dict]) -> Dict:
        """Generate feature engineering report"""
        if not features_list:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(features_list)
        
        report = {
            'total_fixtures': len(features_list),
            'total_features': len(df.columns),
            'feature_types': {},
            'missing_values': {},
            'feature_stats': {}
        }
        
        # Analyze feature types
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype not in report['feature_types']:
                report['feature_types'][dtype] = []
            report['feature_types'][dtype].append(col)
            
            # Missing values
            missing = df[col].isna().sum()
            report['missing_values'][col] = {
                'missing_count': int(missing),
                'missing_percent': float(missing / len(df) * 100)
            }
            
            # Basic statistics for numeric columns
            if np.issubdtype(df[col].dtype, np.number):
                report['feature_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return report
