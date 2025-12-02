"""
Data cleaning and preprocessing with hardware optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from ..system.optimizer import SystemOptimizer

class DataCleaner:
    """Clean and preprocess football data with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer):
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        
        # Setup based on hardware
        self.setup_cleaning_strategy()
        
        # Team name mappings for standardization
        self.team_mappings = self.load_team_mappings()
        
        # League mappings
        self.league_mappings = {
            'Premier League': ['EPL', 'Premier League', 'English Premier League'],
            'La Liga': ['La Liga', 'Spanish La Liga', 'Primera Division'],
            'Bundesliga': ['Bundesliga', 'German Bundesliga'],
            'Serie A': ['Serie A', 'Italian Serie A'],
            'Ligue 1': ['Ligue 1', 'French Ligue 1'],
            'Champions League': ['UEFA Champions League', 'Champions League', 'UCL'],
            'Europa League': ['UEFA Europa League', 'Europa League', 'UEL'],
            'World Cup': ['FIFA World Cup', 'World Cup', 'WC']
        }
    
    def setup_cleaning_strategy(self):
        """Setup cleaning strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.cleaning_config = {
                'max_rows_per_batch': 10000,
                'use_parallel': False,
                'memory_efficient': True,
                'aggressive_filtering': True,
                'min_data_quality': 0.6
            }
        elif profile == 'mid_end':
            self.cleaning_config = {
                'max_rows_per_batch': 50000,
                'use_parallel': True,
                'memory_efficient': True,
                'aggressive_filtering': False,
                'min_data_quality': 0.5
            }
        else:  # high_end
            self.cleaning_config = {
                'max_rows_per_batch': 200000,
                'use_parallel': True,
                'memory_efficient': False,
                'aggressive_filtering': False,
                'min_data_quality': 0.4
            }
        
        self.logger.info(f"Data cleaning strategy: {self.cleaning_config}")
    
    def load_team_mappings(self) -> Dict[str, str]:
        """Load team name mappings for standardization"""
        mappings_file = Path("./config/team_mappings.json")
        
        if mappings_file.exists():
            try:
                with open(mappings_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default mappings
        return {
            'man united': 'Manchester United',
            'man utd': 'Manchester United',
            'manchester utd': 'Manchester United',
            'man city': 'Manchester City',
            'mancity': 'Manchester City',
            'manchester city': 'Manchester City',
            # Add more mappings as needed
        }
    
    def clean_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Clean and preprocess fixtures"""
        if not fixtures:
            return []
        
        self.logger.info(f"Cleaning {len(fixtures)} fixtures...")
        
        cleaned_fixtures = []
        stats = {
            'total': len(fixtures),
            'removed': 0,
            'kept': 0,
            'issues': []
        }
        
        for fixture in fixtures:
            try:
                cleaned = self.clean_single_fixture(fixture)
                if cleaned is not None:
                    cleaned_fixtures.append(cleaned)
                    stats['kept'] += 1
                else:
                    stats['removed'] += 1
            except Exception as e:
                stats['removed'] += 1
                stats['issues'].append(str(e))
                self.logger.debug(f"Failed to clean fixture: {e}")
        
        self.logger.info(f"Cleaning complete: {stats['kept']} kept, {stats['removed']} removed")
        
        return cleaned_fixtures
    
    def clean_single_fixture(self, fixture: Dict) -> Optional[Dict]:
        """Clean a single fixture"""
        # Create a copy to avoid modifying original
        cleaned = fixture.copy()
        
        # 1. Standardize team names
        cleaned['home_team'] = self.standardize_team_name(fixture.get('home_team', ''))
        cleaned['away_team'] = self.standardize_team_name(fixture.get('away_team', ''))
        
        # Check if teams are valid
        if not cleaned['home_team'] or not cleaned['away_team']:
            return None
        
        # 2. Standardize league name
        if 'league' in fixture:
            cleaned['league'] = self.standardize_league_name(fixture['league'])
        
        # 3. Parse and validate date
        date_str = fixture.get('date', '')
        if date_str:
            parsed_date = self.parse_date(date_str)
            if parsed_date:
                cleaned['date'] = parsed_date.isoformat()
                cleaned['timestamp'] = parsed_date.timestamp()
                
                # Check if match is too old
                days_old = (datetime.now() - parsed_date).days
                if days_old > 365 and self.cleaning_config['aggressive_filtering']:
                    return None
            else:
                return None
        else:
            return None
        
        # 4. Clean odds data
        if 'odds' in fixture or 'bookmaker_odds' in fixture:
            cleaned_odds = self.clean_odds_data(fixture)
            if cleaned_odds:
                cleaned.update(cleaned_odds)
        
        # 5. Clean score data
        if 'score' in fixture:
            cleaned['score'] = self.clean_score_data(fixture['score'])
        
        # 6. Clean venue data
        if 'venue' in fixture:
            cleaned['venue'] = self.clean_venue_data(fixture['venue'])
        
        # 7. Clean weather data
        if 'weather' in fixture:
            cleaned['weather'] = self.clean_weather_data(fixture['weather'])
        
        # 8. Add cleaning metadata
        cleaned['cleaned_at'] = datetime.now().isoformat()
        cleaned['data_quality'] = self.calculate_data_quality(cleaned)
        
        # Filter by quality threshold
        if cleaned['data_quality'] < self.cleaning_config['min_data_quality']:
            return None
        
        return cleaned
    
    def standardize_team_name(self, team_name: str) -> str:
        """Standardize team name"""
        if not team_name:
            return ''
        
        # Convert to lowercase for matching
        team_lower = team_name.lower().strip()
        
        # Check mappings
        for pattern, standard in self.team_mappings.items():
            if pattern in team_lower:
                return standard
        
        # Basic cleaning
        cleaned = team_name.strip()
        
        # Remove common prefixes/suffixes and clean
        patterns_to_remove = [
            r'^fc\s+', r'\s+fc$', r'^cf\s+', r'\s+cf$',
            r'^afc\s+', r'\s+afc$', r'^sc\s+', r'\s+sc$'
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Capitalize properly
        words = cleaned.split()
        capitalized_words = []
        
        for word in words:
            if word.upper() == word:  # Already uppercase (like FC, AFC)
                capitalized_words.append(word)
            else:
                capitalized_words.append(word.capitalize())
        
        cleaned = ' '.join(capitalized_words).strip()
        
        return cleaned if cleaned else team_name.strip()
    
    def standardize_league_name(self, league_name: str) -> str:
        """Standardize league name"""
        if not league_name:
            return 'Unknown'
        
        league_lower = league_name.lower().strip()
        
        # Check mappings
        for standard, variations in self.league_mappings.items():
            for variation in variations:
                if variation.lower() in league_lower:
                    return standard
        
        # Basic cleaning
        return league_name.strip()
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        # Try multiple formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',  # ISO with timezone
            '%Y-%m-%dT%H:%M:%S.%f%z',  # ISO with microseconds
            '%Y-%m-%dT%H:%M:%S',  # ISO without timezone
            '%Y-%m-%d %H:%M:%S',  # SQL format
            '%d/%m/%Y %H:%M',  # European format
            '%m/%d/%Y %H:%M',  # US format
            '%Y-%m-%d',  # Date only
        ]
        
        for fmt in formats:
            try:
                # Handle Z timezone
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try Unix timestamp
        try:
            timestamp = float(date_str)
            return datetime.fromtimestamp(timestamp)
        except:
            pass
        
        self.logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def clean_odds_data(self, fixture: Dict) -> Dict:
        """Clean and standardize odds data"""
        cleaned_odds = {}
        
        # Combine odds from different sources
        all_odds = {}
        
        if 'odds' in fixture:
            if isinstance(fixture['odds'], dict):
                all_odds['api_odds'] = fixture['odds']
        
        if 'bookmaker_odds' in fixture:
            if isinstance(fixture['bookmaker_odds'], dict):
                all_odds.update(fixture['bookmaker_odds'])
        
        if not all_odds:
            return {}
        
        # Extract 1X2 odds
        match_winner_odds = self.extract_match_winner_odds(all_odds)
        
        if match_winner_odds:
            cleaned_odds['odds'] = match_winner_odds
            
            # Calculate implied probabilities
            cleaned_odds['implied_probabilities'] = self.calculate_implied_probabilities(
                match_winner_odds
            )
            
            # Calculate bookmaker margin
            cleaned_odds['bookmaker_margin'] = self.calculate_bookmaker_margin(
                cleaned_odds['implied_probabilities']
            )
        
        return cleaned_odds
    
    def extract_match_winner_odds(self, odds_dict: Dict) -> Dict:
        """Extract match winner (1X2) odds from various formats"""
        home_odds, draw_odds, away_odds = None, None, None
        
        for source, odds in odds_dict.items():
            if isinstance(odds, dict):
                # Check for common formats
                if 'homeWin' in odds and 'draw' in odds and 'awayWin' in odds:
                    home_odds = float(odds['homeWin']) if home_odds is None else min(home_odds, float(odds['homeWin']))
                    draw_odds = float(odds['draw']) if draw_odds is None else min(draw_odds, float(odds['draw']))
                    away_odds = float(odds['awayWin']) if away_odds is None else min(away_odds, float(odds['awayWin']))
                
                elif '1' in odds and 'X' in odds and '2' in odds:
                    home_odds = float(odds['1']) if home_odds is None else min(home_odds, float(odds['1']))
                    draw_odds = float(odds['X']) if draw_odds is None else min(draw_odds, float(odds['X']))
                    away_odds = float(odds['2']) if away_odds is None else min(away_odds, float(odds['2']))
                
                elif 'home' in odds and 'draw' in odds and 'away' in odds:
                    home_odds = float(odds['home']) if home_odds is None else min(home_odds, float(odds['home']))
                    draw_odds = float(odds['draw']) if draw_odds is None else min(draw_odds, float(odds['draw']))
                    away_odds = float(odds['away']) if away_odds is None else min(away_odds, float(odds['away']))
        
        if home_odds and draw_odds and away_odds:
            return {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds,
                'source': 'consolidated'
            }
        
        return {}
    
    def calculate_implied_probabilities(self, odds: Dict) -> Dict:
        """Calculate implied probabilities from odds"""
        home_odds = odds.get('home', 0)
        draw_odds = odds.get('draw', 0)
        away_odds = odds.get('away', 0)
        
        if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
            return {}
        
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds
        
        total_prob = home_prob + draw_prob + away_prob
        
        # Normalize to remove bookmaker margin
        if total_prob > 0:
            return {
                'home': home_prob / total_prob,
                'draw': draw_prob / total_prob,
                'away': away_prob / total_prob,
                'raw_total': total_prob
            }
        
        return {}
    
    def calculate_bookmaker_margin(self, probabilities: Dict) -> float:
        """Calculate bookmaker margin"""
        total = sum(probabilities.get(key, 0) for key in ['home', 'draw', 'away'])
        if total > 0:
            return (total - 1) * 100  # Percentage margin
        return 0.0
    
    def clean_score_data(self, score_data: Dict) -> Dict:
        """Clean and validate score data"""
        if not isinstance(score_data, dict):
            return {}
        
        cleaned = {}
        
        # Extract full time score
        if 'fullTime' in score_data:
            ft = score_data['fullTime']
            if isinstance(ft, dict):
                cleaned['home'] = int(ft.get('home', 0))
                cleaned['away'] = int(ft.get('away', 0))
        
        # Extract half time score
        if 'halfTime' in score_data:
            ht = score_data['halfTime']
            if isinstance(ht, dict):
                cleaned['home_ht'] = int(ht.get('home', 0))
                cleaned['away_ht'] = int(ht.get('away', 0))
        
        # Calculate result
        if 'home' in cleaned and 'away' in cleaned:
            if cleaned['home'] > cleaned['away']:
                cleaned['result'] = 'H'
                cleaned['winner'] = 'home'
            elif cleaned['home'] < cleaned['away']:
                cleaned['result'] = 'A'
                cleaned['winner'] = 'away'
            else:
                cleaned['result'] = 'D'
                cleaned['winner'] = 'draw'
        
        return cleaned
    
    def clean_venue_data(self, venue_data: Dict) -> Dict:
        """Clean venue data"""
        if not isinstance(venue_data, dict):
            return {}
        
        cleaned = {}
        
        # Standardize venue fields
        field_mappings = {
            'name': 'name',
            'city': 'city',
            'capacity': 'capacity',
            'surface': 'surface',
            'address': 'address',
            'Country': 'country'  # Handle case variations
        }
        
        for old_key, new_key in field_mappings.items():
            if old_key in venue_data and venue_data[old_key]:
                cleaned[new_key] = str(venue_data[old_key]).strip()
        
        # Clean capacity
        if 'capacity' in cleaned:
            try:
                cleaned['capacity'] = int(cleaned['capacity'].replace(',', '').replace('.', ''))
            except:
                del cleaned['capacity']
        
        return cleaned
    
    def clean_weather_data(self, weather_data: Dict) -> Dict:
        """Clean weather data"""
        if not isinstance(weather_data, dict):
            return {}
        
        cleaned = {}
        
        # Extract and convert values
        conversions = {
            'temperature_c': ('temperature', 'temp', 'temp_c'),
            'humidity_percent': ('humidity', 'humidity_percent'),
            'wind_speed_mps': ('wind_speed', 'wind_speed_mps'),
            'conditions': ('conditions', 'weather', 'main'),
            'precipitation_mm': ('precipitation', 'rain', 'rain_3h_mm')
        }
        
        for cleaned_key, source_keys in conversions.items():
            for source_key in source_keys:
                if source_key in weather_data and weather_data[source_key] is not None:
                    try:
                        value = weather_data[source_key]
                        if isinstance(value, (int, float)):
                            cleaned[cleaned_key] = float(value)
                        elif isinstance(value, str):
                            # Try to extract number
                            match = re.search(r'[-+]?\d*\.?\d+', value)
                            if match:
                                cleaned[cleaned_key] = float(match.group())
                            else:
                                cleaned[cleaned_key] = value
                        break
                    except:
                        continue
        
        return cleaned
    
    def calculate_data_quality(self, fixture: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Basic info (0.3 max)
        max_score += 0.3
        if fixture.get('home_team') and fixture.get('away_team'):
            score += 0.2
        if fixture.get('date'):
            score += 0.1
        
        # League info (0.1 max)
        max_score += 0.1
        if fixture.get('league') and fixture.get('league') != 'Unknown':
            score += 0.1
        
        # Odds info (0.3 max)
        max_score += 0.3
        if 'odds' in fixture:
            score += 0.2
        if 'implied_probabilities' in fixture:
            score += 0.1
        
        # Additional data (0.3 max)
        max_score += 0.3
        additional_fields = ['score', 'venue', 'weather', 'injuries', 'lineups']
        for field in additional_fields:
            if field in fixture and fixture[field]:
                score += 0.06
        
        # Normalize score
        if max_score > 0:
            return score / max_score
        
        return 0.0
    
    def clean_historical_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Clean historical match data"""
        if historical_data.empty:
            return historical_data
        
        self.logger.info(f"Cleaning historical data: {len(historical_data)} rows")
        
        # Create a copy
        df = historical_data.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicates")
        
        # Handle missing values based on hardware
        missing_strategy = 'drop' if self.cleaning_config['aggressive_filtering'] else 'fill'
        
        if missing_strategy == 'drop':
            # Drop rows with critical missing data
            critical_columns = ['home_team', 'away_team', 'date', 'result']
            df = df.dropna(subset=critical_columns)
        else:
            # Fill missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna('Unknown')
        
        # Standardize team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].apply(self.standardize_team_name)
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].apply(self.standardize_team_name)
        
        # Standardize league names
        if 'league' in df.columns:
            df['league'] = df['league'].apply(self.standardize_league_name)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])
        
        # Clean result column
        if 'result' in df.columns:
            # Standardize result codes
            result_mapping = {
                'H': 'H', 'HOME': 'H', '1': 'H',
                'A': 'A', 'AWAY': 'A', '2': 'A',
                'D': 'D', 'DRAW': 'D', 'X': 'D'
            }
            df['result'] = df['result'].astype(str).str.upper().map(result_mapping)
            df = df.dropna(subset=['result'])
        
        # Remove outliers in scores
        if all(col in df.columns for col in ['home_score', 'away_score']):
            # Remove unrealistic scores
            df = df[(df['home_score'] >= 0) & (df['home_score'] <= 20)]
            df = df[(df['away_score'] >= 0) & (df['away_score'] <= 20)]
        
        self.logger.info(f"Historical data cleaning complete: {len(df)} rows remaining")
        
        return df
    
    def process_in_batches(self, data: List[Dict], batch_size: int = None) -> List[Dict]:
        """Process data in batches for memory efficiency"""
        if batch_size is None:
            batch_size = self.cleaning_config['max_rows_per_batch']
        
        if len(data) <= batch_size:
            return self.clean_fixtures(data)
        
        self.logger.info(f"Processing {len(data)} items in batches of {batch_size}")
        
        cleaned_data = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cleaned_batch = self.clean_fixtures(batch)
            cleaned_data.extend(cleaned_batch)
            
            # Clear memory
            if i % (batch_size * 5) == 0:
                import gc
                gc.collect()
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        return cleaned_data
    
    def get_cleaning_report(self, original_count: int, cleaned_count: int, 
                          issues: List[str]) -> Dict:
        """Generate cleaning report"""
        return {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'removed_count': original_count - cleaned_count,
            'retention_rate': cleaned_count / original_count if original_count > 0 else 0,
            'issue_count': len(issues),
            'top_issues': issues[:10] if issues else [],
            'timestamp': datetime.now().isoformat(),
            'hardware_profile': self.optimizer.hardware_info.get('hardware_profile')
        }
