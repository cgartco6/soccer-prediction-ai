"""
Data validation and quality assurance
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
from pathlib import Path

class DataValidator:
    """Validate and ensure data quality"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.validation_rules = {
            'fixture': self.validate_fixture,
            'odds': self.validate_odds,
            'team': self.validate_team,
            'player': self.validate_player
        }
        
        # Setup quality thresholds
        self.quality_thresholds = {
            'minimum': 0.3,
            'acceptable': 0.6,
            'good': 0.8,
            'excellent': 0.9
        }
    
    def validate_fixture(self, fixture: Dict) -> Tuple[bool, List[str], float]:
        """Validate a football fixture"""
        errors = []
        warnings = []
        quality_score = 0.0
        
        # Required fields
        required_fields = ['home_team', 'away_team', 'date']
        for field in required_fields:
            if field not in fixture:
                errors.append(f"Missing required field: {field}")
            elif not fixture[field]:
                errors.append(f"Empty required field: {field}")
        
        if errors:
            return False, errors, quality_score
        
        # Field validation
        home_team = fixture.get('home_team', '').strip()
        away_team = fixture.get('away_team', '').strip()
        date_str = fixture.get('date', '')
        
        # Team name validation
        if not self.is_valid_team_name(home_team):
            errors.append(f"Invalid home team name: {home_team}")
        if not self.is_valid_team_name(away_team):
            errors.append(f"Invalid away team name: {away_team}")
        
        if home_team.lower() == away_team.lower():
            errors.append("Home and away teams cannot be the same")
        
        # Date validation
        try:
            match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            now = datetime.now()
            
            # Check if date is in reasonable range
            if match_date < now - timedelta(days=7):
                warnings.append("Match date is more than 7 days in the past")
            elif match_date > now + timedelta(days=365):
                warnings.append("Match date is more than 1 year in the future")
        except:
            errors.append(f"Invalid date format: {date_str}")
        
        # League validation (if present)
        league = fixture.get('league', '')
        if league and not self.is_valid_league(league):
            warnings.append(f"Unusual league name: {league}")
        
        # Calculate quality score
        quality_score = self.calculate_fixture_quality(fixture)
        
        return len(errors) == 0, errors + warnings, quality_score
    
    def is_valid_team_name(self, team_name: str) -> bool:
        """Check if team name is valid"""
        if not team_name or len(team_name.strip()) < 2:
            return False
        
        # Check for reasonable length
        if len(team_name) > 50:
            return False
        
        # Check for excessive special characters
        special_chars = re.findall(r'[^a-zA-Z0-9\s\-\.]', team_name)
        if len(special_chars) > 3:
            return False
        
        return True
    
    def is_valid_league(self, league: str) -> bool:
        """Check if league name is valid"""
        if not league or len(league.strip()) < 2:
            return False
        
        # Common league patterns
        common_leagues = [
            'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1',
            'champions league', 'europa league', 'world cup', 'euro',
            'fa cup', 'copa del rey', 'dfb pokal', 'coppa italia', 'coupe de france'
        ]
        
        league_lower = league.lower()
        for common in common_leagues:
            if common in league_lower:
                return True
        
        # Check for reasonable format
        words = league_lower.split()
        if len(words) > 5:
            return False
        
        return True
    
    def calculate_fixture_quality(self, fixture: Dict) -> float:
        """Calculate data quality score for fixture (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Basic information (0.3 max)
        max_score += 0.3
        if fixture.get('home_team') and fixture.get('away_team'):
            score += 0.2
        if fixture.get('date'):
            score += 0.1
        
        # League and competition (0.2 max)
        max_score += 0.2
        if fixture.get('league'):
            score += 0.1
        if fixture.get('competition'):
            score += 0.05
        if fixture.get('matchday'):
            score += 0.05
        
        # IDs and references (0.1 max)
        max_score += 0.1
        if fixture.get('home_id') and fixture.get('away_id'):
            score += 0.1
        
        # Odds data (0.2 max)
        max_score += 0.2
        if fixture.get('odds') or fixture.get('bookmaker_odds'):
            score += 0.2
        
        # Additional data (0.2 max)
        max_score += 0.2
        additional_fields = ['weather', 'injuries', 'lineups', 'statistics', 'venue']
        for field in additional_fields:
            if fixture.get(field):
                score += 0.04
        
        # Normalize score
        if max_score > 0:
            return min(score / max_score, 1.0)
        return 0.0
    
    def validate_odds(self, odds: Dict) -> Tuple[bool, List[str], float]:
        """Validate betting odds"""
        errors = []
        warnings = []
        quality_score = 0.0
        
        # Check structure
        if not isinstance(odds, dict):
            errors.append("Odds must be a dictionary")
            return False, errors, quality_score
        
        # Check for required odds
        required_markets = ['1X2', 'match_winner']
        found_markets = []
        
        for market in odds.keys():
            if '1x2' in market.lower() or 'match' in market.lower() or 'winner' in market.lower():
                found_markets.append(market)
        
        if not found_markets:
            warnings.append("No 1X2/match winner odds found")
        
        # Validate individual odds
        for market_name, market_odds in odds.items():
            if isinstance(market_odds, dict):
                # Check for reasonable odds values
                for outcome, odd_value in market_odds.items():
                    if isinstance(odd_value, (int, float)):
                        if odd_value < 1.0:
                            errors.append(f"Invalid odds value {odd_value} for {outcome} in {market_name}")
                        elif odd_value > 1000:
                            warnings.append(f"Extremely high odds {odd_value} for {outcome} in {market_name}")
        
        # Calculate quality score
        quality_score = self.calculate_odds_quality(odds)
        
        return len(errors) == 0, errors + warnings, quality_score
    
    def calculate_odds_quality(self, odds: Dict) -> float:
        """Calculate quality score for odds data"""
        if not odds:
            return 0.0
        
        score = 0.0
        
        # Check for multiple bookmakers
        if isinstance(odds, dict):
            # Count distinct odds sources
            odds_sources = 0
            for key, value in odds.items():
                if isinstance(value, dict) and any(k in key.lower() for k in ['odds', 'bookmaker', 'bet']):
                    odds_sources += 1
            
            score += min(odds_sources * 0.2, 0.6)  # Max 0.6 for multiple sources
            
            # Check for complete market
            for market in odds.values():
                if isinstance(market, dict):
                    if all(outcome in market for outcome in ['home', 'draw', 'away']):
                        score += 0.2
                    if len(market) >= 3:  # At least 3 outcomes
                        score += 0.1
                    break
        
        return min(score, 1.0)
    
    def validate_team(self, team_data: Dict) -> Tuple[bool, List[str], float]:
        """Validate team data"""
        errors = []
        warnings = []
        quality_score = 0.0
        
        # Required fields
        required_fields = ['name', 'id']
        for field in required_fields:
            if field not in team_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, quality_score
        
        # Name validation
        name = team_data.get('name', '').strip()
        if not self.is_valid_team_name(name):
            errors.append(f"Invalid team name: {name}")
        
        # ID validation
        team_id = team_data.get('id')
        if not isinstance(team_id, (int, str)):
            errors.append(f"Invalid team ID type: {type(team_id)}")
        elif isinstance(team_id, str) and not team_id.strip():
            errors.append("Empty team ID")
        
        # Country validation (if present)
        country = team_data.get('country', '')
        if country and len(country) > 50:
            warnings.append(f"Unusually long country name: {country}")
        
        # Calculate quality score
        quality_score = self.calculate_team_quality(team_data)
        
        return len(errors) == 0, errors + warnings, quality_score
    
    def calculate_team_quality(self, team_data: Dict) -> float:
        """Calculate quality score for team data"""
        score = 0.0
        
        # Basic info
        if team_data.get('name'):
            score += 0.2
        if team_data.get('id'):
            score += 0.2
        
        # Additional info
        if team_data.get('country'):
            score += 0.1
        if team_data.get('founded'):
            score += 0.1
        if team_data.get('venue'):
            score += 0.1
        
        # Statistics
        if team_data.get('stats'):
            score += 0.2
        if team_data.get('form'):
            score += 0.1
        
        return min(score, 1.0)
    
    def validate_player(self, player_data: Dict) -> Tuple[bool, List[str], float]:
        """Validate player data"""
        errors = []
        warnings = []
        quality_score = 0.0
        
        # Required fields
        required_fields = ['name']
        for field in required_fields:
            if field not in player_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, quality_score
        
        # Name validation
        name = player_data.get('name', '').strip()
        if len(name) < 2 or len(name) > 50:
            errors.append(f"Invalid player name length: {len(name)}")
        
        # Position validation (if present)
        position = player_data.get('position', '')
        valid_positions = ['GK', 'DF', 'MF', 'FW', 'Goalkeeper', 'Defender', 
                          'Midfielder', 'Forward', 'Striker', 'Winger']
        if position and position not in valid_positions:
            warnings.append(f"Unusual position: {position}")
        
        # Age validation (if present)
        age = player_data.get('age')
        if age is not None:
            if not isinstance(age, (int, float)):
                errors.append(f"Invalid age type: {type(age)}")
            elif age < 16 or age > 50:
                warnings.append(f"Unusual age for professional player: {age}")
        
        # Calculate quality score
        quality_score = self.calculate_player_quality(player_data)
        
        return len(errors) == 0, errors + warnings, quality_score
    
    def calculate_player_quality(self, player_data: Dict) -> float:
        """Calculate quality score for player data"""
        score = 0.0
        
        # Basic info
        if player_data.get('name'):
            score += 0.3
        if player_data.get('position'):
            score += 0.1
        if player_data.get('team'):
            score += 0.1
        
        # Statistics
        if player_data.get('stats'):
            stats = player_data['stats']
            if isinstance(stats, dict):
                if stats.get('goals') is not None:
                    score += 0.1
                if stats.get('assists') is not None:
                    score += 0.1
                if stats.get('rating') is not None:
                    score += 0.1
        
        # Recent form
        if player_data.get('recent_matches'):
            score += 0.2
        
        return min(score, 1.0)
    
    def validate_dataset(self, data: List[Dict], data_type: str = 'fixture') -> Dict[str, Any]:
        """Validate a dataset of items"""
        if data_type not in self.validation_rules:
            return {
                'valid': False,
                'error': f"Unknown data type: {data_type}",
                'statistics': {}
            }
        
        validation_func = self.validation_rules[data_type]
        
        results = {
            'total_items': len(data),
            'valid_items': 0,
            'invalid_items': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'quality_scores': [],
            'item_details': []
        }
        
        for i, item in enumerate(data):
            try:
                is_valid, messages, quality_score = validation_func(item)
                
                item_result = {
                    'index': i,
                    'valid': is_valid,
                    'quality_score': quality_score,
                    'messages': messages
                }
                
                if is_valid:
                    results['valid_items'] += 1
                else:
                    results['invalid_items'] += 1
                
                # Count errors and warnings
                for message in messages:
                    if 'error' in message.lower() or 'invalid' in message.lower():
                        results['total_errors'] += 1
                    else:
                        results['total_warnings'] += 1
                
                results['quality_scores'].append(quality_score)
                results['item_details'].append(item_result)
                
            except Exception as e:
                self.logger.error(f"Validation failed for item {i}: {e}")
                results['invalid_items'] += 1
                results['total_errors'] += 1
        
        # Calculate statistics
        if results['quality_scores']:
            results['avg_quality'] = sum(results['quality_scores']) / len(results['quality_scores'])
            results['min_quality'] = min(results['quality_scores'])
            results['max_quality'] = max(results['quality_scores'])
        else:
            results['avg_quality'] = 0
            results['min_quality'] = 0
            results['max_quality'] = 0
        
        # Determine overall validity
        validity_threshold = self.quality_thresholds['acceptable']
        results['overall_valid'] = (
            results['valid_items'] > 0 and 
            results['avg_quality'] >= validity_threshold
        )
        
        return results
    
    def filter_by_quality(self, data: List[Dict], data_type: str = 'fixture', 
                         min_quality: float = None) -> List[Dict]:
        """Filter data by quality threshold"""
        if min_quality is None:
            min_quality = self.quality_thresholds['acceptable']
        
        filtered_data = []
        
        validation_func = self.validation_rules[data_type]
        
        for item in data:
            try:
                is_valid, _, quality_score = validation_func(item)
                if is_valid and quality_score >= min_quality:
                    filtered_data.append(item)
            except:
                continue
        
        return filtered_data
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nSummary:")
        report.append(f"  Total Items: {validation_results['total_items']}")
        report.append(f"  Valid Items: {validation_results['valid_items']}")
        report.append(f"  Invalid Items: {validation_results['invalid_items']}")
        report.append(f"  Total Errors: {validation_results['total_errors']}")
        report.append(f"  Total Warnings: {validation_results['total_warnings']}")
        
        report.append(f"\nQuality Statistics:")
        report.append(f"  Average Quality: {validation_results.get('avg_quality', 0):.3f}")
        report.append(f"  Minimum Quality: {validation_results.get('min_quality', 0):.3f}")
        report.append(f"  Maximum Quality: {validation_results.get('max_quality', 0):.3f}")
        
        report.append(f"\nOverall Status:")
        overall_status = "PASS" if validation_results.get('overall_valid', False) else "FAIL"
        report.append(f"  Dataset: {overall_status}")
        
        # Add quality thresholds
        report.append(f"\nQuality Thresholds:")
        for level, threshold in self.quality_thresholds.items():
            count = sum(1 for score in validation_results.get('quality_scores', []) 
                       if score >= threshold)
            percentage = (count / validation_results['total_items'] * 100) if validation_results['total_items'] > 0 else 0
            report.append(f"  {level.title()}: {count} items ({percentage:.1f}%)")
        
        # Top issues
        if validation_results['total_errors'] > 0:
            report.append(f"\nTop Issues:")
            error_messages = []
            for item_detail in validation_results.get('item_details', []):
                for message in item_detail.get('messages', []):
                    if 'error' in message.lower() or 'invalid' in message.lower():
                        error_messages.append(message)
            
            from collections import Counter
            common_errors = Counter(error_messages).most_common(5)
            for error, count in common_errors:
                report.append(f"  {error}: {count} occurrences")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_validation_report(self, validation_results: Dict, 
                             filepath: str = "./logs/validation_report.txt"):
        """Save validation report to file"""
        report = self.generate_validation_report(validation_results)
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
