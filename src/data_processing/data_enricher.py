"""
Data enrichment with additional information
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import requests
import json
from pathlib import Path
import asyncio
import aiohttp
from ..system.optimizer import SystemOptimizer

class DataEnricher:
    """Enrich data with additional information from various sources"""
    
    def __init__(self, optimizer: SystemOptimizer, config: Dict = None):
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup enrichment strategy
        self.setup_enrichment_strategy()
        
        # External data sources
        self.external_sources = self.setup_external_sources()
        
        # Cache for external data
        self.cache_dir = Path("./data/cache/enrichment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_enrichment_strategy(self):
        """Setup enrichment strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.enrichment_config = {
                'max_external_calls': 10,
                'use_parallel': False,
                'cache_results': True,
                'cache_ttl': 86400,  # 24 hours
                'enrichment_level': 'basic'
            }
        elif profile == 'mid_end':
            self.enrichment_config = {
                'max_external_calls': 30,
                'use_parallel': True,
                'cache_results': True,
                'cache_ttl': 43200,  # 12 hours
                'enrichment_level': 'standard'
            }
        else:  # high_end
            self.enrichment_config = {
                'max_external_calls': 100,
                'use_parallel': True,
                'cache_results': True,
                'cache_ttl': 21600,  # 6 hours
                'enrichment_level': 'full'
            }
        
        self.logger.info(f"Data enrichment strategy: {self.enrichment_config}")
    
    def setup_external_sources(self) -> Dict:
        """Setup external data sources"""
        return {
            'team_rankings': {
                'url': 'https://api.football-data.org/v4/teams/{team_id}',
                'requires_auth': True,
                'cache_key': 'team_{team_id}'
            },
            'player_stats': {
                'url': 'https://api.football-data.org/v4/players/{player_id}',
                'requires_auth': True,
                'cache_key': 'player_{player_id}'
            },
            'league_standings': {
                'url': 'https://api.football-data.org/v4/competitions/{league_id}/standings',
                'requires_auth': True,
                'cache_key': 'standings_{league_id}'
            },
            'team_squad': {
                'url': 'https://api.football-data.org/v4/teams/{team_id}',
                'requires_auth': True,
                'cache_key': 'squad_{team_id}'
            },
            'match_statistics': {
                'url': 'https://api.football-data.org/v4/matches/{match_id}',
                'requires_auth': True,
                'cache_key': 'match_{match_id}'
            },
            'transfermarkt': {
                'url': 'https://www.transfermarkt.com/{team_name}/startseite/verein/{team_id}',
                'requires_auth': False,
                'cache_key': 'transfermarkt_{team_id}'
            }
        }
    
    def enrich_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Enrich fixtures with additional data"""
        if not fixtures:
            return fixtures
        
        self.logger.info(f"Enriching {len(fixtures)} fixtures...")
        
        enriched_fixtures = []
        
        # Process fixtures based on enrichment level
        if self.enrichment_config['enrichment_level'] == 'basic':
            enriched_fixtures = self._enrich_basic(fixtures)
        elif self.enrichment_config['enrichment_level'] == 'standard':
            enriched_fixtures = self._enrich_standard(fixtures)
        else:  # full
            enriched_fixtures = self._enrich_full(fixtures)
        
        self.logger.info(f"Enrichment complete: {len(enriched_fixtures)} fixtures enriched")
        
        return enriched_fixtures
    
    def _enrich_basic(self, fixtures: List[Dict]) -> List[Dict]:
        """Basic enrichment (minimal external calls)"""
        enriched = []
        
        for fixture in fixtures:
            try:
                enriched_fixture = fixture.copy()
                
                # 1. Add basic derived fields
                self._add_derived_fields(enriched_fixture)
                
                # 2. Add team form (if not already present)
                self._add_basic_form_data(enriched_fixture)
                
                # 3. Add league context
                self._add_league_context(enriched_fixture)
                
                enriched.append(enriched_fixture)
                
            except Exception as e:
                self.logger.warning(f"Basic enrichment failed: {e}")
                enriched.append(fixture)
        
        return enriched
    
    def _enrich_standard(self, fixtures: List[Dict]) -> List[Dict]:
        """Standard enrichment (moderate external calls)"""
        enriched = []
        
        # Group fixtures by league for batch processing
        fixtures_by_league = {}
        for fixture in fixtures:
            league = fixture.get('league', 'unknown')
            if league not in fixtures_by_league:
                fixtures_by_league[league] = []
            fixtures_by_league[league].append(fixture)
        
        # Process each league
        for league, league_fixtures in fixtures_by_league.items():
            try:
                # Get league standings if available
                league_standings = self._get_league_standings(league)
                
                for fixture in league_fixtures:
                    enriched_fixture = fixture.copy()
                    
                    # Basic enrichment
                    self._add_derived_fields(enriched_fixture)
                    self._add_basic_form_data(enriched_fixture)
                    
                    # Add league standings
                    if league_standings:
                        self._add_standings_data(enriched_fixture, league_standings)
                    
                    # Add team rankings
                    self._add_team_rankings(enriched_fixture)
                    
                    enriched.append(enriched_fixture)
                    
            except Exception as e:
                self.logger.warning(f"Standard enrichment failed for league {league}: {e}")
                # Fall back to basic enrichment
                for fixture in league_fixtures:
                    enriched_fixture = fixture.copy()
                    self._add_derived_fields(enriched_fixture)
                    self._add_basic_form_data(enriched_fixture)
                    enriched.append(enriched_fixture)
        
        return enriched
    
    def _enrich_full(self, fixtures: List[Dict]) -> List[Dict]:
        """Full enrichment (maximum external calls)"""
        enriched = []
        
        # Use async for full enrichment
        try:
            enriched = asyncio.run(self._enrich_full_async(fixtures))
        except:
            # Fall back to standard enrichment
            self.logger.warning("Full enrichment failed, falling back to standard")
            enriched = self._enrich_standard(fixtures)
        
        return enriched
    
    async def _enrich_full_async(self, fixtures: List[Dict]) -> List[Dict]:
        """Full enrichment with async calls"""
        # Limit concurrent calls
        semaphore = asyncio.Semaphore(self.optimizer.optimization_config.max_parallel_processes)
        
        async def enrich_single(fixture: Dict) -> Dict:
            async with semaphore:
                try:
                    enriched_fixture = fixture.copy()
                    
                    # Basic enrichment
                    self._add_derived_fields(enriched_fixture)
                    self._add_basic_form_data(enriched_fixture)
                    
                    # Async enrichments
                    tasks = [
                        self._async_get_team_data(enriched_fixture, 'home'),
                        self._async_get_team_data(enriched_fixture, 'away'),
                        self._async_get_league_data(enriched_fixture),
                        self._async_get_player_data(enriched_fixture)
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Combine results
                    for result in results:
                        if not isinstance(result, Exception) and result:
                            enriched_fixture.update(result)
                    
                    return enriched_fixture
                    
                except Exception as e:
                    self.logger.warning(f"Async enrichment failed: {e}")
                    return fixture
        
        # Process all fixtures
        tasks = [enrich_single(fixture) for fixture in fixtures]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        enriched = []
        for result in results:
            if not isinstance(result, Exception):
                enriched.append(result)
        
        return enriched
    
    def _add_derived_fields(self, fixture: Dict):
        """Add derived fields to fixture"""
        # Match importance
        league = fixture.get('league', '').lower()
        
        importance = 1  # Default
        
        if 'champions' in league:
            importance = 10
        elif 'premier' in league or 'la liga' in league or 'bundesliga' in league:
            importance = 8
        elif 'europa' in league:
            importance = 7
        elif 'cup' in league or 'trophy' in league:
            importance = 6
        elif 'world cup' in league:
            importance = 10
        
        fixture['match_importance'] = importance
        
        # Derby match
        home_team = fixture.get('home_team', '').lower()
        away_team = fixture.get('away_team', '').lower()
        
        # Common derby patterns
        derbies = [
            ('manchester', 'manchester'),
            ('liverpool', 'everton'),
            ('arsenal', 'tottenham'),
            ('real madrid', 'barcelona'),
            ('ac milan', 'inter milan'),
            ('celtic', 'rangers')
        ]
        
        is_derby = False
        for team1, team2 in derbies:
            if (team1 in home_team and team2 in away_team) or (team2 in home_team and team1 in away_team):
                is_derby = True
                break
        
        fixture['is_derby'] = is_derby
        
        # Season period
        date_str = fixture.get('date', '')
        if date_str:
            try:
                match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                month = match_date.month
                
                # Early season: Aug-Oct
                # Mid season: Nov-Feb
                # Late season: Mar-Jul
                if month in [8, 9, 10]:
                    fixture['season_period'] = 'early'
                elif month in [11, 12, 1, 2]:
                    fixture['season_period'] = 'mid'
                else:
                    fixture['season_period'] = 'late'
            except:
                fixture['season_period'] = 'unknown'
        
        # Time of day
        if date_str:
            try:
                match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                hour = match_date.hour
                
                if 6 <= hour < 12:
                    fixture['time_of_day'] = 'morning'
                elif 12 <= hour < 18:
                    fixture['time_of_day'] = 'afternoon'
                elif 18 <= hour < 22:
                    fixture['time_of_day'] = 'evening'
                else:
                    fixture['time_of_day'] = 'night'
            except:
                fixture['time_of_day'] = 'unknown'
    
    def _add_basic_form_data(self, fixture: Dict):
        """Add basic form data"""
        # This would normally come from historical data
        # For now, add placeholder values
        if 'home_form' not in fixture:
            fixture['home_form'] = ['W', 'D', 'L', 'W', 'W']  # Placeholder
        
        if 'away_form' not in fixture:
            fixture['away_form'] = ['L', 'W', 'D', 'L', 'W']  # Placeholder
        
        # Calculate form points
        form_points = {'W': 3, 'D': 1, 'L': 0}
        
        home_form_points = sum(form_points.get(result, 0) for result in fixture['home_form'])
        away_form_points = sum(form_points.get(result, 0) for result in fixture['away_form'])
        
        fixture['home_form_points'] = home_form_points
        fixture['away_form_points'] = away_form_points
        fixture['form_points_difference'] = home_form_points - away_form_points
        
        # Current streak
        if fixture['home_form']:
            last_result = fixture['home_form'][0]
            streak = 1
            for result in fixture['home_form'][1:]:
                if result == last_result:
                    streak += 1
                else:
                    break
            fixture['home_streak'] = f"{last_result}{streak}"
        
        if fixture['away_form']:
            last_result = fixture['away_form'][0]
            streak = 1
            for result in fixture['away_form'][1:]:
                if result == last_result:
                    streak += 1
                else:
                    break
            fixture['away_streak'] = f"{last_result}{streak}"
    
    def _add_league_context(self, fixture: Dict):
        """Add league context information"""
        league = fixture.get('league', '')
        
        if not league:
            return
        
        # League tiers (simplified)
        top_tier_leagues = [
            'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1',
            'primeira liga', 'eredivisie', 'premier liga'
        ]
        
        second_tier_leagues = [
            'championship', 'segunda division', '2. bundesliga', 'serie b',
            'ligue 2', 'liga portugal 2', 'eerste divisie'
        ]
        
        league_lower = league.lower()
        
        if any(tier in league_lower for tier in top_tier_leagues):
            fixture['league_tier'] = 1
        elif any(tier in league_lower for tier in second_tier_leagues):
            fixture['league_tier'] = 2
        else:
            fixture['league_tier'] = 3
        
        # League country
        country_mapping = {
            'premier': 'england',
            'la liga': 'spain',
            'bundesliga': 'germany',
            'serie': 'italy',
            'ligue': 'france',
            'primeira': 'portugal',
            'eredivisie': 'netherlands',
            'super lig': 'turkey',
            'mls': 'usa',
            'j-league': 'japan'
        }
        
        fixture['league_country'] = 'unknown'
        for pattern, country in country_mapping.items():
            if pattern in league_lower:
                fixture['league_country'] = country
                break
        
        # League competitiveness (simplified)
        competitive_leagues = ['premier league', 'la liga', 'bundesliga', 'serie a']
        if any(comp_league in league_lower for comp_league in competitive_leagues):
            fixture['league_competitiveness'] = 'high'
        else:
            fixture['league_competitiveness'] = 'medium'
    
    def _get_league_standings(self, league: str) -> Optional[Dict]:
        """Get league standings from cache or API"""
        # Generate cache key
        cache_key = f"standings_{league.replace(' ', '_').lower()}"
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < self.enrichment_config['cache_ttl']:
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        # TODO: Implement actual API call
        # For now, return None
        return None
    
    def _add_standings_data(self, fixture: Dict, standings: Dict):
        """Add standings data to fixture"""
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        
        if not home_team or not away_team:
            return
        
        # Find teams in standings
        home_standing = None
        away_standing = None
        
        for team_standing in standings.get('table', []):
            team_name = team_standing.get('team', {}).get('name', '')
            if home_team.lower() in team_name.lower():
                home_standing = team_standing
            if away_team.lower() in team_name.lower():
                away_standing = team_standing
        
        # Add standing data
        if home_standing:
            fixture['home_position'] = home_standing.get('position', 0)
            fixture['home_points'] = home_standing.get('points', 0)
            fixture['home_played'] = home_standing.get('playedGames', 0)
            fixture['home_wins'] = home_standing.get('won', 0)
            fixture['home_draws'] = home_standing.get('draw', 0)
            fixture['home_losses'] = home_standing.get('lost', 0)
            fixture['home_goal_difference'] = home_standing.get('goalDifference', 0)
        
        if away_standing:
            fixture['away_position'] = away_standing.get('position', 0)
            fixture['away_points'] = away_standing.get('points', 0)
            fixture['away_played'] = away_standing.get('playedGames', 0)
            fixture['away_wins'] = away_standing.get('won', 0)
            fixture['away_draws'] = away_standing.get('draw', 0)
            fixture['away_losses'] = away_standing.get('lost', 0)
            fixture['away_goal_difference'] = away_standing.get('goalDifference', 0)
        
        # Calculate derived standings metrics
        if 'home_position' in fixture and 'away_position' in fixture:
            fixture['position_difference'] = fixture['home_position'] - fixture['away_position']
            
            # Points per game
            if fixture.get('home_played', 0) > 0:
                fixture['home_ppg'] = fixture['home_points'] / fixture['home_played']
            if fixture.get('away_played', 0) > 0:
                fixture['away_ppg'] = fixture['away_points'] / fixture['away_played']
            
            # Win percentages
            if fixture.get('home_played', 0) > 0:
                fixture['home_win_pct'] = fixture['home_wins'] / fixture['home_played']
            if fixture.get('away_played', 0) > 0:
                fixture['away_win_pct'] = fixture['away_wins'] / fixture['away_played']
    
    def _add_team_rankings(self, fixture: Dict):
        """Add team ranking data"""
        # This would normally come from external ranking systems
        # For now, add placeholder values
        
        # FIFA rankings (placeholder)
        fifa_rankings = {
            'brazil': 1, 'argentina': 2, 'france': 3, 'england': 4, 'belgium': 5,
            'netherlands': 6, 'croatia': 7, 'italy': 8, 'portugal': 9, 'spain': 10
        }
        
        home_team = fixture.get('home_team', '').lower()
        away_team = fixture.get('away_team', '').lower()
        
        # Find best matching country
        home_rank = 50  # Default for unranked teams
        away_rank = 50
        
        for country, rank in fifa_rankings.items():
            if country in home_team:
                home_rank = rank
            if country in away_team:
                away_rank = rank
        
        fixture['home_fifa_rank'] = home_rank
        fixture['away_fifa_rank'] = away_rank
        fixture['fifa_rank_difference'] = home_rank - away_rank
        
        # Club coefficients (placeholder)
        club_coefficients = {
            'manchester city': 100, 'real madrid': 98, 'bayern': 96,
            'barcelona': 94, 'liverpool': 92, 'psg': 90, 'chelsea': 88,
            'manchester united': 86, 'ac milan': 84, 'inter': 82
        }
        
        home_coeff = 50
        away_coeff = 50
        
        for club, coeff in club_coefficients.items():
            if club in home_team:
                home_coeff = coeff
            if club in away_team:
                away_coeff = coeff
        
        fixture['home_club_coefficient'] = home_coeff
        fixture['away_club_coefficient'] = away_coeff
        fixture['club_coefficient_difference'] = home_coeff - away_coeff
    
    async def _async_get_team_data(self, fixture: Dict, team_type: str) -> Dict:
        """Async get team data"""
        team_key = f"{team_type}_team"
        team_name = fixture.get(team_key, '')
        
        if not team_name:
            return {}
        
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Placeholder data
        return {
            f'{team_type}_team_size': 25,
            f'{team_type}_avg_age': 26.5,
            f'{team_type}_market_value': 500000000,
            f'{team_type}_coach_experience': 5
        }
    
    async def _async_get_league_data(self, fixture: Dict) -> Dict:
        """Async get league data"""
        league = fixture.get('league', '')
        
        if not league:
            return {}
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        # Placeholder data
        return {
            'league_avg_goals': 2.5,
            'league_avg_corners': 10.2,
            'league_avg_cards': 3.8,
            'league_home_win_rate': 0.45
        }
    
    async def _async_get_player_data(self, fixture: Dict) -> Dict:
        """Async get player data"""
        # Simulate API call
        await asyncio.sleep(0.2)
        
        # Placeholder data
        return {
            'home_top_scorer_goals': 15,
            'away_top_scorer_goals': 12,
            'home_assist_leader': 8,
            'away_assist_leader': 6
        }
    
    def enrich_historical_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Enrich historical match data"""
        if historical_data.empty:
            return historical_data
        
        self.logger.info(f"Enriching historical data: {len(historical_data)} rows")
        
        df = historical_data.copy()
        
        # Add derived features
        df = self._add_historical_derived_features(df)
        
        # Add form features
        df = self._add_historical_form_features(df)
        
        # Add league context
        df = self._add_historical_league_context(df)
        
        self.logger.info(f"Historical data enrichment complete")
        
        return df
    
    def _add_historical_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to historical data"""
        # Ensure required columns exist
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score', 'date']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Result
        df['result'] = 'D'  # Default draw
        df.loc[df['home_score'] > df['away_score'], 'result'] = 'H'
        df.loc[df['home_score'] < df['away_score'], 'result'] = 'A'
        
        # Total goals
        df['total_goals'] = df['home_score'] + df['away_score']
        
        # Goal difference
        df['goal_difference'] = df['home_score'] - df['away_score']
        
        # Both teams scored
        df['both_teams_scored'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
        
        # Over/under 2.5 goals
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['under_2_5'] = (df['total_goals'] < 2.5).astype(int)
        
        # Clean sheet
        df['home_clean_sheet'] = (df['away_score'] == 0).astype(int)
        df['away_clean_sheet'] = (df['home_score'] == 0).astype(int)
        
        # Win margin
        df['win_margin'] = abs(df['goal_difference'])
        df['is_big_win'] = (df['win_margin'] >= 3).astype(int)
        
        return df
    
    def _add_historical_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form features to historical data"""
        # Sort by team and date
        df = df.sort_values(['home_team', 'date']).reset_index(drop=True)
        
        # Initialize form columns
        df['home_form_last_5'] = ''
        df['away_form_last_5'] = ''
        
        # Calculate form for each team (simplified)
        # In practice, this would need more sophisticated rolling calculations
        
        return df
    
    def _add_historical_league_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league context to historical data"""
        if 'league' not in df.columns:
            df['league'] = 'Unknown'
        
        # League tier
        df['league_tier'] = 3  # Default lower tier
        
        top_tier_keywords = ['premier', 'la liga', 'bundesliga', 'serie a', 'ligue 1']
        second_tier_keywords = ['championship', 'segunda', '2. bundesliga', 'serie b', 'ligue 2']
        
        for idx, league in df['league'].astype(str).items():
            league_lower = league.lower()
            
            if any(keyword in league_lower for keyword in top_tier_keywords):
                df.at[idx, 'league_tier'] = 1
            elif any(keyword in league_lower for keyword in second_tier_keywords):
                df.at[idx, 'league_tier'] = 2
        
        # Match importance
        df['match_importance'] = 1
        
        # Increase importance for certain leagues
        important_leagues = ['champions league', 'europa league', 'world cup', 'euro']
        for idx, league in df['league'].astype(str).items():
            league_lower = league.lower()
            if any(important_league in league_lower for important_league in important_leagues):
                df.at[idx, 'match_importance'] = 3
        
        return df
    
    def cleanup_cache(self, max_age_hours: int = 72):
        """Clean up old enrichment cache"""
        try:
            current_time = datetime.now().timestamp()
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Enrichment cache cleanup failed: {e}")
    
    def get_enrichment_report(self, original_count: int, enriched_count: int) -> Dict:
        """Generate enrichment report"""
        return {
            'original_count': original_count,
            'enriched_count': enriched_count,
            'enrichment_rate': enriched_count / original_count if original_count > 0 else 0,
            'strategy': self.enrichment_config['enrichment_level'],
            'timestamp': datetime.now().isoformat(),
            'hardware_profile': self.optimizer.hardware_info.get('hardware_profile')
        }
