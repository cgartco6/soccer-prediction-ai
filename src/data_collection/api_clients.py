"""
API Clients for various football data sources
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from ratelimit import limits, sleep_and_retry

class APIClientBase:
    """Base class for API clients"""
    
    def __init__(self, api_key: str, base_url: str, rate_limit: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(self.get_default_headers())
        
    def get_default_headers(self) -> Dict:
        """Get default headers for API requests"""
        return {
            'User-Agent': 'SoccerPredictionAI/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    @sleep_and_retry
    @limits(calls=10, period=60)  # Rate limiting
    def make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise

class FootballDataAPI(APIClientBase):
    """Client for Football-Data.org API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.football-data.org/v4")
        self.session.headers.update({'X-Auth-Token': api_key})
    
    def get_todays_fixtures(self) -> List[Dict]:
        """Get today's fixtures"""
        today = datetime.now().strftime('%Y-%m-%d')
        endpoint = "matches"
        params = {
            'dateFrom': today,
            'dateTo': today,
            'competitions': 'PL,PD,SA,BL1,FL1,CL,ELC,WC'  # Top competitions
        }
        
        data = self.make_request(endpoint, params)
        return self.parse_fixtures(data)
    
    def parse_fixtures(self, data: Dict) -> List[Dict]:
        """Parse fixtures from API response"""
        fixtures = []
        
        for match in data.get('matches', []):
            fixture = {
                'id': match.get('id'),
                'date': match.get('utcDate'),
                'status': match.get('status'),
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_id': match['homeTeam']['id'],
                'away_id': match['awayTeam']['id'],
                'competition': match['competition']['name'],
                'league': match['competition']['code'],
                'source': 'football-data.org'
            }
            
            # Add odds if available
            if 'odds' in match:
                fixture['odds'] = match['odds']
            
            fixtures.append(fixture)
        
        return fixtures
    
    def get_team_stats(self, team_id: int, season: Optional[str] = None) -> Dict:
        """Get detailed team statistics"""
        endpoint = f"teams/{team_id}"
        params = {'season': season} if season else {}
        
        return self.make_request(endpoint, params)
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Get head-to-head history"""
        endpoint = f"teams/{team1_id}/matches"
        params = {
            'limit': 10,
            'competitions': 'PL,PD,SA,BL1,FL1,CL,ELC,WC'
        }
        
        data = self.make_request(endpoint, params)
        h2h_matches = [
            match for match in data.get('matches', [])
            if match['awayTeam']['id'] == team2_id or match['homeTeam']['id'] == team2_id
        ]
        
        return h2h_matches

class APIFootballClient(APIClientBase):
    """Client for API-Football (RapidAPI)"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://v3.football.api-sports.io")
        self.session.headers.update({'x-rapidapi-key': api_key})
    
    def get_live_odds(self, fixture_id: int) -> Dict:
        """Get live odds for a fixture"""
        endpoint = "odds"
        params = {
            'fixture': fixture_id,
            'bookmaker': 6  # Bet365
        }
        
        return self.make_request(endpoint, params)
    
    def get_injuries(self, fixture_id: int) -> List[Dict]:
        """Get injury information for a fixture"""
        endpoint = "injuries"
        params = {'fixture': fixture_id}
        
        data = self.make_request(endpoint, params)
        return data.get('response', [])
    
    def get_prediction(self, fixture_id: int) -> Dict:
        """Get AI prediction from API-Football"""
        endpoint = "predictions"
        params = {'fixture': fixture_id}
        
        return self.make_request(endpoint, params)

class WeatherAPI:
    """Client for weather data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_match_weather(self, city: str, match_time: datetime) -> Dict:
        """Get weather forecast for match location and time"""
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(f"{self.base_url}/forecast", params=params)
        data = response.json()
        
        # Find closest forecast to match time
        closest_forecast = min(
            data.get('list', []),
            key=lambda x: abs(datetime.fromtimestamp(x['dt']) - match_time)
        )
        
        return {
            'temperature': closest_forecast['main']['temp'],
            'humidity': closest_forecast['main']['humidity'],
            'wind_speed': closest_forecast['wind']['speed'],
            'conditions': closest_forecast['weather'][0]['main'],
            'precipitation': closest_forecast.get('rain', {}).get('3h', 0)
        }

class APIManager:
    """Manager for all API clients"""
    
    def __init__(self, config: Dict):
        self.clients = {}
        self.config = config
        
        # Initialize all API clients
        if config.get('football_data_api_key'):
            self.clients['football_data'] = FootballDataAPI(
                config['football_data_api_key']
            )
        
        if config.get('api_football_key'):
            self.clients['api_football'] = APIFootballClient(
                config['api_football_key']
            )
        
        if config.get('weather_api_key'):
            self.clients['weather'] = WeatherAPI(config['weather_api_key'])
    
    def get_all_fixtures(self) -> List[Dict]:
        """Get fixtures from all available sources"""
        all_fixtures = []
        
        for name, client in self.clients.items():
            try:
                if hasattr(client, 'get_todays_fixtures'):
                    fixtures = client.get_todays_fixtures()
                    all_fixtures.extend(fixtures)
                    
            except Exception as e:
                logging.error(f"Failed to get fixtures from {name}: {e}")
        
        return self.deduplicate_fixtures(all_fixtures)
    
    def deduplicate_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures from different sources"""
        unique_fixtures = {}
        
        for fixture in fixtures:
            key = f"{fixture['home_team']}_{fixture['away_team']}_{fixture.get('date', '')}"
            
            if key not in unique_fixtures:
                unique_fixtures[key] = fixture
            else:
                # Merge data from different sources
                existing = unique_fixtures[key]
                existing.update({k: v for k, v in fixture.items() if v})
        
        return list(unique_fixtures.values())
    
    def enrich_fixture(self, fixture: Dict) -> Dict:
        """Enrich fixture with additional data from APIs"""
        fixture_id = fixture.get('id')
        
        # Get odds
        if 'api_football' in self.clients and fixture_id:
            try:
                odds = self.clients['api_football'].get_live_odds(fixture_id)
                fixture['odds_details'] = odds
            except Exception as e:
                logging.warning(f"Could not get odds for fixture {fixture_id}: {e}")
        
        # Get injuries
        if 'api_football' in self.clients and fixture_id:
            try:
                injuries = self.clients['api_football'].get_injuries(fixture_id)
                fixture['injuries'] = injuries
            except Exception as e:
                logging.warning(f"Could not get injuries for fixture {fixture_id}: {e}")
        
        # Get weather
        if 'weather' in self.clients and fixture.get('venue_city'):
            try:
                match_time = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                weather = self.clients['weather'].get_match_weather(
                    fixture['venue_city'], match_time
                )
                fixture['weather'] = weather
            except Exception as e:
                logging.warning(f"Could not get weather for fixture: {e}")
        
        return fixture
