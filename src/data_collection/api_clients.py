"""
API clients for football data with hardware optimization
"""

import requests
import aiohttp
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from ratelimit import limits, sleep_and_retry
import hashlib
import pickle
from pathlib import Path
import lz4.frame
from ..system.optimizer import SystemOptimizer

class APIClientBase:
    """Base class for API clients with hardware optimization"""
    
    def __init__(self, api_key: str, base_url: str, optimizer: SystemOptimizer):
        self.api_key = api_key
        self.base_url = base_url
        self.optimizer = optimizer
        self.session = requests.Session()
        self.session.headers.update(self.get_default_headers())
        
        # Setup caching
        self.cache_dir = Path("./data/cache/api")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = self.get_cache_ttl()
        self.use_compression = optimizer.optimization_config.use_compression
        
        # Setup rate limiting based on hardware
        self.max_concurrent = optimizer.optimization_config.max_parallel_processes
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        self.logger = logging.getLogger(__name__)
        
    def get_default_headers(self) -> Dict:
        """Get default headers for API requests"""
        return {
            'User-Agent': 'SoccerPredictionAI/2.0 (HardwareOptimized)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    
    def get_cache_ttl(self) -> int:
        """Get cache TTL based on hardware profile"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        ttl_config = {
            'low_end': 7200,    # 2 hours
            'mid_end': 3600,    # 1 hour
            'high_end': 1800,   # 30 minutes
            'custom': 3600
        }
        return ttl_config.get(profile, 3600)
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and parameters"""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > self.cache_ttl:
            try:
                cache_path.unlink()
            except:
                pass
            return None
        
        try:
            if self.use_compression:
                with lz4.frame.open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if self.use_compression:
                with lz4.frame.open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache files"""
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.cache"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def make_request(self, endpoint: str, params: Optional[Dict] = None, 
                    use_cache: bool = True) -> Dict:
        """Make API request with caching and hardware optimization"""
        # Generate cache key
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params)
        
        # Try to load from cache
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for {endpoint}")
                return cached_data
        
        # Make API request
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Adjust timeout based on hardware profile
            profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
            timeout = 45 if profile == 'low_end' else 30
            
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            
            # Parse response
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
            else:
                data = {'raw': response.text}
            
            # Save to cache
            if use_cache and data:
                self._save_to_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.Timeout:
            self.logger.error(f"API request timed out: {url}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    async def make_async_request(self, endpoint: str, params: Optional[Dict] = None,
                               use_cache: bool = True) -> Dict:
        """Make asynchronous API request"""
        async with self.semaphore:
            return await self._make_async_request_impl(endpoint, params, use_cache)
    
    async def _make_async_request_impl(self, endpoint: str, params: Optional[Dict] = None,
                                     use_cache: bool = True) -> Dict:
        """Implementation of async request"""
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params)
        
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        url = f"{self.base_url}/{endpoint}"
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    
                    if response.content_type == 'application/json':
                        data = await response.json()
                    else:
                        data = {'raw': await response.text()}
                    
                    if use_cache and data:
                        self._save_to_cache(cache_key, data)
                    
                    return data
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Async API request timed out: {url}")
            raise
        except aiohttp.ClientError as e:
            self.logger.error(f"Async API request failed: {e}")
            raise

class FootballDataAPI(APIClientBase):
    """Client for Football-Data.org API"""
    
    def __init__(self, api_key: str, optimizer: SystemOptimizer):
        super().__init__(api_key, "https://api.football-data.org/v4", optimizer)
        self.session.headers.update({'X-Auth-Token': api_key})
    
    def get_todays_fixtures(self, use_cache: bool = True) -> List[Dict]:
        """Get today's fixtures with hardware optimization"""
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Adjust competitions based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        if profile == 'low_end':
            competitions = 'PL,PD,SA,BL1,FL1,CL'  # Top leagues only
        elif profile == 'mid_end':
            competitions = 'PL,PD,SA,BL1,FL1,CL,EL,ELC,WC'
        else:
            competitions = 'all'  # All competitions
        
        endpoint = "matches"
        params = {
            'dateFrom': today,
            'dateTo': tomorrow,
            'competitions': competitions
        }
        
        try:
            data = self.make_request(endpoint, params, use_cache=use_cache)
            return self.parse_fixtures(data)
        except Exception as e:
            self.logger.error(f"Failed to get today's fixtures: {e}")
            return []
    
    def parse_fixtures(self, data: Dict) -> List[Dict]:
        """Parse fixtures from API response"""
        fixtures = []
        
        for match in data.get('matches', []):
            try:
                fixture = {
                    'id': match.get('id'),
                    'date': match.get('utcDate'),
                    'status': match.get('status'),
                    'matchday': match.get('matchday'),
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_id': match['homeTeam']['id'],
                    'away_id': match['awayTeam']['id'],
                    'competition': match['competition']['name'],
                    'league': match['competition']['code'],
                    'area': match['area']['name'],
                    'source': 'football-data.org',
                    'last_updated': datetime.now().isoformat()
                }
                
                # Add score if available
                if 'score' in match:
                    fixture['score'] = match['score']
                
                # Add odds if available
                if 'odds' in match:
                    fixture['odds'] = match['odds']
                
                fixtures.append(fixture)
                
            except KeyError as e:
                self.logger.warning(f"Missing key in match data: {e}")
                continue
        
        return fixtures
    
    def get_team_stats(self, team_id: int, season: Optional[str] = None, 
                      use_cache: bool = True) -> Dict:
        """Get detailed team statistics"""
        endpoint = f"teams/{team_id}"
        params = {'season': season} if season else {}
        
        try:
            return self.make_request(endpoint, params, use_cache=use_cache)
        except Exception as e:
            self.logger.error(f"Failed to get team stats for {team_id}: {e}")
            return {}
    
    async def get_multiple_team_stats(self, team_ids: List[int], 
                                    season: Optional[str] = None) -> Dict[int, Dict]:
        """Get statistics for multiple teams asynchronously"""
        tasks = []
        for team_id in team_ids:
            task = self.get_team_stats_async(team_id, season)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        team_stats = {}
        for team_id, result in zip(team_ids, results):
            if not isinstance(result, Exception):
                team_stats[team_id] = result
        
        return team_stats
    
    async def get_team_stats_async(self, team_id: int, season: Optional[str] = None) -> Dict:
        """Async wrapper for team stats"""
        endpoint = f"teams/{team_id}"
        params = {'season': season} if season else {}
        
        return await self.make_async_request(endpoint, params)
    
    def get_head_to_head(self, team1_id: int, team2_id: int, 
                        limit: int = 10) -> List[Dict]:
        """Get head-to-head history"""
        endpoint = f"teams/{team1_id}/matches"
        params = {
            'limit': limit,
            'status': 'FINISHED'
        }
        
        try:
            data = self.make_request(endpoint, params)
            h2h_matches = [
                match for match in data.get('matches', [])
                if match.get('awayTeam', {}).get('id') == team2_id or 
                   match.get('homeTeam', {}).get('id') == team2_id
            ]
            return h2h_matches
        except Exception as e:
            self.logger.error(f"Failed to get H2H for {team1_id} vs {team2_id}: {e}")
            return []

class APIFootballClient(APIClientBase):
    """Client for API-Football (RapidAPI)"""
    
    def __init__(self, api_key: str, optimizer: SystemOptimizer):
        super().__init__(api_key, "https://v3.football.api-sports.io", optimizer)
        self.session.headers.update({
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
        })
    
    def get_fixtures(self, date: str = None, league_id: int = None, 
                    team_id: int = None, use_cache: bool = True) -> List[Dict]:
        """Get fixtures with various filters"""
        endpoint = "fixtures"
        params = {}
        
        if date:
            params['date'] = date
        if league_id:
            params['league'] = league_id
        if team_id:
            params['team'] = team_id
        if not params:
            params['date'] = datetime.now().strftime('%Y-%m-%d')
        
        params['timezone'] = 'UTC'
        
        try:
            data = self.make_request(endpoint, params, use_cache=use_cache)
            return self.parse_fixtures(data)
        except Exception as e:
            self.logger.error(f"Failed to get fixtures: {e}")
            return []
    
    def parse_fixtures(self, data: Dict) -> List[Dict]:
        """Parse API-Football fixtures"""
        fixtures = []
        
        for fixture in data.get('response', []):
            try:
                fixture_data = fixture['fixture']
                teams_data = fixture['teams']
                league_data = fixture['league']
                
                fixture_info = {
                    'id': fixture_data['id'],
                    'date': fixture_data['date'],
                    'timestamp': fixture_data['timestamp'],
                    'status': fixture_data['status'],
                    'venue': fixture_data.get('venue', {}),
                    'home_team': teams_data['home']['name'],
                    'away_team': teams_data['away']['name'],
                    'home_id': teams_data['home']['id'],
                    'away_id': teams_data['away']['id'],
                    'league': league_data['name'],
                    'league_id': league_data['id'],
                    'country': league_data['country'],
                    'season': league_data['season'],
                    'round': fixture.get('league', {}).get('round', ''),
                    'source': 'api-football',
                    'last_updated': datetime.now().isoformat()
                }
                
                # Add odds if available
                if 'odds' in fixture:
                    fixture_info['odds'] = fixture['odds']
                
                # Add predictions if available
                if 'predictions' in fixture:
                    fixture_info['predictions'] = fixture['predictions']
                
                fixtures.append(fixture_info)
                
            except KeyError as e:
                self.logger.warning(f"Missing key in fixture data: {e}")
                continue
        
        return fixtures
    
    def get_odds(self, fixture_id: int, bookmaker: int = None) -> Dict:
        """Get odds for a specific fixture"""
        endpoint = "odds"
        params = {'fixture': fixture_id}
        
        if bookmaker:
            params['bookmaker'] = bookmaker
        
        try:
            data = self.make_request(endpoint, params)
            return self.parse_odds(data)
        except Exception as e:
            self.logger.error(f"Failed to get odds for fixture {fixture_id}: {e}")
            return {}
    
    def parse_odds(self, data: Dict) -> Dict:
        """Parse odds data"""
        odds_data = {}
        
        for odds in data.get('response', []):
            bookmaker = odds.get('bookmaker', {})
            bookmaker_name = bookmaker.get('name', 'Unknown')
            
            for bet in bookmaker.get('bets', []):
                bet_name = bet.get('name', '')
                if bet_name == 'Match Winner':
                    for value in bet.get('values', []):
                        odds_data[value['value']] = float(value['odd'])
        
        return odds_data
    
    def get_injuries(self, fixture_id: int) -> List[Dict]:
        """Get injury information for a fixture"""
        endpoint = "injuries"
        params = {'fixture': fixture_id}
        
        try:
            data = self.make_request(endpoint, params)
            return data.get('response', [])
        except Exception as e:
            self.logger.error(f"Failed to get injuries for fixture {fixture_id}: {e}")
            return []
    
    def get_lineups(self, fixture_id: int) -> Dict:
        """Get lineups for a fixture"""
        endpoint = "lineups"
        params = {'fixture': fixture_id}
        
        try:
            data = self.make_request(endpoint, params)
            return data.get('response', {})
        except Exception as e:
            self.logger.error(f"Failed to get lineups for fixture {fixture_id}: {e}")
            return {}
    
    def get_statistics(self, fixture_id: int) -> Dict:
        """Get match statistics"""
        endpoint = "fixtures/statistics"
        params = {'fixture': fixture_id}
        
        try:
            data = self.make_request(endpoint, params)
            return data.get('response', {})
        except Exception as e:
            self.logger.error(f"Failed to get statistics for fixture {fixture_id}: {e}")
            return {}
    
    async def get_multiple_fixtures_data(self, fixture_ids: List[int]) -> Dict[int, Dict]:
        """Get multiple fixtures data asynchronously"""
        tasks = []
        
        for fixture_id in fixture_ids:
            tasks.append(self.get_fixture_data_async(fixture_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        fixtures_data = {}
        for fixture_id, result in zip(fixture_ids, results):
            if not isinstance(result, Exception):
                fixtures_data[fixture_id] = result
        
        return fixtures_data
    
    async def get_fixture_data_async(self, fixture_id: int) -> Dict:
        """Get comprehensive fixture data asynchronously"""
        endpoints = [
            ('odds', {'fixture': fixture_id}),
            ('injuries', {'fixture': fixture_id}),
            ('lineups', {'fixture': fixture_id}),
            ('fixtures/statistics', {'fixture': fixture_id})
        ]
        
        tasks = []
        for endpoint, params in endpoints:
            tasks.append(self.make_async_request(endpoint, params))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {
            'odds': {},
            'injuries': [],
            'lineups': {},
            'statistics': {}
        }
        
        for i, (endpoint, _) in enumerate(endpoints):
            if not isinstance(results[i], Exception):
                key = endpoint.split('/')[-1]
                data[key] = results[i].get('response', {})
        
        return data

class WeatherAPI(APIClientBase):
    """Client for weather data"""
    
    def __init__(self, api_key: str, optimizer: SystemOptimizer):
        super().__init__(api_key, "https://api.openweathermap.org/data/2.5", optimizer)
        self.api_key = api_key
    
    def get_weather(self, city: str, country: str = None, 
                   date: datetime = None) -> Dict:
        """Get weather forecast"""
        if not date:
            date = datetime.now()
        
        # Build location query
        location = city
        if country:
            location = f"{city},{country}"
        
        endpoint = "forecast"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': 40  # 5-day forecast in 3-hour intervals
        }
        
        try:
            data = self.make_request(endpoint, params)
            return self.extract_match_weather(data, date)
        except Exception as e:
            self.logger.error(f"Failed to get weather for {location}: {e}")
            return {}
    
    def extract_match_weather(self, forecast_data: Dict, match_time: datetime) -> Dict:
        """Extract weather data closest to match time"""
        forecasts = forecast_data.get('list', [])
        
        if not forecasts:
            return {}
        
        # Find closest forecast
        closest_forecast = min(
            forecasts,
            key=lambda x: abs(datetime.fromtimestamp(x['dt']) - match_time)
        )
        
        weather = closest_forecast.get('weather', [{}])[0]
        main = closest_forecast.get('main', {})
        wind = closest_forecast.get('wind', {})
        rain = closest_forecast.get('rain', {})
        snow = closest_forecast.get('snow', {})
        
        return {
            'timestamp': datetime.fromtimestamp(closest_forecast['dt']).isoformat(),
            'temperature_c': main.get('temp'),
            'feels_like_c': main.get('feels_like'),
            'temp_min_c': main.get('temp_min'),
            'temp_max_c': main.get('temp_max'),
            'humidity_percent': main.get('humidity'),
            'pressure_hpa': main.get('pressure'),
            'wind_speed_mps': wind.get('speed'),
            'wind_direction_deg': wind.get('deg'),
            'wind_gust_mps': wind.get('gust'),
            'conditions': weather.get('main'),
            'description': weather.get('description'),
            'icon': weather.get('icon'),
            'rain_3h_mm': rain.get('3h', 0),
            'snow_3h_mm': snow.get('3h', 0),
            'cloudiness_percent': closest_forecast.get('clouds', {}).get('all', 0),
            'visibility_meters': closest_forecast.get('visibility', 10000)
        }
    
    def get_multiple_weather(self, locations: List[tuple]) -> Dict[str, Dict]:
        """Get weather for multiple locations"""
        weather_data = {}
        
        # Limit concurrent requests based on hardware
        max_concurrent = self.optimizer.optimization_config.max_parallel_processes
        
        for i in range(0, len(locations), max_concurrent):
            batch = locations[i:i + max_concurrent]
            
            for city, country in batch:
                try:
                    weather = self.get_weather(city, country)
                    key = f"{city},{country}" if country else city
                    weather_data[key] = weather
                except Exception as e:
                    self.logger.warning(f"Failed to get weather for {city}: {e}")
        
        return weather_data

class APIManager:
    """Manager for all API clients with hardware optimization"""
    
    def __init__(self, config: Dict, optimizer: SystemOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.clients = {}
        self.logger = logging.getLogger(__name__)
        
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize all API clients"""
        api_keys = self.config.get('api_keys', {})
        
        # Football Data API
        football_data_key = api_keys.get('football_data_org')
        if football_data_key:
            self.clients['football_data'] = FootballDataAPI(football_data_key, self.optimizer)
            self.logger.info("Football Data API client initialized")
        
        # API-Football
        api_football_key = api_keys.get('api_football')
        if api_football_key:
            self.clients['api_football'] = APIFootballClient(api_football_key, self.optimizer)
            self.logger.info("API-Football client initialized")
        
        # Weather API
        weather_key = api_keys.get('openweathermap')
        if weather_key:
            self.clients['weather'] = WeatherAPI(weather_key, self.optimizer)
            self.logger.info("Weather API client initialized")
    
    def get_fixtures(self, date: str = None, source: str = 'all') -> List[Dict]:
        """Get fixtures from specified sources"""
        all_fixtures = []
        
        if source == 'all' or source == 'football_data':
            if 'football_data' in self.clients:
                try:
                    fixtures = self.clients['football_data'].get_todays_fixtures()
                    all_fixtures.extend(fixtures)
                except Exception as e:
                    self.logger.error(f"Football Data API failed: {e}")
        
        if source == 'all' or source == 'api_football':
            if 'api_football' in self.clients:
                try:
                    fixtures = self.clients['api_football'].get_fixtures(date)
                    all_fixtures.extend(fixtures)
                except Exception as e:
                    self.logger.error(f"API-Football failed: {e}")
        
        # Deduplicate fixtures
        return self.deduplicate_fixtures(all_fixtures)
    
    def deduplicate_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures"""
        unique_fixtures = {}
        
        for fixture in fixtures:
            # Create unique key
            home_team = fixture.get('home_team', '').lower().strip()
            away_team = fixture.get('away_team', '').lower().strip()
            date_str = fixture.get('date', '').split('T')[0] if fixture.get('date') else ''
            
            key = f"{home_team}_{away_team}_{date_str}"
            
            if key not in unique_fixtures:
                unique_fixtures[key] = fixture
            else:
                # Merge data from different sources
                existing = unique_fixtures[key]
                self.merge_fixture_data(existing, fixture)
        
        return list(unique_fixtures.values())
    
    def merge_fixture_data(self, existing: Dict, new: Dict):
        """Merge fixture data from different sources"""
        # Merge odds
        if 'odds' in new and 'odds' not in existing:
            existing['odds'] = new['odds']
        elif 'odds' in new and 'odds' in existing:
            # Prefer more detailed odds
            if isinstance(new['odds'], dict) and isinstance(existing['odds'], dict):
                existing['odds'].update(new['odds'])
        
        # Merge other data
        for key in ['score', 'predictions', 'injuries', 'lineups', 'statistics']:
            if key in new and key not in existing:
                existing[key] = new[key]
        
        # Update source
        if 'source' in existing and 'source' in new:
            if existing['source'] != new['source']:
                existing['source'] = f"{existing['source']},{new['source']}"
    
    async def enrich_fixtures_async(self, fixtures: List[Dict]) -> List[Dict]:
        """Enrich fixtures with additional data asynchronously"""
        if not fixtures:
            return fixtures
        
        # Limit concurrent processing based on hardware
        max_concurrent = self.optimizer.optimization_config.max_parallel_processes
        
        enriched_fixtures = []
        
        for i in range(0, len(fixtures), max_concurrent):
            batch = fixtures[i:i + max_concurrent]
            batch_tasks = []
            
            for fixture in batch:
                task = self.enrich_fixture_async(fixture)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for fixture, result in zip(batch, batch_results):
                if not isinstance(result, Exception):
                    enriched_fixtures.append(result)
                else:
                    enriched_fixtures.append(fixture)
                    self.logger.warning(f"Failed to enrich fixture: {result}")
        
        return enriched_fixtures
    
    async def enrich_fixture_async(self, fixture: Dict) -> Dict:
        """Enrich a single fixture asynchronously"""
        fixture_id = fixture.get('id')
        
        if not fixture_id or 'api_football' not in self.clients:
            return fixture
        
        try:
            # Get additional data
            fixture_data = await self.clients['api_football'].get_fixture_data_async(fixture_id)
            
            # Update fixture with new data
            if fixture_data.get('odds'):
                fixture['odds'] = fixture_data['odds']
            if fixture_data.get('injuries'):
                fixture['injuries'] = fixture_data['injuries']
            if fixture_data.get('lineups'):
                fixture['lineups'] = fixture_data['lineups']
            if fixture_data.get('statistics'):
                fixture['statistics'] = fixture_data['statistics']
            
            # Get weather if venue info available
            if 'venue' in fixture and 'weather' in self.clients:
                venue = fixture['venue']
                if 'city' in venue:
                    match_time = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                    weather = self.clients['weather'].get_weather(
                        venue['city'], 
                        venue.get('country'),
                        match_time
                    )
                    if weather:
                        fixture['weather'] = weather
            
            return fixture
            
        except Exception as e:
            self.logger.warning(f"Failed to enrich fixture {fixture_id}: {e}")
            return fixture
    
    def cleanup_all_cache(self):
        """Clean up cache for all clients"""
        for name, client in self.clients.items():
            if hasattr(client, 'cleanup_cache'):
                try:
                    client.cleanup_cache()
                    self.logger.info(f"Cache cleaned up for {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup cache for {name}: {e}")
    
    def get_status_report(self) -> Dict:
        """Get API manager status report"""
        report = {
            'clients_initialized': list(self.clients.keys()),
            'total_clients': len(self.clients),
            'optimization': {
                'max_concurrent': self.optimizer.optimization_config.max_parallel_processes,
                'use_cache': True,
                'cache_compression': self.optimizer.optimization_config.use_compression
            }
        }
        
        # Add cache info
        cache_info = {}
        for name, client in self.clients.items():
            if hasattr(client, 'cache_dir'):
                cache_dir = client.cache_dir
                if cache_dir.exists():
                    cache_files = list(cache_dir.glob('*.cache'))
                    cache_info[name] = {
                        'file_count': len(cache_files),
                        'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
                    }
        
        report['cache_info'] = cache_info
        
        return report
