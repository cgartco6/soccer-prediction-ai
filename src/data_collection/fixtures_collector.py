"""
Fixtures collector with hardware optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..system.optimizer import SystemOptimizer

class FixturesCollector:
    """Collect fixtures from all sources with hardware optimization"""
    
    def __init__(self, config: Dict, optimizer: SystemOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        
        # Initialize API manager and scraper manager
        from .api_clients import APIManager
        from .web_scraper import ScraperManager
        
        self.api_manager = APIManager(config, optimizer)
        self.scraper_manager = ScraperManager(config, optimizer)
        
        # Setup data storage
        self.data_dir = Path("./data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.collection_stats = {
            'api_calls': 0,
            'scraping_sessions': 0,
            'total_fixtures': 0,
            'collection_time': 0
        }
    
    def collect_todays_fixtures(self, use_cache: bool = True) -> List[Dict]:
        """Collect today's fixtures from all sources"""
        start_time = datetime.now()
        self.logger.info("Starting fixtures collection...")
        
        all_fixtures = []
        
        try:
            # 1. Collect from APIs
            api_fixtures = self.collect_from_apis(use_cache)
            all_fixtures.extend(api_fixtures)
            self.collection_stats['api_calls'] += 1
            
            # 2. Scrape betting odds
            betting_odds = self.scrape_betting_odds()
            all_fixtures = self.merge_with_odds(all_fixtures, betting_odds)
            self.collection_stats['scraping_sessions'] += 1
            
            # 3. Enrich fixtures with additional data
            enriched_fixtures = asyncio.run(self.enrich_fixtures(all_fixtures))
            
            # 4. Validate and clean data
            validated_fixtures = self.validate_fixtures(enriched_fixtures)
            
            # 5. Save collected data
            self.save_fixtures(validated_fixtures)
            
            # Update stats
            self.collection_stats['total_fixtures'] = len(validated_fixtures)
            collection_time = (datetime.now() - start_time).total_seconds()
            self.collection_stats['collection_time'] = collection_time
            
            self.logger.info(f"Collection completed: {len(validated_fixtures)} fixtures in {collection_time:.2f}s")
            
            return validated_fixtures
            
        except Exception as e:
            self.logger.error(f"Fixtures collection failed: {e}")
            return []
    
    def collect_from_apis(self, use_cache: bool = True) -> List[Dict]:
        """Collect fixtures from APIs"""
        self.logger.info("Collecting fixtures from APIs...")
        
        fixtures = []
        
        # Get fixtures for today and tomorrow
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Determine sources based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            sources = ['football_data']  # Only most reliable source
        elif profile == 'mid_end':
            sources = ['football_data', 'api_football']
        else:
            sources = ['all']  # All available sources
        
        for source in sources:
            try:
                source_fixtures = self.api_manager.get_fixtures(today, source)
                if source == 'all':
                    # Also get tomorrow's fixtures for high-end hardware
                    tomorrow_fixtures = self.api_manager.get_fixtures(tomorrow, source)
                    source_fixtures.extend(tomorrow_fixtures)
                
                fixtures.extend(source_fixtures)
                self.logger.info(f"Collected {len(source_fixtures)} fixtures from {source}")
                
            except Exception as e:
                self.logger.warning(f"Failed to collect from {source}: {e}")
        
        return fixtures
    
    def scrape_betting_odds(self) -> List[Dict]:
        """Scrape betting odds"""
        self.logger.info("Scraping betting odds...")
        
        try:
            # Use async scraping for mid/high-end hardware
            profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
            use_async = profile != 'low_end'
            
            odds = self.scraper_manager.scrape_betting_odds(use_async=use_async)
            self.logger.info(f"Scraped {len(odds)} betting odds")
            
            return odds
            
        except Exception as e:
            self.logger.error(f"Betting odds scraping failed: {e}")
            return []
    
    def merge_with_odds(self, fixtures: List[Dict], betting_odds: List[Dict]) -> List[Dict]:
        """Merge fixtures with betting odds"""
        if not betting_odds:
            return fixtures
        
        # Create mapping for quick lookup
        odds_map = {}
        for odds in betting_odds:
            key = self.create_fixture_key(odds)
            if key not in odds_map:
                odds_map[key] = []
            odds_map[key].append(odds)
        
        # Merge odds with fixtures
        merged_fixtures = []
        
        for fixture in fixtures:
            fixture_key = self.create_fixture_key(fixture)
            
            if fixture_key in odds_map:
                # Merge all matching odds
                for odds in odds_map[fixture_key]:
                    if 'odds' not in fixture:
                        fixture['odds'] = {}
                    
                    # Update with scraped odds
                    if 'bookmaker_odds' not in fixture:
                        fixture['bookmaker_odds'] = {}
                    
                    source = odds.get('source', 'unknown')
                    fixture['bookmaker_odds'][source] = odds.get('odds', {})
                    
                    # Update league if missing
                    if not fixture.get('league') and odds.get('league'):
                        fixture['league'] = odds['league']
            
            merged_fixtures.append(fixture)
        
        return merged_fixtures
    
    def create_fixture_key(self, data: Dict) -> str:
        """Create unique key for fixture matching"""
        home_team = data.get('home_team', '').lower().strip()
        away_team = data.get('away_team', '').lower().strip()
        
        # Clean team names for better matching
        home_team = self.clean_team_name(home_team)
        away_team = self.clean_team_name(away_team)
        
        return f"{home_team}_{away_team}"
    
    def clean_team_name(self, team_name: str) -> str:
        """Clean team name for matching"""
        if not team_name:
            return ''
        
        # Remove common prefixes/suffixes
        removals = ['fc', 'cf', 'afc', 'cfc', 'sc', 'ssc', 'as', 'us', 'sv', 'fk', 'bk', 
                   'ac', 'fc.', 'cf.', 'afc.', 'cfc.']
        
        cleaned = team_name.lower()
        
        # Remove in any position
        for removal in removals:
            pattern = f'\\b{removal}\\b'
            import re
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove extra spaces and special characters
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    async def enrich_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Enrich fixtures with additional data"""
        if not fixtures:
            return fixtures
        
        self.logger.info(f"Enriching {len(fixtures)} fixtures...")
        
        # Enrich with API data
        enriched_fixtures = await self.api_manager.enrich_fixtures_async(fixtures)
        
        # Additional enrichment based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile != 'low_end':
            # Get player/team stats for high-priority fixtures
            priority_fixtures = self.select_priority_fixtures(enriched_fixtures)
            
            if priority_fixtures:
                # Extract team IDs
                team_ids = []
                for fixture in priority_fixtures:
                    if fixture.get('home_id'):
                        team_ids.append(str(fixture['home_id']))
                    if fixture.get('away_id'):
                        team_ids.append(str(fixture['away_id']))
                
                # Get team stats
                if team_ids:
                    team_stats = await self.scraper_manager.scrape_team_stats(team_ids[:10])  # Limit
                    
                    # Merge stats with fixtures
                    for fixture in priority_fixtures:
                        home_id = str(fixture.get('home_id', ''))
                        away_id = str(fixture.get('away_id', ''))
                        
                        if home_id in team_stats:
                            if 'team_stats' not in fixture:
                                fixture['team_stats'] = {}
                            fixture['team_stats']['home'] = team_stats[home_id]
                        
                        if away_id in team_stats:
                            if 'team_stats' not in fixture:
                                fixture['team_stats'] = {}
                            fixture['team_stats']['away'] = team_stats[away_id]
        
        return enriched_fixtures
    
    def select_priority_fixtures(self, fixtures: List[Dict], max_count: int = 20) -> List[Dict]:
        """Select priority fixtures for detailed enrichment"""
        # Sort by league importance and match time
        league_priority = {
            'PL': 10, 'PD': 9, 'SA': 8, 'BL1': 7, 'L1': 6,  # Top leagues
            'CL': 15, 'EL': 14, 'ECL': 13,  # European competitions
            'WC': 20, 'Q': 18  # World Cup and qualifiers
        }
        
        prioritized = []
        
        for fixture in fixtures:
            priority = 0
            
            # League priority
            league = fixture.get('league', '')
            priority += league_priority.get(league, 0)
            
            # Time priority (sooner matches have higher priority)
            match_time = fixture.get('date')
            if match_time:
                try:
                    match_dt = datetime.fromisoformat(match_time.replace('Z', '+00:00'))
                    hours_until = (match_dt - datetime.now()).total_seconds() / 3600
                    if 0 <= hours_until <= 48:  # Next 48 hours
                        priority += 20 - (hours_until / 2.4)  # More priority for sooner matches
                except:
                    pass
            
            prioritized.append((priority, fixture))
        
        # Sort by priority and take top N
        prioritized.sort(key=lambda x: x[0], reverse=True)
        
        return [fixture for _, fixture in prioritized[:max_count]]
    
    def validate_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Validate and clean fixtures"""
        valid_fixtures = []
        
        for fixture in fixtures:
            try:
                # Required fields
                if not all(key in fixture for key in ['home_team', 'away_team', 'date']):
                    continue
                
                # Validate teams
                if not fixture['home_team'].strip() or not fixture['away_team'].strip():
                    continue
                
                # Validate date
                try:
                    datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                except:
                    continue
                
                # Add validation timestamp
                fixture['validated_at'] = datetime.now().isoformat()
                fixture['data_quality'] = self.calculate_data_quality(fixture)
                
                valid_fixtures.append(fixture)
                
            except Exception as e:
                self.logger.debug(f"Fixture validation failed: {e}")
                continue
        
        # Remove duplicates
        unique_fixtures = self.remove_duplicates(valid_fixtures)
        
        return unique_fixtures
    
    def calculate_data_quality(self, fixture: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        
        # Basic data (0.3 max)
        if fixture.get('home_team') and fixture.get('away_team'):
            score += 0.2
        if fixture.get('date'):
            score += 0.1
        
        # Additional data (0.7 max)
        if fixture.get('odds') or fixture.get('bookmaker_odds'):
            score += 0.2
        if fixture.get('league'):
            score += 0.1
        if fixture.get('home_id') and fixture.get('away_id'):
            score += 0.1
        if fixture.get('weather'):
            score += 0.1
        if fixture.get('injuries'):
            score += 0.1
        if fixture.get('team_stats'):
            score += 0.1
        
        return min(score, 1.0)
    
    def remove_duplicates(self, fixtures: List[Dict]) -> List[Dict]:
        """Remove duplicate fixtures"""
        unique_fixtures = {}
        
        for fixture in fixtures:
            key = self.create_fixture_key(fixture)
            
            if key not in unique_fixtures:
                unique_fixtures[key] = fixture
            else:
                # Keep the one with higher data quality
                existing = unique_fixtures[key]
                existing_quality = existing.get('data_quality', 0)
                new_quality = fixture.get('data_quality', 0)
                
                if new_quality > existing_quality:
                    unique_fixtures[key] = fixture
        
        return list(unique_fixtures.values())
    
    def save_fixtures(self, fixtures: List[Dict]):
        """Save fixtures to file"""
        if not fixtures:
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.data_dir / f"fixtures_{timestamp}.json"
        
        # Prepare data for saving
        save_data = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'fixture_count': len(fixtures),
                'hardware_profile': self.optimizer.hardware_info.get('hardware_profile'),
                'collection_stats': self.collection_stats
            },
            'fixtures': fixtures
        }
        
        # Save with compression based on hardware
        use_compression = self.optimizer.optimization_config.use_compression
        
        try:
            if use_compression:
                import lz4.frame
                with lz4.frame.open(filename.with_suffix('.json.lz4'), 'wb') as f:
                    f.write(json.dumps(save_data, indent=2, default=str).encode())
            else:
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(fixtures)} fixtures to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save fixtures: {e}")
    
    def load_fixtures(self, date: str = None) -> List[Dict]:
        """Load fixtures from file for a specific date"""
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        
        pattern = f"fixtures_{date}*.json"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            # Try compressed files
            pattern = f"fixtures_{date}*.json.lz4"
            files = list(self.data_dir.glob(pattern))
        
        if not files:
            return []
        
        # Get latest file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        try:
            if latest_file.suffix == '.lz4':
                import lz4.frame
                with lz4.frame.open(latest_file, 'rb') as f:
                    data = json.loads(f.read().decode())
            else:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
            
            return data.get('fixtures', [])
            
        except Exception as e:
            self.logger.error(f"Failed to load fixtures from {latest_file}: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old data files"""
        try:
            current_time = datetime.now()
            for data_file in self.data_dir.glob("fixtures_*.json*"):
                file_age = current_time - datetime.fromtimestamp(data_file.stat().st_mtime)
                if file_age.days > days_to_keep:
                    data_file.unlink()
                    self.logger.debug(f"Removed old data file: {data_file}")
            
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.warning(f"Data cleanup failed: {e}")
    
    def get_collection_report(self) -> Dict:
        """Get collection statistics report"""
        report = {
            'collection_stats': self.collection_stats,
            'hardware_profile': self.optimizer.hardware_info.get('hardware_profile'),
            'optimization': {
                'max_concurrent': self.optimizer.optimization_config.max_parallel_processes,
                'use_compression': self.optimizer.optimization_config.use_compression,
                'cache_enabled': True
            }
        }
        
        # Add data directory info
        if self.data_dir.exists():
            data_files = list(self.data_dir.glob("fixtures_*.json*"))
            report['data_storage'] = {
                'file_count': len(data_files),
                'total_size_mb': sum(f.stat().st_size for f in data_files) / (1024 * 1024)
            }
        
        return report
