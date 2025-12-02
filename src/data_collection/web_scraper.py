"""
Web scraping module with hardware optimization
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional, Any, Tuple
import logging
from fake_useragent import UserAgent
import cloudscraper
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
from datetime import datetime
import hashlib
import pickle
from pathlib import Path
import lz4.frame
from ..system.optimizer import SystemOptimizer

class BaseScraper:
    """Base class for web scrapers with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer, use_selenium: bool = False):
        self.optimizer = optimizer
        self.use_selenium = use_selenium
        self.ua = UserAgent()
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_dir = Path("./data/cache/scraping")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = self.get_cache_ttl()
        self.use_compression = optimizer.optimization_config.use_compression
        
        # Setup request limits based on hardware
        self.max_concurrent = optimizer.optimization_config.max_parallel_processes
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Initialize cloudscraper
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
        
        # Initialize Selenium if needed
        if use_selenium and self.should_use_selenium():
            self.driver = self.setup_selenium()
        else:
            self.driver = None
    
    def get_cache_ttl(self) -> int:
        """Get cache TTL based on hardware profile"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        ttl_config = {
            'low_end': 10800,   # 3 hours
            'mid_end': 7200,    # 2 hours
            'high_end': 3600,   # 1 hour
            'custom': 7200
        }
        return ttl_config.get(profile, 7200)
    
    def should_use_selenium(self) -> bool:
        """Determine if Selenium should be used based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        # Don't use Selenium on low-end hardware
        if profile == 'low_end':
            return False
        
        # Check available memory
        memory_gb = self.optimizer.hardware_info.get('memory', {}).get('total_gb', 8)
        if memory_gb <= 4:
            return False
        
        # Check if Chrome/Chromium is available
        try:
            from selenium.webdriver.chrome.service import Service
            import subprocess
            # Try to find Chrome
            if self.optimizer.hardware_info.get('os', {}).get('system') == 'Windows':
                result = subprocess.run(['where', 'chrome.exe'], capture_output=True, text=True)
            else:
                result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
            
            return result.returncode == 0
        except:
            return False
    
    def setup_selenium(self):
        """Setup Selenium WebDriver with hardware optimization"""
        chrome_options = Options()
        
        # Basic options
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={self.ua.random}")
        
        # Hardware optimization
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            # Minimal settings for low-end hardware
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-background-networking")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        elif profile == 'mid_end':
            # Balanced settings
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        else:
            # High-end settings
            chrome_options.add_argument("--disable-gpu")  # Still disable GPU in headless
        
        # Memory optimization
        memory_gb = self.optimizer.hardware_info.get('memory', {}).get('total_gb', 8)
        if memory_gb <= 8:
            chrome_options.add_argument("--memory-pressure-off")
            chrome_options.add_argument("--disable-background-timer-throttling")
        
        # Stealth options
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            # Try to use ChromeDriver from PATH
            driver = webdriver.Chrome(options=chrome_options)
            
            # Execute stealth JS
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": self.ua.random
            })
            
            self.logger.info("Selenium WebDriver initialized with hardware optimization")
            return driver
            
        except WebDriverException as e:
            self.logger.warning(f"Failed to initialize Selenium: {e}")
            return None
    
    def get_headers(self) -> Dict:
        """Get random headers to avoid detection"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
    
    def _get_cache_key(self, url: str, params: Dict = None) -> str:
        """Generate cache key from URL and parameters"""
        cache_data = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
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
    
    def fetch_with_selenium(self, url: str, wait_for: str = None, 
                          wait_time: int = 10) -> Optional[str]:
        """Fetch webpage using Selenium"""
        if not self.driver:
            return None
        
        try:
            self.driver.get(url)
            
            # Wait for specific element if specified
            if wait_for:
                try:
                    wait = WebDriverWait(self.driver, wait_time)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for)))
                except TimeoutException:
                    self.logger.warning(f"Timeout waiting for {wait_for} on {url}")
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            return self.driver.page_source
            
        except Exception as e:
            self.logger.error(f"Selenium fetch failed for {url}: {e}")
            return None
    
    def fetch_with_requests(self, url: str, params: Dict = None) -> Optional[str]:
        """Fetch webpage using requests/cloudscraper"""
        headers = self.get_headers()
        
        try:
            # Adjust timeout based on hardware
            profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
            timeout = 60 if profile == 'low_end' else 45
            
            response = self.scraper.get(
                url, 
                headers=headers, 
                params=params, 
                timeout=timeout
            )
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Requests fetch failed for {url}: {e}")
            return None
    
    def fetch_page(self, url: str, use_cache: bool = True, 
                  use_selenium: bool = None) -> Optional[str]:
        """Fetch webpage with caching"""
        if use_selenium is None:
            use_selenium = self.use_selenium
        
        cache_key = self._get_cache_key(url)
        
        if use_cache:
            cached_content = self._load_from_cache(cache_key)
            if cached_content is not None:
                self.logger.debug(f"Cache hit for {url}")
                return cached_content
        
        # Fetch page
        if use_selenium and self.driver:
            content = self.fetch_with_selenium(url)
        else:
            content = self.fetch_with_requests(url)
        
        if content and use_cache:
            self._save_to_cache(cache_key, content)
        
        return content
    
    async def fetch_page_async(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Asynchronously fetch webpage"""
        async with self.semaphore:
            return await self._fetch_page_async_impl(url, use_cache)
    
    async def _fetch_page_async_impl(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Implementation of async page fetch"""
        cache_key = self._get_cache_key(url)
        
        if use_cache:
            cached_content = self._load_from_cache(cache_key)
            if cached_content is not None:
                return cached_content
        
        # Use aiohttp for async requests
        headers = self.get_headers()
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.text()
                    
                    if use_cache and content:
                        self._save_to_cache(cache_key, content)
                    
                    return content
                    
        except Exception as e:
            self.logger.error(f"Async fetch failed for {url}: {e}")
            return None
    
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
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

class BettingSiteScraper(BaseScraper):
    """Scraper for betting sites with hardware optimization"""
    
    def __init__(self, optimizer: SystemOptimizer):
        # Only use Selenium on mid/high-end hardware
        use_selenium = optimizer.hardware_info.get('hardware_profile') != 'low_end'
        super().__init__(optimizer, use_selenium=use_selenium)
        
        self.site_configs = {
            'hollywoodbets': {
                'url': 'https://www.hollywoodbets.net/sport/soccer',
                'selectors': {
                    'match_rows': '.event',
                    'teams': '.team-name',
                    'odds': '.odds',
                    'time': '.time',
                    'league': '.league-name'
                },
                'requires_js': True,
                'wait_for': '.event'
            },
            'betway': {
                'url': 'https://www.betway.co.za/sport/soccer',
                'selectors': {
                    'match_rows': '.events-list__item',
                    'teams': '.participant__name',
                    'odds': '.odds__button',
                    'time': '.event-card__time',
                    'league': '.tournament-header__name'
                },
                'requires_js': True,
                'wait_for': '.events-list__item'
            },
            'sportybet': {
                'url': 'https://www.sportybet.com/za/sport/football',
                'selectors': {
                    'match_rows': '.match',
                    'teams': '.team',
                    'odds': '.odds',
                    'time': '.time',
                    'league': '.tournament'
                },
                'requires_js': True,
                'wait_for': '.match'
            },
            'bet365': {
                'url': 'https://www.bet365.com/#/AC/B1/C1/D8/E877/F10/',
                'selectors': {
                    'match_rows': '.src-MarketGroup',
                    'teams': '.src-ParticipantFixtureDetailsHigher_Team',
                    'odds': '.src-ParticipantOddsOnly80_Odds',
                    'time': '.src-ParticipantFixtureDetailsHigher_Time',
                    'league': '.src-MarketGroup_Header'
                },
                'requires_js': True,
                'wait_for': '.src-MarketGroup'
            }
        }
        
        # Limit sites to scrape based on hardware
        self.active_sites = self.select_active_sites()
    
    def select_active_sites(self) -> List[str]:
        """Select which sites to scrape based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            return ['hollywoodbets', 'betway']  # Only essential sites
        elif profile == 'mid_end':
            return ['hollywoodbets', 'betway', 'sportybet']
        else:
            return list(self.site_configs.keys())  # All sites
    
    def scrape_all_sites(self) -> List[Dict]:
        """Scrape all active betting sites"""
        all_odds = []
        
        for site_name in self.active_sites:
            try:
                self.logger.info(f"Scraping {site_name}")
                odds = self.scrape_site(site_name)
                
                if odds:
                    all_odds.extend(odds)
                    self.logger.info(f"Found {len(odds)} matches on {site_name}")
                else:
                    self.logger.warning(f"No matches found on {site_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to scrape {site_name}: {e}")
        
        return all_odds
    
    def scrape_site(self, site_name: str) -> List[Dict]:
        """Scrape specific betting site"""
        config = self.site_configs[site_name]
        
        # Check cache first
        cache_key = self._get_cache_key(f"site_{site_name}")
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            self.logger.debug(f"Using cached data for {site_name}")
            return cached_data
        
        # Fetch page
        use_selenium = config.get('requires_js', False) and self.driver is not None
        wait_for = config.get('wait_for')
        
        content = self.fetch_page(
            config['url'], 
            use_cache=False,
            use_selenium=use_selenium
        )
        
        if not content:
            return []
        
        # Parse content
        soup = BeautifulSoup(content, 'html.parser')
        matches = self.parse_matches(soup, config['selectors'], site_name)
        
        # Cache results
        if matches:
            self._save_to_cache(cache_key, matches)
        
        return matches
    
    def parse_matches(self, soup: BeautifulSoup, selectors: Dict, site_name: str) -> List[Dict]:
        """Parse matches from HTML"""
        matches = []
        match_rows = soup.select(selectors['match_rows'])
        
        # Limit number of matches based on hardware
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        max_matches = 100 if profile == 'low_end' else 200 if profile == 'mid_end' else 500
        
        for row in match_rows[:max_matches]:
            try:
                match_data = self.parse_match_row(row, selectors, site_name)
                if match_data:
                    matches.append(match_data)
            except Exception as e:
                self.logger.debug(f"Failed to parse match row: {e}")
                continue
        
        return matches
    
    def parse_match_row(self, row, selectors: Dict, site_name: str) -> Optional[Dict]:
        """Parse individual match row"""
        try:
            # Extract teams
            team_elements = row.select(selectors['teams'])
            if len(team_elements) < 2:
                return None
            
            home_team = self.clean_text(team_elements[0].text)
            away_team = self.clean_text(team_elements[1].text)
            
            # Extract odds
            odds_elements = row.select(selectors['odds'])
            if len(odds_elements) < 3:
                return None
            
            home_odds = self.parse_odd(odds_elements[0].text)
            draw_odds = self.parse_odd(odds_elements[1].text)
            away_odds = self.parse_odd(odds_elements[2].text)
            
            # Skip if odds are invalid
            if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
                return None
            
            # Extract time
            time_element = row.select_one(selectors['time'])
            match_time = self.clean_text(time_element.text) if time_element else None
            
            # Extract league
            league_element = row.select_one(selectors.get('league', ''))
            league = self.clean_text(league_element.text) if league_element else 'Unknown'
            
            # Calculate implied probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0
            draw_prob = 1 / draw_odds if draw_odds > 0 else 0
            away_prob = 1 / away_odds if away_odds > 0 else 0
            
            # Adjust for bookmaker margin
            total_prob = home_prob + draw_prob + away_prob
            if total_prob > 0:
                home_prob = home_prob / total_prob
                draw_prob = draw_prob / total_prob
                away_prob = away_prob / total_prob
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'match_time': match_time,
                'odds': {
                    'home_win': home_odds,
                    'draw': draw_odds,
                    'away_win': away_odds,
                    'probabilities': {
                        'home': home_prob,
                        'draw': draw_prob,
                        'away': away_prob
                    }
                },
                'source': site_name,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.debug(f"Error parsing match row on {site_name}: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ''
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.strip().split())
        
        # Remove non-breaking spaces
        text = text.replace('\xa0', ' ')
        
        return text
    
    def parse_odd(self, odd_text: str) -> float:
        """Parse odd text to float"""
        try:
            # Remove non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', odd_text)
            if cleaned:
                return float(cleaned)
            return 0.0
        except:
            return 0.0
    
    async def scrape_sites_async(self) -> List[Dict]:
        """Scrape all sites asynchronously"""
        tasks = []
        
        for site_name in self.active_sites:
            task = self.scrape_site_async(site_name)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_odds = []
        for site_name, result in zip(self.active_sites, results):
            if not isinstance(result, Exception):
                if result:
                    all_odds.extend(result)
                    self.logger.info(f"Found {len(result)} matches on {site_name}")
            else:
                self.logger.error(f"Failed to scrape {site_name}: {result}")
        
        return all_odds
    
    async def scrape_site_async(self, site_name: str) -> List[Dict]:
        """Scrape site asynchronously"""
        config = self.site_configs[site_name]
        
        # Check cache
        cache_key = self._get_cache_key(f"site_{site_name}")
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Fetch page
        content = await self.fetch_page_async(config['url'])
        
        if not content:
            return []
        
        # Parse content
        soup = BeautifulSoup(content, 'html.parser')
        matches = self.parse_matches(soup, config['selectors'], site_name)
        
        # Cache results
        if matches:
            self._save_to_cache(cache_key, matches)
        
        return matches

class FootballStatsScraper(BaseScraper):
    """Scraper for football statistics sites"""
    
    def __init__(self, optimizer: SystemOptimizer):
        super().__init__(optimizer, use_selenium=False)
        
        self.stats_sites = {
            'sofascore': {
                'base_url': 'https://www.sofascore.com',
                'requires_js': False
            },
            'flashscore': {
                'base_url': 'https://www.flashscore.com',
                'requires_js': True
            },
            'whoscored': {
                'base_url': 'https://www.whoscored.com',
                'requires_js': True
            }
        }
    
    async def get_player_stats(self, player_id: str, site: str = 'sofascore') -> Dict:
        """Get player statistics from specified site"""
        if site not in self.stats_sites:
            self.logger.error(f"Unsupported stats site: {site}")
            return {}
        
        config = self.stats_sites[site]
        url = f"{config['base_url']}/player/{player_id}"
        
        # Check cache
        cache_key = self._get_cache_key(f"player_{site}_{player_id}")
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Fetch page
        content = await self.fetch_page_async(url)
        
        if not content:
            return {}
        
        # Parse based on site
        if site == 'sofascore':
            stats = self.parse_sofascore_player(content)
        elif site == 'whoscored':
            stats = self.parse_whoscored_player(content)
        else:
            stats = {}
        
        # Cache results
        if stats:
            stats['source'] = site
            stats['player_id'] = player_id
            stats['scraped_at'] = datetime.now().isoformat()
            self._save_to_cache(cache_key, stats)
        
        return stats
    
    def parse_sofascore_player(self, html: str) -> Dict:
        """Parse player stats from SofaScore"""
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {}
        
        # Extract basic info
        name_elem = soup.select_one('.sc-hKMtZM')
        if name_elem:
            stats['name'] = name_elem.text.strip()
        
        # Extract rating
        rating_elem = soup.select_one('.sc-cBdUnI')
        if rating_elem:
            try:
                stats['rating'] = float(rating_elem.text.strip())
            except:
                pass
        
        # Extract position
        position_elem = soup.select_one('.sc-kDvujY')
        if position_elem:
            stats['position'] = position_elem.text.strip()
        
        # Extract recent performances
        recent_matches = []
        match_rows = soup.select('.sc-eDvSVe')[:10]  # Last 10 matches
        
        for row in match_rows:
            match_data = self.parse_sofascore_match(row)
            if match_data:
                recent_matches.append(match_data)
        
        if recent_matches:
            stats['recent_matches'] = recent_matches
        
        return stats
    
    def parse_sofascore_match(self, row) -> Optional[Dict]:
        """Parse individual match from SofaScore"""
        try:
            # Extract match details
            # This is simplified and would need adjustment based on actual HTML structure
            return {
                'date': 'N/A',
                'rating': 0.0,
                'goals': 0,
                'assists': 0,
                'minutes_played': 0
            }
        except:
            return None
    
    def parse_whoscored_player(self, html: str) -> Dict:
        """Parse player stats from WhoScored"""
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {}
        
        # WhoScored has different structure
        # This would need to be implemented based on actual HTML
        
        return stats
    
    async def get_team_stats(self, team_id: str, site: str = 'sofascore') -> Dict:
        """Get team statistics"""
        if site not in self.stats_sites:
            self.logger.error(f"Unsupported stats site: {site}")
            return {}
        
        config = self.stats_sites[site]
        url = f"{config['base_url']}/team/{team_id}"
        
        # Check cache
        cache_key = self._get_cache_key(f"team_{site}_{team_id}")
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Fetch page
        content = await self.fetch_page_async(url)
        
        if not content:
            return {}
        
        # Parse based on site
        if site == 'sofascore':
            stats = self.parse_sofascore_team(content)
        else:
            stats = {}
        
        # Cache results
        if stats:
            stats['source'] = site
            stats['team_id'] = team_id
            stats['scraped_at'] = datetime.now().isoformat()
            self._save_to_cache(cache_key, stats)
        
        return stats
    
    def parse_sofascore_team(self, html: str) -> Dict:
        """Parse team stats from SofaScore"""
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {}
        
        # Extract form
        form_elem = soup.select_one('.sc-bcXHqe')
        if form_elem:
            form_text = form_elem.text.strip()
            stats['current_form'] = [result for result in form_text.split() if result]
        
        # Extract standings
        standing_elem = soup.select_one('.sc-hLBbgP')
        if standing_elem:
            try:
                stats['league_position'] = int(standing_elem.text.strip())
            except:
                pass
        
        # Extract recent results
        recent_results = []
        result_rows = soup.select('.sc-eDvSVe')[:10]
        
        for row in result_rows:
            result_data = self.parse_sofascore_result(row)
            if result_data:
                recent_results.append(result_data)
        
        if recent_results:
            stats['recent_results'] = recent_results
        
        return stats
    
    def parse_sofascore_result(self, row) -> Optional[Dict]:
        """Parse team result from SofaScore"""
        try:
            # Simplified parsing
            return {
                'date': 'N/A',
                'home_team': 'N/A',
                'away_team': 'N/A',
                'score': 'N/A',
                'result': 'N/A'
            }
        except:
            return None
    
    async def get_multiple_players_stats(self, player_ids: List[str], 
                                       site: str = 'sofascore') -> Dict[str, Dict]:
        """Get statistics for multiple players asynchronously"""
        tasks = []
        
        for player_id in player_ids:
            task = self.get_player_stats(player_id, site)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        player_stats = {}
        for player_id, result in zip(player_ids, results):
            if not isinstance(result, Exception):
                player_stats[player_id] = result
        
        return player_stats
    
    async def get_multiple_teams_stats(self, team_ids: List[str], 
                                     site: str = 'sofascore') -> Dict[str, Dict]:
        """Get statistics for multiple teams asynchronously"""
        tasks = []
        
        for team_id in team_ids:
            task = self.get_team_stats(team_id, site)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        team_stats = {}
        for team_id, result in zip(team_ids, results):
            if not isinstance(result, Exception):
                team_stats[team_id] = result
        
        return team_stats

class ScraperManager:
    """Manager for all scrapers with hardware optimization"""
    
    def __init__(self, config: Dict, optimizer: SystemOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        
        # Initialize scrapers
        self.betting_scraper = BettingSiteScraper(optimizer)
        self.stats_scraper = FootballStatsScraper(optimizer)
        
        # Setup based on hardware
        self.setup_scraping_strategy()
    
    def setup_scraping_strategy(self):
        """Setup scraping strategy based on hardware"""
        profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
        
        if profile == 'low_end':
            self.logger.info("Using conservative scraping strategy (low-end hardware)")
        elif profile == 'mid_end':
            self.logger.info("Using balanced scraping strategy (mid-end hardware)")
        else:
            self.logger.info("Using aggressive scraping strategy (high-end hardware)")
    
    def scrape_betting_odds(self, use_async: bool = None) -> List[Dict]:
        """Scrape odds from betting sites"""
        if use_async is None:
            # Use async for mid/high-end hardware
            profile = self.optimizer.hardware_info.get('hardware_profile', 'mid_end')
            use_async = profile != 'low_end'
        
        if use_async:
            return asyncio.run(self.scrape_betting_odds_async())
        else:
            return self.betting_scraper.scrape_all_sites()
    
    async def scrape_betting_odds_async(self) -> List[Dict]:
        """Scrape odds asynchronously"""
        return await self.betting_scraper.scrape_sites_async()
    
    async def scrape_player_stats(self, player_ids: List[str], 
                                site: str = 'sofascore') -> Dict[str, Dict]:
        """Scrape statistics for multiple players"""
        return await self.stats_scraper.get_multiple_players_stats(player_ids, site)
    
    async def scrape_team_stats(self, team_ids: List[str], 
                              site: str = 'sofascore') -> Dict[str, Dict]:
        """Scrape statistics for multiple teams"""
        return await self.stats_scraper.get_multiple_teams_stats(team_ids, site)
    
    def cleanup_cache(self):
        """Clean up cache for all scrapers"""
        try:
            self.betting_scraper.cleanup_cache()
            self.stats_scraper.cleanup_cache()
            self.logger.info("Scraper cache cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup scraper cache: {e}")
    
    def close(self):
        """Clean up resources"""
        self.betting_scraper.close()
    
    def get_status_report(self) -> Dict:
        """Get scraper manager status report"""
        report = {
            'active_scrapers': {
                'betting': self.betting_scraper.active_sites,
                'stats': list(self.stats_scraper.stats_sites.keys())
            },
            'optimization': {
                'use_selenium': self.betting_scraper.use_selenium,
                'max_concurrent': self.optimizer.optimization_config.max_parallel_processes,
                'use_cache': True,
                'cache_compression': self.optimizer.optimization_config.use_compression
            }
        }
        
        # Add cache info
        cache_info = {}
        for scraper_name, scraper in [('betting', self.betting_scraper), 
                                      ('stats', self.stats_scraper)]:
            if hasattr(scraper, 'cache_dir'):
                cache_dir = scraper.cache_dir
                if cache_dir.exists():
                    cache_files = list(cache_dir.glob('*.cache'))
                    cache_info[scraper_name] = {
                        'file_count': len(cache_files),
                        'total_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
                    }
        
        report['cache_info'] = cache_info
        
        return report
