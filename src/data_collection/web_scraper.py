"""
Web scraper for betting sites and football statistics
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional
import logging
from fake_useragent import UserAgent
import cloudscraper
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

class BaseScraper:
    """Base class for web scrapers"""
    
    def __init__(self, use_selenium: bool = False):
        self.ua = UserAgent()
        self.use_selenium = use_selenium
        
        if use_selenium:
            self.driver = self.setup_selenium()
        
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
    
    def setup_selenium(self):
        """Setup Selenium WebDriver with options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={self.ua.random}")
        
        # Add stealth options
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute stealth JS
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
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

class BettingSiteScraper(BaseScraper):
    """Scraper for betting sites"""
    
    def __init__(self):
        super().__init__(use_selenium=True)
        
        self.site_configs = {
            'hollywoodbets': {
                'url': 'https://www.hollywoodbets.net/sport/soccer',
                'selectors': {
                    'match_rows': '.event',
                    'teams': '.team-name',
                    'odds': '.odds',
                    'time': '.time'
                }
            },
            'betway': {
                'url': 'https://www.betway.co.za/sport/soccer',
                'selectors': {
                    'match_rows': '.events-list__item',
                    'teams': '.participant__name',
                    'odds': '.odds__button',
                    'time': '.event-card__time'
                }
            },
            'sportybet': {
                'url': 'https://www.sportybet.com/za/sport/football',
                'selectors': {
                    'match_rows': '.match',
                    'teams': '.team',
                    'odds': '.odds',
                    'time': '.time'
                }
            }
        }
    
    def scrape_all_sites(self) -> List[Dict]:
        """Scrape all betting sites"""
        all_odds = []
        
        for site_name, config in self.site_configs.items():
            try:
                logging.info(f"Scraping {site_name}")
                odds = self.scrape_site(site_name)
                all_odds.extend(odds)
                
            except Exception as e:
                logging.error(f"Failed to scrape {site_name}: {e}")
        
        return all_odds
    
    def scrape_site(self, site_name: str) -> List[Dict]:
        """Scrape specific betting site"""
        config = self.site_configs[site_name]
        
        if self.use_selenium:
            return self.scrape_with_selenium(config)
        else:
            return self.scrape_with_requests(config)
    
    def scrape_with_selenium(self, config: Dict) -> List[Dict]:
        """Scrape using Selenium for JavaScript-rendered content"""
        try:
            self.driver.get(config['url'])
            
            # Wait for content to load
            wait = WebDriverWait(self.driver, 20)
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, config['selectors']['match_rows'])
                )
            )
            
            # Scroll to load all content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Parse page source
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return self.parse_matches(soup, config['selectors'])
            
        except TimeoutException:
            logging.error("Timeout waiting for page to load")
            return []
    
    def scrape_with_requests(self, config: Dict) -> List[Dict]:
        """Scrape using requests for static content"""
        headers = self.get_headers()
        response = self.scraper.get(config['url'], headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return self.parse_matches(soup, config['selectors'])
        else:
            logging.error(f"Failed to fetch page: {response.status_code}")
            return []
    
    def parse_matches(self, soup: BeautifulSoup, selectors: Dict) -> List[Dict]:
        """Parse matches from HTML"""
        matches = []
        match_rows = soup.select(selectors['match_rows'])
        
        for row in match_rows[:50]:  # Limit to 50 matches
            try:
                match_data = self.parse_match_row(row, selectors)
                if match_data:
                    matches.append(match_data)
            except Exception as e:
                logging.debug(f"Failed to parse match row: {e}")
                continue
        
        return matches
    
    def parse_match_row(self, row, selectors: Dict) -> Optional[Dict]:
        """Parse individual match row"""
        try:
            # Extract teams
            team_elements = row.select(selectors['teams'])
            if len(team_elements) < 2:
                return None
            
            home_team = team_elements[0].text.strip()
            away_team = team_elements[1].text.strip()
            
            # Extract odds
            odds_elements = row.select(selectors['odds'])
            if len(odds_elements) < 3:
                return None
            
            home_odds = self.parse_odd(odds_elements[0].text)
            draw_odds = self.parse_odd(odds_elements[1].text)
            away_odds = self.parse_odd(odds_elements[2].text)
            
            # Extract time if available
            time_element = row.select_one(selectors['time'])
            match_time = time_element.text.strip() if time_element else None
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'odds': {
                    'home_win': home_odds,
                    'draw': draw_odds,
                    'away_win': away_odds
                },
                'match_time': match_time,
                'source': self.site_name
            }
            
        except Exception as e:
            logging.debug(f"Error parsing match row: {e}")
            return None
    
    def parse_odd(self, odd_text: str) -> float:
        """Parse odd text to float"""
        try:
            return float(re.sub(r'[^\d.]', '', odd_text))
        except:
            return 0.0

class FootballStatsScraper(BaseScraper):
    """Scraper for football statistics sites"""
    
    def __init__(self):
        super().__init__(use_selenium=False)
        
        self.stats_sites = {
            'sofascore': 'https://www.sofascore.com',
            'flashscore': 'https://www.flashscore.com',
            'whoscored': 'https://www.whoscored.com',
            'soccerstats': 'https://www.soccerstats.com'
        }
    
    async def fetch_player_stats(self, player_id: str) -> Dict:
        """Fetch player statistics asynchronously"""
        url = f"{self.stats_sites['sofascore']}/player/{player_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    html = await response.text()
                    return self.parse_player_stats(html)
                else:
                    return {}
    
    def parse_player_stats(self, html: str) -> Dict:
        """Parse player statistics from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {}
        
        # Extract basic info
        name_elem = soup.select_one('.player-summary__name')
        if name_elem:
            stats['name'] = name_elem.text.strip()
        
        # Extract performance metrics
        rating_elem = soup.select_one('.sc-rating__value')
        if rating_elem:
            stats['rating'] = float(rating_elem.text.strip())
        
        # Extract recent performances
        recent_matches = []
        match_rows = soup.select('.sc-fEUNkw')
        
        for row in match_rows[:5]:
            match_data = self.parse_match_performance(row)
            if match_data:
                recent_matches.append(match_data)
        
        stats['recent_matches'] = recent_matches
        
        return stats
    
    def parse_match_performance(self, row) -> Optional[Dict]:
        """Parse individual match performance"""
        try:
            # Extract match details
            # This would need to be customized based on actual site structure
            return {
                'date': row.select_one('.sc-kDvujY').text if row.select_one('.sc-kDvujY') else None,
                'rating': float(row.select_one('.sc-jXbUNg').text) if row.select_one('.sc-jXbUNg') else None,
                'goals': int(row.select_one('.sc-ikJyIC').text) if row.select_one('.sc-ikJyIC') else 0,
                'assists': int(row.select_one('.sc-hLBbgP').text) if row.select_one('.sc-hLBbgP') else 0
            }
        except:
            return None
    
    async def fetch_team_stats(self, team_id: str) -> Dict:
        """Fetch team statistics asynchronously"""
        url = f"{self.stats_sites['sofascore']}/team/{team_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    html = await response.text()
                    return self.parse_team_stats(html)
                else:
                    return {}
    
    def parse_team_stats(self, html: str) -> Dict:
        """Parse team statistics from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        stats = {}
        
        # Extract form
        form_elem = soup.select_one('.sc-bcXHqe')
        if form_elem:
            form_text = form_elem.text.strip()
            stats['current_form'] = [result for result in form_text.split()]
        
        # Extract standings if available
        standing_elem = soup.select_one('.sc-hLBbgP')
        if standing_elem:
            stats['league_position'] = int(standing_elem.text.strip())
        
        # Extract recent results
        recent_results = []
        result_rows = soup.select('.sc-eDvSVe')
        
        for row in result_rows[:10]:
            result_data = self.parse_team_result(row)
            if result_data:
                recent_results.append(result_data)
        
        stats['recent_results'] = recent_results
        
        return stats
    
    def parse_team_result(self, row) -> Optional[Dict]:
        """Parse individual team result"""
        try:
            return {
                'date': row.select_one('.sc-jXbUNg').text if row.select_one('.sc-jXbUNg') else None,
                'home_team': row.select_one('.sc-ikJyIC').text if row.select_one('.sc-ikJyIC') else None,
                'away_team': row.select_one('.sc-hLBbgP').text if row.select_one('.sc-hLBbgP') else None,
                'score': row.select_one('.sc-kDvujY').text if row.select_one('.sc-kDvujY') else None,
                'result': row.select_one('.sc-fEUNkw').text if row.select_one('.sc-fEUNkw') else None
            }
        except:
            return None

class ScraperManager:
    """Manager for all scrapers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.betting_scraper = BettingSiteScraper()
        self.stats_scraper = FootballStatsScraper()
        
    def scrape_betting_odds(self) -> List[Dict]:
        """Scrape odds from all betting sites"""
        return self.betting_scraper.scrape_all_sites()
    
    async def scrape_player_stats(self, player_ids: List[str]) -> Dict[str, Dict]:
        """Scrape statistics for multiple players"""
        tasks = []
        for player_id in player_ids:
            task = self.stats_scraper.fetch_player_stats(player_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        player_stats = {}
        for player_id, result in zip(player_ids, results):
            if not isinstance(result, Exception):
                player_stats[player_id] = result
        
        return player_stats
    
    async def scrape_team_stats(self, team_ids: List[str]) -> Dict[str, Dict]:
        """Scrape statistics for multiple teams"""
        tasks = []
        for team_id in team_ids:
            task = self.stats_scraper.fetch_team_stats(team_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        team_stats = {}
        for team_id, result in zip(team_ids, results):
            if not isinstance(result, Exception):
                team_stats[team_id] = result
        
        return team_stats
