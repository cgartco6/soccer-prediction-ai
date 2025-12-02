"""
Deep match analysis module
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict

class MatchAnalyzer:
    """Perform deep analysis of football matches"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis weights
        self.weights = {
            'form': 0.25,
            'h2h': 0.20,
            'home_advantage': 0.15,
            'injuries': 0.15,
            'motivation': 0.10,
            'weather': 0.05,
            'fatigue': 0.05,
            'coaching': 0.05
        }
    
    def analyze_match(self, fixture: Dict, prediction: Dict) -> Dict:
        """Perform comprehensive match analysis"""
        analysis = {
            'key_factors': [],
            'strengths': {'home': [], 'away': []},
            'weaknesses': {'home': [], 'away': []},
            'risk_factors': [],
            'confidence_boosters': [],
            'detailed_analysis': {}
        }
        
        try:
            # 1. Team Form Analysis
            form_analysis = self.analyze_team_form(fixture)
            analysis['detailed_analysis']['form'] = form_analysis
            analysis['key_factors'].extend(form_analysis.get('key_points', []))
            
            # 2. Head-to-Head Analysis
            h2h_analysis = self.analyze_head_to_head(fixture)
            analysis['detailed_analysis']['h2h'] = h2h_analysis
            analysis['key_factors'].extend(h2h_analysis.get('key_points', []))
            
            # 3. Home Advantage Analysis
            home_analysis = self.analyze_home_advantage(fixture)
            analysis['detailed_analysis']['home_advantage'] = home_analysis
            
            # 4. Injury Impact Analysis
            injury_analysis = self.analyze_injuries(fixture)
            analysis['detailed_analysis']['injuries'] = injury_analysis
            if injury_analysis.get('significant_impact'):
                analysis['risk_factors'].extend(injury_analysis['significant_impact'])
            
            # 5. Motivation Analysis
            motivation_analysis = self.analyze_motivation(fixture)
            analysis['detailed_analysis']['motivation'] = motivation_analysis
            analysis['key_factors'].extend(motivation_analysis.get('key_points', []))
            
            # 6. Weather Impact Analysis
            weather_analysis = self.analyze_weather_impact(fixture)
            analysis['detailed_analysis']['weather'] = weather_analysis
            
            # 7. Fatigue Analysis
            fatigue_analysis = self.analyze_fatigue(fixture)
            analysis['detailed_analysis']['fatigue'] = fatigue_analysis
            
            # 8. Coaching Analysis
            coaching_analysis = self.analyze_coaching(fixture)
            analysis['detailed_analysis']['coaching'] = coaching_analysis
            
            # 9. Calculate Composite Score
            composite_score = self.calculate_composite_score(
                form_analysis, h2h_analysis, home_analysis,
                injury_analysis, motivation_analysis
            )
            analysis['composite_score'] = composite_score
            
            # 10. Generate Match Summary
            analysis['summary'] = self.generate_summary(
                fixture, composite_score, analysis['key_factors']
            )
            
            # 11. Identify Value Bets
            if fixture.get('odds'):
                value_bets = self.identify_value_bets(fixture['odds'], prediction)
                analysis['value_bets'] = value_bets
            
            # Limit key factors to most important
            analysis['key_factors'] = analysis['key_factors'][:5]
            
        except Exception as e:
            self.logger.error(f"Error in match analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def analyze_team_form(self, fixture: Dict) -> Dict:
        """Analyze current team form"""
        home_team = fixture.get('home_team', '')
        away_team = fixture.get('away_team', '')
        
        # Get form data (from fixture or database)
        home_form = fixture.get('home_form', [])
        away_form = fixture.get('away_form', [])
        
        analysis = {
            'home_form': home_form,
            'away_form': away_form,
            'home_form_rating': self.calculate_form_rating(home_form),
            'away_form_rating': self.calculate_form_rating(away_form),
            'form_momentum': None,
            'key_points': []
        }
        
        # Calculate form momentum
        if len(home_form) >= 3 and len(away_form) >= 3:
            home_momentum = self.calculate_momentum(home_form[-3:])
            away_momentum = self.calculate_momentum(away_form[-3:])
            analysis['form_momentum'] = {
                'home': home_momentum,
                'away': away_momentum
            }
            
            # Add key points
            if home_momentum > away_momentum + 0.3:
                analysis['key_points'].append(f"{home_team} has strong recent momentum")
            elif away_momentum > home_momentum + 0.3:
                analysis['key_points'].append(f"{away_team} has strong recent momentum")
        
        # Check for streaks
        home_streak = self.identify_streak(home_form)
        away_streak = self.identify_streak(away_form)
        
        if home_streak:
            analysis['key_points'].append(f"{home_team} on {home_streak} streak")
        if away_streak:
            analysis['key_points'].append(f"{away_team} on {away_streak} streak")
        
        return analysis
    
    def calculate_form_rating(self, form: List[str]) -> float:
        """Calculate numerical rating from form string (W/D/L)"""
        if not form:
            return 0.5
        
        points = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        total = sum(points.get(result.upper(), 0) for result in form)
        return total / len(form)
    
    def calculate_momentum(self, recent_form: List[str]) -> float:
        """Calculate momentum from recent form"""
        weights = [1.0, 0.7, 0.4]  # Weight recent matches more
        points = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        
        total = 0
        for i, result in enumerate(recent_form[-3:]):  # Last 3 matches
            if i < len(weights):
                total += points.get(result.upper(), 0) * weights[i]
        
        return total / sum(weights[:min(3, len(recent_form))])
    
    def identify_streak(self, form: List[str]) -> Optional[str]:
        """Identify winning or losing streaks"""
        if len(form) < 3:
            return None
        
        last_3 = form[-3:]
        if all(r == 'W' for r in last_3):
            return "3-game winning"
        elif all(r == 'L' for r in last_3):
            return "3-game losing"
        elif all(r == 'D' for r in last_3):
            return "3-game drawing"
        
        return None
    
    def analyze_head_to_head(self, fixture: Dict) -> Dict:
        """Analyze head-to-head history"""
        h2h_matches = fixture.get('h2h_history', [])
        
        analysis = {
            'total_matches': len(h2h_matches),
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'recent_trend': None,
            'key_points': []
        }
        
        if not h2h_matches:
            return analysis
        
        # Count results
        for match in h2h_matches:
            home_goals = match.get('home_goals', 0)
            away_goals = match.get('away_goals', 0)
            
            if home_goals > away_goals:
                analysis['home_wins'] += 1
            elif away_goals > home_goals:
                analysis['away_wins'] += 1
            else:
                analysis['draws'] += 1
        
        # Calculate percentages
        total = analysis['total_matches']
        if total > 0:
            analysis['home_win_pct'] = analysis['home_wins'] / total
            analysis['away_win_pct'] = analysis['away_wins'] / total
            analysis['draw_pct'] = analysis['draws'] / total
            
            # Analyze recent trend (last 5 matches)
            recent_matches = h2h_matches[-5:]
            if recent_matches:
                recent_home_wins = sum(1 for m in recent_matches 
                                      if m.get('home_goals', 0) > m.get('away_goals', 0))
                recent_away_wins = sum(1 for m in recent_matches 
                                      if m.get('away_goals', 0) > m.get('home_goals', 0))
                
                if recent_home_wins >= 3:
                    analysis['recent_trend'] = 'home_dominant'
                    analysis['key_points'].append("Home team dominant in recent H2H")
                elif recent_away_wins >= 3:
                    analysis['recent_trend'] = 'away_dominant'
                    analysis['key_points'].append("Away team dominant in recent H2H")
        
        return analysis
    
    def analyze_home_advantage(self, fixture: Dict) -> Dict:
        """Analyze home advantage factors"""
        home_team = fixture.get('home_team', '')
        
        analysis = {
            'home_record': fixture.get('home_record', {}),
            'away_record': fixture.get('away_record', {}),
            'stadium_factors': fixture.get('stadium_factors', {}),
            'travel_distance': fixture.get('travel_distance', 0),
            'key_points': []
        }
        
        # Calculate home advantage score
        home_points = analysis['home_record'].get('points_per_game', 0)
        away_points = analysis['away_record'].get('points_per_game', 0)
        
        if home_points > 0 and away_points > 0:
            home_advantage_ratio = home_points / away_points
            analysis['home_advantage_ratio'] = home_advantage_ratio
            
            if home_advantage_ratio > 1.5:
                analysis['key_points'].append(f"Strong home advantage for {home_team}")
            elif home_advantage_ratio < 0.67:
                analysis['key_points'].append(f"Weak home record for {home_team}")
        
        # Consider travel distance
        travel = analysis.get('travel_distance', 0)
        if travel > 500:  # Kilometers
            analysis['key_points'].append("Away team has significant travel")
        
        return analysis
    
    def analyze_injuries(self, fixture: Dict) -> Dict:
        """Analyze impact of injuries and suspensions"""
        injuries = fixture.get('injuries', [])
        suspensions = fixture.get('suspensions', [])
        
        analysis = {
            'home_injuries': [],
            'away_injuries': [],
            'home_suspensions': [],
            'away_suspensions': [],
            'key_player_missing': {'home': False, 'away': False},
            'significant_impact': []
        }
        
        # Categorize injuries by team
        for injury in injuries:
            player = injury.get('player', {})
            team = player.get('team', '')
            importance = player.get('importance', 0)
            
            if 'home' in team.lower() or fixture.get('home_team', '') in team:
                analysis['home_injuries'].append(injury)
                if importance >= 8:  # Key player
                    analysis['key_player_missing']['home'] = True
                    analysis['significant_impact'].append(
                        f"Home team missing key player: {player.get('name')}"
                    )
            elif 'away' in team.lower() or fixture.get('away_team', '') in team:
                analysis['away_injuries'].append(injury)
                if importance >= 8:
                    analysis['key_player_missing']['away'] = True
                    analysis['significant_impact'].append(
                        f"Away team missing key player: {player.get('name')}"
                    )
        
        # Count injuries
        analysis['home_injury_count'] = len(analysis['home_injuries'])
        analysis['away_injury_count'] = len(analysis['away_injuries'])
        
        # Calculate impact score
        analysis['injury_impact_score'] = {
            'home': min(1.0, analysis['home_injury_count'] / 5),
            'away': min(1.0, analysis['away_injury_count'] / 5)
        }
        
        return analysis
    
    def analyze_motivation(self, fixture: Dict) -> Dict:
        """Analyze team motivation factors"""
        league = fixture.get('league', '')
        home_position = fixture.get('home_league_position', 0)
        away_position = fixture.get('away_league_position', 0)
        total_teams = fixture.get('total_teams_in_league', 20)
        
        analysis = {
            'league_context': league,
            'home_position': home_position,
            'away_position': away_position,
            'relegation_battle': False,
            'european_places': False,
            'title_race': False,
            'key_points': []
        }
        
        # Check for relegation battle
        relegation_zone = total_teams - 3
        if home_position >= relegation_zone or away_position >= relegation_zone:
            analysis['relegation_battle'] = True
            analysis['key_points'].append("Teams involved in relegation battle")
        
        # Check for European places
        if home_position <= 6 or away_position <= 6:
            analysis['european_places'] = True
            analysis['key_points'].append("European qualification at stake")
        
        # Check for title race
        if home_position <= 3 or away_position <= 3:
            analysis['title_race'] = True
            analysis['key_points'].append("Title race implications")
        
        # Cup competition motivation
        if any(cup in league.lower() for cup in ['cup', 'champions', 'europa']):
            analysis['key_points'].append("Cup competition - high motivation")
        
        return analysis
    
    def analyze_weather_impact(self, fixture: Dict) -> Dict:
        """Analyze weather impact on match"""
        weather = fixture.get('weather', {})
        
        analysis = {
            'temperature': weather.get('temperature', 20),
            'conditions': weather.get('conditions', 'Clear'),
            'precipitation': weather.get('precipitation', 0),
            'wind_speed': weather.get('wind_speed', 0),
            'humidity': weather.get('humidity', 50),
            'impact': 'neutral',
            'key_points': []
        }
        
        # Assess impact
        precipitation = analysis['precipitation']
        wind_speed = analysis['wind_speed']
        
        if precipitation > 10:  # Heavy rain
            analysis['impact'] = 'high'
            analysis['key_points'].append("Heavy rain expected - could affect play")
        elif wind_speed > 30:  # Strong wind
            analysis['impact'] = 'high'
            analysis['key_points'].append("Strong winds - could affect passing and shots")
        elif analysis['temperature'] > 30:
            analysis['impact'] = 'moderate'
            analysis['key_points'].append("Hot conditions - could lead to fatigue")
        
        return analysis
    
    def analyze_fatigue(self, fixture: Dict) -> Dict:
        """Analyze team fatigue from recent matches"""
        home_fixtures = fixture.get('home_recent_fixtures', [])
        away_fixtures = fixture.get('away_recent_fixtures', [])
        
        analysis = {
            'home_days_rest': self.calculate_rest_days(home_fixtures),
            'away_days_rest': self.calculate_rest_days(away_fixtures),
            'home_travel': self.calculate_travel(home_fixtures),
            'away_travel': self.calculate_travel(away_fixtures),
            'fatigue_factor': {'home': 0, 'away': 0},
            'key_points': []
        }
        
        # Calculate fatigue factor
        home_rest = analysis['home_days_rest']
        away_rest = analysis['away_days_rest']
        
        if home_rest < 3:
            analysis['fatigue_factor']['home'] = 1 - (home_rest / 3)
            analysis['key_points'].append(f"Home team had only {home_rest} days rest")
        
        if away_rest < 3:
            analysis['fatigue_factor']['away'] = 1 - (away_rest / 3)
            analysis['key_points'].append(f"Away team had only {away_rest} days rest")
        
        # Consider travel
        if analysis['away_travel'] > 1000:  # Kilometers
            analysis['fatigue_factor']['away'] = min(1.0, analysis['fatigue_factor']['away'] + 0.3)
            analysis['key_points'].append("Away team with significant recent travel")
        
        return analysis
    
    def calculate_rest_days(self, fixtures: List[Dict]) -> int:
        """Calculate days since last match"""
        if not fixtures:
            return 7  # Default if no data
        
        last_match = max(fixtures, key=lambda x: x.get('date', ''))
        last_date = datetime.fromisoformat(last_match['date'].replace('Z', '+00:00'))
        
        return (datetime.now() - last_date).days
    
    def calculate_travel(self, fixtures: List[Dict]) -> float:
        """Calculate total travel distance for recent matches"""
        # Simplified calculation
        return 0  # Would need location data for accurate calculation
    
    def analyze_coaching(self, fixture: Dict) -> Dict:
        """Analyze coaching factors"""
        home_coach = fixture.get('home_coach', {})
        away_coach = fixture.get('away_coach', {})
        
        analysis = {
            'home_coach_record': home_coach.get('record', {}),
            'away_coach_record': away_coach.get('record', {}),
            'head_to_head_record': self.get_coach_h2h(home_coach, away_coach),
            'tactical_style': {
                'home': home_coach.get('style', 'unknown'),
                'away': away_coach.get('style', 'unknown')
            },
            'key_points': []
        }
        
        # Check for new coach effect
        home_tenure = home_coach.get('tenure_days', 365)
        away_tenure = away_coach.get('tenure_days', 365)
        
        if home_tenure < 30:
            analysis['key_points'].append("Home team has new coach - unpredictable")
        if away_tenure < 30:
            analysis['key_points'].append("Away team has new coach - unpredictable")
        
        return analysis
    
    def get_coach_h2h(self, coach1: Dict, coach2: Dict) -> Dict:
        """Get head-to-head record between coaches"""
        # This would come from historical data
        return {'matches': 0, 'wins1': 0, 'wins2': 0, 'draws': 0}
    
    def calculate_composite_score(self, *analyses) -> Dict:
        """Calculate composite score from all analyses"""
        scores = {
            'home_advantage': 0.5,
            'form': 0.5,
            'h2h': 0.5,
            'injuries': 0.5,
            'motivation': 0.5,
            'weather': 0.5,
            'fatigue': 0.5,
            'total': 0.5
        }
        
        # Extract scores from analyses
        for analysis in analyses:
            if 'form' in analysis:
                scores['form'] = analysis.get('form_rating', 0.5)
            elif 'h2h' in analysis:
                scores['h2h'] = analysis.get('home_win_pct', 0.5)
            elif 'home_advantage' in analysis:
                scores['home_advantage'] = analysis.get('home_advantage_ratio', 1.0) / 2
        
        # Weighted total
        total = sum(scores[key] * self.weights.get(key, 0.1) 
                   for key in scores if key != 'total')
        scores['total'] = total
        
        return scores
    
    def generate_summary(self, fixture: Dict, composite_score: Dict, key_factors: List[str]) -> str:
        """Generate human-readable match summary"""
        home_team = fixture.get('home_team', 'Home')
        away_team = fixture.get('away_team', 'Away')
        league = fixture.get('league', '')
        
        summary_parts = [f"{home_team} vs {away_team} - {league}"]
        
        # Add score context
        total_score = composite_score.get('total', 0.5)
        if total_score > 0.6:
            summary_parts.append(f"Analysis favors {home_team}")
        elif total_score < 0.4:
            summary_parts.append(f"Analysis favors {away_team}")
        else:
            summary_parts.append("Evenly matched contest expected")
        
        # Add key factors
        if key_factors:
            summary_parts.append("Key factors: " + "; ".join(key_factors[:3]))
        
        return " | ".join(summary_parts)
    
    def identify_value_bets(self, odds: Dict, prediction: Dict) -> List[Dict]:
        """Identify value bets based on predictions vs odds"""
        value_bets = []
        
        if not odds:
            return value_bets
        
        # Calculate implied probabilities from odds
        home_odds = odds.get('home_win', 2.0)
        draw_odds = odds.get('draw', 3.0)
        away_odds = odds.get('away_win', 3.0)
        
        # Adjust for bookmaker margin
        home_implied = 1 / home_odds
        draw_implied = 1 / draw_odds
        away_implied = 1 / away_odds
        
        total_implied = home_implied + draw_implied + away_implied
        home_implied_adj = home_implied / total_implied
        draw_implied_adj = draw_implied / total_implied
        away_implied_adj = away_implied / total_implied
        
        # Compare with model probabilities
        model_probs = prediction.get('probabilities', {})
        home_model = model_probs.get('home', 0.33)
        draw_model = model_probs.get('draw', 0.33)
        away_model = model_probs.get('away', 0.33)
        
        # Calculate value (Kelly Criterion simplified)
        home_value = (home_model * home_odds - 1) / (home_odds - 1) if home_odds > 1 else 0
        draw_value = (draw_model * draw_odds - 1) / (draw_odds - 1) if draw_odds > 1 else 0
        away_value = (away_model * away_odds - 1) / (away_odds - 1) if away_odds > 1 else 0
        
        # Identify significant value bets (>5% edge)
        if home_value > 0.05:
            value_bets.append({
                'bet': f'{fixture["home_team"]} to win',
                'odds': home_odds,
                'model_probability': home_model,
                'value': home_value,
                'edge': home_model - home_implied_adj
            })
        
        if draw_value > 0.05:
            value_bets.append({
                'bet': 'Draw',
                'odds': draw_odds,
                'model_probability': draw_model,
                'value': draw_value,
                'edge': draw_model - draw_implied_adj
            })
        
        if away_value > 0.05:
            value_bets.append({
                'bet': f'{fixture["away_team"]} to win',
                'odds': away_odds,
                'model_probability': away_model,
                'value': away_value,
                'edge': away_model - away_implied_adj
            })
        
        # Sort by value
        value_bets.sort(key=lambda x: x['value'], reverse=True)
        
        return value_bets
