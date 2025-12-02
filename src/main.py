#!/usr/bin/env python3
"""
Main execution file for Soccer Prediction AI System
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from src.data_collection.fixtures_collector import FixturesCollector
from src.data_processing.feature_engineering import FeatureEngineer
from src.ai_models.model_predictor import ModelPredictor
from src.prediction.match_analyzer import MatchAnalyzer
from src.utils.logger import setup_logger
from src.utils.database import DatabaseManager

class SoccerPredictionSystem:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the soccer prediction system"""
        self.config = self.load_config(config_path)
        self.logger = setup_logger(__name__)
        self.db = DatabaseManager(self.config['database'])
        
        # Initialize components
        self.collector = FixturesCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.predictor = ModelPredictor(self.config)
        self.analyzer = MatchAnalyzer(self.config)
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def collect_todays_data(self):
        """Collect today's fixtures and odds from all sources"""
        self.logger.info("Starting data collection for today's matches")
        
        # Collect fixtures from APIs
        api_fixtures = self.collector.collect_from_apis()
        
        # Scrape additional data
        scraped_data = self.collector.scrape_betting_sites()
        
        # Combine and validate data
        all_fixtures = self.collector.merge_fixtures(api_fixtures, scraped_data)
        
        # Store in database
        self.db.store_fixtures(all_fixtures)
        
        self.logger.info(f"Collected {len(all_fixtures)} fixtures")
        return all_fixtures
    
    def process_and_predict(self, fixtures):
        """Process fixtures and generate predictions"""
        predictions = []
        
        for fixture in fixtures:
            try:
                # Generate features
                features = self.feature_engineer.create_features(fixture)
                
                # Get prediction from ensemble model
                prediction = self.predictor.predict(features)
                
                # Deep analysis
                analysis = self.analyzer.analyze_match(fixture, prediction)
                
                # Format prediction
                formatted = self.format_prediction(fixture, prediction, analysis)
                predictions.append(formatted)
                
            except Exception as e:
                self.logger.error(f"Error processing fixture {fixture.get('id', 'unknown')}: {str(e)}")
                continue
        
        return predictions
    
    def format_prediction(self, fixture, prediction, analysis):
        """Format prediction for output"""
        return {
            'match_id': fixture.get('id'),
            'date': fixture.get('date'),
            'league': fixture.get('league'),
            'home_team': fixture.get('home_team'),
            'away_team': fixture.get('away_team'),
            'prediction': {
                'winner': prediction.get('winner'),
                'probability': prediction.get('probability'),
                'confidence': prediction.get('confidence'),
                'both_teams_to_score': prediction.get('btts'),
                'btts_probability': prediction.get('btts_probability'),
                'predicted_score': prediction.get('predicted_score')
            },
            'analysis': analysis,
            'odds': fixture.get('odds', {}),
            'key_factors': analysis.get('key_factors', []),
            'timestamp': datetime.now().isoformat()
        }
    
    def train_models(self, retrain=False):
        """Train or retrain models"""
        self.logger.info("Starting model training")
        
        # Load historical data
        historical_data = self.db.get_training_data()
        
        # Train models
        self.predictor.train_models(historical_data, retrain=retrain)
        
        self.logger.info("Model training completed")
    
    def run_daily_pipeline(self):
        """Execute complete daily prediction pipeline"""
        try:
            # Step 1: Collect today's data
            fixtures = self.collect_todays_data()
            
            # Step 2: Process and predict
            predictions = self.process_and_predict(fixtures)
            
            # Step 3: Filter high-confidence predictions
            high_confidence = [
                p for p in predictions 
                if p['prediction']['confidence'] >= self.config['prediction']['min_confidence']
            ]
            
            # Step 4: Save results
            self.save_predictions(high_confidence)
            
            # Step 5: Generate report
            self.generate_report(high_confidence)
            
            return high_confidence
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_predictions(self, predictions):
        """Save predictions to database and file"""
        # Save to database
        self.db.store_predictions(predictions)
        
        # Save to JSON file
        output_dir = Path("output/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(predictions)} predictions to {filename}")
    
    def generate_report(self, predictions):
        """Generate daily prediction report"""
        report = {
            'date': datetime.now().date().isoformat(),
            'total_matches_analyzed': len(predictions),
            'high_confidence_predictions': len(predictions),
            'predictions_by_league': self.group_by_league(predictions),
            'average_confidence': sum(p['prediction']['confidence'] for p in predictions) / len(predictions) if predictions else 0,
            'summary': self.create_summary(predictions)
        }
        
        # Save report
        report_dir = Path("output/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Soccer Prediction AI System")
    parser.add_argument('--mode', choices=['predict', 'train', 'full'], default='full',
                       help='Run mode: predict only, train only, or full pipeline')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retraining of models')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SoccerPredictionSystem(args.config)
    
    if args.mode == 'train' or args.retrain:
        system.train_models(retrain=True)
    
    if args.mode == 'predict' or args.mode == 'full':
        predictions = system.run_daily_pipeline()
        
        # Print top predictions
        print("\n" + "="*80)
        print("TOP SOCCER PREDICTIONS FOR TODAY")
        print("="*80)
        
        for i, pred in enumerate(sorted(predictions, 
                                       key=lambda x: x['prediction']['confidence'], 
                                       reverse=True)[:10], 1):
            print(f"\n{i}. {pred['home_team']} vs {pred['away_team']}")
            print(f"   League: {pred['league']}")
            print(f"   Prediction: {pred['prediction']['winner']} "
                  f"(Confidence: {pred['prediction']['confidence']:.2%})")
            print(f"   BTTS: {'Yes' if pred['prediction']['both_teams_to_score'] else 'No'} "
                  f"(Probability: {pred['prediction']['btts_probability']:.2%})")
            print(f"   Predicted Score: {pred['prediction']['predicted_score']}")
            
            if pred.get('key_factors'):
                print(f"   Key Factors: {', '.join(pred['key_factors'][:3])}")

if __name__ == "__main__":
    main()
