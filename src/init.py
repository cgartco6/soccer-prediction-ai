"""
Soccer Prediction AI System
"""

__version__ = "1.0.0"
__author__ = "Soccer Prediction AI Team"

from src.utils.hardware_detector import HardwareDetector
from src.system.optimizer import SystemOptimizer

# Initialize hardware detection on import
hardware_info = HardwareDetector().detect()
system_optimizer = SystemOptimizer(hardware_info)

# Export main components
__all__ = [
    'hardware_info',
    'system_optimizer',
    'FootballPredictionSystem'
]
