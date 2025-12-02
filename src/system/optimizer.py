"""
System optimization based on hardware detection
"""

import psutil
import os
import sys
import gc
from typing import Dict, Any, Optional, List
import logging
import json
import yaml
from pathlib import Path
import warnings
import threading
import time
from dataclasses import dataclass
from enum import Enum

class OptimizationLevel(Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    MAXIMUM = "maximum"
    AUTO = "auto"

@dataclass
class OptimizationConfig:
    """Configuration for system optimization"""
    max_parallel_processes: int
    batch_size: int
    cache_size_mb: int
    use_compression: bool
    model_complexity: str
    feature_set: str
    memory_safety_margin: float
    gpu_acceleration: bool
    disk_cache_enabled: bool
    logging_level: str

class SystemOptimizer:
    """Optimize system resources based on hardware"""
    
    def __init__(self, hardware_info: Dict[str, Any], config_path: str = "config/config.yaml"):
        self.hardware_info = hardware_info
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.optimization_config = None
        self.monitor_thread = None
        self.monitoring = False
        
        # Initialize optimization
        self.apply_optimization()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def apply_optimization(self):
        """Apply system optimizations based on hardware"""
        profile = self.hardware_info.get('hardware_profile', 'custom')
        
        # Get profile-specific config
        profile_config = self.config.get('profiles', {}).get(profile, {})
        
        # Create optimization config
        self.optimization_config = OptimizationConfig(
            max_parallel_processes=self.calculate_max_processes(profile_config),
            batch_size=self.calculate_batch_size(profile_config),
            cache_size_mb=self.calculate_cache_size(),
            use_compression=self.should_use_compression(profile),
            model_complexity=profile_config.get('model_complexity', 'balanced'),
            feature_set=profile_config.get('feature_set', 'standard'),
            memory_safety_margin=self.config.get('hardware', {}).get('memory_safety_margin', 0.15),
            gpu_acceleration=self.should_use_gpu(),
            disk_cache_enabled=True,
            logging_level=self.config.get('monitoring', {}).get('log_level', 'INFO')
        )
        
        # Apply Python-level optimizations
        self.apply_python_optimizations()
        
        # Apply TensorFlow/GPU optimizations if applicable
        if self.optimization_config.gpu_acceleration:
            self.apply_gpu_optimizations()
        
        # Start resource monitoring
        if self.config.get('monitoring', {}).get('resource_monitoring', True):
            self.start_resource_monitoring()
        
        self.logger.info(f"System optimization applied for {profile.upper()} profile")
        self.logger.info(f"Config: {self.optimization_config}")
    
    def calculate_max_processes(self, profile_config: Dict) -> int:
        """Calculate optimal number of parallel processes"""
        config_value = profile_config.get('max_parallel_processes', 0)
        
        if config_value > 0:
            return config_value
        
        # Auto-calculate based on CPU cores
        cpu_info = self.hardware_info.get('cpu', {})
        physical_cores = cpu_info.get('cores_physical', 2)
        logical_cores = cpu_info.get('cores_logical', 4)
        
        # Consider memory constraints
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        
        if memory_gb <= 4:
            return 1
        elif memory_gb <= 8:
            return min(2, physical_cores)
        elif memory_gb <= 16:
            return min(4, logical_cores // 2)
        else:
            return min(8, logical_cores)
    
    def calculate_batch_size(self, profile_config: Dict) -> int:
        """Calculate optimal batch size for model inference"""
        config_value = profile_config.get('batch_size', 0)
        
        if config_value > 0:
            return config_value
        
        # Auto-calculate based on memory
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        
        if memory_gb <= 4:
            return 8
        elif memory_gb <= 8:
            return 16
        elif memory_gb <= 16:
            return 32
        else:
            return 64
    
    def calculate_cache_size(self) -> int:
        """Calculate optimal cache size in MB"""
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        disk_cache_gb = self.config.get('hardware', {}).get('disk_cache_size_gb', 2)
        
        # Use 10% of RAM or disk cache size, whichever is smaller
        ram_cache_mb = int(memory_gb * 1024 * 0.1)
        disk_cache_mb = disk_cache_gb * 1024
        
        return min(ram_cache_mb, disk_cache_mb)
    
    def should_use_compression(self, profile: str) -> bool:
        """Determine if data compression should be used"""
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        
        # Use compression on low memory systems
        if profile == 'low_end' or memory_gb <= 8:
            return True
        
        # Check if we have SSD for faster compression/decompression
        has_ssd = self.hardware_info.get('disk', {}).get('has_ssd', False)
        
        # Use compression on HDD systems for disk I/O savings
        return not has_ssd
    
    def should_use_gpu(self) -> bool:
        """Determine if GPU acceleration should be used"""
        gpu_info = self.hardware_info.get('gpu', {})
        
        if not gpu_info.get('available', False):
            return False
        
        # Check GPU memory
        primary_gpu = gpu_info.get('primary_gpu', {})
        gpu_memory_mb = primary_gpu.get('memory_total_mb', 0)
        
        # Minimum 2GB GPU memory required
        if gpu_memory_mb < 2048:
            return False
        
        # Check CUDA availability for TensorFlow/PyTorch
        if gpu_info.get('cuda_available', False):
            return True
        
        return False
    
    def apply_python_optimizations(self):
        """Apply Python-specific optimizations"""
        import gc
        
        # Enable automatic garbage collection
        gc.enable()
        
        # Set garbage collection thresholds based on memory
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        
        if memory_gb <= 4:
            gc.set_threshold(700, 10, 10)  # Aggressive GC
        elif memory_gb <= 8:
            gc.set_threshold(1000, 15, 15)  # Moderate GC
        else:
            gc.set_threshold(2000, 25, 25)  # Standard GC
        
        # Optimize recursion limit
        sys.setrecursionlimit(10000)
        
        # Set numpy threading
        os.environ['OMP_NUM_THREADS'] = str(self.optimization_config.max_parallel_processes)
        os.environ['MKL_NUM_THREADS'] = str(self.optimization_config.max_parallel_processes)
        
        # Disable warnings for cleaner output
        if not self.config.get('monitoring', {}).get('debug_mode', False):
            warnings.filterwarnings('ignore')
    
    def apply_gpu_optimizations(self):
        """Apply GPU-specific optimizations"""
        try:
            import tensorflow as tf
            
            # Configure TensorFlow GPU settings
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set GPU memory limit
                    memory_limit = self.calculate_gpu_memory_limit()
                    if memory_limit > 0:
                        tf.config.set_logical_device_configuration(
                            gpus[0],
                            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                        )
                    
                    self.logger.info("TensorFlow GPU optimizations applied")
                    
                except RuntimeError as e:
                    self.logger.warning(f"GPU optimization failed: {e}")
            
        except ImportError:
            self.logger.debug("TensorFlow not available for GPU optimizations")
        
        try:
            import torch
            
            # PyTorch optimizations
            if torch.cuda.is_available():
                # Enable cudnn auto-tuner
                torch.backends.cudnn.benchmark = True
                
                # Set default tensor type
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
                
                self.logger.info("PyTorch GPU optimizations applied")
                
        except ImportError:
            self.logger.debug("PyTorch not available for GPU optimizations")
    
    def calculate_gpu_memory_limit(self) -> int:
        """Calculate GPU memory limit in MB"""
        gpu_info = self.hardware_info.get('gpu', {})
        primary_gpu = gpu_info.get('primary_gpu', {})
        
        total_memory_mb = primary_gpu.get('memory_total_mb', 0)
        
        if total_memory_mb <= 0:
            return 0
        
        # Leave 1GB for system
        return max(512, total_memory_mb - 1024)
    
    def start_resource_monitoring(self):
        """Start resource monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def _monitor_resources(self):
        """Monitor system resources in background"""
        check_interval = self.config.get('monitoring', {}).get('memory_check_interval', 60)
        high_threshold = self.config.get('monitoring', {}).get('high_usage_threshold', 0.85)
        alert_enabled = self.config.get('monitoring', {}).get('alert_on_high_usage', True)
        
        while self.monitoring:
            try:
                # Check memory usage
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Log periodically
                log_interval = self.config.get('monitoring', {}).get('performance_log_interval', 300)
                current_time = time.time()
                
                if hasattr(self, '_last_log_time'):
                    if current_time - self._last_log_time >= log_interval:
                        self.logger.info(
                            f"System Resources - CPU: {cpu_percent:.1f}%, "
                            f"Memory: {memory.percent:.1f}%"
                        )
                        self._last_log_time = current_time
                else:
                    self._last_log_time = current_time
                
                # Check for high usage
                if alert_enabled and memory.percent >= high_threshold * 100:
                    self.logger.warning(
                        f"High memory usage detected: {memory.percent:.1f}%"
                    )
                    self._handle_high_memory_usage()
                
                # Check CPU usage
                max_cpu_usage = self.config.get('hardware', {}).get('max_cpu_usage', 0.85)
                if cpu_percent >= max_cpu_usage * 100:
                    self.logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(check_interval)
    
    def _handle_high_memory_usage(self):
        """Handle high memory usage situations"""
        try:
            # Clear caches
            import gc
            gc.collect()
            
            # Clear TensorFlow/PyTorch cache if available
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Reduce batch size if too high
            current_batch = self.optimization_config.batch_size
            if current_batch > 16:
                new_batch = max(8, current_batch // 2)
                self.optimization_config.batch_size = new_batch
                self.logger.info(f"Reduced batch size to {new_batch} due to memory pressure")
            
        except Exception as e:
            self.logger.error(f"Failed to handle high memory usage: {e}")
    
    def optimize_data_loading(self, data_size: int) -> Dict[str, Any]:
        """Optimize data loading strategy based on data size"""
        memory_gb = self.hardware_info.get('memory', {}).get('total_gb', 8)
        has_ssd = self.hardware_info.get('disk', {}).get('has_ssd', False)
        
        # Estimate memory needed (rough estimation)
        estimated_memory_mb = data_size * 0.1  # Simplified estimation
        
        strategy = {
            'load_full': False,
            'use_chunks': False,
            'chunk_size': 0,
            'use_disk_cache': False,
            'compression': False
        }
        
        if estimated_memory_mb < memory_gb * 1024 * 0.3:  # Less than 30% of RAM
            strategy['load_full'] = True
        else:
            strategy['use_chunks'] = True
            strategy['chunk_size'] = min(10000, int((memory_gb * 1024 * 0.2) / 0.1))
        
        if not has_ssd or estimated_memory_mb > 1024:  # More than 1GB or no SSD
            strategy['use_disk_cache'] = True
            strategy['compression'] = self.optimization_config.use_compression
        
        return strategy
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get optimized model configuration based on hardware"""
        profile = self.hardware_info.get('hardware_profile', 'custom')
        model_configs = self.config.get('models', {}).get('training', {})
        profile_config = model_configs.get(profile, model_configs.get('mid_end', {}))
        
        config = {
            'n_estimators': profile_config.get('n_estimators', 100),
            'max_depth': profile_config.get('max_depth', 4),
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Adjust based on model type
        if model_type == 'xgboost':
            config['tree_method'] = 'gpu_hist' if self.optimization_config.gpu_acceleration else 'hist'
            config['predictor'] = 'gpu_predictor' if self.optimization_config.gpu_acceleration else 'cpu_predictor'
        elif model_type == 'lightgbm':
            config['device'] = 'gpu' if self.optimization_config.gpu_acceleration else 'cpu'
        
        return config
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get optimized feature engineering configuration"""
        profile = self.hardware_info.get('hardware_profile', 'custom')
        feature_configs = self.config.get('features', {})
        profile_config = feature_configs.get(profile, feature_configs.get('mid_end', {}))
        
        return {
            'max_features': profile_config.get('max_features', 50),
            'include_advanced': profile_config.get('include_advanced', True),
            'temporal_features': profile_config.get('temporal_features', 10),
            'interaction_features': profile_config.get('interaction_features', True),
            'use_pca': profile_config.get('max_features', 50) > 30
        }
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Resource monitoring stopped")
    
    def get_optimization_report(self) -> str:
        """Generate optimization report"""
        report = []
        report.append("=" * 60)
        report.append("SYSTEM OPTIMIZATION REPORT")
        report.append("=" * 60)
        
        if self.optimization_config:
            report.append(f"\nOptimization Level: {self.hardware_info.get('hardware_profile', 'custom').upper()}")
            report.append(f"Max Parallel Processes: {self.optimization_config.max_parallel_processes}")
            report.append(f"Batch Size: {self.optimization_config.batch_size}")
            report.append(f"Cache Size: {self.optimization_config.cache_size_mb} MB")
            report.append(f"Compression: {'Enabled' if self.optimization_config.use_compression else 'Disabled'}")
            report.append(f"Model Complexity: {self.optimization_config.model_complexity}")
            report.append(f"Feature Set: {self.optimization_config.feature_set}")
            report.append(f"GPU Acceleration: {'Enabled' if self.optimization_config.gpu_acceleration else 'Disabled'}")
            report.append(f"Disk Cache: {'Enabled' if self.optimization_config.disk_cache_enabled else 'Disabled'}")
        
        report.append(f"\nHardware Profile: {self.hardware_info.get('hardware_profile', 'custom').upper()}")
        report.append(f"Memory Available: {self.hardware_info.get('memory', {}).get('total_gb', 0)} GB")
        report.append(f"CPU Cores: {self.hardware_info.get('cpu', {}).get('cores_logical', 0)}")
        report.append(f"Has SSD: {'Yes' if self.hardware_info.get('disk', {}).get('has_ssd') else 'No'}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_optimization_report(self, filepath: str = "./logs/optimization_report.txt"):
        """Save optimization report to file"""
        report = self.get_optimization_report()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)
        self.logger.info(f"Optimization report saved to {filepath}")
