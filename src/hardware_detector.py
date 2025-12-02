"""
Hardware detection and system profiling
"""

import platform
import psutil
import os
import sys
import subprocess
import json
from typing import Dict, Any, Optional
import logging
import GPUtil
import cpuinfo
import socket

class HardwareDetector:
    """Detect hardware specifications and capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = {}
        
    def detect(self) -> Dict[str, Any]:
        """Comprehensive hardware detection"""
        try:
            self.system_info = {
                'os': self.detect_os(),
                'cpu': self.detect_cpu(),
                'memory': self.detect_memory(),
                'disk': self.detect_disk(),
                'gpu': self.detect_gpu(),
                'network': self.detect_network(),
                'virtualization': self.detect_virtualization(),
                'performance_metrics': self.benchmark_performance(),
                'hardware_profile': self.determine_profile()
            }
            return self.system_info
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            return self.get_fallback_info()
    
    def detect_os(self) -> Dict[str, Any]:
        """Detect operating system details"""
        system = platform.system()
        release = platform.release()
        version = platform.version()
        architecture = platform.architecture()[0]
        machine = platform.machine()
        
        # Detect Windows edition
        windows_edition = None
        if system == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                windows_edition = winreg.QueryValueEx(key, "ProductName")[0]
                winreg.CloseKey(key)
            except:
                windows_edition = "Windows (unknown edition)"
        
        return {
            'system': system,
            'release': release,
            'version': version,
            'architecture': architecture,
            'machine': machine,
            'windows_edition': windows_edition,
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
    
    def detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU specifications"""
        try:
            # Get detailed CPU info
            info = cpuinfo.get_cpu_info()
            
            cpu_count = psutil.cpu_count(logical=False)
            logical_cpus = psutil.cpu_count(logical=True)
            
            # Get CPU frequency
            freq = psutil.cpu_freq()
            current_freq = freq.current if freq else None
            max_freq = freq.max if freq else None
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Check for specific CPU features
            features = {
                'avx': 'avx' in info.get('flags', []),
                'avx2': 'avx2' in info.get('flags', []),
                'avx512': any(f in info.get('flags', []) for f in ['avx512f', 'avx512']),
                'sse4_2': 'sse4_2' in info.get('flags', []),
                'fma': 'fma' in info.get('flags', []),
                'neon': 'neon' in info.get('flags', []),  # ARM
            }
            
            # Determine CPU generation
            generation = self.determine_cpu_generation(info.get('brand_raw', ''))
            
            return {
                'brand': info.get('brand_raw', 'Unknown'),
                'model': info.get('model', 'Unknown'),
                'arch': info.get('arch', 'Unknown'),
                'bits': info.get('bits', 64),
                'cores_physical': cpu_count,
                'cores_logical': logical_cpus,
                'frequency_current_mhz': current_freq,
                'frequency_max_mhz': max_freq,
                'usage_percent': cpu_percent,
                'generation': generation,
                'features': features,
                'vendor': info.get('vendor_id_raw', 'Unknown')
            }
            
        except Exception as e:
            self.logger.warning(f"CPU detection failed: {e}")
            return self.get_cpu_fallback()
    
    def determine_cpu_generation(self, cpu_name: str) -> str:
        """Determine CPU generation from name"""
        cpu_lower = cpu_name.lower()
        
        if 'i3' in cpu_lower:
            return 'i3'
        elif 'i5' in cpu_lower:
            return 'i5'
        elif 'i7' in cpu_lower:
            return 'i7'
        elif 'i9' in cpu_lower:
            return 'i9'
        elif 'ryzen 3' in cpu_lower:
            return 'ryzen3'
        elif 'ryzen 5' in cpu_lower:
            return 'ryzen5'
        elif 'ryzen 7' in cpu_lower:
            return 'ryzen7'
        elif 'ryzen 9' in cpu_lower:
            return 'ryzen9'
        else:
            # Try to extract from common patterns
            import re
            patterns = [
                r'(i[3579])[- ]',
                r'(ryzen[ -][3579])',
                r'(xeon[ -])',
                r'(pentium)',
                r'(celeron)',
                r'(core[ -]2[ -]duo)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, cpu_lower)
                if match:
                    return match.group(1).replace(' ', '_')
            
            return 'unknown'
    
    def detect_memory(self) -> Dict[str, Any]:
        """Detect memory specifications"""
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            # Check memory speed if available (Windows only)
            memory_speed = None
            if platform.system() == "Windows":
                try:
                    import wmi
                    c = wmi.WMI()
                    for mem in c.Win32_PhysicalMemory():
                        memory_speed = mem.Speed
                        break
                except:
                    pass
            
            return {
                'total_gb': round(virtual_memory.total / (1024**3), 2),
                'available_gb': round(virtual_memory.available / (1024**3), 2),
                'used_gb': round(virtual_memory.used / (1024**3), 2),
                'used_percent': virtual_memory.percent,
                'swap_total_gb': round(swap_memory.total / (1024**3), 2),
                'swap_used_gb': round(swap_memory.used / (1024**3), 2),
                'memory_speed_mhz': memory_speed,
                'memory_type': self.detect_memory_type()
            }
        except Exception as e:
            self.logger.warning(f"Memory detection failed: {e}")
            return self.get_memory_fallback()
    
    def detect_memory_type(self) -> str:
        """Try to detect memory type (DDR3, DDR4, DDR5)"""
        if platform.system() == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for mem in c.Win32_PhysicalMemory():
                    memory_type = mem.MemoryType
                    # Map WMI memory type to human readable
                    types = {
                        20: 'DDR',
                        21: 'DDR2',
                        24: 'DDR3',
                        26: 'DDR4',
                        34: 'DDR5'
                    }
                    return types.get(memory_type, f"Unknown ({memory_type})")
            except:
                pass
        return "unknown"
    
    def detect_disk(self) -> Dict[str, Any]:
        """Detect disk specifications"""
        try:
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'used_percent': usage.percent
                    })
                except:
                    continue
            
            # Detect disk type (SSD/HDD)
            disk_types = {}
            if platform.system() == "Windows":
                disk_types = self.detect_disk_type_windows()
            elif platform.system() == "Linux":
                disk_types = self.detect_disk_type_linux()
            elif platform.system() == "Darwin":
                disk_types = self.detect_disk_type_mac()
            
            # Add disk type to info
            for disk in disk_info:
                disk['type'] = disk_types.get(disk['device'], 'unknown')
            
            return {
                'partitions': disk_info,
                'main_disk': disk_info[0] if disk_info else None,
                'has_ssd': any(d['type'] == 'ssd' for d in disk_info)
            }
        except Exception as e:
            self.logger.warning(f"Disk detection failed: {e}")
            return {'partitions': [], 'main_disk': None, 'has_ssd': False}
    
    def detect_disk_type_windows(self) -> Dict[str, str]:
        """Detect disk type on Windows"""
        disk_types = {}
        try:
            import wmi
            c = wmi.WMI()
            for disk in c.Win32_DiskDrive():
                device_id = disk.DeviceID.replace('\\', '\\\\')
                if 'SSD' in disk.Model.upper() or 'SOLID' in disk.Model.upper():
                    disk_types[device_id] = 'ssd'
                else:
                    disk_types[device_id] = 'hdd'
        except:
            pass
        return disk_types
    
    def detect_disk_type_linux(self) -> Dict[str, str]:
        """Detect disk type on Linux"""
        disk_types = {}
        try:
            import subprocess
            # Check for rotational flag
            for device in ['sda', 'sdb', 'nvme0n1']:
                try:
                    result = subprocess.run(
                        ['cat', f'/sys/block/{device}/queue/rotational'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        rotational = result.stdout.strip()
                        disk_types[f'/dev/{device}'] = 'hdd' if rotational == '1' else 'ssd'
                except:
                    continue
        except:
            pass
        return disk_types
    
    def detect_disk_type_mac(self) -> Dict[str, str]:
        """Detect disk type on macOS"""
        disk_types = {}
        try:
            import subprocess
            result = subprocess.run(
                ['system_profiler', 'SPSerialATADataType'],
                capture_output=True, text=True
            )
            if 'SSD' in result.stdout:
                disk_types['main'] = 'ssd'
            else:
                disk_types['main'] = 'hdd'
        except:
            pass
        return disk_types
    
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU specifications"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_free_mb': gpu.memoryFree,
                        'memory_used_mb': gpu.memoryUsed,
                        'temperature_c': gpu.temperature,
                        'load_percent': gpu.load * 100,
                        'driver': gpu.driver,
                        'uuid': gpu.uuid
                    })
                
                # Check for CUDA support
                cuda_available = False
                cuda_version = None
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        cuda_version = torch.version.cuda
                except:
                    pass
                
                return {
                    'available': True,
                    'count': len(gpus),
                    'gpus': gpu_info,
                    'cuda_available': cuda_available,
                    'cuda_version': cuda_version,
                    'primary_gpu': gpu_info[0] if gpu_info else None
                }
            else:
                return {'available': False, 'count': 0}
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            return {'available': False, 'count': 0}
    
    def detect_network(self) -> Dict[str, Any]:
        """Detect network capabilities"""
        try:
            # Get network interfaces
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            network_info = {}
            for interface, addrs in interfaces.items():
                interface_stats = stats.get(interface, {})
                network_info[interface] = {
                    'addresses': [str(addr.address) for addr in addrs],
                    'is_up': interface_stats.get('isup', False),
                    'speed_mbps': interface_stats.get('speed', 0),
                    'mtu': interface_stats.get('mtu', 0)
                }
            
            # Check internet connectivity
            internet_available = self.check_internet_connection()
            
            return {
                'interfaces': network_info,
                'internet_available': internet_available,
                'hostname': socket.gethostname(),
                'ip_address': socket.gethostbyname(socket.gethostname())
            }
        except Exception as e:
            self.logger.warning(f"Network detection failed: {e}")
            return {'interfaces': {}, 'internet_available': False}
    
    def check_internet_connection(self) -> bool:
        """Check if internet connection is available"""
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=3)
            return True
        except:
            return False
    
    def detect_virtualization(self) -> Dict[str, Any]:
        """Detect if running in virtual environment"""
        try:
            is_virtual = False
            virtualization_type = None
            
            if platform.system() == "Windows":
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                       r"SYSTEM\CurrentControlSet\Services\Disk\Enum")
                    device_desc = winreg.QueryValueEx(key, "0")[0]
                    winreg.CloseKey(key)
                    
                    vm_indicators = ['vmware', 'virtual', 'vbox', 'qemu', 'xen', 'hyper-v']
                    if any(indicator in device_desc.lower() for indicator in vm_indicators):
                        is_virtual = True
                        virtualization_type = 'vm'
                except:
                    pass
            elif platform.system() == "Linux":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo_content = f.read()
                        if 'hypervisor' in cpuinfo_content:
                            is_virtual = True
                            virtualization_type = 'vm'
                except:
                    pass
            
            # Check for containers
            if not is_virtual:
                container_indicators = [
                    '/.dockerenv',  # Docker
                    '/run/.containerenv',  # Podman
                    'container=crio',  # CRI-O
                    'container=lxc'  # LXC
                ]
                
                for indicator in container_indicators:
                    if os.path.exists(indicator.split('=')[0] if '=' in indicator else indicator):
                        is_virtual = True
                        virtualization_type = 'container'
                        break
            
            return {
                'is_virtual': is_virtual,
                'type': virtualization_type,
                'in_docker': os.path.exists('/.dockerenv')
            }
        except Exception as e:
            self.logger.warning(f"Virtualization detection failed: {e}")
            return {'is_virtual': False, 'type': None, 'in_docker': False}
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Run simple performance benchmarks"""
        benchmarks = {}
        
        try:
            # CPU benchmark (simple calculation)
            import time
            start = time.time()
            result = 0
            for i in range(10**7):
                result += i * 0.5
            cpu_time = time.time() - start
            benchmarks['cpu_speed_score'] = 1.0 / cpu_time if cpu_time > 0 else 0
            
            # Memory benchmark
            start = time.time()
            large_list = [i for i in range(10**6)]
            memory_time = time.time() - start
            benchmarks['memory_speed_score'] = 1.0 / memory_time if memory_time > 0 else 0
            
            # Disk benchmark (write speed)
            import tempfile
            start = time.time()
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                for i in range(10000):
                    f.write('x' * 100)
            disk_time = time.time() - start
            benchmarks['disk_speed_score'] = 1.0 / disk_time if disk_time > 0 else 0
            
        except Exception as e:
            self.logger.warning(f"Benchmark failed: {e}")
            benchmarks = {
                'cpu_speed_score': 0.5,
                'memory_speed_score': 0.5,
                'disk_speed_score': 0.5
            }
        
        return benchmarks
    
    def determine_profile(self) -> str:
        """Determine hardware profile based on detected specs"""
        if not self.system_info:
            self.detect()
        
        cpu_info = self.system_info.get('cpu', {})
        memory_info = self.system_info.get('memory', {})
        disk_info = self.system_info.get('disk', {})
        
        # Extract key metrics
        cpu_gen = cpu_info.get('generation', '')
        cpu_cores = cpu_info.get('cores_physical', 2)
        memory_gb = memory_info.get('total_gb', 4)
        has_ssd = disk_info.get('has_ssd', False)
        
        # Determine profile
        if 'i3' in cpu_gen or memory_gb <= 8 or cpu_cores <= 4:
            if has_ssd:
                return 'low_end'  # i3/8GB/SSD
            else:
                return 'minimal'
        elif 'i5' in cpu_gen or (8 < memory_gb <= 16) or (4 < cpu_cores <= 6):
            return 'mid_end'  # i5/8-16GB
        elif 'i7' in cpu_gen or 'i9' in cpu_gen or memory_gb > 16 or cpu_cores > 6:
            return 'high_end'  # i7+/16GB+
        else:
            return 'custom'
    
    def get_fallback_info(self) -> Dict[str, Any]:
        """Get fallback hardware info when detection fails"""
        return {
            'os': {
                'system': platform.system(),
                'release': platform.release(),
                'python_version': platform.python_version()
            },
            'cpu': self.get_cpu_fallback(),
            'memory': self.get_memory_fallback(),
            'disk': {'partitions': [], 'main_disk': None, 'has_ssd': False},
            'gpu': {'available': False, 'count': 0},
            'network': {'internet_available': False},
            'virtualization': {'is_virtual': False},
            'performance_metrics': {'cpu_speed_score': 0.5},
            'hardware_profile': 'custom'
        }
    
    def get_cpu_fallback(self) -> Dict[str, Any]:
        """Get fallback CPU info"""
        return {
            'brand': 'Unknown',
            'cores_physical': psutil.cpu_count(logical=False) or 2,
            'cores_logical': psutil.cpu_count(logical=True) or 4,
            'generation': 'unknown',
            'features': {}
        }
    
    def get_memory_fallback(self) -> Dict[str, Any]:
        """Get fallback memory info"""
        try:
            vm = psutil.virtual_memory()
            return {
                'total_gb': round(vm.total / (1024**3), 1),
                'used_percent': vm.percent
            }
        except:
            return {'total_gb': 8.0, 'used_percent': 50.0}
    
    def generate_report(self) -> str:
        """Generate human-readable hardware report"""
        if not self.system_info:
            self.detect()
        
        report = []
        report.append("=" * 60)
        report.append("HARDWARE DETECTION REPORT")
        report.append("=" * 60)
        
        # OS Info
        os_info = self.system_info.get('os', {})
        report.append(f"\nOPERATING SYSTEM:")
        report.append(f"  System: {os_info.get('system', 'Unknown')}")
        report.append(f"  Release: {os_info.get('release', 'Unknown')}")
        if os_info.get('windows_edition'):
            report.append(f"  Edition: {os_info['windows_edition']}")
        report.append(f"  Architecture: {os_info.get('architecture', 'Unknown')}")
        report.append(f"  Python: {os_info.get('python_version', 'Unknown')}")
        
        # CPU Info
        cpu_info = self.system_info.get('cpu', {})
        report.append(f"\nCPU:")
        report.append(f"  Model: {cpu_info.get('brand', 'Unknown')}")
        report.append(f"  Physical Cores: {cpu_info.get('cores_physical', 'N/A')}")
        report.append(f"  Logical Cores: {cpu_info.get('cores_logical', 'N/A')}")
        report.append(f"  Generation: {cpu_info.get('generation', 'Unknown').upper()}")
        if cpu_info.get('frequency_current_mhz'):
            report.append(f"  Frequency: {cpu_info['frequency_current_mhz']} MHz")
        
        # Memory Info
        mem_info = self.system_info.get('memory', {})
        report.append(f"\nMEMORY:")
        report.append(f"  Total RAM: {mem_info.get('total_gb', 'N/A')} GB")
        report.append(f"  Available: {mem_info.get('available_gb', 'N/A')} GB")
        report.append(f"  Used: {mem_info.get('used_percent', 'N/A')}%")
        if mem_info.get('memory_type') != 'unknown':
            report.append(f"  Type: {mem_info['memory_type']}")
        
        # Disk Info
        disk_info = self.system_info.get('disk', {})
        report.append(f"\nDISK:")
        main_disk = disk_info.get('main_disk')
        if main_disk:
            report.append(f"  Main Disk: {main_disk.get('device', 'N/A')}")
            report.append(f"  Total Space: {main_disk.get('total_gb', 'N/A')} GB")
            report.append(f"  Free Space: {main_disk.get('free_gb', 'N/A')} GB")
            report.append(f"  Type: {main_disk.get('type', 'unknown').upper()}")
        
        # GPU Info
        gpu_info = self.system_info.get('gpu', {})
        report.append(f"\nGPU:")
        if gpu_info.get('available'):
            primary = gpu_info.get('primary_gpu', {})
            report.append(f"  Model: {primary.get('name', 'Unknown')}")
            report.append(f"  Memory: {primary.get('memory_total_mb', 'N/A')} MB")
            if gpu_info.get('cuda_available'):
                report.append(f"  CUDA: Available (v{gpu_info.get('cuda_version', 'N/A')})")
        else:
            report.append("  No dedicated GPU detected")
        
        # Network Info
        net_info = self.system_info.get('network', {})
        report.append(f"\nNETWORK:")
        report.append(f"  Internet: {'Available' if net_info.get('internet_available') else 'Unavailable'}")
        report.append(f"  Hostname: {net_info.get('hostname', 'N/A')}")
        
        # Virtualization
        virt_info = self.system_info.get('virtualization', {})
        if virt_info.get('is_virtual'):
            report.append(f"\nVIRTUALIZATION:")
            report.append(f"  Environment: {virt_info.get('type', 'virtual').upper()}")
            if virt_info.get('in_docker'):
                report.append("  Running in Docker container")
        
        # Hardware Profile
        profile = self.system_info.get('hardware_profile', 'custom')
        report.append(f"\nHARDWARE PROFILE:")
        report.append(f"  Detected: {profile.upper()}")
        
        # Performance Scores
        perf = self.system_info.get('performance_metrics', {})
        if any(perf.values()):
            report.append(f"\nPERFORMANCE SCORES:")
            for key, value in perf.items():
                if value:
                    score_name = key.replace('_score', '').replace('_', ' ').title()
                    report.append(f"  {score_name}: {value:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str = "./logs/hardware_report.txt"):
        """Save hardware report to file"""
        report = self.generate_report()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)
        self.logger.info(f"Hardware report saved to {filepath}")
