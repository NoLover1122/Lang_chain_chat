import psutil
import time
from datetime import datetime
from typing import Dict, Any


class SystemMonitor:
    """
    系统资源监控工具，用于监控CPU、内存和磁盘使用情况
    """
    
    _last_metrics = None
    _last_update_time = None
    _update_interval = 5  # 更新间隔(秒)
    
    @classmethod
    def get_system_metrics(cls) -> Dict[str, Any]:
        """
        获取系统指标，包括CPU、内存和磁盘使用情况
        如果距离上次更新不足指定间隔，则返回缓存的指标
        """
        current_time = time.time()
        
        # 如果是首次调用或者已超过更新间隔，则更新指标
        if (cls._last_metrics is None or 
            cls._last_update_time is None or 
            current_time - cls._last_update_time > cls._update_interval):
            
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 获取内存使用情况
            memory = psutil.virtual_memory()
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 更新指标
            cls._last_metrics = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "cpu": {
                        "percent": cpu_percent,
                        "cores": psutil.cpu_count(logical=True),
                        "physical_cores": psutil.cpu_count(logical=False)
                    },
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "used": memory.used,
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024 ** 3),
                        "used_gb": disk.used / (1024 ** 3),
                        "free_gb": disk.free / (1024 ** 3),
                        "percent": disk.percent
                    }
                }
            }
            
            cls._last_update_time = current_time
        
        return cls._last_metrics
    
    @classmethod
    def get_api_format_metrics(cls) -> Dict[str, Any]:
        """
        获取API格式的系统指标 - 扁平化格式，直接用于API响应
        
        Returns:
            Dict[str, Any]: API格式的系统指标
        """
        # 获取原始指标
        raw_metrics = cls.get_system_metrics()
        metrics_data = raw_metrics.get("metrics", {})
        
        return {
            "status": "success",
            "system_metrics": {
                "cpu_percent": metrics_data.get("cpu", {}).get("percent", 0),
                "memory_percent": metrics_data.get("memory", {}).get("percent", 0),
                "disk_total_gb": metrics_data.get("disk", {}).get("total_gb", 0),
                "disk_used_gb": metrics_data.get("disk", {}).get("used_gb", 0),
                "disk_free_gb": metrics_data.get("disk", {}).get("free_gb", 0),
                "disk_percent": metrics_data.get("disk", {}).get("percent", 0)
            },
            "timestamp": raw_metrics.get("timestamp")
        }
    