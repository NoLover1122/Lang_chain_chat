import asyncio
import time
import psutil
from datetime import datetime
from typing import Dict, Any
from config.settings import settings
from utils.logging import logger
from utils.system_monitor import SystemMonitor

# 动态资源限制
_dynamic_limits = {
    "max_cache_size": settings.MAX_CACHE_SIZE,
    "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
}

_last_adjustment_time = 0
_adjustment_interval = 60  # 60秒调整一次

def adjust_resource_limits() -> Dict[str, Any]:
    """
    根据系统资源状态调整限制
    
    Returns:
        调整后的限制配置
    """
    global _dynamic_limits, _last_adjustment_time
    
    current_time = time.time()
    
    # 如果距离上次调整时间不够，跳过
    if current_time - _last_adjustment_time < _adjustment_interval:
        return _dynamic_limits.copy()
    
    # 获取系统指标
    system_metrics = SystemMonitor.get_system_metrics()
    cpu_percent = system_metrics.get("metrics", {}).get("cpu", {}).get("percent", 0)
    memory_percent = system_metrics.get("metrics", {}).get("memory", {}).get("percent", 0)
    
    # 保存调整前的值
    old_limits = _dynamic_limits.copy()
    
    # 根据CPU使用率调整并发请求数
    if cpu_percent > 90:
        _dynamic_limits["max_concurrent_requests"] = max(1, _dynamic_limits["max_concurrent_requests"] - 1)
        logger.warning(f"CPU使用率过高({cpu_percent}%)，降低最大并发请求数至 {_dynamic_limits['max_concurrent_requests']}")
    elif cpu_percent < 50 and _dynamic_limits["max_concurrent_requests"] < 10:
        _dynamic_limits["max_concurrent_requests"] += 1
        logger.info(f"CPU使用率较低({cpu_percent}%)，提高最大并发请求数至 {_dynamic_limits['max_concurrent_requests']}")
    
    # 根据内存使用率调整缓存大小
    if memory_percent > 85:
        _dynamic_limits["max_cache_size"] = max(5, _dynamic_limits["max_cache_size"] - 2)
        logger.warning(f"内存使用率过高({memory_percent}%)，降低最大缓存大小至 {_dynamic_limits['max_cache_size']}")
    elif memory_percent < 60 and _dynamic_limits["max_cache_size"] < 30:
        _dynamic_limits["max_cache_size"] += 2
        logger.info(f"内存使用率较低({memory_percent}%)，提高最大缓存大小至 {_dynamic_limits['max_cache_size']}")
    
    # 记录调整时间
    _last_adjustment_time = current_time
    
    # 如果有调整，记录日志
    if _dynamic_limits != old_limits:
        logger.info(f"资源限制自动调整完成:")
        logger.info(f"  最大缓存: {old_limits['max_cache_size']} → {_dynamic_limits['max_cache_size']}")
        logger.info(f"  最大并发: {old_limits['max_concurrent_requests']} → {_dynamic_limits['max_concurrent_requests']}")
        logger.info(f"  系统状态: CPU={cpu_percent}%, 内存={memory_percent}%")
    
    return _dynamic_limits.copy()

def get_current_limits() -> Dict[str, Any]:
    """
    获取当前资源限制
    
    Returns:
        当前的资源限制配置
    """
    return _dynamic_limits.copy()

def reset_limits_to_default():
    """
    重置资源限制为默认值
    """
    global _dynamic_limits
    
    _dynamic_limits = {
        "max_cache_size": settings.MAX_CACHE_SIZE,
        "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
    }
    logger.info("资源限制已重置为默认值")

async def start_resource_monitor():
    """
    启动资源监控后台任务
    """
    logger.info("启动资源监控后台任务")
    
    while True:
        try:
            adjust_resource_limits()
            await asyncio.sleep(_adjustment_interval)
        except Exception as e:
            logger.error(f"资源监控任务出错: {e}")
            await asyncio.sleep(_adjustment_interval)

def get_resource_stats() -> Dict[str, Any]:
    """
    获取资源监控统计信息
    
    Returns:
        资源统计信息
    """
    system_metrics = SystemMonitor.get_system_metrics()
    
    return {
        "current_limits": get_current_limits(),
        "default_limits": {
            "max_cache_size": settings.MAX_CACHE_SIZE,
            "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS
        },
        "system_metrics": {
            "cpu_percent": system_metrics.get("metrics", {}).get("cpu", {}).get("percent", 0),
            "memory_percent": system_metrics.get("metrics", {}).get("memory", {}).get("percent", 0),
            "last_update": system_metrics.get("timestamp")
        },
        "last_adjustment": datetime.fromtimestamp(_last_adjustment_time).isoformat() if _last_adjustment_time > 0 else None,
        "next_adjustment_in": max(0, _adjustment_interval - (time.time() - _last_adjustment_time))
    } 