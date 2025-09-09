import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional


class ApiStatsManager:
    """
    API统计管理器，用于收集和管理API调用统计信息
    """
    
    def __init__(self):
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "requests_per_scholar": {},
            "average_processing_time": 0.0,
            "start_time": datetime.now().isoformat(),
            "last_reset": datetime.now().isoformat()
        }
        self._lock = threading.Lock()  # 线程安全
    
    def update_stats(self, processing_time: float, success: bool = True, scholar_open_id: Optional[str] = None):
        """
        更新API统计信息
        
        Args:
            processing_time: 处理时间（秒）
            success: 是否成功
            scholar_open_id: 学者ID（可选）
        """
        with self._lock:
            self._stats["total_requests"] += 1
            self._stats["total_processing_time"] += processing_time
            
            if success:
                self._stats["successful_requests"] += 1
            else:
                self._stats["failed_requests"] += 1
            
            # 更新平均处理时间
            if self._stats["total_requests"] > 0:
                self._stats["average_processing_time"] = (
                    self._stats["total_processing_time"] / self._stats["total_requests"]
                )
            
            # 按学者统计
            if scholar_open_id:
                if scholar_open_id not in self._stats["requests_per_scholar"]:
                    self._stats["requests_per_scholar"][scholar_open_id] = 0
                self._stats["requests_per_scholar"][scholar_open_id] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取当前统计信息
        
        Returns:
            包含所有统计信息的字典
        """
        with self._lock:
            # 返回统计信息的深拷贝，避免外部修改
            return {
                "total_requests": self._stats["total_requests"],
                "successful_requests": self._stats["successful_requests"],
                "failed_requests": self._stats["failed_requests"],
                "average_processing_time": round(self._stats["average_processing_time"], 4),
                "total_processing_time": round(self._stats["total_processing_time"], 4),
                "requests_per_scholar": dict(self._stats["requests_per_scholar"]),
                "start_time": self._stats["start_time"],
                "last_reset": self._stats["last_reset"],
                "uptime_seconds": (datetime.now() - datetime.fromisoformat(self._stats["start_time"].replace('Z', '+00:00'))).total_seconds()
            }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        获取摘要统计信息（用于health check等场景）
        
        Returns:
            简化的统计信息
        """
        stats = self.get_stats()
        return {
            "total_requests": stats["total_requests"],
            "successful_requests": stats["successful_requests"],
            "failed_requests": stats["failed_requests"],
            "average_processing_time": stats["average_processing_time"]
        }
    
    def reset_stats(self):
        """
        重置所有统计信息
        """
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_processing_time": 0.0,
                "requests_per_scholar": {},
                "average_processing_time": 0.0,
                "start_time": self._stats["start_time"],  # 保持启动时间
                "last_reset": datetime.now().isoformat()
            }
    
    def get_top_scholars(self, limit: int = 10) -> list:
        """
        获取请求最多的学者列表
        
        Args:
            limit: 返回数量限制
            
        Returns:
            按请求数排序的学者列表
        """
        with self._lock:
            scholars = [
                {"scholar_open_id": scholar_open_id, "requests": count}
                for scholar_open_id, count in self._stats["requests_per_scholar"].items()
            ]
            return sorted(scholars, key=lambda x: x["requests"], reverse=True)[:limit]


# 全局单例实例
_api_stats_manager = ApiStatsManager()


def get_api_stats_manager() -> ApiStatsManager:
    """
    获取API统计管理器实例（单例模式）
    
    Returns:
        ApiStatsManager: API统计管理器实例
    """
    return _api_stats_manager


# 便捷函数
def update_api_stats(processing_time: float, success: bool = True, scholar_open_id: Optional[str] = None):
    """
    更新API统计信息的便捷函数
    """
    _api_stats_manager.update_stats(processing_time, success, scholar_open_id)


def get_api_stats() -> Dict[str, Any]:
    """
    获取API统计信息的便捷函数
    """
    return _api_stats_manager.get_summary_stats()


def reset_api_stats():
    """
    重置API统计信息的便捷函数
    """
    _api_stats_manager.reset_stats() 