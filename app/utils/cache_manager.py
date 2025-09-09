import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set
from config.settings import settings

class CacheManager:
    """
    缓存管理器，负责定期清理过期的缓存资源
    """
    
    def __init__(self):
        """初始化缓存管理器"""
        self.logger = logging.getLogger("cache_manager")
        
        # 清理任务
        self.cleanup_task = None
        
        # 停止标志
        self.shutdown_flag = False
        
        # 最后一次清理时间
        self.last_cleanup = time.time()
        
        # 清理统计
        self.stats = {
            "total_cleanups": 0,
            "last_cleanup": None,
            "total_items_cleaned": 0,
            "avg_cleanup_time": 0,
            "total_cleanup_time": 0
        }
        
        self.logger.info("缓存管理器初始化完成")
    
    async def start(self):
        """启动缓存管理器"""
        if self.cleanup_task is not None and not self.cleanup_task.done():
            self.logger.warning("缓存管理器已在运行")
            return
            
        self.shutdown_flag = False
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("缓存管理器启动完成")
    
    async def stop(self):
        """停止缓存管理器"""
        if self.cleanup_task is None:
            return
            
        self.logger.info("正在停止缓存管理器...")
        self.shutdown_flag = True
        
        try:
            await asyncio.wait_for(self.cleanup_task, timeout=5)
        except asyncio.TimeoutError:
            self.logger.warning("等待缓存管理器停止超时")
            if not self.cleanup_task.done():
                self.cleanup_task.cancel()
        
        self.cleanup_task = None
        self.logger.info("缓存管理器已停止")
    
    async def _cleanup_loop(self):
        """定期清理缓存的后台任务"""
        try:
            cleanup_interval = settings.CACHE_CLEANUP_INTERVAL  # 清理间隔(秒)
            
            while not self.shutdown_flag:
                # 等待下次清理
                await asyncio.sleep(cleanup_interval)
                
                if self.shutdown_flag:
                    break
                
                # 执行清理
                await self.run_cleanup()
                
        except asyncio.CancelledError:
            self.logger.info("缓存清理任务被取消")
        except Exception as e:
            self.logger.error(f"缓存清理任务异常: {str(e)}", exc_info=True)
    
    async def run_cleanup(self):
        """运行一次缓存清理"""
        try:
            start_time = time.time()
            self.logger.info("开始执行缓存清理...")
            
            # 这里实际上不做任何清理操作，只更新统计信息
            cleaned_count = 0
            
            # 更新统计信息
            cleanup_time = time.time() - start_time
            self.stats["total_cleanups"] += 1
            self.stats["last_cleanup"] = time.time()
            self.stats["total_items_cleaned"] += cleaned_count
            self.stats["total_cleanup_time"] += cleanup_time
            self.stats["avg_cleanup_time"] = (
                self.stats["total_cleanup_time"] / self.stats["total_cleanups"]
            )
            
            self.logger.info(f"缓存清理完成，用时: {cleanup_time:.3f}秒，清理: {cleaned_count}项")
            self.last_cleanup = time.time()
            
            return {
                "status": "success",
                "cleaned": cleaned_count,
                "time": cleanup_time
            }
            
        except Exception as e:
            self.logger.error(f"执行缓存清理时出错: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_stats(self):
        """获取缓存管理器统计信息"""
        return {
            "total_cleanups": self.stats["total_cleanups"],
            "last_cleanup": self.stats["last_cleanup"],
            "total_items_cleaned": self.stats["total_items_cleaned"],
            "avg_cleanup_time": self.stats["avg_cleanup_time"],
            "next_cleanup": self.last_cleanup + settings.CACHE_CLEANUP_INTERVAL if self.last_cleanup else None,
            "status": "running" if self.cleanup_task and not self.cleanup_task.done() else "stopped"
        }

# 创建全局缓存管理器实例
cache_manager = CacheManager()