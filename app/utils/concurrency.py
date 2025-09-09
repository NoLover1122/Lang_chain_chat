import asyncio
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings
from app.utils.logging import logger

# 全局线程池
thread_pool = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)

# 全局资源控制器缓存
_concurrency_controllers: Dict[str, Any] = {}
_last_active_time: Dict[str, float] = {}
_global_lock = asyncio.Lock()

class ConcurrencyController:
    """
    并发控制器，控制并发请求数
    """
    
    def __init__(self, max_concurrent: int):
        """
        初始化并发控制器
        
        Args:
            max_concurrent: 最大并发数
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
        self._lock = asyncio.Lock()  # 保护active_tasks的锁
    
    def try_acquire(self) -> bool:
        """
        尝试获取信号量，不阻塞
        
        Returns:
            bool: 是否成功获取
        """
        if self.semaphore.locked():
            return False
        acquired = False
        try:
            acquired = self.semaphore.acquire_nowait()
            if acquired:
                self.active_tasks += 1
            return acquired
        except:
            if acquired:
                self.semaphore.release()
            return False
        
    async def acquire(self):
        """
        获取信号量，会阻塞
        """
        await self.semaphore.acquire()
        async with self._lock:
            self.active_tasks += 1
        
    def release(self):
        """
        释放信号量
        """
        if self.active_tasks > 0:  # 防止重复释放
            self.semaphore.release()
            self.active_tasks -= 1
    
    @property
    def available(self) -> int:
        """
        可用的并发数
        """
        return self.semaphore._value
    
    @property
    def busy(self) -> int:
        """
        已用的并发数
        """
        return self.active_tasks

async def get_or_create_concurrency_controller(key: str, max_concurrent: int) -> ConcurrencyController:
    """
    获取或创建并发控制器
    
    Args:
        key: 控制器唯一标识
        max_concurrent: 最大并发数
        
    Returns:
        ConcurrencyController: 并发控制器实例
    """
    global _concurrency_controllers, _last_active_time, _global_lock
    
    async with _global_lock:
        # 更新活跃时间
        _last_active_time[key] = time.time()
        
        # 如果不存在则创建
        if key not in _concurrency_controllers:
            logger.info(f"创建新的并发控制器: {key}, 最大并发: {max_concurrent}")
            _concurrency_controllers[key] = ConcurrencyController(max_concurrent)
            
    return _concurrency_controllers[key]

async def cleanup_inactive_controllers(timeout: int = 3600):
    """
    清理不活跃的控制器
    
    Args:
        timeout: 超时时间(秒)，默认1小时
    """
    global _concurrency_controllers, _last_active_time, _global_lock
    
    current_time = time.time()
    keys_to_remove = []
    
    async with _global_lock:
        for key, last_time in _last_active_time.items():
            if current_time - last_time > timeout:
                if _concurrency_controllers[key].active_tasks == 0:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            logger.info(f"清理不活跃的并发控制器: {key}")
            del _concurrency_controllers[key]
            del _last_active_time[key]
            
        if keys_to_remove:
            logger.info(f"已清理 {len(keys_to_remove)} 个不活跃的并发控制器")

async def run_in_thread(func, *args, **kwargs):
    """
    在线程池中运行同步函数
    
    Args:
        func: 要运行的函数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        Any: 函数执行结果
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool, 
        lambda: func(*args, **kwargs)
    )

async def cleanup_controller_by_key(key: str) -> bool:
    """
    清理特定key的并发控制器
    
    Args:
        key: 要清理的控制器唯一标识
        
    Returns:
        bool: 是否成功清理
    """
    global _concurrency_controllers, _last_active_time, _global_lock
    
    async with _global_lock:
        removed = False
        if key in _concurrency_controllers:
            # 只有在没有活跃任务时才清理
            if _concurrency_controllers[key].active_tasks == 0:
                logger.info(f"清理并发控制器: {key}")
                del _concurrency_controllers[key]
                if key in _last_active_time:
                    del _last_active_time[key]
                removed = True
            else:
                logger.warning(f"并发控制器 {key} 仍有活跃任务，无法清理")
        
        return removed 