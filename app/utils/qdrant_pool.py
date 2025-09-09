import asyncio
from typing import Dict
from qdrant_client import AsyncQdrantClient
from src.utils.logging import logger

# 全局Qdrant客户端池
_qdrant_clients: Dict[str, AsyncQdrantClient] = {}
_client_lock = asyncio.Lock()

async def get_qdrant_client(host: str = "localhost", port: int = 6333) -> AsyncQdrantClient:
    """
    获取Qdrant客户端实例（连接池模式）
    
    Args:
        host: Qdrant服务器主机
        port: Qdrant服务器端口
        
    Returns:
        AsyncQdrantClient: Qdrant客户端实例
    """
    global _qdrant_clients
    
    key = f"{host}:{port}"
    
    async with _client_lock:
        if key not in _qdrant_clients:
            url = f"http://{host}:{port}"
            _qdrant_clients[key] = AsyncQdrantClient(url=url)
            logger.info(f"创建新的Qdrant客户端: {url}")
    
    return _qdrant_clients[key]

async def close_qdrant_clients():
    """
    关闭所有Qdrant客户端连接
    """
    global _qdrant_clients
    
    async with _client_lock:
        count = 0
        for key, client in _qdrant_clients.items():
            try:
                await client.close()
                count += 1
            except Exception as e:
                logger.error(f"关闭Qdrant客户端 {key} 失败: {e}")
        
        _qdrant_clients.clear()
        if count > 0:
            logger.info(f"已关闭 {count} 个Qdrant客户端连接")

async def get_qdrant_pool_stats() -> dict:
    """
    获取Qdrant客户端池统计信息
    
    Returns:
        统计信息字典
    """
    global _qdrant_clients
    
    async with _client_lock:
        return {
            "total_clients": len(_qdrant_clients),
            "endpoints": list(_qdrant_clients.keys())
        } 