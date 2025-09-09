import asyncio
from typing import Optional
import aiohttp
from aiohttp import ClientSession, TCPConnector
from src.utils.logging import logger

# 全局HTTP会话池
_http_session: Optional[ClientSession] = None
_session_lock = asyncio.Lock()

async def get_http_session() -> ClientSession:
    """
    获取全局HTTP会话实例（单例模式）
    
    Returns:
        ClientSession: HTTP会话实例
    """
    global _http_session
    
    async with _session_lock:
        if _http_session is None or _http_session.closed:
            connector = TCPConnector(
                limit=100,                # 总连接数限制
                limit_per_host=30,        # 每个主机连接数限制
                force_close=False,        # 不强制关闭连接
                keepalive_timeout=60,     # 保持连接时间
                enable_cleanup_closed=True # 启用清理已关闭连接
            )
            _http_session = ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=300,        # 总超时增加到10分钟
                connect=30,       # 连接超时30秒
                sock_read=300,    # 明确设置读取超时为9分钟
                sock_connect=30   # 套接字连接超时30秒
            )
        )
            logger.info("HTTP会话池初始化完成")
    
    return _http_session

async def close_http_session():
    """
    关闭HTTP会话池
    """
    global _http_session
    
    async with _session_lock:
        if _http_session and not _http_session.closed:
            await _http_session.close()
            _http_session = None
            logger.info("HTTP会话池已关闭")

async def get_session_stats() -> dict:
    """
    获取HTTP会话池统计信息
    
    Returns:
        统计信息字典
    """
    global _http_session
    
    if _http_session is None or _http_session.closed:
        return {"status": "closed", "connections": 0}
    
    connector = _http_session.connector
    if hasattr(connector, '_conns'):
        total_connections = sum(len(conns) for conns in connector._conns.values())
    else:
        total_connections = 0
    
    return {
        "status": "active",
        "total_connections": total_connections,
        "limit": connector.limit if hasattr(connector, 'limit') else 100,
        "limit_per_host": connector.limit_per_host if hasattr(connector, 'limit_per_host') else 30
    } 