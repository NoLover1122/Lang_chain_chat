# ~/lang_chain_chat_test/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router  # 导入主路由
from app.utils.cache_manager import cache_manager
from app.utils.logging import logger
from config.settings import settings
from app.core.agent_manager import agent_manager  # 导入agent_manager以便在启动时初始化

# 创建FastAPI应用
app = FastAPI(
    title="学者数字分身API",
    description="基于LangChain的学者数字分身问答系统",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(router)

_app_initialized = False

@app.on_event("startup")
async def startup_event():
    """启动事件处理函数"""
    global _app_initialized
    if _app_initialized:
        logger.warning("应用程序启动事件已经执行过，跳过")
        return
    
    logger.info("应用程序启动中...")
    
    # 启动缓存管理器
    await cache_manager.start()
    
    # 启动Agent管理器
    await agent_manager.start()
    
    logger.info("应用程序启动完成")
    _app_initialized = True

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件处理函数"""
    logger.info("应用程序关闭中...")
    
    # 停止缓存管理器
    await cache_manager.stop()
    
    # 关闭Agent管理器
    await agent_manager.shutdown()
    
    logger.info("应用程序关闭完成")

# 运行应用
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )