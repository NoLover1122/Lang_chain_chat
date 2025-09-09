import os
import sys
from loguru import logger
from config.settings import settings

# 默认日志格式
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

def setup_logger():
    """
    配置全局日志系统
    """
    # 移除默认处理器
    logger.remove()
    
    # 添加标准输出处理器
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
    )
    
    # 确保日志目录存在
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # 添加文件处理器 - 全部日志
    logger.add(
        os.path.join(settings.LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
        rotation="00:00",  # 每天零点创建新文件
        retention="7 days",  # 保留7天的日志
        format=LOG_FORMAT,
        level=settings.LOG_LEVEL,
        compression="zip",  # 压缩旧日志
        encoding="utf-8",
    )
    
    # 添加文件处理器 - 仅错误日志
    logger.add(
        os.path.join(settings.LOG_DIR, "error_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        format=LOG_FORMAT,
        level="ERROR",
        compression="zip",
        encoding="utf-8",
    )
    
    # 添加文件处理器 - DEBUG日志（开发调试用）
    logger.add(
        os.path.join(settings.LOG_DIR, "debug_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="3 days",  # DEBUG日志只保留3天
        format=LOG_FORMAT,
        level="DEBUG",
        compression="zip",
        encoding="utf-8",
    )
    
    logger.info("日志系统初始化完成")
    logger.debug("DEBUG级别日志已启用")
    return logger

# 初始化日志配置
setup_logger() 