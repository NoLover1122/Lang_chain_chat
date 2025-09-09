import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    应用全局配置类，通过环境变量或.env文件加载配置
    """
    # 服务配置
    SERVICE_NAME: str = "Digital Chat"
    SERVICE_HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 8009))
    WORKERS: int = int(os.getenv("WORKERS", 4))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # LLM配置
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "sk-32e118fa6f794bb58d2805a30786572d")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://144.214.55.148:8000/v1")
    LLM_DEFAULT_MODEL: str = os.getenv("LLM_DEFAULT_MODEL", "qwen3-235b-int4-flat")
    LLM_RETRY_COUNT: int = int(os.getenv("LLM_RETRY_COUNT", "3"))
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "2.0"))
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "300.0"))
    
    # 向量检索配置
    VECTOR_HOST: str = os.getenv("VECTOR_HOST", "localhost")
    VECTOR_PORT: int = int(os.getenv("VECTOR_PORT", 6333))
    VECTOR_COLLECTION_PAPERS: str = os.getenv("VECTOR_COLLECTION_PAPERS", "scholar_papers_main") 
    VECTOR_COLLECTION_SCHOLARS: str = os.getenv("VECTOR_COLLECTION_SCHOLARS", "scholar_all_main")
    EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")
    
    # 线程池配置
    THREAD_POOL_SIZE: int = int(os.getenv("THREAD_POOL_SIZE", 20))

    # CORS设置
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 资源管理与缓存配置
    # AgentManager相关
    MAX_SCHOLAR_CONCURRENCY: int = int(os.getenv("MAX_SCHOLAR_CONCURRENCY", 5))  # 每个学者Agent能同时处理的最大请求数
    LLM_TEMPERATURE: float = 0.7 # LLM温度
    MAX_USER_CONCURRENCY: int = int(os.getenv("MAX_USER_CONCURRENCY", 10)) # 每个用户能同时处理的最大请求数
    AGENT_INACTIVE_TIMEOUT: int = int(os.getenv("AGENT_INACTIVE_TIMEOUT", 3600))  # 学者Agent实例闲置超时时间(秒)，超时后从内存清理
    SESSION_INACTIVE_TIMEOUT: int = int(os.getenv("SESSION_INACTIVE_TIMEOUT", 7200)) # 用户会话闲置超时时间(秒)，超时后从内存清理
    CACHE_CLEANUP_INTERVAL: int = int(os.getenv("CACHE_CLEANUP_INTERVAL", 1800))  # 缓存自动清理任务的运行间隔(秒)
    
    # 添加大写的字段名，以匹配代码中的使用方式
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", 20))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", 5))
    INACTIVE_TIMEOUT: int = int(os.getenv("INACTIVE_TIMEOUT", 3600))
    
    # 速率限制配置
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
    RATE_LIMIT_HEALTH: str = os.getenv("RATE_LIMIT_HEALTH", "60/minute")
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "15/minute")
    RATE_LIMIT_CHAT_PER_SCHOLAR: str = os.getenv("RATE_LIMIT_CHAT_PER_SCHOLAR", "15/minute")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    # LangChain相关配置
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "3"))
    MAX_TOKEN_LIMIT: int = int(os.getenv("MAX_TOKEN_LIMIT", "4000"))
    
    @property
    def QDRANT_URL(self) -> str:
        return f"http://{self.VECTOR_HOST}:{self.VECTOR_PORT}"
    
    # 使用 model_config 替代 Config 类（Pydantic v2 的方式）
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

# 创建全局设置实例
settings = Settings()