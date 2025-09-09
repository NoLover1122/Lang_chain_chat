# app/retrieval/embeddings.py
"""
嵌入服务配置
"""
import logging
import requests
from typing import List
from langchain.embeddings.base import Embeddings
from config.settings import settings

logger = logging.getLogger("app")

class ExternalEmbeddingService(Embeddings):
    """连接外部嵌入服务的自定义嵌入类"""
    
    def __init__(self, api_url: str = None):
        """初始化嵌入服务连接器"""
        self.api_url = api_url or settings.EMBEDDING_SERVICE_URL
        logger.info(f"初始化外部嵌入服务连接: {self.api_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取文档嵌入向量"""
        try:
            # 创建一个空的结果列表
            results = []
            
            # 逐个处理文本，因为API只接受单个文本
            for text in texts:
                response = requests.post(
                    f"{self.api_url}/encode",
                    json={"text": text}  # 修改为API接受的格式：text而不是texts
                )
                response.raise_for_status()
                data = response.json()
                # 添加向量到结果列表
                results.append(data["vector"])
            
            return results
        except Exception as e:
            logger.error(f"获取文档嵌入时出错: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """获取查询嵌入向量"""
        try:
            response = requests.post(
                f"{self.api_url}/encode",
                json={"text": text}  # 修改为API接受的格式：text而不是texts数组
            )
            response.raise_for_status()
            return response.json()["vector"]  # 直接返回向量
        except Exception as e:
            logger.error(f"获取查询嵌入时出错: {e}")
            raise

def get_embeddings():
    """
    获取嵌入服务实例
    
    Returns:
        配置好的嵌入服务
    """
    try:
        logger.info(f"连接嵌入服务: {settings.EMBEDDING_SERVICE_URL}")
        embeddings = ExternalEmbeddingService()
        logger.info("嵌入服务连接成功")
        return embeddings
    
    except Exception as e:
        logger.error(f"连接嵌入服务时出错: {e}")
        raise