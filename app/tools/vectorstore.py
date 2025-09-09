# app/retrieval/vectorstore.py
"""
向量存储配置
"""
import logging
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from app.tools.embeddings import get_embeddings
from config.settings import settings

logger = logging.getLogger("app")

def setup_vectorstore(collection_name=None):
    """
    设置Qdrant向量存储（用于语义检索）
    
    Args:
        collection_name: 集合名称，默认使用论文集合
        
    Returns:
        配置好的向量存储
    """
    try:
        # 使用指定的集合名称或默认使用论文集合
        collection = collection_name or settings.VECTOR_COLLECTION_PAPERS
        logger.info(f"连接Qdrant向量存储: {settings.VECTOR_HOST}:{settings.VECTOR_PORT}, 集合: {collection}")
        
        # 获取嵌入服务
        embeddings = get_embeddings()
        
        # 连接到Qdrant
        client = QdrantClient(host=settings.VECTOR_HOST, port=settings.VECTOR_PORT)
        
        # 创建向量存储
        vectorstore = Qdrant(
            client=client,
            collection_name=collection,
            embeddings=embeddings,
        )
        
        logger.info("向量存储设置成功")
        return vectorstore
    
    except Exception as e:
        logger.error(f"设置向量存储时出错: {e}")
        raise

def get_qdrant_client():
    """
    获取Qdrant客户端（用于ID检索）
    
    Returns:
        Qdrant客户端实例
    """
    try:
        logger.info(f"连接Qdrant客户端: {settings.VECTOR_HOST}:{settings.VECTOR_PORT}")
        client = QdrantClient(host=settings.VECTOR_HOST, port=settings.VECTOR_PORT)
        logger.info("Qdrant客户端连接成功")
        return client
    except Exception as e:
        logger.error(f"连接Qdrant客户端时出错: {e}")
        raise

def get_scholar_by_id(scholar_open_id):
    """
    通过ID直接获取学者信息
    
    Args:
        scholar_open_id: 学者ID
        
    Returns:
        学者信息字典，如果未找到则返回None
    """
    try:
        client = get_qdrant_client()
        # 确保 scholar_open_id 是字符串
        scholar_open_id_str = str(scholar_open_id)
        logger.info(f"通过ID检索学者: {scholar_open_id_str}")
        
        # 创建精确匹配过滤条件
        filter_condition = {"must": [{"key": "scholar_open_id", "match": {"value": scholar_open_id_str}}]}
        
        # 执行过滤查询（不使用向量相似度）
        results = client.scroll(
            collection_name=settings.VECTOR_COLLECTION_SCHOLARS,
            scroll_filter=filter_condition,
            limit=1
        )
        
        # 检查结果结构和内容
        if isinstance(results, tuple) and len(results) > 0 and results[0]:
            points = results[0]
            if points and len(points) > 0:
                logger.info(f"找到学者: {scholar_open_id_str}")
                return points[0].payload
        elif hasattr(results, 'points') and results.points and len(results.points) > 0:
            logger.info(f"找到学者: {scholar_open_id_str}")
            return results.points[0].payload
        
        # 尝试其他可能的格式
        try:
            # 如果是API v1.0.x格式
            if results and len(results) > 0 and hasattr(results[0], 'payload'):
                logger.info(f"找到学者(格式1): {scholar_open_id_str}")
                return results[0].payload
            # 如果是API v0.x格式
            elif results and isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and 'payload' in results[0]:
                    logger.info(f"找到学者(格式2): {scholar_open_id_str}")
                    return results[0]['payload']
        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"处理结果时出错: {e}, 结果类型: {type(results)}")
        
        logger.info(f"未找到学者: {scholar_open_id_str}")
        return None
        
    except Exception as e:
        logger.error(f"检索学者时出错: {e}")
        return None
        
def setup_papers_vectorstore():
    """设置论文向量存储（用于语义检索）"""
    return setup_vectorstore(settings.VECTOR_COLLECTION_PAPERS)