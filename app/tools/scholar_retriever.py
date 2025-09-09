# app/retrieval/scholar_retriever.py
"""
学者检索器实现
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from config.settings import settings
from app.tools.vectorstore import get_scholar_by_id, setup_papers_vectorstore

logger = logging.getLogger("app")

class ScholarRetriever:
    """学者知识检索器"""
    
    def __init__(self, scholar_open_id: str = None):
        """
        初始化学者检索器
        
        Args:
            scholar_open_id: 学者ID，用于过滤检索结果
        """
        self.scholar_open_id = scholar_open_id
        self.papers_vectorstore = setup_papers_vectorstore()
        logger.info(f"初始化学者检索器: scholar_open_id={scholar_open_id}")
    
    def get_scholar_info(self) -> Optional[Dict[str, Any]]:
        """
        获取学者基本信息
        
        Returns:
            学者信息字典
        """
        if not self.scholar_open_id:
            return None
            
        # 使用直接ID查询获取学者信息
        return get_scholar_by_id(self.scholar_open_id)
    
    def retrieve_relevant_papers(self, query: str, top_k: int = None) -> List[Document]:
        """
        检索与查询相关的论文
        
        Args:
            query: 查询文本
            top_k: 返回的最大文档数
            
        Returns:
            相关论文文档列表
        """
        try:
            k = top_k or settings.RETRIEVER_K
            logger.info(f"检索相关论文: query='{query}', top_k={k}, scholar_open_id={self.scholar_open_id}")
            
            # 创建过滤条件
            filter_condition = None
            if self.scholar_open_id:
                filter_condition = {"must": [{"key": "scholar_open_id", "match": {"value": self.scholar_open_id}}]}
            
            # 检索相关文档
            retriever = self.papers_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                    "filter": filter_condition
                }
            )
            
            documents = retriever.get_relevant_documents(query)
            logger.info(f"找到 {len(documents)} 篇相关论文")
            return documents
            
        except Exception as e:
            logger.error(f"检索相关论文时出错: {e}")
            return []