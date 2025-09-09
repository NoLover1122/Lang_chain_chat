# app/tools/scholar_tools.py
import logging
from typing import Optional, List, Dict, Any, ClassVar
from langchain.tools import BaseTool
from app.tools.scholar_retriever import ScholarRetriever
from app.prompt.scholar_config import ScholarConfig
from pydantic import Field

logger = logging.getLogger("app")

class PaperRetrievalTool(BaseTool):
    """学术论文检索工具"""
    
    # 类变量的类型注解
    name: ClassVar[str] = "paper_retrieval"
    description: ClassVar[str] = "当问题涉及学者的专业研究、学术论文或需要专业知识回答时使用此工具。该工具可以检索与问题相关的学术文献。"
    return_direct: ClassVar[bool] = False
    
    # 实例属性的类型注解
    retriever: ScholarRetriever = Field(description="论文检索器")
    
    def __init__(self, retriever: ScholarRetriever):
        super().__init__(retriever=retriever)  # 传递给父类构造函数
    
    def _run(self, query: str) -> str:
        logger.info(f"执行论文检索工具: '{query}'")
        documents = self.retriever.retrieve_relevant_papers(query)
        
        if not documents:
            return "没有找到相关的研究文献。"
        
        # 格式化检索结果
        results = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            metadata = doc.metadata
            title = metadata.get("title", "未知标题")
            year = metadata.get("year", "")
            authors = ", ".join(metadata.get("authors", []))
            
            results.append(f"文献 {i}:\n标题: {title}\n年份: {year}\n作者: {authors}\n内容摘要: {content}\n")
        
        return "以下是检索到的相关文献:\n\n" + "\n".join(results)
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class ScholarInfoTool(BaseTool):
    """学者信息查询工具"""
    
    # 类变量的类型注解
    name: ClassVar[str] = "scholar_info"
    description: ClassVar[str] = "当问题涉及学者的个人背景、工作经历、研究领域或其他基本信息时使用此工具。适用于一般性问题。"
    return_direct: ClassVar[bool] = False
    
    # 实例属性的类型注解
    config: ScholarConfig = Field(description="学者配置")
    
    def __init__(self, config: ScholarConfig):
        super().__init__(config=config)  # 传递给父类构造函数
    
    def _run(self, query: str) -> str:
        logger.info(f"执行学者信息工具: '{query}'")
        
        # 获取学者场景信息
        scene_info = self.config.get_scene_info()
        
        # 返回学者信息
        return f"以下是学者信息:\n\n{scene_info}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)