from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色，如user或assistant")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户当前消息")
    history: Optional[List[Message]] = Field(default=[], description="历史消息列表")
    scholar_open_id: str = Field(..., description="学者ID")
    stream: bool = Field(default=True, description="是否使用流式响应")

class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    limit: int = Field(default=10, description="返回结果数量限制")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件") 