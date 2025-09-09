from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型 - 所有API响应的基类"""
    status: str = Field(..., description="响应状态，如success或error")
    message: str = Field(..., description="响应消息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")
    
    class Config:
        """模型配置"""
        # 允许额外字段，以兼容原项目的响应格式
        extra = "allow"



# 健康检查相关模型
class HealthCheckResponse(BaseResponse):
    """健康检查响应模型"""
    cached_scholars: int = Field(..., description="缓存的学者数量")
    active_scholars: int = Field(..., description="活跃的学者数量")
    system_metrics: Dict[str, Any] = Field(..., description="系统指标")
    api_stats: Dict[str, Any] = Field(..., description="API统计")
    performance_stats: Dict[str, Any] = Field(..., description="性能统计")
    resource_limits: Dict[str, Any] = Field(..., description="资源限制")

# 缓存相关模型
class ClearCacheResponse(BaseResponse):
    """清理缓存响应模型"""
    cleared_count: int = Field(..., description="清理的缓存数量")

class CacheStatusResponse(BaseResponse):
    """缓存状态响应模型"""
    cached_scholars: int = Field(..., description="缓存的学者数量")
    max_cache_size: int = Field(..., description="最大缓存大小")
    scholar_details: List[Dict[str, Any]] = Field(default_factory=list, description="学者详情")

# 学者相关模型
class ScholarInfo(BaseResponse):
    """学者信息响应模型"""
    scholar_open_id: str = Field(..., description="学者ID")
    name: Optional[str] = Field(default="", description="学者姓名")
    institution: Optional[str] = Field(default="", description="所属机构")
    research_keywords: Optional[List[str]] = Field(default_factory=list, description="研究关键词")
    total_papers: Optional[int] = Field(default=0, description="论文总数")

class ScholarListResponse(BaseResponse):
    """学者列表响应模型"""
    scholars: List[Dict[str, Any]] = Field(default_factory=list, description="学者列表")

class ScholarPapersResponse(BaseResponse):
    """学者论文响应模型"""
    scholar_open_id: str = Field(..., description="学者ID")
    papers: List[Dict[str, Any]] = Field(default_factory=list, description="论文列表")
    total_count: int = Field(default=0, description="论文总数")

class ResetScholarResponse(BaseResponse):
    """重置学者缓存响应模型"""
    scholar_open_id: str = Field(..., description="学者ID")

# 系统指标相关模型
class SystemMetricsResponse(BaseResponse):
    """系统指标响应模型"""
    cpu_usage: float = Field(..., description="CPU使用率")
    memory_usage: float = Field(..., description="内存使用率")
    disk_usage: float = Field(..., description="磁盘使用率")
    network_io: Dict[str, float] = Field(..., description="网络输入输出")

# 搜索相关模型
class SearchResult(BaseModel):
    """搜索结果项模型"""
    id: str = Field(..., description="结果ID")
    score: float = Field(..., description="相关性得分")
    content: Dict[str, Any] = Field(..., description="结果内容")

class SearchResponse(BaseResponse):
    """搜索响应模型"""
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果")
    total_count: int = Field(default=0, description="结果总数")