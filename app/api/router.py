# app/api/routes/scholar_router.py
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks, Response, Query
from fastapi.responses import StreamingResponse
import time
import json
import asyncio
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from app.models.request import ChatRequest, SearchRequest
from app.models.response import (
    HealthCheckResponse, ClearCacheResponse, ScholarInfo, 
    ScholarListResponse, ResetScholarResponse, SystemMetricsResponse, 
    CacheStatusResponse
)
from app.core.agent_manager import agent_manager
from app.utils.system_monitor import SystemMonitor
from app.utils.logging import logger
from app.utils.cache_manager import cache_manager
from config.settings import settings
from app.utils.concurrency import get_or_create_concurrency_controller, cleanup_controller_by_key
from app.utils.rate_limit import limiter, get_scholar_open_id  # 导入共享的限流器实例和获取学者ID的函数

# 从settings导入资源管理配置
MAX_CACHE_SIZE = settings.MAX_CACHE_SIZE  # 最大缓存学者数量
MAX_CONCURRENT_REQUESTS = settings.MAX_CONCURRENT_REQUESTS  # 每个学者最大并发请求数
INACTIVE_TIMEOUT = settings.INACTIVE_TIMEOUT  # 学者不活跃超时时间(秒)

# API统计信息
api_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_time": 0,
    "requests_per_scholar": {},
    "average_processing_time": 0
}

# 主路由
router = APIRouter()

@router.get("/health")
@limiter.limit(settings.RATE_LIMIT_HEALTH)
async def health_check(request: Request):
    """健康检查端点，同时返回性能统计信息"""
    try:
        start_time = time.time()
        
        # 获取AgentManager状态
        agent_stats = agent_manager.get_stats()
        system_metrics = SystemMonitor.get_system_metrics()
        
        # 构建健康信息
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cached_scholars": len(agent_manager.agent_cache),
            "active_scholars": sum(1 for _, last_active in agent_manager.last_active.items() 
                               if time.time() - last_active < INACTIVE_TIMEOUT/2),
            "system_metrics": {
                "cpu_percent": system_metrics.get("metrics", {}).get("cpu", {}).get("percent", 0),
                "memory_percent": system_metrics.get("metrics", {}).get("memory", {}).get("percent", 0),
                "last_update": system_metrics.get("timestamp")
            },
            "api_stats": {
                "total_requests": api_stats["total_requests"],
                "successful_requests": api_stats["successful_requests"],
                "failed_requests": api_stats["failed_requests"],
                "average_processing_time": api_stats["average_processing_time"]
            },
            "performance_stats": {
                "response_time": 0
            },
            "resource_limits": {
                "max_cache_size": MAX_CACHE_SIZE,
                "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "inactive_timeout_seconds": INACTIVE_TIMEOUT
            },
            "session_info": {
                "total_sessions": len(agent_manager.sessions),
                "active_users": len(agent_manager.user_sessions)
            },
            "queues": {
                "total_pending": agent_stats["queues"]["total_pending"],
                "active_requests": agent_stats["queues"]["active_requests"]
            }
        }
        
        # 添加响应时间
        health_info["performance_stats"]["response_time"] = time.time() - start_time
        
        logger.info("健康检查执行完成")
        logger.info(f"响应时间: {health_info['performance_stats']['response_time']:.6f}秒")
        
        return health_info
        
    except Exception as e:
        logger.error("健康检查失败", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# 在 chat 路由中，我们需要确保流式回调能够正确处理新格式的回复
@router.post("/chat_stream")
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat(request: Request, chat_request: ChatRequest):
    """
    统一聊天接口 - 使用流式处理
    """
    scholar_open_id = chat_request.scholar_open_id
    request.state.scholar_open_id = scholar_open_id
    
    controller = await get_or_create_concurrency_controller(
        f"scholar:{scholar_open_id}", 
        settings.MAX_CONCURRENT_REQUESTS
    )
    
    async def stream_generator():
        try:
            await controller.acquire()
            logger.debug(f"获取并发控制器信号量: scholar:{scholar_open_id}, 当前并发: {controller.active_tasks}")
            
            start_time = time.time()
            api_stats["total_requests"] += 1
            
            try:
                history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in chat_request.history
                ]
                
                session_id = None
                for msg in reversed(chat_request.history):
                    if hasattr(msg, 'metadata') and msg.metadata and 'session_id' in msg.metadata:
                        session_id = msg.metadata['session_id']
                        break
                
                user_id = getattr(chat_request, 'user_id', f"user_{int(time.time())}")
                
                # 在流式处理开始前就创建会话
                session = await agent_manager.get_or_create_session(
                    user_id=user_id,
                    scholar_open_id=scholar_open_id,
                    session_id=session_id
                )
                session_id = session.session_id  # 使用确定的会话ID
                
                # 先将用户消息添加到会话
                session.messages.append({
                    "role": "user",
                    "content": chat_request.message
                })
                
                response_queue = asyncio.Queue()
                full_response = ""  # 直接累积字符串而不是列表
                
                # 创建一个自定义回调，处理 ReAct 格式的回应
                async def stream_callback(token: str):
                    # 如果是结束标志，直接传递
                    if token == "[DONE]":
                        await response_queue.put(token)
                        return
                    
                    # 其他情况下直接传递token，因为 ScholarAgent 已经处理过滤
                    await response_queue.put(token)
                
                start_event = {
                    "event": "start",
                    "data": {
                        "answer": "",
                        "retrievalList": [],
                        "refContent": [],
                        "status": "start"
                    }
                }
                yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                
                # 使用确定的会话ID
                task_id = await agent_manager.enqueue_request(
                    user_id=user_id,
                    scholar_open_id=scholar_open_id,
                    query=chat_request.message,
                    session_id=session_id,  # 使用已创建的会话ID
                    streaming=True,
                    callback=stream_callback
                )
                
                # 使用超时保护和错误处理提升稳定性
                timeout_seconds = 180  # 总超时时间
                start_time = time.time()
                last_activity_time = start_time
                ACTIVITY_TIMEOUT = 15.0  # 15秒无活动视为卡住
                
                while True:
                    try:
                        # 使用更长的超时来等待响应，提高在网络延迟情况下的稳定性
                        token = await asyncio.wait_for(response_queue.get(), timeout=0.5)
                        last_activity_time = time.time()
                        
                        if token == "[DONE]":
                            logger.debug("收到流式响应完成标志")
                            break
                        
                        # 直接累积到完整响应字符串
                        full_response += token
                        
                        # 向客户端发送数据事件
                        event_data = {
                            "event": "data",
                            "data": {
                                "answer": token,
                                "retrievalList": [],
                                "refContent": [],
                                "status": "generating"
                            }
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        
                    except asyncio.TimeoutError:
                        # 检查任务是否仍在处理
                        current_time = time.time()
                        
                        # 总超时检查
                        if current_time - start_time > timeout_seconds:
                            logger.warning(f"任务 {task_id} 总处理时间超过 {timeout_seconds} 秒，强制结束")
                            break
                        
                        # 无活动超时检查
                        if current_time - last_activity_time > ACTIVITY_TIMEOUT:
                            logger.warning(f"任务 {task_id} 超过 {ACTIVITY_TIMEOUT} 秒无活动，可能卡住")
                            
                            # 检查是否已有足够响应
                            if len(full_response) > 100:
                                logger.info("已有足够响应内容，强制结束流式处理")
                                break
                            
                            # 检查任务是否仍在队列或处理中
                            if (task_id not in agent_manager.active_requests and 
                                task_id not in agent_manager.pending_results):
                                logger.info(f"任务 {task_id} 已不在处理队列中，结束等待")
                                break
                            
                            # 继续等待
                            continue
                        
                        # 任务仍在处理，继续等待
                        continue
                        
                    except Exception as e:
                        logger.error(f"获取流式响应时出错: {str(e)}")
                        error_response = {
                            "event": "error", 
                            "data": {
                                "error": f"生成过程中出错: {str(e)}"
                            }
                        }
                        yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                        return
                
                # 检查响应是否为空
                if not full_response.strip():
                    logger.warning(f"任务 {task_id} 生成了空响应")
                    error_response = {
                        "event": "error", 
                        "data": {
                            "error": "生成内容为空，请重新提问"
                        }
                    }
                    yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                    return
                
                # 流式处理完成后，使用同一个会话添加助手回复
                session.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                session.last_active = time.time()
                
                # 发送完成事件
                end_event = {
                    "event": "end",
                    "data": {
                        "answer": full_response,
                        "retrievalList": [],
                        "refContent": [],
                        "status": "done",
                        "session_id": session_id,
                        "processing_time": time.time() - start_time  # 添加处理时间统计
                    }
                }
                yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
                
                # 更新API统计
                processing_time = time.time() - start_time
                api_stats["successful_requests"] += 1
                api_stats["total_processing_time"] += processing_time
                api_stats["average_processing_time"] = (
                    api_stats["total_processing_time"] / api_stats["successful_requests"]
                )
                
                # 更新每个学者的请求计数
                if scholar_open_id not in api_stats["requests_per_scholar"]:
                    api_stats["requests_per_scholar"][scholar_open_id] = 0
                api_stats["requests_per_scholar"][scholar_open_id] += 1
                
                logger.info(f"流式处理完成，生成内容长度: {len(full_response)}，处理时间: {processing_time:.2f}秒")
                
            finally:
                controller.release()
                logger.debug(f"释放并发控制器信号量: scholar:{scholar_open_id}, 剩余并发: {controller.active_tasks}")
                
        except Exception as e:
            try:
                controller.release()
                logger.debug(f"异常情况下释放信号量: scholar:{scholar_open_id}, 剩余并发: {controller.active_tasks}")
            except:
                pass
                
            api_stats["failed_requests"] += 1
            logger.error(f"流式处理失败: {str(e)}")
            error_response = {
                "event": "error", 
                "data": {
                    "error": f"抱歉，生成过程中出现错误: {str(e)}"
                }
            }
            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
    
    # 确保返回正确的StreamingResponse
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"  # 明确设置内容类型
        }
    )

@router.get("/list_scholars")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def list_scholars(request: Request):
    """
    列出所有已缓存的学者 - 运行时监控接口
    """
    try:
        current_time = datetime.now().timestamp()
        
        # 获取Agent管理器状态
        stats = agent_manager.get_stats()
        agent_details = stats["agents"]["details"]
        
        scholar_info = []
        for agent in agent_details:
            scholar_open_id = agent["scholar_open_id"]
            profile = agent_manager.scholar_profiles.get(scholar_open_id)
            
            scholar_info.append({
                "scholar_open_id": scholar_open_id,
                "name": profile.display_name if profile else "未知学者",
                "last_active": datetime.fromtimestamp(current_time - agent["idle_time"]).isoformat(),
                "inactive_for": f"{agent['idle_time']/60:.1f}分钟",
                "status": "活跃" if agent["idle_time"] < INACTIVE_TIMEOUT/2 else "不活跃"
            })
        
        return {
            "status": "success",
            "scholars": scholar_info,
            "count": len(scholar_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("列出学者失败", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/clear_cache")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def clear_cache(request: Request, background_tasks: BackgroundTasks):
    """清除Agent缓存"""
    try:
        # 获取当前缓存大小
        cache_size = len(agent_manager.agent_cache)
        
        # 使用后台任务清理所有缓存
        background_tasks.add_task(agent_manager.clean_cache, clean_all=True)
        
        return {
            "status": "success",
            "message": f"已开始清除 {cache_size} 个学者的Agent缓存",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("清除缓存失败", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/system_metrics")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_system_metrics(request: Request):
    """
    获取系统指标
    """
    try:
        # 直接获取API格式的系统指标
        metrics_data = SystemMonitor.get_api_format_metrics()
        
        # 添加资源限制信息
        metrics_data["resource_limits"] = {
            "max_cache_size": MAX_CACHE_SIZE,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
        }
        
        # 添加Agent管理器状态
        agent_stats = agent_manager.get_stats()
        metrics_data["agent_manager"] = {
            "agents": agent_stats["agents"]["total"],
            "sessions": agent_stats["sessions"]["total"],
            "active_sessions": agent_stats["sessions"]["active"],
            "queued_requests": agent_stats["queues"]["total_pending"],
            "active_requests": agent_stats["queues"]["active_requests"]
        }
        
        return metrics_data
    except Exception as e:
        logger.error("获取系统指标失败", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/reset_scholar/{scholar_open_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def reset_scholar(request: Request, scholar_open_id: str):
    """
    重置指定学者的缓存
    """
    try:
        # 检查学者是否在缓存中
        if scholar_open_id not in agent_manager.agent_cache:
            return {
                "status": "warning",
                "message": f"学者 {scholar_open_id} 不在缓存中",
                "timestamp": datetime.now().isoformat()
            }
        
        # 获取学者名称用于日志
        profile = agent_manager.scholar_profiles.get(scholar_open_id)
        scholar_name = profile.display_name if profile else scholar_open_id
        
        # 清理Agent缓存
        async with agent_manager.agent_lock:
            if scholar_open_id in agent_manager.agent_cache:
                del agent_manager.agent_cache[scholar_open_id]
            
            if scholar_open_id in agent_manager.last_active:
                del agent_manager.last_active[scholar_open_id]
                
            if scholar_open_id in agent_manager.scholar_concurrency:
                del agent_manager.scholar_concurrency[scholar_open_id]
        
        # 清理流式代理池
        async with agent_manager.streaming_agent_lock:
            streaming_agents_count = 0
            if scholar_open_id in agent_manager.streaming_agent_pools:
                streaming_agents_count = len(agent_manager.streaming_agent_pools[scholar_open_id])
                del agent_manager.streaming_agent_pools[scholar_open_id]
        
        # 清理学者的会话
        if scholar_open_id in agent_manager.scholar_sessions:
            session_ids = agent_manager.scholar_sessions[scholar_open_id].copy()
            
            async with agent_manager.session_lock:
                for session_id in session_ids:
                    if session_id in agent_manager.sessions:
                        session = agent_manager.sessions[session_id]
                        user_id = session.user_id
                        
                        # 从用户会话中移除
                        if user_id in agent_manager.user_sessions and session_id in agent_manager.user_sessions[user_id]:
                            agent_manager.user_sessions[user_id].remove(session_id)
                        
                        # 删除会话
                        del agent_manager.sessions[session_id]
                
                # 清空学者会话列表
                agent_manager.scholar_sessions[scholar_open_id] = []
        
        # 取消学者的所有队列中请求
        async with agent_manager.queue_lock:
            if scholar_open_id in agent_manager.request_queues:
                queue = agent_manager.request_queues[scholar_open_id]
                cancelled_count = len(queue)
                
                # 取消所有任务
                for task in queue:
                    if task.task_id in agent_manager.pending_results:
                        future = agent_manager.pending_results[task.task_id]
                        if not future.done():
                            future.set_result({
                                "status": "cancelled",
                                "message": f"学者 {scholar_name} 已重置，任务已取消"
                            })
                
                # 清空队列
                queue.clear()
            else:
                cancelled_count = 0
        
        # 清理并发控制器
        controller_key = f"scholar:{scholar_open_id}"
        await cleanup_controller_by_key(controller_key)
        
        return {
            "status": "success",
            "message": f"已重置学者 {scholar_name} 的缓存",
            "cancelled_requests": cancelled_count,
            "streaming_agents_cleared": streaming_agents_count,  # 添加流式代理清理信息
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"重置学者 {scholar_open_id} 缓存失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/sessions/{user_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_user_sessions(request: Request, user_id: str):
    """获取用户的所有会话"""
    try:
        sessions = await agent_manager.get_user_sessions(user_id)
        return {
            "status": "success",
            "user_id": user_id,
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取用户 {user_id} 的会话时出错: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/sessions/{user_id}/{session_id}/history")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_session_history(request: Request, user_id: str, session_id: str):
    """获取会话历史"""
    try:
        # 验证session_id是否属于该用户
        if user_id not in agent_manager.user_sessions or session_id not in agent_manager.user_sessions[user_id]:
            return {
                "status": "error",
                "message": f"未找到用户 {user_id} 的会话 {session_id}",
                "timestamp": datetime.now().isoformat()
            }
        
        history = await agent_manager.get_session_history(session_id)
        session = agent_manager.sessions.get(session_id)
        
        # 获取学者信息
        scholar_open_id = session.scholar_open_id if session else None
        profile = agent_manager.scholar_profiles.get(scholar_open_id) if scholar_open_id else None
        
        return {
            "status": "success",
            "user_id": user_id,
            "session_id": session_id,
            "scholar_open_id": scholar_open_id,
            "scholar_name": profile.display_name if profile else "未知学者",
            "created_at": datetime.fromtimestamp(session.created_at).isoformat() if session else None,
            "last_active": datetime.fromtimestamp(session.last_active).isoformat() if session else None,
            "messages": history,
            "message_count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except KeyError:
        return {
            "status": "error",
            "message": f"未找到会话 {session_id}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取会话 {session_id} 历史时出错: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.delete("/sessions/{user_id}/{session_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def delete_session(request: Request, user_id: str, session_id: str):
    """删除会话"""
    try:
        # 验证session_id是否属于该用户
        if user_id not in agent_manager.user_sessions or session_id not in agent_manager.user_sessions[user_id]:
            return {
                "status": "error",
                "message": f"未找到用户 {user_id} 的会话 {session_id}",
                "timestamp": datetime.now().isoformat()
            }
        
        async with agent_manager.session_lock:
            # 获取会话信息
            session = agent_manager.sessions.get(session_id)
            if not session:
                return {
                    "status": "error",
                    "message": f"未找到会话 {session_id}",
                    "timestamp": datetime.now().isoformat()
                }
            
            scholar_open_id = session.scholar_open_id
            
            # 从用户会话列表中移除
            agent_manager.user_sessions[user_id].remove(session_id)
            
            # 从学者会话列表中移除
            if scholar_open_id in agent_manager.scholar_sessions and session_id in agent_manager.scholar_sessions[scholar_open_id]:
                agent_manager.scholar_sessions[scholar_open_id].remove(session_id)
            
            # 删除会话
            del agent_manager.sessions[session_id]
        
        return {
            "status": "success",
            "message": f"已删除会话 {session_id}",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"删除会话 {session_id} 时出错: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/scholar_info/{scholar_open_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_scholar_info(request: Request, scholar_open_id: str):
    """获取学者信息"""
    try:
        # 尝试获取学者Agent以加载学者信息
        agent = await agent_manager.get_or_create_agent(scholar_open_id)
        scholar_info = agent.get_scholar_info()
        
        # 获取学者档案
        profile = agent_manager.scholar_profiles.get(scholar_open_id)
        
        # 获取学者会话数量
        session_count = 0
        if scholar_open_id in agent_manager.scholar_sessions:
            session_count = len(agent_manager.scholar_sessions[scholar_open_id])
        
        # 获取队列长度
        queue_length = 0
        if scholar_open_id in agent_manager.request_queues:
            queue_length = len(agent_manager.request_queues[scholar_open_id])
        
        response = {
            "status": "success",
            "scholar_open_id": scholar_open_id,
            "name": profile.display_name if profile else scholar_info.get("name", "未知学者"),
            "zh_name": scholar_info.get("zh_name", ""),
            "en_name": scholar_info.get("en_name", ""),
            "institution": scholar_info.get("institution", ""),
            "brief": scholar_info.get("brief", ""),
            "research_keywords": scholar_info.get("research_keywords", []),
            "total_papers": scholar_info.get("total_papers", 0),
            "session_count": session_count,
            "queue_length": queue_length,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"获取学者 {scholar_open_id} 信息失败: {str(e)}")
        return {
            "status": "error",
            "scholar_open_id": scholar_open_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/cache_status")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def cache_status(request: Request):
    """
    获取缓存状态信息
    """
    try:
        current_time = datetime.now().timestamp()
        
        # 获取Agent管理器状态
        stats = agent_manager.get_stats()
        agent_details = stats["agents"]["details"]
        
        # 添加流式代理池信息
        streaming_agents_total = stats["agents"].get("streaming_pools", 0)
        
        cache_info = {
            "total_scholars": len(agent_manager.agent_cache),
            "active_scholars": sum(1 for a in agent_details if a["idle_time"] < INACTIVE_TIMEOUT/2),
            "inactive_scholars": sum(1 for a in agent_details if a["idle_time"] >= INACTIVE_TIMEOUT/2),
            "streaming_agents": streaming_agents_total,  # 添加流式代理池总数
            "scholars_info": []
        }
        
        for agent in agent_details:
            scholar_open_id = agent["scholar_open_id"]
            profile = agent_manager.scholar_profiles.get(scholar_open_id)
            
            # 获取流式代理数量
            streaming_agents = agent.get("streaming_agents", 0)
            
            cache_info["scholars_info"].append({
                "scholar_open_id": scholar_open_id,
                "name": profile.display_name if profile else "未知学者",
                "last_active": datetime.fromtimestamp(current_time - agent["idle_time"]).isoformat(),
                "inactive_for": f"{agent['idle_time']/60:.1f}分钟",
                "status": "活跃" if agent["idle_time"] < INACTIVE_TIMEOUT/2 else "不活跃",
                "concurrent_requests": agent["concurrent_requests"],
                "streaming_agents": streaming_agents  # 添加流式代理数量
            })
        
        # 添加会话信息
        cache_info["sessions"] = {
            "total": stats["sessions"]["total"],
            "active": stats["sessions"]["active"],
            "users": stats["sessions"]["users"]
        }
        
        # 添加队列信息
        cache_info["queues"] = {
            "pending_requests": stats["queues"]["total_pending"],
            "active_requests": stats["queues"]["active_requests"],
            "per_scholar": [
                {
                    "scholar_open_id": q["scholar_open_id"],
                    "name": q["name"],
                    "queue_length": q["queue_length"],
                    "concurrent_requests": q["concurrent_requests"]
                }
                for q in stats["queues"]["details"]
            ]
        }
        
        # 添加缓存管理器信息
        cache_info["cache_manager"] = cache_manager.get_stats()
        
        # 添加资源共享信息
        cache_info["shared_resources"] = {
            "enabled": True,
            "description": "使用ScholarResources进行资源共享，减少内存占用和初始化时间"
        }
        
        return {
            "status": "success",
            "cache_info": cache_info,
            "max_cache_size": MAX_CACHE_SIZE,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "inactive_timeout": f"{INACTIVE_TIMEOUT/60}分钟",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("获取缓存状态失败", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.post("/cache/cleanup")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def run_cache_cleanup(request: Request, force_all: bool = False):
    """手动运行缓存清理"""
    try:
        if force_all:
            result = await agent_manager.clean_cache(clean_all=True)
            message = "已强制清理所有缓存"
        else:
            result = await cache_manager.run_cleanup()
            message = "已执行缓存清理"
        
        return {
            "status": "success",
            "message": message,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"执行缓存清理时出错: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }