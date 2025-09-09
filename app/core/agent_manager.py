"""
学者Agent管理器，负责管理不同学者的Agent实例并支持并发和队列
"""
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Deque, Set, Callable
from collections import deque
import logging
from pydantic import BaseModel, Field

# 导入LangChain核心消息类型
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# 导入真实的组件
from app.core.scholar_agent import ScholarAgent, ScholarResources  # 导入新的ScholarResources类
from app.core.llm import initialize_llm, CustomStreamingCallbackHandler
from app.prompt.scholar_config import ScholarConfig
from config.settings import settings

logger = logging.getLogger("app")

class ConversationSession(BaseModel):
    """会话模型，存储与特定用户的对话状态"""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    scholar_open_id: str
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RequestTask(BaseModel):
    """请求任务模型"""
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    scholar_open_id: str
    session_id: Optional[str] = None
    query: str
    created_at: float = Field(default_factory=time.time)
    streaming: bool = False
    callback: Optional[Any] = None  # 流式回调处理器
    result_future: Any = None  # asyncio.Future实例
    priority: int = 0  # 优先级，数字越小优先级越高
    
    class Config:
        arbitrary_types_allowed = True

class ScholarProfile(BaseModel):
    """学者档案模型"""
    scholar_open_id: str
    name: str = "未知学者"
    institution: str = ""
    research_interests: List[str] = []
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        if self.name and self.institution:
            return f"{self.name} ({self.institution})"
        return self.name or self.scholar_open_id

class AgentManager:
    """学者Agent管理器，负责管理不同学者的Agent实例并支持并发和队列"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """确保单例模式"""
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化Agent管理器"""
        if hasattr(self, 'initialized'):
            return
                
        self.agent_cache: Dict[str, ScholarAgent] = {}
        self.last_active: Dict[str, float] = {}
        self.scholar_profiles: Dict[str, ScholarProfile] = {}
        
        # 新增：流式Agent缓存池（限制每个学者的流式Agent数量）
        self.streaming_agent_pools: Dict[str, List[ScholarAgent]] = {}
        self.streaming_agent_lock = asyncio.Lock()
        self.MAX_STREAMING_AGENTS_PER_SCHOLAR = 3  # 每个学者最多保留3个流式Agent实例
        
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}
        self.scholar_sessions: Dict[str, List[str]] = {}
        
        self.request_queues: Dict[str, Deque[RequestTask]] = {}
        self.active_requests: Set[str] = set()
        self.pending_results: Dict[str, asyncio.Future] = {}
        
        self.scholar_concurrency: Dict[str, int] = {}
        self.user_concurrency: Dict[str, int] = {}
        
        self.workers: Dict[str, asyncio.Task] = {}
        self.shutdown_flag = False
        self.task_signal = asyncio.Event()
        
        self.queue_lock = asyncio.Lock()
        self.agent_lock = asyncio.Lock()
        self.session_lock = asyncio.Lock()
        self.clean_lock = asyncio.Lock()
        
        self.worker_task = None
        
        self.initialized = True
        logger.info("AgentManager初始化完成")
    
    def get_chat_history_as_messages(self, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """将消息历史转换为LangChain消息列表"""
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
        return lc_messages
    
    async def start(self):
        """启动Agent管理器"""
        if self.worker_task is not None and not self.worker_task.done():
            return
            
        self.shutdown_flag = False
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("AgentManager启动完成")
    
    async def get_or_create_agent(self, scholar_open_id: str) -> ScholarAgent:
        """获取或创建特定学者的Agent实例"""
        async with self.agent_lock:
            if scholar_open_id in self.agent_cache:
                logger.info(f"从缓存获取学者 {scholar_open_id} 的Agent实例")
                self.last_active[scholar_open_id] = time.time()
                return self.agent_cache[scholar_open_id]
            
            logger.info(f"创建学者 {scholar_open_id} 的新Agent实例")
            
            try:
                # 使用异步方法创建ScholarAgent
                llm, _ = initialize_llm(streaming=False)
                
                # 获取或创建学者资源
                resources = await ScholarAgent.get_or_create_resources(scholar_open_id)
                
                # 创建新的Agent实例，使用标准Agent框架
                agent = await ScholarAgent.create(
                    scholar_open_id=scholar_open_id,
                    custom_llm=llm,
                    streaming=False  # 非流式模式
                )
                
                # 获取学者信息
                scholar_info = agent.get_scholar_info()
                
                profile = ScholarProfile(
                    scholar_open_id=scholar_open_id,
                    name=scholar_info.get("name", "未知学者"),
                    institution=scholar_info.get("institution", ""),
                    research_interests=scholar_info.get("research_keywords", [])
                )
                
                self.scholar_profiles[scholar_open_id] = profile
                self.agent_cache[scholar_open_id] = agent
                self.last_active[scholar_open_id] = time.time()
                
                logger.info(f"学者 {profile.display_name} 的Agent初始化完成")
                return agent
                
            except Exception as e:
                logger.error(f"创建学者 {scholar_open_id} 的Agent实例时出错: {e}")
                raise ValueError(f"无法创建学者Agent: {str(e)}")
    
    async def get_or_create_agent_with_llm(self, scholar_open_id: str, streaming_callback=None) -> ScholarAgent:
        """获取或创建使用指定LLM的Agent实例"""
        logger.info(f"为学者 {scholar_open_id} 创建带自定义LLM的Agent实例")
        
        # 尝试从流式代理池获取实例
        async with self.streaming_agent_lock:
            # 初始化学者的流式代理池（如果不存在）
            if scholar_open_id not in self.streaming_agent_pools:
                self.streaming_agent_pools[scholar_open_id] = []
            
            try:
                # 获取或创建共享资源
                resources = await ScholarAgent.get_or_create_resources(scholar_open_id)
                
                # 创建新的流式LLM - 仅当提供了回调时才使用
                stream_llm = None
                if streaming_callback:
                    stream_llm, _ = initialize_llm(
                        temperature=settings.LLM_TEMPERATURE,
                        streaming=True,
                        streaming_callback=streaming_callback
                    )
                
                # 首先尝试重用现有代理
                if len(self.streaming_agent_pools[scholar_open_id]) > 0:
                    # 从池中获取一个代理
                    agent = self.streaming_agent_pools[scholar_open_id].pop(0)
                    # 只有当提供了回调时才更新LLM
                    if stream_llm:
                        agent.update_llm(stream_llm)
                    logger.info(f"重用学者 {scholar_open_id} 的流式代理")
                else:
                    # 如果没有可重用的代理，创建新的
                    agent = await ScholarAgent.create(
                        scholar_open_id=scholar_open_id,
                        custom_llm=stream_llm,  # 可能为None
                        streaming=True  # 始终标记为流式
                    )
                    logger.info(f"为学者 {scholar_open_id} 创建新的流式代理")
                
                self.last_active[scholar_open_id] = time.time()
                return agent
                
            except Exception as e:
                logger.error(f"创建学者 {scholar_open_id} 的自定义LLM Agent实例时出错: {e}")
                raise ValueError(f"无法创建学者Agent: {str(e)}")
    
    async def release_streaming_agent(self, scholar_open_id: str, agent: ScholarAgent):
        """释放流式代理回池中"""
        async with self.streaming_agent_lock:
            # 确保池存在
            if scholar_open_id not in self.streaming_agent_pools:
                self.streaming_agent_pools[scholar_open_id] = []
            
            # 如果池未满，将代理放回池中
            if len(self.streaming_agent_pools[scholar_open_id]) < self.MAX_STREAMING_AGENTS_PER_SCHOLAR:
                self.streaming_agent_pools[scholar_open_id].append(agent)
                logger.debug(f"将流式代理释放回池中: 学者={scholar_open_id}, 池大小={len(self.streaming_agent_pools[scholar_open_id])}")
            else:
                # 池已满，丢弃代理（它会被垃圾回收）
                logger.debug(f"流式代理池已满，丢弃代理: 学者={scholar_open_id}")
    
    async def get_or_create_session(self, user_id: str, scholar_open_id: str, session_id: Optional[str] = None) -> ConversationSession:
        """获取或创建会话"""
        async with self.session_lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if session.user_id == user_id and session.scholar_open_id == scholar_open_id:
                    session.last_active = time.time()
                    return session
                else:
                    logger.warning(f"会话ID {session_id} 与用户 {user_id} 和学者 {scholar_open_id} 不匹配")
            
            session = ConversationSession(
                user_id=user_id,
                scholar_open_id=scholar_open_id
            )
            
            self.sessions[session.session_id] = session
            
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session.session_id)
            
            if scholar_open_id not in self.scholar_sessions:
                self.scholar_sessions[scholar_open_id] = []
            self.scholar_sessions[scholar_open_id].append(session.session_id)
            
            logger.info(f"为用户 {user_id} 和学者 {scholar_open_id} 创建了新会话 {session.session_id}")
            return session
    
    async def enqueue_request(self, 
                         user_id: str, 
                         scholar_open_id: str, 
                         query: str, 
                         session_id: Optional[str] = None,
                         streaming: bool = True, # 默认使用流式处理
                         callback: Optional[Callable] = None,
                         priority: int = 0) -> str:
        """将请求加入队列"""
        # 仅在没有提供会话ID时创建新会话
        if not session_id:
            session = await self.get_or_create_session(user_id, scholar_open_id)
            session_id = session.session_id
        else:
            # 确保会话存在
            if session_id not in self.sessions:
                logger.warning(f"提供的会话ID {session_id} 不存在，创建新会话")
                session = await self.get_or_create_session(user_id, scholar_open_id)
                session_id = session.session_id
        
        # 将用户消息添加到会话历史
        session = self.sessions[session_id]
        session.messages.append({"role": "user", "content": query})
        session.last_active = time.time()
        
        result_future = asyncio.Future()
        
        task = RequestTask(
            user_id=user_id,
            scholar_open_id=scholar_open_id,
            session_id=session_id,  # 使用确定的会话ID
            query=query,
            streaming=streaming,
            callback=callback,
            result_future=result_future,
            priority=priority
        )
        
        async with self.queue_lock:
            if scholar_open_id not in self.request_queues:
                self.request_queues[scholar_open_id] = deque()
            
            queue = self.request_queues[scholar_open_id]
            if priority > 0:
                inserted = False
                for i, existing_task in enumerate(queue):
                    if existing_task.priority > priority:
                        queue.insert(i, task)
                        inserted = True
                        break
                if not inserted:
                    queue.append(task)
            else:
                queue.append(task)
            
            self.pending_results[task.task_id] = result_future
            
            logger.info(f"任务 {task.task_id} 加入队列: 用户={user_id}, 学者={scholar_open_id}, "
                    f"会话={session_id}, 优先级={priority}")
        
        self.task_signal.set()
        return task.task_id
    
    async def wait_for_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """等待任务结果"""
        if task_id not in self.pending_results:
            raise KeyError(f"未找到任务 {task_id}")
        
        future = self.pending_results[task_id]
        
        try:
            result = await asyncio.wait_for(future, timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"等待任务 {task_id} 结果超时")
        finally:
            if task_id in self.pending_results:
                del self.pending_results[task_id]
    
    async def cancel_request(self, task_id: str) -> bool:
        """取消请求"""
        async with self.queue_lock:
            for scholar_open_id, queue in self.request_queues.items():
                for i, task in enumerate(list(queue)): # 迭代副本以安全删除
                    if task.task_id == task_id:
                        del queue[i]
                        if task_id in self.pending_results:
                            future = self.pending_results.pop(task_id)
                            if not future.done():
                                future.set_result({
                                    "status": "cancelled",
                                    "message": "任务已取消"
                                })
                        logger.info(f"已取消任务 {task_id}")
                        return True
            
            if task_id in self.active_requests:
                logger.warning(f"任务 {task_id} 已在处理中，无法取消")
                return False
            
            logger.warning(f"未找到任务 {task_id}，无法取消")
            return False
    
    async def _worker_loop(self):
        """工作循环，负责处理队列中的任务"""
        logger.info("启动Agent管理器工作循环")
        while not self.shutdown_flag:
            await self.task_signal.wait()
            self.task_signal.clear()
            
            processed = await self._process_queues()
            
            if not processed and not self.shutdown_flag:
                # 如果没有处理任何任务，检查是否还有任务在队列中
                async with self.queue_lock:
                    total_pending = sum(len(q) for q in self.request_queues.values())
                if total_pending > 0:
                    # 仍有任务但无法处理（可能因为并发限制），稍后重试
                    self.task_signal.set()
    
    async def _process_queues(self) -> bool:
        """处理所有队列中的任务"""
        processed = False
        async with self.queue_lock:
            scholar_open_ids = list(self.request_queues.keys())
        
        for scholar_open_id in scholar_open_ids:
            scholar_concurrency = self.scholar_concurrency.get(scholar_open_id, 0)
            if scholar_concurrency >= settings.MAX_SCHOLAR_CONCURRENCY:
                continue
            
            task = None
            async with self.queue_lock:
                queue = self.request_queues.get(scholar_open_id)
                if queue:
                    task = queue.popleft()
            
            if task:
                user_concurrency = self.user_concurrency.get(task.user_id, 0)
                if user_concurrency >= settings.MAX_USER_CONCURRENCY:
                    logger.debug(f"用户 {task.user_id} 已达到并发限制，任务 {task.task_id} 放回队列。")
                    async with self.queue_lock:
                        self.request_queues[scholar_open_id].appendleft(task)
                    continue
                
                self.scholar_concurrency[scholar_open_id] = scholar_concurrency + 1
                self.user_concurrency[task.user_id] = user_concurrency + 1
                self.active_requests.add(task.task_id)
                
                asyncio.create_task(self._process_task(task, scholar_open_id))
                processed = True
        
        return processed
    
    async def _process_task(self, task: RequestTask, scholar_open_id: str):
        """处理单个任务 - 更新为使用新的ScholarAgent标准ReAct框架"""
        logger.info(f"开始处理任务 {task.task_id}: 用户={task.user_id}, 学者={scholar_open_id}")
        
        result = {"status": "error", "message": "未知错误", "task_id": task.task_id}
        streaming_agent = None  # 用于记录流式Agent以便后续释放回池
        
        try:
            session = self.sessions.get(task.session_id)
            if not session:
                raise ValueError(f"未找到会话 {task.session_id}")

            # 直接从会话中获取历史记录
            messages_history = session.messages
            
            # 处理有回调的流式请求
            if task.callback:
                # 获取流式处理Agent - 创建时已包含流式回调
                stream_callback = task.callback  # 保存原始回调
                
                # 创建自定义回调以处理流式结果
                class AgentResponseCallback:
                    def __init__(self, original_callback):
                        self.original_callback = original_callback
                        
                    async def __call__(self, token):
                        await self.original_callback(token)
                
                # 获取流式Agent
                streaming_agent = await self.get_or_create_agent_with_llm(
                    scholar_open_id, streaming_callback=None
                )
                
                # 收集完整响应
                response_text = ""
                
                try:
                    # 使用新的流式ask_stream方法
                    async for token in streaming_agent.ask_stream(task.query, history=messages_history):
                        # 处理完成标志
                        if token == "[DONE]":
                            await stream_callback("[DONE]")
                            break
                        
                        # 发送token给原始回调
                        await stream_callback(token)
                        response_text += token
                    
                except Exception as e:
                    logger.error(f"流式生成过程中出错: {e}", exc_info=True)
                    
                    # 检查是否已经生成了足够内容
                    if len(response_text) > 100:
                        logger.info("虽然出错，但已生成足够内容，不发送错误消息")
                        await stream_callback("[DONE]")
                    else:
                        # 内容不足，发送错误消息
                        await stream_callback(f"生成回答时出错: {str(e)}")
                        await stream_callback("[DONE]")
                
                # 将助手的完整响应添加到历史记录
                session.messages.append({"role": "assistant", "content": response_text})
                
                result = {
                    "status": "success", 
                    "message": "流式响应完成", 
                    "task_id": task.task_id,
                    "session_id": session.session_id, 
                    "response": response_text,
                    "tokens_count": len(response_text)  # 简化为响应长度
                }
            
            # 处理无回调的非流式请求
            else:
                # 使用非流式Agent
                agent = await self.get_or_create_agent(scholar_open_id)
                
                # 使用新的ask方法获取完整响应
                response_text = await agent.ask(task.query, history=messages_history)
                
                # 将助手的响应添加到历史记录
                session.messages.append({"role": "assistant", "content": response_text})
                
                result = {
                    "status": "success", 
                    "task_id": task.task_id, 
                    "session_id": session.session_id,
                    "response": response_text
                }
            
            session.last_active = time.time()
            
        except Exception as e:
            logger.error(f"处理任务 {task.task_id} 时出错: {e}", exc_info=True)
            result = {"status": "error", "message": f"处理请求时出错: {str(e)}", "task_id": task.task_id}
        
        finally:
            try:
                # 释放流式Agent回池中（如果使用了流式Agent）
                if streaming_agent:
                    await self.release_streaming_agent(scholar_open_id, streaming_agent)
                
                if task.task_id in self.pending_results:
                    future = self.pending_results.pop(task.task_id)
                    if not future.done():
                        future.set_result(result)
                
                self.scholar_concurrency[scholar_open_id] -= 1
                self.user_concurrency[task.user_id] -= 1
                self.active_requests.remove(task.task_id)
                
                logger.info(f"完成任务 {task.task_id} 处理")
                
                # 触发工作循环以检查是否有更多任务
                self.task_signal.set()
                
            except Exception as e:
                logger.error(f"完成任务 {task.task_id} 后清理时出错: {e}", exc_info=True)
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        if session_id not in self.sessions:
            raise KeyError(f"未找到会话 {session_id}")
        
        # 直接从会话对象返回消息
        return self.sessions[session_id].messages
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有会话"""
        if user_id not in self.user_sessions:
            return []
        
        sessions_info = []
        for session_id in self.user_sessions[user_id]:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                scholar_profile = self.scholar_profiles.get(session.scholar_open_id)
                
                sessions_info.append({
                    "session_id": session.session_id,
                    "scholar_open_id": session.scholar_open_id,
                    "scholar_name": scholar_profile.display_name if scholar_profile else "未知学者",
                    "created_at": session.created_at,
                    "last_active": session.last_active,
                    "message_count": len(session.messages) # 直接从会话获取消息数量
                })
        
        return sorted(sessions_info, key=lambda x: x["last_active"], reverse=True)
    
    async def clean_cache(self, max_idle_time: Optional[float] = None, clean_all: bool = False) -> Dict[str, int]:
        """清理缓存实例和过期会话"""
        agent_timeout = max_idle_time or settings.AGENT_INACTIVE_TIMEOUT
        session_timeout = max_idle_time or settings.SESSION_INACTIVE_TIMEOUT
        
        async with self.clean_lock:
            stats = {"agents_cleaned": 0, "sessions_cleaned": 0, "streaming_agents_cleaned": 0}
            
            if clean_all:
                stats["agents_cleaned"] = len(self.agent_cache)
                self.agent_cache.clear()
                self.last_active.clear()
                self.scholar_concurrency.clear()
                
                # 清理流式Agent池
                async with self.streaming_agent_lock:
                    for scholar_id, agents in self.streaming_agent_pools.items():
                        stats["streaming_agents_cleaned"] += len(agents)
                    self.streaming_agent_pools.clear()
                
                stats["sessions_cleaned"] = len(self.sessions)
                self.sessions.clear()
                self.user_sessions.clear()
                self.scholar_sessions.clear()
                
                logger.info(f"已清理所有缓存: {stats['agents_cleaned']} 个Agent, "
                           f"{stats['streaming_agents_cleaned']} 个流式Agent, "
                           f"{stats['sessions_cleaned']} 个会话")
                return stats
            
            current_time = time.time()
            
            # 清理非流式Agent
            agents_to_remove = [
                sid for sid, last_active in self.last_active.items()
                if current_time - last_active > agent_timeout and self.scholar_concurrency.get(sid, 0) == 0
            ]
            for sid in agents_to_remove:
                del self.agent_cache[sid]
                del self.last_active[sid]
                if sid in self.scholar_concurrency: del self.scholar_concurrency[sid]
            stats["agents_cleaned"] = len(agents_to_remove)
            
            # 清理流式Agent池
            async with self.streaming_agent_lock:
                for scholar_id in list(self.streaming_agent_pools.keys()):
                    # 如果学者最后活跃时间超过超时时间，清理其流式Agent池
                    if scholar_id in self.last_active and current_time - self.last_active[scholar_id] > agent_timeout:
                        stats["streaming_agents_cleaned"] += len(self.streaming_agent_pools[scholar_id])
                        del self.streaming_agent_pools[scholar_id]

            # 清理过期会话
            sessions_to_remove = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_active > session_timeout
            ]
            for sid in sessions_to_remove:
                session = self.sessions.pop(sid)
                if session.user_id in self.user_sessions and sid in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].remove(sid)
                if session.scholar_open_id in self.scholar_sessions and sid in self.scholar_sessions[session.scholar_open_id]:
                    self.scholar_sessions[session.scholar_open_id].remove(sid)
            stats["sessions_cleaned"] = len(sessions_to_remove)

            if stats["agents_cleaned"] > 0 or stats["sessions_cleaned"] > 0 or stats["streaming_agents_cleaned"] > 0:
                logger.info(f"缓存清理完成: 清理了 {stats['agents_cleaned']} 个Agent, "
                           f"{stats['streaming_agents_cleaned']} 个流式Agent, "
                           f"{stats['sessions_cleaned']} 个会话")
            
            return stats

    def get_stats(self) -> dict:
        """获取Agent管理器状态"""
        stats = {
            "agents": {
                "total": len(self.agent_cache),
                "streaming_pools": sum(len(pool) for pool in self.streaming_agent_pools.values()),
                "details": []
            },
            "sessions": {
                "total": len(self.sessions),
                "active": sum(1 for s in self.sessions.values() if time.time() - s.last_active < settings.SESSION_ACTIVE_THRESHOLD),
                "users": len(self.user_sessions)
            },
            "queues": {
                "total_pending": sum(len(q) for q in self.request_queues.values()),
                "active_requests": len(self.active_requests),
                "details": []
            }
        }
        
        # 添加Agent详情
        current_time = time.time()
        for scholar_open_id, last_active in self.last_active.items():
            idle_time = current_time - last_active
            profile = self.scholar_profiles.get(scholar_open_id)
            streaming_count = len(self.streaming_agent_pools.get(scholar_open_id, []))
            
            stats["agents"]["details"].append({
                "scholar_open_id": scholar_open_id,
                "name": profile.display_name if profile else "未知学者",
                "idle_time": idle_time,
                "idle_minutes": round(idle_time / 60, 1),
                "concurrent_requests": self.scholar_concurrency.get(scholar_open_id, 0),
                "streaming_agents": streaming_count
            })
        
        # 添加队列详情
        for scholar_open_id, queue in self.request_queues.items():
            profile = self.scholar_profiles.get(scholar_open_id)
            stats["queues"]["details"].append({
                "scholar_open_id": scholar_open_id,
                "name": profile.display_name if profile else "未知学者",
                "queue_length": len(queue),
                "concurrent_requests": self.scholar_concurrency.get(scholar_open_id, 0)
            })
        
        return stats

    async def shutdown(self):
        """关闭Agent管理器，停止所有任务"""
        logger.info("正在关闭Agent管理器...")
        self.shutdown_flag = True
        
        async with self.queue_lock:
            for queue in self.request_queues.values():
                while queue:
                    task = queue.popleft()
                    if task.task_id in self.pending_results:
                        future = self.pending_results.pop(task.task_id)
                        if not future.done():
                            future.set_result({"status": "cancelled", "message": "系统关闭，任务已取消"})
        
        if self.worker_task and not self.worker_task.done():
            self.task_signal.set()
            try:
                await asyncio.wait_for(self.worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("工作循环未能在超时时间内结束，强制取消")
                self.worker_task.cancel()
        
        logger.info("Agent管理器已关闭")

# 创建全局管理器实例
agent_manager = AgentManager()