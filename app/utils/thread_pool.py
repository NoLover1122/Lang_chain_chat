"""
线程池管理器 - 实现主从线程间的数据通道

这个模块实现了基于ThreadPoolExecutor + Queue的线程池方案，
用于处理LLM响应的流式数据传输。

主要特性：
1. 主线程提交任务到线程池，立即获得通信桥梁
2. 工作线程执行LLM调用，通过Queue传输数据 
3. 主线程异步读取Queue数据，实现流式处理
4. 支持多并发请求的独立上下文管理
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass

from src.utils.logging import logger


class IncrementalContentProcessor:
    """专门处理增量内容计算的处理器"""
    
    def __init__(self):
        self.last_content = ""
        self.total_chunks = 0
    
    def process_response_list(self, response_list) -> str:
        """处理单个response_list，返回增量内容
        
        Args:
            response_list: QwenAgent返回的响应列表
            
        Returns:
            str: 增量内容，如果没有新内容则返回空字符串
        """
        for response in response_list:
            if self._is_valid_content(response):
                content = response["content"].strip()
                
                # 计算增量内容
                incremental = content[len(self.last_content):]
                logger.debug(f"内容处理: 全量长度={len(content)}, 上次长度={len(self.last_content)}, 增量长度={len(incremental)}")
                
                # 更新状态
                self.last_content = content
                
                if incremental:
                    self.total_chunks += 1
                    logger.debug(f"找到增量内容，当前总块数: {self.total_chunks}")
                    return incremental
        
        return ""
    
    def _is_valid_content(self, response) -> bool:
        """判断是否是有效的内容响应
        
        Args:
            response: 单个响应对象
            
        Returns:
            bool: 是否是有效的内容响应
        """
        # 检查基本结构
        if not ("role" in response and response["role"] == "assistant"):
            return False
        if not "content" in response:
            return False
        
        content = response["content"]
        if not isinstance(content, str):
            return False
        
        # 过滤空内容
        if not content.strip():
            return False
            
        # 过滤工具返回和特殊内容
        if any(keyword in content for keyword in [
            "status", "category", "confidence", "reason", 
            "identity", "identity_reason"
        ]):
            return False
            
        return True
    
    def get_stats(self) -> dict:
        """获取处理统计信息"""
        return {
            "total_chunks": self.total_chunks,
            "last_content_length": len(self.last_content)
        }


class EventFormatter:
    """专门处理事件格式化的格式化器"""
    
    @staticmethod
    def content_event(content: str) -> dict:
        """生成内容事件
        
        Args:
            content: 增量内容
            
        Returns:
            dict: 格式化的内容事件
        """
        return {
            "event": "content",
            "data": {
                "content": content
            }
        }
    
    @staticmethod 
    def error_event(error: str) -> dict:
        """生成错误事件
        
        Args:
            error: 错误信息
            
        Returns:
            dict: 格式化的错误事件
        """
        return {
            "event": "error",
            "data": {
                "error": error,
                "status": "error"
            }
        }


@dataclass
class StreamContext:
    """流式数据传输上下文"""
    context_id: str
    data_queue: Queue
    finished_event: threading.Event
    error: Optional[Exception] = None


class ThreadPoolManager:
    """线程池管理器 - 主从线程数据通道实现"""
    
    def __init__(self, max_workers: int = 7):
        """初始化线程池管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_contexts: Dict[str, StreamContext] = {}
        self._lock = threading.Lock()
        
        logger.info(f"线程池管理器初始化完成，最大工作线程数: {max_workers}")
    
    def submit_llm_task(self, context_id: str, assistant, messages) -> StreamContext:
        """提交LLM任务到线程池
        
        Args:
            context_id: 唯一上下文标识
            assistant: QwenAgent Assistant实例
            messages: 消息列表
            
        Returns:
            StreamContext: 通信桥梁，用于数据传输
        """
        # 创建流式上下文
        context = StreamContext(
            context_id=context_id,
            data_queue=Queue(),
            finished_event=threading.Event()
        )
        
        with self._lock:
            self.active_contexts[context_id] = context
        
        # 提交任务到线程池
        future = self.executor.submit(self._execute_llm_task, context, assistant, messages)
        
        logger.info(f"任务提交到线程池 [context_id={context_id}]")
        return context
    
    def _execute_llm_task(self, context: StreamContext, assistant, messages):
        """在工作线程中执行LLM任务
        
        Args:
            context: 流式上下文
            assistant: QwenAgent Assistant实例  
            messages: 消息列表
        """
        context_id = context.context_id
        logger.info(f"工作线程开始执行LLM任务 [context_id={context_id}]")
        
        try:
            # 调用QwenAgent的run方法获取生成器
            response_generator = assistant.run(messages=messages)
            
            # 处理流式响应
            response_list = []
            for response_data in response_generator:
                # 将数据放入队列
                context.data_queue.put(response_data)
                response_list.append(response_data)
                
                # 记录调试信息
                if hasattr(response_data, 'get'):
                    content = response_data.get('content', '')
                    if content:
                        logger.debug(f"工作线程放入数据 [context_id={context_id}], 内容长度: {len(content)}")
            
            # 任务完成，发送结束信号
            context.data_queue.put(None)  # 结束标记
            context.finished_event.set()
            
            logger.info(f"工作线程任务完成 [context_id={context_id}], 总数据块: {len(response_list)}")
            
        except Exception as e:
            logger.error(f"工作线程执行失败 [context_id={context_id}]: {str(e)}", exc_info=True)
            context.error = e
            context.data_queue.put(None)  # 确保主线程能够退出
            context.finished_event.set()
    
    def cleanup_context(self, context_id: str):
        """清理指定的上下文
        
        Args:
            context_id: 上下文ID
        """
        with self._lock:
            if context_id in self.active_contexts:
                del self.active_contexts[context_id]
                logger.debug(f"清理上下文 [context_id={context_id}]")
    
    def shutdown(self):
        """关闭线程池管理器"""
        logger.info("正在关闭线程池管理器...")
        self.executor.shutdown(wait=True)
        
        with self._lock:
            self.active_contexts.clear()
        
        logger.info("线程池管理器已关闭")


async def stream_from_thread_context(
    stream_context: StreamContext,
    timing_obj: Any
) -> AsyncGenerator[Dict[str, Any], None]:
    """从线程上下文异步流式读取数据并处理
    
    这个函数职责单一：
    1. 从队列获取数据
    2. 处理增量内容  
    3. 格式化事件
    4. 返回给用户
    
    Args:
        stream_context: 流式上下文
        timing_obj: 计时对象
        
    Yields:
        Dict[str, Any]: 格式化的事件数据
    """
    context_id = stream_context.context_id
    logger.info(f"开始异步流式读取 [context_id={context_id}]")
    
    # 创建处理器和格式化器
    processor = IncrementalContentProcessor()
    formatter = EventFormatter()
    
    processed_chunks = 0
    
    try:
        while True:
            # 检查工作线程是否出错
            if stream_context.error:
                logger.error(f"工作线程出错 [context_id={context_id}]: {stream_context.error}")
                yield formatter.error_event(f"LLM调用失败: {str(stream_context.error)}")
                raise stream_context.error
            
            try:
                # 非阻塞地尝试获取数据
                data = stream_context.data_queue.get_nowait()
                
                if data is None:  # 结束标记
                    logger.info(f"收到结束标记 [context_id={context_id}]")
                    break
                
                # 更新计时统计
                chunk_wait = timing_obj.update_chunk_timing()
                if processed_chunks > 0:
                    logger.debug(f"等待chunk耗时: {chunk_wait:.6f}秒")
                
                # 处理增量内容
                incremental_content = processor.process_response_list(data)
                
                # 如果有增量内容，格式化并返回
                if incremental_content:
                    processed_chunks += 1
                    logger.debug(f"输出增量内容 [context_id={context_id}]: '{incremental_content}'")
                    yield formatter.content_event(incremental_content)
                
            except Empty:
                # 队列为空，等待一小段时间
                await asyncio.sleep(0.01)  # 10ms
                
                # 如果任务已完成且队列为空，退出循环
                if stream_context.finished_event.is_set():
                    # 再次检查队列，防止遗漏最后的数据
                    try:
                        data = stream_context.data_queue.get_nowait()
                        if data is None:
                            break
                        
                        incremental_content = processor.process_response_list(data)
                        if incremental_content:
                            processed_chunks += 1
                            yield formatter.content_event(incremental_content)
                    except Empty:
                        break
                
                continue
        
        # 输出统计信息
        stats = processor.get_stats()
        logger.info(f"异步流式读取完成 [context_id={context_id}], "
                   f"处理数据块: {processed_chunks}, 总字符数: {stats['last_content_length']}")
        
    except Exception as e:
        logger.error(f"异步流式读取出错 [context_id={context_id}]: {str(e)}", exc_info=True)
        yield formatter.error_event(f"流式处理失败: {str(e)}")
        raise