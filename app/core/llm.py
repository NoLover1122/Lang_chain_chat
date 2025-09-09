import logging
import asyncio
from typing import Optional, List, Union, Dict, Any, Callable
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from config.settings import settings

logger = logging.getLogger("app")

class CustomStreamingCallbackHandler(BaseCallbackHandler):
    """优化的异步流式输出回调处理器"""
    
    def __init__(self, 
                 queue=None, 
                 on_llm_new_token: Optional[Callable[[str], None]] = None):
        """初始化回调处理器
        
        Args:
            queue: 可选的异步队列，用于传输生成的文本片段
            on_llm_new_token: 可选的回调函数，当有新token时调用
        """
        self.queue = queue
        self._on_llm_new_token = on_llm_new_token
        self.text = ""
        self.last_token = None
        
        # 使用一个专用的事件循环来处理回调
        self._loop = None
        self._background_task = None
        self._token_buffer = asyncio.Queue()
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """当新的文本片段生成时调用
        
        Args:
            token: 新生成的文本片段
        """
        # 去重逻辑 - 跳过重复的token
        if token == self.last_token:
            logger.debug(f"跳过重复token: {token}")
            return
        self.last_token = token
        
        # 更新文本
        self.text += token
        
        # 如果提供了队列，将新令牌发送到队列
        if self.queue:
            try:
                # 尝试同步放入队列
                if hasattr(self.queue, 'put_nowait'):
                    self.queue.put_nowait(token)
                else:
                    # 假设队列是可调用对象
                    self.queue(token)
            except Exception as e:
                logger.error(f"向队列发送token时出错: {e}")
        
        # 如果提供了回调函数并且需要异步处理
        if self._on_llm_new_token:
            try:
                # 确保事件循环已初始化
                if self._loop is None:
                    self._init_background_task()
                
                # 将token添加到处理队列
                asyncio.run_coroutine_threadsafe(
                    self._token_buffer.put(token), 
                    self._loop
                )
            except Exception as e:
                logger.error(f"处理token时出错: {e}")
        else:
            # 否则打印到控制台（调试用）
            print(token, end="", flush=True)
    
    def _init_background_task(self):
        """初始化后台任务和事件循环"""
        # 创建新的事件循环
        self._loop = asyncio.new_event_loop()
        
        # 在新线程中启动事件循环
        import threading
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            # 启动token处理任务
            self._background_task = asyncio.ensure_future(
                self._process_tokens(), 
                loop=loop
            )
            loop.run_forever()
        
        # 启动事件循环线程
        thread = threading.Thread(target=run_event_loop, args=(self._loop,), daemon=True)
        thread.start()
    
    async def _process_tokens(self):
        """处理队列中的tokens"""
        while True:
            try:
                # 获取下一个token
                token = await self._token_buffer.get()
                
                # 调用回调处理token
                if asyncio.iscoroutinefunction(self._on_llm_new_token):
                    await self._on_llm_new_token(token)
                else:
                    # 如果回调不是协程函数，同步调用
                    self._on_llm_new_token(token)
                
                # 标记任务完成
                self._token_buffer.task_done()
            except Exception as e:
                logger.error(f"处理token队列时出错: {e}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM生成完成时调用
        
        Args:
            response: LLM生成的结果
        """
        # 重置去重状态
        self.last_token = None
        
        # 如果提供了队列，发送结束信号
        if self.queue:
            try:
                if hasattr(self.queue, 'put_nowait'):
                    self.queue.put_nowait("[DONE]")
                else:
                    self.queue("[DONE]")
            except Exception as e:
                logger.error(f"向队列发送结束信号时出错: {e}")
        
        # 如果提供了回调函数，发送结束信号
        if self._on_llm_new_token and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._token_buffer.put("[DONE]"),
                    self._loop
                )
            except Exception as e:
                logger.error(f"发送结束信号时出错: {e}")
    
    def get_generated_text(self) -> str:
        """获取到目前为止生成的完整文本"""
        return self.text
    
    def __del__(self):
        """析构函数，确保事件循环被关闭"""
        try:
            if self._loop:
                # 停止事件循环
                asyncio.run_coroutine_threadsafe(
                    self._cleanup(), 
                    self._loop
                )
        except Exception as e:
            logger.error(f"关闭事件循环时出错: {e}")
    
    async def _cleanup(self):
        """清理资源"""
        try:
            # 停止事件循环
            self._loop.stop()
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")


def initialize_llm(
    temperature: float = 0.1, 
    streaming: bool = True,
    streaming_callback: Optional[BaseCallbackHandler] = None
):
    """
    初始化LLM模型
    
    Args:
        temperature: 控制输出随机性，0为确定性输出，1为最大随机性
        streaming: 是否启用流式输出
        streaming_callback: 自定义流式输出回调，如果为None则创建新的回调
        
    Returns:
        配置好的LLM模型实例和流回调(如果启用)的元组
    """
    try:
        logger.info(f"初始化LLM模型: {settings.LLM_DEFAULT_MODEL} (temperature={temperature}, streaming={streaming})")
        
        # 处理回调
        callback = None
        callbacks = []
        
        if streaming:
            # 使用提供的回调或创建新的
            callback = streaming_callback or CustomStreamingCallbackHandler()
            callbacks = [callback]
            logger.debug("已启用流式输出")
        
        # 配置API访问
        llm = ChatOpenAI(
            model=settings.LLM_DEFAULT_MODEL,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            request_timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_RETRY_COUNT
        )
        
        logger.info("LLM模型初始化成功")
        return llm, callback
    
    except Exception as e:
        logger.error(f"初始化LLM时出错: {e}")
        raise