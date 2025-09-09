"""
Scholar Agent Implementation - 基于LangChain标准Agent框架和ReAct模式
"""
import logging
import asyncio
import time
import re
from typing import Dict, Any, List, Optional, Tuple

# LangChain 导入
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 应用导入
from app.core.llm import initialize_llm, CustomStreamingCallbackHandler
from app.prompt.scholar_config import ScholarConfig
from app.tools.scholar_tools import PaperRetrievalTool, ScholarInfoTool
from config.settings import settings

logger = logging.getLogger("app")

class ScholarResources:
    """持有学者相关的共享资源，可以被多个Agent实例共用"""
    
    def __init__(self, scholar_open_id: str):
        """初始化学者资源"""
        self.scholar_open_id = scholar_open_id
        
        # 初始化学者配置
        self.config = ScholarConfig(scholar_open_id)
        
        # 预加载学者信息
        self.scholar_info = self.config.get_scene_info()
        
        # 获取检索器
        self.retriever = self.config.get_retriever()
        
        logger.info(f"Scholar resources initialized: scholar_open_id={scholar_open_id}")
    
    def get_system_instruction(self) -> str:
        """获取系统指令"""
        return self.config.get_system_instruction()
    
    def get_scholar_info(self) -> Dict[str, Any]:
        """获取学者基本信息"""
        return {
            "scholar_open_id": self.scholar_open_id,
            "name": self.config.get("name", ""),
            "zh_name": self.config.get("zh_name", ""),
            "en_name": self.config.get("en_name", ""),
            "institution": self.config.get("institution", ""),
            "brief": self.config.get("brief", ""),
            "research_keywords": self.config.get("research_keywords", []),
            "total_papers": self.config.get("total_count", 0)
        }


class ScholarAgent:
    """Scholar Digital Persona Agent - 使用LangChain标准Agent框架和ReAct模式"""
    
    # 全局资源缓存
    _resources_cache: Dict[str, ScholarResources] = {}
    _resources_lock = asyncio.Lock()
    
    # ReAct提示词模板
    REACT_TEMPLATE = """
你是{name}教授，一位在{research_fields}领域的专家学者。

{system_instruction}

当回答问题时，请遵循以下流程：
1. 如果需要特定研究信息，请使用可用工具。
2. 考虑是否需要检索论文或使用你已有的知识。
3. 仅在需要专业研究信息时使用工具。
4. 对于一般性问题，直接回答。
5. 确保最终答案与思考过程分开。

工具列表：
------
{tools}

工具使用格式：
------
如需使用工具，请按照以下格式：
思考: 我需要思考回答这个问题需要什么信息
行动: 要执行的动作，应该是[{tool_names}]之一
输入: 动作的输入参数
观察: 工具返回的结果
当你有了回答，请按照以下格式：
思考: 我知道这个问题的答案
回答: [你的最终回答，不包含任何推理、思考过程或工具使用信息]
开始！

问题: {input}
{agent_scratchpad}
"""
    
    @classmethod
    async def get_or_create_resources(cls, scholar_open_id: str) -> ScholarResources:
        """获取或创建学者资源"""
        async with cls._resources_lock:
            if scholar_open_id not in cls._resources_cache:
                cls._resources_cache[scholar_open_id] = ScholarResources(scholar_open_id)
            return cls._resources_cache[scholar_open_id]
    
    def __init__(self, scholar_open_id: str, custom_llm=None, resources: Optional[ScholarResources] = None, streaming: bool = False):
        """
        初始化学者Agent
        
        Args:
            scholar_open_id: 学者ID
            custom_llm: 可选的自定义LLM
            resources: 可选的共享资源
            streaming: 是否为流式模式
        """
        self.scholar_open_id = scholar_open_id
        self.llm = custom_llm
        self.streaming = streaming
        
        # 如果没有提供自定义LLM，初始化一个非流式LLM
        if self.llm is None:
            self.llm, _ = initialize_llm(streaming=self.streaming)
        
        # 使用预加载资源或初始化同步创建资源
        if resources:
            self.resources = resources
        else:
            # 同步方式创建资源 - 异步环境中应使用get_or_create_resources
            self.resources = ScholarResources(scholar_open_id)
        
        # 从资源获取学者信息
        self.scholar_info = self.resources.scholar_info
        self.config = self.resources.config
        self.retriever = self.resources.retriever
        
        # 初始化工具
        self.paper_retrieval_tool = PaperRetrievalTool(self.retriever)
        self.scholar_info_tool = ScholarInfoTool(self.config)
        self.tools = [self.paper_retrieval_tool, self.scholar_info_tool]
        
        # 初始化记忆组件
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化Agent Executor
        self.agent_executor = self._create_agent_executor(streaming=self.streaming)
        
        # 创建输出解析器
        self.output_parser = self._create_output_parser()
        
        logger.info(f"Scholar Agent initialization complete: scholar_open_id={scholar_open_id}, name={self.config.get('name')}")
    
    def _create_output_parser(self):
        """创建结构化输出解析器"""
        response_schemas = [
            ResponseSchema(name="answer", description="The final answer to the user's question"),
            ResponseSchema(name="sources", description="Optional: Sources or references used for the answer")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def _create_agent_executor(self, streaming=False) -> AgentExecutor:
        """创建Agent执行器"""
        # 获取学者基本信息
        scholar_name = self.config.get("name", "")
        research_fields = ", ".join(self.config.get("research_keywords", [])[:5])
        system_instruction = self.resources.get_system_instruction()
        
        # 初始化Agent
        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            max_iterations=3  # 限制最大迭代次数
        )
        
        # 覆盖默认提示词
        agent_executor.agent.llm_chain.prompt = ChatPromptTemplate.from_template(
            template=self.REACT_TEMPLATE,
            partial_variables={
                "name": scholar_name,
                "research_fields": research_fields,
                "system_instruction": system_instruction,
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools]),
            }
        )
        
        return agent_executor
    
    @classmethod
    async def create(cls, scholar_open_id: str, custom_llm=None, streaming=True, streaming_callback=None):
        """
        异步创建学者Agent
        
        Args:
            scholar_open_id: 学者ID
            custom_llm: 可选的自定义LLM
            streaming: 是否使用流式LLM
            streaming_callback: 流式回调处理器
        
        Returns:
            ScholarAgent实例
        """
        # 获取或创建共享资源
        resources = await cls.get_or_create_resources(scholar_open_id)
        
        # 如果没有提供自定义LLM且需要流式处理
        if custom_llm is None and streaming:
            custom_llm, _ = initialize_llm(streaming=True, streaming_callback=streaming_callback)
        
        # 创建Agent实例
        return cls(scholar_open_id, custom_llm=custom_llm, resources=resources, streaming=streaming)
    
    def update_llm(self, new_llm):
        """
        更新LLM
        
        Args:
            new_llm: 新的LLM实例
        """
        self.llm = new_llm
        # 更新Agent执行器
        self.agent_executor = self._create_agent_executor(streaming=self.streaming)
    
    def _format_chat_history(self, history: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        格式化对话历史为LangChain Agent所需的元组列表
        
        Args:
            history: 对话历史
            
        Returns:
            格式化的对话历史列表 [(human_message, ai_message), ...]
        """
        if not history:
            return []
            
        formatted_history = []
        human_msg = None
        
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                human_msg = content
            elif role == "assistant" and human_msg:
                # 形成一对对话并添加到历史中
                formatted_history.append((human_msg, content))
                human_msg = None
        
        return formatted_history
    
    async def ask(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        处理问题 - 使用标准Agent执行流程
        
        Args:
            query: 用户问题
            history: 可选的对话历史
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"处理学者问题: scholar_open_id={self.scholar_open_id}, query='{query}'")
            
            # 格式化历史对话
            formatted_history = self._format_chat_history(history) if history else []
            
            # 清空旧的聊天记忆
            self.memory.clear()
            
            # 如果存在历史对话，添加到内存中
            for human_msg, ai_msg in formatted_history[-5:]:  # 只使用最近的5个交互
                self.memory.chat_memory.add_user_message(human_msg)
                self.memory.chat_memory.add_ai_message(ai_msg)
            
            # 执行Agent
            response = await self.agent_executor.arun(
                input=query
            )
            
            # 提取最终答案
            final_answer = self._extract_answer(response)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"处理问题时出错: {e}", exc_info=True)
            return f"抱歉，在处理您的问题时遇到了错误: {str(e)}"
    
    def _extract_answer(self, text: str) -> str:
        """提取最终答案，去除思考过程"""
        # 如果包含"回答:"标记
        if "回答:" in text:
            parts = text.split("回答:")
            return parts[1].strip()
        
        # 如果包含英文"Answer:"标记
        if "Answer:" in text:
            parts = text.split("Answer:")
            return parts[1].strip()
        
        # 移除所有思考和行动部分
        cleaned = re.sub(r'(思考:|行动:|输入:|观察:|Thought:|Action:|Action Input:|Observation:).*?(\n|$)', '', 
                         text, flags=re.DOTALL)
        return cleaned.strip()
    
    async def ask_stream(self, query: str, history: Optional[List[Dict[str, Any]]] = None):
        """
        流式处理问题 - 使用Agent执行流程并提供流式输出
        
        Args:
            query: 用户问题
            history: 可选的对话历史
        """
        try:
            logger.info(f"流式处理学者问题: scholar_open_id={self.scholar_open_id}, query='{query}'")
            start_time = time.time()
            
            # 创建队列用于传递tokens
            queue = asyncio.Queue()
            
            # 创建带过滤功能的流式回调
            class AgentStreamingCallback(CustomStreamingCallbackHandler):
                def __init__(self, queue):
                    super().__init__()
                    self.queue = queue
                    self.in_final_answer = False
                    self.buffer = ""
                    self.answer_pattern = re.compile(r'(回答:|Answer:)', re.IGNORECASE)
                    
                async def on_llm_new_token(self, token: str, **kwargs) -> None:
                    # 更新缓冲区
                    self.buffer += token
                    
                    # 检查是否进入最终答案部分
                    if not self.in_final_answer and self.answer_pattern.search(self.buffer):
                        self.in_final_answer = True
                        # 找到Answer:后的部分
                        match = self.answer_pattern.search(self.buffer)
                        answer_start = match.end()
                        answer_part = self.buffer[answer_start:]
                        # 发送Answer:后的部分
                        await self.queue.put(answer_part)
                        self.buffer = ""
                        return
                    
                    # 如果已经在最终答案部分，直接发送token
                    if self.in_final_answer:
                        await self.queue.put(token)
                
                async def on_llm_end(self, response, **kwargs) -> None:
                    # 如果结束时还有缓冲内容但未进入最终答案
                    if not self.in_final_answer and self.buffer:
                        # 尝试找出可能的答案部分
                        match = self.answer_pattern.search(self.buffer)
                        if match:
                            answer_start = match.end()
                            answer_part = self.buffer[answer_start:]
                            await self.queue.put(answer_part)
                        else:
                            # 如果没有找到明确的答案标记，尝试清理内容并发送
                            cleaned = re.sub(r'(思考:|行动:|输入:|观察:|Thought:|Action:|Action Input:|Observation:).*?(\n|$)', 
                                          '', self.buffer, flags=re.DOTALL)
                            await self.queue.put(cleaned.strip())
                    
                    await self.queue.put("[DONE]")
            
            # 创建带流式回调的LLM
            streaming_callback = AgentStreamingCallback(queue)
            stream_llm, _ = initialize_llm(streaming=True, streaming_callback=streaming_callback)
            
            # 更新Agent执行器使用流式LLM
            self.update_llm(stream_llm)
            
            # 格式化历史对话
            formatted_history = self._format_chat_history(history) if history else []
            
            # 清空旧的聊天记忆
            self.memory.clear()
            
            # 如果存在历史对话，添加到内存中
            for human_msg, ai_msg in formatted_history[-5:]:  # 只使用最近的5个交互
                self.memory.chat_memory.add_user_message(human_msg)
                self.memory.chat_memory.add_ai_message(ai_msg)
            
            # 异步执行Agent
            task = asyncio.create_task(self.agent_executor.arun(input=query))
            
            # 读取并yield结果
            while True:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    if token == "[DONE]":
                        yield "[DONE]"
                        break
                    
                    yield token
                    
                except asyncio.TimeoutError:
                    # 如果任务已完成但队列为空，可以退出
                    if task.done() and queue.empty():
                        yield "[DONE]"
                        break
                    
                    # 如果太长时间没有收到新token，可能需要结束
                    if time.time() - start_time > 60.0:  # 总超时1分钟
                        logger.warning("生成超时")
                        if not task.done():
                            task.cancel()
                        yield "[DONE]"
                        break
                    
                    # 继续等待
                    continue
                
        except Exception as e:
            logger.error(f"流式处理问题时出错: {e}", exc_info=True)
            yield f"处理您的问题时出错: {str(e)}"
            yield "[DONE]"
    
    def get_scholar_info(self) -> Dict[str, Any]:
        """
        获取学者基本信息
        
        Returns:
            学者信息字典
        """
        return self.resources.get_scholar_info()