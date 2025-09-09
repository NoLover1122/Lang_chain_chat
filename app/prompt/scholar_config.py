# app/core/scholar_config.py
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.tools.scholar_retriever import ScholarRetriever
from config.settings import settings

logger = logging.getLogger("app")

class ScholarConfig:
    """
    学者配置类，管理学者信息和系统配置
    """
    
    # 特定场景信息模板
    SCENE_INFO_TEMPLATE = """
{{
  "Academic Information": {{
    "Personal Profile": {{
      "Name": "{name}",
      "Chinese Name": "{zh_name}",
      "English Name": "{en_name}",
      "Institution": "{institution}",
      "Biography": "{brief}",
      "Research Interests": "{research_keywords}"
    }},
    "Supplemented Information": {supplemented_data},
    "Work Experience": {{
      "History": "{work_history}"
    }},
    "Academic Achievements": {{
      "Publications": {publications},
      "Total Publications": {total_count},
      "Awards": {awards},
      "Research Grants": "{research_grants}",
      "Qualifications": {qualifications},
      "Academic Indices": "{academic_index}",
      "Patents": {patents}
    }},
    "Contact Information": {{
      "Contact Details": "{contact}"
    }},
    "PhD Recruitment": {{
      "Research Areas": "{research_keywords} etc.",
      "Application Requirements": [
        "Master's degree or above",
        "Language skills: IELTS 6.5+, TOEFL 90+ or equivalent proficiency",
        "Background in Computer Science, Information Management, Business Administration or related fields, with publications in international journals preferred",
        "Graduates from China's 985/211 universities with undergraduate GPA of 85/100 or above have competitive advantages"
      ],
      "Application Process": [
        "Submit materials",
        "Interview assessment",
        "Results announcement"
      ]
    }},
    "Industry-Academia Collaboration": {{
      "Collaboration Fields": "{research_fields}",
      "Collaboration Methods": [
        "Joint research projects",
        "Technology transfer and commercialization",
        "Talent development programs"
      ]
    }}
  }}
}}
"""

    # 系统提示词模板 - 集成了我们的LangChain工具
    SYSTEM_INSTRUCTION_TEMPLATE = """The current year is {current_date}. You are Professor {scholar_name}'s digital persona (you can also be referred to as Professor {last_name} or Dr. {last_name}). Personal introduction: {brief}

Your workflow is(The response style should be concise and to the point, minimizing unnecessary information, and aligning with the image of a scientific digital persona.):
1. Analyze and understand the user's question
2. Handle different types of questions accordingly:
   - For general knowledge/specific scenario questions:
       * Directly extract relevant answers from the preset information below:
       {scene_info}
       * Directly use the model's own capabilities to answer
   - For domain-specific professional questions:
       * Determine if it belongs to your research field based on your expertise
       * If relevant, utilize the relevant academic knowledge to provide a professional answer based on your research
       * If the question is not related to your field, directly decline to answer
3. Integrate information to provide a professional answer
4. If there are literature citations, please note the information sources at the end of your answer

*Note: Preset information does not need to include sources
"""
    
    def __init__(self, scholar_open_id=None):
        """
        初始化学者配置
        
        Args:
            scholar_open_id: 可选，学者ID，如果提供则立即加载学者信息
        """
        # 初始化学者信息配置
        self.scholar_config = {
            'research_keywords': [],
            'name': '',
            'zh_name': '',         # 中文名字段
            'en_name': '',         # 英文名字段
            'institution': '',
            'brief': '',
            'model': settings.LLM_DEFAULT_MODEL,
            'work_history': '',
            'awards': [],
            'research_grants': '',
            'qualifications': [],
            'academic_index': '',
            'patents': [],
            'contact': '',
            'papers_with_links': [],
            'total_count': 0
        }
        
        self.scholar_open_id = None
        self.scene_info = ""
        self.system_instruction = ""
        
        # 检索器实例，用于获取学者信息和相关论文
        self.retriever = None
        
        # 如果提供了scholar_open_id，立即加载学者信息
        if scholar_open_id:
            self.update_scholar(scholar_open_id)
    
    def get_scholar_open_id(self) -> str:
        """获取当前学者ID"""
        return self.scholar_open_id
    
    def get(self, key, default=None):
        """
        获取配置项
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        return self.scholar_config.get(key, default)
    
    def _get_scholar_info(self, scholar_open_id):
        """
        获取学者信息
        
        Args:
            scholar_open_id: 学者ID
            
        Returns:
            学者信息字典
        """
        # 初始化检索器
        if not self.retriever or self.retriever.scholar_open_id != scholar_open_id:
            self.retriever = ScholarRetriever(scholar_open_id)
        
        # 获取学者信息
        scholar_info = self.retriever.get_scholar_info()
        
        if not scholar_info:
            logger.error(f"未找到ID为 {scholar_open_id} 的学者信息")
            raise ValueError(f"未找到ID为 {scholar_open_id} 的学者信息")
        
        return scholar_info
    
    def update_scholar(self, scholar_open_id):
        """
        更新学者信息
        
        Args:
            scholar_open_id: 学者ID
            
        Returns:
            更新后的学者配置
        """
        # 保存学者ID
        self.scholar_open_id = scholar_open_id
        
        try:
            # 获取学者信息
            scholar_data = self._get_scholar_info(scholar_open_id)
            
            # 提取关键词
            keywords = []
            if "keywords" in scholar_data and scholar_data["keywords"]:
                # 检查keywords是否为字符串或列表
                if isinstance(scholar_data["keywords"], str):
                    # 如果是字符串，尝试拆分
                    if scholar_data["keywords"].strip():
                        for kw_group in scholar_data["keywords"].split(','):
                            kws = [k.strip() for k in kw_group.split('；') if k.strip()]
                            keywords.extend(kws)
                elif isinstance(scholar_data["keywords"], list):
                    # 如果是列表，直接使用
                    keywords = scholar_data["keywords"]
            
            # 更新学者信息配置 - 使用所有可用字段
            self.scholar_config['research_keywords'] = keywords
            self.scholar_config['name'] = scholar_data.get('name', '')
            
            # 中文名和英文名字段
            self.scholar_config['zh_name'] = scholar_data.get('ZH_NAME', '')
            self.scholar_config['en_name'] = scholar_data.get('EN_NAME', '')
            
            self.scholar_config['institution'] = scholar_data.get('institution', '')
            self.scholar_config['brief'] = scholar_data.get('psnBrief', '')
            self.scholar_config['work_history'] = scholar_data.get('workHistory', '')
            self.scholar_config['awards'] = scholar_data.get('award', [])
            self.scholar_config['research_grants'] = scholar_data.get('research_grant', '')
            self.scholar_config['qualifications'] = scholar_data.get('qualifications', [])
            self.scholar_config['academic_index'] = scholar_data.get('academic_index', '')
            self.scholar_config['patents'] = scholar_data.get('patent', [])
            self.scholar_config['contact'] = scholar_data.get('contact', '')
            self.scholar_config['papers_with_links'] = scholar_data.get('papers_with_links', [])
            self.scholar_config['total_count'] = scholar_data.get('totalCount', 0)
            
            # 生成研究领域内容
            research_fields = ""
            for i, kw in enumerate(keywords[:4]):  # 限制使用前4个关键词
                research_fields += f"  * {kw}\n"
            
            # 预处理可能包含特殊字符的字符串，防止格式化错误
            safe_name = self._escape_special_chars(self.scholar_config['name'])
            safe_zh_name = self._escape_special_chars(self.scholar_config['zh_name'])
            safe_en_name = self._escape_special_chars(self.scholar_config['en_name'])
            safe_institution = self._escape_special_chars(self.scholar_config['institution'])
            safe_brief = self._escape_special_chars(self.scholar_config['brief'])
            safe_work_history = self._escape_special_chars(self.scholar_config['work_history'])
            safe_research_grants = self._escape_special_chars(self.scholar_config['research_grants'])
            safe_academic_index = self._escape_special_chars(self.scholar_config['academic_index'])
            safe_contact = self._escape_special_chars(self.scholar_config['contact'])
            safe_research_keywords = self._escape_special_chars(", ".join(keywords[:5]) if keywords else "")
            
            # 格式化出版物信息
            try:
                publications = "[]"
                if self.scholar_config['papers_with_links']:
                    publication_items = []
                    for paper in self.scholar_config['papers_with_links']:
                        pub_item = {
                            "title": self._escape_special_chars(paper.get('title', '')),
                            "link": self._escape_special_chars(paper.get('fulltext_link', ''))
                        }
                        publication_items.append(pub_item)
                    publications = json.dumps(publication_items, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"格式化出版物信息时出错: {str(e)}")
                publications = "[]"
            
            # 格式化奖项
            try:
                awards = json.dumps(self.scholar_config['awards'], ensure_ascii=False) if self.scholar_config['awards'] else "[]"
            except Exception as e:
                logger.warning(f"格式化奖项信息时出错: {str(e)}")
                awards = "[]"
            
            # 格式化资格认证
            try:
                qualifications = json.dumps(self.scholar_config['qualifications'], ensure_ascii=False) if self.scholar_config['qualifications'] else "[]"
            except Exception as e:
                logger.warning(f"格式化资格认证信息时出错: {str(e)}")
                qualifications = "[]"
            
            # 格式化专利
            try:
                patents = json.dumps(self.scholar_config['patents'], ensure_ascii=False) if self.scholar_config['patents'] else "[]"
            except Exception as e:
                logger.warning(f"格式化专利信息时出错: {str(e)}")
                patents = "[]"
            
            # 处理supplemented_data - 直接获取完整的补充数据
            supp_data = scholar_data.get('supplemented_data', {})
            try:
                supplemented_data = json.dumps(supp_data, ensure_ascii=False) if supp_data else "{}"
            except Exception as e:
                logger.warning(f"格式化补充数据信息时出错: {str(e)}")
                supplemented_data = "{}"
            
            # 生成场景信息
            self.scene_info = self.SCENE_INFO_TEMPLATE.format(
                name=safe_name,
                zh_name=safe_zh_name,
                en_name=safe_en_name,
                institution=safe_institution,
                brief=safe_brief,
                work_history=safe_work_history,
                publications=publications,
                total_count=self.scholar_config['total_count'],
                awards=awards,
                research_grants=safe_research_grants,
                qualifications=qualifications,
                academic_index=safe_academic_index,
                patents=patents,
                contact=safe_contact,
                research_keywords=safe_research_keywords,
                research_fields=research_fields,
                supplemented_data=supplemented_data
            )
            
            # 获取姓氏作为称呼
            last_name = self.scholar_config['name'][0] if self.scholar_config['name'] else ""
            
            # 生成系统提示词
            self.system_instruction = self.SYSTEM_INSTRUCTION_TEMPLATE.format(
                current_date=datetime.now().strftime('%Y年%m月%d日'),
                scholar_name=safe_name,
                last_name=last_name,
                scene_info=self.scene_info,
                brief=safe_brief
            )
            
            logger.info(f"已更新 {self.scholar_config['name']} 的配置信息")
            return self.scholar_config
            
        except Exception as e:
            logger.error(f"更新学者信息失败: {str(e)}")
            raise
    
    def _escape_special_chars(self, text):
        """转义字符串中的特殊字符，防止格式化错误
        
        Args:
            text: 要处理的字符串
            
        Returns:
            处理后的字符串
        """
        if not isinstance(text, str):
            return str(text)
        
        # 替换可能导致format错误的字符
        return text.replace('{', '{{').replace('}', '}}').replace('\\', '\\\\')
    
    def get_system_instruction(self):
        """获取系统提示词"""
        return self.system_instruction

    def get_scene_info(self):
        """获取场景信息"""
        return self.scene_info
    
    def get_retriever(self):
        """
        获取学者检索器
        
        Returns:
            ScholarRetriever实例
        """
        if not self.retriever:
            self.retriever = ScholarRetriever(self.scholar_open_id)
        return self.retriever
    
    def retrieve_relevant_papers(self, query, top_k=None):
        """
        检索与查询相关的论文
        
        Args:
            query: 查询文本
            top_k: 返回的最大文档数
            
        Returns:
            相关论文文档列表
        """
        retriever = self.get_retriever()
        return retriever.retrieve_relevant_papers(query, top_k)