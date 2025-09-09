# yibocao2
# 快速开始🚀
# lang_chain_agent框架，test
langchain-agent-project/
│
├── requirements.txt                  # 依赖文件
├── .env                              # 环境变量配置
├── .gitignore                        # Git忽略文件
├── README.md                         # 项目文档
├── setup.py                          # 项目安装脚本
│
├── config/                           # 配置文件目录
│   ├── __init__.py
│   ├── settings.py                   # 全局设置
│   └── logging_config.py             # 日志配置
│
├── app/                              # 主应用代码
│   ├── __init__.py
│   ├── main.py                       # 主入口点
│   │
│   ├── core/                         # 核心功能
│   │   ├── __init__.py
│   │   ├── agent.py                  # Agent主逻辑
│   │   ├── llm.py                    # 本地LLM连接
│   │   └── memory.py                 # 记忆/历史管理
│   │
│   ├── tools/                        # 工具集合
│   │   ├── __init__.py
│   │   ├── qdrant_tool.py            # Qdrant向量库工具
│   │   ├── search_tool.py            # 搜索相关工具
│   │   └── custom_tools.py           # 自定义工具
│   │
│   ├── retrieval/                    # 检索相关组件
│   │   ├── __init__.py
│   │   ├── vectorstore.py            # 向量存储管理
│   │   ├── embeddings.py             # 嵌入模型
│   │   └── retriever.py              # 检索器实现
│   │
│   ├── chains/                       # 链组件
│   │   ├── __init__.py
│   │   ├── qa_chain.py               # 问答链
│   │   └── custom_chains.py          # 自定义链
│   │
│   ├── prompts/                      # 提示模板
│   │   ├── __init__.py
│   │   ├── qa_prompts.py             # 问答提示
│   │   └── agent_prompts.py          # Agent提示
│   │
│   └── utils/                        # 工具函数
│       ├── __init__.py
│       ├── text_processing.py        # 文本处理工具
│       ├── performance.py            # 性能监控工具
│       └── async_utils.py            # 异步工具函数
│
├── data/                             # 数据目录
│   ├── cache/                        # 缓存数据
│   └── models/                       # 本地模型存储
│
├── tests/                            # 测试代码
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_retrieval.py
│   └── test_tools.py
│
└── examples/                         # 示例代码
    ├── simple_query.py
    ├── batch_processing.py
    └── async_example.py