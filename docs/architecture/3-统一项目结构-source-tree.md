# 3\. 统一项目结构 (Source Tree)

```plaintext
a_share_rag_project/
│
├── 📜 pyproject.toml         # 项目元数据和uv配置
├── 📜 requirements.txt         # (由uv管理) 项目依赖库
├── 📜 uv.lock                  # (由uv管理) 锁定依赖版本
│
├── 📂 .github/                # GitHub Actions CI/CD 工作流
│   └── 📂 workflows/
│       ├── 📄 ci.yaml          # 持续集成工作流
│       └── 📄 cd.yaml          # 持续部署工作流
│
├── 📂 .env.example            # 环境变量模板，用于存放API密钥等敏感信息
│
├── 📂 config/                  # 存放所有配置和Prompt模板
│   ├── 📄 prompts.yaml          # 将所有prompt集中存放于此
│   └── 📄 settings.py          # (使用Pydantic-Settings)加载环境变量和配置
│
├── 📂 data/                    # 原始数据
│   └── 📄 corpus.json          # A股上市公司数据
│
├── 📂 output/                  # 所有生成的数据资产存放处
│   ├── 📂 graph/              # 存放生成的igraph图谱文件
│   └── 📂 vector_store/       # 存放LanceDB向量数据库文件
│
├── 📂 src/                     # 项目核心源码
│   │
│   ├── 📂 adapters/           # 与第三方库交互的适配器
│   │   ├── 📄 __init__.py
│   │   ├── 📄 llm_adapter.py     
│   │   ├── 📄 embedding_adapter.py 
│   │   └── 📄 reranker_adapter.py  
│   │
│   ├── 📂 components/         # 七大核心组件的实现
│   │   ├── 📄 __init__.py
│   │   └── ... (7个组件的.py文件)
│   │
│   ├── 📂 server/             # 在线查询API服务 (使用FastAPI)
│   │   └── 📄 main.py            # API服务入口
│   │
│   └── 📄 pipeline.py            # 离线数据处理与索引构建的流程脚本
│
├── 📂 notebooks/               # 人类监督者用于实验、分析和验证的Jupyter Notebooks
│
└── 📂 tests/                   # 自动化测试
    ├── 📂 unit/               # 单元测试
    ├── 📂 integration/        # 集成测试
    └── 📄 conftest.py         # Pytest的配置文件
```
