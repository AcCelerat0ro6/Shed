# Shed

Shed 是一个用 Python 实现的编码 Agent，面向 Python 项目的依赖升级与 API 迁移。编码 Agent 可以读取和修改工作区文件、运行命令，并通过 MCP 扩展工具。迁移检索模块按依赖包和版本查找 API 变更、参数说明等文档片段，为后续代码修改提供依据。

## 当前进度

RAG 已实现粗排阶段的流程编排和检索。

- 根据知识库和检索模式生成请求，按存储分组并行执行。向量检索可以接收预先计算的查询向量，也可以通过注入的模型服务生成向量。
- Elasticsearch 8 使用 BM25 召回关键词候选，使用余弦相似度召回语义候选。检索支持按包名、文档类型和 PEP 440 版本区间缩小候选范围。
- 召回结果保留各路候选的顺序、分数和来源信息，可供后续融合和重排。RAG 的资源装配入口负责管理 PostgreSQL 和 Elasticsearch 连接。

当前可通过 Python 调用粗排模块。版本化文档自动采集、最终排序和迁移文档工具接入属于后续阶段。

## 快速开始

运行终端需要 Python 3.11 或更高版本、`uv` 和 Node.js 22.19 或更高版本。在仓库根目录执行以下命令。

```bash
uv sync --locked
npm ci
npm run build
npm start
```

首次进入终端可用 `/login` 配置模型服务。需要在另一个项目中运行时，指定该项目的工作区路径。

```bash
node shed-tui/bin/shed.js --workspace /path/to/project
```

若要单独开发 RAG，使用 `uv sync --locked --extra rag` 安装 PostgreSQL 客户端依赖，并参考 [`.env.example`](.env.example) 配置 PostgreSQL 和 Elasticsearch 8。粗排和资源装配的入口分别在 [`hybrid_search.py`](src/shed_agent/RAG/hybrid_search.py) 和 [`container.py`](src/shed_agent/RAG/container.py)。
