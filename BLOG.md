# 我搭建了一个 RAG 策略对比实验室，发现 Hybrid 检索不一定比纯 Vector 强

> 开源项目：**[enterprise-rag-eval-lab](https://github.com/13331800076/enterprise-rag-eval-lab)** — 一个开箱即用的企业级 RAG 评估系统，支持文档解析、混合检索、重排序和可视化对比。

## 为什么做这个

面试时被问到："你们线上 RAG 用 BM25 还是向量检索？"我回答 Hybrid，因为直觉上"两者结合应该更强"。

但面试官追问："有数据支撑吗？"我答不上来。

所以我决定自己搭建一个**RAG 策略对比实验室**，用真实数据验证每一种组合的效果。

## 项目架构

```
文档上传 → 解析(.md/.txt/.html/.pdf) → 文本切分
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓               ↓               ↓
                BM25 检索      Vector 检索      Hybrid 融合
                    └───────────────┬───────────────┘
                                    ↓
                              重排序(Rerank)
                                    ↓
                              答案生成
                                    ↓
                              评估对比
```

## 评测结果（真实数据）

我在 3 份文档（RAG 概述、BM25 指南、向量搜索）上跑了 5 个问答对，对比了 6 种策略组合：

| Rank | 策略 | Recall@1 | Recall@5 | MRR@5 | nDCG@5 |
|------|------|:---:|:---:|:---:|:---:|
| 🥇 | 纯 Vector | 0.80 | 0.90 | **1.00** | **0.91** |
| 🥈 | Hybrid + Keyword Rerank | 0.70 | 0.90 | 0.84 | 0.85 |
| 🥉 | Vector + Keyword Rerank | 0.70 | 0.90 | 0.85 | 0.84 |
| 4 | Hybrid | 0.60 | 0.90 | 0.75 | 0.79 |
| 5 | BM25 | 0.60 | 0.80 | 0.72 | 0.71 |
| 6 | BM25 + Keyword Rerank | 0.60 | 0.80 | 0.71 | 0.71 |

### 意外发现

**Vector 纯语义检索在这个数据集上打败了 Hybrid。**

原因分析：
- 测试文档是**概念性文本**（定义、解释），语义相似度比关键词匹配更稳定
- Hybrid 的加权融合（alpha=0.5）可能过度平滑了 Vector 的高分
- BM25 在 Recall@5 上不差（0.80），但 MRR 明显低，说明排名质量差

这说明：**没有万能策略，必须针对数据类型做评测选型。**

## 核心功能

### 1. 文档解析（4 种格式）

```python
from rag_lab.ingestion import IngestionPipeline
pipeline = IngestionPipeline()
docs = pipeline.ingest_directory("./docs/")  # .md .txt .html .pdf
```

- PDF 支持 PyMuPDF 解析，保留页码和表格结构
- Markdown 保留标题层级，支持 HeadingBased 切分

### 2. 多种检索策略

```python
from rag_lab.retrieval import BM25Retriever, VectorRetriever, HybridRetriever

bm25 = BM25Retriever()
vector = VectorRetriever(model_name="all-MiniLM-L6-v2")
hybrid = HybridRetriever(bm25, vector, alpha=0.5, fusion="rrf")
```

### 3. 重排序对比

```python
from rag_lab.reranking import KeywordOverlapReranker, CrossEncoderReranker

keyword_rerank = KeywordOverlapReranker()
ce_rerank = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### 4. 一键评测排行榜

```bash
python -m examples.run_leaderboard
```

自动生成 6 种策略的对比表格，按 nDCG@5 排序。

### 5. Streamlit 可视化界面

```bash
streamlit run app.py
```

- 上传文档 → 实时切分
- 输入问题 → 三栏并排对比（BM25 / Vector / Hybrid）
- 评测标签 → 上传 QA 对，生成图表和排行榜
- 导出报告 → JSON/Markdown 格式

## 技术栈

| 模块 | 技术 |
|------|------|
| 文档解析 | `markdown`, `beautifulsoup4`, `PyMuPDF` |
| 文本切分 | FixedSize / HeadingBased / Layout-aware |
| 向量检索 | `sentence-transformers` + `FAISS` |
| 词汇检索 | `rank-bm25` |
| 重排序 | KeywordOverlap / CrossEncoder |
| 评估 | Recall@k, MRR, nDCG |
| API | `FastAPI` + `uvicorn` |
| UI | `Streamlit` + `Plotly` |
| 测试 | `pytest` + GitHub Actions CI |
| 部署 | `Docker` + `docker-compose` |

## 快速开始（5 分钟跑起来）

```bash
# 1. 克隆
git clone https://github.com/13331800076/enterprise-rag-eval-lab.git
cd enterprise-rag-eval-lab

# 2. 安装
pip install -r requirements.txt

# 3. 跑测试
pytest

# 4. 启动 UI
streamlit run app.py

# 5. 或启动 API
docker compose up
```

## 学到的教训

1. **不要假设 Hybrid 一定更强** — 必须跑评测
2. **重排序效果看数据类型** — KeywordOverlap 对短查询有效，CrossEncoder 对语义模糊查询更有效
3. **Chunking 策略影响检索质量** — HeadingBased 对结构化文档（论文、手册）明显优于 FixedSize
4. **评测数据集要反映真实分布** — 5 个问答对不够，下一步要接入 MS MARCO 子集

## 下一步

- [ ] 接入 OpenAI/Claude 做真实 LLM 生成对比
- [ ] 支持 Milvus 向量数据库后端
- [ ] 用 MS MARCO 1000 对跑标准评测
- [ ] 加 PDF 表格结构保留（layout-aware chunking）

## 关于作者

中软国际 AI 应用工程师，专注于企业级文档检索和 NLP 系统。这个项目是为了验证 RAG 策略选型而做的实验平台，欢迎 Issue 和 PR。

---

⭐ GitHub 开源地址：[https://github.com/13331800076/enterprise-rag-eval-lab](https://github.com/13331800076/enterprise-rag-eval-lab)
