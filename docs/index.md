# RAG Lab 在线体验

无需安装，直接体验：

## 方式一：Streamlit Cloud（推荐）

点击按钮一键部署：

[![Deploy to Streamlit](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-red?style=for-the-badge)](https://streamlit.io/cloud)

## 方式二：Docker 一键运行

```bash
docker run -p 8501:8501 ghcr.io/13331800076/enterprise-rag-eval-lab:latest
```

然后打开 http://localhost:8501

## 方式三：本地快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 快速评测

```bash
python -m examples.run_leaderboard
```
