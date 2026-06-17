# 快速部署指南

## 方式一：Docker 一键运行（无需安装依赖）

```bash
docker run -p 8000:8000 ghcr.io/13331800076/enterprise-rag-eval-lab:latest
```

然后访问 http://localhost:8000/docs

## 方式二：Streamlit 界面

```bash
docker run -p 8501:8501 ghcr.io/13331800076/enterprise-rag-eval-lab:latest streamlit run app.py
```

访问 http://localhost:8501

## 方式三：GitHub Codespaces（零配置）

点击仓库首页的 "Code" → "Codespaces" → "Create codespace on main"

VS Code 网页版自动打开，依赖已安装，直接运行：
```bash
streamlit run app.py
```

## 方式四：本地开发

```bash
git clone https://github.com/13331800076/enterprise-rag-eval-lab.git
cd enterprise-rag-eval-lab
pip install -r requirements.txt
pytest
streamlit run app.py
```

## API 快速测试

```bash
curl http://localhost:8000/health
```
