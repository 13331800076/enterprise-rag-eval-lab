"""Streamlit UI for RAG Lab — Retrieval Strategy Comparison Dashboard."""
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
    format_markdown_table,
)

st.set_page_config(page_title="RAG Lab", page_icon="🔬", layout="wide")

st.title("🔬 RAG Lab — Retrieval Strategy Comparison")
st.markdown("Upload documents, compare BM25 vs Vector vs Hybrid retrieval, and evaluate with metrics.")


@st.cache_resource
def get_retrievers():
    return {
        "BM25": BM25Retriever(),
        "Vector": VectorRetriever(),
        "Hybrid": HybridRetriever(),
    }


# ── Sidebar: Upload & Settings ──
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop .txt, .md, or .html files",
        accept_multiple_files=True,
        type=["txt", "md", "html", "markdown"],
    )

    st.header("⚙️ Chunking Settings")
    chunk_strategy = st.selectbox("Strategy", ["Fixed Size", "Heading Based"])
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50) if chunk_strategy == "Fixed Size" else None
    overlap = st.slider("Overlap", 0, 200, 50, 10) if chunk_strategy == "Fixed Size" else None

    st.header("🔍 Reranker")
    reranker_choice = st.selectbox("Reranker", ["None", "Keyword Overlap"])


# ── State Management ──
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "docs" not in st.session_state:
    st.session_state.docs = []


# ── Ingestion ──
if uploaded_files:
    with st.spinner("Processing documents..."):
        pipeline = IngestionPipeline()
        docs = []
        temp_dir = Path(tempfile.mkdtemp())
        for file in uploaded_files:
            path = temp_dir / file.name
            path.write_bytes(file.getvalue())
            try:
                docs.append(pipeline.ingest_file(path))
            except Exception as e:
                st.error(f"Failed to parse {file.name}: {e}")

        st.session_state.docs = docs

        chunks = []
        for doc in docs:
            if chunk_strategy == "Heading Based":
                chunker = HeadingBasedChunker()
            else:
                chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
            chunks.extend(chunker.chunk(doc))

        st.session_state.chunks = chunks
        st.success(f"Indexed {len(docs)} documents, {len(chunks)} chunks")


# ── Tabs ──
tab_search, tab_eval, tab_docs = st.tabs(["🔎 Search & Compare", "📊 Evaluation", "📄 Documents"])


with tab_search:
    if not st.session_state.chunks:
        st.info("Upload documents in the sidebar to start searching.")
    else:
        query = st.text_input("Enter your question:", value="What is RAG?")
        top_k = st.slider("Top K", 1, 10, 5)

        # Index retrievers
        retrievers = get_retrievers()
        for name, retriever in retrievers.items():
            retriever.index(st.session_state.chunks)

        reranker = NoOpReranker() if reranker_choice == "None" else KeywordOverlapReranker()

        # Search all strategies
        results = {}
        for name, retriever in retrievers.items():
            raw = retriever.search(query, top_k=top_k)
            results[name] = reranker.rerank(query, raw)

        # Display side-by-side
        cols = st.columns(3)
        for idx, (name, res) in enumerate(results.items()):
            with cols[idx]:
                st.subheader(name)
                if not res:
                    st.caption("No results")
                for rank, r in enumerate(res, 1):
                    with st.container():
                        st.markdown(f"**#{rank}** | Score: `{r.score:.4f}`")
                        st.caption(f"Chunk: `{r.chunk_id}`")
                        st.text(r.text[:300] + ("..." if len(r.text) > 300 else ""))
                        st.divider()

        # Comparison chart
        if query and results:
            scores = {name: [r.score for r in res] for name, res in results.items()}
            max_len = max(len(v) for v in scores.values())
            for name in scores:
                scores[name] += [None] * (max_len - len(scores[name]))
            scores["Rank"] = list(range(1, max_len + 1))

            df = pd.DataFrame(scores).set_index("Rank")
            fig = go.Figure()
            colors = {"BM25": "#EF553B", "Vector": "#636EFA", "Hybrid": "#00CC96"}
            for name in ["BM25", "Vector", "Hybrid"]:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[name], mode="lines+markers",
                    name=name, line=dict(color=colors[name]),
                ))
            fig.update_layout(
                title="Score by Rank (Higher = Better)",
                xaxis_title="Rank", yaxis_title="Score",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)


with tab_eval:
    if not st.session_state.chunks:
        st.info("Upload documents first.")
    else:
        st.markdown("### Upload Evaluation Dataset (JSON)")
        eval_file = st.file_uploader("QA pairs with relevant_chunk_ids", type="json")

        if eval_file:
            raw = json.loads(eval_file.read())
            examples = [
                QAExample(
                    question=e["question"],
                    relevant_chunk_ids=set(e.get("relevant_chunk_ids", [])),
                    answer=e.get("answer", ""),
                )
                for e in raw
            ]

            k_values = st.multiselect("Metrics @k", [1, 3, 5, 10], default=[5])

            with st.spinner("Running evaluation..."):
                retrievers = get_retrievers()
                for name, retriever in retrievers.items():
                    retriever.index(st.session_state.chunks)

                all_results = {}
                for name, retriever in retrievers.items():
                    res = [evaluate_single(ex, retriever, name, k_values) for ex in examples]
                    all_results[name] = aggregate_metrics(res)

            # Table
            st.markdown("### Results")
            st.markdown(format_markdown_table(all_results, k_values))

            # Bar chart
            metric = st.selectbox("Metric", [f"recall@{k}" for k in k_values] + [f"mrr@{k}" for k in k_values] + [f"ndcg@{k}" for k in k_values])
            df_chart = pd.DataFrame({
                "Retriever": list(all_results.keys()),
                metric: [v.get(metric, 0) for v in all_results.values()],
            })
            fig = px.bar(df_chart, x="Retriever", y=metric, color="Retriever", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Export
            st.download_button(
                "Download Results JSON",
                data=json.dumps(all_results, indent=2),
                file_name="eval_results.json",
                mime="application/json",
            )


with tab_docs:
    if not st.session_state.docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.docs:
            with st.expander(f"📄 {doc.title} ({doc.metadata.get('format', 'unknown')})"):
                st.markdown(f"**Source:** `{doc.source_path}`")
                st.markdown(f"**Size:** {doc.metadata.get('size', '?')} chars")
                st.text(doc.text[:500] + "...")

st.markdown("---")
st.caption("Built with ❤️ using RAG Lab | [GitHub](https://github.com)")
