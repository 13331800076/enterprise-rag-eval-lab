"""Streamlit RAG Evaluation Lab UI."""

import json
import tempfile
from pathlib import Path

import streamlit as st
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

st.set_page_config(page_title="RAG Evaluation Lab", layout="wide")

st.title("🔬 RAG Evaluation Lab")
st.markdown("Upload documents, compare retrieval strategies, and evaluate results.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["Fixed Size", "Heading Based (Markdown only)"],
    )
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    overlap = st.slider("Overlap", 0, 200, 50, 10)

    st.subheader("Retrieval Strategy")
    use_bm25 = st.checkbox("BM25", value=True)
    use_vector = st.checkbox("Vector (Dense)", value=True)
    use_hybrid = st.checkbox("Hybrid", value=True)

    st.subheader("Reranker")
    reranker_choice = st.selectbox(
        "Reranker",
        ["NoOp", "KeywordOverlap"],
    )

    st.subheader("Evaluation")
    k_value = st.slider("k for metrics", 1, 10, 5)

    if st.button("🗑️ Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# --- State ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}
if "eval_examples" not in st.session_state:
    st.session_state.eval_examples = []


# --- Upload ---
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload .txt, .md, or .html files",
    accept_multiple_files=True,
    type=["txt", "md", "html"],
)

if uploaded_files and st.button("🚀 Process Documents"):
    with st.spinner("Processing..."):
        pipeline = IngestionPipeline()
        chunks = []

        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            doc = pipeline.ingest_file(tmp_path)

            if chunk_strategy == "Heading Based (Markdown only)" and doc.metadata.get("format") == "markdown":
                chunker = HeadingBasedChunker()
            else:
                chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)

            chunks.extend(chunker.chunk(doc))

        st.session_state.chunks = chunks
        st.session_state.retrievers = {}

        # Index retrievers
        if use_bm25:
            bm25 = BM25Retriever()
            bm25.index(chunks)
            st.session_state.retrievers["BM25"] = bm25

        if use_vector:
            with st.spinner("Loading embedding model..."):
                vector = VectorRetriever()
                vector.index(chunks)
                st.session_state.retrievers["Vector"] = vector

        if use_hybrid and use_bm25 and use_vector:
            bm25 = st.session_state.retrievers.get("BM25")
            vector = st.session_state.retrievers.get("Vector")
            if bm25 and vector:
                hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)
                hybrid.index(chunks)
                st.session_state.retrievers["Hybrid"] = hybrid

        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} documents")


# --- Search ---
st.header("🔍 Search")
query = st.text_input("Enter your question:", placeholder="What is RAG?")

col1, col2, col3 = st.columns(3)

if st.session_state.chunks and query:
    reranker = NoOpReranker() if reranker_choice == "NoOp" else KeywordOverlapReranker()

    for name, retriever in st.session_state.retrievers.items():
        results = retriever.search(query, top_k=k_value)
        reranked = reranker.rerank(query, results)

        with col1 if name == "BM25" else col2 if name == "Vector" else col3:
            st.subheader(f"📌 {name}")
            for i, r in enumerate(reranked[:k_value], 1):
                with st.expander(f"{i}. [{r.chunk_id}] score={r.score:.3f}"):
                    st.text(r.text[:300])


# --- Evaluation ---
st.header("📊 Evaluation")

eval_tab, upload_tab = st.tabs(["Run Evaluation", "Upload QA Dataset"])

with upload_tab:
    eval_file = st.file_uploader("Upload eval_dataset.json", type=["json"])
    if eval_file:
        data = json.loads(eval_file.getvalue().decode())
        st.session_state.eval_examples = [
            QAExample(
                question=e["question"],
                relevant_chunk_ids=set(e["relevant_chunk_ids"]),
                answer=e.get("answer", ""),
            )
            for e in data
        ]
        st.success(f"Loaded {len(st.session_state.eval_examples)} examples")

with eval_tab:
    if not st.session_state.retrievers:
        st.info("Process documents first")
    elif not st.session_state.eval_examples:
        st.info("Upload a QA dataset or use the default fixtures")
        if st.button("▶️ Run with Default Fixtures"):
            with st.spinner("Running evaluation..."):
                pipeline = IngestionPipeline()
                docs = pipeline.ingest_directory("tests/fixtures")
                chunks = []
                for doc in docs:
                    if chunk_strategy == "Heading Based (Markdown only)" and doc.metadata.get("format") == "markdown":
                        chunker = HeadingBasedChunker()
                    else:
                        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
                    chunks.extend(chunker.chunk(doc))

                st.session_state.chunks = chunks
                retrievers = {}

                bm25 = BM25Retriever()
                bm25.index(chunks)
                retrievers["BM25"] = bm25

                vector = VectorRetriever()
                vector.index(chunks)
                retrievers["Vector"] = vector

                hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)
                hybrid.index(chunks)
                retrievers["Hybrid"] = hybrid

                st.session_state.retrievers = retrievers

                # Load default eval dataset
                eval_path = Path("examples/eval_dataset.json")
                if eval_path.exists():
                    with open(eval_path) as f:
                        data = json.load(f)
                    st.session_state.eval_examples = [
                        QAExample(
                            question=e["question"],
                            relevant_chunk_ids=set(e["relevant_chunk_ids"]),
                            answer=e.get("answer", ""),
                        )
                        for e in data
                    ]

                st.rerun()
    else:
        if st.button("▶️ Run Evaluation"):
            with st.spinner("Evaluating..."):
                k_values = [k_value]
                all_results = {}

                for name, retriever in st.session_state.retrievers.items():
                    results = []
                    for example in st.session_state.eval_examples:
                        results.append(evaluate_single(example, retriever, name, k_values))
                    all_results[name] = results

                aggregated = {
                    name: aggregate_metrics(results) for name, results in all_results.items()
                }

                st.session_state.aggregated = aggregated
                st.session_state.all_results = all_results

                st.success("Evaluation complete!")

        if "aggregated" in st.session_state:
            st.markdown("### Results Table")
            st.markdown(format_markdown_table(st.session_state.aggregated, [k_value]))

            st.markdown("### Chart")
            names = list(st.session_state.aggregated.keys())
            metrics = [f"recall@{k_value}", f"mrr@{k_value}", f"ndcg@{k_value}"]

            fig = go.Figure()
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=names,
                    y=[st.session_state.aggregated[n].get(metric, 0) for n in names],
                ))
            fig.update_layout(
                barmode="group",
                title=f"Retrieval Metrics @ {k_value}",
                yaxis_title="Score",
                xaxis_title="Retriever",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Export Report")
            report = f"# RAG Evaluation Report\n\n"
            report += format_markdown_table(st.session_state.aggregated, [k_value])
            report += "\n\n## Detailed Results\n\n"

            for name, results in st.session_state.all_results.items():
                report += f"### {name}\n\n"
                for r in results:
                    report += f"- **Q**: {r.question}\n"
                    report += f"  - Retrieved: {', '.join(r.retrieved_ids[:5])}\n"
                    report += f"  - Metrics: {r.metrics}\n\n"

            st.download_button(
                "Download Markdown Report",
                report,
                file_name="rag_eval_report.md",
                mime="text/markdown",
            )


# --- Footer ---
st.markdown("---")
st.markdown(
    "Built with ❤️ using [RAG Evaluation Lab](https://github.com/yourusername/enterprise-rag-eval-lab)"
)
