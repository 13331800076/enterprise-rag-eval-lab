# RAG Overview

## What is RAG

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to ground the LLM's responses in factual information.

## Why RAG Matters

Large language models can hallucinate facts. RAG reduces hallucination by providing external evidence. The retrieved passages act as context windows for the generation model.

## Key Components

A complete RAG pipeline includes document ingestion, text chunking, vector embeddings, retrieval algorithms, and answer generation.

## Evaluation

Common metrics for RAG retrieval include Recall@k, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG). These metrics help compare different retrieval strategies.
