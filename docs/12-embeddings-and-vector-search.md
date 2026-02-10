# Embeddings And Vector Search

Last reviewed: 2026-02-10

## Summary
Embeddings map text (and sometimes images/audio) into vectors so you can do similarity search, clustering, and semantic filtering. Vector search is the backbone of most RAG systems.

## When To Use
- Semantic search over documents.
- Retrieval for RAG.
- Deduplication and clustering.
- Similarity-based routing (pick a workflow/tool based on nearest examples).

## How It Works
### Embeddings
- An embedding model converts content into a fixed-length vector.
- Similarity is typically computed via cosine similarity or dot product.

### Nearest Neighbor Search
- Exact search is expensive at scale.
- Approximate nearest neighbor (ANN) indexes trade a small amount of recall for speed and memory improvements.

## Practical Design Choices
### Chunking And Metadata
- Chunk by meaning; keep chunks reasonably self-contained.
- Store metadata alongside vectors (title, URL, timestamps, permissions, doc id).
- Enforce access control at query time (filtering) before generation.

### Retrieval Strategy
- Start simple: vector similarity + metadata filters.
- Add a reranker when precision matters (and budget allows).
- Consider hybrid search (BM25 + vectors) for short queries and exact terms.

### Evaluation
- Retrieval: recall@k / precision@k on a labeled query set.
- End-to-end: answer faithfulness and usefulness on real tasks.

## Pitfalls
- Garbage in, garbage out: poor chunking and noisy text kill retrieval quality.
- Missing metadata makes debugging and access control harder.
- Embedding model changes can silently shift neighbors; version your index.

## Checklist
- Do you version the embedding model and the index build?
- Do you log retrieved ids (not raw content) for debugging?
- Do you have a labeled retrieval eval set?
- Do you have reranking for hard queries?
- Do you enforce ACL filtering during retrieval?

## References
- FAISS (vector similarity search library). https://github.com/facebookresearch/faiss
- FAISS paper (efficient similarity search and clustering). https://arxiv.org/abs/2401.08281
- HNSW paper (graph-based ANN). https://arxiv.org/abs/1603.09320
- Sentence-BERT paper (strong baseline for embeddings). https://arxiv.org/abs/1908.10084
- pgvector (Postgres vector extension). https://github.com/pgvector/pgvector

