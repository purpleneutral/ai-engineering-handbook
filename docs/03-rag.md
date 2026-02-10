# Retrieval-Augmented Generation (RAG)

Last reviewed: 2026-02-10

## Summary
RAG grounds model outputs in external knowledge by retrieving relevant documents and using them as context.

## When To Use
- You need up-to-date or domain-specific facts.
- You need traceability (citations) to source material.
- Fine-tuning is not feasible, too slow, or too expensive for the change rate.

## High-Level Pipeline
- Ingest: collect documents, normalize formats, extract text.
- Chunk: split into searchable units.
- Embed + index: create embeddings and store in a vector index.
- Retrieve: similarity search (+ filters).
- Re-rank: optional, improves precision.
- Synthesize: answer using retrieved chunks, ideally with citations.

## Chunking Heuristics
- Chunk by semantics, not fixed length when possible.
- Preserve metadata (title, section headers, timestamps, permissions).
- Avoid mixing unrelated topics in a single chunk.

## RAG Is Not Automatically Safe
- Retrieved text is untrusted input and can contain prompt injection attempts.
- Keep system rules separate from retrieved text.
- Consider extracting facts into a structured intermediate format before generation.

## Evaluation
- Retrieval quality: does it fetch the right stuff?
- Answer faithfulness: does the answer match the retrieved text?
- Coverage: does it handle hard/rare queries?

## Checklist
- Can you reproduce an answer from logs (query + retrieved chunk ids)?
- Do you have an eval set of real queries?
- Do you have a way to detect and handle empty/low-confidence retrieval?
- Are you enforcing document access control at retrieval time?

## References
- RAG paper (Lewis et al., 2020). https://arxiv.org/abs/2005.11401
- Dense Passage Retrieval (DPR). https://arxiv.org/abs/2004.04906
