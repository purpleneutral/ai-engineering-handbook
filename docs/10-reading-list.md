# Reading List (Curated)

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](09-glossary.md) | [Next](README.md)

## Summary

This is a deliberately short list of high-signal resources for AI/LLM engineering. Every entry earns its place by being either a foundational primary source or a consistently useful reference. When in doubt, read the vendor documentation for the tools you actually use — it is almost always more accurate and current than any summary.

## See Also
- [Staying Current (Without Chasing Hype)](13-staying-current.md)
- [Scope and Update Policy](00-scope-and-update-policy.md)

## Primary Documentation (Start Here)

These are the official sources you should bookmark and consult regularly. They define how the tools actually work, not how someone else thinks they work.

- **OpenAI docs: Structured outputs.** The definitive reference for schema-constrained generation with OpenAI models. Essential reading before using JSON mode or structured outputs in production. https://platform.openai.com/docs/guides/structured-outputs
- **OpenAI docs: Function calling.** Covers tool definition, invocation, and parallel tool calls. The foundation for building any tool-using system with OpenAI. https://platform.openai.com/docs/guides/function-calling
- **Anthropic docs: Tool use.** Anthropic's approach to tool calling, including tool definition schemas, multi-turn tool use, and integration patterns. https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- **OWASP Top 10 for LLM Applications.** The most widely referenced threat taxonomy for LLM systems. Covers prompt injection, insecure output handling, excessive agency, and more. Required reading for anyone shipping AI features. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **NIST AI RMF 1.0.** The U.S. government's framework for managing AI risk. Provides a structured vocabulary and process for risk identification, assessment, and mitigation. Useful even outside regulated industries. https://www.nist.gov/itl/ai-risk-management-framework
- **NIST Generative AI Profile.** An extension of the AI RMF specifically addressing generative AI risks, including hallucination, data privacy, and content provenance. https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile

## Foundational Papers

These papers define the core ideas that modern LLM engineering is built on. Reading the abstracts and key sections is sufficient for practitioners; full reads are worthwhile for anyone doing research or building infrastructure.

- **"Attention Is All You Need" (Vaswani et al., 2017).** The transformer architecture paper. Everything in modern LLMs descends from this work. If you read one paper, read this one. https://arxiv.org/abs/1706.03762
- **GPT-3: "Language Models are Few-Shot Learners" (Brown et al., 2020).** Demonstrated that scaling language models enables few-shot learning through prompting alone, without fine-tuning. Established the paradigm of prompt engineering. https://arxiv.org/abs/2005.14165
- **InstructGPT / RLHF: "Training language models to follow instructions" (Ouyang et al., 2022).** Showed how reinforcement learning from human feedback transforms a base model into an instruction-following assistant. The blueprint for alignment. https://arxiv.org/abs/2203.02155
- **Chain-of-Thought Prompting (Wei et al., NeurIPS 2022).** Demonstrated that prompting models to show reasoning steps dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks. https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html
- **"Lost in the Middle" (Liu et al., 2024).** Showed that models struggle to use information placed in the middle of long contexts, with a U-shaped attention pattern favoring the beginning and end. Directly relevant to RAG chunk ordering. https://doi.org/10.1162/tacl_a_00638
- **RAG: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020).** The original RAG paper. Introduced the idea of combining parametric (model) and non-parametric (retrieval) knowledge. https://arxiv.org/abs/2005.11401
- **Dense Passage Retrieval (Karpukhin et al., 2020).** Showed that learned dense embeddings outperform BM25 for open-domain question answering. A key building block for modern RAG systems. https://arxiv.org/abs/2004.04906
- **ReAct: "Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023).** Introduced the interleaved reasoning-and-acting paradigm for agent systems. The conceptual basis for most production agent architectures. https://arxiv.org/abs/2210.03629
- **Toolformer (Schick et al., 2023).** Showed that language models can learn to use tools (calculators, search engines, APIs) through self-supervised training. Foundational for tool-calling capabilities. https://arxiv.org/abs/2302.04761
- **MT-Bench / "Judging LLM-as-a-Judge" (Zheng et al., 2023).** Established methodology for using LLMs to evaluate other LLMs, with analysis of agreement rates with human judges. The reference for model-graded evaluation. https://arxiv.org/abs/2306.05685

## Embeddings and Vector Search

- **FAISS (Facebook AI Similarity Search).** The most widely used library for efficient similarity search and clustering of dense vectors. Essential infrastructure for any vector search system. https://github.com/facebookresearch/faiss
- **HNSW: "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018).** The algorithm behind most production vector database indexes. https://arxiv.org/abs/1603.09320
- **Sentence-BERT (Reimers & Gurevych, 2019).** Adapted BERT for producing semantically meaningful sentence embeddings. A strong baseline for text similarity tasks. https://arxiv.org/abs/1908.10084

## Evaluation Frameworks

- **OpenAI Evals.** OpenAI's open-source framework for evaluating LLM outputs. Provides a library of eval templates and a runner for custom evaluations. https://github.com/openai/evals
- **promptfoo.** A developer-friendly tool for testing prompts across models and configurations. Supports assertions, comparisons, and CI integration. https://github.com/promptfoo/promptfoo
- **Ragas.** A framework specifically designed for evaluating RAG pipelines, with metrics for faithfulness, answer relevancy, and context precision. https://docs.ragas.io/en/stable/getstarted/install/

## Staying Current (High-Signal Sources)

These sources have a high signal-to-noise ratio and are worth checking regularly. See [Staying Current](13-staying-current.md) for a framework on how to process new information without chasing hype.

- **The Batch (deeplearning.ai).** Andrew Ng's weekly newsletter. Concise, curated, and accessible. Good for keeping a pulse on the field without reading papers. https://www.deeplearning.ai/the-batch/
- **Hugging Face: Daily Papers.** Community-curated academic papers with discussion. Good for spotting important new research early. https://huggingface.co/papers
- **Import AI.** Jack Clark's long-running newsletter covering AI research and policy. More depth than most newsletters. https://importai.substack.com/
- **Papers with Code.** Tracks papers alongside their implementations and benchmark results. Useful for finding reproducible research. https://paperswithcode.com/about
- **NLP News.** A curated newsletter focused on natural language processing research and applications. https://www.nlpnews.org/

---
[Contents](README.md) | [Prev](09-glossary.md) | [Next](README.md)
