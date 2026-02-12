# Reading List (Curated)

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](09-glossary.md) | [Next](19-history-of-ai.md)

## Summary

This is a deliberately short list of high-signal resources for AI/LLM engineering. Every entry earns its place by being either a foundational primary source or a consistently useful reference. When in doubt, read the vendor documentation for the tools you actually use — it is almost always more accurate and current than any summary.

## See Also
- [Staying Current (Without Chasing Hype)](13-staying-current.md)
- [Scope and Update Policy](00-scope-and-update-policy.md)

## Primary Documentation (Start Here)

These are the official sources you should bookmark and consult regularly. They define how the tools actually work, not how someone else thinks they work.

- **[OpenAI docs: Structured outputs](https://platform.openai.com/docs/guides/structured-outputs).** The definitive reference for schema-constrained generation with OpenAI models. Essential reading before using JSON mode or structured outputs in production.
- **[OpenAI docs: Function calling](https://platform.openai.com/docs/guides/function-calling).** Covers tool definition, invocation, and parallel tool calls. The foundation for building any tool-using system with OpenAI.
- **[Anthropic docs: Tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).** Anthropic's approach to tool calling, including tool definition schemas, multi-turn tool use, and integration patterns.
- **[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).** The most widely referenced threat taxonomy for LLM systems. Covers prompt injection, insecure output handling, excessive agency, and more. Required reading for anyone shipping AI features.
- **[NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework).** The U.S. government's framework for managing AI risk. Provides a structured vocabulary and process for risk identification, assessment, and mitigation. Useful even outside regulated industries.
- **[NIST Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile).** An extension of the AI RMF specifically addressing generative AI risks, including hallucination, data privacy, and content provenance.

## Foundational Papers

These papers define the core ideas that modern LLM engineering is built on. Reading the abstracts and key sections is sufficient for practitioners; full reads are worthwhile for anyone doing research or building infrastructure.

- **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).** The transformer architecture paper. Everything in modern LLMs descends from this work. If you read one paper, read this one.
- **[GPT-3: "Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (Brown et al., 2020).** Demonstrated that scaling language models enables few-shot learning through prompting alone, without fine-tuning. Established the paradigm of prompt engineering.
- **[InstructGPT / RLHF: "Training language models to follow instructions"](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022).** Showed how reinforcement learning from human feedback transforms a base model into an instruction-following assistant. The blueprint for alignment.
- **[Chain-of-Thought Prompting](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) (Wei et al., NeurIPS 2022).** Demonstrated that prompting models to show reasoning steps dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks.
- **["Lost in the Middle"](https://doi.org/10.1162/tacl_a_00638) (Liu et al., 2024).** Showed that models struggle to use information placed in the middle of long contexts, with a U-shaped attention pattern favoring the beginning and end. Directly relevant to RAG chunk ordering.
- **[RAG: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020).** The original RAG paper. Introduced the idea of combining parametric (model) and non-parametric (retrieval) knowledge.
- **[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020).** Showed that learned dense embeddings outperform BM25 for open-domain question answering. A key building block for modern RAG systems.
- **[ReAct: "Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) (Yao et al., 2023).** Introduced the interleaved reasoning-and-acting paradigm for agent systems. The conceptual basis for most production agent architectures.
- **[Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023).** Showed that language models can learn to use tools (calculators, search engines, APIs) through self-supervised training. Foundational for tool-calling capabilities.
- **[MT-Bench / "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023).** Established methodology for using LLMs to evaluate other LLMs, with analysis of agreement rates with human judges. The reference for model-graded evaluation.

## Embeddings and Vector Search

- **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss).** The most widely used library for efficient similarity search and clustering of dense vectors. Essential infrastructure for any vector search system.
- **[HNSW: "Efficient and robust approximate nearest neighbor using Hierarchical Navigable Small World graphs"](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2018).** The algorithm behind most production vector database indexes.
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019).** Adapted BERT for producing semantically meaningful sentence embeddings. A strong baseline for text similarity tasks.

## Evaluation Frameworks

- **[OpenAI Evals](https://github.com/openai/evals).** OpenAI's open-source framework for evaluating LLM outputs. Provides a library of eval templates and a runner for custom evaluations.
- **[promptfoo](https://github.com/promptfoo/promptfoo).** A developer-friendly tool for testing prompts across models and configurations. Supports assertions, comparisons, and CI integration.
- **[Ragas](https://docs.ragas.io/en/stable/).** A framework specifically designed for evaluating RAG pipelines, with metrics for faithfulness, answer relevancy, and context precision.

## Staying Current (High-Signal Sources)

These sources have a high signal-to-noise ratio and are worth checking regularly. See [Staying Current](13-staying-current.md) for a framework on how to process new information without chasing hype.

- **[The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/).** Andrew Ng's weekly newsletter. Concise, curated, and accessible. Good for keeping a pulse on the field without reading papers.
- **[Hugging Face: Daily Papers](https://huggingface.co/papers).** Community-curated academic papers with discussion. Good for spotting important new research early.
- **[Import AI](https://importai.substack.com/).** Jack Clark's long-running newsletter covering AI research and policy. More depth than most newsletters.
- **[Papers with Code](https://paperswithcode.com/about).** Tracks papers alongside their implementations and benchmark results. Useful for finding reproducible research.
- **[NLP News](https://www.nlpnews.org/).** A curated newsletter focused on natural language processing research and applications.

## Recent Developments and Surveys

These papers and reports cover important developments since the foundational work listed above. They represent significant shifts in how models are trained, aligned, scaled, and made efficient.

- **["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020).** Established empirical power-law relationships between model performance and model size, dataset size, and compute budget. These scaling laws directly informed the design decisions behind GPT-3 and subsequent large models, and remain the theoretical basis for compute-optimal training.
- **["Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) (Bai et al., 2022).** Introduced Anthropic's method for training harmless AI assistants using a written set of principles (a "constitution") rather than relying entirely on human feedback. The two-phase approach — supervised self-critique followed by reinforcement learning from AI feedback (RLAIF) — reduces the need for human labelers while making alignment criteria explicit and auditable.
- **["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (Hu et al., 2021).** Proposed freezing pre-trained weights and injecting trainable low-rank matrices into transformer layers, reducing trainable parameters by 10,000x and GPU memory by 3x compared to full fine-tuning. LoRA is now the dominant method for parameter-efficient fine-tuning across the industry.
- **["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023).** Extended LoRA to work on 4-bit quantized models, enabling fine-tuning of 65B-parameter models on a single 48GB GPU. Introduced 4-bit NormalFloat quantization and demonstrated that the quality-memory trade-off is far more favorable than previously assumed. Published at NeurIPS 2023.
- **["Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017).** The foundational Mixture-of-Experts paper that introduced sparsely-gated MoE layers, enabling 1000x increases in model capacity with only minor computation overhead. This architecture underpins modern MoE models like Mixtral and DeepSeek-V3.
- **["A Survey on Mixture of Experts in Large Language Models"](https://arxiv.org/abs/2407.06204) (Cai et al., 2024).** A comprehensive survey covering MoE taxonomy, core designs, routing mechanisms, and open-source implementations. Useful as an entry point for understanding how MoE is applied in modern LLMs. Accepted at IEEE TKDE 2025.
- **["DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"](https://arxiv.org/abs/2501.12948) (Guo et al., 2025).** Demonstrated that reasoning capabilities can emerge from pure reinforcement learning without human-labeled reasoning trajectories. The open-source release of DeepSeek-R1 and its distilled variants made frontier-level reasoning accessible outside closed-model providers. Published in Nature 2025.
- **["Gemini: A Family of Highly Capable Multimodal Models"](https://arxiv.org/abs/2312.11805) (Gemini Team, Google, 2023).** Technical report for Google's Gemini model family, covering multimodal capabilities across text, image, audio, and video. The first model to achieve human-expert performance on MMLU.
- **["Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context"](https://arxiv.org/abs/2403.05530) (Gemini Team, Google, 2024).** Introduced million-token context windows with near-perfect retrieval up to 10M tokens, a generational leap in long-context capability. Covers both Gemini 1.5 Pro and the efficiency-optimized Gemini 1.5 Flash.
- **[Anthropic: Claude's constitution](https://www.anthropic.com/news/claude-new-constitution).** Anthropic's public description of the principles and values that guide Claude's behavior, released under Creative Commons CC0. A practical example of how Constitutional AI is applied to a production model.
- **[Model Context Protocol (MCP) specification](https://modelcontextprotocol.io/specification/2025-11-25).** The open specification for connecting LLM applications with external tools and data sources. As of 2026-02-11, MCP is governed by the Agentic AI Foundation under the Linux Foundation and supported by Anthropic, OpenAI, Google, and Microsoft.

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](09-glossary.md) | [Next](19-history-of-ai.md)
