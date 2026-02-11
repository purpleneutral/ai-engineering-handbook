# Architecture Recipes

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](06-safety-privacy-security.md) | [Next](08-ops.md)

## Summary
These are common AI feature shapes and the pieces that usually matter for correctness and reliability. Each recipe describes a pattern that appears repeatedly across production LLM systems. The goal is not to provide copy-paste architectures but to give you a mental model for each pattern: what the data flow looks like, where the risks concentrate, and what you need to get right before shipping. Most real systems combine multiple recipes, so understanding the building blocks is more valuable than memorizing any single configuration.

## See Also
- [Stacks And Difficulty](16-stacks-and-difficulty.md)
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md)
- [Retrieval-Augmented Generation (RAG)](03-rag.md)
- [Agents](04-agents.md)

## Chat Assistant (General Q&A)

A chat assistant is the most familiar LLM application: a conversational interface where users ask questions and receive natural language answers. Despite its apparent simplicity, a production chat assistant involves a surprising number of design decisions that affect quality, safety, and user experience.

### Data Flow

The typical data flow begins with the user's message, which is combined with a conversation history (previous turns in the session), a system prompt (defining the assistant's behavior, tone, and boundaries), and optionally user profile information (name, role, preferences) and retrieved knowledge (from a knowledge base or previous conversations). This assembled context is sent to the model, which generates a response that is then validated, logged, and returned to the user.

The conversation history deserves particular attention. As conversations grow long, the accumulated context can exceed the model's context window or degrade response quality by burying the current question under pages of earlier discussion. Common strategies include truncating older turns, summarizing them into a condensed representation, or using a sliding window that keeps only the most recent N turns in full detail while summarizing earlier ones.

### When To Combine With Other Patterns

A standalone chat assistant is appropriate for general conversation, brainstorming, and tasks where the model's training knowledge is sufficient. However, most production assistants benefit from combining with other patterns. Add RAG when users ask questions that require domain-specific or up-to-date information. Add tool calling when users need the assistant to take actions (look up data, perform calculations, interact with systems). Add structured outputs when the assistant produces artifacts (code, configurations, structured data) that downstream systems consume.

### Risks and Mitigations

**Hallucinations** are the primary risk for any generative system. The model will confidently state things that are not true, especially when it does not have relevant information in context. Mitigate with RAG (grounding in source material), with explicit instructions to acknowledge uncertainty, and with user-facing caveats about the system's limitations.

**Policy violations** occur when the model generates content that violates your organization's policies or the model provider's usage policies. This includes harmful content, biased outputs, and responses that exceed the assistant's intended scope. Mitigate with clear system prompt boundaries, output content filtering, and regular safety evaluations.

**Inconsistent tone** erodes user trust over time. If the assistant is formal in one response and casual in the next, or if it contradicts something it said earlier in the conversation, users lose confidence. Mitigate with detailed system prompt instructions about voice and style, and with consistency evaluations in your eval suite.

### Checklist
- Clear refusal and escalation paths for out-of-scope or sensitive requests
- Memory policy defining what persists across sessions and what does not
- Logging and redaction strategy that captures enough for debugging without retaining sensitive data
- Conversation history management that keeps context relevant without exceeding limits
- Safety evaluations covering adversarial inputs and policy boundary cases

## Document Q&A (RAG)

Document Q&A is the most common RAG application: users ask questions and the system answers by retrieving and synthesizing information from a document corpus. This pattern is widely used for customer support knowledge bases, internal documentation search, legal research, and any domain where answers must be grounded in authoritative sources.

### Data Flow

The data flow has two phases: an offline indexing phase and an online query phase.

During indexing, documents are collected from their sources, converted to clean text, chunked into searchable units, embedded using a vector embedding model, and stored in a vector index alongside their metadata (source, title, access control tags, timestamps). This pipeline runs periodically or in response to document changes.

During query processing, the user's question is embedded using the same model, and a similarity search retrieves the top-k most relevant chunks. Optionally, a re-ranker scores these chunks more precisely. The question and the retrieved chunks are then assembled into a synthesis prompt, and the model generates an answer with citations pointing back to the source chunks.

### When To Combine With Other Patterns

Document Q&A often benefits from combining with a chat interface (allowing follow-up questions and conversational refinement), with structured outputs (when the answer needs to be in a specific format for downstream consumption), and with tool calling (for example, to let the model request additional searches or access external APIs when the initial retrieval is insufficient).

A particularly powerful combination is document Q&A with a summarization step: first retrieve relevant chunks, then generate a structured summary, then answer the question based on the summary. This two-stage approach improves faithfulness because the summarization step acts as a filter that strips away irrelevant details and potential injection attempts.

### Risks and Mitigations

**Wrong retrieval** is the most common failure: the system retrieves chunks that are not relevant to the question, leading to wrong or irrelevant answers. This can be caused by poor chunking (relevant information split across chunks), poor embedding quality (the embedding model does not capture the right notion of similarity for your domain), or missing content (the answer simply is not in the corpus). Diagnose by evaluating retrieval quality separately from answer quality.

**Prompt injection via documents** is a security risk unique to RAG. If an attacker can influence the contents of your document corpus, they can embed instructions that the model may follow. See the [Safety, Privacy, And Security](06-safety-privacy-security.md) chapter for detailed mitigations.

**Fake citations** occur when the model generates a citation that does not actually support the claim, or cites a source that does not exist. This is a form of hallucination that is particularly damaging because citations are supposed to provide trust. Mitigate with faithfulness evaluation that verifies each citation against the cited chunk.

### Checklist
- Access control at retrieval time enforcing document-level permissions
- Faithfulness evaluations that verify answers match retrieved content
- "No answer" behavior for low-confidence retrieval (better to say "I don't know" than to hallucinate)
- Citation verification in the eval pipeline
- Monitoring for retrieval quality drift as the document corpus grows and changes

## Extraction / Structuring

Extraction is the pattern of converting unstructured text into structured data: pulling entities, relationships, dates, amounts, and other fields from documents, emails, logs, or any other text source. This is one of the most commercially valuable LLM applications because it automates work that previously required either custom NLP pipelines or manual data entry.

### Data Flow

The input is unstructured text (an invoice, a contract clause, a medical note, a customer email). The text is passed to the model along with a schema that defines the expected output structure (JSON Schema, Pydantic model, or equivalent) and instructions that describe how to extract each field, including examples of tricky cases. The model produces a structured output that is validated against the schema. If validation fails, the system can retry with an error message explaining what went wrong.

For long documents that exceed the context window, the text must be processed in segments. This introduces the challenge of merging partial extractions: if an entity spans two segments, or if context from an earlier segment is needed to interpret a later one, the extraction logic must handle this.

### When To Combine With Other Patterns

Extraction often serves as a preprocessing step for other patterns. A RAG system might use extraction to create structured metadata from documents during indexing. An agent might use extraction to parse tool outputs before deciding on next steps. A chat assistant might use extraction behind the scenes to structure user requests before processing them.

Extraction can also be combined with validation pipelines that go beyond schema checking. For example, after extracting dates from a contract, a validation step might verify that the dates are logically consistent (end date after start date, dates in the past marked appropriately). After extracting monetary amounts, a validation step might check that line items sum to the stated total.

### Risks and Mitigations

**Partial extraction** occurs when the model misses fields that are present in the text. This is common for fields that appear in unusual formats, in unexpected locations, or that require inference rather than direct extraction. Mitigate with a comprehensive test set that includes edge cases and with few-shot examples in the prompt that demonstrate tricky extraction scenarios.

**Format drift** happens when model updates or prompt changes cause subtle shifts in how the model formats extracted data. A date that was previously extracted as "2026-02-10" might suddenly appear as "February 10, 2026." Schema validation catches type errors, but format drift within a valid type can break downstream systems. Use strict format specifications in your schema and test for format consistency.

**Silent mis-parses** are the most dangerous failure mode: the model extracts a value that is structurally valid but factually wrong. An amount of "$1,234" might be extracted as "$12,34" or "$1234." A name might be swapped with another name in the same document. These errors pass schema validation and can only be caught by correctness evaluation against ground truth.

### Checklist
- Strict JSON schema validation on every output
- Test set covering tricky formats, edge cases, and known failure modes
- Deterministic model settings (temperature zero or near-zero) when consistency matters
- Retry logic with informative error messages for schema validation failures
- Format consistency tests beyond basic type checking
- Confidence scoring or abstention for low-confidence extractions

## Workflow Automation (Agent)

Workflow automation uses an agent to accomplish multi-step tasks by interacting with external systems. This is the most powerful and most dangerous recipe because it involves taking real-world actions. Examples include automated incident response (read alert, check logs, apply fix, notify team), automated data processing (fetch data, transform, validate, load), and automated project management (read requirements, create tickets, assign, track progress).

### Data Flow

The agent receives a goal (either from a user request or a triggered event), formulates a plan (either explicitly as a plan-and-execute architecture or implicitly as ReAct reasoning), and then iterates through a loop of tool calls and observations. Each tool call interacts with an external system (an API, a database, a file system, a messaging platform), and the result feeds back into the agent's decision-making for the next step.

The critical data flow consideration is the feedback loop between the agent and its environment. Unlike a chat assistant where data flows in one direction (user question to model answer), an agent's actions change the environment, which changes the observations, which change the agent's decisions. This feedback loop can amplify errors: a wrong action produces a confusing observation, which leads to another wrong action.

### When To Combine With Other Patterns

Agent workflows almost always incorporate other patterns. RAG provides the agent with access to documentation and knowledge bases. Extraction structures the outputs of tool calls into forms the agent can reason about. Structured outputs ensure that the agent's tool calls have valid parameters. Chat interfaces let users interact with the agent, provide guidance, and approve actions.

A common pattern is a "supervised agent" that combines full automation for low-risk steps with human approval for high-risk ones. The agent proceeds autonomously through read operations and analysis, then pauses for human review before executing write operations or external communications.

### Risks and Mitigations

**Unintended actions** are the primary risk. An agent that misunderstands the goal, misinterprets a tool's purpose, or follows a hallucinated plan can take actions that are difficult or impossible to reverse. Mitigate with the principle of least privilege (limit what each tool can do), idempotency (design tools so that repeating an action is safe), and reversibility (ensure that actions can be undone where possible).

**Tool misuse** occurs when the agent calls a tool with incorrect parameters or in an inappropriate context. A deployment tool called with the wrong environment parameter could push code to production instead of staging. A deletion tool called with a wildcard path could remove far more than intended. Mitigate with strict parameter validation, confirmation prompts for destructive operations, and dry-run modes.

**Infinite loops** happen when the agent cannot make progress but does not recognize the impasse. It keeps trying variations of the same approach, consuming budget without moving toward the goal. Mitigate with explicit loop detection (track action history and detect repetition), progress metrics (has anything meaningful changed in the last N steps?), and hard budget limits (stop after N tool calls regardless of progress).

### Checklist
- Least-privilege tools with explicit permission boundaries
- Idempotency and rollback capability for all write operations
- Stop conditions and loop detection with graceful termination
- Human-in-the-loop checkpoints for high-risk or irreversible actions
- Comprehensive audit logging of all tool calls and their results
- Budget limits (tokens, tool calls, wall-clock time, cost) with clear behavior when exhausted

## Combining Recipes

Most production systems are not pure instances of a single recipe. A customer support system might combine chat (for the conversational interface) with RAG (for knowledge base grounding) with extraction (for parsing customer information from the conversation) with workflow automation (for filing tickets and updating account records).

When combining recipes, pay attention to the interfaces between them. The output of one stage is the input to the next, and errors propagate across boundaries. A retrieval failure in the RAG component might cause the chat component to hallucinate, which might cause the extraction component to produce wrong structured data, which might cause the automation component to take a wrong action. Each stage should validate its inputs and handle errors from upstream stages gracefully.

Design for observability across the full pipeline. When a user reports a problem, you need to be able to trace the issue back through every stage to find the root cause. This requires consistent request identifiers, structured logging at each stage, and tooling that lets you reconstruct the full processing chain for any given request.

## References
- OpenAI docs: Production best practices. https://platform.openai.com/docs/guides/production-best-practices
- OpenAI docs: Safety best practices. https://platform.openai.com/docs/guides/safety-best-practices

---
[Contents](README.md) | [Prev](06-safety-privacy-security.md) | [Next](08-ops.md)
