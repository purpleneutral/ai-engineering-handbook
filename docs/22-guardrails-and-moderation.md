# Guardrails And Content Moderation

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](21-cost-engineering.md) | [Next](24-prompt-management.md)

## Summary

Guardrails are validation layers that inspect LLM inputs and outputs to enforce safety, quality, and policy constraints. They are not optional for production systems. A model that generates toxic content, leaks PII, follows a jailbreak, or hallucinates a medical diagnosis is not a quality issue --- it is an incident. This chapter covers the practical tools and patterns for preventing those incidents: moderation APIs, guardrail frameworks, PII detection pipelines, and the layered architecture that ties them together.

## See Also
- [Safety, Privacy, And Security](06-safety-privacy-security.md) --- The broader security framework, including prompt injection, supply chain attacks, and agent security.
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md) --- Schema validation as an output guardrail.
- [Evals And Testing](05-evals.md) --- Evaluation as a quality guardrail during development.
- [Governance And Risk](14-governance-and-risk.md) --- Organizational policies that guardrails enforce.

## Moderation APIs

Moderation APIs are the simplest guardrail: send text (or images) to a classification endpoint and get back category scores. They are fast, cheap (often free), and catch the most obvious violations.

### OpenAI Moderation API

The [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) is free to use and classifies text across 11 categories: harassment, hate, sexual content, violence, self-harm, and their subcategories. The latest model (`omni-moderation-latest`) also accepts images.

```python
from openai import OpenAI
client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="Some text to check",
)

result = response.results[0]
if result.flagged:
    print("Blocked categories:", [
        cat for cat, flagged in result.categories.__dict__.items()
        if flagged
    ])
# Each category also has a float score (0-1) in result.category_scores
```

The `category_scores` are floats between 0 and 1 that allow you to set custom thresholds rather than relying on the binary `flagged` field. This is important because the default thresholds may be too aggressive or too lenient for your use case.

### Azure AI Content Safety

[Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview) provides severity-scored moderation across four categories (hate, sexual, violence, self-harm) on a 0--7 scale. It also includes specialized capabilities: **prompt shield** for detecting jailbreak and injection attempts, **groundedness detection** for hallucination checking, and **protected material detection** for copyrighted content.

### Google Safety Settings

[Gemini safety settings](https://ai.google.dev/gemini-api/docs/safety-settings) are configured per-request via the API. You set a threshold for each harm category (harassment, hate speech, sexually explicit, dangerous content) to `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, or `BLOCK_HIGH_ONLY`.

### Anthropic Content Moderation

Anthropic provides [guidance on using Claude for content moderation](https://docs.anthropic.com/en/docs/about-claude/use-case-guides/content-moderation), using the model itself as a classifier with structured output. Their [constitutional classifiers](https://www.anthropic.com/research/constitutional-classifiers) research demonstrates a two-stage architecture: a lightweight activation probe screens most inputs quickly, and a full classifier evaluates only suspicious ones.

### When to Use Moderation APIs

Moderation APIs are best suited as a fast first-pass filter. They are not sufficient on their own --- they miss subtle policy violations, domain-specific concerns, and context-dependent content --- but they catch the obvious cases cheaply and quickly. Use them as the first layer in a multi-layer defense.

## Guardrail Frameworks

For more sophisticated guardrails, several open-source and commercial frameworks provide composable validation pipelines.

### NVIDIA NeMo Guardrails

[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) is an event-driven framework that uses [Colang](https://docs.nvidia.com/nemo/guardrails/colang_2/overview.html), a domain-specific language for defining guardrail logic. It supports five rail types: input rails (applied to user input before the model), output rails (applied to model responses before the user), dialog rails (influence how the LLM is prompted), retrieval rails (applied to retrieved chunks in RAG), and execution rails (applied to tool execution).

```yaml
# config.yml
models:
  - type: main
    engine: openai
    model: gpt-4o

rails:
  input:
    flows:
      - self check input
      - check jailbreak
  output:
    flows:
      - self check output
      - self check facts
```

NeMo Guardrails is well-suited for conversational systems where you need fine-grained control over dialog flow, topic boundaries, and factuality checking.

### Guardrails AI

[Guardrails AI](https://github.com/guardrails-ai/guardrails) uses composable **validators** from a community [Hub](https://hub.guardrailsai.com/) of 100+ pre-built checks. Each validator targets a specific risk (toxicity, PII, competitor mentions, regex compliance, valid choices) and is configured with a failure action: raise an exception, attempt automatic correction, ask the LLM to regenerate, or log and pass through.

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, CompetitorCheck

guard = Guard().use_many(
    CompetitorCheck(["Apple", "Microsoft"], on_fail=OnFailAction.EXCEPTION),
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.FIX),
)

result = guard.validate("The product is better than Apple's offering.")
# Raises exception due to competitor mention
```

Guardrails AI also supports streaming validation (checking LLM responses in real-time as they stream) and Pydantic model integration for type-safe structured output validation.

### LLM Guard (Protect AI)

[LLM Guard](https://github.com/protectai/llm-guard) provides 15 input scanners and 20 output scanners covering PII anonymization, code detection, prompt injection, toxicity, gibberish detection, URL safety, and more. It can be deployed as a standalone API server or integrated directly into Python applications.

Notable scanners include `Anonymize`/`Deanonymize` (reversible PII replacement), `PromptInjection` (fine-tuned classifier), `FactualConsistency` (NLI-based hallucination detection), and `MaliciousURLs` (URL safety checking in model outputs).

### Lakera Guard

[Lakera Guard](https://www.lakera.ai/lakera-guard) is a commercial API focused on prompt injection and jailbreak detection. It analyzes 100,000+ new attacks daily via its [Gandalf](https://gandalf.lakera.ai/) research platform, which provides continuously updated threat intelligence.

### Choosing an Approach

| Dimension | Rule-Based | ML Classifier | LLM-as-Judge |
|-----------|-----------|---------------|--------------|
| Latency | Microseconds | 10--100ms | 1--5 seconds |
| Coverage | Low; brittle against novel attacks | Medium; captures patterns | High; understands context and nuance |
| Adaptability | Low; manual rule updates | Medium; requires retraining | High; policy changes via prompt |
| Cost | Negligible | Low--moderate | High (full LLM API call) |
| Bypass resistance | Low; keyword obfuscation defeats them | Moderate | Higher, but can be jailbroken itself |

The practical recommendation is to use a **layered approach**: fast, cheap checks first (regex, blocklists), ML classifiers for patterned violations, and LLM-as-judge for high-stakes or ambiguous cases.

## PII Detection And Redaction

PII in LLM prompts, outputs, and logs is both a privacy risk and a compliance liability. A systematic detection-and-redaction pipeline prevents sensitive data from reaching the model or persisting in logs.

### Microsoft Presidio

[Presidio](https://microsoft.github.io/presidio/) is the most mature open-source PII detection framework. It has two components: an **analyzer** that identifies PII entities by type and position (using regex, NER models, and checksum validation), and an **anonymizer** that replaces identified PII with configurable transformations (redact, replace, mask, hash, or encrypt).

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

results = analyzer.analyze(
    text="My SSN is 123-45-6789 and my email is jane@example.com",
    language="en",
)
anonymized = anonymizer.anonymize(text="My SSN is 123-45-6789 and my email is jane@example.com", analyzer_results=results)
print(anonymized.text)  # "My SSN is <US_SSN> and my email is <EMAIL_ADDRESS>"
```

Built-in entity types include names, phone numbers, email addresses, credit card numbers, SSNs, passport numbers, IP addresses, and more. Custom recognizers can be added for domain-specific PII types.

### Cloud PII Services

**[AWS Comprehend](https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html)** detects and redacts PII in English and Spanish text across 22 universal and 14 country-specific entity types. Available as a real-time API or async batch job.

**[Google Sensitive Data Protection](https://cloud.google.com/sensitive-data-protection/docs)** (formerly DLP API) provides 120+ built-in infoType detectors with de-identification techniques including masking, tokenization, and format-preserving encryption. Supports reversible transformations for authorized re-identification.

### The Redaction Pipeline

The standard pattern for PII-safe LLM processing:

```
User Input
    │
    ▼
[1. DETECT]  ── Presidio/Comprehend/DLP identifies PII with positions and types
    │
    ▼
[2. REDACT]  ── Replace PII with placeholders; store mapping securely
    │            {"<PERSON_1>": "Jane Smith", "<SSN_1>": "123-45-6789"}
    ▼
[3. PROCESS] ── Send redacted text to LLM; model never sees actual PII
    │            Response contains placeholders: "Dear <PERSON_1>, your account..."
    ▼
[4. RE-ID]   ── Replace placeholders with original values (if authorized)
    │
    ▼
Final Response
```

**Key considerations:**
- Store the mapping securely with a short TTL. Do not persist it unnecessarily.
- LLMs can sometimes infer or hallucinate PII in outputs even from redacted inputs. Always scan outputs too.
- For irreversible use cases (logging, analytics), use one-way hashing or masking instead of reversible tokenization.
- Run PII detection on both inputs and outputs (belt and suspenders).

## Input Guardrails

Input guardrails inspect user messages before they reach the model. They are the first line of defense and should be fast enough to add negligible latency.

### Topic Restriction

Define allowed or denied topics and classify each input against the list. Implementations range from keyword matching (fast but brittle) to LLM-as-classifier (accurate but slower). [AWS Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) supports up to 30 denied topics per guardrail, evaluated in natural language. NeMo Guardrails uses Colang flows to define topical boundaries.

### Jailbreak and Injection Detection

Multiple layers reduce the risk:

1. **Fine-tuned classifiers.** Small BERT-based models trained on known jailbreak datasets. Fast (10--100ms) and effective against known patterns. Used by LLM Guard and Azure AI Content Safety.
2. **Prompt shields.** Azure's [Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection) API distinguishes trusted from untrusted inputs using the spotlighting technique.
3. **Continuously updated detection.** Lakera Guard updates its detection models from 100,000+ daily attack samples.
4. **Canary tokens.** Embed hidden markers in system prompts. If they appear in the output, injection has likely occurred.

### Input Length and Rate Limits

Set `max_tokens` or character limits appropriate to your use case. Long inputs increase cost, latency, and injection surface area. Rate limits per user, per IP, or per API key serve both cost control and security (preventing model extraction and DoS).

## Output Guardrails

Output guardrails inspect model responses before they reach the user. They are the last line of defense.

### Toxicity Filtering

Run model output through a toxicity classifier. The OpenAI Moderation API (free), LLM Guard's `Toxicity` scanner, Guardrails AI's `ToxicLanguage` validator, and Azure AI Content Safety all serve this purpose.

### Factuality and Groundedness

For RAG systems, verify that the model's response is grounded in the retrieved source material. Azure AI Content Safety provides a [groundedness detection](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness) API that checks responses against source documents and can auto-correct ungrounded claims. LLM Guard's `FactualConsistency` scanner uses NLI (natural language inference) models for the same purpose. See also [Evals and Testing](05-evals.md) for faithfulness evaluation patterns.

### Schema Validation

For structured outputs, validate against the expected schema before passing results downstream. This catches format drift, missing fields, and type errors. OpenAI's [structured outputs](11-structured-outputs-and-tool-calling.md) enforce JSON schema at the API level. Guardrails AI provides native Pydantic model integration with field-level validators.

### PII in Outputs

Scan model outputs for PII even when inputs were clean. The model may hallucinate or recall PII from training data. Run the same PII detection pipeline on outputs as you run on inputs.

## Policy Enforcement Architecture

### The Layered Pipeline

Guardrails should be layered, with fast and cheap checks running first and expensive checks running only when earlier layers pass:

```
User Request
    │
    ▼
[Layer 1: Fast]  ── Regex, blocklists, length limits, rate limits
    │                 Microsecond latency. Catches obvious violations.
    ▼
[Layer 2: ML]    ── Classifiers for toxicity, injection, topic
    │                 10-100ms latency. Catches patterned violations.
    ▼
[Layer 3: LLM]   ── LLM-as-judge for nuanced policy compliance
    │                 1-5 seconds latency. For high-stakes decisions only.
    ▼
[Model Inference]
    │
    ▼
[Output Rails]   ── Toxicity, PII, factuality, schema validation
    │
    ▼
User Response
```

### Performance Budget

For interactive applications, total guardrail overhead should stay under ~200ms to avoid perceptible degradation. This means Layer 1 and Layer 2 for most requests, with Layer 3 reserved for flagged or high-risk inputs. Batch and offline workloads can afford more thorough (and slower) checking.

### False Positive Management

Guardrails that block too aggressively are as harmful as guardrails that miss violations. Users who encounter frequent false positives will lose trust in the system or find workarounds.

**Configurable thresholds.** Most guardrail systems expose confidence scores. Tune thresholds per use case: a children's education app needs stricter thresholds than an internal developer tool.

**Tiered responses.** Low-confidence flags trigger logging and monitoring. Medium-confidence flags trigger human review. High-confidence flags trigger blocking. This prevents binary all-or-nothing enforcement.

**Failure actions.** Guardrails AI's `OnFailAction` pattern is instructive: `EXCEPTION` (hard block), `FIX` (attempt automatic correction), `REASK` (ask the LLM to regenerate), or `NOOP` (log but pass through). Different validators warrant different failure actions.

### Appeals and Overrides

For user-facing systems, provide a path for legitimate content that was incorrectly blocked. Log all override decisions with rationale for compliance. Track false positive rates and use them to retune classifiers over time.

## Emerging Standards

### OWASP

The [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) includes several entries directly relevant to guardrails: LLM01 (Prompt Injection), LLM02 (Sensitive Information Disclosure), LLM05 (Improper Output Handling), and LLM09 (Misinformation). OWASP recommends guardrails external to the LLM itself --- an independent system that inspects inputs and outputs for compliance.

The [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) (released December 2025) introduces the principle of **least agency**: grant agents only the minimum autonomy required for safe, bounded tasks. It covers agent-specific risks including goal hijacking, memory poisoning, cascading failures, and rogue agent behavior.

### NIST

The [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework) provides a voluntary framework emphasizing transparency, human oversight, bias mitigation, testing, and accountability. [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) extends this to generative AI-specific risks. Both frameworks recommend guardrails and human-in-the-loop controls as core risk mitigations.

## Pitfalls

**Relying on a single guardrail layer.** No single check catches everything. Regex misses obfuscated attacks. Classifiers degrade on novel prompts. LLM-as-judge can be jailbroken. Use layered defense.

**Applying the same thresholds to every use case.** A medical Q&A system needs stricter guardrails than an internal brainstorming tool. Tune thresholds to match the risk profile of each application.

**Ignoring guardrail latency in UX.** A 3-second output guardrail check on every response will degrade user experience. Budget guardrail latency as part of your total response time target.

**Treating guardrails as a substitute for good system design.** Guardrails are a safety net, not a foundation. A well-designed system prompt, proper instruction-data separation, and careful tool permissions prevent most issues. Guardrails catch what slips through.

**Not monitoring guardrail effectiveness.** Track block rates, false positive rates, and the types of violations caught. If your guardrails never fire, they may not be working. If they fire too often, your thresholds may be too aggressive or your system design may have deeper issues.

**Skipping output guardrails because input guardrails are in place.** Input guardrails protect against malicious users. Output guardrails protect against model misbehavior, hallucination, and PII leakage --- problems that occur even with perfectly clean inputs.

## Checklist
- Do you have input guardrails that check for injection, jailbreak, and policy violations before the model sees the input?
- Do you have output guardrails that check for toxicity, PII, and factuality before the response reaches the user?
- Are guardrail thresholds tuned to your specific use case, not left at defaults?
- Is the total guardrail latency within your response time budget?
- Do you track false positive and false negative rates for your guardrails?
- Is PII detected and redacted on both inputs and outputs?
- For RAG systems, do you validate groundedness of model responses against source material?
- Do you have a path for appealing incorrectly blocked content?
- Are guardrail configurations versioned and testable?
- Do you test guardrails with adversarial inputs, not just benign ones?

## References

### Moderation APIs
- OpenAI Moderation API. https://platform.openai.com/docs/guides/moderation
- Azure AI Content Safety. https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview
- Google Gemini Safety Settings. https://ai.google.dev/gemini-api/docs/safety-settings
- Anthropic Content Moderation Guide. https://docs.anthropic.com/en/docs/about-claude/use-case-guides/content-moderation

### Frameworks
- NVIDIA NeMo Guardrails. https://github.com/NVIDIA/NeMo-Guardrails
- NeMo Guardrails Colang 2.0. https://docs.nvidia.com/nemo/guardrails/colang_2/overview.html
- Guardrails AI. https://github.com/guardrails-ai/guardrails
- Guardrails AI Hub. https://hub.guardrailsai.com/
- LLM Guard (Protect AI). https://github.com/protectai/llm-guard
- Lakera Guard. https://www.lakera.ai/lakera-guard

### PII Detection
- Microsoft Presidio. https://microsoft.github.io/presidio/
- AWS Comprehend PII Detection. https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html
- Google Sensitive Data Protection. https://cloud.google.com/sensitive-data-protection/docs

### Standards
- OWASP Top 10 for LLM Applications 2025. https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/
- OWASP Top 10 for Agentic Applications 2026. https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
- NIST AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI 600-1 (Generative AI Profile). https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- AWS Bedrock Guardrails. https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](21-cost-engineering.md) | [Next](24-prompt-management.md)
