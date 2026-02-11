# Governance And Risk

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](08-ops.md) | [Next](13-staying-current.md)

## Summary
Shipping AI safely is mostly about governance: knowing what system you built, what it can do, how it can fail, and how you control and audit it. Governance is not bureaucracy for its own sake. It is the set of practices that let an organization deploy AI systems with confidence, respond effectively when things go wrong, and demonstrate to stakeholders (users, regulators, executives, auditors) that appropriate controls are in place. Good governance makes teams faster, not slower, because it replaces ad hoc decision-making with clear processes and shared understanding.

## See Also
- [Safety, Privacy, And Security](06-safety-privacy-security.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Evals And Testing](05-evals.md)

## Risk Areas

Understanding where risk concentrates in LLM systems is the prerequisite for managing it. Risk in AI systems is not fundamentally different from risk in other software, but the probability distributions are unusual: failures are harder to predict, harder to detect, and harder to bound.

### Safety

Safety risk encompasses harmful content generation, policy violations, and unsafe instructions. A customer-facing chatbot that generates offensive content, a medical assistant that provides dangerous advice, or an automation agent that takes destructive actions all represent safety failures.

The distinctive challenge of safety risk in LLM systems is that failures are contextual and probabilistic. The same model that handles 99.9 percent of queries safely can produce harmful output on a rare query that happens to exploit a gap in its training or alignment. Safety risk cannot be eliminated through testing alone; it requires layered defenses (system prompt constraints, output filtering, human review for high-risk domains) and ongoing monitoring.

### Privacy

Privacy risk involves the exposure of personal, confidential, or regulated information through model outputs, logs, or the training pipelines of third-party providers. LLM systems are particularly prone to privacy risk because they process natural language, which frequently contains PII, and because their outputs can surface information from their context in unexpected ways.

Privacy risk extends beyond the model itself. Conversation histories stored for debugging may contain sensitive user disclosures. Retrieved documents may contain confidential information that the user should not access. Logs shipped to monitoring services may contain PII that was not redacted. Each data flow in the system is a potential privacy exposure point.

### Security

Security risk in LLM systems includes prompt injection (manipulating model behavior through crafted inputs), data exfiltration (using the model as a channel to extract sensitive information), and tool abuse (leveraging an agent's capabilities for unintended purposes). These risks are covered in detail in the [Safety, Privacy, And Security](06-safety-privacy-security.md) chapter.

The key governance concern with security risk is that the attack surface is novel and evolving. Traditional security teams may not be familiar with prompt injection or indirect injection via retrieved documents. Ensure that your security review process includes LLM-specific threat modeling.

### Reliability

Reliability risk manifests as hallucinations (confidently wrong answers), brittle output formats (responses that sometimes fail to parse), and regressions after model or prompt updates. Unlike safety or security failures, which are occasional and often dramatic, reliability failures are chronic and insidious. A system that is 95 percent accurate sounds good until you realize that one in twenty answers is wrong, and users cannot tell which ones.

Reliability governance focuses on evaluation practices (measuring quality systematically), change management (testing changes before deployment), and monitoring (detecting degradation in production). These practices are covered in the [Evals And Testing](05-evals.md) and [Ops](08-ops.md) chapters, but governance ensures that they are actually followed consistently rather than being aspirational best practices that get skipped under time pressure.

### Compliance

Compliance risk varies dramatically by organization and jurisdiction. Healthcare organizations face [HIPAA](https://www.hhs.gov/hipaa/index.html) requirements for PHI. Financial institutions face regulations around automated decision-making. Organizations operating in the EU face the [AI Act](https://artificialintelligenceact.eu/)'s requirements for high-risk AI systems. Consumer-facing applications face data protection regulations ([GDPR](https://gdpr.eu/), [CCPA](https://oag.ca.gov/privacy/ccpa), and their counterparts worldwide).

The governance challenge is that compliance requirements for AI are still evolving, and many regulations were written before LLM systems existed. This means that compliance teams may not have clear guidance on how to apply existing regulations to LLM systems, and the organization must make reasonable interpretations that may be tested later. Document these interpretations and the reasoning behind them so that you can demonstrate good faith if questions arise.

## Practical Governance Moves

### Document The System

Documentation is the foundation of governance. You cannot manage what you have not described. A well-documented AI system has a clear record of what it is, what it does, and how it is controlled.

**System inventory.** Maintain a catalog of every AI-powered feature in your organization, including the model and provider, the prompt versions in use, the tools and permissions available, the data sources (retrieval corpora, databases, APIs), and the intended use case and user population. This inventory should be a living document updated as part of every deployment.

**[Model cards](https://arxiv.org/abs/1810.03993).** For each model deployment, create a model card that describes the model's capabilities and limitations, the tasks it is used for, the data it has access to, known failure modes, and any restrictions on its use. Model cards were originally proposed for ML models in general, but they are equally valuable for LLM deployments. They do not need to be elaborate; a one-page summary that captures the essentials is far better than nothing.

**Data classification and handling.** Document what data enters the system (inputs, retrieved documents, user profiles), what data the system produces (outputs, logs, artifacts), and the classification level of each data type. Map each data flow against your data handling policies to identify gaps. See the [Safety, Privacy, And Security](06-safety-privacy-security.md) chapter for a data classification framework.

**Human-in-the-loop specification.** For systems that involve human oversight, document exactly when human review is required, what information the human reviewer sees, what decisions they are empowered to make, and how their decisions are recorded. Vague statements like "a human reviews high-risk outputs" are not governance; specific procedures like "outputs flagged by the safety classifier are queued for review by a member of the Trust & Safety team within 4 hours" are.

### Measure And Monitor

Governance without measurement is policy without enforcement. The measurement practices described in [Evals And Testing](05-evals.md) and [Ops](08-ops.md) are governance tools as much as they are engineering tools.

**Offline evaluations** (golden sets) measure system quality under controlled conditions before changes are deployed. Governance requires that evals are run on every significant change (prompt, model, index, tool) and that results are recorded. This creates an audit trail that demonstrates the organization tested changes before deploying them.

**Online monitoring** (production metrics, user feedback, safety signals) measures system quality under real conditions after deployment. Governance requires that monitoring dashboards are reviewed regularly (not just when something breaks), that anomalies are investigated, and that findings are documented.

**Incident tracking.** Every safety, privacy, security, or significant quality incident should be tracked in a formal system (not just Slack threads). Track the incident timeline, root cause, impact, remediation, and preventive measures. This record is essential for demonstrating that the organization learns from failures and improves its controls over time.

### Control Change

Change management is where governance has the most direct impact on day-to-day operations. The goal is not to slow things down but to ensure that changes are intentional, tested, and reversible.

**Release process.** Define an explicit release process for each type of change: prompt updates, model version changes, retrieval index rebuilds, tool modifications, and policy configuration changes. The process should include review (who approves the change), testing (what evals must pass), deployment (how the change is rolled out), and validation (how you confirm the change is working as expected in production).

**Rollback plan.** Every release should have a rollback plan that can be executed quickly if the change causes problems. This requires maintaining previous versions of all components and having the infrastructure to switch between them. See the [Ops](08-ops.md) chapter for details on rollback procedures.

**Failure reproduction.** When a problem is discovered (through monitoring, user reports, or incident response), the team must be able to reproduce it. This requires that the system logs enough information to reconstruct the inputs, configuration, and context that produced the problematic output. Sanitize artifacts for privacy, but preserve enough detail for diagnosis. Reproduced failures become regression test cases that prevent recurrence.

### Periodic Review

Governance is not a one-time setup; it is an ongoing practice. Schedule periodic reviews (quarterly is a common cadence) to assess whether governance practices are being followed, whether they are effective, and whether they need to be updated.

Review topics should include: eval coverage (are there important failure modes not covered by the eval suite?), incident trends (are the same types of incidents recurring?), compliance updates (have regulations or organizational policies changed?), and system evolution (have new features or data sources been added without corresponding governance updates?).

## Risk Assessment Framework

For organizations that want a structured approach to risk assessment, a simple framework can be built around three dimensions: likelihood of failure, severity of consequences, and detectability of the failure.

**Likelihood** is how often the failure mode occurs. An LLM that hallucinates on 5 percent of queries has a higher likelihood than one that hallucinates on 0.1 percent. Likelihood is measured through evaluation and production monitoring.

**Severity** is the consequence of the failure. A chatbot that gives a mildly unhelpful answer has low severity. A medical advisor that gives dangerous advice has extreme severity. Severity is determined by domain analysis and stakeholder input.

**Detectability** is how quickly and reliably the failure is noticed. A schema validation failure is immediately detected. A subtle factual error might not be detected until a user acts on bad information. Detectability is improved through monitoring, automated quality checks, and user feedback mechanisms.

Risk priority is determined by the combination of all three: high-likelihood, high-severity, low-detectability failures are the most dangerous and deserve the most investment in mitigation and monitoring.

## Checklist
- Do you have an inventory of all AI-powered features, their models, tools, and data sources?
- Do you have model cards or equivalent documentation for each deployment?
- Do you have a documented data handling policy covering inputs, outputs, logs, and retention?
- Do you run evals on every prompt, model, and index change with recorded results?
- Do you have a defined release process with rollback capability?
- Do you have a formal incident tracking system with root cause analysis?
- Do you conduct periodic governance reviews?
- Are human-in-the-loop procedures specified concretely, not just aspirationally?
- Can you reproduce any flagged output from logged artifacts?

## References
- [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST Generative AI Profile (GenAI risk guidance)](https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenAI usage policies](https://openai.com/policies/usage-policies/)
- [OpenAI "Your data" guide (data handling controls)](https://platform.openai.com/docs/guides/your-data)
- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

---
[Contents](README.md) | [Prev](08-ops.md) | [Next](13-staying-current.md)
