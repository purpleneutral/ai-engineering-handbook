# Legal And Intellectual Property

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](29-migration-and-vendor-strategy.md) | [Next](09-glossary.md)

> **This chapter is educational content, not legal advice.** The legal landscape around AI is evolving rapidly, with active litigation, pending legislation, and regulatory guidance that may change materially between the time this is written and the time you read it. Nothing here should be treated as a substitute for qualified legal counsel familiar with your jurisdiction, your industry, and the specific facts of your situation. Consult a lawyer before making decisions that carry legal risk.

## Summary

Teams building and deploying AI features operate in a legal environment that is unsettled, jurisdictionally fragmented, and changing fast. This chapter maps the key areas -- copyright, licensing, data residency, liability, and regulation -- so that you can spot issues early, ask the right questions, and build systems that are easier to defend legally. It is not a compliance manual; it is a field guide for anyone involved in AI products --- engineers, product managers, and business leaders alike --- who needs to understand the terrain.

## See Also
- [Governance And Risk](14-governance-and-risk.md)
- [Safety, Privacy, And Security](06-safety-privacy-security.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Scope And Update Policy](00-scope-and-update-policy.md)

## When To Use

Read this chapter when you are evaluating whether to use AI-generated content in a product, choosing between open-weight and proprietary models, processing data subject to privacy regulations, designing audit trails for AI-assisted workflows, or trying to understand what your obligations are under emerging AI regulation. This chapter will not give you answers -- those depend on your facts and your jurisdiction -- but it will give you the vocabulary and framework to have productive conversations with your legal team.

## Copyright For AI-Generated Content

The central question is deceptively simple: can AI-generated content be copyrighted? The answer, as of 2026-02-10, is that pure AI-generated output without meaningful human authorship generally cannot receive copyright protection in the United States, but the boundaries of "meaningful human authorship" remain actively contested.

### The Human Authorship Requirement

The [U.S. Copyright Office](https://www.copyright.gov/) has consistently held that copyright requires human authorship. This position was articulated in a series of decisions and guidance documents beginning in 2022:

The Copyright Office refused registration for an AI-generated image called "A Recent Entrance to Paradise" created by the [DABUS](https://artificialinventor.com/) system. Stephen Thaler, the creator of DABUS, challenged this refusal in [*Thaler v. Perlmutter*](https://www.copyright.gov/ai/Thaler%20v%20Perlmutter.pdf), where the U.S. District Court for the District of Columbia ruled in August 2023 that works generated autonomously by AI without human creative input are not eligible for copyright protection. The court affirmed the Copyright Office's longstanding requirement of human authorship.

In February 2023, the Copyright Office addressed a more nuanced case: Kris Kashtanova's graphic novel *Zarya of the Dawn*, which used Midjourney to generate illustrations while the author arranged them and wrote the text. The Office [granted copyright for the text and the selection and arrangement of images](https://www.copyright.gov/docs/zarya-of-the-dawn.pdf) but denied protection for the individual AI-generated images themselves. This case is important because it established that works containing AI-generated elements can receive partial copyright protection for the human-authored portions.

In August 2023, the Copyright Office issued a [Federal Register notice](https://www.federalregister.gov/documents/2023/08/30/2023-18624/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence) formalizing its guidance: applicants must disclose AI-generated content in registration applications, and copyright protection extends only to the human-authored elements of a work.

### What This Means In Practice

As of 2026-02-10, the practical implications are:

**Code generated entirely by an LLM** (where you provide a brief prompt and use the output verbatim) likely has weak or no copyright protection. This does not mean you cannot use it -- it means you may not be able to prevent others from using identical code, because it may be in the public domain.

**Code you write with AI assistance** (where you provide substantial direction, review, modify, select among alternatives, and integrate into a larger work) likely retains copyright protection for your creative contributions. The more human judgment and creative decision-making involved, the stronger the copyright claim.

**The disclosure obligation matters.** If you register a copyright for a work that contains AI-generated material and fail to disclose the AI's contribution, the registration may be invalidated. This applies to any copyrightable work, including software.

The situation outside the United States varies. As of 2026-02-10, the [UK Copyright, Designs and Patents Act 1988](https://www.legislation.gov.uk/ukpga/1988/48/section/9) contains a provision (Section 9(3)) that assigns authorship of computer-generated works to the person who made the arrangements necessary for the creation of the work, which could potentially cover AI-generated output. The EU has not established a unified position, and member states differ in their approaches.

## Training Data Copyright

The question of whether training AI models on copyrighted material constitutes infringement is the subject of major ongoing litigation. The outcome will shape the economics and availability of AI systems for years.

### Key Litigation

As of 2026-02-10, several landmark cases are working through the courts:

[*The New York Times Co. v. Microsoft Corp. et al.*](https://nytco-assets.nytimes.com/2023/12/NYT_Complaint_Dec2023.pdf) was filed in December 2023 in the Southern District of New York. The Times alleges that OpenAI and Microsoft trained models on millions of Times articles without authorization and that the models can reproduce near-verbatim excerpts of copyrighted content. This case is significant because it directly challenges the fair use argument for training data and involves a well-resourced plaintiff with strong copyright claims.

[*Andersen v. Stability AI Ltd.*](https://stablediffusionlitigation.com/) is a class action filed by visual artists against Stability AI, Midjourney, and DeviantArt, alleging that text-to-image models were trained on copyrighted artwork without consent. The case raises questions about whether the output of image generation models constitutes a derivative work.

[*Thomson Reuters Enterprise Centre GmbH v. ROSS Intelligence Inc.*](https://law.justia.com/cases/federal/district-courts/delaware/dedce/1:2020cv00613/73618/244/) resulted in a February 2025 ruling from the District of Delaware that denied ROSS Intelligence's fair use defense for training a legal AI system on Westlaw headnotes. The court found that the use was commercial, the headnotes were creative works, and the AI system competed directly with the original. As of 2026-02-10, this is one of the few cases where a court has ruled against fair use in an AI training context.

### The Fair Use Argument

Defendants in training data cases generally argue that training is [fair use](https://www.copyright.gov/fair-use/) under U.S. copyright law, relying on the four-factor test in [17 U.S.C. Section 107](https://www.law.cornell.edu/uscode/text/17/107). The argument is that training is "transformative" because the model learns statistical patterns rather than copying the works, similar to how a search engine index was found to be fair use in [*Authors Guild v. Google*](https://law.justia.com/cases/federal/appellate-courts/ca2/13-4829/13-4829-2015-10-16.html). Whether courts will accept this analogy for generative models -- which can produce output that competes with the original works -- remains an open question that the current litigation will resolve.

### Opt-Out Mechanisms

In the absence of clear legal resolution, a practical ecosystem of opt-out mechanisms has emerged. The [robots.txt](https://www.robotstxt.org/) protocol, originally designed for search engine crawlers, has been extended by major AI companies to support AI-specific directives (for example, `User-agent: GPTBot` for OpenAI's crawler). Additionally, some organizations have adopted [ai.txt](https://site.spawning.ai/spawning-ai-txt) proposals to specify preferences for AI training use.

As of 2026-02-10, these mechanisms are voluntary and not legally binding. Their effectiveness depends entirely on whether AI companies choose to honor them. OpenAI has stated that it respects `robots.txt` directives for GPTBot. Not all providers make the same commitment.

The practical takeaway: if you are building systems that crawl or ingest content, respect opt-out signals. If you are fine-tuning models on collected data, document the provenance and licensing of your training data. The legal standards are unclear, but demonstrating good faith and respect for content creators' preferences is both ethically sound and strategically prudent.

## Terms Of Service For LLM APIs

When you use a hosted LLM API, the terms of service govern critical questions: who owns the outputs, whether the provider can train on your inputs, and how your data is retained. These terms differ materially between providers and have changed over time.

### Ownership Of Outputs

As of 2026-02-10, the major providers generally assign output ownership to the customer:

**OpenAI** states in its [Terms of Use](https://openai.com/policies/terms-of-use/) that the user owns the output generated by the API, subject to compliance with the terms. OpenAI assigns all rights in the output to the user.

**Anthropic** states in its [Terms of Service](https://www.anthropic.com/legal/consumer-terms) and commercial agreements that customers own their outputs. Anthropic does not claim ownership of API outputs.

**Google** states in its [Generative AI Additional Terms of Service](https://ai.google.dev/gemini-api/terms) that customers retain ownership of content generated through the Gemini API.

However, "ownership" in the terms of service does not create copyright where none exists. If the output is not copyrightable (because it lacks sufficient human authorship), the provider's assignment of rights does not change that. You own whatever rights exist, which may be limited.

### Training On Your Data

Whether providers use your API inputs and outputs to train their models is a critical concern, especially for organizations processing confidential or regulated data.

As of 2026-02-10, [OpenAI's API data usage policy](https://platform.openai.com/docs/models/how-we-use-your-data) states that data submitted through the API is not used to train models by default. However, the consumer ChatGPT product has different defaults, and users must opt out. OpenAI offers a [Data Processing Addendum (DPA)](https://openai.com/policies/data-processing-addendum/) for enterprise customers.

As of 2026-02-10, [Anthropic's commercial terms](https://www.anthropic.com/legal/commercial-terms) state that Anthropic does not train on customer API inputs or outputs by default. Anthropic offers a DPA and has committed to this position in its commercial agreements.

As of 2026-02-10, [Google's API terms for Gemini](https://ai.google.dev/gemini-api/terms) state that Google does not use API data to improve its products unless the customer opts in. Google offers workspace-grade data handling for enterprise customers.

The patterns to watch for: consumer-tier products often have broader data usage rights than API or enterprise products. Free tiers may have different terms than paid tiers. Terms change over time. Read the current terms for the specific product tier you are using, and re-read them when they are updated. If you are processing confidential data, get a DPA.

### Data Retention

Providers retain data for different periods and purposes (abuse monitoring, debugging, safety). As of 2026-02-10, OpenAI retains API data for up to 30 days for abuse and misuse monitoring, with a [zero-day retention option](https://platform.openai.com/docs/models/how-we-use-your-data) available for eligible endpoints. Anthropic retains data for safety monitoring purposes as described in its usage policy. Retention policies vary by provider and product tier. If your data is subject to regulatory retention requirements (or deletion requirements), verify that the provider's retention policy is compatible and get it in writing.

## Open-Weight Model Licenses

The terms "open source" and "open weights" are used loosely in the AI community, but they mean different things and the distinction matters commercially.

### Open Source Versus Open Weights

The [Open Source Initiative (OSI)](https://opensource.org/) has published a definition of [Open Source AI](https://opensource.org/ai/open-source-ai-definition) that requires freedom to use, study, modify, and share the system, including access to training data, model weights, and the code used to train and run it. By this definition, very few AI models qualify as truly "open source" because most providers do not release their training data or full training code.

What most providers release is "open weights": the trained model parameters that allow you to run and fine-tune the model, but without the training data, training code, or full reproduction recipe. This is analogous to distributing a compiled binary with the right to modify and redistribute it, but without the full source code and build system.

### Notable Licenses

As of 2026-02-10, the most commercially relevant open-weight licenses are:

**Meta's Llama license** (used for [Llama 3](https://llama.meta.com/llama3/) and subsequent releases) is a bespoke license that permits commercial use, modification, and redistribution, but with restrictions. The license includes a monthly active user threshold (700 million MAU as of the Llama 3.1 release) above which a separate commercial license from Meta is required. The license also includes acceptable use restrictions that prohibit certain applications. This is not an OSI-approved open source license, and the MAU threshold means it does not grant unrestricted commercial freedom. Read the [full license text](https://llama.meta.com/llama3/license/) before assuming you can use it freely.

**Apache 2.0 models** from providers like [Mistral](https://mistral.ai/) (for some of their models, such as [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)) are released under the standard [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), which is a well-understood OSI-approved open source license. Apache 2.0 permits commercial use, modification, and redistribution with minimal restrictions (attribution, no trademark use, patent grant). Models released under Apache 2.0 provide the strongest commercial freedom.

**Other bespoke licenses** exist from various providers, each with their own restrictions. [Google's Gemma](https://ai.google.dev/gemma/terms) models have a specific terms of use. [Microsoft's Phi](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE) models have had varying license terms across versions. Always read the specific license for the specific model version you intend to use. Do not assume that because a model is "open" it is unrestricted.

### Practical Guidance

Before deploying an open-weight model commercially, verify: (1) the license permits your intended use, (2) you comply with any usage thresholds or restrictions, (3) you satisfy attribution requirements, (4) the acceptable use policy does not prohibit your application category, and (5) you understand whether fine-tuned derivatives inherit the base model's license restrictions (they usually do).

## Data Residency And Sovereignty

When you send data to a cloud-hosted LLM API, that data crosses network boundaries and may be processed and stored in jurisdictions with different legal frameworks than your own. For organizations subject to data protection regulations, this creates compliance obligations that cannot be ignored.

### GDPR Considerations

The [General Data Protection Regulation (GDPR)](https://gdpr.eu/) imposes specific requirements on the transfer of personal data outside the European Economic Area (EEA). As of 2026-02-10, sending personal data to a US-based API provider requires a valid transfer mechanism, such as [Standard Contractual Clauses (SCCs)](https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/standard-contractual-clauses-scc_en) or the [EU-U.S. Data Privacy Framework](https://www.dataprivacyframework.gov/), which was adopted by the European Commission in July 2023. The Data Privacy Framework provides a lawful transfer mechanism for organizations that have self-certified, but its durability is uncertain given the history of prior frameworks ([Safe Harbor](https://en.wikipedia.org/wiki/International_Safe_Harbor_Privacy_Principles) was invalidated in *Schrems I*, [Privacy Shield](https://en.wikipedia.org/wiki/EU%E2%80%93US_Privacy_Shield) was invalidated in *Schrems II*).

Practical steps:

**Minimize personal data in prompts.** The most robust compliance strategy is to avoid sending personal data to external APIs in the first place. Strip, anonymize, or pseudonymize personal data before it enters the LLM pipeline. See the [Safety, Privacy, And Security](06-safety-privacy-security.md) chapter for data classification and minimization techniques.

**Use regional API endpoints where available.** As of 2026-02-10, major providers are expanding regional availability. [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) offers endpoints in EU regions. [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/docs/general/locations) offers regional deployment options. [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/regions) offers a European region for API access. Regional endpoints can satisfy data residency requirements, but verify with the provider exactly where data is processed, stored, and logged.

**Execute Data Processing Agreements (DPAs).** If you are a data controller under GDPR sending personal data to an API provider, you need a DPA that establishes the provider as a data processor and specifies the lawful basis for processing, the categories of data, the processing purposes, and the technical and organizational measures in place. Major providers offer standard DPAs.

**Consider self-hosting for sensitive workloads.** For data that cannot leave your jurisdiction under any circumstances (certain government data, healthcare data in some jurisdictions, financial data subject to local regulations), self-hosting an open-weight model within your own infrastructure may be the only viable option. This eliminates the cross-border transfer issue entirely but introduces operational complexity. See [Ops](08-ops.md) for self-hosting considerations.

## IP Ownership In AI-Assisted Development

When an engineer uses an AI coding assistant (GitHub Copilot, Claude, Cursor, or similar tools) to write code, the question of who owns the resulting code involves multiple parties: the engineer, the employer, and the tool provider.

### The Employer-Employee Dimension

In most employment relationships, code written by an employee in the course of their duties is a "work made for hire" and the employer owns the copyright. This analysis does not change simply because the employee used an AI tool. The tool is an instrument, like a compiler or an IDE. However, if the AI-generated portions of the code are not copyrightable (because they lack human authorship), then the employer's copyright may extend only to the human-authored portions.

### Tool Provider Terms

AI coding tool providers generally do not claim ownership of the code you generate using their tools. As of 2026-02-10, [GitHub Copilot's terms](https://github.com/customer-terms/github-copilot-product-specific-terms) state that suggestions belong to the user. [Anthropic's terms](https://www.anthropic.com/legal/consumer-terms) assign output rights to the user. However, some tools may retain the right to use your inputs (the code context you share with the tool) for model improvement unless you opt out. If you are working with proprietary code, verify the tool's data usage policy and opt out of training data collection where available.

### Practical Implications

For most engineering teams, the practical risk is low: you use AI tools to accelerate development, you review and modify the output, and the result is a human-authored work with AI assistance. The legal ambiguity lies at the margins. To manage the risk, establish clear policies about which AI tools are approved for use with proprietary code, document the extent of AI assistance in your development process (your organization's legal team may want this for IP due diligence), and ensure that engineers are reviewing and modifying AI-generated code rather than using it verbatim without judgment.

## Liability For AI Outputs

When an AI system produces output that causes harm -- incorrect medical advice, biased hiring recommendations, defamatory statements, faulty code that causes data loss -- the question of who bears responsibility is legally unsettled.

### Current Frameworks

As of 2026-02-10, there is no comprehensive liability framework specifically for AI-generated harm in the United States. Courts and regulators are applying existing frameworks:

**Product liability** may apply when an AI system is embedded in a product and the AI's output causes harm. Traditional product liability theories (manufacturing defect, design defect, failure to warn) are being tested against AI systems. The challenge is that AI outputs are probabilistic and context-dependent, which does not map cleanly onto defect theories designed for physical products.

**Professional liability** may apply when AI is used to assist professional judgments (medical diagnosis, legal advice, financial planning) and the professional relies on incorrect AI output without adequate independent judgment.

**Negligence** may apply when an organization deploys an AI system without reasonable safeguards (testing, monitoring, human oversight) and the system causes foreseeable harm.

**Section 230 of the Communications Decency Act** ([47 U.S.C. Section 230](https://www.law.cornell.edu/uscode/text/47/230)) provides immunity to platforms for user-generated content, but as of 2026-02-10 it is unclear whether this immunity extends to AI-generated content. If a chatbot generates defamatory content, is the operator a "publisher" of that content or an "interactive computer service" immune from liability for third-party content? This question has not been definitively resolved.

### Practical Implications

You cannot control the legal outcomes, but you can build systems that are more defensible. Document your design decisions and the safety measures you implemented. Run evaluations and keep the results. Implement monitoring and incident response. Add appropriate disclaimers to AI-generated output, especially in high-risk domains. These practices do not guarantee immunity, but they demonstrate reasonable care, which matters in a negligence analysis.

## The EU AI Act

The [EU AI Act](https://artificialintelligenceact.eu/) is the world's first comprehensive AI regulation. It entered into force on August 1, 2024, with provisions phasing in over a staggered timeline through 2027. As of 2026-02-10, the prohibitions on unacceptable-risk AI practices are in effect, and many other obligations are being phased in.

### Risk Classification

The AI Act classifies AI systems into risk tiers, with obligations proportional to the risk:

**Unacceptable risk** (prohibited): social scoring by governments, real-time biometric identification in public spaces (with limited exceptions), manipulation techniques that exploit vulnerabilities, and emotion recognition in workplaces and schools. These are banned outright. As of February 2, 2025, these prohibitions are enforceable.

**High risk**: AI systems used in critical infrastructure, education, employment, essential services, law enforcement, migration, and justice. These systems face mandatory requirements including risk management, data governance, technical documentation, transparency, human oversight, accuracy, robustness, and cybersecurity. Most high-risk obligations apply from August 2026.

**Limited risk**: AI systems that interact with people (chatbots), generate synthetic content (deepfakes), or perform emotion recognition. These face transparency obligations -- users must be informed they are interacting with AI, and synthetic content must be labeled.

**Minimal risk**: all other AI systems, which face no specific obligations under the Act beyond existing law.

### General-Purpose AI (GPAI) Models

The AI Act includes specific provisions for general-purpose AI models (which includes large language models). As of 2026-02-10, providers of GPAI models must comply with transparency obligations including providing technical documentation, making information available to downstream deployers, publishing a sufficiently detailed summary of training data content, and complying with EU copyright law. GPAI models with "systemic risk" (defined by compute thresholds and other criteria) face additional obligations including model evaluation, adversarial testing, incident tracking, and cybersecurity measures. These GPAI obligations apply from August 2, 2025.

### What It Means For AI Teams

If you are building AI features that serve EU users or are deployed within the EU, the AI Act likely applies to your system. The key engineering implications:

**Classification matters.** Determine which risk tier your AI system falls into. This determines your obligations. A customer support chatbot is likely "limited risk" (transparency obligations). An AI system that screens job applicants is "high risk" (extensive obligations).

**Transparency is mandatory.** If your system interacts with users, they must know they are interacting with AI. If your system generates synthetic content, that content must be machine-readably labeled. Build these disclosure mechanisms into your product from the start.

**Documentation is required.** High-risk systems require technical documentation, risk assessments, data governance records, and conformity assessments. The [Governance And Risk](14-governance-and-risk.md) chapter covers documentation practices that align with these requirements.

**Human oversight must be real.** The AI Act requires that high-risk systems be designed to allow effective human oversight. A checkbox that says "human reviewed" without meaningful review capability does not satisfy this requirement.

## Patent Considerations

The question of whether AI-generated inventions can be patented parallels the copyright debate but has been addressed more definitively in some jurisdictions.

As of 2026-02-10, the [U.S. Patent and Trademark Office (USPTO)](https://www.uspto.gov/) issued [guidance in February 2024](https://www.federalregister.gov/documents/2024/02/13/2024-02623/inventorship-guidance-for-ai-assisted-inventions) stating that AI-assisted inventions are not categorically unpatentable, but that an AI system cannot be listed as an inventor. At least one natural person must have made a "significant contribution" to the invention. The guidance followed the Federal Circuit's decision in [*Thaler v. Vidal*](https://cafc.uscourts.gov/opinions-orders/21-2347.OPINION.8-5-2022_1988142.pdf), which held that an AI cannot be an "inventor" under U.S. patent law.

The UK Supreme Court reached the same conclusion in [*Thaler v. Comptroller-General of Patents*](https://www.supremecourt.uk/cases/uksc-2021-0201.html), ruling in December 2023 that an AI system cannot be an inventor under UK patent law.

In practice, this means: if you use AI to assist in developing a patentable invention, ensure that human inventors made significant intellectual contributions to the conception of the invention. Document the human contributions. An invention where the human's only contribution was typing a prompt and the AI produced the complete inventive concept may not be patentable.

## Practical Compliance Patterns

The legal landscape is uncertain, but you can build systems that are well-positioned regardless of how the law develops. These patterns reduce legal risk and make it easier to adapt to new requirements.

### Documentation And Audit Trails

**Document AI usage in your development process.** Maintain records of which AI tools are used, for what purposes, and with what level of human oversight. This documentation supports IP due diligence, regulatory compliance, and litigation defense.

**Log AI interactions for regulated workloads.** For AI systems that make or assist consequential decisions (hiring, lending, medical, legal), maintain audit trails that include the inputs, the model's output, any human review or override, and the final decision. These logs support explainability requirements and enable post-hoc investigation.

**Version everything.** Prompts, model versions, evaluation results, training data manifests, and configuration. Reproducibility is both good engineering and good legal practice.

### Disclosure Practices

**Disclose AI involvement where required or expected.** The EU AI Act mandates disclosure for certain AI interactions. Even where not legally required, voluntary disclosure builds trust and reduces the risk of later allegations of deception. Establish organizational guidelines for when and how to disclose AI involvement in content creation, decision-making, and customer interactions.

**Label synthetic content.** For generated text, images, audio, or video, apply metadata labels (such as [C2PA](https://c2pa.org/) content credentials) that identify the content as AI-generated. This is increasingly expected by platforms, regulators, and users.

### Provenance And Licensing Hygiene

**Track the licensing of every model you deploy.** Maintain a register of model names, versions, license types, and any usage restrictions. Treat model licenses with the same rigor you apply to software dependency licenses.

**Track training data provenance.** If you fine-tune models, document the sources of your training data, the licenses under which it was obtained, and any consent or opt-out mechanisms you respect. This protects you if training data copyright law develops in a direction that requires such documentation.

**Review API terms of service periodically.** Provider terms change. Set a calendar reminder to review the terms of your key providers at least annually, and whenever you receive a notification of terms changes.

## Pitfalls

**Assuming "open" means "unrestricted."** Open-weight models come with licenses that may include commercial use restrictions, user thresholds, acceptable use policies, and derivative work requirements. Read the license. The word "open" in marketing materials does not mean the same thing as an OSI-approved open source license.

**Ignoring data residency until a regulator asks.** Sending personal data to a US-based API without a valid transfer mechanism violates GDPR. The consequences include fines of up to 4 percent of global annual revenue. Engineer data residency compliance into your architecture from the start, not as an afterthought.

**Relying on provider terms without reading them.** Assumptions like "they don't train on API data" or "I own the output" may be true for one provider tier but not another, or may have changed since the last time you checked. Read the current terms for the specific product and tier you use.

**Treating AI-generated content as equivalent to human-authored content for IP purposes.** Code, text, or images generated entirely by AI may not be copyrightable. If your business model depends on copyright protection of AI-generated output, that assumption may not hold.

**Building high-risk AI systems without legal review.** Systems that make or assist consequential decisions about people (employment, credit, healthcare, law enforcement) face the highest regulatory scrutiny and the greatest liability exposure. Involve legal counsel early, not after the system is built and deployed.

**Waiting for legal certainty before acting.** The law will not be settled for years. Waiting for definitive answers is not a strategy. Build systems that document their behavior, respect user preferences, implement human oversight, and can adapt to new requirements. These practices are good engineering regardless of how the law develops.

## Checklist
- Have you identified which AI regulations apply to your system (EU AI Act risk tier, GDPR, sector-specific rules)?
- Do you have a register of all AI models deployed, their licenses, and any usage restrictions?
- Have you read the current terms of service for each LLM API provider you use?
- Do you have Data Processing Agreements in place for providers that process personal data?
- Is personal data minimized or anonymized before being sent to external AI APIs?
- Do you use regional API endpoints or self-hosting where data residency requires it?
- Are AI-generated outputs disclosed to users where required by regulation or organizational policy?
- Do you maintain audit trails for AI-assisted decisions in high-risk domains?
- Is there a documented policy for AI tool usage in your development process?
- Have you consulted legal counsel for your specific jurisdiction and use case?
- Do you track training data provenance and licensing for any fine-tuned models?
- Are you periodically reviewing provider terms of service for changes?

## References

### Copyright And Authorship
- U.S. Copyright Office, Copyright Registration Guidance: Works Containing Material Generated by Artificial Intelligence. Federal Register, August 2023. https://www.federalregister.gov/documents/2023/08/30/2023-18624/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence
- *Thaler v. Perlmutter*, No. 22-1564 (D.D.C. 2023). Copyright Office decision document. https://www.copyright.gov/ai/Thaler%20v%20Perlmutter.pdf
- U.S. Copyright Office, *Zarya of the Dawn* registration decision. https://www.copyright.gov/docs/zarya-of-the-dawn.pdf
- U.S. Copyright Office, fair use index. https://www.copyright.gov/fair-use/

### Training Data Litigation
- *The New York Times Co. v. Microsoft Corp. et al.*, complaint filed December 2023. https://nytco-assets.nytimes.com/2023/12/NYT_Complaint_Dec2023.pdf
- *Authors Guild v. Google*, 804 F.3d 202 (2d Cir. 2015). https://law.justia.com/cases/federal/appellate-courts/ca2/13-4829/13-4829-2015-10-16.html
- 17 U.S.C. Section 107 (Fair Use). https://www.law.cornell.edu/uscode/text/17/107

### Provider Terms And Policies
- OpenAI API data usage policy. https://platform.openai.com/docs/models/how-we-use-your-data
- OpenAI Terms of Use. https://openai.com/policies/terms-of-use/
- Anthropic Commercial Terms. https://www.anthropic.com/legal/commercial-terms
- Google Gemini API Additional Terms of Service. https://ai.google.dev/gemini-api/terms

### Open-Weight Licenses
- Open Source Initiative, Open Source AI Definition. https://opensource.org/ai/open-source-ai-definition
- Meta Llama 3 Community License Agreement. https://llama.meta.com/llama3/license/
- Apache License 2.0. https://www.apache.org/licenses/LICENSE-2.0

### Regulation
- EU AI Act full text and guidance. https://artificialintelligenceact.eu/
- GDPR official text. https://gdpr.eu/
- EU-U.S. Data Privacy Framework. https://www.dataprivacyframework.gov/
- EU Standard Contractual Clauses. https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/standard-contractual-clauses-scc_en

### Patents
- USPTO, Inventorship Guidance for AI-Assisted Inventions. Federal Register, February 2024. https://www.federalregister.gov/documents/2024/02/13/2024-02623/inventorship-guidance-for-ai-assisted-inventions
- *Thaler v. Vidal*, No. 21-2347 (Fed. Cir. 2022). https://cafc.uscourts.gov/opinions-orders/21-2347.OPINION.8-5-2022_1988142.pdf
- *Thaler v. Comptroller-General of Patents*, UKSC 2021/0201. https://www.supremecourt.uk/cases/uksc-2021-0201.html

### Content Provenance
- Coalition for Content Provenance and Authenticity (C2PA). https://c2pa.org/

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](29-migration-and-vendor-strategy.md) | [Next](09-glossary.md)
