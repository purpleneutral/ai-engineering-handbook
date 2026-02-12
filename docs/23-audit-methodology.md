# Audit Methodology

Last reviewed: 2026-02-10

[Contents](README.md)

## Summary

Every chapter in this handbook carries two dates: a **Last reviewed** date (when the content was last read and considered current) and a **Last audited** date (when factual claims, links, and code examples were systematically verified against primary sources). This page documents how audits are conducted, what they check, and how issues are classified. The goal is transparency: readers should know exactly what "audited" means and how much to trust the result.

## What An Audit Covers

An audit is a structured review of a chapter's content against external sources. It is not a peer review of writing quality or a judgment of pedagogical approach. It is a fact-checking and link-checking exercise with a defined scope.

### Factual Claims

Every verifiable factual claim is checked against a primary source. This includes:

- **Paper citations.** ArXiv IDs, DOIs, and proceedings links are verified to resolve to the correct paper with the correct title, authors, and year. Claims attributed to specific papers (e.g., "Research X showed Y") are cross-checked against the paper's actual findings.
- **Statistics and numbers.** Specific figures (percentages, dollar amounts, parameter counts, dates) are verified against the cited source or, if uncited, against the best available primary source. Unverifiable statistics are flagged for removal or softening.
- **Dates.** Publication dates, release dates, and event dates are verified. Common errors include confusing arXiv submission dates with publication dates, or attributing a paper to the wrong conference.
- **Product and service descriptions.** Claims about what a tool, API, or framework does are checked against current official documentation.
- **Venue attributions.** Claims that a paper was "presented at" or "published at" a specific conference are verified against the conference proceedings.

### Links

Every URL in the chapter is checked:

- **Resolution.** Does the URL return a 200 status, or does it 404, redirect, or timeout?
- **Content match.** Does the page at the URL actually contain the content described by the link text? Documentation URLs are especially prone to reorganization.
- **Staleness.** Has the linked documentation moved to a new URL structure? (Common with Chroma, Weaviate, and other fast-moving projects.)
- **Format consistency.** ArXiv links should use the `/abs/` format, not `/html/` or `/pdf/`, unless there is a specific reason.

### Code Examples

Code examples are checked for:

- **Syntactic correctness.** Does the code parse without errors?
- **API accuracy.** Do method names, parameter names, and return types match the current version of the SDK or library?
- **Best practices.** Does the code follow basic production practices (context managers for file handles, proper error handling patterns)?
- **Namespace stability.** Are APIs used from stable namespaces, or are they in `beta`/`preview` namespaces that may change? Beta usage is flagged with a date stamp.

### Structure And Cross-References

- **Prev/Next navigation.** Does the navigation chain match the reading order defined in the table of contents?
- **See Also links.** Do all cross-references point to files that exist? Does the link text match the destination chapter's actual title?
- **Heading capitalization.** Are headings consistent with the handbook's title case convention?
- **Required sections.** Does the chapter include the sections specified in the [template](_template.md) and [CONTRIBUTING.md](../CONTRIBUTING.md)?

## Severity Classification

Audit findings are classified into three levels:

| Severity | Criteria | Action |
|----------|----------|--------|
| **Must-Fix** | Factual errors, fabricated URLs, misleading attributions, broken links to primary sources | Fix before next publication |
| **Should-Fix** | Stale links, imprecise claims, missing date stamps on volatile content, beta API usage without warning, dangling references | Fix in next review cycle |
| **Low Priority** | Minor formatting inconsistencies, slightly imprecise numerical ranges, style nits | Fix at convenience |

## Volatile Claims And Date Stamps

Claims that are likely to change within a year are prefixed with `As of YYYY-MM-DD` so readers know when to re-verify. This applies to:

- Pricing and token costs
- Model capabilities and availability
- Framework version-specific behavior
- Provider support and feature lists
- Statistics that are snapshots in time (citation counts, market figures)

Evergreen concepts (architectural patterns, failure modes, mathematical properties) do not need date stamps.

## Audit Frequency

- **Content review.** Each chapter is reviewed when its subject area has significant developments (new model releases, API changes, framework updates). The "Last reviewed" date at the top of each chapter reflects this.
- **Systematic audit.** A full fact-checking and link-checking audit is conducted periodically. The "Last audited" date at the bottom of each chapter reflects when this was last done.
- **Continuous.** Readers who spot errors are encouraged to [report them](https://github.com/purpleneutral/ai-engineering-handbook/issues/new?template=correction.md&labels=correction). Corrections from reader reports are applied promptly and the audit date is updated.

## Limitations

An audit improves confidence but does not guarantee correctness. Specific limitations:

- **Web search coverage.** Verification relies on publicly available sources. Claims from paywalled or restricted sources may not be fully verifiable.
- **Temporal accuracy.** A claim that was correct on the audit date may become incorrect as tools, APIs, and research evolve. Date stamps mitigate this but do not eliminate it.
- **Code examples.** Examples are checked for syntactic and API correctness but are not executed against live APIs during the audit. Runtime behavior may differ from documentation.
- **Link durability.** URLs that resolved on the audit date may break later. This is especially common for documentation sites that reorganize frequently.

## How To Report An Error

If you find something wrong, outdated, or misleading, open a [Correction issue](https://github.com/purpleneutral/ai-engineering-handbook/issues/new?template=correction.md&labels=correction). Include:

1. A link to the specific section.
2. What the current text says.
3. What you believe is correct, with a primary source if possible.

---
[Contents](README.md)
