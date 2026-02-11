# Contributing

This is a living reference. The bar for inclusion is "useful in six months," not "perfect today." Contributions should make the material more accurate, more complete, or easier to use.

## Principles

**Be explicit about volatility.** AI tooling, pricing, and capabilities change rapidly. Any claim that is likely to change within a year should include an `As of YYYY-MM-DD, ...` prefix so readers know when to re-verify. Evergreen concepts (architectural patterns, failure modes, design principles) do not need date stamps.

**Separate facts from guidance.** Facts should cite sources — vendor documentation, academic papers, or specifications. Guidance should explain trade-offs and assumptions so readers can evaluate whether the advice applies to their situation.

**Write for decision-making, not for comprehensiveness.** Prefer checklists, decision criteria, small examples, and trade-off analyses over exhaustive coverage. The goal is to help someone make a good decision, not to catalog every option.

**Avoid fragile claims.** Do not write "Model X is the best at Y." Instead, write selection criteria and trade-offs. The reader's constraints determine the right choice, and the landscape changes faster than any document can track.

## Style

### Tone

Write as a senior engineer explaining concepts to a capable colleague. Be direct, precise, and practical. Avoid marketing language, hype, and unnecessary hedging. If something is uncertain, say so plainly.

### Structure

Each chapter should follow a consistent structure. Not every section is required for every page, but the ordering should be consistent:

1. `## Summary` — One to three sentences that capture the key insight.
2. `## See Also` — Links to related chapters for navigation.
3. Content sections — The body of the chapter. Use descriptive headings.
4. `## Pitfalls` — Common mistakes and how to avoid them (where applicable).
5. `## Checklist` — A concise yes/no list for production readiness.
6. `## References` — Primary sources (vendor docs, papers, specifications).

### Formatting

- Use short sections and descriptive headings for skimmability.
- Prefer explanatory paragraphs over bare bullet lists for teaching content.
- Use bullet lists for checklists, enumerations, and quick-reference material.
- Use code blocks for commands, schemas, and prompt examples.
- Use Mermaid diagrams sparingly for architectural flows.

## Curated Link Entries

Links in the [Tool Index](README.md#tool-index-curated) and [Reading List](docs/10-reading-list.md) should meet these criteria:

- Prefer official documentation and official GitHub repositories.
- Each entry should answer two questions: "what is it?" and "why should I care?"
- If you include install commands, date-stamp volatile ones with `As of YYYY-MM-DD`.
- Include a difficulty tag where applicable: **Easy**, **Medium**, or **Hard**.

## Suggested Workflow

1. Add or update a page in `docs/`.
2. If you add a new page, update `docs/README.md` (the table of contents) and wire up Prev/Next links on the adjacent pages.
3. If you add a volatile claim, include an `As of YYYY-MM-DD` line and cite a source in the References section.
4. Keep commits small and topic-focused.
