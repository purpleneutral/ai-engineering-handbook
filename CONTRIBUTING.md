# Contributing

This is a living reference. The bar is "useful in 6 months", not "perfect today".

## Principles
- Be explicit about volatility: use `As of YYYY-MM-DD, ...` for anything likely to change.
- Separate facts from guidance.
- Facts: cite sources (vendor docs, papers, specs).
- Guidance: explain tradeoffs and assumptions.
- Prefer checklists, small examples, and decision criteria over long prose.
- Avoid fragile "best model" claims. Write "selection criteria" instead.

## Style
- Keep files in `docs/` focused and skimmable.
- Use short sections and lists; avoid nested bullets when possible.
- Use consistent headings.
- `## Summary`
- `## When To Use`
- `## How It Works`
- `## Pitfalls`
- `## Checklist`
- `## References`

## Curated Link Entries
- Prefer official docs and official GitHub repos.
- Each entry should be one line and answer: "what is it" and "why should I care".
- If you include install commands, date-stamp volatile ones with `As of YYYY-MM-DD`.
- If relevant, add a quick difficulty tag: `Easy`, `Medium`, or `Hard`.

## Suggested Workflow
1. Add or update a page in `docs/`.
2. If you add a volatile claim, include an `As of YYYY-MM-DD` line and a source in `References`.
3. Keep commits small and topic-focused.
