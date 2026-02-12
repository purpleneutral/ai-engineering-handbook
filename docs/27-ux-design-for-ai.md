# UX Design For AI Features

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](26-caching-and-latency.md) | [Next](28-multi-tenancy.md)

## Summary

UX for AI features is an engineering discipline, not a visual design exercise. The hard problems are streaming responses without jarring the user, communicating uncertainty honestly, failing gracefully when the model times out or hallucinates, and choosing the right interaction pattern for the task. Get these wrong and it does not matter how good your model is --- users will not trust it, will not use it, or will use it in ways that produce bad outcomes. This chapter covers the engineering side: what to build, how to build it, and what to avoid.

## See Also
- [Architecture Recipes](07-architecture-recipes.md) --- interaction patterns map directly to architecture choices.
- [Agents](04-agents.md) --- human-in-the-loop and approval gates.
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md) --- when structured UI is better than chat.
- [Ops: Shipping And Running LLM Systems](08-ops.md) --- error handling, timeouts, and monitoring.
- [Safety, Privacy, And Security](06-safety-privacy-security.md) --- user trust and data handling.

## When To Use

Every AI-powered feature has a UX surface, so this chapter applies whenever your system interacts with a human. That said, the patterns here are most critical for user-facing applications where latency is perceptible (anything over ~200ms), where the model's output is uncertain or potentially wrong, or where the user needs to understand, correct, or approve the AI's work before it takes effect.

If your AI feature is purely backend (a classification pipeline, a batch extraction job), the UX concerns here are less relevant. If a human ever sees the output, reads the response, or acts on the result, they are directly relevant.

## How It Works

### Streaming Responses And Progressive Rendering

LLM inference is slow. A typical response takes 2--15 seconds to generate fully, which is an eternity in user-facing applications. **Streaming** eliminates the perception of waiting by displaying tokens as they are generated, reducing time-to-first-token (TTFT) from seconds to milliseconds.

The standard transport for streaming is **Server-Sent Events (SSE)**. The client opens a long-lived HTTP connection, and the server pushes chunks of the response as they arrive from the model. Here is a minimal backend implementation using Python and FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.get("/chat")
async def chat(q: str):
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": q}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

On the client side, the `EventSource` API handles SSE natively, but for more control (custom headers, POST requests, abort signals), use the [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API) with a streaming reader:

```typescript
async function streamChat(
  query: string,
  onChunk: (text: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch("/chat?q=" + encodeURIComponent(query), {
    signal,
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    const lines = text.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        onChunk(line.slice(6));
      }
    }
  }
}
```

**Markdown rendering mid-stream** is trickier than it sounds. Partial markdown is often invalid --- an unclosed code fence, a half-formed table, a dangling link. Naive renderers will produce flickering or broken output. Two practical approaches work well. The first is to render markdown only on completed blocks: accumulate text until you detect a block boundary (a blank line, a closing fence), then render the completed block. The second is to use a streaming-aware markdown parser that handles partial input gracefully. Libraries like [marked](https://marked.js.org/) can be configured to be tolerant of incomplete input, and [react-markdown](https://github.com/remarkjs/react-markdown) can re-render incrementally as new content arrives.

A React component for streaming display:

```tsx
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

function StreamingMessage({ query }: { query: string }) {
  const [content, setContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    setContent("");
    setIsStreaming(true);

    streamChat(
      query,
      (chunk) => setContent((prev) => prev + chunk),
      controller.signal
    )
      .then(() => setIsStreaming(false))
      .catch((err) => {
        if (err.name !== "AbortError") setIsStreaming(false);
      });

    return () => controller.abort();
  }, [query]);

  return (
    <div className="message">
      <ReactMarkdown>{content}</ReactMarkdown>
      {isStreaming && <span className="cursor" aria-label="Generating" />}
    </div>
  );
}
```

The blinking cursor (or a pulsing dot, or a subtle animation) is not decoration. It communicates that the system is still working. Remove it the instant streaming completes. Users are sensitive to mismatches between the indicator and reality.

### Showing Uncertainty And Confidence

LLMs do not reliably know when they are wrong. Model confidence (logprobs, softmax scores) does not correlate well with factual accuracy --- a model can be highly confident about a hallucinated answer. This makes **calibration** a hard problem, but not one you can ignore.

There are several practical approaches, ordered by engineering effort. The simplest is **hedging language in the system prompt**. Instruct the model to say "I'm not sure" or "Based on the information I have" when it lacks strong grounding. This is cheap but unreliable --- the model may hedge when it should not, or fail to hedge when it should.

The next level is **retrieval-based confidence**. If you are running a RAG system, you have a natural confidence signal: the similarity score of retrieved documents. When the best retrieval score is below a threshold, display a warning like "I could not find strong supporting information for this answer." This is more trustworthy than the model's self-assessment because it is grounded in a measurable signal.

A more robust approach is **consistency-based confidence**. Run the same query multiple times (or sample multiple completions) and measure agreement. If five completions all say the same thing, confidence is higher than if they diverge. This costs more in latency and tokens, so reserve it for high-stakes decisions. You can also use this technique offline in your eval pipeline to identify query patterns where the model is unreliable.

Whatever approach you use, the display pattern matters. Avoid numeric confidence scores ("87% confident") --- users either ignore them or misinterpret them. Instead, use categorical indicators: a subtle badge ("Verified from source" vs. "AI-generated answer"), color coding (but never as the only signal --- see accessibility below), or explicit caveats inline with the response.

### Conversation Design And Multi-Turn State

Multi-turn conversation introduces state management problems that single-turn interactions do not have. The key decisions are what to remember, how long to remember it, and how to handle the context window constraint.

**Conversation memory** falls into three tiers. **Session memory** persists for the duration of a conversation and is discarded when the session ends. This is the simplest tier and is appropriate for most consumer chat interfaces. **Cross-session memory** persists across conversations, allowing the system to remember user preferences, past decisions, or ongoing projects. This requires explicit storage (a database, a user profile) and raises privacy questions --- users should be able to view and delete what the system remembers. **Semantic memory** extracts facts and preferences from conversations and stores them in a structured format for future retrieval. This is the most powerful tier and the most complex to build correctly.

From a UX perspective, the critical principle is **transparency**. Users should know what the system remembers. A "memory" panel, a visible context indicator, or an explicit "I remember that you prefer X" acknowledgment all help build trust. Silently remembering things feels surveillance-like; explicitly remembering things feels helpful.

**Context window management** affects UX because it determines when the conversation degrades. As the context fills up, earlier turns are either truncated or summarized. The user does not see this happening, but they notice when the system "forgets" something they said ten messages ago. Practical strategies include displaying a subtle indicator when the conversation is long ("This is a long conversation --- the assistant may not recall earlier details"), offering a "start new conversation" prompt when context is getting full, and using a sliding window with summarization so that the system retains the gist of earlier turns even when it cannot retain the full text.

### Error States And Graceful Degradation

AI features fail in ways that traditional software does not. The model might time out, return gibberish, hallucinate, refuse to answer, or produce an answer that is technically correct but useless. Each failure mode needs a distinct UX response.

**Timeout and rate limits.** When the model provider returns a 429 (rate limit) or the request times out, show a clear, non-alarming message: "This is taking longer than expected. You can wait or try again." Offer a retry button. Do not show raw error codes or stack traces. If timeouts are frequent, implement a queue with position indicators ("Your request is queued, estimated wait: ~30 seconds").

**Model refusals.** Models sometimes refuse to answer questions they consider harmful, out of scope, or otherwise problematic. The refusal message from the model is often unhelpful ("I can't help with that"). Detect refusals (pattern matching on common refusal phrases, or checking for a refusal flag in the API response) and replace them with a more useful message that explains what the system can help with, or offers to connect the user with a human.

**Hallucination and low-quality output.** This is the hardest failure to handle because it looks like success --- the model returned a plausible-sounding response that happens to be wrong. Your best defense is design-level: include citations so users can verify, mark AI-generated content clearly, and build feedback mechanisms so users can flag problems. Do not design interfaces that imply the AI's output is authoritative.

**Partial failures in multi-step workflows.** When an agent or tool-calling system fails partway through a task, show the user what was completed, what failed, and what remains. "I found 3 relevant documents but could not access the billing system to verify the amounts" is far more useful than a generic error.

### Chat Vs. Structured UI

Chat is the default interface for AI features, but it is often the wrong one. Chat excels when the task is exploratory, open-ended, or when the user does not know exactly what they want. It is poorly suited for tasks that have a known structure, require specific inputs, or produce results that need to be compared or manipulated.

**Use structured UI (forms, wizards, dashboards) when:**
- The task has defined inputs. If you know you need a date range, a department, and a metric, a form with three fields is faster and less error-prone than asking the user to type a sentence.
- The output has a defined structure. A table of results, a chart, or a comparison view is more useful than a paragraph describing the same information.
- The user needs to iterate on parameters. Adjusting a slider or changing a dropdown is faster than rephrasing a query.
- Accuracy matters more than flexibility. Forms constrain input to valid values; chat accepts anything.

**Use chat when:**
- The task is genuinely open-ended ("Help me brainstorm names for this project").
- The user does not know the right question to ask.
- The interaction benefits from back-and-forth refinement.
- The domain is too broad to capture in a fixed set of form fields.

A hybrid approach often works best: use chat for the initial interaction, then transition to structured UI for refinement. For example, a user asks "Show me sales trends for Q4," and the system responds with a chart and filter controls rather than a paragraph of text.

### Loading States And Perceived Performance

Perceived performance matters more than actual performance. A system that shows progress feels faster than one that shows a spinner, even if wall-clock time is identical.

**Skeleton screens** work well for AI features that produce structured output. If you know the response will contain a title, a summary, and a list of recommendations, render the skeleton immediately and fill in each section as it becomes available. This gives the user something to scan while the rest loads.

**Progressive disclosure of results** is effective for multi-step processes. If your system retrieves documents, then synthesizes an answer, show "Searching knowledge base..." with the retrieved document titles appearing as they are found, then "Synthesizing answer..." as the model generates. Each visible step reassures the user that work is happening.

**Optimistic UI for reversible actions.** When the user clicks "Summarize this document," show the UI state as if the action is already in progress (the document panel slides aside, a summary panel appears with a loading state) rather than waiting for the first byte of the response. If the request fails, roll back gracefully.

Avoid indeterminate spinners for operations longer than a few seconds. They communicate "something is happening" but not "progress is being made," which makes waits feel longer. A determinate progress indicator, even an approximate one ("Analyzing... 2 of 5 sections complete"), is significantly better for perceived performance.

### Feedback Collection

User feedback is the cheapest eval signal you have, and most teams under-invest in collecting it.

**Explicit feedback** is the thumbs up/down pattern. It is simple and widely understood. The critical design choice is minimizing friction: the feedback control should be visible but not intrusive, and providing feedback should require exactly one click. Adding a required text field ("Tell us why") drops completion rates by 80% or more. Instead, make the text field optional and show it after the initial click.

**Correction-based feedback** is more valuable than binary signals. When a user edits an AI-generated draft, the diff between the original and the edited version is a rich training signal. When a user re-asks a question with different phrasing, the rephrased query paired with the original tells you what the system misunderstood. Capture these signals automatically, with appropriate consent and privacy controls.

**Implicit signals** include whether the user used the response (copied it, clicked a link in it, took an action based on it), how long they spent reading it, whether they immediately asked a follow-up question (which may indicate the first answer was insufficient), and whether they abandoned the conversation. These signals are noisy individually but powerful in aggregate.

Store feedback linked to the specific prompt version, model version, and retrieval results that produced the response. Feedback without context is almost useless for debugging or improvement.

### Accessibility Considerations

AI interfaces introduce accessibility challenges beyond those of traditional web applications.

**Screen reader compatibility with streaming text** is a non-trivial problem. A screen reader that announces every token as it arrives will produce an incomprehensible stream of fragments. Use ARIA live regions with `aria-live="polite"` so the screen reader waits for a natural pause before announcing new content. Batch updates at sentence or paragraph boundaries rather than token boundaries.

```html
<div role="log" aria-live="polite" aria-atomic="false">
  <div aria-relevant="additions">
    <!-- Completed paragraphs are appended here -->
  </div>
</div>
```

**Do not rely on color alone** to communicate confidence, status, or error states. A green/yellow/red confidence indicator is meaningless to a color-blind user. Pair colors with text labels, icons, or patterns. The [WCAG 2.1 guidelines](https://www.w3.org/WAI/WCAG21/quickref/) require a contrast ratio of at least 4.5:1 for normal text.

**Keyboard navigation** must work for all AI interactions. Chat inputs, feedback buttons, citation links, approval gates, and conversation controls must all be reachable and operable with keyboard alone. Test with Tab, Enter, Escape, and arrow keys.

**Timing and timeouts.** AI operations can be slow, and users with motor or cognitive disabilities may need more time to read and respond to AI output. Do not auto-dismiss AI responses or auto-advance to the next step without user confirmation. If you implement conversation timeouts, provide a way to extend them.

### Progressive Disclosure Of AI Capabilities

Users do not read documentation. They discover features by trying things. Design your AI interface so that basic capabilities are immediately obvious and advanced capabilities are discoverable through use.

**Start simple.** The initial interaction should require no instruction. A text input with a clear placeholder ("Ask a question about your data...") sets the right expectation. Avoid feature tours, onboarding modals, and capability lists on first use --- they are skipped or forgotten.

**Suggest capabilities contextually.** When the user's query is close to a capability they have not used, surface a suggestion: "You can also ask me to generate a chart from these numbers." This is more effective than listing all capabilities up front because it is relevant to the user's current task.

**Provide example queries.** A set of 3--5 starter queries (clickable) shows the user what the system can do better than any description. Choose examples that demonstrate the range of capabilities: one simple factual question, one analytical request, one action-oriented command.

### Human-In-The-Loop Patterns

When AI output has real-world consequences --- sending an email, making a purchase, updating a record --- the user needs to review and approve before execution. The design of the **approval gate** determines whether this is a useful safety check or an annoying speed bump.

**Edit-before-send** is the strongest pattern. The AI generates a draft (an email, a report, a code change), and the user can read, edit, and then approve it. This works because the AI handles the tedious part (drafting) while the human handles the judgment part (is this correct and appropriate?). Present the draft in an editable format, not a read-only preview. Users who can edit feel ownership; users who can only accept or reject feel like rubber stamps.

**Diff view for changes.** When the AI proposes modifications to existing content (editing a document, updating a configuration, refactoring code), show a clear diff view: what will change, what will stay the same. This lets the user focus their review on the changes rather than re-reading the entire artifact.

**Batch approval with exceptions.** When the AI processes multiple items (classifying 50 support tickets, generating 20 product descriptions), do not require individual approval for each one. Show the full batch with the AI's proposed action, let the user scan for problems and override specific items, then approve the rest in bulk. Highlight items where the AI's confidence is low or where its decision was close to a different outcome.

**Escalation paths.** Not every decision should go to the same person. Design your approval flow so that routine items can be approved by the user, ambiguous items are flagged for review, and high-risk items are escalated to a designated approver. The AI's confidence signal (however imperfect) can drive this routing.

### Managing User Expectations

User trust in AI follows a predictable curve: initial over-trust (the AI seems magical), followed by a trust collapse (the AI makes a visible mistake), followed by calibrated trust (the user learns the AI's strengths and limits). Your UX should accelerate this curve toward calibrated trust without requiring the user to experience a painful failure.

**Set boundaries explicitly.** Tell users what the system is designed to do and, more importantly, what it is not designed to do. "I can help with questions about company policies and benefits. I cannot provide legal advice or make changes to your account" is more useful than "How can I help you today?"

**Label AI-generated content clearly.** Every piece of content the AI produces should be visually distinguishable from human-authored content and system content. A subtle "Generated by AI" label, a different background color (with a text label for accessibility), or a distinct avatar all work. This is not just a UX nicety --- as of 2026-02-10, the [EU AI Act](https://artificialintelligenceact.eu/) requires labeling AI-generated content in many contexts.

**Do not anthropomorphize beyond utility.** Giving the AI a name and a conversational tone is fine --- it sets expectations about interaction style. Giving it a backstory, emotional responses, or claims about its own understanding is misleading and erodes trust when the illusion breaks. Users who believe they are talking to something that understands them are more hurt by failures than users who understand they are using a sophisticated tool.

### Citation And Source Display

When AI output is grounded in retrieved sources (RAG, web search, knowledge base), showing those sources is essential for trust and verifiability.

**Inline citations** are the strongest pattern. Number references in the text ("[1]", "[2]") and display the corresponding sources in a sidebar or footer. Each citation should include the source title, a relevant snippet (so the user can verify without clicking through), and a link to the full source.

**Source cards** work well when the number of sources is small (1--5). Display each source as a card with title, snippet, and metadata (date, author, document type). Position them adjacent to the response, not hidden behind a click.

**Highlight the cited passage.** When the user clicks a citation, scroll to and highlight the specific passage in the source document that supports the claim. This is the most trustworthy citation pattern because it lets the user verify the connection between the claim and the evidence without reading the entire source document.

**Handle missing or weak sources honestly.** When the AI cannot find supporting sources, say so. "I could not find specific documentation on this topic" is better than presenting an unsourced answer without comment. When the retrieved sources are tangentially related rather than directly relevant, indicate that: "Related information (may not directly answer your question)."

## Design Notes

**Test with real users early.** AI features behave differently in the hands of real users than in demos. Users will ask questions you did not anticipate, interpret hedging language in unexpected ways, and find workflows you did not design for. Usability testing with five users will surface more UX problems than a month of internal dogfooding.

**Measure task completion, not satisfaction.** User satisfaction scores for AI features are unreliable because users often cannot tell when the AI gave them a wrong answer. Measure whether users actually completed their task correctly. This requires knowing what "correct" looks like, which ties back to evals.

**Instrument everything.** Log the user's query, the system's response, the feedback signal, the latency, the retrieval scores, and whether the user took action on the response. You will need all of this to debug quality issues and prioritize improvements. See the [Ops](08-ops.md) chapter for logging and observability patterns.

## Pitfalls

**Defaulting to chat for everything.** Chat interfaces are familiar and easy to build, which makes them the default choice. But chat is a high-friction interface for structured tasks. If users are repeatedly asking the same type of question with the same parameters, that is a signal you should build a form, a dashboard, or a dedicated view instead.

**Over-streaming.** Streaming every character creates a typewriter effect that is initially engaging but becomes fatiguing in long responses. Consider buffering to word or clause boundaries rather than emitting individual tokens. This produces smoother reading and reduces rendering overhead.

**Confidence theater.** Displaying confidence scores or "thinking" animations that are not connected to any real signal is worse than displaying nothing. Users will calibrate their trust based on these signals, and if the signals are meaningless, the calibration will be wrong. Only show confidence indicators that are backed by a genuine signal (retrieval score, consistency check, source verification).

**Ignoring the empty state.** The first thing a user sees is the empty state --- before they have typed anything. A blank chat window with a blinking cursor communicates nothing. Use the empty state to set expectations, show example queries, and guide the user toward a productive first interaction.

**Treating feedback as optional.** Teams often add thumbs up/down buttons and never look at the data. Build a pipeline that aggregates feedback, surfaces low-rated responses, and feeds them into your eval cycle. Feedback that is not acted on is wasted effort --- both yours and the user's.

**Inaccessible streaming.** Streaming text that works beautifully for sighted mouse users but is unusable for screen reader users or keyboard-only users. Test with assistive technology. The [ARIA live regions specification](https://www.w3.org/TR/wai-aria/#aria-live) governs how dynamic content should be announced.

## Checklist
- Does the interface stream responses and show a clear working indicator?
- Are error states (timeout, rate limit, refusal, hallucination) handled with user-friendly messages?
- Is AI-generated content clearly labeled as such?
- Do citations link to verifiable sources with relevant snippets?
- Is there a low-friction feedback mechanism (one-click minimum)?
- Does the interface work with keyboard navigation and screen readers?
- Are confidence signals backed by real measurements, not heuristics?
- Do high-consequence actions require explicit user approval (edit-before-send)?
- Has the interface been tested with real users, not just internal stakeholders?
- Is the empty state informative and actionable?
- Is feedback data collected, stored with context, and fed into the eval pipeline?

## References
- MDN: Using server-sent events. https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- MDN: Fetch API. https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API
- WCAG 2.1 Quick Reference. https://www.w3.org/WAI/WCAG21/quickref/
- W3C WAI-ARIA: aria-live. https://www.w3.org/TR/wai-aria/#aria-live
- Nielsen Norman Group: AI UX guidelines. https://www.nngroup.com/articles/ai-ux/
- react-markdown (streaming-friendly markdown renderer). https://github.com/remarkjs/react-markdown
- marked (JavaScript markdown parser). https://marked.js.org/
- EU AI Act. https://artificialintelligenceact.eu/

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](26-caching-and-latency.md) | [Next](28-multi-tenancy.md)
