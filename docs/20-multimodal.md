# Multimodal AI

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](13-staying-current.md) | [Next](09-glossary.md)

## Summary

Multimodal AI refers to models that process and reason across multiple input types --- text, images, audio, video, and documents --- within a single request. As of early 2026, multimodal capabilities are a baseline feature of every major frontier model, not an experimental add-on. For practitioners, this means the same model that processes text can now process screenshots, scanned documents, audio recordings, and video clips, which fundamentally changes the design space for applications like document processing, accessibility, content moderation, and data extraction.

## See Also
- [LLM Fundamentals](01-llm-fundamentals.md) --- How models process inputs, including token economics.
- [Embeddings and Vector Search](12-embeddings-and-vector-search.md) --- Multimodal embeddings extend the concepts covered here.
- [Retrieval-Augmented Generation (RAG)](03-rag.md) --- Multimodal RAG architectures build on the retrieval pipeline.
- [Safety, Privacy, and Security](06-safety-privacy-security.md) --- Visual prompt injection and cross-modal attacks.
- [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md) --- Structured output is essential for reliable extraction from images and documents.

## What Multimodal Means in Practice

Before multimodal models, processing an image meant running a separate computer vision model, processing audio meant running a separate speech-to-text model, and integrating these outputs with an LLM required glue code, format conversion, and careful error handling. Multimodal models collapse this pipeline: you send the image, audio, or video directly to the language model alongside your text prompt, and the model reasons about all of it together.

This is not just a convenience improvement. It changes what is possible. A multimodal model can read a chart image and answer questions about trends. It can look at a screenshot of a UI and identify layout issues. It can listen to a customer service call and summarize both the content and the caller's tone. These tasks were technically possible before, but they required stitching together multiple specialized models, each with its own failure modes. A single multimodal model handles the integration internally.

The practical implication for engineers is straightforward: if your application involves images, documents, audio, or video, you should evaluate whether a multimodal model can handle the task directly before building a multi-model pipeline. The answer is increasingly yes, and the unified approach is usually simpler, cheaper, and more reliable.

## Vision and Image Understanding

Vision is the most mature multimodal capability. Every major frontier model accepts images as input and can describe, analyze, extract text from, and reason about visual content.

### Provider Capabilities

As of 2026-02-10, the major providers support the following image capabilities:

| Feature | [OpenAI (GPT-4o/4.1)](https://platform.openai.com/docs/guides/images-vision) | [Anthropic (Claude 3/4)](https://docs.anthropic.com/en/docs/build-with-claude/vision) | [Google (Gemini 2.x)](https://ai.google.dev/gemini-api/docs/image-understanding) |
|---------|-------------|-------------------|-------------------|
| Formats | PNG, JPEG, GIF, WebP | JPEG, PNG, GIF, WebP | PNG, JPEG, GIF, WebP, plus PDF natively |
| Max file size | 20 MB | 30 MB | 20 MB inline; larger via Files API |
| Max resolution | No hard limit (auto-scaled) | 8000 x 8000 px | Auto-scaled |
| Max images/request | ~50 (context-dependent) | 100 | ~16 inline; more via Files API |
| Detail control | `detail: low/high/auto` | Automatic | Automatic |

**Open-weight multimodal models** have closed much of the gap with proprietary offerings. [LLaVA](https://arxiv.org/abs/2304.08485) (2023) established the dominant open-source architecture --- a visual encoder (typically [CLIP](https://arxiv.org/abs/2103.00020)) connected to an LLM via a projection layer. Meta's [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (2025) is natively multimodal with a mixture-of-experts architecture. [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) (2024) offers strong vision capabilities in an open-weight package. For teams that need data sovereignty or offline processing, open multimodal models are now a credible option.

### How Image Tokens Work

Images consume tokens just like text, but the accounting works differently across providers. Understanding this is critical for cost management and context window budgeting.

**OpenAI** uses a tile-based system. In `low` detail mode, every image costs a flat 85 tokens regardless of size. In `high` detail mode, the image is divided into 512px tiles, and each tile costs 170 tokens plus a base of 85. A 1024x1024 image in high detail costs `85 + (170 * 4) = 765` tokens. Use `low` for triage and classification; use `high` when fine detail matters (OCR, small text, charts).

**Anthropic** uses a simpler formula: `tokens = (width * height) / 750`. A 1000x1000 image costs roughly 1,333 tokens. Anthropic recommends keeping images under 1.15 megapixels (about 1568px on the longest side) for optimal performance.

**Google Gemini** offers configurable `media_resolution` that adjusts token cost, ranging from approximately 280 tokens at low resolution to 1120+ at high resolution.

The cost implications are significant when processing many images. At high detail, 1,000 images can consume hundreds of thousands of tokens. Budget accordingly and use the lowest resolution that achieves acceptable quality for your task.

### Code Example: Vision API

The API format for sending images differs across providers. Here is the OpenAI pattern, which is representative:

```python
import base64
from openai import OpenAI

client = OpenAI()

# Encode a local image
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract the total amount from this receipt."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image('receipt.jpg')}",
                    "detail": "high",
                },
            },
        ],
    }],
)
print(response.choices[0].message.content)
```

For Anthropic, images use a different structure with `type: "image"` and a `source` object specifying the encoding. For Google Gemini, you can pass `PIL.Image` objects directly or upload files via the Files API. See each provider's documentation for exact formats.

**Base64 vs. URL:** Base64 encoding works for local files but increases payload size by ~33%. URLs are more efficient but require the image to be publicly accessible. For production pipelines processing local files, base64 is the standard approach. For user-submitted URLs, pass them directly.

## Audio

Audio capabilities fall into two categories: dedicated transcription models (like Whisper) and native audio understanding built into multimodal models.

### Transcription with Whisper

[OpenAI Whisper](https://github.com/openai/whisper) remains the most widely used speech-to-text model. It is available both as an API and as an open-source model you can run locally.

```python
from openai import OpenAI

client = OpenAI()

# API transcription
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.mp3", "rb"),
    language="en",
    response_format="verbose_json",
    timestamp_granularities=["segment"],
    prompt="ZyntriQix, Currentex",  # hint domain-specific terms
)
```

As of 2026-02-10, the Whisper API (`whisper-1`) costs $0.006 per minute. OpenAI also offers `gpt-4o-transcribe`, which uses the GPT-4o model for transcription and achieves lower word error rates, at $0.01 per minute (audio input) plus standard text output pricing.

For local deployment, [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based) runs approximately 4x faster than the original, and [Distil-Whisper](https://github.com/huggingface/distil-whisper) offers 6x speedup with minimal quality loss. These are practical options for teams processing large volumes of audio or requiring data sovereignty.

**Practical tips for audio pipelines:**
- Split files longer than 25 MB at natural pauses (silence detection) before sending to the API.
- Use the `prompt` parameter to hint domain-specific terminology, product names, or acronyms that the model might otherwise misspell.
- For speaker identification, pair Whisper with [pyannote](https://github.com/pyannote/pyannote-audio) for diarization (identifying who spoke when).
- Chain transcription with an LLM for summarization, action item extraction, or sentiment analysis.

### Native Audio Understanding

As of 2026-02-10, [Google Gemini](https://ai.google.dev/gemini-api/docs/audio) is the leader in native audio understanding. Gemini can process audio directly --- not just transcribe it, but understand tone, emotion, speaker characteristics, and musical content. Audio consumes approximately 25 tokens per second. You can upload audio files via the Files API and prompt the model to analyze content, detect speakers, or answer questions about the recording.

OpenAI's [Realtime API](https://platform.openai.com/docs/guides/realtime) enables live audio conversations with GPT-4o, supporting both audio input and output. This is designed for voice assistant applications rather than batch audio processing.

Anthropic does not currently offer audio input capabilities.

## Video

Video understanding is the least mature multimodal capability but is advancing rapidly.

### Native Video Support

As of 2026-02-10, [Google Gemini](https://ai.google.dev/gemini-api/docs/vision) is the only major provider with native video input. Gemini accepts MP4 files up to approximately 1 hour at default resolution, consuming about 263 tokens per second of video. You upload video via the Files API and prompt the model to describe, analyze, or answer questions about the content. Gemini can also accept YouTube URLs directly in some configurations.

Neither OpenAI nor Anthropic offer native video input as of this writing.

### Frame Sampling for Non-Native Providers

For providers without native video support, the standard approach is to extract frames from the video and send them as images. The choice of sampling strategy significantly affects both cost and quality:

| Strategy | Method | Best For |
|----------|--------|----------|
| Uniform sampling | Extract 1 frame per second (or per N seconds) | General-purpose video understanding |
| Scene-change detection | Extract frames when visual content changes significantly | Videos with distinct scenes |
| Keyframe extraction | Extract I-frames from the video codec | Quick, no extra compute needed |
| Adaptive sampling | Dense during action, sparse during static periods | Surveillance, sports, presentations |

```bash
# Uniform: 1 frame per second
ffmpeg -i video.mp4 -vf "fps=1" frames/frame_%04d.jpg

# Scene-change detection (threshold 0.3)
ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)'" -vsync vfr frames/scene_%04d.jpg
```

**Rule of thumb:** 8--32 frames is sufficient for most video understanding tasks. For longer videos, consider a two-pass approach: sample sparsely first to identify interesting segments, then sample densely within those segments. Watch your token budget --- 32 high-detail images can easily consume 25,000+ tokens.

## Document Processing with Vision Models

One of the highest-value applications of multimodal models is document processing: extracting structured data from PDFs, scanned documents, receipts, invoices, forms, and screenshots. Vision models can replace or augment traditional OCR pipelines.

### Vision LLM vs. Traditional OCR

The choice between a vision LLM and traditional OCR depends on your documents and requirements:

| Scenario | Traditional OCR | Vision LLM |
|----------|----------------|-------------|
| Standard printed forms | 99%+ accuracy | 95--98% accuracy |
| Variable / unfamiliar layouts | 80--90% accuracy | 92--97% accuracy |
| Handwriting | 60--80% accuracy | 85--95% accuracy |
| Mixed content (text + charts + images) | Text extraction only | Holistic understanding |
| Semantic understanding (what does this mean?) | Not possible | Strong |
| Cost per 1,000 pages | $0.09--$1.50 | $5--$50+ |
| Speed | Fast (milliseconds) | Slower (seconds) |

**When to use traditional OCR:** High-volume processing of standardized forms where cost and speed matter more than flexibility. Well-established tools include [Amazon Textract](https://aws.amazon.com/textract/), [Google Cloud Document AI](https://cloud.google.com/document-ai), and [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/).

**When to use a vision LLM:** Variable layouts, handwritten content, mixed media, or when you need semantic understanding (not just text extraction but comprehension of what the document means). Also useful for documents where the structure itself carries meaning --- tables, flowcharts, org charts.

**Hybrid approach:** Use OCR for text extraction (fast, cheap, high accuracy on printed text) and feed the extracted text to an LLM for semantic understanding, classification, and structured extraction. This gets the best of both worlds for many use cases.

### Best Practices for Document Extraction

- **Use structured output.** Pair vision input with [JSON Schema constraints](11-structured-outputs-and-tool-calling.md) to enforce the expected output format. This dramatically reduces format errors.
- **Render at adequate resolution.** For PDFs, render at 200--300 DPI. Ensure the shortest side is at least 768px. Low resolution is the most common cause of extraction failures.
- **Crop regions of interest.** For dense documents, crop to the specific region you need (a table, a signature block, a specific field) rather than sending the entire page. This reduces token cost and improves accuracy.
- **Validate numerical data.** Vision models hallucinate exact numbers from charts and tables. Cross-validate extracted numbers against known constraints (totals should sum, dates should be plausible, percentages should add to 100%).
- **Provide context.** Tell the model what kind of document it is looking at and what information you need. "Extract the invoice number, date, and line items from this invoice" performs much better than "What's in this image?"

Specialized document models are worth evaluating for high-volume use cases. [Mistral OCR](https://mistral.ai/products/mistral-ocr) (2025) is optimized for document understanding. [Florence-2](https://huggingface.co/microsoft/Florence-2-large) (Microsoft) handles dense captioning and OCR. [olmOCR](https://github.com/allenai/olmocr) (Allen AI) is open-source and competitive.

## Multimodal Embeddings and Retrieval

Standard text embeddings represent text as vectors. Multimodal embeddings extend this to images, audio, and video, enabling cross-modal search: find images using text queries, find text using image queries, or find similar items across modalities.

### CLIP and Cross-Modal Search

[CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training, OpenAI, 2021) is the foundational model for multimodal embeddings. Trained on 400 million image-text pairs, CLIP produces embeddings where images and text describing similar concepts are close together in vector space.

```python
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cpu")

# Encode image and text into the same embedding space
image_features = model.encode_image(preprocess(Image.open("photo.jpg")).unsqueeze(0))
text_features = model.encode_text(clip.tokenize(["a golden retriever", "a black cat"]))

# Cosine similarity determines which text best matches the image
similarity = (image_features @ text_features.T).softmax(dim=-1)
```

CLIP enables zero-shot image classification (compare an image against text labels), image search (find images matching a text query), and reverse image search (find text matching an image). It is the backbone of most production image search systems.

For production use, index CLIP embeddings in a vector database ([FAISS](https://github.com/facebookresearch/faiss), [Qdrant](https://qdrant.tech/), [pgvector](https://github.com/pgvector/pgvector)) just as you would text embeddings. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md) for indexing and retrieval patterns.

### Unified Multimodal Embeddings

Newer embedding models go beyond CLIP's image-text pairing to support unified embeddings across more modalities:

- **[Cohere Embed v4](https://docs.cohere.com/docs/multimodal-embeddings)** supports text and images in a single embedding call, with configurable dimensions (256--1536) and support for tables, graphs, and handwriting.
- **[Voyage multimodal-3](https://blog.voyageai.com/2024/11/12/voyage-multimodal-3/)** produces state-of-the-art multimodal embeddings as of late 2024.
- **[Jina CLIP v1](https://jina.ai/news/jina-clip-v1-a-truly-multimodal-embeddings-model-for-text-and-image/)** is an open-source alternative that outperforms the original CLIP.
- **[Google Multimodal Embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings)** (Vertex AI) support image, text, and video in a single embedding space.

### Multimodal RAG Architectures

There are three primary approaches to building RAG systems that handle both text and images:

**Separate indices with fusion.** Embed text chunks with a text embedding model and image content with CLIP (or similar) into separate vector indices. At query time, search both indices and merge results. This is the simplest approach and works well when text and images are largely independent.

**Unified multimodal embeddings.** Use a single embedding model (Cohere Embed v4, Voyage multimodal-3) that handles both text and images, storing everything in one index. This simplifies the architecture and enables natural cross-modal retrieval. The tradeoff is that unified models may not match the quality of specialized models on either modality alone.

**Visual document retrieval ([ColPali](https://arxiv.org/abs/2407.01449)/ColQwen).** Treat each document page as a screenshot and embed the visual representation directly, bypassing OCR and text chunking entirely. This approach uses multi-vector embeddings with late interaction matching. It is particularly effective for documents where layout carries meaning (tables, forms, slides) and eliminates the error-prone OCR step. See the [ColPali paper](https://arxiv.org/abs/2407.01449) and [implementation](https://github.com/illuin-tech/colpali).

## Multimodal Security

Multimodal inputs introduce attack surfaces that do not exist in text-only systems. See [Safety, Privacy, and Security](06-safety-privacy-security.md) for the broader security framework; this section covers multimodal-specific threats.

### Visual Prompt Injection

Just as text inputs can contain instructions that override system behavior, images can contain hidden or embedded text that the model reads and follows. An attacker can embed instructions in an image --- as small text, as part of a pattern, or as steganographic content --- that manipulate the model's behavior.

Research presented at [ICML 2025](https://arxiv.org/html/2509.05883v1) demonstrated that visual prompt injection attacks achieve an 84.8% success rate against GPT-4o. The [mind map injection technique](https://www.mdpi.com/2079-9292/14/10/1907) achieved over 90% attack success rates on both GPT-4o and Gemini by embedding instructions within mind map diagrams. These are not theoretical concerns --- they work against production models today.

### Cross-Modal Attacks

Cross-modal attacks exploit the boundary between modalities. An image that looks innocuous to a human viewer may contain text that the model reads and interprets as instructions. Audio files can contain ultrasonic or low-volume content that the model processes but humans cannot hear. Documents can contain white-on-white text that is invisible to human readers but visible to the model.

### Defenses

Defenses against multimodal injection mirror text-based defenses but require modality-specific implementation:

- **Modality-specific sanitization.** Strip metadata from images (EXIF data can contain instructions). Normalize audio levels. Render documents at standard resolution to eliminate hidden content.
- **Content isolation.** Process untrusted multimodal content separately from trusted instructions. Do not include user-uploaded images in the same message as sensitive system prompts if avoidable.
- **Output validation.** Validate model outputs against expected schemas and content policies regardless of input modality. The same output controls that protect against text-based injection protect against visual injection.
- **The [spotlighting](https://arxiv.org/abs/2403.14720) approach** (Microsoft, 2024) marks data boundaries within multimodal inputs, helping the model distinguish between instructions and content.

## Evaluation of Multimodal Systems

### Benchmarks

The multimodal evaluation landscape has standardized around several benchmarks:

| Benchmark | What It Measures | Scale | Link |
|-----------|-----------------|-------|------|
| [MMMU](https://mmmu-benchmark.github.io/) | College-level multimodal reasoning across 30 subjects | 11.5K questions | [Paper](https://arxiv.org/abs/2311.16502) |
| [MMMU-Pro](https://arxiv.org/abs/2409.02813) | Harder variant of MMMU with augmented challenge | Extended | [Paper](https://arxiv.org/abs/2409.02813) |
| [MMBench](https://github.com/open-compass/MMBench) | 20 ability dimensions for vision-language models | ~3K questions | [GitHub](https://github.com/open-compass/MMBench) |
| [Video-MME](https://github.com/MME-Benchmarks/Video-MME) | Video understanding across durations and domains | Varies | [GitHub](https://github.com/MME-Benchmarks/Video-MME) |
| TextVQA | OCR + visual question answering | 45K questions | Standard |
| ChartQA | Chart and graph understanding | 32K questions | Standard |
| DocVQA | Document visual question answering | 50K questions | Standard |

These benchmarks are useful for comparing models but do not replace task-specific evaluation on your own data. A model that scores well on MMMU may still fail on your specific document types or image formats.

### Practical Evaluation Strategies

For production multimodal systems, build evaluation around your actual use cases:

**Ground truth datasets.** Label a representative set of inputs with expected outputs. Measure field-level accuracy (exact match, fuzzy match, semantic match) for extraction tasks. Measure description quality with model-graded evaluation for understanding tasks.

**Hallucination detection.** Track false positives (the model "sees" something that is not there) separately from false negatives (the model misses something that is present). Vision models hallucinate objects, text, and relationships. For document extraction, cross-validate extracted values against known constraints.

**Cost and latency tracking.** Multimodal requests are significantly more expensive than text-only requests. Track cost per document, cost per image, and latency per request. These metrics often determine whether a multimodal approach is viable at scale.

## Pitfalls

**Sending images at unnecessarily high resolution.** Every extra pixel costs tokens. Use `detail: "low"` (or equivalent) for classification and triage tasks. Reserve high resolution for tasks that genuinely need it (OCR, small text, fine detail).

**Assuming vision models can count.** Models struggle with counting more than ~10 items in an image. If you need accurate counts, use traditional computer vision (object detection) or ask the model to list and number items individually.

**Trusting extracted numbers without validation.** Vision models hallucinate numerical values from charts, tables, and financial documents. Always validate extracted numbers against constraints (sums, ranges, plausibility checks).

**Processing video by sending every frame.** Token costs explode with many images. Use frame sampling strategies and start with the minimum number of frames that captures the content. 8--32 frames is usually sufficient.

**Ignoring visual prompt injection.** If your system processes user-uploaded images, those images are an injection surface. Apply the same skepticism to image content that you apply to user text input.

**Building complex multi-model pipelines when a single multimodal model suffices.** Evaluate whether the multimodal model can handle the task directly before building OCR-to-LLM or transcription-to-LLM chains. The simpler architecture is usually more reliable.

**Neglecting cost at scale.** Processing 1,000 images with a frontier vision model can cost $2--$50+ depending on resolution and provider. Model selection, resolution settings, and caching strategies matter enormously for cost-sensitive applications.

## Checklist

- Have you selected the appropriate resolution/detail level for your task (not defaulting to maximum)?
- Are you using structured output for extraction tasks to enforce the expected schema?
- Do you validate extracted numerical data against known constraints?
- For document processing, have you compared vision LLM vs. traditional OCR vs. hybrid for your specific document types?
- Are user-uploaded images treated as untrusted input (potential injection surface)?
- Is your frame sampling strategy for video appropriate for the content type?
- Do you track per-request cost and token consumption for multimodal inputs?
- Have you built task-specific evals rather than relying solely on public benchmarks?
- For audio pipelines, are you handling domain-specific terminology (via prompts or post-processing)?
- If using multimodal embeddings, have you evaluated cross-modal retrieval quality on your data?

## References

### Foundational Papers
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021). https://arxiv.org/abs/2103.00020
- LLaVA: "Visual Instruction Tuning" (Liu et al., 2023). https://arxiv.org/abs/2304.08485
- Flamingo: "A Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022). https://arxiv.org/abs/2204.14198
- GPT-4V System Card (OpenAI, 2023). https://cdn.openai.com/papers/GPTV_System_Card.pdf
- ColPali: "Efficient Document Retrieval with Vision Language Models" (2024). https://arxiv.org/abs/2407.01449

### Surveys
- "A Survey on Multimodal Large Language Models" (2024). https://arxiv.org/abs/2306.13549
- "Hallucination of Multimodal Large Language Models: A Survey" (2025). https://arxiv.org/abs/2404.18930
- "A Survey of Multimodal RAG" (2025). https://arxiv.org/abs/2504.08748

### Provider Documentation
- OpenAI Vision guide. https://platform.openai.com/docs/guides/images-vision
- OpenAI Speech-to-Text guide. https://platform.openai.com/docs/guides/speech-to-text
- Anthropic Vision guide. https://docs.anthropic.com/en/docs/build-with-claude/vision
- Google Gemini Image Understanding. https://ai.google.dev/gemini-api/docs/image-understanding
- Google Gemini Audio guide. https://ai.google.dev/gemini-api/docs/audio

### Security
- "Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs" (2025). https://arxiv.org/abs/2509.05883
- Spotlighting: "Defending Against Indirect Prompt Injection Attacks" (Microsoft, 2024). https://arxiv.org/abs/2403.14720
- OWASP LLM01:2025 --- Prompt Injection. https://genai.owasp.org/llmrisk/llm01-prompt-injection/

### Tools and Models
- Whisper (OpenAI). https://github.com/openai/whisper
- Faster-Whisper. https://github.com/SYSTRAN/faster-whisper
- Cohere Embed v4 (multimodal). https://docs.cohere.com/docs/multimodal-embeddings
- FAISS. https://github.com/facebookresearch/faiss
- ColPali implementation. https://github.com/illuin-tech/colpali

---
[Contents](README.md) | [Prev](13-staying-current.md) | [Next](09-glossary.md)
