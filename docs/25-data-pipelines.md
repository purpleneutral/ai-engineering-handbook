# Data Pipelines For AI

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](24-prompt-management.md) | [Next](26-caching-and-latency.md)

## Summary

Data pipelines are the upstream machinery that converts messy real-world documents into clean, structured, searchable content that AI systems can use. If your AI application answers questions from a knowledge base, processes uploaded documents, or searches internal files, a data pipeline is what prepares that content for the AI to work with. This is the unglamorous foundation that determines the ceiling of every downstream task: if your ingestion pipeline drops tables from PDFs, corrupts encodings, or silently skips documents, no amount of prompt engineering or model sophistication will compensate. This chapter is deeply technical --- it covers the end-to-end pipeline from raw document acquisition through parsing, cleaning, chunking, and quality measurement, with practical guidance on libraries, orchestration, and operating these systems at scale.

## See Also
- [Retrieval-Augmented Generation (RAG)](03-rag.md)
- [Embeddings And Vector Search](12-embeddings-and-vector-search.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Evals And Testing](05-evals.md)

## When To Use

Every AI system that consumes documents needs a data pipeline. If your system reads PDFs, scrapes web pages, processes emails, ingests Office documents, or indexes any form of unstructured content, you are building a data pipeline whether you recognize it or not. The question is not whether you need one but whether you build it deliberately or let it emerge as a tangle of ad hoc scripts.

You need a deliberate pipeline when document volume exceeds what you can inspect manually, when documents arrive in multiple formats, when you need reproducible processing (rerun after a bug fix and get the same results), when data quality directly impacts a user-facing system, or when you need to update the corpus incrementally as new documents arrive.

You can get away with simpler approaches -- loading a handful of files directly into context, for example -- when the corpus is small (tens of documents), stable (rarely changes), and homogeneous (single format, clean content). But systems that start simple have a way of growing, and retrofitting a proper pipeline onto a system that was never designed for one is significantly harder than building it right the first time.

## How It Works

### The Pipeline Stages

A document ingestion pipeline follows a consistent sequence of stages regardless of the specific technologies involved. Understanding this sequence helps you diagnose where quality problems originate and where to invest optimization effort.

```mermaid
flowchart LR
  A[Acquire] --> B[Detect format]
  B --> C[Parse / extract text]
  C --> D[Clean and normalize]
  D --> E[Extract metadata]
  E --> F[Chunk]
  F --> G[Quality check]
  G --> H[Embed / index]
```

Each stage transforms the data and each stage can introduce errors. A robust pipeline treats each stage as an explicit, testable step with its own inputs, outputs, and failure modes.

### Document Acquisition

The first challenge is getting documents into the pipeline reliably. Sources include file systems, S3 buckets, SharePoint, email inboxes, web crawlers, API endpoints, and database exports. Each source has its own access patterns, authentication requirements, and failure modes.

Design your acquisition layer to be **idempotent**: processing the same document twice should produce the same result without creating duplicates. This means assigning a stable document identifier (a content hash is reliable; a file path is not, because files move) and tracking which documents have been processed. Idempotency is what makes it safe to retry after failures and to reprocess after bug fixes.

**Incremental vs. full reprocessing** is a fundamental design choice. Incremental processing handles only new or changed documents, which is faster and cheaper but requires reliable change detection. Full reprocessing rebuilds the entire index from scratch, which is simpler and guarantees consistency but can be prohibitively expensive for large corpora. In practice, you want both: incremental processing for routine updates and the ability to trigger a full reprocess when you change your parsing logic, chunking strategy, or embedding model.

Change detection can use file modification timestamps, content hashes, or source-provided change feeds (S3 event notifications, database change data capture, email IMAP flags). Content hashes are the most reliable because they detect actual changes regardless of timestamp manipulation, but they require reading the full file to compute.

### Format Detection

Never trust file extensions. A `.pdf` file might be a scanned image masquerading as a PDF. A `.docx` file might actually be an `.xlsx` renamed by a confused user. A `.txt` file might be UTF-16 encoded with a BOM, or it might be Latin-1 without any encoding declaration.

Use **magic bytes** (the first few bytes of a file that identify its format) for reliable format detection. The [python-magic](https://github.com/ahupp/python-magic) library wraps libmagic and handles this well. For more nuanced detection, combine magic bytes with extension and MIME type as a tiebreaker.

```python
import magic

def detect_format(file_path: str) -> str:
    mime = magic.from_file(file_path, mime=True)
    # mime will be something like 'application/pdf', 'text/html',
    # 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    return mime
```

**Encoding detection** for text files deserves special attention. The [charset-normalizer](https://github.com/Ousret/charset-normalizer) library (which replaced chardet in the requests library) provides reliable encoding detection. When encoding detection fails or produces low-confidence results, log a warning and fall back to UTF-8 with error replacement rather than crashing the pipeline. Silent encoding errors produce garbled text that will degrade embedding quality and retrieval accuracy downstream.

```python
from charset_normalizer import from_path

def read_text_file(file_path: str) -> str:
    result = from_path(file_path)
    best = result.best()
    if best is None:
        raise ValueError(f"Cannot detect encoding for {file_path}")
    return str(best)
```

### Parsing Strategies And Libraries

Parsing is where the most consequential quality decisions happen. Different document formats require different strategies, and no single library handles everything well.

**PDF parsing** is the hardest common case because PDFs are a page-description format, not a document-structure format. A PDF encodes instructions for drawing characters at specific coordinates on a page; it does not inherently know what a paragraph is, where a table starts, or how columns are laid out. The major approaches are:

[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) is fast and handles most straightforward PDFs well. It extracts text by reading the character stream in the order it appears in the PDF, which is usually correct for single-column documents but can produce garbled output for multi-column layouts, complex tables, or documents with text boxes. PyMuPDF also provides page-level image extraction and basic structure detection.

```python
import fitz  # PyMuPDF

def extract_pdf_text(file_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        pages.append({
            "page": page_num + 1,
            "text": page.get_text("text"),
        })
    return pages
```

[pdfplumber](https://github.com/jsvine/pdfplumber) is built on top of [pdfminer.six](https://github.com/pdfminer/pdfminer.six) and excels at table extraction. It provides fine-grained access to character positions, lines, and rectangles, which makes it possible to reconstruct table structures that other parsers miss. It is slower than PyMuPDF but significantly more capable for documents with complex layouts.

[Unstructured](https://github.com/Unstructured-IO/unstructured) is a higher-level library that handles multiple document formats (PDF, DOCX, HTML, email, images) through a unified API. It combines multiple extraction strategies including layout analysis, OCR, and table detection, and it produces structured elements (titles, paragraphs, tables, lists) rather than raw text. This structure-awareness makes it particularly useful for pipelines that need to preserve document hierarchy.

```python
from unstructured.partition.auto import partition

elements = partition(filename="report.pdf")
for element in elements:
    print(f"{element.category}: {element.text[:100]}")
    # Output like: "Title: Q3 Financial Report"
    #              "NarrativeText: Revenue increased by 12%..."
    #              "Table: | Quarter | Revenue | ..."
```

[Apache Tika](https://tika.apache.org/) is a Java-based content analysis toolkit that handles an enormous range of formats (over 1,000 file types). It runs as a server process and is typically accessed from Python via [tika-python](https://github.com/chrismattmann/tika-python). Tika is the right choice when you need broad format coverage without maintaining format-specific parsing code, but its text extraction for any single format is rarely best-in-class.

[Docling](https://github.com/DS4SD/docling), developed by IBM Research, focuses on document understanding with strong layout analysis and table extraction. As of 2026-02-10, it provides high-quality PDF conversion with structure preservation, making it a strong option for documents with complex layouts, academic papers, and technical documents with figures and tables.

The right strategy is often to combine libraries. Use format detection to route documents to the parser that handles their format best. For PDFs specifically, you might use PyMuPDF as a fast default and fall back to pdfplumber or Unstructured for documents where PyMuPDF's output fails quality checks (low character count relative to page count, garbled text patterns, missing table content).

**HTML parsing** is deceptively complex because real-world HTML is rarely clean. Web pages contain navigation, ads, headers, footers, cookie banners, and scripts that you want to strip, leaving only the main content. Libraries like [trafilatura](https://github.com/adbar/trafilatura) and [readability-lxml](https://github.com/buriy/python-readability) are designed for main content extraction and work well for article-style pages. For more structured scraping where you need specific elements, [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) combined with a clean rendering step is the standard approach.

**Office documents** (DOCX, XLSX, PPTX) are ZIP archives containing XML. [python-docx](https://python-docx.readthedocs.io/) handles Word documents, [openpyxl](https://openpyxl.readthedocs.io/) handles Excel, and [python-pptx](https://python-pptx.readthedocs.io/) handles PowerPoint. These libraries provide structured access to document content, which is a significant advantage over PDF parsing. However, they only handle the Office Open XML formats (.docx, .xlsx, .pptx); older binary formats (.doc, .xls, .ppt) require either LibreOffice conversion or Apache Tika.

**Email** (RFC 5322, .eml, .msg) parsing needs to handle MIME structure, encodings, attachments, and inline images. Python's built-in `email` library handles .eml files. For Outlook .msg files, [extract-msg](https://github.com/TeamMsgExtractor/msg-extractor) is the standard library. Treat email attachments as separate documents that re-enter the pipeline for their own format detection and parsing.

### OCR For Scanned Documents

When a PDF contains scanned images rather than digital text, or when you are processing standalone images of documents, optical character recognition (OCR) is required. Detection of whether a PDF needs OCR is straightforward: if a page has images but minimal extractable text, it is likely scanned.

[Tesseract](https://github.com/tesseract-ocr/tesseract) is the open-source standard for OCR. It supports over 100 languages, runs locally, and produces reasonable results for clean scans. Access it from Python via [pytesseract](https://github.com/madmaze/pytesseract). Tesseract struggles with low-resolution scans, skewed images, complex layouts, and handwriting.

```python
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

def ocr_pdf_page(pdf_path: str, page_num: int, dpi: int = 300) -> str:
    """OCR a single page of a scanned PDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    # Render page to image at specified DPI
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text
```

**Preprocessing** dramatically improves OCR quality. Deskewing (correcting rotation), binarization (converting to black and white), denoising, and resolution upscaling can turn a failed OCR attempt into a usable one. The [OpenCV](https://opencv.org/) library provides the image processing primitives needed for these operations.

**Cloud OCR APIs** from AWS (Textract), Google Cloud (Document AI), and Azure (AI Document Intelligence) offer significantly higher accuracy than Tesseract, especially for complex layouts, tables, and forms. They also handle layout analysis, table extraction, and key-value pair extraction natively. The tradeoff is cost, latency, and data privacy: your documents leave your infrastructure. For sensitive documents, this may be unacceptable.

Choose your OCR strategy based on your quality requirements and constraints. Tesseract is sufficient for clean scans of standard documents. Cloud APIs are worth the cost for documents with complex layouts, poor scan quality, or when extraction accuracy directly impacts downstream business decisions.

### Metadata Extraction

Every document carries metadata beyond its text content, and capturing this metadata is essential for filtering, access control, citation, and debugging.

**Intrinsic metadata** comes from the document itself: title, author, creation date, modification date, page count, language, and format-specific properties (PDF producer, Word template, email headers). Extract this during parsing and store it alongside the text.

**Extrinsic metadata** comes from the document's context: the source system (which SharePoint site, which S3 bucket, which email account), the collection it belongs to, access control labels, the ingestion timestamp, and any organizational taxonomy tags. This metadata typically comes from the acquisition layer and must be threaded through the entire pipeline.

**Derived metadata** is computed during processing: detected language, document category (inferred from content), summary, key entities, and quality scores. This metadata can be generated by heuristics, by classifiers, or by LLM-based extraction.

Store metadata in a structured format alongside your document index. You will need to query on it (find all documents from source X modified after date Y that the current user has permission to see), so it should be in a system that supports efficient filtering -- typically the same database or vector store that holds your chunks.

### Data Cleaning And Normalization

Raw extracted text is almost never ready for embedding or retrieval without cleaning. The specific cleaning steps depend on the source format, but common operations include:

**Whitespace normalization:** Collapse multiple spaces, remove trailing whitespace, normalize line endings. PDF extraction frequently produces erratic whitespace because character positioning does not always map cleanly to word and line boundaries.

**Boilerplate removal:** Strip headers, footers, page numbers, copyright notices, and other repeated content that appears on every page. These fragments dilute embeddings and waste context window space. Detect boilerplate by looking for text that appears identically (or near-identically) across multiple pages of the same document.

**Encoding repair:** Fix mojibake (garbled text from encoding mismatches), replace smart quotes and special characters with their standard equivalents, and normalize Unicode (NFC normalization is the standard choice). The [ftfy](https://github.com/rspeer/python-ftfy) library automates many of these repairs.

```python
import ftfy
import unicodedata
import re

def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)                        # Fix encoding issues
    text = unicodedata.normalize("NFC", text)         # Normalize Unicode
    text = re.sub(r"[ \t]+", " ", text)               # Collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)            # Limit consecutive newlines
    text = text.strip()
    return text
```

**Table linearization:** Tables extracted from PDFs often lose their structure. If you detect table content (via pdfplumber, Unstructured, or a cloud API), convert it to a format that preserves relationships between cells. Markdown tables work well for simple structures. For complex tables, a structured representation (JSON or CSV) embedded in the text with a clear delimiter can be more reliable.

**OCR error correction:** OCR output frequently contains systematic errors (1/l/I confusion, 0/O confusion, missing spaces). Post-processing with a spell checker or a small language model can improve quality, but be careful not to "correct" domain-specific terminology.

### Chunking For Ingestion

Chunking is discussed in the [RAG chapter](03-rag.md) from the retrieval perspective. From the pipeline perspective, the key considerations are slightly different.

**Chunk at ingestion time, not at query time.** Pre-computing chunks allows you to embed and index them once, and it ensures that every query hits the same chunk boundaries. Chunking at query time would require re-processing documents on every search, which is prohibitively expensive and introduces inconsistency.

**Preserve document structure in chunk boundaries.** The best chunk boundaries align with the document's own structure: section breaks, paragraph boundaries, list item boundaries. This requires that your parsing step preserves structural information (headings, lists, tables) rather than producing a flat text stream. Libraries like Unstructured that produce structured elements make this significantly easier.

**Attach parent context to each chunk.** A chunk that says "the third option is preferred" is meaningless without knowing what the three options were. Include enough context with each chunk for it to be interpretable in isolation. Common techniques include prepending the section hierarchy (e.g., "Document: User Guide > Chapter 3: Configuration > Section 3.2: Advanced Settings") and including a small overlap with adjacent chunks.

**Handle tables and figures as atomic units.** A table split across two chunks is useless in both. If a table fits within your chunk size limit, keep it as a single chunk. If it exceeds the limit, consider summarizing it or splitting by logical groupings (rows or column groups) rather than by raw character count.

### Deduplication

Duplicate documents are a common and underestimated problem. The same report uploaded to three different SharePoint sites, forwarded emails with the original attached, versioned documents where only the header date changed -- duplicates waste storage, inflate index size, and bias retrieval toward documents that happen to exist in multiple copies.

**Exact deduplication** uses content hashes (SHA-256 of the normalized text) to identify identical documents. This is fast and precise but misses near-duplicates.

**Near-deduplication** uses techniques like MinHash or SimHash to identify documents that are substantially similar but not identical. This catches versioned documents, documents with different headers or footers, and documents that were reformatted. Near-deduplication is more expensive to compute but catches the cases that matter most in practice.

Deduplicate at the document level before chunking. If you deduplicate only at the chunk level, you save some index space but still waste processing time parsing and cleaning the duplicate document.

### Data Quality Measurement

You cannot maintain pipeline quality without measuring it. Quality measurement should be automated and run on every pipeline execution, not just during development.

**Extraction completeness** measures whether the parser extracted all the content from the document. Compare the character count of extracted text against expected ranges for the document type and page count. A 50-page PDF that yields only 200 characters of text almost certainly had a parsing failure. A page with zero text but nonzero images likely needs OCR.

**Structural integrity** checks whether the extracted structure matches expectations. Do detected headings follow a logical hierarchy? Do tables have consistent column counts across rows? Are list items properly delineated?

**Text quality scoring** uses heuristics to flag problematic extractions. High ratios of non-alphabetic characters suggest garbled content. Extremely long "words" (strings without spaces) suggest concatenation errors. High repetition rates (the same phrase appearing many times) suggest boilerplate that was not removed.

**Sampling and human review** remains necessary even with automated checks. Periodically sample processed documents and have a human verify that the extracted text faithfully represents the original. This catches subtle quality issues that automated heuristics miss and provides ground truth for calibrating your automated checks.

```python
def quality_score(text: str, page_count: int) -> dict:
    """Compute basic quality metrics for extracted text."""
    chars = len(text)
    words = len(text.split())
    chars_per_page = chars / max(page_count, 1)
    avg_word_len = chars / max(words, 1)
    # Flag suspiciously short extractions
    low_content = chars_per_page < 100
    # Flag potential garbled text (average word length > 20 is unusual)
    garbled = avg_word_len > 20
    return {
        "chars": chars,
        "words": words,
        "chars_per_page": round(chars_per_page, 1),
        "avg_word_len": round(avg_word_len, 1),
        "flags": {
            "low_content": low_content,
            "possibly_garbled": garbled,
        },
    }
```

## Design Notes

**Route by format, not by filename.** Build a dispatcher that examines the actual file content (magic bytes, MIME type) and routes to the appropriate parser. This is more robust than switching on file extensions and handles mislabeled files gracefully.

**Make every stage idempotent and restartable.** Pipeline runs fail partway through -- network errors, OOM kills, transient API failures. Each stage should be safe to rerun without producing duplicates or corrupted state. The simplest way to achieve this is to write each stage's output to a new location and only promote it to the "current" location atomically on success.

**Separate parsing from chunking from embedding.** These are distinct concerns with different change frequencies. You will change your chunking strategy more often than your parsing logic, and you will change your embedding model less often than either. Separating them lets you re-chunk without re-parsing, and re-embed without re-chunking, saving significant processing time and cost.

**Store intermediate artifacts.** Keep the raw document, the parsed text, and the chunks as separate artifacts. When you discover a parsing bug and fix it, you want to re-parse from the raw document without re-acquiring it. When you change your chunking strategy, you want to re-chunk from parsed text without re-parsing. Disk is cheap; re-processing at scale is not.

**Use content-addressable storage where practical.** Storing documents and artifacts keyed by their content hash naturally deduplicates, makes caching trivial, and provides a stable identifier that does not change when files are moved or renamed.

**Consider LLM-assisted parsing for high-value documents.** For documents where extraction quality directly impacts business outcomes (contracts, regulatory filings, medical records), you can use an LLM to clean up or validate parser output. This is expensive and slow, so it is not appropriate for bulk processing, but for a targeted subset of critical documents it can dramatically improve quality. Send the raw parsed text alongside an image of the page and ask the model to produce clean, structured text.

## Pipeline Orchestration

For small pipelines processing hundreds of documents, a Python script with error handling and logging is sufficient. As pipelines grow, you need proper orchestration.

**Task queues** ([Celery](https://docs.celeryq.dev/), [RQ](https://python-rq.org/), cloud-native equivalents like AWS SQS with Lambda or GCP Cloud Tasks) work well for pipelines where each document is processed independently. The acquisition step enqueues documents, and worker processes handle parsing, cleaning, chunking, and embedding. This provides natural parallelism, retry handling, and failure isolation.

**Workflow engines** ([Apache Airflow](https://airflow.apache.org/), [Prefect](https://www.prefect.io/), [Dagster](https://dagster.io/)) are appropriate when you need to manage dependencies between stages, schedule periodic runs, monitor pipeline health through a dashboard, and handle complex branching logic (different parsers for different formats, conditional OCR). Dagster's asset-based model is particularly well-suited to data pipelines where intermediate artifacts need to be versioned and tracked.

**Key orchestration requirements** regardless of tool choice: retry with exponential backoff for transient failures, dead-letter queues for documents that fail repeatedly, per-document status tracking (pending, processing, completed, failed), and the ability to reprocess specific documents or date ranges without touching the rest of the corpus.

## Dealing With Scale

Scale introduces problems that do not exist at small volumes. A pipeline that works for 1,000 documents may fail catastrophically at 1,000,000.

**Memory management** becomes critical when processing large documents. A 500-page PDF rendered for OCR at 300 DPI consumes gigabytes of memory. Process pages in batches rather than loading entire documents into memory. Stream text through cleaning stages rather than accumulating it in a single string.

**Parallelism** is essential for throughput but introduces coordination challenges. Process documents in parallel (they are independent), but be mindful of resource limits: API rate limits for cloud OCR, GPU memory for local embedding models, file descriptor limits for concurrent I/O. A bounded worker pool with configurable concurrency is the right pattern.

**Cost management** matters when processing millions of documents through cloud APIs. Estimate costs before starting large jobs. Cloud OCR services charge per page; embedding APIs charge per token. A million-document corpus can easily cost thousands of dollars to process. Consider processing a representative sample first to validate quality before committing to a full run.

**Incremental processing** becomes non-negotiable at scale. Full reprocessing of a million-document corpus might take days and cost thousands of dollars. Design for incremental updates from the start: track which documents have been processed, detect changes, and process only what is new or modified.

## Monitoring Pipeline Health

A data pipeline is an ongoing operational system, not a one-time script. Monitor it with the same rigor you apply to production services.

**Throughput metrics:** documents processed per hour, documents in the queue, processing time per document (broken down by stage). These tell you whether the pipeline is keeping up with incoming documents.

**Quality metrics:** extraction failure rate, OCR confidence scores, quality check failure rate, average text length per page. Track these over time; a gradual decline often indicates a shift in the incoming document mix (more scanned documents, different formats, lower-quality sources).

**Error tracking:** documents that failed processing (with the specific stage and error), documents stuck in retry loops, documents that produced suspiciously short or empty output. Set up alerts for error rate spikes and for individual documents that fail repeatedly.

**Freshness metrics:** time between document creation/modification and availability in the index. If your users expect near-real-time updates and your pipeline has a 6-hour lag, that is an operational issue regardless of whether the pipeline is "working."

## Pitfalls

**Trusting the parser blindly.** Every parser has failure modes. PyMuPDF mangles multi-column layouts. Tesseract hallucinates text on noisy backgrounds. HTML extractors miss content loaded by JavaScript. Build quality checks that catch these failures rather than assuming clean output.

**Ignoring encoding issues.** A single encoding error in a document can propagate through the pipeline and corrupt chunks, embeddings, and search results. Detect and fix encoding issues early, and fail loudly when you cannot.

**Chunking without context.** A chunk that references "the above table" or "as mentioned in section 2" is useless in isolation. Ensure your chunking strategy attaches enough surrounding context for each chunk to stand on its own. This often means including the section title and a brief summary of the document scope.

**Skipping deduplication.** Duplicate documents bias retrieval toward over-represented content, waste embedding budget, and inflate index size. Deduplicate before embedding, not after.

**No intermediate storage.** If your pipeline goes directly from raw document to embedded chunks with no intermediate artifacts, every bug fix requires a full reprocess from acquisition. Store the parsed text and metadata as an intermediate artifact so you can restart from midpoint.

**Building for one format and discovering ten.** Real-world document corpora are messy. You plan for PDFs and discover scanned PDFs, password-protected PDFs, PDFs with embedded fonts that do not map to Unicode, PDF portfolios containing other PDFs, and PDFs that are actually images saved with a .pdf extension. Build your pipeline to handle format diversity from the start, with graceful fallbacks and clear error reporting for formats you have not explicitly handled.

**Underestimating the cost of OCR at scale.** OCR is orders of magnitude slower and more expensive than text extraction from digital documents. Detect which documents actually need OCR (scanned vs. digital) and only OCR those. Do not OCR a digital PDF just because your pipeline routes all PDFs through the same path.

## Checklist
- Is format detection based on file content (magic bytes), not just file extensions?
- Do you handle encoding detection and repair explicitly?
- Can you reprocess from intermediate artifacts without re-acquiring raw documents?
- Is every pipeline stage idempotent and safe to retry?
- Do you deduplicate before embedding?
- Do you have automated quality checks on extracted text (completeness, garbled text detection)?
- Is there a periodic human review of sampled pipeline output?
- Do you track processing status per document (pending, completed, failed)?
- Are error rates and throughput monitored with alerts for anomalies?
- Can you trigger incremental reprocessing for a specific date range or source?
- Do chunks carry enough context (section titles, document metadata) to be interpretable in isolation?
- Is OCR applied selectively to documents that actually need it?

## References
- PyMuPDF documentation. https://pymupdf.readthedocs.io/
- pdfplumber (PDF table and text extraction). https://github.com/jsvine/pdfplumber
- Unstructured library (multi-format document parsing). https://github.com/Unstructured-IO/unstructured
- Apache Tika (content detection and extraction). https://tika.apache.org/
- Docling (document understanding by IBM Research). https://github.com/DS4SD/docling
- Tesseract OCR engine. https://github.com/tesseract-ocr/tesseract
- ftfy (fixes text encoding issues). https://github.com/rspeer/python-ftfy
- charset-normalizer (encoding detection). https://github.com/Ousret/charset-normalizer
- trafilatura (web content extraction). https://github.com/adbar/trafilatura
- Dagster (data orchestration platform). https://dagster.io/

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](24-prompt-management.md) | [Next](26-caching-and-latency.md)
