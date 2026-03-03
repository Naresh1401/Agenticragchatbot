# 3 Fundamental Flaws of LLMs — And How This Agentic RAG Solves Them

Large Language Models (LLMs) like GPT-4 are powerful, but they come with **3 fundamental flaws** that make them unreliable for enterprise and production use. This document explains each flaw, why it's dangerous, and exactly how this Agentic RAG chatbot **detects and auto-corrects** every one of them.

---

## Table of Contents

1. [Flaw 1 — Knowledge Cutoff](#flaw-1--knowledge-cutoff)
2. [Flaw 2 — Hallucination](#flaw-2--hallucination)
3. [Flaw 3 — No Source Attribution](#flaw-3--no-source-attribution)
4. [How the Auto-Correction Pipeline Works](#how-the-auto-correction-pipeline-works)
5. [Before vs After — Side-by-Side Examples](#before-vs-after--side-by-side-examples)
6. [Architecture Diagram](#architecture-diagram)

---

## Flaw 1 — Knowledge Cutoff

### What is it?

Every LLM has a **training data cutoff date**. For example, GPT-4o-mini's knowledge ends at a certain point in 2023–2024. Anything that happened after that date — policy changes, new regulations, updated company handbooks — the LLM simply doesn't know about.

### Why is it dangerous?

Imagine you upload your company's **2025 employee handbook** and ask:
> "What is our current parental leave policy?"

A raw LLM might respond:
> "As of my last update, parental leave policies typically range from 6 to 12 weeks…"

This is **completely wrong** — it's answering from its training data, ignoring your uploaded document entirely. The user gets a generic, outdated answer instead of the actual 2025 policy.

### How RAG solves it (Retrieval)

Instead of relying on the LLM's frozen knowledge, the system **retrieves** relevant chunks from your uploaded documents first:

```
User Question → Embed → Search ChromaDB → Retrieve Top Chunks → Feed to LLM
```

The LLM never answers from memory. It only sees the retrieved document chunks in its context window. This means:
- Upload a **2025 policy** → the LLM answers based on the 2025 policy
- **Delete** the 2025 policy, upload a **2026 revision** → the LLM immediately answers based on the 2026 version
- **No retraining, no fine-tuning, no waiting** — just swap the documents

### How the auto-corrector catches it anyway

Even with retrieval, the LLM sometimes **still** injects training-data phrases like "as of my last update" or "generally speaking, in most cases." The auto-corrector catches these with 15 regex patterns:

**Detection** — `detect_knowledge_cutoff()` in `answer_verifier.py`:

| Pattern Type | Examples Caught |
|---|---|
| Training data disclaimers | "As of my last update", "Based on my training data", "As an AI language model" |
| External knowledge phrases | "Research has shown", "Studies suggest", "Experts say", "It is widely known" |
| Factual claims without citations | LLM states numbers, percentages, or dates but no document chunk supports them |

**Auto-Correction** — `correct_knowledge_cutoff()`:

| What It Does | Example |
|---|---|
| Strips training-data disclaimers | "As of my last update, the policy…" → "The policy…" |
| Replaces external knowledge phrases | "Research has shown that…" → "Based on the uploaded documents…" |
| Adds transparency disclaimer | Appends: "⚠️ Parts of this response could not be verified against uploaded documents" |

### Real-world impact

A company changes its travel reimbursement policy every quarter. Without this correction:
- The LLM might say "Travel reimbursement is typically $50/day" (from training data)
- With this system, it says "According to Q1-2025-Travel-Policy.pdf, reimbursement is $75/day" (from the actual document)

---

## Flaw 2 — Hallucination

### What is it?

LLMs **generate** text — they don't look things up. This means they can (and do) **invent facts** that sound completely plausible but are completely fabricated. This includes:
- Fake statistics ("Revenue grew by 23.7% in Q3")
- Invented dates ("The regulation was enacted on March 15, 2024")
- Fabricated URLs ("See https://company.com/policy/v3 for details")
- Made-up names and entities ("As noted by Dr. Sarah Mitchell in the appendix")

### Why is it dangerous?

Hallucinations are uniquely dangerous because **they look exactly like real answers**. There's no visual cue that the information is fabricated. A user reading "Revenue grew by 23.7%" has no reason to doubt it — the LLM stated it with full confidence.

In enterprise settings, this can lead to:
- Wrong business decisions based on fabricated data
- Legal liability from citing non-existent regulations
- Broken links from invented URLs
- Loss of trust when errors are discovered

### How RAG reduces it (Retrieval + Augmentation)

RAG significantly reduces hallucination by **grounding** the LLM's answer in retrieved document chunks:

```
Retrieved Chunks (actual content from your files)
        ↓
    Injected into LLM's context window
        ↓
    LLM generates answer BASED ON these chunks
```

The system prompt explicitly instructs the LLM:
> "Answer ONLY from the retrieved chunks. If the information is not in the chunks, say so."

This dramatically reduces hallucination — but doesn't eliminate it entirely. LLMs can still inject plausible-sounding facts even when instructed not to.

### How the auto-corrector catches it

**Detection** — `detect_hallucination()` in `answer_verifier.py`:

The detector extracts **every verifiable fact** from the LLM's answer and cross-references it against the retrieved chunks:

| Fact Type | How It's Extracted | How It's Verified |
|---|---|---|
| Numbers & statistics | Regex: `$123`, `45.6%`, `2.5 million` | Must appear somewhere in the chunk text |
| Dates | Regex: `January 15, 2024`, `Q3 2023`, `2024-01-15` | Must appear in the chunk text |
| URLs | Regex: `https://...` | Must appear verbatim in a chunk |
| Proper nouns | Regex: Multi-word capitalized names like `John Smith` | Must appear in the chunk text |

For each fact extracted from the answer, the system checks: **"Does this fact exist anywhere in the retrieved document chunks?"** If not, it's flagged as a potential hallucination.

**Auto-Correction** — `correct_hallucination()`:

| Severity | Action Taken |
|---|---|
| Fabricated URL | Struck through: `~~https://fake.url~~ *(link not found in documents)*` |
| ≥ 3 unverified facts | Adds: "⚠️ **Hallucination Warning:** This response contains **N facts** that could not be verified" |
| 1–2 unverified facts | Adds: "ℹ️ **Note:** The following items could not be verified: [items]" |

### Example

**LLM says:**
> "The annual revenue was $4.2 billion (up 18.3% year-over-year). For more details, visit https://company.com/financials/2024."

**Auto-corrector finds:**
- `$4.2 billion` → ✅ Found in document chunk
- `18.3%` → ❌ Not found in any chunk (LLM invented this)
- `https://company.com/financials/2024` → ❌ Not found (fabricated URL)

**Corrected output:**
> "The annual revenue was $4.2 billion (up 18.3% year-over-year). For more details, visit ~~https://company.com/financials/2024~~ *(link not found in documents)*.
>
> ---
> ℹ️ **Note:** The following items could not be verified in the uploaded documents and may be approximate: 18.3%, https://company.com/financials/2024"

---

## Flaw 3 — No Source Attribution

### What is it?

When you ask a regular LLM a question, it gives you an answer — but **never tells you where that answer came from**. There's no footnote, no page number, no document reference. You have no way to:
- Verify if the answer is correct
- Find the original context for more detail
- Know which document version was used
- Audit the source for compliance purposes

### Why is it dangerous?

In professional settings, **an answer without a source is essentially an opinion**. If someone asks:
> "What are our data retention requirements?"

And the LLM responds:
> "Data must be retained for 7 years for financial records and 3 years for employee records."

Is this from your actual compliance document? From a generic internet article? From the LLM's imagination? Without source attribution, there's no way to tell — and no way to hold anyone accountable for acting on it.

### How RAG solves it (Generation with Citations)

Every answer from this RAG system comes with **full citations** tracking exactly which document, chunk, page, and relevance score the information came from:

```json
{
  "answer": "Data must be retained for 7 years for financial records...",
  "citations": [
    {
      "source": "compliance_handbook_2025.pdf",
      "chunk_id": "compliance_handbook_2025_chunk_12",
      "page_number": 34,
      "score": 0.89,
      "snippet": "Financial records: 7 year retention period..."
    }
  ]
}
```

The user sees:
- **Which file** the answer came from
- **Which page** to check
- **How relevant** the match was (0–100%)
- **The actual text** from the document that supports the answer

### How the auto-corrector enforces it

Even with full citation infrastructure, the LLM might generate a long answer without actually mentioning the source documents. The auto-corrector catches this:

**Detection** — `detect_missing_attribution()` in `answer_verifier.py`:

| Check | Trigger Condition |
|---|---|
| No citations at all | Substantive answer (>100 chars) with factual claims but zero citations |
| Citations exist but not referenced | Answer is >150 chars but doesn't mention any source document by name |

**Auto-Correction** — `correct_missing_attribution()`:

| Scenario | Action |
|---|---|
| Citations available but not referenced | Appends a **📄 Sources** section listing each document, page numbers, and relevance scores |
| No citations at all | Appends: "⚠️ **No Source Attribution:** This response could not be linked to any specific uploaded document" |

### Example

**LLM says:**
> "The company's work-from-home policy allows employees to work remotely up to 3 days per week, with manager approval required for each instance."

**Citations exist** for `HR_Policy_2025.pdf` (page 12, score 0.91) but the answer doesn't mention it.

**Corrected output:**
> "The company's work-from-home policy allows employees to work remotely up to 3 days per week, with manager approval required for each instance.
>
> ---
> 📄 **Sources:**
> - **HR_Policy_2025.pdf** (Pages: 12) — relevance: 91%"

Now the user can **verify** the answer by checking page 12 of the actual document.

---

## How the Auto-Correction Pipeline Works

The auto-correction runs as the **last step** before the answer reaches the user, inside the `output_guard_node` of the LangGraph pipeline:

```
User Question
    │
    ▼
┌──────────────┐
│  Input Guard  │ ← PII redaction, injection detection
└──────┬───────┘
       ▼
┌──────────────┐
│    Agent      │ ← LLM decides which tools to call
└──────┬───────┘
       ▼
┌──────────────┐
│    Tools      │ ← retrieve_documents, query_structured_data
└──────┬───────┘
       ▼
┌──────────────┐
│  Extract      │ ← Parse citations from tool results
│  Citations    │
└──────┬───────┘
       ▼
┌──────────────┐
│  Output Guard │ ← Confidence scoring
│               │
│  ┌──────────┐ │
│  │ VERIFIER │ │ ← ① Detect knowledge cutoff
│  │          │ │   ② Detect hallucination
│  │          │ │   ③ Detect missing attribution
│  │          │ │   → Auto-correct all flaws
│  └──────────┘ │
└──────┬───────┘
       ▼
   Final Answer  ← Clean, verified, cited response
```

### What happens at each step:

1. **`verify_and_correct(answer, citations)`** is called with the LLM's answer and all retrieved citations
2. **Flaw 1 check** → `detect_knowledge_cutoff()` runs 15 regex patterns → `correct_knowledge_cutoff()` strips/replaces
3. **Flaw 2 check** → `detect_hallucination()` extracts facts, cross-references chunks → `correct_hallucination()` flags unverified
4. **Flaw 3 check** → `detect_missing_attribution()` checks citation presence → `correct_missing_attribution()` appends sources
5. Returns `VerificationResult` with `corrected_answer`, `issues` list, and `corrections_applied` list

### The API response includes verification metadata:

```json
{
  "answer": "...(corrected answer)...",
  "citations": [...],
  "verification_issues": [
    {
      "flaw_type": "hallucination",
      "severity": "high",
      "description": "URL was fabricated — not found in any document.",
      "evidence": "https://fake.url/doc"
    }
  ],
  "corrections_applied": [
    "Struck fabricated URL: https://fake.url/doc",
    "Added verification note for: https://fake.url/doc"
  ]
}
```

This gives full transparency into what was detected and corrected.

---

## Before vs After — Side-by-Side Examples

### Example 1: Knowledge Cutoff

| | Raw LLM Response | After Auto-Correction |
|---|---|---|
| **Answer** | "As of my last update in 2023, company leave policies typically offer 10 days of PTO. I don't have access to your specific policy." | "According to Employee_Handbook_2025.pdf, the company offers 15 days of PTO for employees with 1–3 years of tenure, and 20 days for 3+ years." |
| **What changed** | Training-data disclaimer removed; external knowledge replaced with document-grounded answer | Answer now comes from the actual uploaded document |

### Example 2: Hallucination

| | Raw LLM Response | After Auto-Correction |
|---|---|---|
| **Answer** | "The project budget was $2.3 million with a 15.7% contingency reserve. See https://company.com/budget for details." | "The project budget was $2.3 million. ~~https://company.com/budget~~ *(link not found in documents)*.<br><br>ℹ️ **Note:** The following items could not be verified: 15.7%" |
| **What changed** | `$2.3 million` was in the document (kept). `15.7%` was invented (flagged). URL was fabricated (struck). |

### Example 3: No Source Attribution

| | Raw LLM Response | After Auto-Correction |
|---|---|---|
| **Answer** | "Employees must complete safety training within 30 days of their start date. Annual refresher training is mandatory." | "Employees must complete safety training within 30 days of their start date. Annual refresher training is mandatory.<br><br>📄 **Sources:**<br>- **Safety_Manual_2025.pdf** (Pages: 5, 6) — relevance: 94%" |
| **What changed** | Source attribution automatically added so the user knows exactly where to verify the information |

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │          THE 3 LLM FLAWS                │
                    │                                         │
                    │  ┌───────────┐  ┌──────────┐  ┌──────┐ │
                    │  │ Knowledge │  │Hallucin- │  │ No   │ │
                    │  │ Cutoff    │  │ation     │  │Source│ │
                    │  └─────┬─────┘  └────┬─────┘  └──┬───┘ │
                    │        │             │            │     │
                    └────────┼─────────────┼────────────┼─────┘
                             │             │            │
                             ▼             ▼            ▼
                    ┌─────────────────────────────────────────┐
                    │        RAG PIPELINE SOLUTIONS           │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │ RETRIEVAL (ChromaDB)               │  │
                    │  │ → Fetches current documents        │  │
                    │  │ → Replaces LLM's frozen knowledge  │  │
                    │  │ → Solves: Knowledge Cutoff         │  │
                    │  └───────────────────────────────────┘  │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │ AUGMENTATION (Context Injection)   │  │
                    │  │ → Grounds LLM in real document text│  │
                    │  │ → Constrains generation to facts   │  │
                    │  │ → Solves: Hallucination             │  │
                    │  └───────────────────────────────────┘  │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │ GENERATION (Cited Answers)         │  │
                    │  │ → Every answer has document refs   │  │
                    │  │ → Page numbers + relevance scores  │  │
                    │  │ → Solves: No Source Attribution     │  │
                    │  └───────────────────────────────────┘  │
                    │                                         │
                    │  ┌───────────────────────────────────┐  │
                    │  │ AUTO-CORRECTOR (answer_verifier)   │  │
                    │  │ → 15 regex patterns for cutoff     │  │
                    │  │ → Fact extraction + cross-ref      │  │
                    │  │ → Source injection + disclaimers   │  │
                    │  │ → Catches anything RAG missed      │  │
                    │  └───────────────────────────────────┘  │
                    │                                         │
                    └─────────────────────────────────────────┘
```

---

## Summary

| Flaw | Root Cause | RAG Solution | Auto-Corrector Safety Net |
|---|---|---|---|
| **Knowledge Cutoff** | LLM's training data is frozen | Retrieve from current documents | Strips training-data phrases, adds disclaimer |
| **Hallucination** | LLMs generate, not retrieve | Ground answers in chunk text | Cross-references every fact against chunks |
| **No Source Attribution** | LLMs don't track sources | Full citation pipeline | Forces source blocks onto uncited answers |

The key insight: **RAG handles 90% of the problem** by routing answers through real documents. The auto-corrector in `answer_verifier.py` is the **safety net** that catches the remaining 10% — the cases where the LLM still manages to inject training knowledge, fabricate a fact, or skip over its citations.

This makes the system suitable for enterprise use where **accuracy, auditability, and freshness** are non-negotiable.
