---
name: paper-reader
description: Read and process academic papers (arXiv or local PDF) into structured, section-level Markdown files under papers/. Prefers HTML when available; falls back to PDF+MinerU. When the user wants to add/process/archive a paper, produces a multi-file structured breakdown with README index.
argument-hint: <arXiv-URL-or-PDF-path> [short_name]
arguments: [source, short_name?]
disable-model-invocation: false
---

# Paper Reader Skill

This skill helps the AI **read and process academic papers** into a structured, browsable knowledge base. The deliverable is a set of section files under `papers/<short_name>/`, indexed by a `README.md`, with the paper tracked in `CLAUDE.md`.

## Core Principle

> **HTML first, PDF as fallback. Structured multi-file output.**

- **HTML first** — If the publisher provides a high-quality HTML rendering (e.g. `https://arxiv.org/html/<id>`), preprocess it with `tools/arxiv_html_to_md.py` and use the generated Markdown intermediates as the source of truth. No PDF download, no MinerU.
- **PDF fallback** — If HTML is missing or low quality, download the PDF and parse it with MinerU into clean Markdown.
- **Structured output** — Regardless of input format, produce section files + README + CLAUDE.md update.

**Input:**
- `$source` (required) — arXiv URL or local PDF path.
- `$short_name` (optional) — snake_case folder name. **If the user does not provide one, the AI picks it.** Derive it from the paper title: lowercase, underscore-separated, 1–3 words, memorable and unambiguous (e.g. `deepseek_r1`, `grpo`, `capacity_interference`). Avoid generic names. If a folder with that name already exists, pick a different name or ask.

---

## Quick Decision

- HTML available and complete → Strategy 1 (current-agent HTML transcription)
- HTML missing/incomplete, or input is PDF → Strategy 2 (MinerU Markdown)
- All paths converge to **Phase 1–3** below for structured output.

---

## Strategy 1: HTML First (Recommended for arXiv with HTML)

### HTML detection
Extract the arXiv ID from the source when possible, then try `https://arxiv.org/html/<id>` directly with available web tools. Treat the HTML path as available only when the page loads successfully and appears complete. If the page is missing, blocked, truncated, or lacks major sections/equations/references, switch to **Strategy 2**.

### HTML reading model
Use `https://arxiv.org/html/<id>` as the source and read the full HTML paper in the current agent. Verify that the HTML rendering contains the complete paper — all major sections, equations, and references must be present, not truncated or summarized. If the full HTML cannot be fetched or appears incomplete, fall back to **Strategy 2**. Do not dispatch subagents for the HTML path.

Older or unusual papers (scanned PDFs, non-LaTeX) often have no HTML version.

---

## Strategy 2: PDF Fallback

Use when HTML is missing, incomplete, or the input is a PDF.

### PDF conversion
Normalize arXiv IDs or arXiv `abs`/`html`/`pdf` URLs to `https://arxiv.org/pdf/<id>`. For non-arXiv URLs, use the URL as-is. For local PDFs, use the local path directly.

For a remote PDF, download it into the working output directory:
```bash
mkdir -p <OUTPUT_DIR>
curl -L -o <OUTPUT_DIR>/<pdf_base_name>.pdf <PDF_URL>
```

Run MinerU directly on the PDF:
```bash
mineru -p <PDF_PATH> -o <OUTPUT_DIR> -b pipeline
```

Locate the MinerU-generated Markdown and use it as `<parsed.md>`:
```bash
find <OUTPUT_DIR> -name '*.md' -print
```

Use the Markdown file matching the PDF base name when multiple Markdown files exist. If the generated Markdown is nested, copy it to `<OUTPUT_DIR>/<pdf_base_name>.md` for convenience. Do not inject helper content into the parsed Markdown.

---

## Non-arXiv Papers

Same principle applies: prefer HTML from OpenReview / ACL Anthology / etc.; probe directly to decide. Use available web tools and page completeness checks; if no reliable HTML source exists, fall back to PDF/MinerU.

---

## Phase 1 — Outline and Partition Plan

**Goal:** Decide the paper's file layout before writing section files.

**HTML path:** Ensure the full HTML paper has been read. If it has not been read yet, read it now. Then plan the file partition from the full content.

**PDF/MinerU path:** Use grep/search on the parsed Markdown to list all H1–H3 headings, and use that outline as the main partition basis. Example command: `grep -E '^#{1,3}[[:space:]]' <parsed.md>`. (Note: macOS `grep` does not support `\s` in basic/regex mode; use `[[:space:]]` instead. On Linux, `grep -P '^#{1,3}\s'` also works. If the grep returns nothing, check the regex and fall back to reading the first ~100 lines to locate headings manually.) Read only the front matter (title, authors, abstract, introduction) and skim the tail (conclusion/discussion, references position, appendices). Do not read full body sections during planning; methods, experiments, and results are handled during file writing.

Then pick a `short_name` if not provided.

---

## Phase 2 — Write Structured Files

Create `papers/<short_name>/` and write all section files. If using MinerU and an images directory exists, copy it alongside the section files, for example: `cp -r <mineru_output_dir>/auto/images <output_dir>/images`.

**HTML path:** Do not dispatch subagents. The current agent should write the section files directly from the already-read full HTML content, preserving narrative flow and paragraph-level detail.

**PDF/MinerU path:** When sub-agents are available, dispatch them in parallel (1–3 files each) using the **`code-explorer-writer`** subagent type (`.codebuddy/agents/code-explorer-writer.md`). This agent has write permission (`write_to_file`, `replace_in_file`, `delete_file`) in addition to full codebase exploration tools — use it for any task that involves both reading source content and writing section files. Do NOT use the plain `code-explorer` subagent for writing tasks, as it is read-only. Provide each subagent with the parsed Markdown path, assigned section headings/ranges, the content rules below, and `max_turns`. Each must return a 2–5 sentence substantive summary. When writing directly (no subagents), apply the same quality standard — concrete mechanisms, named methods, specific numbers, not generic labels.

### Content Rules

**Primary goal:** Structured transcription, not summarization. The section file should read like the paper itself — preserving narrative flow, argument progression, and paragraph-level detail. Omit only genuinely redundant or off-topic passages.

1. **Preserve the original text.** Keep the paper's own wording, paragraph structure, and argument flow. Remove only: (a) verbatim repetition of a point already made, (b) filler/padding, (c) irrelevant boilerplate (legal, ethics statements, author lists). If the text is clear and concise — keep every sentence. A section that spans several pages in the original should produce a file with commensurate depth, not a bullet-point abstract.
2. **Use tables for structured data** — results, comparisons, hyperparameter configs. Bold the most important values.
3. **Equations — ALWAYS use LaTeX math notation** so they render properly in Markdown viewers:
   - **Block equations** (display math): `$$...$$` on its own line
   - **Inline math**: `$...$` within text
   - **Never** use fenced code blocks (\`\`\`) for equations — code fences prevent math rendering
   - Preserve clean equations from the original paper. If MinerU has garbled an equation (broken symbols, missing terms), include a best-effort LaTeX reconstruction and mark it `[OCR uncertain]` rather than dropping it silently.

### Formatting Conventions

| Element | Format | Example |
|---------|--------|---------|
| Math variables | `$...$` (inline) or `$$...$$` (block) | `$\pi_\theta$`, `$\alpha$`, `$\beta_t$` |
| Model & dataset names | backticks | `` `Llama-3.1-8B` ``, `` `GSM8K` `` |
| Key concepts (1st mention) | **bold** | **chain-of-thought** |
| Important results | **bold** | **96.1%** |
| Citations | `[N]` | Zou et al. [29] |
| Figure / table refs | capitalized | Figure 1, Table 2 |
| Acronyms | introduce once | Attack Success Rate (**ASR**) |
| Equations | `$inline$` or `$$block$$` (NOT code fences) | `$$L(N, D) = L_0 + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$` |

---

## Phase 3 — README + CLAUDE.md Update

### Write `README.md`

Write `papers/<short_name>/README.md` compiling the 2–4 sentence file descriptions. Include the paper title, arXiv ID, lead author, affiliation, year, and code URL header.

**README.md format:**

```markdown
# <Full Paper Title>
arXiv: <ID> | <Lead Author> et al. (<Affiliation>, <Year>)
Code: <GitHub URL or "—">

## Files

- `s01_<name>.md` — <2–5 sentence substantive description>
- `s02_<name>.md` — <2–5 sentence substantive description>
- ...
- `s<NN>_references.md` — Full <N>-entry reference list
- `s<LAST>_<name>.md` — <appendix or other trailing sections, if present>
```

**IMPORTANT:** Each file description should be 2–5 sentences, substantive enough that someone reading only the README gets a real sense of the paper's content. Focus on the core mechanism, key concepts, notable findings, and specific numbers. Generic statements like "this section describes the experiments" are not acceptable.

### Update CLAUDE.md

Add a row to the "Papers index" table in `CLAUDE.md` (create the table if it doesn't exist):
```
| `papers/<short_name>/` | "<Title>" — <Author> et al. (<Affiliation>, <Year>) | <arXiv ID> |
```

---

## File Layout

```
papers/<short_name>/
├── README.md                              # Index with substantive file descriptions
├── s01_<name>.md                          # One file per major section
├── s02_<name>.md
├── ...
├── s<NN>_<MM>_<combined_name>.md          # Optional: 2–3 short/related sections combined
├── s<NN>_references.md                    # Reference list (wherever it appears in the paper)
└── s<LAST>_<name>.md                      # Appendix or last section if present
```

**Naming:** lowercase, underscores, zero-padded two-digit section numbers. Use `s01`–`s99` by default; reserve `s00` only for a separate front-matter/overview file if one is explicitly needed. Multi-section files list all numbers: `s04_05_experiments_conclusion.md`. Keep names compact (2–4 words). Combine sections only when each is short or tightly coupled. Follow the paper's own section order — references and appendices get numbers matching their position.

### Section Files (s01_, s02_, …)

Heading hierarchy using the paper's own numbering:
```markdown
## <N>. Section Title           ← H2
### <N>.<M> Subsection          ← H3
#### <N>.<M>.<P> Sub-subsection ← H4, rare
```

### Multi-Section Files

Separate each section with `---`, each keeping its own `##` heading and numbering.

### References File (s<NN>_references.md)

```markdown
## References

| # | Authors | Title | Venue/Source | Year |
|---|---------|-------|-------------|------|
| [1] | First Author et al. | Full title | NeurIPS / arXiv / GitHub / etc. | 2024 |
```

Include every reference cited in the paper. Use compact author names: "First Author et al." for 3+ authors.

---

