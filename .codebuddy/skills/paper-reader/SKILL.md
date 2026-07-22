---
name: paper-reader
description: Read and process academic papers from arXiv, public HTML pages, or PDFs into structured, section-level Markdown files under papers/. Prefers native arXiv HTML, then MinerU HTML for other accessible webpages, with PDF+MinerU as fallback. When the user wants to add/process/archive a paper, produces a multi-file structured breakdown with README index.
argument-hint: <arXiv-URL-or-public-HTML-URL-or-PDF-path> [short_name]
arguments: [source, short_name?]
disable-model-invocation: false
---

# Paper Reader Skill

This skill helps the AI **read and process academic papers** into a structured, browsable knowledge base. The deliverable is a set of section files under `papers/<short_name>/`, indexed by a `README.md`, with the paper tracked in `CLAUDE.md`.

## Core Principle

> **HTML first, PDF as fallback. Structured multi-file output.**

- **HTML first** — For arXiv, if the publisher provides a high-quality HTML rendering (e.g. `https://arxiv.org/html/<id>`), preprocess it with `tools/arxiv_html_to_md.py` and use the generated Markdown intermediates as the source of truth. No PDF download or MinerU is needed.
- **Generic HTML (optional)** — For a non-arXiv public HTML page, MinerU's `MinerU-HTML` model is an optional extraction tool when direct HTML reading is insufficient or the user requests it. When selected and service access succeeds, it returns extracted `main.html` and `full.md`.
- **PDF fallback** — If no reliable HTML route is available, download the PDF and parse it with MinerU into clean Markdown.
- **Structured output** — Regardless of input format, produce section files + README + CLAUDE.md update.

**Input:**
- `$source` (required) — arXiv URL, public HTML URL, or local/remote PDF path.
- `$short_name` (optional) — snake_case folder name. **If the user does not provide one, the AI picks it.** Derive it from the paper title: lowercase, underscore-separated, 1–3 words, memorable and unambiguous (e.g. `deepseek_r1`, `grpo`, `capacity_interference`). Avoid generic names. If a folder with that name already exists, pick a different name or ask.

**Category subdirectory:** Papers are organized under `papers/<category>/<short_name>/`, not `papers/<short_name>/`. Check `CLAUDE.md` for existing categories (e.g. `rl_policy_optimization`, `peft_lora`, `learning_dynamics_capacity`, `efficient_architectures`, `llm_security`). Pick the best-fitting existing category; create a new one only if no existing category fits, and add it as a new `### <Category Name>` section in `CLAUDE.md`.

---

## Quick Decision

- arXiv HTML available and complete → Strategy 1 (HTML transcription via `arxiv_html_to_md.py`)
- Non-arXiv public HTML page → read the source directly first; optionally select Strategy 1B (MinerU `MinerU-HTML`) when extra extraction is needed and the service can reach the URL
- arXiv HTML missing/incomplete → Strategy 2 (HF Papers — fast, clean formatting, but may be incomplete)
- No reliable HTML/HF route → Strategy 3 (MinerU PDF — cloud VLM API preferred, local pipeline fallback)
- **Always run `hf papers info <arXiv-ID>` for metadata** (title, authors, GitHub repo, etc.) regardless of content strategy.
- All paths converge to **Phase 1–3** below for structured output.

> **Fetch HTML and HF Papers in parallel, not as a strict fallback chain.** `hf papers info` + `hf papers read` + the `arxiv_html_to_md.py` tool are independent — kick them all off at once. Two reasons: (1) metadata is needed regardless of content strategy; (2) HF Papers preserves the connective prose around citations that the HTML parser sometimes drops to bare citation clusters (see "Handling HTML-extraction gaps" below). If HTML succeeds, use it as the primary source (better tables/equations) and mine the HF Papers output only to fill any bare-citation gaps; if HTML fails, HF Papers is already in hand as Strategy 2.

---

## Strategy 1: HTML First (Recommended for arXiv with HTML)

### HTML detection and preprocessing

Extract the arXiv ID from the source, then **run `tools/arxiv_html_to_md.py` directly** — it uses `curl` to download the complete HTML (unlike web-fetch tools which may truncate long papers) and produces structured Markdown intermediates:

```bash
python3 tools/arxiv_html_to_md.py <arXiv-ID-or-URL> --out /tmp/<work_dir>
```

Inspect the generated `outline.json` and `chunks/` directory to verify completeness — all major sections, equations, tables, and references must be present. The tool outputs:
- `source.html` — raw HTML backup
- `parsed.md` — full concatenated Markdown
- `outline.json` — metadata + chunk index
- `references.md` — extracted reference table
- `chunks/sNN_*.md` — one file per section

If the HTML page is missing (404), blocked, or the outline shows missing major sections, switch to **Strategy 2**.

### HTML reading model
Use the generated `chunks/` files as the source of truth. Read the chunk files to understand the full paper content before planning the file partition. If the tool fails or the outline appears incomplete, fall back to **Strategy 2**.

Older or unusual papers (scanned PDFs, non-LaTeX) often have no HTML version.

---

## Strategy 1B: Optional Generic HTML Extraction (MinerU-HTML)

Use only when direct reading of a non-arXiv web page is inadequate, a structured Markdown extraction is useful, or the user explicitly requests MinerU. It is not a required step for public HTML sources. If selected, first verify that the page is publicly accessible, then submit the URL to MinerU. **HTML requires `model_version: "MinerU-HTML"`; do not submit it to the PDF `vlm` model.**

```bash
curl -s --request POST 'https://mineru.net/api/v4/extract/task' \
  --header "Authorization: Bearer $MINERU_API_TOKEN" \
  --header 'Content-Type: application/json' \
  --data-raw '{"url": "<HTML_URL>", "model_version": "MinerU-HTML"}'
# → returns {"data": {"task_id": "..."}}
```

Poll and download the task result with the same commands as Strategy 3, Option A. Use `full.md` as the parsed source and retain `main.html` as the extracted-body backup. Parameters such as `language`, `is_ocr`, `enable_formula`, and `enable_table` do not apply to `MinerU-HTML`.

**Service-access restriction:** MinerU must be able to retrieve the URL from its own servers. If it returns a regional-regulation, network, or access error (as it did for `alignment.anthropic.com`), do not retry the same route. Instead, use a linked paper PDF or another accessible source; then follow Strategy 2 or Strategy 3 as appropriate.

---

## Strategy 2: HF Papers (Fast Markdown, May Be Incomplete)

Use when arXiv HTML is unavailable. HuggingFace's `hf papers read` command converts papers to clean Markdown with proper bold/italic formatting and LaTeX math — often higher quality than MinerU's OCR. However, it can be **incomplete**: tables, images, and later appendices may be missing. Always verify completeness before using.

### Reading the paper

```bash
hf papers read <arXiv-ID> > <OUTPUT_DIR>/hf_parsed.md
```

### Completeness check

After saving, verify the output contains all expected content:
1. **References**: `grep -c 'References' <file>` — confirm a References section with actual entries exists
2. **Appendices**: `grep -E '^#{1,4}[[:space:]].*[A-Z]\.' <file>` — check that all appendix sections are present
3. **Tables**: `grep -c '^|' <file>` — if the paper has tables but this returns 0, tables are missing
4. **Length sanity**: `wc -l <file>` — compare against expected (a 40-page paper should produce ~800+ lines)

If significant content is missing (appendices, references, or all tables), fall back to **Strategy 3** (MinerU). Minor gaps (e.g., a single table) can be supplemented from MinerU while using HF Papers for the main body.

### Limitations to be aware of

- **No tables**: HF Papers often drops tables entirely (no Markdown table syntax in output)
- **No images**: No image references or extraction (MinerU extracts images)
- **Heading hierarchy**: Can be inconsistent (e.g., `###` for a top-level section, `####` for a subsection)
- **Truncation**: May stop before later appendices

---

## Strategy 3: PDF Fallback (MinerU)

Use when no reliable HTML or HF Papers route is available. Prefer the cloud VLM API over local parsing — it produces better block equation quality, doesn't require a local GPU, and is faster.

### PDF conversion
Normalize arXiv IDs or arXiv `abs`/`html`/`pdf` URLs to `https://arxiv.org/pdf/<id>`. For non-arXiv URLs, use the URL as-is. For local PDFs, use the local path directly.

For a remote PDF, download it into the working output directory:
```bash
mkdir -p <OUTPUT_DIR>
curl -L -o <OUTPUT_DIR>/<pdf_base_name>.pdf <PDF_URL>
```

### Option A: Cloud VLM API (preferred)

Submit the PDF URL to the MinerU cloud API with `model_version: "vlm"` for highest quality. Requires a Bearer token from https://mineru.net/apiManage, stored in the `MINERU_API_TOKEN` environment variable.

```bash
# 1. Submit task
curl -s --request POST 'https://mineru.net/api/v4/extract/task' \
  --header "Authorization: Bearer $MINERU_API_TOKEN" \
  --header 'Content-Type: application/json' \
  --data-raw '{"url": "<PDF_URL>", "model_version": "vlm", "language": "en"}'
# → returns {"data": {"task_id": "..."}}

# 2. Poll until state=done (usually 20-60 seconds)
curl -s --request GET "https://mineru.net/api/v4/extract/task/<TASK_ID>" \
  --header "Authorization: Bearer $MINERU_API_TOKEN"
# → state: running → done, returns full_zip_url

# 3. Download and extract result
curl -L -o <OUTPUT_DIR>/result.zip "<FULL_ZIP_URL>"
cd <OUTPUT_DIR> && unzip -o result.zip
# → full.md is the parsed Markdown, images/ contains extracted figures
```

The result zip contains `full.md` (parsed Markdown), `images/` (extracted figures), and JSON intermediates. Use `full.md` as `<parsed.md>`.

For local PDF files, use the batch upload API (`POST /api/v4/file-urls/batch`) to get an OSS upload URL, then PUT the file. See https://mineru.net/apiManage/docs for details.

### Option B: Local MinerU (fallback)

Use when the cloud API is unavailable (no token, quota exhausted, network issues).

```bash
mineru -p <PDF_PATH> -o <OUTPUT_DIR>
```

**Do not specify `-b pipeline` explicitly.** MinerU defaults to `hybrid-engine` (VLM-based, higher accuracy). Only fall back to `-b pipeline` if `hybrid-engine` fails (see known issues below).

Locate the MinerU-generated Markdown:
```bash
find <OUTPUT_DIR> -name '*.md' -print
```

### Known issues

**`hybrid-engine` on Apple Silicon (M1–M4):** Fails with `RuntimeError: There is no Stream(gpu, 1) in current thread` — MLX's GPU context is thread-local and doesn't propagate to MinerU's FastAPI worker threads. If this occurs, fall back to `-b pipeline` or use the cloud VLM API (Option A).

**transformers version incompatibility:** MinerU 3.4.x requires `transformers<5.0`. With `transformers>=5.0`, both `hybrid-engine` and `pipeline -m txt` fail with import errors. If MinerU fails on import, downgrade: `pip install 'transformers<5.0'`.

**Formula quality limitations (all backends):** Inline math may contain character-level errors regardless of backend: spaces in numbers (`1 0` instead of `10`), spaces in `\mathrm{}` blocks, and misidentified symbols (`\dot{2}`, `\gimel`, `\breve{5}`). The VLM cloud API fixes block equations (`\min`, `\mathrm{benign}`, `\text{if}` are clean) but inline math quality is the same as local pipeline. **Specifying `-m txt` does NOT improve formula quality.** These errors are expected to be corrected during the LLM transcription phase (Phase 2), where the agent recognizes and fixes OCR artifacts using its LaTeX knowledge.

---

## Non-arXiv Papers

Prefer a high-quality publisher HTML page and read it directly when its structure is sufficient. Strategy 1B's `MinerU-HTML` model is optional for cases needing structured Markdown extraction; if selected, first confirm that MinerU can access the URL, since a browser-accessible page can still be blocked from MinerU's servers by regional or network restrictions. For non-arXiv papers, `hf papers` commands will not work (they require arXiv IDs). If the HTML route is inaccessible or unreliable, use a linked PDF and Strategy 3.

---

## Phase 1 — Outline and Partition Plan

**Goal:** Decide the paper's file layout before writing section files.

**Metadata:** For arXiv sources, always run `hf papers info <arXiv-ID>` to get clean metadata — title, authors, published date, GitHub repo URL, AI summary, and keywords. This is more reliable than scraping arXiv abs pages and often provides the code URL that arXiv doesn't list. For non-arXiv HTML or PDFs, extract equivalent metadata from the source page or PDF. Use the resulting metadata for the README header in Phase 3.

**arXiv HTML path:** The `arxiv_html_to_md.py` tool already generated `outline.json` (with metadata + chunk index) and `chunks/sNN_*.md` (one per section). Use the outline as the partition basis — inspect chunk titles and sizes to decide which sections to combine or keep separate. Read front matter and skim conclusion/appendix chunks for context.

**Optional MinerU-HTML path:** Only if Strategy 1B was selected, use `full.md` as the source and `main.html` to resolve missing structure or text. List headings with `grep -E '^#{1,3}[[:space:]]' <full.md>`; if the headings are sparse, inspect `main.html` and establish the paper's section order before partitioning.

**HF Papers path:** Use grep/search on the HF Papers Markdown to list all headings: `grep -E '^#{1,4}[[:space:]]' <hf_parsed.md>`. Note that heading hierarchy may be inconsistent — normalize during planning. If the completeness check (Strategy 2) found missing appendices, note which sections are absent and plan to supplement from MinerU.

**PDF/MinerU path:** Use grep/search on the parsed Markdown to list all H1–H3 headings, and use that outline as the main partition basis. Example command: `grep -E '^#{1,3}[[:space:]]' <parsed.md>`. (Note: macOS `grep` does not support `\s` in basic/regex mode; use `[[:space:]]` instead. On Linux, `grep -P '^#{1,3}\s'` also works. If the grep returns nothing, check the regex and fall back to reading the first ~100 lines to locate headings manually.) Read only the front matter (title, authors, abstract, introduction) and skim the tail (conclusion/discussion, references position, appendices). Do not read full body sections during planning; methods, experiments, and results are handled during file writing.

Then pick a `short_name` if not provided.

---

## Phase 2 — Write Structured Files

Create `papers/<short_name>/` and write all section files. If using MinerU and an images directory exists, copy it alongside the section files, for example: `cp -r <mineru_output_dir>/auto/images <output_dir>/images`.

**Writing strategy (all paths):** The current agent writes all section files directly — no subagents. This applies to the arXiv HTML path (using `chunks/sNN_*.md` as source), the optional MinerU-HTML path when Strategy 1B was selected (using `full.md`, with `main.html` as a fidelity check), the HF Papers path (using `hf_parsed.md`, supplementing missing sections from MinerU if needed), and the PDF/MinerU path (using `<parsed.md>` line ranges as source). For each section file, read the corresponding source content and write the enriched section file immediately. This approach ensures:
- **Consistent formatting** — one agent applies all conventions uniformly across every file
- **No redundant reads** — the main agent builds up the paper context once, instead of N subagents each re-reading the same source
- **Full narrative coherence** — the agent sees the paper holistically, preserving cross-references and flow between sections
- **Lower overhead** — no prompt construction, subagent startup, or coordination cost

**When to use subagents (exception only):** For exceptionally large papers (60+ pages or 25+ section files) where the main agent's context window cannot accommodate all read+write operations, dispatch **`code-explorer-writer`** subagents in parallel (1–3 files each). This agent has write permission (`write_to_file`, `replace_in_file`, `delete_file`) — do NOT use the plain `code-explorer` subagent for writing tasks, as it is read-only. Provide each subagent with the source content path, assigned section headings/ranges, the content rules below, and `max_turns`. Each must return a 2–5 sentence substantive summary. This exception should be rare; for typical papers (30–50 pages), direct writing is strongly preferred.

Regardless of approach, apply the same quality standard — concrete mechanisms, named methods, specific numbers, not generic labels.

### Content Rules

**Primary goal:** Structured transcription, not summarization. The section file should read like the paper itself — preserving narrative flow, argument progression, and paragraph-level detail. Omit only genuinely redundant or off-topic passages.

1. **Preserve the original text.** Keep the paper's own wording, paragraph structure, and argument flow. Remove only: (a) verbatim repetition of a point already made, (b) filler/padding, (c) irrelevant boilerplate (legal, ethics statements, author lists). If the text is clear and concise — keep every sentence. A section that spans several pages in the original should produce a file with commensurate depth, not a bullet-point abstract.
2. **Use tables for structured data** — results, comparisons, hyperparameter configs. Bold the most important values. The `arxiv_html_to_md.py` tool now extracts table data with colspan/rowspan handling, but verify the output: empty cells from rowspan may need context (e.g., repeat the model name), and multi-row headers are merged with ` → `. Enrich tables by bolding key results and adding backticks around model/dataset names.
3. **Equations — ALWAYS use LaTeX math notation** so they render properly in Markdown viewers:
   - **Block equations** (display math): `$$...$$` on its own line
   - **Inline math**: `$...$` within text
   - **Never** use fenced code blocks (\`\`\`) for equations — code fences prevent math rendering
   - **Fix MinerU extraction artifacts actively.** When using MinerU output (Strategy 1B or 3), formulas may contain systematic errors: spaces in numbers (`1 0` → `10`), spaces in `\mathrm{}` blocks (`b e n i g n` → `benign`), and misidentified symbols (`\dot{2}` → `2`, `\gimel` → correct value, `\breve{5}` → `5`, `\mathsf{l}` → `1`). The agent should recognize and fix these using its LaTeX knowledge and the surrounding context — this is a key advantage of LLM-based transcription over raw tool output.
   - If MinerU has garbled an equation beyond simple fixes (broken symbols, missing terms), include a best-effort LaTeX reconstruction and mark it `[OCR uncertain]` rather than dropping it silently.

4. **Faithful transcription of sensitive content.** Transcribe the paper's content faithfully by default, even when the content itself is harmful (e.g., a security paper's attack transcripts or proof-of-concept outputs). Such content is the paper's *evidence* of a vulnerability, not instructions to the reader, and the paper is already publicly available. In the rare extreme case where specific, directly-actionable steps cannot be transcribed verbatim (e.g., step-by-step instructions for manufacturing explosives or weapons), write a concise summary of what the content is and note that the paper reproduces it verbatim, then **label the omission explicitly** (e.g., `[verbatim steps omitted; paper reproduces them in full]`). **Never silently drop or skip content** — silent omission violates the transcription principle. A labeled summary preserves the documentary record while being transparent about the edge case.

### Handling HTML-extraction gaps

The HTML→Markdown conversion can occasionally lose connective prose around citations, reducing a section (most often **Related Work**) to bare citation clusters such as `(Yang et al., 2023; Zhan et al., 2023)`. When a chunk is citation-only:

- **First, check the HF Papers output** (`hf_parsed.md`, already fetched per the parallel-fetch guidance above). HF Papers typically preserves the full sentences around citations that the HTML parser dropped, making it the fastest and cleanest recovery source — lift the surrounding prose and re-attach the same citations.
- If HF Papers is also missing the prose, check whether the paper has an **"Additional Related Work" appendix** — many do, with full prose. Reconstruct the section's narrative from the appendix, citing the same works.
- Otherwise, consult the raw `source.html` (already downloaded into the work directory by `arxiv_html_to_md.py`) or re-fetch the section from the arXiv HTML page to recover the surrounding sentences.
- Do not leave a section as a bare list of citation clusters; reconstruct a coherent narrative.

### Handling large reference / inventory tables

Distinguish two kinds of tables:

- **Findings tables** (results, comparisons, ablations) carry the paper's claims — transcribe them **in full** and bold the key cells. These are rarely huge (typically ≤ a few dozen rows).
- **Inventory tables** (full model / dataset / hyperparameter catalogs that live in appendices) are reference data whose per-row detail is not load-bearing for understanding the paper. When such a table exceeds **~30 rows**, **condense** it instead of transcribing verbatim:

1. Lead with **aggregate statistics** — the total count plus the breakdowns that matter to a reader (e.g., *"305 LLMs from 153 organizations; decoder-only + encoder-decoder architectures; 100M–70B parameters; licenses span Apache-2.0, MIT, Llama-2, …"*).
2. Keep a **representative subset** of ~15–25 rows chosen to span the variation (major orgs/families, size extremes, every architecture/category, varied licenses).
3. Add a one-line pointer so the exhaustive list is recoverable, e.g., *"Full 305-row table in source Appendix F (`chunks/s11_appendix.md`)."*
4. Only if the user explicitly asks for the complete catalog, write it to a companion `*_full.md` file so the main section file stays browsable.

Rationale: a knowledge-base reader almost never needs all 300+ rows; the aggregate scope plus representative samples carry the information that matters, and the full table is always recoverable from the source paper.

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

Write `papers/<short_name>/README.md` compiling the 2–4 sentence file descriptions. Include the paper title, arXiv ID, lead author, affiliation, year, and code URL header. Use the metadata from `hf papers info` (Phase 1) for the code URL — it often provides a GitHub repo that arXiv doesn't list.

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
| `papers/<category>/<short_name>/` | "<Title>" — <Author> et al. (<Affiliation>, <Year>) | <arXiv ID> |
```

`CODEBUDDY.md` at the project root is typically a **symlink to `CLAUDE.md`** (or auto-regenerated from it), so editing `CLAUDE.md` already updates both — do not edit `CODEBUDDY.md` separately unless you have confirmed it is a genuinely independent file (e.g., `ls -l CODEBUDDY.md` does not show `-> CLAUDE.md`).

---

## File Layout

```
papers/<category>/<short_name>/
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

**Verify years for arXiv-only entries.** The tool infers the year from an explicit date when present and falls back to the arXiv ID (`YYMM.NNNNN` → `20YY`) otherwise. Still spot-check entries whose only source is an arXiv preprint — if a year looks implausible (e.g., 1903, 1978 for a recent paper), correct it from the arXiv ID's first two digits.

---

