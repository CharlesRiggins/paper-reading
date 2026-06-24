---
name: paper-reader
description: Read and process academic papers (arXiv or local PDF) into structured, section-level Markdown files under papers/. Prefers HTML when available; falls back to PDF+MinerU, embeds figures with captions, cleans up scratch artifacts, and updates the README + CLAUDE.md index. Use when the user wants to add, process, or archive a paper. Invoke as /paper-reader <arXiv-URL-or-PDF-path> [short_name].
---

# Paper Reader Skill

This skill helps you **read and process academic papers** into a structured, browsable knowledge base. The deliverable is a set of section files under `papers/<category>/<short_name>/`, indexed by a `README.md`, with the paper tracked in the categorized "Papers index" in `CLAUDE.md`.

## Core Principle

> **HTML first, PDF as fallback. Structured multi-file output.**

- **HTML first** — If the publisher provides a high-quality HTML rendering (e.g. `https://arxiv.org/html/<id>`), read it directly (raw HTML preferred over `WebFetch` for fidelity). No PDF download, no MinerU.
- **PDF fallback** — If HTML is missing or low quality, download the PDF and parse it with MinerU into clean Markdown.
- **Structured output** — Regardless of input format, produce section files + README + CLAUDE.md update.

## Inputs

Arguments may arrive as positional values (`$1`, `$2`) or woven into the natural-language request. Parse them as:

- **`source`** (`$1`, required) — arXiv URL or local PDF path.
- **`short_name`** (`$2`, optional) — snake_case folder name. **If the user does not provide one, you pick it.** Derive it from the paper title: lowercase, underscore-separated, 1–3 words, memorable and unambiguous (e.g. `deepseek_r1`, `grpo`, `capacity_interference`). Avoid generic names. If a folder with that name already exists under `papers/<category>/`, pick a different name or ask.

The paper lands at `papers/<category>/<short_name>/`. You also choose the **category** (see Phase 0) — a third implicit input that the user may state directly ("file this under PEFT").

---

## Quick Decision

- HTML available and complete → Strategy 1 (current-agent HTML transcription)
- HTML missing/incomplete, or input is PDF → Strategy 2 (MinerU Markdown)
- All paths converge to **Phase 1–4** below for structured output.

---

## Phase 0 — Pre-flight

Before fetching anything:

1. **Resolve `short_name`.** If the user gave one, validate it (snake_case, 1–3 words). If not, you'll derive it after you know the title (Phase 1) — but reserve a guess now if the title is obvious from the URL.
2. **Check for duplicates.** List the `papers/` tree (folders are nested one level under a category, `papers/<category>/<short_name>/`) and scan the "Papers index" in `CLAUDE.md`. If this paper (or a folder named `<short_name>` in any category) already exists, stop and ask the user whether to overwrite, pick a new name, or abort. Do not silently clobber.
3. **Pick a category.** Papers are grouped by topic. Read the category section headers in `CLAUDE.md`'s Papers index and place the paper in the best-fitting existing category. The current categories (subject to growth) are:
   - `rl_policy_optimization` — RL & Policy Optimization (GRPO/PPO-family, credit assignment, agent RL)
   - `peft_lora` — Parameter-Efficient Fine-Tuning (LoRA and related)
   - `learning_dynamics_capacity` — Learning Dynamics & Capacity (how training updates propagate, scaling/capacity effects)
   - `calibration_self_knowledge` — Calibration & Self-Knowledge (difficulty perception, abstention, confidence)
   - `efficient_architectures` — Efficient Architectures & Frontier Systems (attention/inference efficiency, frontier model reports)

   If the paper fits an existing category, use it. If it clearly belongs to a new topic, propose a new category (snake_case folder + human-readable title) and confirm with the user before creating it. When genuinely ambiguous, ask.
4. **Decide HTML vs PDF** per the Quick Decision above.

---

## Strategy 1: HTML First (Recommended for arXiv with HTML)

### HTML detection
Extract the arXiv ID from the source when possible, then probe `https://arxiv.org/html/<id>`. Treat the HTML path as available only when the page loads successfully and appears complete. If the page is missing, blocked, truncated, or lacks major sections/equations/references, switch to **Strategy 2**.

### HTML reading model
Fidelity matters here — the goal is transcription, not a summary. Prefer fetching the **raw HTML to a scratch file and reading it directly**, which gives you the paper's verbatim text:
```bash
curl -L -o /tmp/<short_name>.html https://arxiv.org/html/<id>
```
Then `Read` that file (it's large — read in offset/limit chunks, or `Grep` for section headings first to map it). This preserves exact wording, equations (as LaTeX in the source), and the full reference list.

If `curl` is unavailable or the page is awkward to parse, fall back to `WebFetch` — but note it summarizes against a prompt and caps content, so drive it **section by section** with focused prompts requesting verbatim text of specific sections/equations/references, never a single "summarize the paper" call.

> **Optional quick-look — `hf papers read <id>`.** The Hugging Face CLI (`hf papers read 2601.18734`) dumps the arXiv HTML as one flat Markdown file. Handy for a fast skim, but **not** a substitute for the transcription above and verified worse in practice: (1) it resolves the **bare ID to v1**, not the latest version (hard-coded `…/<id>v1/…` image URLs, no version flag) — claims/numbers can differ across versions, so never trust it for a versioned archive; (2) it renders **each equation twice** (mangled Unicode fused with raw LaTeX) — unusable without heavy cleanup; (3) **tables collapse to captions only**; (4) the **References section comes through empty** (same glitch as below). Use it at most to grab verbatim prose, and always cross-check the version.

Either way, verify you have captured the complete paper — all major sections, equations, and references present, not truncated. If the full HTML cannot be obtained or appears incomplete, fall back to **Strategy 2**. Do not dispatch subagents for the HTML path.

> **Pin the version.** The bare arXiv ID can resolve to different versions across tools/time. When you know the version (e.g. the `abs` page shows v3), fetch `https://arxiv.org/html/<id>v<N>` explicitly and record it in the README ID line, so the archive is reproducible.

> **Empty-bibliography glitch (common).** arXiv's LaTeXML HTML sometimes renders the bibliography as an **empty** `<ul class="ltx_biblist"></ul>` even though in-text author–year citations are present. When this happens, don't degrade the whole paper to OCR — keep transcribing the body from HTML and recover **only the reference list** from the PDF (`pypdf`/`pdftotext`, or MinerU if available). Watch for running-header lines (e.g. the paper title repeated mid-list) that merge two references into one; split them back out.

### Figures on the HTML path
arXiv LaTeXML papers **do** expose vector/raster figures as files at predictable URLs: `https://arxiv.org/html/<id>v<N>/x1.png`, `x2.png`, … (probe sequentially until one 404s to learn the count). Map each `xK.png` to its `Figure N` by matching the `<img>` tag to the nearest `<figcaption>` in the HTML — **the file index K does not equal the figure number N** (typeset figures and tables are skipped, so e.g. `x2.png` may be "Figure 3"). Download the real figures into `papers/<category>/<short_name>/images/` with descriptive names, embed them with the paper's actual caption as alt text, and place each at the point in the narrative where the paper introduces it. Figures that are typeset (not bitmaps) — prompt boxes, algorithm listings — and result/config **tables rendered as images** won't have an `xK` file; describe those from their captions/prose instead.

Older or unusual papers (scanned PDFs, non-LaTeX) often have no HTML version.

---

## Strategy 2: PDF Fallback

Use when HTML is missing, incomplete, or the input is a PDF.

### PDF conversion
Normalize arXiv IDs or arXiv `abs`/`html`/`pdf` URLs to `https://arxiv.org/pdf/<id>`. For non-arXiv URLs, use the URL as-is. For local PDFs, use the local path directly.

**Convert in a scratch directory, not inside `papers/`.** MinerU emits a large tree (duplicate images, layout/span/origin PDFs, model JSON) plus the source PDF — none of that belongs in the committed knowledge base. Use a temp working dir and copy only what you need into `papers/<category>/<short_name>/` in Phase 2.

For a remote PDF, download it into the scratch dir:
```bash
WORK=$(mktemp -d)
curl -L -o "$WORK/<pdf_base_name>.pdf" <PDF_URL>
```

Run MinerU on the PDF:
```bash
mineru -p "$WORK/<pdf_base_name>.pdf" -o "$WORK" -b pipeline
```

Locate the MinerU-generated Markdown and use it as `<parsed.md>`:
```bash
find "$WORK" -name '*.md' -print
```

Use the Markdown file matching the PDF base name when multiple Markdown files exist. The parsed Markdown and its `images/` dir are typically nested under `<WORK>/<pdf_base_name>/auto/`. Do not inject helper content into the parsed Markdown.

> Note: `mineru` must be installed and on `PATH`. If it is not available, tell the user the PDF path requires MinerU and offer to proceed via the HTML path instead (or ask them to install it).

### Figures (PDF path)
MinerU extracts figures into an `images/` folder and references them in the Markdown as **empty** `![](images/<hash>.jpg)` tags. When transcribing, **fill in the alt text with the figure's real caption** from the paper, e.g. `![Figure 2 | Overall architecture of the model.](images/<hash>.jpg)`, and place the image at the point in the narrative where the paper introduces it. Keep image paths relative (`images/...`) so they resolve inside `papers/<category>/<short_name>/`. Drop decorative/duplicate images that carry no information.

---

## Non-arXiv Papers

Same principle applies: prefer HTML from OpenReview / ACL Anthology / etc.; probe directly with `WebFetch` to decide. Check page completeness; if no reliable HTML source exists, fall back to PDF/MinerU.

---

## Phase 1 — Outline and Partition Plan

**Goal:** Decide the paper's file layout before writing section files.

**HTML path:** Ensure the full HTML paper has been read. If it has not been read yet, read it now. Then plan the file partition from the full content.

**PDF/MinerU path:** Use `Grep` on the parsed Markdown to list all H1–H3 headings, and use that outline as the main partition basis. Example: `Grep` with pattern `^#{1,3}[[:space:]]` and `output_mode: content`. (Note: macOS `grep` does not support `\s` in basic mode; the `[[:space:]]` class works everywhere. If the search returns nothing, re-check the pattern and fall back to `Read`ing the first ~100 lines to locate headings manually.) Read only the front matter (title, authors, abstract, introduction) and skim the tail (conclusion/discussion, references position, appendices). Do not read full body sections during planning; methods, experiments, and results are handled during file writing.

Then pick a `short_name` if not provided. Use `TodoWrite` to track the planned section files as tasks when there are several.

---

## Phase 2 — Write Structured Files

Create `papers/<category>/<short_name>/` and write all section files. If using MinerU and figures are referenced, copy **only the images directory** into the folder (never the source PDF, the raw parsed `.md`, or MinerU's `auto/` intermediates):
```bash
cp -r "$WORK"/<pdf_base_name>/auto/images papers/<category>/<short_name>/images
```
Then embed figures with real captions per the **Figures (PDF path)** rules above.

**HTML path:** Do not dispatch subagents. Write the section files directly from the already-read full HTML content with the `Write` tool, preserving narrative flow and paragraph-level detail.

**PDF/MinerU path:** Dispatch subagents in parallel (1–3 files each) using the **`Agent` tool with `subagent_type: general-purpose`** — this agent can read source content and write files (`Read`, `Grep`, `Write`, `Edit`, `Bash`). To run them concurrently, issue multiple `Agent` calls in a single message. Provide each subagent with: the absolute parsed Markdown path, its assigned section headings/ranges, the exact output file path(s) to write, the **Content Rules**, **Formatting Conventions**, and **Figures** rules below, and an instruction to return a 2–5 sentence substantive summary of what it wrote. When writing directly (no subagents), apply the same quality standard — concrete mechanisms, named methods, specific numbers, not generic labels.

### Content Rules

**Primary goal:** Structured transcription, not summarization. The section file should read like the paper itself — preserving narrative flow, argument progression, and paragraph-level detail. Omit only genuinely redundant or off-topic passages.

1. **Preserve the original text.** Keep the paper's own wording, paragraph structure, and argument flow. Remove only: (a) verbatim repetition of a point already made, (b) filler/padding, (c) irrelevant boilerplate (legal, ethics statements, author lists). If the text is clear and concise — keep every sentence. A section that spans several pages in the original should produce a file with commensurate depth, not a bullet-point abstract.
2. **Use tables for structured data** — results, comparisons, hyperparameter configs. Bold the most important values.
3. **Equations — ALWAYS use LaTeX math notation** so they render properly in Markdown viewers:
   - **Block equations** (display math): `$$...$$` on its own line
   - **Inline math**: `$...$` within text
   - **Never** use fenced code blocks (```) for equations — code fences prevent math rendering
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

Write `papers/<category>/<short_name>/README.md` compiling the 2–4 sentence file descriptions. Include the paper title, identifier, lead author, affiliation, year, and code URL header.

**README.md format:**

```markdown
# <Full Paper Title>
<ID line> | <Lead Author> et al. (<Affiliation>, <Year>)
Code: <GitHub URL or "—">

## Files

- `s01_<name>.md` — <2–5 sentence substantive description>
- `s02_<name>.md` — <2–5 sentence substantive description>
- ...
- `s<NN>_references.md` — Full <N>-entry reference list
- `s<LAST>_<name>.md` — <appendix or other trailing sections, if present>
```

For the **`<ID line>`**, use whatever identifier the paper actually has — match the style already in `CLAUDE.md`: `arXiv: <id>` for arXiv papers, `DOI: <doi>`, `Blog Post`, `HuggingFace`, or `—` when there is none (e.g. anonymous submissions under review). Don't fabricate an arXiv ID.

**IMPORTANT:** Each file description should be 2–5 sentences, substantive enough that someone reading only the README gets a real sense of the paper's content. Focus on the core mechanism, key concepts, notable findings, and specific numbers. Generic statements like "this section describes the experiments" are not acceptable.

### Update CLAUDE.md

The "Papers index" in `CLAUDE.md` is grouped into per-category subsections, each with its own table (columns: Folder | Title | ID). Add a row under the matching category's `### <Category Title>` header:
```
| `papers/<category>/<short_name>/` | "<Title>" — <Author> et al. (<Affiliation>, <Year>) | <ID> |
```
Use the same ID value style as the README's ID line (`2409.15355`, `HuggingFace`, `Blog Post`, `—`, …). If you introduced a **new** category in Phase 0, add a new `### <Category Title>` subsection with its table header in the appropriate place before adding the row.

---

## Phase 4 — Verify and Clean Up

Before declaring done:

1. **Cleanup.** Confirm no scratch artifacts landed in `papers/<category>/<short_name>/` — only `README.md`, the `s*.md` files, and (if applicable) `images/` should be present. Remove any stray source PDF, raw parsed `.md`, or MinerU `auto/`/intermediate tree. Delete the scratch dir (`rm -rf "$WORK"`).
2. **Completeness.** Every major section of the paper maps to a file; references and appendices are captured; section numbering follows the paper's own order.
3. **Rendering.** Spot-check that equations use `$`/`$$` (no code-fenced math), and that embedded figures point to existing files under `images/` with real captions (no empty `![]()`).
4. **Indexes agree.** The `README.md` ID line and the new `CLAUDE.md` row use the same identifier, and the folder name matches `<short_name>`.

Report a brief summary to the user: the folder created, number of section files, the source path used (HTML vs PDF), and anything that needed reconstruction (e.g. `[OCR uncertain]` equations).

---

## File Layout

```
papers/
└── <category>/                            # Topic bucket, mirrors a CLAUDE.md index subsection
    └── <short_name>/
        ├── README.md                      # Index with substantive file descriptions
        ├── s01_<name>.md                  # One file per major section
        ├── s02_<name>.md
        ├── ...
        ├── s<NN>_<MM>_<combined_name>.md  # Optional: 2–3 short/related sections combined
        ├── s<NN>_references.md            # Reference list (wherever it appears in the paper)
        ├── s<LAST>_<name>.md              # Appendix or last section if present
        └── images/                        # Figures (PDF path only), referenced as images/<hash>.jpg
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
