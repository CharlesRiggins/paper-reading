#!/usr/bin/env python3
"""Convert arXiv LaTeXML HTML into structured Markdown intermediates.

Outputs under --out:
  - source.html
  - parsed.md
  - outline.json
  - references.md
  - chunks/sNN_*.md
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

SPACE_RE = re.compile(r"[ \t\r\f\v]+")
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|html|pdf)/)?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I)


def clean(text: str) -> str:
    text = html.unescape(text or "")
    text = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u200b", "")
    )
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +([,.;:?!])", r"\1", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"\[ ", "[", text)
    text = re.sub(r" \]", "]", text)
    return text.strip()


def slugify(text: str, max_words: int = 5) -> str:
    text = re.sub(r"^Appendix\s+", "appendix_", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    text = re.sub(r"^(\d+_)+", "", text)
    parts = [p for p in text.split("_") if p]
    return "_".join(parts[:max_words]) or "section"


def arxiv_id(source: str) -> str | None:
    match = ARXIV_RE.search(source)
    return match.group(1) if match else None


def resolve_html_source(source: str, out_dir: Path) -> tuple[Path, str | None, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source).expanduser()
    dst = out_dir / "source.html"
    if source_path.exists():
        shutil.copyfile(source_path, dst)
        return dst, arxiv_id(source), str(source_path)

    paper_id = arxiv_id(source)
    if not paper_id:
        raise SystemExit(f"Cannot infer arXiv id from source: {source}")
    url = f"https://arxiv.org/html/{paper_id}"
    cmd = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "60", "-o", str(dst), url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("curl is required for downloading arXiv HTML") from exc
    return dst, paper_id, url


def tex_of_math(node: Tag) -> str:
    ann = node.find("annotation", attrs={"encoding": "application/x-tex"})
    tex = ann.get_text("", strip=True) if ann else (node.get("alttext") or node.get_text(" ", strip=True))
    return SPACE_RE.sub(" ", tex.replace("\n", " ")).strip()


def inline_text(node) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "annotation":
        return ""
    if node.name == "math":
        tex = tex_of_math(node)
        return f"${tex}$" if tex else ""
    if node.name == "br":
        return "\n"
    if node.name in {"script", "style"}:
        return ""
    if node.name == "a":
        text = clean("".join(inline_text(c) for c in node.children))
        href = node.get("href") or ""
        if href and text.startswith("http"):
            return text
        return text
    return "".join(inline_text(c) for c in node.children)


def paragraph(node: Tag) -> str:
    return clean(inline_text(node))


def heading_md(node: Tag) -> str:
    level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####"}.get(node.name, "##")
    text = paragraph(node)
    return f"{level} {text}" if text else ""


def display_equation(node: Tag) -> str:
    maths: list[str] = []
    for math_node in node.find_all("math"):
        if math_node.find_parent("math") is None:
            tex = tex_of_math(math_node)
            if tex and tex not in maths:
                maths.append(tex)
    if not maths:
        return ""
    body = " \\\n".join(maths)
    tag = node.find(class_=re.compile("ltx_tag"))
    label = clean(tag.get_text(" ", strip=True)) if tag else ""
    result = f"$$\n{body}\n$$"
    return f"{result}\n\n*{label}*" if label else result


def markdown_table(node: Tag) -> str:
    rows: list[list[str]] = []
    for tr in node.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        row = [clean(inline_text(cell)).replace("|", "\\|") for cell in cells]
        if row:
            rows.append(row)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def table_block(node: Tag) -> str:
    classes = set(node.get("class", []))
    if "ltx_tabular" in classes:
        return markdown_table(node)
    return display_equation(node)


def list_block(node: Tag, ordered: bool = False, indent: int = 0) -> str:
    lines: list[str] = []
    index = 1
    for li in node.find_all("li", recursive=False):
        marker = f"{index}." if ordered else "-"
        text_parts: list[str] = []
        nested: list[str] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested.append(list_block(child, child.name == "ol", indent + 2))
            else:
                text_parts.append(inline_text(child))
        text = clean("".join(text_parts))
        if text:
            lines.append(" " * indent + marker + " " + text)
        lines.extend(n for n in nested if n)
        index += 1
    return "\n".join(lines)


def figure_block(node: Tag) -> str:
    parts: list[str] = []
    caption = node.find("figcaption")
    if caption:
        text = paragraph(caption)
        if text:
            if ": " in text[:50]:
                tag, rest = text.split(": ", 1)
                parts.append(f"> **{tag}:** {rest}")
            else:
                parts.append(f"> {text}")
    for table in node.find_all("table", recursive=False):
        block = table_block(table)
        if block:
            parts.append(block)
    if not parts:
        text = paragraph(node)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def blocks(node: Tag) -> list[str]:
    output: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString) or not isinstance(child, Tag):
            continue
        if child.name in {"script", "style"}:
            continue
        classes = set(child.get("class", []))
        block = ""
        if child.name in {"h1", "h2", "h3", "h4", "h5"}:
            block = heading_md(child)
        elif child.name in {"p", "div"} and (child.name == "p" or "ltx_para" in classes or "ltx_abstract" in classes):
            nested = blocks(child)
            block = "\n\n".join(nested) if nested else paragraph(child)
        elif child.name == "figure":
            block = figure_block(child)
        elif child.name == "table":
            block = table_block(child)
        elif child.name in {"ul", "ol"}:
            block = list_block(child, child.name == "ol")
        elif child.name == "section":
            block = "\n\n".join(blocks(child))
        elif child.name == "blockquote":
            text = paragraph(child)
            block = f"> {text}" if text else ""
        else:
            nested = blocks(child)
            if nested:
                block = "\n\n".join(nested)
            else:
                text = paragraph(child)
                block = text if len(text) > 40 else ""
        if block:
            output.append(block)
    return output


def section_markdown(node: Tag) -> str:
    text = "\n\n".join(blocks(node))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def parse_references(bib: Tag | None) -> str:
    if not bib:
        return "## References\n\n"
    rows = ["## References", "", "| # | Authors | Title | Venue/Source | Year |", "|---|---------|-------|--------------|------|"]
    for idx, li in enumerate(bib.select("li.ltx_bibitem"), 1):
        tag = li.find(class_=re.compile("ltx_tag_bibitem"))
        label = clean(tag.get_text(" ", strip=True)) if tag else ""
        bib_blocks = [clean(span.get_text(" ", strip=True)) for span in li.select("span.ltx_bibblock")]
        years = re.findall(r"(?:19|20)\d{2}", " ".join([label] + bib_blocks))
        year = years[-1] if years else ""
        authors = re.sub(r"\s*\((?:19|20)\d{2}[a-z]?\)\s*$", "", label).strip() or (bib_blocks[0] if bib_blocks else "")
        title = bib_blocks[1] if len(bib_blocks) > 1 else (bib_blocks[0] if bib_blocks else "")
        venue = " ".join(bib_blocks[2:]) if len(bib_blocks) > 2 else ""
        esc = lambda x: clean(x).replace("|", "\\|")
        rows.append(f"| [{idx}] | {esc(authors)} | {esc(title)} | {esc(venue)} | {esc(year)} |")
    return "\n".join(rows).strip() + "\n"


def direct_sections(article: Tag) -> Iterable[Tag]:
    for child in article.children:
        if isinstance(child, Tag) and child.name == "section":
            yield child


def title_of_section(section: Tag) -> str:
    heading = section.find(["h2", "h3", "h4"], recursive=False) or section.find(["h2", "h3", "h4"])
    return paragraph(heading) if heading else section.get("id", "section")


def metadata(article: Tag, paper_id: str | None, source_url: str) -> dict:
    title_node = article.find("h1")
    authors_node = article.find(class_=re.compile("ltx_authors"))
    abstract_node = article.find(class_=re.compile("ltx_abstract"))
    return {
        "title": paragraph(title_node) if title_node else "",
        "authors": paragraph(authors_node) if authors_node else "",
        "abstract": paragraph(abstract_node.find("p")) if abstract_node and abstract_node.find("p") else "",
        "arxiv_id": paper_id or "",
        "source": source_url,
    }


def build_front_matter(article: Tag, meta: dict) -> str:
    parts = [f"# {meta['title']}" if meta.get("title") else "# Untitled", ""]
    if meta.get("authors"):
        parts.extend([meta["authors"], ""])
    if meta.get("abstract"):
        parts.extend(["## Abstract", "", meta["abstract"], ""])
    for child in article.children:
        if isinstance(child, Tag) and child.name == "figure":
            fig = figure_block(child)
            if fig:
                parts.extend([fig, ""])
        if isinstance(child, Tag) and child.name == "section":
            break
    return "\n".join(parts).strip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(re.sub(r"\n{3,}", "\n\n", text).strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert arXiv HTML to structured Markdown intermediates.")
    parser.add_argument("source", help="arXiv id, arXiv abs/html/pdf URL, or local HTML file")
    parser.add_argument("--out", required=True, help="output directory for source.html, parsed.md, outline.json, references.md, chunks/")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    html_path, paper_id, source_url = resolve_html_source(args.source, out_dir)
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    article = soup.select_one("article.ltx_document") or soup.select_one("div.ltx_page_content article")
    if not article:
        raise SystemExit("Could not find arXiv article.ltx_document in HTML")

    meta = metadata(article, paper_id, source_url)
    chunks_dir = out_dir / "chunks"
    chunks: list[dict] = []

    front = build_front_matter(article, meta)
    front_file = chunks_dir / "s00_front_matter.md"
    write_text(front_file, front)
    chunks.append({"level": 0, "title": "Front matter", "id": "front", "file": str(front_file.relative_to(out_dir))})

    parsed_parts = [front]
    seq = 1
    refs_md = ""
    for section in direct_sections(article):
        classes = set(section.get("class", []))
        title = title_of_section(section)
        if "ltx_bibliography" in classes or title.lower() == "references":
            refs_md = parse_references(section)
            filename = f"s{seq:02d}_references.md"
            write_text(chunks_dir / filename, refs_md)
            chunks.append({"level": 2, "title": "References", "id": section.get("id", "bib"), "file": f"chunks/{filename}"})
            parsed_parts.append(refs_md)
            seq += 1
            continue
        md = section_markdown(section)
        filename = f"s{seq:02d}_{slugify(title)}.md"
        write_text(chunks_dir / filename, md)
        chunks.append({"level": 2, "title": title, "id": section.get("id", ""), "file": f"chunks/{filename}"})
        parsed_parts.append(md)
        seq += 1

    if not refs_md:
        refs_md = parse_references(article.find("section", class_="ltx_bibliography"))
    write_text(out_dir / "references.md", refs_md)
    write_text(out_dir / "parsed.md", "\n\n---\n\n".join(parsed_parts))
    outline = {"metadata": meta, "chunks": chunks}
    write_text(out_dir / "outline.json", json.dumps(outline, ensure_ascii=False, indent=2))
    print(json.dumps({"out": str(out_dir), "title": meta.get("title"), "arxiv_id": paper_id, "chunks": len(chunks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
