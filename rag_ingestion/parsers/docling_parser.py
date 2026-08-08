"""Docling parser — pdf, docx, pptx (spec sections 5.1 / 5.1.1).

Design: Docling-specific extraction is confined to :meth:`DoclingParser._extract`,
which normalizes a ``DoclingDocument`` into a flat list of :class:`ParsedElement`.
Everything else — the parent-strategy ladder, heading/page/LLM section assembly,
the quality gate — is a pure function over that list. That keeps the ladder logic
unit-testable without downloading Docling's OCR/TableFormer models.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import IngestionConfig
from ..models import IRDocument, Section, TableBlock, TextBlock
from ..utils.ids import doc_id_for_path
from .base import LowExtractionQualityError

logger = logging.getLogger("rag_ingestion.parsers.docling")

_NUMBERING_RE = re.compile(r"^\d+(\.\d+)*\s")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", ",")

# A callable that takes the JSON candidate payload and returns the raw model
# response text. Injected in tests; built lazily against Azure in production.
TocClassifier = Callable[[str], str]


class ParentStrategy(str, Enum):
    HEADING = "HEADING"
    PAGE = "PAGE"
    LLM_INFERRED = "LLM_INFERRED"
    FLAT = "FLAT"  # docx with no usable headings
    SLIDE = "SLIDE"  # pptx


@dataclass
class ParsedElement:
    """Format-neutral element extracted from a source document."""

    kind: str  # "heading" | "text" | "table"
    text: str = ""
    page: int | None = None
    end_page: int | None = None  # > page when a paragraph straddles a page break
    level: int | None = None  # heading level
    is_page_start: bool = False  # first line on its page (LLM candidate hint)
    # table payload
    table_rows: list[list[str]] | None = None
    table_markdown: str | None = None
    caption: str | None = None


# --------------------------------------------------------------------------- #
# Section assembly (pure)
# --------------------------------------------------------------------------- #
def _mk_text_block(section_id: str, order: int, el: ParsedElement) -> TextBlock:
    return TextBlock(block_id=f"{section_id}-b{order}", text=el.text, page=el.page, order=order)


def _mk_table_block(doc_id: str, idx: int, el: ParsedElement, section_path: list[str]) -> TableBlock:
    return TableBlock(
        table_id=f"{doc_id}-tbl-{idx}",
        markdown=el.table_markdown or "",
        rows=el.table_rows or [],
        caption=el.caption,
        page=el.page,
        section_path=list(section_path),
    )


def build_sections_heading(doc_id: str, elements: list[ParsedElement]) -> list[Section]:
    """Heading-anchored assembly with a level stack (spec 5.1)."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    table_idx = 0
    order = 0

    def current_path() -> list[str]:
        return [h for _, h in stack]

    def new_section() -> Section:
        path = current_path()
        sec = Section(
            section_id=f"{doc_id}-sec-{len(sections)}",
            heading=stack[-1][1] if stack else None,
            level=stack[-1][0] if stack else 0,
            section_path=path,
        )
        sections.append(sec)
        return sec

    # synthetic root for pre-heading content
    root = new_section()
    current = root

    for el in elements:
        if el.kind == "heading":
            level = el.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.text.strip()))
            current = new_section()
            order = 0
        elif el.kind == "table":
            current.tables.append(_mk_table_block(doc_id, table_idx, el, current.section_path))
            table_idx += 1
        else:  # text
            if el.text.strip():
                current.text_blocks.append(_mk_text_block(current.section_id, order, el))
                order += 1
                if current.page_start is None:
                    current.page_start = el.page
                current.page_end = el.page or current.page_end

    # drop the synthetic root if it stayed empty
    if not root.text_blocks and not root.tables and len(sections) > 1:
        sections.pop(0)
    return sections


def build_sections_page(doc_id: str, elements: list[ParsedElement], cfg: IngestionConfig) -> list[Section]:
    """One Section per page, with paragraph bleed and short-page merge (spec 5.1.1 tier 2)."""
    from ..utils.tokens import count_tokens

    by_page: dict[int, list[ParsedElement]] = defaultdict(list)
    tables_by_page: dict[int, list[ParsedElement]] = defaultdict(list)

    for el in elements:
        page = el.page or 1
        if el.kind == "table":
            tables_by_page[page].append(el)
            continue
        if not el.text.strip():
            continue
        by_page[page].append(el)
        # paragraph bleed: a straddling paragraph is included on both pages
        if el.end_page and el.end_page != page:
            for extra in range(page + 1, el.end_page + 1):
                by_page[extra].append(el)

    all_pages = sorted(set(by_page) | set(tables_by_page))
    sections: list[Section] = []
    for page in all_pages:
        sec = Section(
            section_id=f"{doc_id}-sec-{len(sections)}",
            heading=None,
            level=0,
            section_path=[f"Page {page}"],
            page_start=page,
            page_end=page,
        )
        for order, el in enumerate(by_page.get(page, [])):
            sec.text_blocks.append(_mk_text_block(sec.section_id, order, el))
        for t_idx, el in enumerate(tables_by_page.get(page, [])):
            sec.tables.append(_mk_table_block(doc_id, len(sections) * 1000 + t_idx, el, sec.section_path))
        sections.append(sec)

    # merge very short pages forward into the next page
    merged: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            sec.text_blocks = carry.text_blocks + sec.text_blocks
            sec.tables = carry.tables + sec.tables
            sec.page_start = carry.page_start
            carry = None
        if count_tokens("\n".join(tb.text for tb in sec.text_blocks)) < cfg.min_child_tokens and not sec.tables:
            carry = sec
            continue
        merged.append(sec)
    if carry is not None:  # trailing short page — keep it rather than drop content
        merged.append(carry)
    return merged


def build_sections_flat(doc_id: str, elements: list[ParsedElement]) -> list[Section]:
    """Single synthetic section for docx with no usable headings.

    The semantic chunker splits this into parents by ``max_parent_tokens``, so no
    token-window pre-splitting is needed here.
    """
    sec = Section(section_id=f"{doc_id}-sec-0", heading=None, level=0, section_path=[])
    order = 0
    table_idx = 0
    for el in elements:
        if el.kind == "table":
            sec.tables.append(_mk_table_block(doc_id, table_idx, el, []))
            table_idx += 1
        elif el.text.strip():
            sec.text_blocks.append(_mk_text_block(sec.section_id, order, el))
            if sec.page_start is None:
                sec.page_start = el.page
            sec.page_end = el.page or sec.page_end
            order += 1
    return [sec]


def build_sections_slides(doc_id: str, elements: list[ParsedElement], num_slides: int) -> list[Section]:
    """One Section per slide (spec 5.1 pptx row + 6.2).

    Sections are emitted for every slide index 1..num_slides so the slide chunker
    can detect and count empty (image-only) slides.
    """
    by_slide: dict[int, list[ParsedElement]] = defaultdict(list)
    for el in elements:
        by_slide[el.page or 1].append(el)

    total = max(num_slides, max(by_slide, default=0))
    sections: list[Section] = []
    for slide in range(1, total + 1):
        els = by_slide.get(slide, [])
        title = next((e.text.strip() for e in els if e.kind == "heading"), None)
        if title is None and els:
            title = els[0].text.strip() or None
        sec = Section(
            section_id=f"{doc_id}-sec-{slide - 1}",
            heading=title,
            level=1,
            section_path=[title] if title else [f"Slide {slide}"],
            page_start=slide,
            page_end=slide,
        )
        order = 0
        for el in els:
            if el.kind == "heading" and el.text.strip() == title:
                continue  # title captured as heading, don't duplicate in body
            if el.kind == "table":
                continue  # tables handled by table chunker via section.tables
            if el.text.strip():
                sec.text_blocks.append(_mk_text_block(sec.section_id, order, el))
                order += 1
        for t_idx, el in enumerate(e for e in els if e.kind == "table"):
            sec.tables.append(_mk_table_block(doc_id, slide * 1000 + t_idx, el, sec.section_path))
        sections.append(sec)
    return sections


# --------------------------------------------------------------------------- #
# Tier 3 — LLM candidate classification (pure over an injected classifier)
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    line_id: str
    text: str
    page: int | None
    context_before: str = ""
    context_after: str = ""
    element_index: int = 0
    line_index: int = 0


def _is_heading_candidate(line: str, is_page_start: bool, cfg: IngestionConfig) -> bool:
    line = line.strip()
    if not line or len(line) > cfg.max_heading_chars:
        return False
    ends_clean = not line.endswith(_TERMINAL_PUNCT)
    return ends_clean or bool(_NUMBERING_RE.match(line)) or is_page_start


def _extract_candidates(elements: list[ParsedElement], cfg: IngestionConfig) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for ei, el in enumerate(elements):
        if el.kind != "text":
            continue
        lines = [ln for ln in el.text.splitlines() if ln.strip()]
        for li, line in enumerate(lines):
            page_start = el.is_page_start and li == 0
            if _is_heading_candidate(line, page_start, cfg):
                candidates.append(
                    _Candidate(
                        line_id=f"L{len(candidates)}",
                        text=line.strip(),
                        page=el.page,
                        context_before=lines[li - 1].strip() if li > 0 else "",
                        context_after=lines[li + 1].strip() if li + 1 < len(lines) else "",
                        element_index=ei,
                        line_index=li,
                    )
                )
    return candidates


def _classify_candidates(
    candidates: list[_Candidate],
    classifier: TocClassifier,
) -> dict[str, int] | None:
    """Call the classifier, validate, retry once. Returns line_id -> level, or None."""
    submitted = {c.line_id for c in candidates}
    payload = json.dumps(
        [
            {"line_id": c.line_id, "text": c.text, "page": c.page,
             "before": c.context_before, "after": c.context_after}
            for c in candidates
        ]
    )

    for attempt in range(2):
        try:
            raw = classifier(payload)
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
            confirmed: dict[str, int] = {}
            for entry in parsed:
                lid = entry.get("line_id")
                if lid not in submitted:  # reject hallucinated line_ids (spec 9.6c)
                    logger.warning("toc inference: discarding unsubmitted line_id %r", lid)
                    continue
                if not entry.get("is_heading"):
                    continue
                level = entry.get("level", 1)
                try:
                    level = int(level)
                except (TypeError, ValueError):
                    level = 1
                confirmed[lid] = max(1, min(4, level))  # clamp 1..4
            return confirmed
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as exc:
            logger.warning("toc inference attempt %d failed: %s", attempt + 1, exc)
            continue
    return None


def build_sections_llm(
    doc_id: str,
    elements: list[ParsedElement],
    cfg: IngestionConfig,
    classifier: TocClassifier,
) -> list[Section] | None:
    """Tier 3 assembly. Returns None to signal a safe fall-through to PAGE."""
    candidates = _extract_candidates(elements, cfg)
    if not candidates or len(candidates) > cfg.max_heading_candidates:
        logger.warning("toc inference skipped: %d candidates (cap %d)", len(candidates), cfg.max_heading_candidates)
        return None

    confirmed = _classify_candidates(candidates, classifier)
    if not confirmed:
        return None

    # rebuild the element stream, promoting confirmed lines to heading elements
    cand_by_pos = {(c.element_index, c.line_index): c for c in candidates}
    promoted: list[ParsedElement] = []
    for ei, el in enumerate(elements):
        if el.kind != "text":
            promoted.append(el)
            continue
        lines = el.text.splitlines()
        for li, line in enumerate(lines):
            cand = cand_by_pos.get((ei, li))
            if cand is not None and cand.line_id in confirmed:
                promoted.append(ParsedElement(kind="heading", text=line.strip(), page=el.page, level=confirmed[cand.line_id]))
            elif line.strip():
                promoted.append(ParsedElement(kind="text", text=line.strip(), page=el.page))

    sections = build_sections_heading(doc_id, promoted)
    return sections


# --------------------------------------------------------------------------- #
# Strategy resolution
# --------------------------------------------------------------------------- #
def _heading_coverage(elements: list[ParsedElement], page_count: int) -> float:
    if page_count <= 0:
        return 0.0
    pages = {e.page for e in elements if e.kind == "heading" and e.page is not None}
    return len(pages) / page_count


def resolve_strategy_and_build(
    doc_id: str,
    source_uri: str,
    file_type: str,
    elements: list[ParsedElement],
    page_count: int,
    cfg: IngestionConfig,
    *,
    classifier: TocClassifier | None = None,
    num_slides: int = 0,
) -> IRDocument:
    """Run the §5.1.1 ladder and assemble the IR. Records the winning strategy."""
    doc = IRDocument(doc_id=doc_id, source_uri=source_uri, file_type=file_type)

    if file_type == "pptx":
        doc.sections = build_sections_slides(doc_id, elements, num_slides)
        doc.raw_metadata["parent_strategy"] = ParentStrategy.SLIDE.value
        return doc

    headings = [e for e in elements if e.kind == "heading"]
    has_headings = len(headings) >= cfg.min_headings_for_structure

    # For PDF the coverage guard applies; docx generally lacks page numbers.
    coverage_ok = True
    if file_type == "pdf":
        coverage = _heading_coverage(elements, page_count)
        coverage_ok = coverage >= cfg.min_heading_page_coverage
        doc.raw_metadata["heading_coverage"] = round(coverage, 3)

    if has_headings and coverage_ok:
        doc.sections = build_sections_heading(doc_id, elements)
        doc.raw_metadata["parent_strategy"] = ParentStrategy.HEADING.value
        return doc

    # Tier 1 failed.
    if file_type == "docx":
        doc.sections = build_sections_flat(doc_id, elements)
        doc.raw_metadata["parent_strategy"] = ParentStrategy.FLAT.value
        return doc

    # PDF tiers 2/3
    if cfg.enable_llm_toc_inference and classifier is not None:
        llm_sections = build_sections_llm(doc_id, elements, cfg, classifier)
        if llm_sections is not None and file_type == "pdf":
            if _heading_coverage_from_sections(llm_sections, page_count) >= cfg.min_heading_page_coverage:
                doc.sections = llm_sections
                doc.raw_metadata["parent_strategy"] = ParentStrategy.LLM_INFERRED.value
                return doc

    doc.sections = build_sections_page(doc_id, elements, cfg)
    doc.raw_metadata["parent_strategy"] = ParentStrategy.PAGE.value
    return doc


def _heading_coverage_from_sections(sections: list[Section], page_count: int) -> float:
    if page_count <= 0:
        return 1.0  # unknown page count: don't block LLM tier
    pages = {s.page_start for s in sections if s.heading is not None and s.page_start is not None}
    return len(pages) / page_count if page_count else 1.0


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #
_EXT_TO_TYPE = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx"}


class DoclingParser:
    supported_extensions: set[str] = {".pdf", ".docx", ".pptx"}

    def __init__(self, config: IngestionConfig, classifier: TocClassifier | None = None) -> None:
        self.cfg = config
        self._classifier = classifier
        self._converter: Any = None

    # -- docling boundary ---------------------------------------------------- #
    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        return self._converter

    def _extract(self, path: Path) -> tuple[list[ParsedElement], int, int]:
        """Convert with Docling and normalize to (elements, page_count, num_slides)."""
        from docling_core.types.doc import (
            SectionHeaderItem,
            TableItem,
            TextItem,
        )

        converter = self._get_converter()
        result = converter.convert(str(path))
        dl_doc = result.document

        def page_of(item: object) -> int | None:
            prov = getattr(item, "prov", None)
            if prov:
                return getattr(prov[0], "page_no", None)
            return None

        elements: list[ParsedElement] = []
        picture_count = 0
        seen_page_first: set[int] = set()

        for item, _ in dl_doc.iterate_items():
            if isinstance(item, TableItem):
                try:
                    df = item.export_to_dataframe()
                    rows = [[str(c) for c in df.columns]] + [[str(c) for c in r] for r in df.values.tolist()]
                    markdown = item.export_to_markdown()
                except Exception:  # noqa: BLE001 - defensive around docling table export
                    rows, markdown = [], ""
                caption = None
                cap_fn = getattr(item, "caption_text", None)
                if callable(cap_fn):
                    try:
                        caption = cap_fn(dl_doc) or None
                    except Exception:  # noqa: BLE001
                        caption = None
                elements.append(
                    ParsedElement(kind="table", page=page_of(item), table_rows=rows,
                                  table_markdown=markdown, caption=caption)
                )
            elif isinstance(item, SectionHeaderItem):
                elements.append(
                    ParsedElement(kind="heading", text=item.text, page=page_of(item),
                                  level=getattr(item, "level", 1) or 1)
                )
            elif isinstance(item, TextItem):
                label = str(getattr(item, "label", "")).lower()
                page = page_of(item)
                if "picture" in label:
                    picture_count += 1
                    continue
                is_first = page is not None and page not in seen_page_first
                if page is not None:
                    seen_page_first.add(page)
                if "header" in label or "title" in label or "section" in label:
                    elements.append(ParsedElement(kind="heading", text=item.text, page=page, level=1))
                else:
                    elements.append(ParsedElement(kind="text", text=item.text, page=page, is_page_start=is_first))

        page_count = self._page_count(dl_doc)
        if picture_count:
            logger.info("skipped %d picture(s) in %s", picture_count, path.name)
        return elements, page_count, page_count

    @staticmethod
    def _page_count(dl_doc: object) -> int:
        pages = getattr(dl_doc, "pages", None)
        if pages is not None:
            try:
                return len(pages)
            except TypeError:
                pass
        num_pages = getattr(dl_doc, "num_pages", None)
        if callable(num_pages):
            try:
                return int(num_pages())
            except Exception:  # noqa: BLE001
                return 0
        return 0

    # -- default LLM classifier (lazy) --------------------------------------- #
    def _build_llm(self) -> Any:
        """Build the Tier-3 chat LLM per config (local Ollama default, or Azure)."""
        provider = self.cfg.llm_provider.lower()
        if provider in ("ollama", "local"):
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.cfg.ollama_model,
                base_url=self.cfg.ollama_base_url,
                temperature=0,
            )
        if provider == "azure":
            from langchain_openai import AzureChatOpenAI

            if not self.cfg.azure_openai_endpoint or self.cfg.azure_openai_api_key is None:
                raise ValueError(
                    "azure llm_provider requires azure_openai_endpoint and azure_openai_api_key, "
                    "or use RAG_INGEST_LLM_PROVIDER=ollama for a local model."
                )
            return AzureChatOpenAI(
                azure_endpoint=self.cfg.azure_openai_endpoint,
                api_key=self.cfg.azure_openai_api_key.get_secret_value(),
                azure_deployment=self.cfg.toc_inference_deployment,
                temperature=0,
            )
        raise ValueError(f"Unknown llm_provider {self.cfg.llm_provider!r}; expected 'ollama' or 'azure'.")

    def _default_classifier(self) -> TocClassifier:
        llm = self._build_llm()

        def classify(payload: str) -> str:
            prompt = (
                "You are given candidate lines from a PDF. For each, decide if it is a "
                "section heading. Return ONLY a JSON array of "
                '{"line_id": str, "is_heading": bool, "level": int}. No prose.\n\n'
                f"Candidates:\n{payload}"
            )
            resp = llm.invoke(prompt)
            return str(resp.content)

        return classify

    # -- public API ---------------------------------------------------------- #
    def parse(self, path: str | Path) -> IRDocument:
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in _EXT_TO_TYPE:  # pragma: no cover - registry guards this
            raise ValueError(f"DoclingParser cannot handle extension {ext!r}")
        file_type = _EXT_TO_TYPE[ext]

        doc_id = doc_id_for_path(p)
        elements, page_count, num_slides = self._extract(p)

        # quality gate (spec 5.1): raise rather than emit an empty document
        if file_type == "pdf":
            total_chars = sum(len(e.text) for e in elements)
            pages = max(page_count, 1)
            if total_chars / pages < self.cfg.min_chars_per_page:
                raise LowExtractionQualityError(
                    f"{p.name}: {total_chars} chars over {pages} page(s) "
                    f"(< {self.cfg.min_chars_per_page}/page). Route to fallback parser."
                )

        classifier = self._classifier
        if file_type == "pdf" and self.cfg.enable_llm_toc_inference and classifier is None:
            classifier = self._default_classifier()

        doc = resolve_strategy_and_build(
            doc_id,
            p.resolve().as_uri(),
            file_type,
            elements,
            page_count,
            self.cfg,
            classifier=classifier,
            num_slides=num_slides,
        )
        doc.title = p.stem
        doc.raw_metadata["page_count"] = page_count
        return doc
