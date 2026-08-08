"""§9.6 / 6a / 6b / 6c — parent strategy ladder (HEADING -> PAGE -> LLM_INFERRED)."""

from __future__ import annotations

from conftest import build_pdf_ir

from rag_ingestion.config import IngestionConfig
from rag_ingestion.parsers.docling_parser import ParsedElement


def _headed_elements() -> list[ParsedElement]:
    els: list[ParsedElement] = []
    for pg in range(1, 4):
        els.append(ParsedElement(kind="heading", text=f"{pg}. Section", page=pg, level=1))
        els.append(ParsedElement(kind="text", text=f"Body of section {pg}. " * 20, page=pg))
    return els


def test_headed_pdf_resolves_to_heading(cfg: IngestionConfig) -> None:
    doc = build_pdf_ir(_headed_elements(), page_count=3, cfg=cfg)
    assert doc.raw_metadata["parent_strategy"] == "HEADING"


def test_no_heading_pdf_resolves_to_page(cfg: IngestionConfig) -> None:
    els = [ParsedElement(kind="text", text=f"page {p} content. " * 30, page=p) for p in range(1, 4)]
    doc = build_pdf_ir(els, page_count=3, cfg=cfg)
    assert doc.raw_metadata["parent_strategy"] == "PAGE"


def test_half_structured_pdf_falls_to_page(cfg: IngestionConfig) -> None:
    # 2 headings across 40 pages -> coverage 0.05 < 0.6 -> PAGE, not HEADING (§9.6)
    els: list[ParsedElement] = [
        ParsedElement(kind="heading", text="Executive Summary", page=1, level=1),
        ParsedElement(kind="heading", text="Appendix", page=2, level=1),
    ]
    for p in range(1, 41):
        els.append(ParsedElement(kind="text", text=f"Page {p} content. " * 20, page=p))
    doc = build_pdf_ir(els, page_count=40, cfg=cfg)
    assert doc.raw_metadata["parent_strategy"] == "PAGE"
    assert doc.raw_metadata["heading_coverage"] < cfg.min_heading_page_coverage


def test_page_parents_carry_paragraph_bleed(cfg: IngestionConfig) -> None:
    # §6a — a paragraph straddling a page break appears complete in its parent.
    full_sentence = "This complete paragraph starts on page one and finishes on page two without truncation. " * 3
    els = [
        ParsedElement(kind="text", text="Page one filler. " * 40, page=1),
        ParsedElement(kind="text", text=full_sentence, page=1, end_page=2),
        ParsedElement(kind="text", text="Page two filler. " * 40, page=2),
    ]
    doc = build_pdf_ir(els, page_count=2, cfg=cfg)
    assert doc.raw_metadata["parent_strategy"] == "PAGE"
    # the straddling paragraph must appear complete on both pages it spans
    pages_with_full = [
        s for s in doc.sections if any(full_sentence.strip() in tb.text for tb in s.text_blocks)
    ]
    assert len(pages_with_full) >= 2


def test_llm_tier_degrades_safely_on_malformed_json(cfg: IngestionConfig) -> None:
    # §6b — malformed JSON from the classifier => safe fall-through to PAGE.
    def bad_classifier(_payload: str) -> str:
        return "this is not json at all {"

    llm_cfg = cfg.model_copy(update={"enable_llm_toc_inference": True})
    els = [ParsedElement(kind="text", text=f"Some Heading Line {p}\nbody text here. " * 5, page=p) for p in range(1, 4)]
    doc = build_pdf_ir(els, page_count=3, cfg=llm_cfg, classifier=bad_classifier)
    assert doc.raw_metadata["parent_strategy"] == "PAGE"  # no exception escaped


def test_llm_tier_rejects_hallucinated_line_ids(cfg: IngestionConfig) -> None:
    # §6c — a returned line_id not in the submitted set is discarded, not indexed.
    import json

    def hallucinating_classifier(payload: str) -> str:
        submitted = [c["line_id"] for c in json.loads(payload)]
        # confirm exactly one real candidate as a heading, plus a fabricated one
        resp = [{"line_id": submitted[0], "is_heading": True, "level": 1}] if submitted else []
        resp.append({"line_id": "L_HALLUCINATED", "is_heading": True, "level": 1})
        return json.dumps(resp)

    llm_cfg = cfg.model_copy(update={"enable_llm_toc_inference": True, "min_heading_page_coverage": 0.0})
    els = [
        ParsedElement(kind="text", text="Real Candidate Heading\nfollowed by body content.", page=1, is_page_start=True),
        ParsedElement(kind="text", text="More body text on page two here.", page=2),
    ]
    doc = build_pdf_ir(els, page_count=2, cfg=llm_cfg, classifier=hallucinating_classifier)
    # the hallucinated id must never surface as a section heading
    headings = [s.heading for s in doc.sections if s.heading]
    assert "L_HALLUCINATED" not in headings
    assert not any(h and "HALLUCINATED" in h for h in headings)
