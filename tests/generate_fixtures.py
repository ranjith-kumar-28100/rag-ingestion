"""Generate the real sample files required by spec section 9.

Run directly (`python tests/generate_fixtures.py`) or via the session-scoped
`fixtures_dir` conftest fixture, which calls :func:`generate_all` if files are
missing. Kept out of the unit tests' hot path — most assertions use synthetic
ParsedElement lists so they never depend on these files or on Docling.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _pdf_multi_heading(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = [
        Paragraph("1. Introduction", styles["Heading1"]),
        Paragraph("This document introduces the architecture. " * 8, styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("2. Results", styles["Heading1"]),
        Paragraph("The measured results are summarized below. " * 8, styles["BodyText"]),
        Table(
            [["Region", "Revenue"], ["North", "1200"], ["South", "SENTINEL_CELL_9999"]],
            style=[("GRID", (0, 0), (-1, -1), 0.5, colors.black)],
        ),
        Paragraph("3. Conclusion", styles["Heading1"]),
        Paragraph("We conclude with next steps. " * 8, styles["BodyText"]),
    ]
    doc.build(story)


def _pdf_flat(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    body = [Paragraph("Uniform paragraph with no heading structure. " * 12, styles["BodyText"]) for _ in range(8)]
    doc.build(body)


def _pdf_half_structured(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = []
    for page in range(40):
        if page == 0:
            story.append(Paragraph("Executive Summary", styles["Heading1"]))
        elif page == 1:
            story.append(Paragraph("Appendix", styles["Heading1"]))
        story.append(Paragraph(f"Page {page + 1} body content. " * 20, styles["BodyText"]))
        story.append(PageBreak())
    doc.build(story)


def _pdf_scanned(path: Path) -> None:
    """Image-only PDF with no readable text (OCR should extract ~nothing)."""
    import random

    from PIL import Image
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    random.seed(0)
    img = Image.new("RGB", (400, 300))
    img.putdata([(random.randint(200, 255),) * 3 for _ in range(400 * 300)])
    tmp = path.with_suffix(".noise.png")
    img.save(tmp)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(ImageReader(str(tmp)), 100, 400, width=400, height=300)
    c.showPage()
    c.save()
    tmp.unlink(missing_ok=True)


def _docx_headings(path: Path) -> None:
    from docx import Document

    d = Document()
    d.add_heading("Overview", level=1)
    d.add_paragraph("The overview section explains the goal. " * 6)
    d.add_heading("Design", level=1)
    d.add_paragraph("The design section explains the approach. " * 6)
    d.add_heading("Implementation", level=2)
    d.add_paragraph("Implementation details follow here. " * 6)
    d.save(str(path))


def _pptx_notes(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    for i in range(3):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Slide {i + 1} Title"
        body = slide.placeholders[1]
        body.text = f"Body text for slide {i + 1}. Refers to the previous slide."
        notes = slide.notes_slide
        notes.notes_text_frame.text = f"Speaker notes for slide {i + 1}."
    # an image-only (empty) slide
    blank = prs.slides.add_slide(prs.slide_layouts[6])
    blank.shapes.add_textbox(Inches(1), Inches(1), Inches(1), Inches(1))
    prs.save(str(path))


def _xlsx_multi_sheet(path: Path) -> None:
    import pandas as pd

    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"region": ["North", "South"], "revenue": [1200, 3400]}).to_excel(
            w, sheet_name="sales", index=False
        )
        pd.DataFrame({"product": ["A", "B", "C"], "units": [10, 20, 30]}).to_excel(
            w, sheet_name="inventory", index=False
        )
    return None


def _csv_headerless(path: Path) -> None:
    path.write_text("1,2,3\n4,5,6\n7,8,9\n10,11,12\n")


def generate_all(dest: Path = FIXTURES) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "pdf_multi_heading": dest / "multi_heading.pdf",
        "pdf_flat": dest / "flat.pdf",
        "pdf_half_structured": dest / "half_structured.pdf",
        "pdf_scanned": dest / "scanned.pdf",
        "docx_headings": dest / "headings.docx",
        "pptx_notes": dest / "notes.pptx",
        "xlsx_multi_sheet": dest / "multi_sheet.xlsx",
        "csv_headerless": dest / "headerless.csv",
    }
    _pdf_multi_heading(files["pdf_multi_heading"])
    _pdf_flat(files["pdf_flat"])
    _pdf_half_structured(files["pdf_half_structured"])
    _pdf_scanned(files["pdf_scanned"])
    _docx_headings(files["docx_headings"])
    _pptx_notes(files["pptx_notes"])
    _xlsx_multi_sheet(files["xlsx_multi_sheet"])
    _csv_headerless(files["csv_headerless"])
    return files


if __name__ == "__main__":
    made = generate_all()
    for name, p in made.items():
        print(f"{name}: {p} ({p.stat().st_size} bytes)")
