#!/usr/bin/env python3
"""Render paper-table report JSON as a landscape, paginated PDF.

Usage: python render_report.py INPUT_JSON OUTPUT_PDF
Requires ReportLab. DejaVu Sans is embedded when available; REPORT_FONT_DIR can
point to a directory containing DejaVuSans.ttf and DejaVuSans-Bold.ttf.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle
import reportlab


PAGE_W, PAGE_H = landscape(A4)
MARGIN = 34
WIDTH = PAGE_W - 2 * MARGIN
BOTTOM = 45
NAVY = colors.HexColor("#17334C")
TEAL = colors.HexColor("#12665D")
INK = colors.HexColor("#25384B")
MUTED = colors.HexColor("#617286")
RULE = colors.HexColor("#D3DFE7")
ZEBRA = colors.HexColor("#F3F6F9")
ACCENT_LIGHT = colors.HexColor("#EAF4F0")
ACCENT_ALT = colors.HexColor("#DFEEE8")


def clean(value: object) -> str:
    """Treat all report fields as literal text, including any markup."""
    text = str(value)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def register_fonts() -> None:
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    candidates = [
        Path(os.environ["REPORT_FONT_DIR"]) if os.environ.get("REPORT_FONT_DIR") else None,
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/local/share/fonts/dejavu"),
        runtime / "native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype",
    ]
    for directory in filter(None, candidates):
        regular, bold = directory / "DejaVuSans.ttf", directory / "DejaVuSans-Bold.ttf"
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("Report", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            break
    else:
        directory = Path(reportlab.__file__).parent / "fonts"
        pdfmetrics.registerFont(TTFont("Report", str(directory / "Vera.ttf")))
        pdfmetrics.registerFont(TTFont("ReportBold", str(directory / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")


def style(size: float, *, bold: bool = False, color=INK, right: bool = False,
          leading: float | None = None) -> ParagraphStyle:
    return ParagraphStyle(
        "report", fontName="ReportBold" if bold else "Report", fontSize=size,
        leading=leading or size * 1.22, textColor=color,
        alignment=TA_RIGHT if right else TA_LEFT,
        spaceBefore=0, spaceAfter=0, allowWidows=0, allowOrphans=0,
    )


def paragraph(text: object, paragraph_style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(clean(text)).replace("\n", "<br/>"), paragraph_style)


def validate_report(report: object) -> dict:
    if not isinstance(report, dict):
        raise ValueError("Report must be a JSON object")
    for name in ("title", "subtitle", "date"):
        if not isinstance(report.get(name), str):
            raise ValueError(f"Report {name} must be a string")
    sections = report.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Report sections must be a nonempty array")
    for index, section in enumerate(sections, 1):
        if not isinstance(section, dict):
            raise ValueError(f"Section {index} must be an object")
        for name in ("title", "caption"):
            if not isinstance(section.get(name), str):
                raise ValueError(f"Section {index} {name} must be a string")
        headers = section.get("headers")
        if not isinstance(headers, list) or not 2 <= len(headers) <= 12 or not all(isinstance(h, str) for h in headers):
            raise ValueError(f"Section {index} needs 2-12 string headers")
        rows = section.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Section {index} rows must be a nonempty array")
        if any(not isinstance(row, list) or len(row) != len(headers) for row in rows):
            raise ValueError(f"Section {index} rows must match its header count")
        if any(not isinstance(cell, (str, int, float)) for row in rows for cell in row):
            raise ValueError(f"Section {index} cells must be strings or numbers")
        if any(isinstance(cell, float) and not math.isfinite(cell) for row in rows for cell in row):
            raise ValueError(f"Section {index} has a nonfinite value")
        notes = section.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            raise ValueError(f"Section {index} notes must be an array of strings")
    return report


def verify_glyphs(report: dict) -> None:
    supported = pdfmetrics.getFont("Report").face.charWidths
    strings = [report[key] for key in ("title", "subtitle", "date")]
    for section in report["sections"]:
        strings += [section["title"], section["caption"], *section["headers"], *section.get("notes", [])]
        strings += [str(cell) for row in section["rows"] for cell in row]
    missing = sorted({c for text in strings for c in clean(text) if not c.isspace() and ord(c) not in supported})
    if missing:
        raise ValueError("Font lacks glyphs: " + ", ".join(f"{c!r} U+{ord(c):04X}" for c in missing) + "; set REPORT_FONT_DIR to DejaVu Sans")


def numeric(text: object) -> bool:
    return bool(re.fullmatch(r"[\d\s.,+%/xX()<>≤≥=-]+", clean(text)))


def column_layout(section: dict) -> tuple[list[float], list[bool]]:
    rows, headers = section["rows"], section["headers"]
    right = [index != 0 and all(numeric(row[index]) for row in rows) for index in range(len(headers))]
    desired = []
    minimums = []
    for index, header in enumerate(headers):
        body_width = max(pdfmetrics.stringWidth(clean(row[index]), "Report", 8.1) for row in rows)
        words = clean(header).split()
        header_width = max((pdfmetrics.stringWidth(word, "ReportBold", 7.8) for word in words), default=0)
        minimum = max(45, min(110, header_width + 13))
        if not right[index] and index != 0:
            minimum = max(minimum, min(148, body_width + 15))
        desired.append(max(minimum, min(155, body_width + 18), min(115, len(clean(header)) * 3.5)))
        minimums.append(minimum)
    if sum(minimums) > WIDTH:
        scale = WIDTH / sum(minimums)
        return [value * scale for value in minimums], right
    spare = WIDTH - sum(minimums)
    priorities = [max(9, want - small) for want, small in zip(desired, minimums)]
    return [small + spare * priority / sum(priorities) for small, priority in zip(minimums, priorities)], right


def header_blocks(report: dict, section: dict, continued: bool) -> tuple[list[tuple[Paragraph, float]], float]:
    y = PAGE_H - 26
    blocks = []
    entries = [
        (report["title"], style(12.1, bold=True, color=NAVY), 4),
        (report["subtitle"], style(8.1, color=MUTED), 10),
        (section["title"] + (" (continued)" if continued else ""), style(16.1, bold=True, color=NAVY), 4),
        (section["caption"], style(8.15, color=MUTED), 8),
    ]
    for text, text_style, gap in entries:
        item = paragraph(text, text_style)
        _, height = item.wrap(WIDTH, PAGE_H)
        y -= height
        blocks.append((item, y))
        y -= gap
    return blocks, y


def make_table(section: dict, rows: list, font_size: float, padding: float) -> Table:
    widths, alignments = column_layout(section)
    columns = len(widths)
    emphasize = 2 if columns >= 6 else 1
    body_styles = [style(font_size, bold=i >= columns - emphasize, color=TEAL if i >= columns-emphasize else INK,
                         right=alignments[i], leading=font_size * 1.17) for i in range(columns)]
    head_styles = [style(7.7, bold=True, color=colors.white, right=alignments[i], leading=9.15) for i in range(columns)]
    cells = [[paragraph(value, head_styles[i]) for i, value in enumerate(section["headers"])]]
    cells += [[paragraph(value, body_styles[i]) for i, value in enumerate(row)] for row in rows]
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 1), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 1), (-1, -1), padding),
        ("LINEBELOW", (0, -1), (-1, -1), 0.55, RULE),
    ]
    for row_index in range(1, len(cells)):
        commands += [
            ("BACKGROUND", (0, row_index), (-1, row_index), colors.white if row_index % 2 else ZEBRA),
            ("BACKGROUND", (columns-emphasize, row_index), (-1, row_index), ACCENT_LIGHT if row_index % 2 else ACCENT_ALT),
        ]
        if row_index > 1 and clean(rows[row_index-1][0]) != clean(rows[row_index-2][0]):
            commands.append(("LINEABOVE", (0, row_index), (-1, row_index), 0.6, RULE))
    table = Table(cells, colWidths=widths, style=TableStyle(commands), hAlign="LEFT")
    table.wrap(WIDTH, PAGE_H)
    return table


def make_notes(section: dict) -> list[Paragraph]:
    return [paragraph(note, style(7.55, color=MUTED, leading=9.3)) for note in section.get("notes", [])]


def notes_height(notes: list[Paragraph]) -> float:
    return 0 if not notes else 14 + sum(note.wrap(WIDTH, PAGE_H)[1] + 4 for note in notes)


@dataclass
class Page:
    section: int
    continued: bool
    header: list[tuple[Paragraph, float]]
    top: float
    table: Table | None
    notes: list[Paragraph]
    rows: int
    font_size: float


def plan_pages(report: dict) -> list[Page]:
    pages = []
    for section_index, section in enumerate(report["sections"]):
        remaining = list(section["rows"])
        pending_notes = make_notes(section)
        continued = False
        while remaining:
            header, top = header_blocks(report, section, continued)
            available = top - BOTTOM
            # Try to keep each section on one page. Never reduce body text below 8pt.
            chosen = None
            candidates = [(9.0, 3.1), (8.6, 2.3), (8.2, 1.5), (8.0, 0.7)]
            if len(remaining) <= 30:
                for size, pad in candidates:
                    table = make_table(section, remaining, size, pad)
                    if table._height + notes_height(pending_notes) + (10 if pending_notes else 0) <= available:
                        chosen = (table, len(remaining), size, pending_notes)
                        pending_notes = []
                        break
            if chosen is None:
                # On continuation pages, prefer comfortably padded rows to tiny type.
                size, pad = 8.2, 1.5
                count = min(30, len(remaining))
                while count > 0:
                    table = make_table(section, remaining[:count], size, pad)
                    if table._height <= available:
                        break
                    count -= 1
                if count == 0:
                    raise ValueError(f"Section {section_index+1}: a table row cannot fit on a page")
                notes = []
                if count == len(remaining) and pending_notes:
                    # Avoid a notes-only page when a smaller final table fits with notes.
                    fitting = count
                    while fitting > 1:
                        candidate = make_table(section, remaining[:fitting], size, pad)
                        if candidate._height + notes_height(pending_notes) + 10 <= available:
                            break
                        fitting -= 1
                    if fitting < count and len(remaining) > 1:
                        count = max(1, count - max(3, count - fitting))
                        table = make_table(section, remaining[:count], size, pad)
                chosen = (table, count, size, notes)
            table, count, size, notes = chosen
            pages.append(Page(section_index, continued, header, top, table, notes, count, size))
            del remaining[:count]
            continued = True
        while pending_notes:
            # Long notes can share remaining table space or continue with full headers.
            page = pages[-1]
            used = page.table._height if page.table else 0
            available = page.top - BOTTOM - used - 24
            if page.notes or available < 35:
                header, top = header_blocks(report, section, True)
                page = Page(section_index, True, header, top, None, [], 0, 8.0)
                pages.append(page)
                available = top - BOTTOM - 14
            while pending_notes and available >= 10:
                item = pending_notes[0]
                height = item.wrap(WIDTH, PAGE_H)[1]
                if height + 4 <= available:
                    page.notes.append(pending_notes.pop(0))
                    available -= height + 4
                else:
                    parts = item.split(WIDTH, available - 4)
                    if parts:
                        page.notes.append(parts[0])
                        pending_notes[0:1] = parts[1:]
                    break
            if not page.notes and pending_notes:
                raise ValueError(f"Section {section_index+1}: notes cannot fit on a page")
    return pages


def draw_page(canvas: Canvas, report: dict, page: Page, number: int, total: int) -> None:
    canvas.setFillColor(TEAL)
    canvas.rect(MARGIN, PAGE_H-18, 28, 2.8, fill=1, stroke=0)
    for item, y in page.header:
        item.drawOn(canvas, MARGIN, y)
    y = page.top
    if page.table:
        y -= page.table._height
        page.table.drawOn(canvas, MARGIN, y)
    if page.notes:
        y -= 17 if page.table else 7
        canvas.setFont("ReportBold", 6.8)
        canvas.setFillColor(TEAL)
        canvas.drawString(MARGIN, y, "MEASUREMENT NOTES")
        y -= 7
        for note in page.notes:
            height = note.wrap(WIDTH, PAGE_H)[1]
            y -= height
            note.drawOn(canvas, MARGIN, y)
            y -= 4
    if y < BOTTOM - 0.2:
        raise ValueError(f"Layout overflow on page {number}: content bottom {y:.1f}pt")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 31, PAGE_W-MARGIN, 31)
    canvas.setFillColor(MUTED)
    canvas.setFont("Report", 7.0)
    canvas.drawString(MARGIN, 18, f"{clean(report['date'])}   |   Section {page.section+1} of {len(report['sections'])}")
    canvas.drawRightString(PAGE_W-MARGIN, 18, f"{number:02d} / {total:02d}")
    canvas.showPage()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    report = validate_report(json.loads(args.input_json.read_text(encoding="utf-8")))
    register_fonts()
    verify_glyphs(report)
    pages = plan_pages(report)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(args.output_pdf), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    canvas.setTitle(clean(report["title"]))
    canvas.setSubject(clean(report["subtitle"]))
    canvas.setAuthor("Weighted BTX benchmark experiments")
    for number, page in enumerate(pages, 1):
        draw_page(canvas, report, page, number, len(pages))
    canvas.save()
    print(json.dumps({"output": str(args.output_pdf), "pages": len(pages), "layout": [
        {"page": i, "section": page.section+1, "rows": page.rows, "body_font_pt": page.font_size,
         "continued": page.continued} for i, page in enumerate(pages, 1)]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"render_report: {exc}", file=sys.stderr)
        sys.exit(1)
