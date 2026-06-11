"""
Render a Markdown report to PDF using ReportLab (simple layout).

Generates: docs/sprint4_report.pdf

Usage:
    python scripts/render_report_pdf.py --md docs/sprint4_report.md --out docs/sprint4_report.pdf

This is a lightweight renderer (headings, paragraphs, images, lists).
"""
import argparse
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem


def parse_markdown_lines(lines):
    elems = []
    bullets = []
    for raw in lines:
        line = raw.rstrip('\n')
        if not line.strip():
            # flush bullets if any
            if bullets:
                elems.append(('bullets', bullets.copy()))
                bullets.clear()
            elems.append(('spacer', None))
            continue

        if line.startswith('# '):
            if bullets:
                elems.append(('bullets', bullets.copy()))
                bullets.clear()
            elems.append(('h1', line[2:].strip()))
        elif line.startswith('## '):
            if bullets:
                elems.append(('bullets', bullets.copy()))
                bullets.clear()
            elems.append(('h2', line[3:].strip()))
        elif line.startswith('### '):
            if bullets:
                elems.append(('bullets', bullets.copy()))
                bullets.clear()
            elems.append(('h3', line[4:].strip()))
        elif line.startswith('- '):
            bullets.append(line[2:].strip())
        elif line.startswith('!['):
            # image syntax ![alt](path)
            try:
                start = line.index('](')
                path = line[start+2:-1]
                elems.append(('img', path))
            except Exception:
                elems.append(('p', line))
        else:
            elems.append(('p', line))

    if bullets:
        elems.append(('bullets', bullets.copy()))
    return elems


def build_pdf(md_path, out_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    elems = parse_markdown_lines(lines)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            rightMargin=inch/2, leftMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)
    styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=20, leading=22)
    style_h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=16, leading=18)
    style_h3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=14, leading=16)
    style_p = ParagraphStyle('P', parent=styles['BodyText'], fontSize=11, leading=14)
    story = []

    md_dir = os.path.dirname(md_path)

    for typ, val in elems:
        if typ == 'h1':
            story.append(Paragraph(val, style_h1))
            story.append(Spacer(1, 12))
        elif typ == 'h2':
            story.append(Paragraph(val, style_h2))
            story.append(Spacer(1, 8))
        elif typ == 'h3':
            story.append(Paragraph(val, style_h3))
            story.append(Spacer(1, 6))
        elif typ == 'p':
            story.append(Paragraph(val, style_p))
            story.append(Spacer(1, 6))
        elif typ == 'spacer':
            story.append(Spacer(1, 6))
        elif typ == 'bullets':
            items = [ListItem(Paragraph(b, style_p)) for b in val]
            story.append(ListFlowable(items, bulletType='bullet'))
            story.append(Spacer(1, 6))
        elif typ == 'img':
            # resolve path relative to markdown
            img_path = val
            if not os.path.isabs(img_path):
                # try relative to md file, and project root
                cand = os.path.join(md_dir, img_path)
                if os.path.exists(cand):
                    img_path = cand
                else:
                    cand2 = os.path.normpath(os.path.join(md_dir, '..', img_path))
                    if os.path.exists(cand2):
                        img_path = cand2

            if os.path.exists(img_path):
                try:
                    im = Image(img_path)
                    # scale image width to page width minus margins
                    im.drawWidth = doc.width
                    im.drawHeight = im.drawHeight * (doc.width / im.drawWidth)
                    story.append(im)
                    story.append(Spacer(1, 12))
                except Exception:
                    story.append(Paragraph(f'[Image: {img_path}]', style_p))
            else:
                story.append(Paragraph(f'[Missing image: {img_path}]', style_p))

    doc.build(story)
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--md', default='docs/sprint4_report.md')
    parser.add_argument('--out', default='docs/sprint4_report.pdf')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_pdf(args.md, args.out)


if __name__ == '__main__':
    main()
