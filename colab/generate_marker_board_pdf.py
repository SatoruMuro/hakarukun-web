from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from colab.volume_pipeline import BoardSpec, render_charuco_board


def build_pdf(output_path: Path) -> None:
    spec = BoardSpec()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        board_png = render_charuco_board(Path(directory) / "charuco-board.png", spec)
        page_width, page_height = A4
        pdf = canvas.Canvas(str(output_path), pagesize=A4, pageCompression=1)
        pdf.setTitle("Volume Hakarukun A4 ChArUco Measurement Board")
        pdf.setAuthor("Volume Hakarukun")

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawCentredString(page_width / 2, page_height - 9 * mm, "VOLUME HAKARUKUN")
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(
            page_width / 2,
            page_height - 14 * mm,
            "A4 ChArUco measurement board - print at 100% / Actual size",
        )

        board_width = spec.width_m * 1000 * mm
        board_height = spec.height_m * 1000 * mm
        board_x = (page_width - board_width) / 2
        board_y = 22 * mm
        pdf.drawImage(
            str(board_png),
            board_x,
            board_y,
            width=board_width,
            height=board_height,
            preserveAspectRatio=False,
            mask="auto",
        )
        pdf.setLineWidth(0.25)
        pdf.rect(board_x, board_y, board_width, board_height, stroke=1, fill=0)

        ruler_x = (page_width - 100 * mm) / 2
        ruler_y = 12 * mm
        pdf.setLineWidth(0.8)
        pdf.line(ruler_x, ruler_y, ruler_x + 100 * mm, ruler_y)
        for index in range(11):
            x = ruler_x + index * 10 * mm
            tick = 3 * mm if index in (0, 10) else 2 * mm
            pdf.line(x, ruler_y - tick / 2, x, ruler_y + tick / 2)
        pdf.setFont("Helvetica", 7)
        pdf.drawCentredString(page_width / 2, 6.5 * mm, "This line must measure exactly 100 mm")
        pdf.showPage()
        pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_pdf(args.output)


if __name__ == "__main__":
    main()
