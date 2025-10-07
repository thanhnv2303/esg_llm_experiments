from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_page_as_image(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    dpi: int = 200,
    filename_prefix: Optional[str] = None,
) -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    _ensure_output_dir(output_dir)
    prefix = filename_prefix or pdf_path.stem
    output_path = output_dir / f"{prefix}_p{page_number}.png"

    if fitz is not None:
        with fitz.open(pdf_path) as doc:  # pragma: no cover - requires PyMuPDF
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(dpi=dpi)
            pix.save(output_path)
        return output_path

    return _extract_page_image_via_cli(pdf_path, page_number, output_dir, output_path)


def _extract_page_image_via_cli(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    output_path: Path,
) -> Path:
    import shutil

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:  # pragma: no cover - depends on system command
        raise RuntimeError(
            "Cannot export PDF page as image: PyMuPDF not installed and 'pdftoppm' command not available."
        )

    base = output_path.with_suffix("")
    cmd = [
        pdftoppm,
        "-singlefile",
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-png",
        str(pdf_path),
        str(base),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def extract_page_as_text(pdf_path: Path, page_number: int) -> str:
    pdf_path = pdf_path.expanduser().resolve()

    if fitz is not None:
        with fitz.open(pdf_path) as doc:  # pragma: no cover - requires PyMuPDF
            page = doc.load_page(page_number - 1)
            return page.get_text()

    return _extract_page_text_via_cli(pdf_path, page_number)


def _extract_page_text_via_cli(pdf_path: Path, page_number: int) -> str:
    import shutil

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:  # pragma: no cover - depends on system command
        raise RuntimeError(
            "Cannot extract text from PDF: PyMuPDF not installed and 'pdftotext' command not available."
        )

    with subprocess.Popen(
        [
            pdftotext,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-layout",
            str(pdf_path),
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        stdout, stderr = proc.communicate()
    if proc.returncode != 0:  # pragma: no cover
        raise RuntimeError(f"pdftotext failed with exit code {proc.returncode}: {stderr.strip()}")
    return stdout


__all__ = ["extract_page_as_image", "extract_page_as_text"]
