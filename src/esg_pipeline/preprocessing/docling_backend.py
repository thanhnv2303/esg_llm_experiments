from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import DoclingDocument
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling_core.types.doc import Size
    from docling_core.types.doc.base import ImageRefMode
    from docling_core.types.doc.document import ImageRef, PictureItem
except ImportError:  # pragma: no cover - optional dependency
    DoclingParseDocumentBackend = None  # type: ignore[assignment]
    InputFormat = None  # type: ignore[assignment]
    DoclingDocument = None  # type: ignore[assignment]
    PdfPipelineOptions = None  # type: ignore[assignment]
    DocumentConverter = None  # type: ignore[assignment]
    FormatOption = None  # type: ignore[assignment]
    StandardPdfPipeline = None  # type: ignore[assignment]
    Size = None  # type: ignore[assignment]
    ImageRefMode = None  # type: ignore[assignment]
    ImageRef = None  # type: ignore[assignment]
    PictureItem = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


DoclingImageMode = Literal["embedded", "referenced"]


@dataclass
class DoclingPageArtifacts:
    markdown_path: Path
    markdown_text: str
    image_paths: List[Path]
    combined_image_path: Optional[Path] = None


_DOCLING_CONVERTER: Optional[DocumentConverter] = None


def _docling_available() -> bool:
    return (
        DocumentConverter is not None
        and FormatOption is not None
        and InputFormat is not None
        and PdfPipelineOptions is not None
        and StandardPdfPipeline is not None
        and DoclingParseDocumentBackend is not None
        and PictureItem is not None
        and ImageRef is not None
        and Size is not None
        and ImageRefMode is not None
    )


def _get_docling_converter() -> DocumentConverter:
    if not _docling_available():  # pragma: no cover - guarded by caller
        raise RuntimeError(
            "Docling is not available. Install 'docling' to enable docling-based extraction."
        )

    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        pdf_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            generate_table_images=True,
        )
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pdf_options,
                backend=DoclingParseDocumentBackend,
            )
        }
        _DOCLING_CONVERTER = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options=format_options,
        )
    return _DOCLING_CONVERTER


def _page_picture_items(doc: DoclingDocument, page_no: int) -> List[PictureItem]:
    items: List[PictureItem] = []
    for item, _level in doc.iterate_items(with_groups=False):
        if isinstance(item, PictureItem) and item.prov:
            if any(prov.page_no == page_no for prov in item.prov):
                items.append(item)
    return items


def _save_image_ref(
    item: PictureItem,
    doc: DoclingDocument,
    image_dir: Path,
    rel_root: Path,
    base_name: str,
    index: int,
) -> Optional[Tuple[Path, str]]:
    image = item.get_image(doc=doc)
    if image is None:
        return None

    filename = f"{base_name}_img{index}.png"
    output_path = image_dir / filename
    image.save(output_path, format="PNG")

    try:
        dpi_info = image.info.get("dpi")  # type: ignore[attr-defined]
        dpi = int(dpi_info[0]) if isinstance(dpi_info, tuple) and dpi_info else 72
    except Exception:  # pragma: no cover - best effort
        dpi = 72

    relative_path = Path(os.path.relpath(output_path, rel_root))
    item.image = ImageRef(
        mimetype="image/png",
        dpi=dpi,
        size=Size(width=image.width, height=image.height),
        uri=relative_path,
    )
    return output_path, relative_path.as_posix()


def _combine_images_with_labels(
    labeled_images: List[Tuple[Path, str]],
    output_path: Path,
) -> Optional[Path]:
    if Image is None or ImageDraw is None or ImageFont is None:
        return None
    if not labeled_images:
        return None

    images_with_labels: List[Tuple[Image.Image, str]] = []
    for path, label in labeled_images:
        try:
            img = Image.open(path).convert("RGB")
            images_with_labels.append((img, label))
        except OSError:
            continue

    if not images_with_labels:
        return None

    font = ImageFont.load_default()
    padding = 10
    gap_between_blocks = 15
    label_padding = 6

    def _text_size(text: str) -> Tuple[int, int]:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height
        width, height = font.getsize(text)
        return width, height

    max_width = 0
    total_height = padding
    measurements: List[Tuple[Image.Image, str, int, int]] = []

    for img, label in images_with_labels:
        label_width, label_height = _text_size(label)
        block_width = max(img.width, label_width)
        block_height = img.height + label_height + label_padding * 2
        measurements.append((img, label, label_width, label_height))
        max_width = max(max_width, block_width)
        total_height += block_height + gap_between_blocks

    total_height = total_height - gap_between_blocks + padding
    canvas_width = max_width + padding * 2
    combined = Image.new("RGB", (canvas_width, total_height), color="white")
    draw = ImageDraw.Draw(combined)

    y_offset = padding
    for img, label, label_width, label_height in measurements:
        x_img = padding + (max_width - img.width) // 2
        combined.paste(img, (x_img, y_offset))
        y_offset += img.height + label_padding
        x_label = padding + (max_width - label_width) // 2
        draw.text((x_label, y_offset), label, fill="black", font=font)
        y_offset += label_height + label_padding + gap_between_blocks

    combined.save(output_path, format="PNG")

    for img, _ in images_with_labels:
        img.close()

    return output_path


def extract_page_with_docling(
    pdf_path: Path,
    page_number: int,
    markdown_dir: Path,
    images_dir: Optional[Path] = None,
    image_mode: DoclingImageMode = "embedded",
    filename_prefix: Optional[str] = None,
) -> DoclingPageArtifacts:
    pdf_path = pdf_path.expanduser().resolve()
    markdown_dir.mkdir(parents=True, exist_ok=True)

    converter = _get_docling_converter()

    conversion = converter.convert(
        str(pdf_path),
        # max_num_pages=1,
        page_range=(page_number, page_number),
    )

    doc = conversion.document
    prefix = filename_prefix or pdf_path.stem
    markdown_path = markdown_dir / f"{prefix}_p{page_number}.md"

    saved_images: List[Path] = []
    combined_image_path: Optional[Path] = None
    export_mode = ImageRefMode.EMBEDDED

    if image_mode == "referenced":
        if images_dir is None:
            raise ValueError("images_dir must be provided when image_mode='referenced'")
        images_dir.mkdir(parents=True, exist_ok=True)
        pictures = _page_picture_items(doc, page_number)
        labeled_images: List[Tuple[Path, str]] = []
        for index, picture in enumerate(pictures, start=1):
            saved = _save_image_ref(
                picture,
                doc,
                images_dir,
                markdown_dir,
                f"{prefix}_p{page_number}",
                index,
            )
            if saved:
                path, label = saved
                saved_images.append(path)
                labeled_images.append((path, label))
        export_mode = ImageRefMode.REFERENCED
        combined_output = images_dir / f"{prefix}_combined.png"
        combined_created = _combine_images_with_labels(labeled_images, combined_output)
        if combined_created:
            combined_image_path = combined_created
    elif image_mode != "embedded":
        raise ValueError(f"Unsupported image_mode '{image_mode}'.")

    markdown_text = doc.export_to_markdown(
        page_no=page_number,
        image_mode=export_mode,
    )
    markdown_path.write_text(markdown_text, encoding="utf-8")

    return DoclingPageArtifacts(
        markdown_path=markdown_path,
        markdown_text=markdown_text,
        image_paths=saved_images,
        combined_image_path=combined_image_path,
    )


__all__ = [
    "DoclingImageMode",
    "DoclingPageArtifacts",
    "extract_page_with_docling",
]
