"""Image and PDF preparation, borrowed from VoucherVisionGO's app.py.

HEIC/HEIF uploads are decoded through pillow-heif and always re-encoded as
JPEG. Images above the pixel cap are downscaled with Lanczos resampling.
"""

from __future__ import annotations

import io
import logging
import math
import os

from PIL import Image
from pillow_heif import register_heif_opener
from werkzeug.datastructures import FileStorage

register_heif_opener()

logger = logging.getLogger(__name__)

DEFAULT_MAX_PIXELS = 5_200_000
MAX_PDF_PAGES = 200
PDF_RENDER_DPI = 150


def resize_image_to_max_pixels(image, max_pixels=DEFAULT_MAX_PIXELS):
    """Resize an image to at most `max_pixels`, keeping the aspect ratio."""
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_pixels:
        return image

    scale_factor = math.sqrt(max_pixels / current_pixels)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Ensure we don't exceed the pixel limit due to rounding
    while new_width * new_height > max_pixels:
        if new_width > new_height:
            new_width -= 1
        else:
            new_height -= 1

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Image resized: {width}x{height} -> {new_width}x{new_height}")
    return resized_image


def _to_rgb_for_jpeg(image):
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        return background
    if image.mode != "RGB":
        # Catches YCbCr, CMYK, etc. (common in HEIC)
        return image.convert("RGB")
    return image


def encode_resized(image, filename, max_pixels=DEFAULT_MAX_PIXELS):
    """Resize and re-encode; returns (BytesIO, filename, content_type). HEIC becomes JPEG."""
    resized_image = resize_image_to_max_pixels(image, max_pixels)
    image_format = "PNG" if filename and filename.lower().endswith(".png") else "JPEG"
    if image_format == "JPEG":
        resized_image = _to_rgb_for_jpeg(resized_image)
    save_kwargs = {"format": image_format}
    if image_format == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["optimize"] = True
    buffer = io.BytesIO()
    resized_image.save(buffer, **save_kwargs)
    buffer.seek(0)
    if image_format == "JPEG" and filename and not filename.lower().endswith((".jpg", ".jpeg")):
        filename = f"{os.path.splitext(filename)[0]}.jpg"
    return buffer, filename, f"image/{image_format.lower()}"


def process_uploaded_file_with_resize(file, max_pixels=DEFAULT_MAX_PIXELS):
    """Return the upload itself, or a new FileStorage with a resized/re-encoded image.

    Raises on an unreadable image. (VoucherVisionGO returned the original file
    on error and failed later; raising lets the route answer 400 right away.)
    """
    image = Image.open(file.stream)
    image.load()
    original_filename = file.filename
    width, height = image.size
    is_heic = bool(original_filename) and original_filename.lower().endswith((".heic", ".heif"))
    if width * height <= max_pixels and not is_heic:
        file.stream.seek(0)
        return file

    buffer, filename, content_type = encode_resized(image, original_filename, max_pixels)
    return FileStorage(stream=buffer, filename=filename, content_type=content_type)


def convert_pdf_to_page_images(pdf_bytes, pdf_filename, dpi=PDF_RENDER_DPI, max_pages=None):
    """Render each PDF page to an in-memory JPEG FileStorage.

    Pages are named {pdf_stem}__page_0001.jpg, {pdf_stem}__page_0002.jpg, ...
    With `max_pages`, a longer PDF raises ValueError before rendering anything.
    """
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if max_pages is not None and len(doc) > max_pages:
            raise PdfTooLong(len(doc), max_pages)
        page_files = []
        base_name = os.path.splitext(pdf_filename)[0]
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
            page_files.append(FileStorage(
                stream=io.BytesIO(img_bytes),
                filename=f"{base_name}__page_{page_num + 1:04d}.jpg",
                content_type="image/jpeg",
            ))
        return page_files
    finally:
        doc.close()


class PdfTooLong(ValueError):
    def __init__(self, page_count, max_pages):
        super().__init__(f"PDF has {page_count} pages, maximum allowed is {max_pages}")
        self.page_count = page_count
        self.max_pages = max_pages
