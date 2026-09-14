"""
Image encoding/decoding tool

Provides image processing functions such as base64 encoding.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


MAX_ASPECT_RATIO = 10


def encode_image_to_base64(
    image_input: Union[str, bytes, Path],
    max_size: Optional[int] = None,
) -> Optional[str]:
    """Encode image to base64 string.

    Args:
        image_input: Image file path or byte data.
        max_size: Maximum pixel size of the longer side. If exceeded, the image is resized and re-encoded using PNG.
                  None means no resizing, directly encode the original file.

    Returns:
        base64 string; if the image is distorted (aspect ratio > MAX_ASPECT_RATIO), return None.
    """
    if max_size is None:
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                return base64.standard_b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.standard_b64encode(image_input).decode("utf-8")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    w, h = img.size
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        logger.warning(
            f"Image aspect ratio too large ({w}x{h}, ratio={aspect:.0f}:1), skip"
        )
        return None

    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"Image resized: {w}x{h} -> {new_w}x{new_h}")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
