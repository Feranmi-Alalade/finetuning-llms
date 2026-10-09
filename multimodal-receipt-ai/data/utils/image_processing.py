"""Image processing utilities for vision-language models."""

import math
import random
from abc import ABC, abstractmethod
from typing import Optional, List
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class StandardImageProcessor:
    """Standardizes resolution to balance quality and token efficiency.

    Uses max_side and max_pixels constraints.
    """

    def __init__(
        self,
        max_pixels: int = 1024 * 1024,  # 1.0 MP
        max_side: int = 1024,
        patch_size: Optional[int] = None,
    ):
        self.max_pixels = max_pixels
        self.max_side = max_side
        self.patch_size = patch_size

    def process(self, image: Image.Image) -> Image.Image:
        """Resize image to target constraints while maintaining aspect ratio."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        # 1. Handle max_side constraint
        if self.max_side is not None:
            if max(width, height) > self.max_side:
                scale = self.max_side / max(width, height)
                width = int(width * scale)
                height = int(height * scale)

        # 2. Handle max_pixels constraint
        if self.max_pixels is not None:
            if width * height > self.max_pixels:
                scale = math.sqrt(self.max_pixels / (width * height))
                width = int(width * scale)
                height = int(height * scale)

        # 3. Handle patch_size alignment if requested
        if self.patch_size is not None:
            width = max(self.patch_size, (width // self.patch_size) * self.patch_size)
            height = max(self.patch_size, (height // self.patch_size) * self.patch_size)

        if (width, height) != image.size:
            return image.resize((width, height), resample=Image.Resampling.LANCZOS)

        return image

    def process_for_inference(self, image: Image.Image) -> Image.Image:
        """Alias for process."""
        return self.process(image)


class Qwen3ImageProcessor(StandardImageProcessor):
    """Specialized image processor for Qwen3-VL models."""

    def __init__(
        self,
        max_pixels: int = 1024 * 1024,
        max_side: int = 1024,
        patch_size: int = 16,
    ):
        super().__init__(max_pixels=max_pixels, max_side=max_side, patch_size=patch_size)


class ImageTransformation(ABC):
    """Base class for image transformations."""

    @abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        pass


class BlurTransform(ImageTransformation):
    def __init__(self, radius: float = 2.0):
        self.radius = radius

    def apply(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius))


class PerspectiveTransform(ImageTransformation):
    def __init__(self, distortion_level: float = 0.1):
        self.distortion = distortion_level

    def apply(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        x_shift = width * self.distortion * random.uniform(-1, 1)
        y_shift = height * self.distortion * random.uniform(-1, 1)

        # Simple affine shift as a proxy for mild skew
        coeffs = (1, 0, x_shift, 0, 1, y_shift, 0, 0)
        return image.transform((width, height), Image.AFFINE, coeffs, Image.BICUBIC)


class VerticalPartialCropTransform(ImageTransformation):
    """Crops only height (top/bottom) to simulate partial receipt scans."""

    def __init__(self, scale=(0.5, 0.8)):
        self.scale = scale

    def apply(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        # Keep full width (maybe slight 2% margin to avoid edge artifacts)
        # But per user request "not by the side"
        new_h = int(height * random.uniform(*self.scale))
        top = random.randint(0, height - new_h)
        return image.crop((0, top, width, top + new_h))


class LightingTransform(ImageTransformation):
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, is_random: bool = True):
        """
        If is_random: parameters are margins around 1.0 (e.g. 0.2 -> 0.8 to 1.2)
        If not is_random: parameters are direct multipliers for the enhancer.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.is_random = is_random

    def apply(self, image: Image.Image) -> Image.Image:
        if self.is_random:
            if self.brightness:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.0 + random.uniform(-self.brightness, self.brightness))
            if self.contrast:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.0 + random.uniform(-self.contrast, self.contrast))
        else:
            if self.brightness != 0.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(self.brightness)
            if self.contrast != 0.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(self.contrast)
        return image


class PartialMaskTransform(ImageTransformation):
    """Applies a nested transformation only to a portion (top or bottom) of the image."""

    def __init__(self, transform: ImageTransformation, portion: float = 0.2):
        self.transform = transform
        self.portion = portion

    def apply(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        # Decide top or bottom, 80% top, 20% bottom
        is_top = random.choice([True] * 8 + [False] * 2)

        # Create mask
        mask = Image.new("L", (width, height), 0)
        split_y = int(height * self.portion)

        if is_top:
            mask_box = (0, 0, width, split_y)
            mask.paste(255, mask_box)
        else:
            mask_box = (0, split_y, width, height)
            mask.paste(255, mask_box)

        # Apply transformation to whole image
        transformed = self.transform.apply(image.copy())

        return Image.composite(transformed, image, mask)


class ImageTransformEngine:
    """OOP Image Transformation Engine for robustification."""

    def __init__(self, transforms: List[ImageTransformation] = None):
        self.transforms = transforms or []

    def add_transform(self, transform: ImageTransformation):
        self.transforms.append(transform)

    def apply_all(self, image: Image.Image) -> Image.Image:
        for t in self.transforms:
            image = t.apply(image)
        return image

    @classmethod
    def phone_realistic_pipeline(cls):
        """Pre-configured pipeline for phone-like distortions."""
        engine = cls()
        engine.add_transform(PerspectiveTransform(0.05))
        engine.add_transform(LightingTransform(0.1, 0.1))
        engine.add_transform(BlurTransform(random.uniform(0.5, 1.5)))
        return engine
