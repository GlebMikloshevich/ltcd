import random

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePath

import albumentations as A
import cv2
import numpy as np
import yaml

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from ltcd.generators.utils import generate_gaussian

class DocumentField(BaseModel):
    x: int
    y: int
    lines: int = Field(default=0)
    first_line_indent: int = Field(default=0)
    anchor: str = "la"
    dx: int = 0
    dy: int = 0
    field_type: str = "text"
    font_name: str = "roboto.ttf"
    font_size: int = 14
    color: list[str] = ["#000000"]


class Table(BaseModel):
    table_bbox: list[int] = Field(default_factory=list)
    line_intersection: list[list[int]] = Field(default_factory=list)


class DocumentConfig(BaseModel):
    doc_name: str
    fields: dict[str, DocumentField] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)


class DocumentGenerator:
    config: DocumentConfig

    def __init__(
        self,
        config: DocumentConfig,
        document_form_image: Image,
        fonts: dict,
        backgrounds: list,
    ) -> None:
        self.config = config
        self.document_form_image = document_form_image
        self.fonts = fonts
        self.backgrounds = backgrounds

        self.text_augmentation = self._get_default_text_augmentation()
        self.document_augmentation = self._get_default_document_augmentation()
        self.scene_augmentation = self._get_default_scene_augmentation()

    @classmethod
    def from_folder(cls, path_to_folder: str | PurePath) -> "DocumentGenerator":
        path_to_folder = Path(path_to_folder)
        path_to_config = path_to_folder/ "config.yaml"
        with path_to_config.open() as f:
            config = DocumentConfig.model_validate(yaml.safe_load(f))

        # Load document form image and force it into memory (thread-safe)
        document_form_image = Image.open(path_to_folder / "document.png").convert("RGBA")
        document_form_image.load()  # Force full load into memory for thread safety

        fonts_folder = path_to_folder / "fonts"
        fonts = {}
        for field_config in config.fields.values():
            if (field_config.font_name, field_config.font_size) in fonts:
                continue
            fonts[(field_config.font_name, field_config.font_size)] = ImageFont.truetype(
                font=fonts_folder / field_config.font_name,
                size=field_config.font_size,
            )
        backgrounds = []

        for bg_image_path in (path_to_folder / "backgrounds").iterdir():
            # Load backgrounds and force into memory (thread-safe)
            bg_img = Image.open(bg_image_path).convert("RGB")
            bg_img.load()
            backgrounds.append(bg_img)

        return cls(config, document_form_image, fonts, backgrounds)

    def generate(
        self,
        batch_data: list[dict[str, Sequence[str] | str]],
        max_workers: int = 6,
        chunk_size: int = 500,  # Process in chunks to avoid OOM
    ) -> list[tuple[Image, list[float]]]:
        """
        Generate document images from batch data.

        Args:
            batch_data: List of field data dictionaries
            max_workers: Number of parallel workers per chunk
            chunk_size: Maximum samples per chunk (prevents OOM on large batches)

        Returns:
            List of (image, keypoints, mask) tuples in same order as batch_data
        """
        # For small batches, process directly
        if len(batch_data) <= chunk_size:
            return self._generate_chunk(batch_data, max_workers)

        # For large batches, process in chunks
        all_results = []
        num_chunks = (len(batch_data) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(batch_data))
            chunk = batch_data[start_idx:end_idx]

            chunk_results = self._generate_chunk(chunk, max_workers)
            all_results.extend(chunk_results)

            # Explicit garbage collection for large batches
            if len(batch_data) > 1000 and (i + 1) % 5 == 0:
                import gc
                gc.collect()

        return all_results

    def generate_to_folder(
        self,
        path: str,
        batch_data: list[dict[str, Sequence[str] | str]],
        subfolder_name: str = "train",
        max_workers: int = 6,
        chunk_size: int = 500,  # Process in chunks to avoid OOM
    ) -> list[tuple[Image, list[float]]]:
        """
        Generate document images from batch data.

        Args:
            batch_data: List of field data dictionaries
            max_workers: Number of parallel workers per chunk
            chunk_size: Maximum samples per chunk (prevents OOM on large batches)

        Returns:
            List of (image, keypoints, mask) tuples in same order as batch_data
        """
        path = Path(path)
        # For small batches, process directly
        subfolder = path / subfolder_name
        subfolder.mkdir()
        data_folders = ("images", "keypoints", "masks")
        for data_folder in data_folders:
            (subfolder / data_folder).mkdir()

        num_chunks = (len(batch_data) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            print(f"generating chunk {str(i+1).zfill(len(str(num_chunks)))}/{num_chunks}")
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(batch_data))
            chunk = batch_data[start_idx:end_idx]

            chunk_results = self._generate_chunk(chunk, max_workers)

            # print(len(chunk_results))
            # print(chunk_results[0])
            for j, data_sample in enumerate(chunk_results):

                # for data_folder in data_folders:
                #     (subfolder / data_folder).mkdir()
                # for j, sample in enumerate(data_split):
                    # print(sample)
                image, keypoints, mask = data_sample

                image.save(subfolder / data_folders[0] / f"{start_idx + j}.png")

                with (subfolder / data_folders[1] / f"{start_idx + j}.npy").open("wb") as f:
                    np.save(f, keypoints)

                with (subfolder / data_folders[2] / f"{start_idx + j}.npy").open("wb") as f:
                    np.save(f, mask)

            # all_results.extend(chunk_results)

            # Explicit garbage collection for large batches
            if len(batch_data) > 1000 and (i + 1) % 5 == 0:
                import gc
                gc.collect()

        # return all_results

    def _generate_chunk(
        self,
        batch_data: list[dict[str, Sequence[str] | str]],
        max_workers: int = 6,
    ) -> list[tuple[Image, list[float]]]:
        """Generate a single chunk of data."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate, data) for data in batch_data]

            # Collect results in order
            ordered_results = [f.result() for f in futures]
        # print(ordered_results[0])
        return ordered_results

    def _get_default_text_augmentation(self) -> A.augmentations:
        text_augmentation = A.Compose([
            A.RandomBrightnessContrast(
                p=0.6,
            ),
            A.HueSaturationValue(
                p=0.5,
            ),
            A.RGBShift(
                p=0.4,
            ),

            A.GaussNoise(p=0.4),
            A.MotionBlur(
                p=0.3,
            ),
            A.ImageCompression(
                quality_range=(70, 100),
                p=0.2,
            ),
            A.ISONoise(p=0.3),
        ])
        return text_augmentation

    def _get_default_document_augmentation(self) -> A.augmentations:
        document_augmentation = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=600,
                    # pad_height_divisor=32,
                    min_width=600,
                    # pad_width_divisor=32,
                    border_mode=cv2.BORDER_CONSTANT,
                    # value=(255, 255, 255),
                    # mask_value=0,
                ),
                A.Affine(
                    scale=(0.7, 1.2),
                    rotate=8,
                    p=0.8,
                    fit_output=True,
                ),

                A.RandomBrightnessContrast(
                    p=0.5,
                ),
                A.RGBShift(
                    p=0.4,
                ),
                A.GaussNoise(p=0.3),
                A.MotionBlur(
                    p=0.3,
                ),
                A.ElasticTransform(
                    alpha=30,
                    sigma=5,
                    p=0.2,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    p=0.2,
                ),
                A.ImageCompression(
                    quality_range=(70, 100),
                    p=0.7,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
        )
        return document_augmentation

    def _get_default_scene_augmentation(self) -> A.augmentations:
        scene_augmentation = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=600,
                    # pad_height_divisor=32,
                    min_width=600,
                    border_mode=cv2.BORDER_CONSTANT,
                    # value=(255, 255, 255),
                    # mask_value=0,
                ),
                A.Affine(
                    shear=(-5, 5),
                    scale=(0.8, 1.2),
                    translate_percent=[-0.05, 0.05],
                    rotate=(-30, 30),
                    fit_output=True,
                    p=0.6,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6,
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=0.4,
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    p=0.3,
                ),
                A.MotionBlur(
                    blur_limit=3,
                    p=0.3,
                ),
                A.ImageCompression(
                    quality_range=(60, 100),
                    p=0.4,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
        )
        return scene_augmentation

    def _generate(
        self,
        data: dict[str, Sequence[str] | str],
    ) -> tuple[Image, list[float]]:
        keypoints = self.config.tables[0].line_intersection
        # TODO(Gleb): make mask optional
        form_layer, text_layer, mask = self._generate_form_text_layer_mask(data, keypoints)
        background_layer = self._get_background()

        text_layer = self._augment_text_layer(text_layer)
        document = self._add_text(form_layer, text_layer)
        document, keypoints, mask = self._augment_form_layer(document, keypoints=np.array(keypoints), mask=mask)
        # augment heatmap2
        scene_image, keypoints, mask = self._add_background(background_layer, document, keypoints, mask=mask)
        # augment heatmap3
        scene_image, keypoints, _ = self._augment_scene_image(scene_image, keypoints, mask=mask)

        if len(keypoints) < len(self.config.tables[0].line_intersection):
            return self._generate(data)

        # Regenerate mask from final float-precision keypoints to avoid bilinear
        # interpolation drift that accumulates when the heatmap is warped through
        # two augmentation stages.
        width, height = scene_image.size
        mask = self._mask_from_keypoints(keypoints, width, height)
        return scene_image, keypoints, mask

    def _mask_from_keypoints(
        self,
        keypoints,
        width: int,
        height: int,
        keypoint_focal_scale: int = 31,
    ) -> np.ndarray:
        kfs = int(min(width / keypoint_focal_scale, height / keypoint_focal_scale))
        mask = np.zeros((height, width), dtype=np.float32)
        gaussian = generate_gaussian(kfs, kfs)
        for kp in keypoints:
            kx, ky = int(round(kp[0])), int(round(kp[1]))
            gsx = kx - kfs // 2
            gsy = ky - kfs // 2
            my1, my2 = max(0, gsy), min(height, gsy + kfs)
            mx1, mx2 = max(0, gsx), min(width, gsx + kfs)
            gy1 = my1 - gsy
            gx1 = mx1 - gsx
            mask[my1:my2, mx1:mx2] = np.maximum(
                mask[my1:my2, mx1:mx2],
                gaussian[gy1:gy1 + (my2 - my1), gx1:gx1 + (mx2 - mx1)],
            )
        return mask


    def _generate_form_text_layer_mask(
        self,
        data: dict[str, Sequence[str] | str],
        keypoints: list[list[float]],
        keypoint_focal_scale: int = 31,
    ) -> list[list[float]]:
        form_layer = self.document_form_image.copy()
        width, height = form_layer.size
        keypoint_focal_size = int(min(width / keypoint_focal_scale, height / keypoint_focal_scale))

        mask = np.zeros((height, width), dtype=np.float32)
        text_layer = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8))
        text_layer_draw = ImageDraw.Draw(text_layer)

        for field_name, field_value in data.items():
            field_config = self.config.fields[field_name]
            font = self.fonts[(field_config.font_name, field_config.font_size)]
            dx = np.random.randint(0, field_config.dx, size=1)
            dy = np.random.randint(0, field_config.dy, size=1)

            text_layer_draw.text(
                xy=(field_config.x + dx, field_config.y + dy),
                text=field_value,
                fill=random.choice(field_config.color),  # noqa: S311
                font=font,
                anchor=field_config.anchor,
            )

        gaussian = generate_gaussian(keypoint_focal_size, keypoint_focal_size)
        for (keypoint_x, keypoint_y) in keypoints:
            gaussian_start_x = keypoint_x - keypoint_focal_size // 2
            gaussian_start_y = keypoint_y - keypoint_focal_size // 2
            mask[
                gaussian_start_y: gaussian_start_y + keypoint_focal_size,
                gaussian_start_x: gaussian_start_x + keypoint_focal_size,
            ] = gaussian

        return form_layer, text_layer, mask

    def _add_text(self, form_layer: Image, text_layer: Image) -> Image:
        return Image.alpha_composite(form_layer, text_layer)

    def _add_background(
            self,
            background: Image,
            foreground: Image,
            keypoints: list[list[float]],
            mask: np.ndarray,
        ) -> tuple[Image, list[list[float]]]:
        bg_width, bg_height = background.size
        fg_width, fg_height = foreground.size

        if bg_width > fg_width or bg_height > fg_height:
            x_start = np.random.randint(0, bg_width - fg_width, size=1)[0]
            y_start = np.random.randint(0, bg_height - fg_height, size=1)[0]

            background = background.crop(
                (
                    max(x_start, 0), max(y_start, 0),
                    max(bg_width, bg_width-fg_width), max(bg_height, bg_height-fg_height),
                ),
            )

        bg_width, bg_height = background.size
        if bg_width != fg_width or bg_height != fg_height:
            background = background.resize(
                size=(fg_width, fg_height),
                resample=Image.Resampling.BICUBIC,
            )
            mask = cv2.resize(mask, (fg_width, fg_height))

        background = background.convert("RGBA")
        foreground = foreground.convert("RGBA")
        return Image.alpha_composite(background, foreground), keypoints, mask

    def _get_background(self) -> Image:
        return random.choice(self.backgrounds).copy()  # noqa: S311

    def _augment_text_layer(self, text_layer: Image) -> Image:
        image_np = np.array(text_layer)
        rgb_image_np = image_np[..., :3]
        alpha_image_np = image_np[..., 3]
        augmented = self.text_augmentation(image=rgb_image_np)

        augmented_image_np = np.dstack([augmented["image"], alpha_image_np])
        augmented_image_pil = Image.fromarray(augmented_image_np.astype(np.uint8), mode="RGBA")

        return augmented_image_pil

    def _augment_form_layer(
        self,
        form_layer: Image,
        keypoints: list = None,
        mask: np.ndarray | None = None,
    ) -> tuple[Image, list[list[int]], np.ndarray]:
        image_np = np.array(form_layer)
        rgb_image_np = image_np[..., :3]
        alpha_image_np = image_np[..., 3]

        if keypoints is None:
            keypoints = []


        masks = np.stack((alpha_image_np, mask), dtype=np.float32)
        augmented = self.document_augmentation(image=rgb_image_np, masks=masks, keypoints=keypoints)
        augmented_alpha = augmented["masks"][0].astype(np.uint8)
        augmented_mask = augmented["masks"][1]
        augmented_image = np.dstack([augmented["image"], augmented_alpha])
        augmented_keypoints = augmented["keypoints"]

        augmented_image_pil = Image.fromarray(augmented_image, mode="RGBA")

        return augmented_image_pil, augmented_keypoints, augmented_mask

    def _augment_scene_image(
        self,
        scene_image: Image,
        keypoints: list = None,
        mask: np.ndarray | None = None,
    ) -> tuple[Image, list[list[int]], np.ndarray]:
        image_np = np.array(scene_image)
        rgb_image_np = image_np[..., :3]
        alpha_image_np = image_np[..., 3]

        if keypoints is None:
            keypoints = []


        masks = np.stack((alpha_image_np, mask), dtype=np.float32)
        augmented = self.scene_augmentation(image=rgb_image_np, masks=masks, keypoints=keypoints)
        augmented_alpha = augmented["masks"][0].astype(np.uint8)
        augmented_mask = augmented["masks"][1]
        augmented_image = np.dstack([augmented["image"], augmented_alpha])
        augmented_keypoints = augmented["keypoints"]

        augmented_image_pil = Image.fromarray(augmented_image, mode="RGBA")

        return augmented_image_pil, augmented_keypoints, augmented_mask
    # [                      #  sample
    #     [[0, 0], [0, 1]],  #  first row
    #     [[0, 5], [0, 5]],  #  second row
    # ]                      #  sample end
