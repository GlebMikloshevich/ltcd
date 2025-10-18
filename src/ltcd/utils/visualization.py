import numpy as np

from PIL import Image, ImageDraw

def draw_keypoints(
    image: Image,
    keypoints: list[float] | list[list[float]],
    # colors: tuple[int] | list[tuple[int]] = (255, 0, 0),
    colors: tuple[int] = (255, 0, 0),
    fill: bool = True,
) -> Image:
    image_cp = image.copy()
    image_draw = ImageDraw.Draw(image_cp)
    if isinstance(keypoints[0], float | np.number):
        keypoints = [keypoints]
        colors = colors[colors]

    if len(keypoints) != len(colors):
        msg = (
            "len of `colors` must be equal to len of `keypoints`. "
            f"Got {len(keypoints)} and {len(colors)}.\n"
            "Make sure that if you pass single keypoints list you must pass single color.\n"
            "if you pass list of keypoints list you must pass list of colors.\n"
        )
        raise ValueError(msg)


    keypoint_size = max(1, int(min(image.size) / 100))
    width_size = max(1, int(keypoint_size / 2.5))

    for table_keypoints in keypoints:
        # image_draw.point(keypoint, fill=colors)
        fill_color = colors if fill else None
        image_draw.circle(table_keypoints, radius=keypoint_size, fill=fill_color, outline=colors, width=width_size)
    return image_cp

