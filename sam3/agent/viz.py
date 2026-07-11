# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import random

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .helpers.visualizer import Visualizer
from .helpers.zoom_in import render_zoom_in


def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """Deterministic wrapper around the renderer.

    Mask overlay colors are drawn with ``np.random`` / ``random`` (see
    ``helpers/color_map.py``). Left unseeded, the SAME masks get DIFFERENT colors
    on every call, which changes the rendered image and makes any downstream VLM
    judgement non-reproducible across runs (especially for color queries like
    "blue vest", where the random tint corrupts the perceived color). We seed both
    RNGs from a fixed value so identical masks always render identically, and
    save/restore the global RNG state so nothing else is affected.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(0)
    random.seed(0)
    try:
        return _visualize_impl(
            input_json,
            zoom_in_index,
            mask_alpha,
            label_mode,
            font_size_multiplier,
            boarder_width_multiplier,
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _visualize_impl(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """
    Unified visualization function.

    If zoom_in_index is None:
        - Render all masks in input_json (equivalent to visualize_masks_from_result_json).
        - Returns: PIL.Image

    If zoom_in_index is provided:
        - Returns two PIL.Images:
            1) Output identical to zoom_in_and_visualize(input_json, index).
            2) The same instance rendered via the general overlay using the color
               returned by (1), equivalent to calling visualize_masks_from_result_json
               on a single-mask json_i with color=color_hex.
    """
    # Common fields
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json["original_image_path"]

    # ---------- Mode A: Full-scene render ----------
    if zoom_in_index is None:
        boxes = np.array(input_json["pred_boxes"])
        rle_masks = [
            {"size": (orig_h, orig_w), "counts": rle}
            for rle in input_json["pred_masks"]
        ]
        binary_masks = [mask_utils.decode(rle) for rle in rle_masks]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        viz = Visualizer(
            img_rgb,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz.overlay_instances(
            boxes=boxes,
            masks=rle_masks,
            binary_masks=binary_masks,
            assigned_colors=None,
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_all_masks = Image.fromarray(viz.output.get_image())
        return pil_all_masks

    # ---------- Mode B: Zoom-in pair ----------
    else:
        idx = int(zoom_in_index)
        num_masks = len(input_json.get("pred_masks", []))
        if idx < 0 or idx >= num_masks:
            raise ValueError(
                f"zoom_in_index {idx} is out of range (0..{num_masks - 1})."
            )

        # (1) Replicate zoom_in_and_visualize
        object_data = {
            "labels": [{"noun_phrase": f"mask_{idx}"}],
            "segmentation": {
                "counts": input_json["pred_masks"][idx],
                "size": [orig_h, orig_w],
            },
        }
        pil_img = Image.open(img_path)
        pil_mask_i_zoomed, color_hex = render_zoom_in(
            object_data, pil_img, mask_alpha=mask_alpha
        )

        # (2) Single-instance render with the same color
        boxes_i = np.array([input_json["pred_boxes"][idx]])
        rle_i = {"size": (orig_h, orig_w), "counts": input_json["pred_masks"][idx]}
        bin_i = mask_utils.decode(rle_i)

        img_bgr_i = cv2.imread(img_path)
        if img_bgr_i is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb_i = cv2.cvtColor(img_bgr_i, cv2.COLOR_BGR2RGB)

        viz_i = Visualizer(
            img_rgb_i,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz_i.overlay_instances(
            boxes=boxes_i,
            masks=[rle_i],
            binary_masks=[bin_i],
            assigned_colors=[color_hex],
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_mask_i = Image.fromarray(viz_i.output.get_image())

        return pil_mask_i, pil_mask_i_zoomed
