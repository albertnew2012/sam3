"""
Study 02 — Image segmentation with VISUAL (box) prompts + exemplars.

Sometimes a concept is hard to name ("that specific striped thing"). SAM 3 lets
you point at an example with a box instead of / in addition to text. The model
then finds everything that RESEMBLES your positive boxes while AVOIDING things
that look like your negative boxes.

This uses the same Sam3Processor as study 01, but instead of set_text_prompt we
call add_geometric_prompt(box, label). Boxes are given in NORMALIZED cxcywh
(center-x, center-y, width, height, all in [0,1]) — the helpers below convert
from the more familiar pixel xywh (top-left + size).

We demo two things on the same image:
  A) one positive box  -> "find more like this"
  B) positive + negative box -> "find like #1 but NOT like #2"

Open study/outputs/02_*.png to compare.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results

from _common import ASSETS, out, setup_runtime


def pixel_xywh_to_norm_cxcywh(box_xywh, w, h):
    """[x, y, w, h] pixels (top-left) -> [cx, cy, w, h] normalized to [0,1]."""
    t = torch.tensor(box_xywh, dtype=torch.float32).view(-1, 4)
    cxcywh = box_xywh_to_cxcywh(t)
    return normalize_bbox(cxcywh, w, h).flatten().tolist()


def main():
    setup_runtime()

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    image_path = f"{ASSETS}/images/test_image.jpg"
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    state = processor.set_image(image)

    # --- A) single positive box (an exemplar of one shoe) --------------------
    pos_box = [480.0, 290.0, 110.0, 360.0]  # pixel xywh
    processor.reset_all_prompts(state)
    processor.add_geometric_prompt(
        state=state, box=pixel_xywh_to_norm_cxcywh(pos_box, w, h), label=True
    )
    print(f"A) 1 positive box -> {len(state['scores'])} detection(s)")
    plot_results(Image.open(image_path), state)
    plt.savefig(out("02_visual_positive_only.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # --- B) positive + negative box ------------------------------------------
    # Positive box selects a target; negative box tells the model to suppress
    # things that look like that second region.
    boxes = [[480.0, 290.0, 110.0, 360.0], [370.0, 280.0, 115.0, 375.0]]
    labels = [True, False]
    processor.reset_all_prompts(state)
    for box, label in zip(boxes, labels):
        processor.add_geometric_prompt(
            state=state, box=pixel_xywh_to_norm_cxcywh(box, w, h), label=label
        )
    print(f"B) positive + negative box -> {len(state['scores'])} detection(s)")
    plot_results(Image.open(image_path), state)
    plt.savefig(out("02_visual_pos_and_neg.png"), bbox_inches="tight", dpi=150)
    plt.close()

    print("saved 02_visual_positive_only.png and 02_visual_pos_and_neg.png")


if __name__ == "__main__":
    main()
