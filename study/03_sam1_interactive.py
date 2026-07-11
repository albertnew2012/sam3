"""
Study 03 — SAM 1-style interactive segmentation (point / box clicks).

SAM 3 also subsumes the classic SAM 1 "click a point, get one object's mask"
workflow. This path is OFF by default; you turn it on with
    build_sam3_image_model(enable_inst_interactivity=True)
which attaches an interactive mask decoder. You then call model.predict_inst().

Contrast with studies 01/02:
  - Text/visual prompts  -> EXHAUSTIVE: every instance of a concept.
  - Point/box clicks here -> ONE object, the thing you clicked, SAM 1 semantics.

We demo the three canonical SAM 1 interactions:
  A) one positive point, multimask_output=True -> 3 candidate masks + scores
  B) positive + negative points to refine a single object
  C) a box prompt

Coordinates here are ABSOLUTE PIXELS (x, y). Points get labels 1=foreground,
0=background. Open study/outputs/03_*.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from _common import ASSETS, out, setup_runtime


def show_mask(mask, ax, color=(30 / 255, 144 / 255, 255 / 255, 0.6)):
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1))


def show_points(coords, labels, ax, s=300):
    pos, neg = coords[labels == 1], coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color="lime", marker="*", s=s, edgecolor="white")
    ax.scatter(neg[:, 0], neg[:, 1], color="red", marker="*", s=s, edgecolor="white")


def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, ec="lime", fc=(0, 0, 0, 0), lw=2)
    )


def save(image, masks, scores, tag, points=None, labels=None, box=None):
    # masks: (num_masks, H, W). Keep the highest-scoring mask for the figure.
    best = int(np.argmax(scores))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(image)
    show_mask(masks[best], ax)
    if points is not None:
        show_points(points, labels, ax)
    if box is not None:
        show_box(box, ax)
    ax.set_title(f"{tag}  best score={scores[best]:.3f}  ({len(masks)} candidate masks)")
    ax.axis("off")
    path = out(f"03_{tag}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    saved {path}")


def main():
    setup_runtime()

    # enable_inst_interactivity=True attaches the SAM 1-style interactive decoder.
    model = build_sam3_image_model(enable_inst_interactivity=True)
    processor = Sam3Processor(model)

    image = Image.open(f"{ASSETS}/images/truck.jpg").convert("RGB")
    state = processor.set_image(image)

    # A) single positive click; ask for 3 candidate masks (ambiguity resolution).
    pts = np.array([[520, 375]])
    lbl = np.array([1])
    masks, scores, logits = model.predict_inst(
        state, point_coords=pts, point_labels=lbl, multimask_output=True
    )
    order = np.argsort(scores)[::-1]
    masks, scores, logits = masks[order], scores[order], logits[order]
    print(f"A) 1 point, multimask -> {len(masks)} masks, scores={np.round(scores,3)}")
    save(image, masks, scores, "point_multimask", points=pts, labels=lbl)

    # B) refine with an extra negative click, feeding back the best mask logits.
    pts = np.array([[500, 375], [1125, 625]])
    lbl = np.array([1, 0])
    mask_input = logits[np.argmax(scores)][None]  # best mask from step A
    masks, scores, _ = model.predict_inst(
        state,
        point_coords=pts,
        point_labels=lbl,
        mask_input=mask_input,
        multimask_output=False,
    )
    print(f"B) refine with +/- points -> score={scores[0]:.3f}")
    save(image, masks, scores, "point_refine", points=pts, labels=lbl)

    # C) a single box prompt (xyxy pixels).
    box = np.array([425, 600, 700, 875])
    masks, scores, _ = model.predict_inst(
        state, point_coords=None, point_labels=None, box=box[None], multimask_output=False
    )
    print(f"C) box prompt -> score={scores[0]:.3f}")
    save(image, masks, scores, "box", box=box)


if __name__ == "__main__":
    main()
