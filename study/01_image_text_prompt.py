"""
Study 01 — Open-vocabulary image segmentation with TEXT prompts.

This is SAM 3's headline capability and the best place to start. Unlike SAM 1/2
(which segment ONE object per click), SAM 3 takes a short noun phrase and
EXHAUSTIVELY finds every instance of that concept in the image.

Pipeline:
    build_sam3_image_model()          # detector+segmenter, weights from HF
      -> Sam3Processor(model)         # thin wrapper: preprocessing + prompting
        -> processor.set_image(img)   # runs the vision backbone ONCE
          -> processor.set_text_prompt(state, "car")   # cheap, re-run per prompt

Key idea: set_image() is the expensive part (backbone). You can then fire many
different text prompts against the same cached image state for free-ish.

Each detection comes back with a mask, an xyxy box, and a confidence score.
Only detections above `confidence_threshold` are returned.

Run it and open study/outputs/01_*.png.
"""

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

from _common import ASSETS, out, setup_runtime


def main():
    setup_runtime()

    # Build the image model (downloads facebook/sam3 checkpoint on first run).
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    image_path = f"{ASSETS}/images/test_image.jpg"
    image = Image.open(image_path).convert("RGB")

    # Expensive step: encode the image once, reuse for every prompt below.
    state = processor.set_image(image)

    # Try several concepts on the SAME cached image to build intuition for the
    # "concept granularity" SAM 3 handles. This image is a sneaker display:
    #   "sneaker"  -> whole objects (~12)
    #   "sole"     -> a PART of each object (~11)
    #   "hand"     -> a different concept entirely, the people handling them (~9)
    # Swap in your own phrases and watch the presence token discriminate.
    prompts = ["sneaker", "sole", "hand"]

    for prompt in prompts:
        # reset_all_prompts clears the previous prompt but keeps the cached image.
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        n = len(state["scores"])
        print(f'prompt="{prompt}"  ->  {n} instance(s) above threshold')

        plot_results(Image.open(image_path), state)
        save_path = out(f"01_text_{prompt.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"    saved {save_path}")


if __name__ == "__main__":
    main()
