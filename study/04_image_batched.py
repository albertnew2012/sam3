"""
Study 04 — Batched, low-level image inference (multiple images x prompts).

Studies 01-03 use the friendly Sam3Processor. This one drops one level down to
show how the model is actually fed: you build "datapoints" (an image + a list of
queries), collate them into a batch, run a single model(batch) forward, and
post-process. This is the path you'd use to run SAM 3 efficiently over a dataset,
and it lets you mix text and visual prompts, positive and negative, in one shot.

Concepts:
  - Datapoint  = one image + N FindQueries (each query gets its own result).
  - FindQuery  = a text phrase and/or input boxes with per-box labels.
  - collate    = pad/stack datapoints into a batch the model accepts.
  - PostProcessImage = turn raw logits into masks/boxes/scores at original size.

We run two images in ONE batch:
  image A (test_image): text "person", text "shoe"
  image B (groceries):  text "bottle", and a NEGATIVE visual box to suppress a
                        region while keeping the text concept.

Each query's result is retrieved by the id returned when it was added.
Open study/outputs/04_*.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.visualization_utils import plot_results

from _common import ASSETS, out, setup_runtime

_COUNTER = [1]  # unique id generator shared by the helpers below


def new_datapoint():
    return Datapoint(find_queries=[], images=[])


def set_image(dp, pil_image):
    w, h = pil_image.size
    dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def _metadata(w, h):
    _COUNTER[0] += 1
    return InferenceMetadata(
        coco_image_id=_COUNTER[0],
        original_image_id=_COUNTER[0],
        original_category_id=1,
        original_size=[w, h],
        object_id=0,
        frame_index=0,
    )


def add_text_query(dp, text):
    """Add a text prompt; returns the id used to fetch its result later."""
    w, h = dp.images[0].size
    dp.find_queries.append(
        FindQueryLoaded(
            query_text=text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=_metadata(w, h),
        )
    )
    return _COUNTER[0]


def add_visual_query(dp, boxes, labels, text="visual"):
    """Add a box (exemplar) prompt. boxes: xyxy pixels; labels: True=pos/False=neg."""
    w, h = dp.images[0].size
    dp.find_queries.append(
        FindQueryLoaded(
            query_text=text,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1, 4),
            input_bbox_label=torch.tensor(labels, dtype=torch.bool).view(-1),
            inference_metadata=_metadata(w, h),
        )
    )
    return _COUNTER[0]


def main():
    device = setup_runtime()

    model = build_sam3_image_model()

    # The model expects a square 1008px, normalized input.
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    # --- image A: two prompt styles on the sneaker display ---
    imgA = Image.open(f"{ASSETS}/images/test_image.jpg").convert("RGB")
    dpA = new_datapoint()
    set_image(dpA, imgA)
    idA_shoe = add_text_query(dpA, "shoe")  # text -> ~12 shoes
    # visual prompt: a positive box (xyxy pixels) around ONE shoe as an exemplar.
    # The model generalizes it to find the other similar shoes.
    idA_exemplar = add_visual_query(dpA, boxes=[[480, 290, 590, 650]], labels=[True])
    dpA = transform(dpA)

    # --- image B: text prompt on a different image, same batch ---
    imgB = Image.open(f"{ASSETS}/images/groceries.jpg").convert("RGB")
    dpB = new_datapoint()
    set_image(dpB, imgB)
    idB_food = add_text_query(dpB, "food")
    dpB = transform(dpB)

    # --- one batched forward over BOTH images and ALL queries ---
    # The fused kernels require grad to be off, so run under inference_mode.
    batch = collate([dpA, dpB], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
    with torch.inference_mode():
        output = model(batch)
        results = postprocessor.process_results(output, batch.find_metadatas)

    for name, qid, img in [
        ("A_shoe_text", idA_shoe, imgA),
        ("A_shoe_exemplar", idA_exemplar, imgA),
        ("B_food_text", idB_food, imgB),
    ]:
        res = results[qid]
        print(f"{name}: {len(res['scores'])} detection(s)")
        plot_results(img, res)
        plt.savefig(out(f"04_{name}.png"), bbox_inches="tight", dpi=150)
        plt.close()
        print(f"    saved 04_{name}.png")


if __name__ == "__main__":
    main()
