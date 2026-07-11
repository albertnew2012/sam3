# SAM 3 — Study Plan & Runnable Examples

A guided path to understanding **SAM 3 (Segment Anything with Concepts)** on this
machine, with five progressively deeper example scripts that all run on the local
RTX 3090. Everything here has been executed end-to-end and produces real outputs
in [`outputs/`](outputs/).

> **TL;DR** — Open the *Run and Debug* panel in VSCode and pick a
> `SAM3 · NN …` config (they're in [`.vscode/launch.json`](../.vscode/launch.json)),
> or run e.g. `python study/01_image_text_prompt.py`. Results land in `study/outputs/`.

---

## 1. What makes SAM 3 different (the mental model)

| Model | Prompt | What you get back |
|-------|--------|-------------------|
| SAM 1 | a click / box on **one** object | that **one** object's mask |
| SAM 2 | a click on one object, in video | that one object, **tracked** across frames |
| **SAM 3** | a **text phrase** or **visual exemplar** of a *concept* | **every instance** of that concept, segmented (and in video, each one tracked with a stable id) |

The leap is **Promptable Concept Segmentation (PCS)**: instead of "segment this
thing I clicked," you say *"car"* and it exhaustively finds all cars. Two
architectural ideas make this work (see the paper / [`../README.md`](../README.md)):

- **Presence token** — a dedicated token that decides *whether the concept is
  present at all*, which sharpens discrimination between close phrases
  (*"player in white"* vs *"player in red"*). You'll see this when a plausible
  prompt correctly returns **0** detections.
- **Decoupled detector + tracker** — the image detector finds instances; a
  separate tracker propagates them through video. SAM 3.1 adds a shared
  **"multiplex" memory** so many objects are tracked jointly and fast.

SAM 3 still *contains* the SAM 1/2 behavior (point/box → one object) — that's
study 03.

---

## 2. Suggested order

Work through the scripts in numeric order. Each is self-contained, heavily
commented, and writes a labeled image/GIF you should open and inspect.

### `01_image_text_prompt.py` — open-vocabulary text → masks  ★ start here
The headline feature. One image is encoded once (`set_image`), then several text
prompts are fired against it cheaply. On the sneaker-display test image you get
`sneaker`→~12, `sole`→~11 (a *part*), `hand`→~9 (a *different* concept).
**Concepts:** `build_sam3_image_model`, `Sam3Processor.set_image` /
`set_text_prompt`, confidence threshold, concept granularity.

### `02_image_visual_prompt.py` — visual (box) exemplars, positive & negative
When a concept is hard to name, point at an example with a box. The model finds
things that *resemble* your positive box and *avoids* your negative box. Compare
`02_visual_positive_only.png` vs `02_visual_pos_and_neg.png`.
**Concepts:** `add_geometric_prompt`, normalized `cxcywh` boxes, exemplar-based
prompting, negatives for suppression.

### `03_sam1_interactive.py` — the classic SAM 1 click workflow
Turn on `enable_inst_interactivity=True` and use `model.predict_inst()` with
point/box clicks to segment **one** object at a time — with `multimask_output`
to resolve ambiguity, and iterative +/- point refinement.
**Concepts:** interactive decoder, foreground/background points, mask-logit
feedback, box prompts. This is where SAM 3 ⊇ SAM 1.

### `04_image_batched.py` — the low-level, efficient path
Drops below `Sam3Processor` to show how the model is actually fed: build
`Datapoint`s (image + queries), `collate` into a batch, one `model(batch)`
forward over **multiple images × multiple prompts**, then `PostProcessImage`.
Mixes a text prompt and a visual-exemplar prompt in the same batch.
**Concepts:** `Datapoint` / `FindQueryLoaded`, collation, `PostProcessImage`,
running under `torch.inference_mode()` (the fused kernels require grad off).
This is the pattern to use for evaluating over a dataset.

### `05_video_tracking.py` — SAM 3.1 open-vocabulary video tracking  ★ the big one
Give the concept `"person"` on frame 0 and the model detects + tracks everyone
across 270 frames, each with a stable `obj_id`. Then it demonstrates the *editing*
side of the API by removing one tracked object and re-propagating. Produces
`05_tracking.gif`, `05_tracking_removed_one.gif`, and key-frame stills.
**Concepts:** `build_sam3_predictor(version="sam3.1")`, the request/response API
(`start_session` → `add_prompt` → `propagate_in_video` → `remove_object` →
`close_session`), stable ids, per-frame numpy outputs.

---

## 3. Hardware notes for THIS box (RTX 3090, 24 GB) — read before video

The image scripts (01–04) run comfortably (~6–8 GB). The **video** model is
memory-hungry and the library defaults assume a bigger datacenter GPU. Three
settings (already baked into `05_video_tracking.py`) make it fit in 24 GB:

1. **`use_fa3=False`** — FlashAttention-3 isn't installed and doesn't support
   Ampere anyway. The predictor defaults to `True`, so this override is required
   or you get `ModuleNotFoundError: flash_attn_interface`.
2. **`predictor.model.batched_grounding_batch_size = 4`** — the default `16`
   grounds 16 frames at once on the first propagate step and spikes to ~22 GB →
   OOM. Dropping to 4 keeps the full run around **~14 GB**.
3. **`offload_video_to_cpu=True`** in `start_session` — keeps the raw frame
   buffer off the GPU.

Also set process-wide (in `_common.py` and the launch configs):
`PYTORCH_ALLOC_CONF=expandable_segments:True` to curb fragmentation.

> If you point the video script at a **much longer** video and still OOM, cap it
> with `max_frame_num_to_track=N` in the `propagate_in_video` request — GPU memory
> grows with the number of frames because backbone features are cached per frame.

---

## 4. Environment (already set up in this container)

- Python 3.12, PyTorch 2.9.1+cu129, CUDA available on an RTX 3090.
- `pip install -e .` (editable) plus example deps.
- **Pinned versions that matter:** `numpy<2` (the repo requires it), so we use
  `opencv-python-headless<4.12`, `scikit-image 0.24`, and `scikit-learn`.
  Installing a numpy≥2 build of OpenCV will silently break `import sam3`.
- Checkpoints auto-download from Hugging Face on first use and cache under
  `~/.cache/huggingface/hub`: `facebook/sam3` (image + base video, ~3.4 GB) and
  `facebook/sam3.1` (multiplex video, ~3 GB).

---

## 5. Where to go next in the codebase

- [`../sam3/model_builder.py`](../sam3/model_builder.py) — how every model is
  assembled (backbone, transformer, geometry encoder, segmentation head, tracker,
  multiplex). The single best file for architecture study.
- [`../sam3/model/sam3_image_processor.py`](../sam3/model/sam3_image_processor.py)
  — the friendly image API used by studies 01–03.
- [`../sam3/model/sam3_multiplex_tracking.py`](../sam3/model/sam3_multiplex_tracking.py)
  — SAM 3.1 detect+track+multiplex logic (where the memory knobs live).
- [`../examples/`](../examples/) — the official notebooks these scripts are
  distilled from (including an interactive ipywidgets segmentation UI).
- [`AGENT_NOTES.md`](AGENT_NOTES.md) — running [`../examples/sam3_agent.ipynb`](../examples/sam3_agent.ipynb),
  the LLM agent that turns referring phrases ("the leftmost child in a blue vest")
  into SAM 3 calls. Includes the **single-GPU vLLM command** (the notebook's default
  `--tensor-parallel-size 4` assumes 4 GPUs; you have 1).
- [`../scripts/`](../scripts/) — evaluation and speed-measurement entry points.
- [`../README.md`](../README.md) / [`../RELEASE_SAM3p1.md`](../RELEASE_SAM3p1.md)
  / [`../README_TRAIN.md`](../README_TRAIN.md) — model overview and training.
