# SAM 3 — Codebase Structure & Architecture Map

A guide to *how the SAM 3 code is organized* and *how the model fits together*, so you
can navigate the repo and read the right file for any question. Everything here is
anchored to real files (links are clickable).

> Companion docs: [`STUDY_PLAN.md`](STUDY_PLAN.md) (runnable examples 01–05) and
> [`AGENT_NOTES.md`](AGENT_NOTES.md) (the LLM agent on top of SAM 3).

---

## 1. The big picture

**SAM 3 = one 848M-parameter model for promptable segmentation in images *and* video.**
Its headline capability over SAM 2 is **Promptable Concept Segmentation (PCS)**: give it
a short **text phrase** ("car") or **visual exemplars** (a box around an example) and it
**exhaustively finds every instance** of that concept — not just the one thing you
clicked. It still contains the classic SAM 1/2 click-to-segment behavior too.

Two architectural ideas make PCS work (from the [paper / README](../README.md)):

- **Presence token** — a dedicated token that decides *whether the concept is present at
  all*, sharpening discrimination between close phrases ("player in white" vs "player in
  red"). You see this when a plausible prompt correctly returns **0** detections.
- **Decoupled detector + tracker sharing one vision encoder** — the **detector** (a
  DETR-style model conditioned on text / geometry / image exemplars) finds instances in a
  frame; a separate **tracker** (SAM 2-style memory transformer) propagates them through
  video. SAM 3.1 adds a shared **"multiplex" memory** so many objects track jointly and fast.

```
                         ┌──────────────── shared vision encoder (ViTDet) ───────────────┐
   image / video frame ──►                                                                │
                         └──────────────┬───────────────────────────────┬────────────────┘
                                        │                               │
                    ┌───────────────────▼──────────────┐   ┌────────────▼─────────────┐
   text prompt ────►│  DETECTOR  (DETR-based)           │   │  TRACKER  (SAM 2-style)  │
   box/mask   ────►│  VL fusion → enc/dec → seg head    │   │  memory enc/dec, per-    │──► masks +
   exemplars  ────►│  + presence token + scoring        │──►│  object propagation      │   stable ids
                    └───────────────────────────────────┘   │  (+ multiplex in 3.1)    │   across frames
                                                             └──────────────────────────┘
```

---

## 2. Public entry points (start here)

Only **two** symbols are exported from the package ([`sam3/__init__.py`](../sam3/__init__.py)),
both defined in [`sam3/model_builder.py`](../sam3/model_builder.py):

| Function | Use it for | Returns |
|---|---|---|
| `build_sam3_image_model(bpe_path=…)` | **Images** — text/exemplar → all instances, and SAM-1 clicks | a `Sam3Image` model (wrap with `Sam3Processor`) |
| `build_sam3_predictor(version="sam3" \| "sam3.1", …)` | **Video** — detect + track a concept across frames | a predictor with a request/response API |

`model_builder.py` is the **single best file for architecture study**: it's a set of
`_create_*` factory functions that assemble every component, wired together by the
`build_*` entry points. Skim the `def`s to see the whole model as a parts list:

- `_create_vision_backbone` / `_create_vit_backbone` → the shared ViTDet encoder
- `_create_vl_backbone`, `_create_text_encoder` → language conditioning
- `_create_transformer_encoder` / `_create_transformer_decoder` → the DETR core
- `_create_geometry_encoder` → box/mask/exemplar prompts
- `_create_segmentation_head`, `_create_dot_product_scoring` → masks + presence/scoring
- `build_tracker`, `_create_multiplex_*` → the video tracker (and SAM 3.1 multiplex)

---

## 3. Repository map

```
sam3/
├── model_builder.py      ★ assembles every model; the build_* entry points live here
├── model/                the model itself — detector, tracker, video, multiplex
├── sam/                  classic SAM 1/2 interactive decoder (clicks/boxes → one mask)
├── agent/                LLM-agent layer that drives SAM 3 as a tool (see AGENT_NOTES.md)
├── eval/                 benchmark evaluators (COCO, LVIS, YTVIS, SA-CO)
├── train/                training loop, Hungarian matcher, losses
├── perflib/              fused CUDA/Triton kernels + Flash-Attention-3 wrappers
└── visualization_utils.py

assets/    test images, videos, tokenizer BPE vocab, model diagram
scripts/   eval + speed/qualitative entry points (measure_speed.py, qualitative_test.py)
study/     ← you are here: learning materials
```

---

## 4. `sam3/model/` — the model, file by file

### Shared vision + language front-end
| File | What it is |
|---|---|
| [`vitdet.py`](../sam3/model/vitdet.py) | the ViT / ViTDet **vision backbone** (shared by detector & tracker) |
| [`necks.py`](../sam3/model/necks.py) | feature-pyramid **necks** on top of the backbone |
| [`position_encoding.py`](../sam3/model/position_encoding.py) | sine / RoPE position encodings |
| [`text_encoder_ve.py`](../sam3/model/text_encoder_ve.py), [`tokenizer_ve.py`](../sam3/model/tokenizer_ve.py) | **text encoder** + BPE tokenizer (needs `bpe_path`) |
| [`vl_combiner.py`](../sam3/model/vl_combiner.py) | **vision-language fusion** (`SAM3VLBackbone`) |
| [`geometry_encoders.py`](../sam3/model/geometry_encoders.py) | encodes **box / mask / exemplar** prompts (`Prompt`, `MaskEncoder`, `SequenceGeometryEncoder`) |

### The image detector (DETR-based)
| File | What it is |
|---|---|
| [`sam3_image.py`](../sam3/model/sam3_image.py) | ★ **`Sam3Image`** — the detector. Key methods: `forward_grounding` (text/exemplar → instances), `forward_segmentation_from_state`, `predict_inst` (SAM-1 clicks) |
| [`sam3_image_processor.py`](../sam3/model/sam3_image_processor.py) | ★ **`Sam3Processor`** — the friendly image API (see §6) |
| [`encoder.py`](../sam3/model/encoder.py), [`decoder.py`](../sam3/model/decoder.py) | transformer **encoder/decoder** (fusion + query decoding) |
| [`maskformer_segmentation.py`](../sam3/model/maskformer_segmentation.py) | **segmentation head** (query → mask) |
| [`sam3_base_predictor.py`](../sam3/model/sam3_base_predictor.py) | request/response base (`start_session`, `add_prompt`, `propagate_in_video`, `remove_object`, `close_session`) |

### The video tracker (SAM 2 lineage)
| File | What it is |
|---|---|
| [`memory.py`](../sam3/model/memory.py) | the tracker's **memory** modules |
| [`sam3_tracker_base.py`](../sam3/model/sam3_tracker_base.py), [`sam3_tracker_utils.py`](../sam3/model/sam3_tracker_utils.py) | tracker core |
| [`sam3_video_base.py`](../sam3/model/sam3_video_base.py), [`sam3_video_predictor.py`](../sam3/model/sam3_video_predictor.py), [`sam3_video_inference.py`](../sam3/model/sam3_video_inference.py) | base video detect-and-track pipeline |
| [`sam3_tracking_predictor.py`](../sam3/model/sam3_tracking_predictor.py) | ties detector + tracker together |

### SAM 3.1 "multiplex" (shared-memory multi-object tracking)
| File | What it is |
|---|---|
| [`sam3_multiplex_tracking.py`](../sam3/model/sam3_multiplex_tracking.py) | ★ detect + track + **multiplex** logic (the memory knobs live here) |
| [`multiplex_mask_decoder.py`](../sam3/model/multiplex_mask_decoder.py), [`multiplex_utils.py`](../sam3/model/multiplex_utils.py) | multiplex decoder + helpers |
| [`sam3_multiplex_base.py`](../sam3/model/sam3_multiplex_base.py), [`sam3_multiplex_detector.py`](../sam3/model/sam3_multiplex_detector.py), [`sam3_multiplex_video_predictor.py`](../sam3/model/sam3_multiplex_video_predictor.py) | multiplex model + video predictor |

## 5. Other subpackages

- **[`sam3/sam/`](../sam3/sam/)** — the classic **SAM 1/2 interactive decoder**: [`prompt_encoder.py`](../sam3/sam/prompt_encoder.py) (points/boxes), [`mask_decoder.py`](../sam3/sam/mask_decoder.py), [`transformer.py`](../sam3/sam/transformer.py), [`rope.py`](../sam3/sam/rope.py). This is the "click → one mask" path (study 03).
- **[`sam3/perflib/`](../sam3/perflib/)** — performance kernels: [`fa3.py`](../sam3/perflib/fa3.py) (Flash-Attention-3 wrappers — the `use_fa3` flag), [`fused.py`](../sam3/perflib/fused.py) (fused linear+activation), [`nms.py`](../sam3/perflib/nms.py), [`iou.py`](../sam3/perflib/iou.py), [`connected_components.py`](../sam3/perflib/connected_components.py). On a consumer GPU (no FA-3) pass `use_fa3=False`.
- **[`sam3/eval/`](../sam3/eval/)** — offline evaluators for COCO/LVIS ([`coco_eval.py`](../sam3/eval/coco_eval.py)), YTVIS ([`ytvis_eval.py`](../sam3/eval/ytvis_eval.py)), and the **SA-CO** benchmark ([`saco_veval_eval.py`](../sam3/eval/saco_veval_eval.py)), plus `postprocessors.py`.
- **[`sam3/train/`](../sam3/train/)** — [`trainer.py`](../sam3/train/trainer.py), [`train.py`](../sam3/train/train.py), and the **`BinaryHungarianMatcherV2`** in [`matcher.py`](../sam3/train/matcher.py) (DETR-style bipartite matching). See [`../README_TRAIN.md`](../README_TRAIN.md).
- **[`sam3/agent/`](../sam3/agent/)** — an LLM agent that turns hard referring phrases into SAM 3 calls; documented separately in [`AGENT_NOTES.md`](AGENT_NOTES.md).

---

## 6. The two user-facing APIs

### Images — `Sam3Processor` ([`sam3_image_processor.py`](../sam3/model/sam3_image_processor.py))
Encode an image **once**, then fire many cheap prompts against it:
```python
processor = Sam3Processor(model, confidence_threshold=0.5)
state = processor.set_image(image)              # encode once  (also: set_image_batch)
processor.set_text_prompt("sneaker", state)     # open-vocabulary text → all instances
processor.add_geometric_prompt(box, label=True, state=state)   # +/- visual exemplars
processor.set_confidence_threshold(0.5, state)  # tune recall/precision
processor.reset_all_prompts(state)
```

### Video — predictor request/response ([`build_sam3_predictor`](../sam3/model_builder.py))
A stateful session API (`version="sam3.1"` = multiplex, `"sam3"` = base):
```python
predictor = build_sam3_predictor(version="sam3.1")
r = predictor.handle_request({"type": "start_session", "resource_path": video_dir})
sid = r["session_id"]
predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": "person"})
for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
    masks = out["out_binary_masks"]             # per-frame masks, each with a stable obj id
# also: remove_object, reset_session, close_session
```

---

## 7. Suggested reading path

1. **[`model_builder.py`](../sam3/model_builder.py)** — read the `_create_*` factories top-to-bottom to see the whole model as a parts list, then `build_sam3_image_model`.
2. **[`sam3_image.py`](../sam3/model/sam3_image.py)** `forward_grounding` — how text/exemplar prompts become instance masks (the detector's core).
3. **[`vl_combiner.py`](../sam3/model/vl_combiner.py)** + **[`geometry_encoders.py`](../sam3/model/geometry_encoders.py)** — how language and geometry condition the detector.
4. **[`sam3_multiplex_tracking.py`](../sam3/model/sam3_multiplex_tracking.py)** — how detection is propagated + tracked across video (the SAM 3.1 path).
5. Run the **[study scripts 01–05](STUDY_PLAN.md)** alongside, editing prompts and watching outputs.

> **Params/perf:** 848M total (README §Model). On a 24 GB GPU, pass `use_fa3=False`
> (no Flash-Attention-3 on Ampere) and cap video memory — see the hardware notes in
> [`STUDY_PLAN.md`](STUDY_PLAN.md#3-hardware-notes-for-this-box-rtx-3090-24-gb--read-before-video).
