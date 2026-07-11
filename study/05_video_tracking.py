"""
Study 05 — SAM 3.1 open-vocabulary VIDEO tracking (the "multiplex" model).

This is SAM 3's other headline: give a text concept on ONE frame and the model
detects every instance AND tracks each one through the whole video, assigning a
stable object id per instance. Under the hood it's a decoupled detector+tracker
with a shared "multiplex" memory (the SAM 3.1 speedup).

The API is request/response oriented (built for a server/demo):
    predictor = build_sam3_predictor(version="sam3.1")
    predictor.handle_request({"type": "start_session", "resource_path": <dir|mp4>})
    predictor.handle_request({"type": "add_prompt", "session_id": ..., "text": "person"})
    for out in predictor.handle_stream_request({"type": "propagate_in_video", ...}):
        out["outputs"]  # per-frame masks/boxes/ids, as numpy on CPU

Each per-frame output dict has:
    out_obj_ids     (N,)          stable integer id per tracked instance
    out_probs       (N,)          confidence
    out_boxes_xywh  (N, 4)        boxes, NORMALIZED xywh (top-left + size, [0,1])
    out_binary_masks (N, H, W)    boolean masks at original frame resolution

--- Hardware notes (important for the RTX 3090 / 24 GB in this container) ---
  * use_fa3=False: FlashAttention-3 isn't installed and doesn't support Ampere
    anyway. The default is True, so we MUST override it.
  * batched_grounding_batch_size: defaults to 16, which grounds 16 frames at once
    on the first propagate step and peaks ~22 GB -> OOM. We drop it to 4, which
    keeps the whole 270-frame run around ~14 GB.
  * offload_video_to_cpu=True: keeps the raw frame buffer off the GPU.

Output: study/outputs/05_tracking.gif plus a few key-frame PNGs.
"""

import glob
import os

import imageio.v2 as imageio
import numpy as np

from sam3.model_builder import build_sam3_predictor
from sam3.visualization_utils import load_frame, render_masklet_frame

from _common import ASSETS, out, setup_runtime

PROMPT = "person"
VIDEO_DIR = f"{ASSETS}/videos/0001"  # 270 JPEG frames of dancers


def propagate(predictor, session_id):
    """Run tracking over the whole video, returning {frame_idx: outputs}."""
    outputs = {}
    for resp in predictor.handle_stream_request(
        dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs[resp["frame_index"]] = resp["outputs"]
    return outputs


def save_gif(frame_paths, outputs, gif_path, stride=3):
    """Overlay masks on each frame and stitch an animated GIF (no ffmpeg needed)."""
    frames = []
    for fi in sorted(outputs.keys())[::stride]:
        img = load_frame(frame_paths[fi])  # HxWx3 uint8/float
        overlay = render_masklet_frame(img, outputs[fi], frame_idx=fi, alpha=0.5)
        frames.append(overlay)
    imageio.mimsave(gif_path, frames, fps=10, loop=0)
    print(f"    saved {gif_path} ({len(frames)} frames)")


def main():
    setup_runtime()

    # Ordered list of frame files: 0.jpg, 1.jpg, ... (numeric sort, not lexical).
    frame_paths = sorted(
        glob.glob(os.path.join(VIDEO_DIR, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    print(f"video: {len(frame_paths)} frames, prompt='{PROMPT}'")

    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False)
    predictor.model.batched_grounding_batch_size = 4  # 24 GB-friendly (see docstring)

    # --- Pass 1: text prompt -> detect + track everyone --------------------
    resp = predictor.handle_request(
        dict(type="start_session", resource_path=VIDEO_DIR, offload_video_to_cpu=True)
    )
    session_id = resp["session_id"]

    r0 = predictor.handle_request(
        dict(type="add_prompt", session_id=session_id, frame_index=0, text=PROMPT)
    )
    ids = list(r0["outputs"]["out_obj_ids"])
    print(f"frame 0: detected {len(ids)} instance(s), ids={ids}")

    outputs = propagate(predictor, session_id)
    save_gif(frame_paths, outputs, out("05_tracking.gif"))

    # A couple of key-frame stills so you can inspect ids/masks closely.
    for fi in [0, len(frame_paths) // 2, len(frame_paths) - 1]:
        img = load_frame(frame_paths[fi])
        overlay = render_masklet_frame(img, outputs[fi], frame_idx=fi)
        imageio.imwrite(out(f"05_frame_{fi:03d}.png"), overlay)

    # --- Pass 2: interactively remove one tracked object -------------------
    # This shows the editing side of the API: drop obj_id and re-propagate.
    if ids:
        drop = ids[0]
        predictor.handle_request(
            dict(type="remove_object", session_id=session_id, obj_id=int(drop))
        )
        outputs2 = propagate(predictor, session_id)
        save_gif(frame_paths, outputs2, out("05_tracking_removed_one.gif"))
        print(f"removed obj_id={drop} and re-propagated")

    predictor.handle_request(dict(type="close_session", session_id=session_id))
    print("done — open study/outputs/05_tracking.gif")


if __name__ == "__main__":
    main()
