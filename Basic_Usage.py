import torch
from contextlib import nullcontext
#################################### For Image ####################################
import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


def merge_prompt_outputs(processor, image, text_prompt):
    if isinstance(text_prompt, str):
        prompts = [text_prompt]
    elif isinstance(text_prompt, list):
        prompts = text_prompt
    else:
        raise TypeError("text_prompt must be a string or a list of strings")

    if len(prompts) == 0:
        raise ValueError("text_prompt list must not be empty")

    merged = {"masks": [], "boxes": [], "scores": []}
    prompt_labels = []

    for prompt in prompts:
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        merged["masks"].append(output["masks"])
        merged["boxes"].append(output["boxes"])
        merged["scores"].append(output["scores"])
        prompt_labels.extend([prompt] * len(output["scores"]))

    return {
        "masks": torch.cat(merged["masks"], dim=0),
        "boxes": torch.cat(merged["boxes"], dim=0),
        "scores": torch.cat(merged["scores"], dim=0),
        "labels": prompt_labels,
    }


# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("cars.jpg")
text_prompt = ["car", "pedestrian"]
prompt_slug = "_".join(text_prompt) if isinstance(text_prompt, list) else text_prompt
visualization_path = f"basic_usage_visualization_{prompt_slug}.png"
inference_context = (
    torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    if torch.cuda.is_available()
    else nullcontext()
)
with inference_context:
    output = merge_prompt_outputs(processor, image, text_prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
plot_results(image, output)
plt.savefig(visualization_path, bbox_inches="tight", dpi=200)
plt.close()
print(f"Prompt: {text_prompt}")
print(f"Detections: {len(scores)}")
print(f"Labels: {output['labels']}")
print(f"Saved visualization to {visualization_path}")

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
