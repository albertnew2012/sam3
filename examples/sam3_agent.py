# Copyright (c) Meta Platforms, Inc. and affiliates.

# # SAM 3 Agent
#
# Example of using an MLLM to drive SAM 3 as a tool ("SAM 3 Agent") so it can resolve
# complex referring expressions like "the leftmost child wearing blue vest". The LLM
# decides which concept to segment and which resulting mask(s) answer the query; SAM 3
# does the segmentation. See ``study/AGENT_NOTES.md`` for how it works and how it was
# tuned to run correctly on a single 24 GB GPU.
#
# Install `sam3` first: https://github.com/facebookresearch/sam3#installation

import os

import torch

# TensorFloat-32, bfloat16 autocast, and inference mode for the whole script (Ampere+;
# use float16 if your card lacks bf16). Entered globally and never exited.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

# Run from the repo root, derived from the installed package. Idempotent: safe to
# re-run without walking the working directory up a level (which would break the
# relative asset paths below).
import sam3

os.chdir(os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), "..")))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU is plenty for this demo
print("working dir:", os.getcwd())

# ## Build the SAM 3 image model
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.dirname(sam3.__file__)
model = build_sam3_image_model(bpe_path=f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz")
processor = Sam3Processor(model, confidence_threshold=0.5)

# ## Choose the agent LLM
#
# Either a local model served by vLLM, or any OpenAI-compatible external API. The local
# Qwen3-VL-8B fits on one 24 GB GPU and, together with the agent's structured-output and
# focused-selection steps, returns correct and reproducible results. Point at an external
# frontier VLM for harder reasoning, or if you'd rather not run a local server.
LLM_CONFIGS = {
    "qwen3_vl_8b_instruct": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "external_frontier_vlm": {
        "provider": "external",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # or gpt-4o, gemini, ...
        "base_url": "https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1",
    },
}

model_name = "qwen3_vl_8b_instruct"  # or "external_frontier_vlm"
LLM_API_KEY = "DUMMY_API_KEY"  # set a real key when using the external provider

llm_config = LLM_CONFIGS[model_name]
llm_config["api_key"] = LLM_API_KEY
llm_config["name"] = model_name
LLM_SERVER_URL = (
    "http://0.0.0.0:8002/v1"
    if llm_config["provider"] == "vllm"
    else llm_config["base_url"]
)

# ## (Local only) Start the vLLM server first, then run this script
#
# Skip if you use an external API. On a single 24 GB GPU, serve with TP=1 + fp8 and wait
# for "Application startup complete". The port must match LLM_SERVER_URL above.
#
#   vllm serve Qwen/Qwen3-VL-8B-Instruct \
#     --tensor-parallel-size 1 --quantization fp8 --max-model-len 24576 \
#     --gpu-memory-utilization 0.70 --allowed-local-media-path / \
#     --enforce-eager --port 8002

# ## Run inference on one image
from functools import partial

from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

image = os.path.abspath("assets/images/test_image.jpg")
prompt = "the leftmost child wearing blue vest"

send_generate_request = partial(
    send_generate_request_orig,
    server_url=LLM_SERVER_URL,
    model=llm_config["model"],
    api_key=llm_config["api_key"],
)
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

output_image_path = run_single_image_inference(
    image,
    prompt,
    llm_config,
    send_generate_request,
    call_sam_service,
    debug=True,
    output_dir="agent_output",
)
if output_image_path is not None:
    print("output image saved to:", output_image_path)
