# Running the SAM 3 Agent (`examples/sam3_agent.ipynb`) — the full story

The agent uses a vision-LLM (served by **vLLM**) to turn a hard referring request
— *"the leftmost child wearing a blue vest"* — into concrete SAM 3 calls. The LLM
runs in a separate server; SAM 3 runs in the notebook. Both share this box's single
**RTX 3090 (24 GB)**.

Getting it working on one consumer GPU took several fixes. This documents all of
them so it's reproducible.

## ⚠️ Reality check: the agent needs a frontier model

The desktop-crash and config bugs below are **fixed and verified**. But there's a
hard limitation independent of those fixes: this agent's protocol (a ~66 KB system
prompt with multi-step reasoning + strict `<tool>`/`<verdict>` JSON) was designed
for a **frontier VLM** — the code's default model is
`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`.

An **8B model that fits on a single 24 GB 3090 cannot reliably drive it** in the
raw protocol (verified): `Qwen3-VL-8B-Instruct` **echoes the system prompt's
instruction list and never emits a `<tool>` call** (same at temp 0 and 0.7, even
with an explicit forcing directive); `Qwen3-VL-8B-Thinking` emits 18 K-char dumps
and often omits `<verdict>`.

### Structured decoding makes the local 8B *run* end-to-end (new)
We added **vLLM structured decoding** so the 8B is forced to emit a valid
`{reasoning, tool}` / `{reasoning, verdict}` JSON object, which the client converts
back into the `<tool>`/`<verdict>` string the agent parses. With this, the local 8B
now **runs the whole loop end-to-end without the "Invalid JSON in tool call" crash**
— e.g. `segment_phrase("child")` → 6 masks → `select_masks_and_return([1])` → a
clean single-mask PNG at 0.957 confidence. See the implementation:
[`client_llm.py`](../sam3/agent/client_llm.py) (`structured=`/`wrap=` args) and
[`agent_core.py`](../sam3/agent/agent_core.py) (`_tool_structured`, `_VERDICT_STRUCTURED`,
per-state allowed-tool schemas). vLLM **0.24** honors `{"structured_outputs": {"json": …}}`
(the legacy top-level `guided_json`/`guided_regex`/`guided_choice` are silently ignored).

### The wrong answers were the *harness*, not the 8B (verified)
At first the 8B picked the wrong instance ("leftmost child in a blue vest" → it chose
the right-most blue-vest child, or bailed with `report_no_mask`). It's tempting to
blame model size — **but that's wrong.** Handed the *same rendered image* with a
short, focused prompt ("for each numbered mask, note its vest colour and left-to-right
position, then pick the leftmost blue vest"), the 8B answers **correctly and
deterministically** (mask 4), even through the structured-output schema. What broke it
was the **66 KB agent protocol prompt**: it buries the actual visual question under
pages of tool-routing rules and asks the model to "choose a tool," and the small model
loses the plot. (Tellingly, the per-mask *verdict* step already sidesteps this with its
own small prompt — see `system_prompt_iterative_checking.txt`.)

### The general fix: a focused, query-agnostic selection step
Rather than trust the mask indices chosen under the giant prompt, the `select` step now
**re-asks the same model one focused question** — "which numbered mask(s) match this
query?" — using a small, generic prompt
([`system_prompts/system_prompt_selection.txt`](../sam3/agent/system_prompts/system_prompt_selection.txt))
with step-by-step structured reasoning. Nothing in it is specialised to any example;
the query is a passed-in variable and it names no specific object/attribute/relation.
Implementation: `_focused_select_masks` + `_SELECT_STRUCTURED` in
[`agent_core.py`](../sam3/agent/agent_core.py). Plus two general control-flow guards:
**Round 1 must segment** (you can't conclude "no mask" before asking SAM3), and once
candidates exist the model **can't lazily `report_no_mask`** (a true no-match still
emerges when the focused step returns an empty list). `temperature=0` throughout for
reproducibility.

### Run-to-run instability was RANDOM MASK COLORS (verified), not the LLM
Symptom: the same image+query gave different answers on different runs. Root cause
(proven): the mask visualizer picks overlay colors with **unseeded `np.random` /
`random`** (`helpers/color_map.py`, `helpers/visualizer.py`; there's a commented-out
`# np.random.seed(0)`). So every render tinted each object a different color, and since
the queries are color-sensitive ("blue vest"), the model's answer flipped. Proof: with
a *fixed* SAM3 output, re-rendering 5× (new random colors each time) flipped the answer
`[4],[4],[4],[],[]`; sending one *fixed* image 10× returned the same answer 10/10.

Fixes:
- **Seed the overlay colors** — `visualize()` in [`viz.py`](../sam3/agent/viz.py) now
  seeds both RNGs (save/restore) so identical masks render byte-identically.
- **Give the selection step the raw image too** and tell it the overlay colors are
  arbitrary (judge real colors from the raw pixels). See `_focused_select_masks` /
  `_SELECT_STRUCTURED` in [`agent_core.py`](../sam3/agent/agent_core.py) and
  [`system_prompt_selection.txt`](../sam3/agent/system_prompts/system_prompt_selection.txt).

### Keep the selection prompt MINIMAL (the mistake I made, and undid)
There is NO residual "8B can't compose 'leftmost X wearing Y'" limit — that earlier
conclusion was wrong. It came from an **over-engineered selection prompt**: a two-stage
schema (an intermediate `masks_matching_attributes` list) plus a heavy "superlative
scoping rule" plus a worked example. That extra machinery is exactly what pushed the 8B
into reading the query as *(leftmost) AND (blue)* → "the leftmost child (mask 2) is red
→ `[]`". Measured: with that prompt, `"the leftmost child wearing blue vest"` returned
`[]` 8/8.

The fix was to **delete the complexity**. A plain *"go through each numbered candidate,
note its query-relevant attributes and left-to-right position, then output the
number(s) the query refers to"* prompt with a simple `{reasoning, final_answer_masks}`
schema composes the query correctly and deterministically. Measured **8/8 → `[4]`**
across four prompt/image variants, and the full agent picks the correct child on every
run. Lesson: for a small VLM, a short direct question beats an elaborate rubric —
instruct less, not more.

**Result (verified, 3–10× each):** every query is now stable **and correct** —
`"the leftmost child wearing blue vest"` → the centre girl in the blue vest,
`"the rightmost child"` → the far-right boy, `"children wearing red vests"` → all three.
The local 8B is entirely capable of this class of referring expression; a frontier VLM
(`external_frontier_vlm`) is only needed for genuinely harder reasoning.

If you only need SAM 3's core capabilities (not the LLM agent), the
[study scripts](STUDY_PLAN.md) (`01`–`05`) run fully locally and cover text/visual
prompts, interactive clicks, batched inference, and video tracking.

---

## TL;DR — what to run

1. **Start the server** (downloads the Instruct model once, ~16 GB):
   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct \
     --tensor-parallel-size 1 \
     --quantization fp8 \
     --max-model-len 24576 \
     --gpu-memory-utilization 0.70 \
     --allowed-local-media-path / \
     --enforce-eager \
     --port 8002
   ```
   Wait for **`Application startup complete`**.
2. **Run the notebook** top-to-bottom. Cell 10 already points at
   `Qwen/Qwen3-VL-8B-Instruct` on `http://0.0.0.0:8002/v1`.

That's it. The sections below explain *why* each choice matters.

---

## The five problems that were fixed

### 1. `--tensor-parallel-size 4` → "World size (4) > available GPUs (1)"
You have one GPU. TP must be `1`.

### 2. bf16 8B won't fit beside SAM 3 → use `--quantization fp8`
`Qwen3-VL-8B` in bf16 is ~16 GB of weights; SAM 3 needs ~4–7 GB on the same card.
fp8 (~11 GB served; the vision tower stays bf16) leaves room. Runs on the 3090 via
Marlin.

### 3. **`max-model-len` too small → HTTP 400 → "Generated text is None"**
This was the real cause of the immediate failure. The agent's **system prompt is
~66 KB (~8 K tokens)**. With a small context window (e.g. `--max-model-len 12288`),
`prompt (~8 K) + requested output (4096)` exceeds the window, so vLLM returns
**HTTP 400**. The agent's LLM client swallows *any* exception and returns `None`
([`client_llm.py`](../sam3/agent/client_llm.py)), which surfaces as
*"Generated text is None"*. **Use `--max-model-len 24576`** (fits the 8 K system
prompt + up to 3 images + output; KV cache still fits at util 0.70).

### 4. **The 80 GB RAM crash was NOT this repo — it was a VS Code Server leak**
Earlier notes here blamed `client_llm.py` (a per-call `OpenAI()` client). **That was
wrong** and is corrected here. Measured root cause: **VS Code Server leaks orphaned
port-forwarding relay processes** — tiny `node -e "net.createConnection({port:NNNNN})
… pipe stdio↔socket"` helpers it spawns for remote port forwarding. They get
orphaned (PPID 0) and never reaped; **2,390 of them accumulated at ~40 MB each ≈
91 GB RSS**, which fills RAM and makes the OOM-killer reap Firefox / VS Code.

Evidence (measured on this box):
- A single real agent request grew `client_llm.py`'s own process by only **0.67 GB**
  (just the base64 image). The OpenAI SDK / `requests.post` path does **not** leak.
- At crash time: **1,970 total processes**, ~2,100 named `MainThread`, all the
  identical vscode-server `node` relay to `127.0.0.1:<forward-port>`.
- `pkill -9 -f "createConnection"` dropped procs **1,970 → 33** and freed **~50 GB**
  instantly (MemAvailable 14 GB → 64 GB, AnonPages 50 GB → 13 GB).

**Fix / prevention:**
- Reap the leak: `pkill -9 -f "createConnection"` (safe — they're orphaned relays;
  VS Code respawns legitimate ones on demand). Or run the watchdog in
  [`../study/reap_vscode_relays.sh`](../study/reap_vscode_relays.sh).
- Reduce spawning: in VS Code settings set `"remote.autoForwardPorts": false`, and
  reload the window periodically during long sessions.
- Cap the blast radius so a future leak can't take the desktop down:
  run the container with `--memory=48g --memory-swap=48g`.

(The `requests.post` rewrite in `client_llm.py` is retained — it's clean and
stateless — but it was **not** the fix; the SDK was never the cause.)

### 5. **Thinking model → slow + unparseable → use Instruct**
`Qwen3-VL-8B-Thinking` emits enormous free-form `<think>` reasoning. Measured: a
single mask-checking response was **18,828 characters** (~8 minutes at ~40 tok/s),
and it frequently did **not** contain the `<verdict>Accept/Reject</verdict>` tags
the agent's parser requires → `ValueError: Unexpected verdict`. The agent makes one
such call *per candidate mask per round*, so this compounds into very long,
often-failing runs. **`Qwen3-VL-8B-Instruct` follows the required format and is
much faster.** The notebook's cell 10 now selects it.

---

## Why the numbers (24 GB budget)

- vLLM Instruct fp8 + 24 K context at `--gpu-memory-utilization 0.70` ≈ **15 GB**,
  and it holds a KV cache of ~31 K tokens (enough to serve one 24 K-token request).
- That leaves ~7 GB for the notebook's SAM 3 (~3.7 GB resident, peaks under 7 GB).
- **Start the server first, then the notebook.** Don't run a second SAM 3 (e.g. a
  separate script) at the same time — three consumers overflow 24 GB.
- `--gpu-memory-utilization 0.55` fails a KV-cache check; **0.70 is the sweet spot**
  here. Going higher starves SAM 3.

## Health check (no notebook)

```bash
curl -s http://0.0.0.0:8002/v1/models | python -m json.tool     # server alive?
```

## If you ever see the RAM climb again

- Confirm you're on the **patched** `client_llm.py` (cached client), not an old copy.
- Make sure only **one** vLLM server is running (`pgrep -af "vllm serve"`), and only
  the notebook (not extra scripts) is holding SAM 3 on the GPU.
- The agent writes many small PNGs under `agent_output/` and `sam3_output/`; that's
  disk, not RAM, but you can prune it between runs.

## Faster / lighter option

If 8B is still heavier than you want, `Qwen/Qwen3-VL-4B-Instruct` (~8 GB, ~2× faster)
is a drop-in: change the model id in cell 10 and in `vllm serve`. It leaves much more
GPU headroom for SAM 3 and longer context.
