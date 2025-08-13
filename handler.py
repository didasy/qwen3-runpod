import os
import time
import json
import math
import runpod
import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# ----------------------------
# Environment & defaults
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-8B")
HF_TOKEN = os.getenv("HF_TOKEN")  # set as a Secret in RunPod if needed

# dtype heuristic
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
    else (torch.float16 if torch.cuda.is_available() else torch.float32)

# Faster matmul on GPU
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ----------------------------
# Load model once per pod
# ----------------------------
print(f"[boot] loading model: {MODEL_ID} dtype={DTYPE} device_map=auto")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    token=HF_TOKEN,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)
# pad token id fallback
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ----------------------------
# Helpers
# ----------------------------
class StringListStops(StoppingCriteria):
    """Stop on any of the provided strings (decoded)."""
    def __init__(self, stops: List[str], tokenizer: AutoTokenizer):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        # Heuristic: keep a rolling decode window
        self.window_tokens = 64

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check the last window for speed
        seq = input_ids[0].tolist()
        tail = seq[-self.window_tokens:]
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return any(s in text for s in self.stops)


def build_prompt_from_messages(
    messages: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
    user: Optional[str] = None
) -> str:
    """
    Accepts OpenAI-like messages or raw strings.
    Priority: messages -> (system,user) -> raw user prompt.
    """
    if messages and isinstance(messages, list):
        # Ensure roles are valid; fallback to 'user'
        norm = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "user"
            norm.append({"role": role, "content": content})
        return tokenizer.apply_chat_template(norm, tokenize=False, add_generation_prompt=True)

    # Build from system + user strings if given
    chat: List[Dict[str, str]] = []
    if system:
        chat.append({"role": "system", "content": system})
    if user:
        chat.append({"role": "user", "content": user})
    if chat:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Otherwise assume 'user' is the prompt
    return str(user or "")

def clamp(v, lo, hi, default):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return max(lo, min(hi, f))
    except Exception:
        return default

# ----------------------------
# Core generate
# ----------------------------
def generate_text(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input schema (all optional except one of: prompt|messages|user):
    {
      "prompt": "string",                // raw prompt (if you don't use messages)
      "messages": [{"role":"system|user|assistant","content":"..."}],
      "system": "optional system prompt",
      "user": "optional user message (used if no messages)",
      "max_new_tokens": 512,             // alias: max_tokens
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 50,
      "repetition_penalty": 1.05,
      "seed": 42,
      "stop": ["</s>", "User:"],        // list of stop strings
      "return_prompt": false             // echo the rendered prompt
    }
    """
    # Build prompt
    messages = inp.get("messages")
    prompt = inp.get("prompt")
    system = inp.get("system")
    user = inp.get("user")
    rendered = None

    if messages:
        rendered = build_prompt_from_messages(messages=messages)
    elif prompt is not None:
        # If a raw prompt string is provided, try to treat it as "user"
        rendered = build_prompt_from_messages(messages=None, system=system, user=prompt)
    else:
        rendered = build_prompt_from_messages(messages=None, system=system, user=user)

    if not isinstance(rendered, str) or len(rendered.strip()) == 0:
        return {"error": "Missing prompt/messages. Provide 'prompt' or 'messages' (or 'user')."}

    # Sampling params
    max_new_tokens = int(inp.get("max_new_tokens") or inp.get("max_tokens") or 512)
    max_new_tokens = max(1, min(max_new_tokens, 4096))

    temperature = clamp(inp.get("temperature", 0.7), 0.0, 2.0, 0.7)
    top_p = clamp(inp.get("top_p", 0.9), 0.0, 1.0, 0.9)
    top_k = int(inp.get("top_k", 50))
    repetition_penalty = clamp(inp.get("repetition_penalty", 1.0), 0.8, 2.0, 1.0)

    # Seed for reproducibility
    seed = inp.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            seed = None

    # Stops
    stop = inp.get("stop", None)
    stopping_criteria = StoppingCriteriaList()
    if isinstance(stop, list) and stop:
        stopping_criteria.append(StringListStops(stop, tokenizer))

    # Tokenize
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
    elapsed = round(time.time() - t0, 3)

    # Decode
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Try to remove the prompt portion to return only the generation
    # If apply_chat_template was used, the prompt is a prefix of full_text.
    # Fallback: return full_text if slicing fails.
    try:
        prefix_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        gen_text = full_text[len(prefix_text):].strip()
        if not gen_text:
            gen_text = full_text.strip()
    except Exception:
        gen_text = full_text.strip()

    # Post-stop trimming (if user passed stop strings and we didn't catch them inside generation)
    if isinstance(stop, list) and stop:
        for s in stop:
            idx = gen_text.find(s)
            if idx != -1:
                gen_text = gen_text[:idx].strip()
                break

    # Usage (best-effort; if you want exact, enable `return_dict_in_generate=True, output_scores=True`)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    total_tokens = int(output_ids.shape[1])
    completion_tokens = max(0, total_tokens - prompt_tokens)

    return {
        "text": gen_text,
        "finish_reason": "stop",   # best-effort
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "latency_sec": elapsed,
        "model": MODEL_ID,
        "returned_max_new_tokens": max_new_tokens,
        "returned_temperature": temperature,
        "returned_top_p": top_p,
        "returned_top_k": top_k,
        "returned_repetition_penalty": repetition_penalty,
        **({"rendered_prompt": rendered} if inp.get("return_prompt") else {})
    }

# ----------------------------
# RunPod handler
# ----------------------------
def handler(job):
    """
    RunPod invokes this with:
    {
      "input": {
        ... generation params as in generate_text() ...
      }
    }
    """
    try:
        params = job.get("input", {}) or {}
        result = generate_text(params)
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

# Start serverless worker
runpod.serverless.start({"handler": handler})
