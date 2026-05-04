"""LLM wrapper using HuggingFace transformers (lazy-loaded to save memory)."""

import logging
import time

from pydantic import BaseModel

from src.config import config

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def get_device() -> str:
    """Detect best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_llm():
    """Lazy-load the LLM and tokenizer (singleton)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    model_name = config.model.llm_model
    device = get_device()
    logger.info(f"Loading LLM: {model_name} on {device}")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load with appropriate precision
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        _model = _model.to(device)

    _model.eval()
    logger.info(f"LLM loaded successfully on {device}")
    return _tokenizer, _model


class LLMResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about League of Legends patch notes.
Use ONLY the provided context to answer. If the context doesn't contain enough information to answer, say "I don't have enough information from the patch notes to answer this question."
Be specific and cite patch versions when possible."""

RAG_PROMPT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Context from patch notes:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""


def generate_answer(question: str, context_chunks: list[dict]) -> LLMResponse:
    """
    Generate an answer using retrieved context chunks.

    Args:
        question: The user's question.
        context_chunks: List of dicts with 'content', 'patch_version', 'similarity'.

    Returns:
        LLMResponse with the generated answer and metrics.
    """
    tokenizer, model = load_llm()
    device = get_device()

    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        patch = chunk.get("patch_version", "unknown")
        context_parts.append(f"[Patch {patch}] {chunk['content']}")
    context_str = "\n\n".join(context_parts)

    # Build the full prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context_str,
        question=question,
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate
    import torch

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.model.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start_time

    # Decode only the generated portion
    generated_ids = outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return LLMResponse(
        answer=answer,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        latency_seconds=latency,
    )
