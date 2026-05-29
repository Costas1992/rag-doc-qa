import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # see note below

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import retrieve, format_context

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_model = None
_tokenizer = None


def get_device() -> str:
    """Pick the fastest backend available: Apple MPS, CUDA GPU, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_llm():
    """Load the LLM and tokenizer once, on the fastest device, then reuse."""
    global _model, _tokenizer
    if _model is None:
        device = get_device()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )
        _model.to(device)        # move the ~3 GB of weights onto MPS/GPU/CPU
        _model.eval()
        print(f"[generator] model loaded on: {device}")
    return _model, _tokenizer


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "sources provided. If the answer is not in the sources, say you do not have "
    "that information. Keep the answer concise and factual."
)


def generate_answer(question: str, context: str, max_new_tokens: int = 256) -> str:
    """Feed the question + retrieved context to the LLM, return its answer."""
    model, tokenizer = get_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)   # match model's device

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    question = "How many vacation days do employees get?"

    chunks = retrieve(question, k=3)
    context = format_context(chunks)

    print(f"Question: {question}\n")
    answer = generate_answer(question, context)
    print("Answer:\n" + answer)