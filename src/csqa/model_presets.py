from __future__ import annotations


QWEN25_INSTRUCT_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


LLAMA31_INSTRUCT_MODELS = {
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
}


def resolve_qwen25_instruct_model_id(size: str) -> str:
    try:
        return QWEN25_INSTRUCT_MODELS[str(size).upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported Qwen 2.5 Instruct size: {size}") from e


def resolve_llama31_instruct_model_id(size: str) -> str:
    try:
        return LLAMA31_INSTRUCT_MODELS[str(size).upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported Llama 3.1 Instruct size: {size}") from e
