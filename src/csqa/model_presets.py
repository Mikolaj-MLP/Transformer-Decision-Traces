from __future__ import annotations


QWEN25_INSTRUCT_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


LLAMA32_INSTRUCT_MODELS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
}


def resolve_qwen25_instruct_model_id(size: str) -> str:
    try:
        return QWEN25_INSTRUCT_MODELS[str(size).upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported Qwen 2.5 Instruct size: {size}") from e


def resolve_llama32_instruct_model_id(size: str) -> str:
    try:
        return LLAMA32_INSTRUCT_MODELS[str(size).upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported Llama 3.2 Instruct size: {size}") from e
