from __future__ import annotations


LLAMA31_INSTRUCT_MODELS = {
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
}


def resolve_llama31_instruct_model_id(size: str) -> str:
    try:
        return LLAMA31_INSTRUCT_MODELS[str(size).upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported Llama 3.1 Instruct size: {size}") from e
