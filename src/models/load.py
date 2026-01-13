# src/models/load.py
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_base(
    model_name: str = "roberta-base",
    device: str | None = None,
    model_path: str | None = None,
):
    """
    model_name: HF hub id (e.g. 'gpt2', 'roberta-base')
    model_path: local checkpoint directory in HF format; overrides model_name when provided
    """
    model_id = model_path or model_name

    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)

    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))
    model_type = str(getattr(cfg, "model_type", "")).lower()
    decoder_families = {
        "gpt2","gptj","gpt_neox","llama","mistral","falcon","opt","bloom",
        "qwen2","xglm","mpt","phi","gemma","mixtral"
    }
    is_decoder_only = (not is_encdec) and (
        bool(getattr(cfg, "is_decoder", False)) or model_type in decoder_families
    )

    if is_encdec:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif is_decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    else:
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager")

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.eval()

    if device is None:
        device = device_auto()
    model.to(device)
    return tok, model, device
