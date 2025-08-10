# src/models/load.py
import torch
from transformers import AutoTokenizer, AutoModel

def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_base(model_name: str = "roberta-base", device: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.eval()
    if device is None:
        device = device_auto()
    model.to(device)
    return tok, model, device
