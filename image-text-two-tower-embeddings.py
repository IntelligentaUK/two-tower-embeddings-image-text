# 

import json
import time
from pprint import pprint
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

# CKPT_old = "google/siglip2-so400m-patch16-512"
CKPT = "google/siglip2-giant-opt-patch16-384"
TEXT_CKPT = "google/embeddinggemma-300m"

text_input = "a photo of a cat sitting on the ground surrounded by fallen leaves, playing with a ladybird"

def load_local_image(path: str) -> Tuple[Image.Image, float]:
    """Load an image from disk and return it along with the elapsed load time in ms."""
    start = time.perf_counter()
    image = Image.open(path)
    load_ms = (time.perf_counter() - start) * 1000.0
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image, load_ms


def load_model() -> Tuple[ProcessorMixin, PreTrainedModel]:
    processor = AutoProcessor.from_pretrained(CKPT)
    try:
        model = AutoModel.from_pretrained(CKPT, device_map="cuda")
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(CKPT)
    model.eval()
    return processor, model
def load_text_model() -> Tuple[AutoTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CKPT)
    try:
        model = AutoModel.from_pretrained(TEXT_CKPT, device_map="cuda")
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(TEXT_CKPT)
    model.eval()
    return tokenizer, model


def compute_image_embedding(
    image: Image.Image, processor: ProcessorMixin, model: PreTrainedModel
) -> Tuple[List[float], List[float], float]:
    """Generate a normalized embedding vector along with generation time in ms."""
    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    embedding_ms = (time.perf_counter() - start) * 1000.0
    raw_vector = image_features.squeeze(0).cpu()
    normalized_vector = torch.nn.functional.normalize(image_features, p=2, dim=-1).squeeze(0).cpu()
    return raw_vector.tolist(), normalized_vector.tolist(), embedding_ms


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * mask).sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def compute_text_embedding(
    inputs: BatchEncoding, model: PreTrainedModel
) -> Tuple[List[float], List[float], float]:
    """Generate raw and normalized text embeddings along with generation time in ms."""
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    embedding_ms = (time.perf_counter() - start) * 1000.0
    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    raw_vector = pooled.squeeze(0).cpu()
    normalized_vector = torch.nn.functional.normalize(pooled, p=2, dim=-1).squeeze(0).cpu()
    return raw_vector.tolist(), normalized_vector.tolist(), embedding_ms

def summarize_timings(times: List[float]) -> dict:
    steady_state = times[1:] if len(times) > 1 else times
    steady_avg = sum(steady_state) / len(steady_state)
    return {
        "first_ms": times[0],
        "last_ms": times[-1],
        "avg_ms": sum(times) / len(times),
        "steady_state_avg_ms": steady_avg,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _value_size_bytes(model: PreTrainedModel) -> int:
    """Return the byte width of a single embedding value for the given model."""
    dtype = getattr(model, "dtype", torch.float32)
    try:
        return torch.zeros((), dtype=dtype).element_size()
    except (RuntimeError, TypeError):
        return torch.finfo(torch.float32).bits // 8


def main():
    image_path = "cat.jpg"

    tokenizer, text_model = load_text_model()
    text_prompt = f"task: classification | query: {text_input}"
    text_inputs = tokenizer(text_prompt, return_tensors="pt").to(text_model.device)

    text_runs = []
    raw_text_embedding: List[float] = []
    normalized_text_embedding: List[float] = []
   
    raw_text_embedding, normalized_text_embedding, embedding_ms = compute_text_embedding(text_inputs, text_model)

    image, download_ms = load_local_image(image_path)
    processor, model = load_model()
    
    raw_embedding: List[float] = []
    normalized_embedding: List[float] = []
    raw_embedding, normalized_embedding, embedding_ms = compute_image_embedding(image, processor, model)

    text_value_size_bytes = _value_size_bytes(text_model)
    image_value_size_bytes = _value_size_bytes(model)
    text_vector_dimensions = len(raw_text_embedding)
    vector_dimensions = len(raw_embedding)
    text_vector_size_bytes = text_vector_dimensions * text_value_size_bytes
    vector_size_bytes = vector_dimensions * image_value_size_bytes

    pprint(
        {
            "text_model": TEXT_CKPT,
            "text_prompt": text_prompt,
            "text_vector_dimensions": text_vector_dimensions,
            "text_vector_size_bytes": text_vector_size_bytes,
            "image_model": CKPT,
            "download_ms": round(download_ms, 2),
            "vector_dimensions": vector_dimensions,
            "vector_size_bytes": vector_size_bytes,
        }
    )

    result = {
        "text": {
            "model": TEXT_CKPT,
            "prompt": text_prompt,            
            "raw_embedding": raw_text_embedding,
            "normalized_embedding": normalized_text_embedding,
            "vector_dimensions": text_vector_dimensions,
            "vector_size_bytes": text_vector_size_bytes,
        },
        "image": {
            "model": CKPT,
            "image_path": image_path,
            "download_ms": download_ms,
            "raw_embedding": raw_embedding,
            "normalized_embedding": normalized_embedding,
            "vector_dimensions": vector_dimensions,
            "vector_size_bytes": vector_size_bytes,
        },
    }

    with open("image-text-embeddings-output.json", "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=2)

    print(
        "Vector storage: text={} dims (~{} bytes), image={} dims (~{} bytes)".format(
            text_vector_dimensions,
            text_vector_size_bytes,
            vector_dimensions,
            vector_size_bytes,
        )
    )

if __name__ == "__main__":
    main()
