"""
FastAPI service for two-tower text and image embeddings.

Uses:
- Image embeddings: google/siglip2-giant-opt-patch16-384 (1536 dims)
- Text embeddings: google/embeddinggemma-300m (768 dims)
"""

import base64
import io
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import httpx
import torch
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

# Model checkpoints
IMAGE_CKPT = "google/siglip2-giant-opt-patch16-384"
TEXT_CKPT = "google/embeddinggemma-300m"

# Global model instances
image_processor: Optional[ProcessorMixin] = None
image_model: Optional[PreTrainedModel] = None
text_tokenizer: Optional[AutoTokenizer] = None
text_model: Optional[PreTrainedModel] = None
http_client: Optional[httpx.AsyncClient] = None


def load_image_model() -> tuple[ProcessorMixin, PreTrainedModel]:
    """Load the SigLIP2 image embedding model."""
    processor = AutoProcessor.from_pretrained(IMAGE_CKPT)
    try:
        model = AutoModel.from_pretrained(IMAGE_CKPT, device_map="cuda")
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(IMAGE_CKPT)
    model.eval()
    return processor, model


def load_text_model() -> tuple[AutoTokenizer, PreTrainedModel]:
    """Load the EmbeddingGemma text embedding model."""
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CKPT)
    try:
        model = AutoModel.from_pretrained(TEXT_CKPT, device_map="cuda")
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(TEXT_CKPT)
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for text embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * mask).sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def compute_image_embedding(image: Image.Image) -> List[float]:
    """Generate a normalized image embedding vector."""
    inputs = image_processor(images=[image], return_tensors="pt").to(image_model.device)
    with torch.no_grad():
        image_features = image_model.get_image_features(**inputs)
    normalized_vector = torch.nn.functional.normalize(image_features, p=2, dim=-1).squeeze(0).cpu()
    return normalized_vector.tolist()


def compute_text_embedding(text: str) -> List[float]:
    """Generate a normalized text embedding vector."""
    text_prompt = f"task: classification | query: {text}"
    inputs = text_tokenizer(text_prompt, return_tensors="pt").to(text_model.device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    normalized_vector = torch.nn.functional.normalize(pooled, p=2, dim=-1).squeeze(0).cpu()
    return normalized_vector.tolist()


async def load_image_from_url(url: str) -> Image.Image:
    """Fetch an image from a URL."""
    try:
        response = await http_client.get(url, timeout=30.0)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image from URL: {str(e)}")


def load_image_from_base64(base64_string: str) -> Image.Image:
    """Decode a base64 string to an image."""
    try:
        # Handle data URI scheme if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")


# Pydantic models
class EmbedItem(BaseModel):
    """Single item for embedding generation."""
    image_url: Optional[str] = Field(default=None, description="URL of the image to embed")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data")
    text_input: Optional[str] = Field(default=None, description="Text to embed")

    @model_validator(mode="after")
    def validate_has_input(self):
        if not self.image_url and not self.image_base64 and not self.text_input:
            raise ValueError("At least one of image_url, image_base64, or text_input must be provided")
        if self.image_url and self.image_base64:
            raise ValueError("Cannot provide both image_url and image_base64")
        return self


class EmbedRequest(BaseModel):
    """Request body for embedding endpoint."""
    items: Union[EmbedItem, List[EmbedItem]] = Field(
        ..., 
        description="Single item or list of items to generate embeddings for"
    )


class EmbedResult(BaseModel):
    """Result for a single embedding request."""
    image_embedding: Optional[List[float]] = Field(
        default=None, 
        description="Normalized image embedding vector (1536 dimensions)"
    )
    text_embedding: Optional[List[float]] = Field(
        default=None, 
        description="Normalized text embedding vector (768 dimensions)"
    )
    image_dimensions: Optional[int] = Field(default=None, description="Image embedding dimensions")
    text_dimensions: Optional[int] = Field(default=None, description="Text embedding dimensions")


class EmbedResponse(BaseModel):
    """Response body for embedding endpoint."""
    results: List[EmbedResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    image_model: str
    text_model: str
    device: str


class PrettyJSONResponse(Response):
    """Custom response class that pretty-prints JSON."""
    media_type = "application/json"

    def render(self, content) -> bytes:
        return json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    global image_processor, image_model, text_tokenizer, text_model, http_client
    
    print("Loading image model...")
    image_processor, image_model = load_image_model()
    print(f"Image model loaded on device: {image_model.device}")
    
    print("Loading text model...")
    text_tokenizer, text_model = load_text_model()
    print(f"Text model loaded on device: {text_model.device}")
    
    http_client = httpx.AsyncClient()
    
    yield
    
    # Cleanup
    await http_client.aclose()


app = FastAPI(
    title="Two-Tower Embeddings API",
    description="Generate text and image embeddings using SigLIP2 and EmbeddingGemma models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, response_class=PrettyJSONResponse)
async def health_check():
    """Check the health status of the API and loaded models."""
    device = str(image_model.device) if image_model else "not loaded"
    return HealthResponse(
        status="healthy",
        image_model=IMAGE_CKPT,
        text_model=TEXT_CKPT,
        device=device,
    ).model_dump()


@app.post("/embed", response_model=EmbedResponse, response_class=PrettyJSONResponse)
async def create_embeddings(request: EmbedRequest):
    """
    Generate embeddings for text and/or images.
    
    Accepts either a single item or an array of items. Each item can contain:
    - image_url: URL to fetch the image from
    - image_base64: Base64 encoded image data
    - text_input: Text to generate embeddings for
    
    Returns normalized embedding vectors for each input.
    """
    # Normalize input to list
    items = request.items if isinstance(request.items, list) else [request.items]
    
    results = []
    for item in items:
        image_emb = None
        image_dims = None
        text_emb = None
        text_dims = None
        
        # Process image if provided
        if item.image_url or item.image_base64:
            if item.image_url:
                image = await load_image_from_url(item.image_url)
            else:
                image = load_image_from_base64(item.image_base64)
            
            image_emb = compute_image_embedding(image)
            image_dims = len(image_emb)
            print(f"[DEBUG] Computed IMAGE embedding with {image_dims} dimensions")
        
        # Process text if provided
        if item.text_input:
            text_emb = compute_text_embedding(item.text_input)
            text_dims = len(text_emb)
            print(f"[DEBUG] Computed TEXT embedding with {text_dims} dimensions")
        
        result = EmbedResult(
            image_embedding=image_emb,
            image_dimensions=image_dims,
            text_embedding=text_emb,
            text_dimensions=text_dims,
        )
        results.append(result)
    
    response = EmbedResponse(results=results)
    # Use model_dump to include None values explicitly
    return response.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

