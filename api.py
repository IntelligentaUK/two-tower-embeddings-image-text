"""
FastAPI service for two-tower text and image embeddings.

Uses:
- Image embeddings: google/siglip2-giant-opt-patch16-384 (1536 dims)
- Text embeddings: google/embeddinggemma-300m (768 dims)
"""

import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Server port (for RunPod compatibility)
PORT = int(os.environ.get("PORT", 8000))

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
class EmbedMode(str, Enum):
    """Mode for embedding generation."""
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    BOTH = "both"


class EmbedItem(BaseModel):
    """Single item for embedding generation."""
    id: str = Field(..., description="Unique identifier for this item")
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
    items: List[EmbedItem] = Field(
        ..., 
        description="List of items to generate embeddings for"
    )
    embed_mode: EmbedMode = Field(
        default=EmbedMode.BOTH,
        description="Which embeddings to generate: text_only, image_only, or both (default)"
    )


class EmbedResult(BaseModel):
    """Result for a single embedding request."""
    id: str = Field(..., description="Unique identifier for this item")
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
    
    logger.info("Loading image model: %s", IMAGE_CKPT)
    image_processor, image_model = load_image_model()
    logger.info("Image model loaded on device: %s", image_model.device)
    
    logger.info("Loading text model: %s", TEXT_CKPT)
    text_tokenizer, text_model = load_text_model()
    logger.info("Text model loaded on device: %s", text_model.device)
    
    http_client = httpx.AsyncClient()
    
    logger.info("Server ready on port %d", PORT)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    await http_client.aclose()


app = FastAPI(
    title="Two-Tower Embeddings API",
    description="Generate text and image embeddings using SigLIP2 and EmbeddingGemma models",
    version="1.0.1",
    lifespan=lifespan,
)

@app.get("/ping")
async def ping():
    """Simple ping endpoint - returns healthy only after models are loaded."""
    if image_model is None or text_model is None:
        return {"status": "loading"}
    return {"status": "healthy"}


@app.post("/embed", response_model=EmbedResponse, response_class=PrettyJSONResponse)
async def create_embeddings(request: EmbedRequest):
    """
    Generate embeddings for text and/or images.
    
    Accepts an array of items. Each item must contain:
    - id: Unique identifier for the item
    
    And at least one of:
    - image_url: URL to fetch the image from
    - image_base64: Base64 encoded image data
    - text_input: Text to generate embeddings for
    
    Returns normalized embedding vectors for each input with matching id.
    """
    # Generate request ID and start timing
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    items = request.items
    
    # Count request types
    num_images = sum(1 for item in items if item.image_url or item.image_base64)
    num_texts = sum(1 for item in items if item.text_input)
    
    logger.info(
        "[%s] Request received: %d item(s) - %d image(s), %d text(s), mode=%s",
        request_id, len(items), num_images, num_texts, request.embed_mode.value
    )
    
    results = []
    for idx, item in enumerate(items):
        image_emb = None
        image_dims = None
        text_emb = None
        text_dims = None
        
        # Process image if provided and mode allows
        should_process_image = request.embed_mode in (EmbedMode.IMAGE_ONLY, EmbedMode.BOTH)
        if should_process_image and (item.image_url or item.image_base64):
            source = "url" if item.image_url else "base64"
            logger.info("[%s] Processing image %d/%d (%s)", request_id, idx + 1, len(items), source)
            
            if item.image_url:
                image = await load_image_from_url(item.image_url)
            else:
                image = load_image_from_base64(item.image_base64)
            
            image_emb = compute_image_embedding(image)
            image_dims = len(image_emb)
        
        # Process text if provided and mode allows
        should_process_text = request.embed_mode in (EmbedMode.TEXT_ONLY, EmbedMode.BOTH)
        if should_process_text and item.text_input:
            text_preview = item.text_input[:50] + "..." if len(item.text_input) > 50 else item.text_input
            logger.info("[%s] Processing text %d/%d: '%s'", request_id, idx + 1, len(items), text_preview)
            
            text_emb = compute_text_embedding(item.text_input)
            text_dims = len(text_emb)
        
        result = EmbedResult(
            id=item.id,
            image_embedding=image_emb,
            image_dimensions=image_dims,
            text_embedding=text_emb,
            text_dimensions=text_dims,
        )
        results.append(result)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("[%s] Request completed in %.2fms", request_id, elapsed_ms)
    
    response = EmbedResponse(results=results)
    # Use model_dump to include None values explicitly
    return response.model_dump()

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port %d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

