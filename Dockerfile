# Two-Tower Embeddings API Dockerfile
# Uses official NVIDIA PyTorch image for optimal GPU support

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set Python to not buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Default port (RunPod may override this)
ENV PORT=8000

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install additional Python dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers accelerate pillow fastapi "uvicorn[standard]" httpx

# Copy application code
COPY api.py .
COPY image-text-two-tower-embeddings.py .

# Create cache directory for Hugging Face models
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

# Expose the API port
EXPOSE ${PORT}

# Health check using PORT variable
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the API server with dynamic port
CMD python -m uvicorn api:app --host 0.0.0.0 --port ${PORT}
