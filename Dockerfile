# Two-Tower Embeddings API Dockerfile
# Uses official NVIDIA PyTorch image for optimal GPU support

FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

# Default port (RunPod may override this)
ENV PORT=8000

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .

# Expose the API port
EXPOSE ${PORT}

CMD ["python3", "api.py"]
