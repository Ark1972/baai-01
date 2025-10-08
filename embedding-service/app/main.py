from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
import os
import sys
from contextlib import asynccontextmanager
from FlagEmbedding import BGEM3FlagModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global embedding model
embedding_model = None

class EmbeddingModel:
    """Wrapper for FlagEmbedding BGEM3FlagModel"""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.device = "cuda" if use_fp16 else "cpu"
        logger.info(f"Model loaded successfully on device: {self.device}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (1024-dimensional)
        """
        result = self.model.encode(texts, max_length=8192)
        embeddings = result['dense_vecs'].tolist()
        return embeddings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global embedding_model

    # Load model on startup
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-m3")
    use_fp16 = os.getenv("USE_FP16", "false").lower() == "true"

    logger.info(f"Initializing embedding model: {model_name}")
    embedding_model = EmbeddingModel(model_name=model_name, use_fp16=use_fp16)
    logger.info("Embedding model ready")

    yield

    # Cleanup
    logger.info("Shutting down embedding model")
    embedding_model = None

# Initialize FastAPI app
app = FastAPI(
    title="BAAI BGE-M3 Embedding Service",
    description="Web service for generating text embeddings using BAAI/bge-m3",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)

    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text or text.isspace():
                raise ValueError('Texts cannot be empty or only whitespace')
            if len(text) > 50000:
                raise ValueError('Each text must be less than 50000 characters')
        return v

class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Dimensionality of embeddings (1024)")
    texts_count: int = Field(..., description="Number of texts processed")
    model: str = Field(..., description="Model name used")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    version: str = Field(..., description="API version")
    dimensions: int = Field(..., description="Embedding dimensions")

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "BAAI BGE-M3 Embedding Service",
        "model": os.getenv("MODEL_NAME", "BAAI/bge-m3"),
        "version": "1.0.0",
        "backend": "FlagEmbedding",
        "dimensions": 1024,
        "max_input_length": 8192,
        "endpoints": {
            "/": "Service information",
            "/health": "Health check endpoint",
            "/embed": "Generate embeddings for texts",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for model status"""
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not initialized"
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=embedding_model.model_name,
        device=embedding_model.device,
        version="1.0.0",
        dimensions=1024
    )

@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def generate_embeddings(request: EmbedRequest):
    """Generate embeddings for input texts"""
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not available"
        )

    try:
        embeddings = embedding_model.encode(request.texts)

        return EmbedResponse(
            embeddings=embeddings,
            dimensions=1024,
            texts_count=len(request.texts),
            model=embedding_model.model_name
        )
    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting BGE-M3 embedding service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
