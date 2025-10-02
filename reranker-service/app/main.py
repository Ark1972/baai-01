from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Union, Optional
from FlagEmbedding import FlagReranker
import uvicorn
import logging
import os
import sys
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global reranker instance
reranker_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown"""
    global reranker_model
    try:
        model_name = os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        use_fp16 = os.getenv("USE_FP16", "true").lower() == "true"
        cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"FP16 enabled: {use_fp16}")
        logger.info(f"Cache directory: {cache_dir}")
        
        reranker_model = FlagReranker(
            model_name, 
            use_fp16=use_fp16,
            cache_dir=cache_dir
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down and cleaning up resources")
    reranker_model = None

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BAAI Reranker Service",
    description="Web service for reranking text pairs using BAAI/bge-reranker-v2-m3 model",
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
class TextPair(BaseModel):
    query: str = Field(..., description="The query text", min_length=1, max_length=10000)
    passage: str = Field(..., description="The passage text to rank against the query", min_length=1, max_length=10000)
    
    @validator('query', 'passage')
    def validate_not_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('Text cannot be empty or only whitespace')
        return v

class SingleRerankRequest(BaseModel):
    query: str = Field(..., description="The query text", min_length=1, max_length=10000)
    passage: str = Field(..., description="The passage text", min_length=1, max_length=10000)
    normalize: Optional[bool] = Field(False, description="Whether to normalize scores to 0-1 range using sigmoid")

class BatchRerankRequest(BaseModel):
    pairs: List[TextPair] = Field(..., description="List of text pairs to rerank", min_items=1, max_items=100)
    normalize: Optional[bool] = Field(False, description="Whether to normalize scores to 0-1 range using sigmoid")

class SingleRerankResponse(BaseModel):
    score: float = Field(..., description="Reranking score")
    normalized: bool = Field(..., description="Whether the score is normalized")
    query_length: int = Field(..., description="Length of the query in characters")
    passage_length: int = Field(..., description="Length of the passage in characters")

class BatchRerankResponse(BaseModel):
    scores: List[float] = Field(..., description="List of reranking scores")
    normalized: bool = Field(..., description="Whether scores are normalized")
    pairs_count: int = Field(..., description="Number of pairs processed")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    version: str = Field(..., description="API version")

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "BAAI Reranker Service",
        "model": os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "version": "1.0.0",
        "endpoints": {
            "/": "Service information",
            "/health": "Health check endpoint",
            "/rerank": "Single text pair reranking",
            "/rerank/batch": "Batch text pairs reranking",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for container monitoring"""
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        version="1.0.0"
    )

@app.post("/rerank", response_model=SingleRerankResponse, tags=["Reranking"])
async def rerank_single(request: SingleRerankRequest):
    """
    Rerank a single text pair
    
    - **query**: The query text
    - **passage**: The passage to rank against the query
    - **normalize**: Apply sigmoid normalization to get 0-1 range
    """
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        score = reranker_model.compute_score(
            [request.query, request.passage],
            normalize=request.normalize
        )
        
        return SingleRerankResponse(
            score=float(score),
            normalized=request.normalize,
            query_length=len(request.query),
            passage_length=len(request.passage)
        )
    except Exception as e:
        logger.error(f"Error during single reranking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {str(e)}"
        )

@app.post("/rerank/batch", response_model=BatchRerankResponse, tags=["Reranking"])
async def rerank_batch(request: BatchRerankRequest):
    """
    Rerank multiple text pairs in batch
    
    - **pairs**: List of text pairs (max 100)
    - **normalize**: Apply sigmoid normalization to get 0-1 range
    """
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert pairs to format expected by the model
        pairs_list = [[pair.query, pair.passage] for pair in request.pairs]
        
        # Compute scores for all pairs
        scores = reranker_model.compute_score(
            pairs_list,
            normalize=request.normalize
        )
        
        # Ensure scores is a list
        if not isinstance(scores, list):
            scores = [scores]
        
        return BatchRerankResponse(
            scores=[float(score) for score in scores],
            normalized=request.normalize,
            pairs_count=len(request.pairs)
        )
    except Exception as e:
        logger.error(f"Error during batch reranking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch reranking failed: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )