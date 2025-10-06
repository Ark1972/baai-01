from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
import os
import sys
from contextlib import asynccontextmanager
from FlagEmbedding import FlagReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global reranker
reranker_model = None

class RerankModel:
    """Wrapper for FlagEmbedding FlagReranker"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.model_name = model_name
        logger.info(f"Loading reranker model: {model_name}")
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        self.device = "cuda" if use_fp16 else "cpu"
        logger.info(f"Model loaded successfully on device: {self.device}")

    def compute_score(self, query: str, passage: str) -> float:
        """
        Compute relevance score for a single query-passage pair

        Args:
            query: The query text
            passage: The passage text to rank

        Returns:
            Relevance score (higher = more relevant)
        """
        score = self.reranker.compute_score([query, passage])
        # FlagReranker returns float for single pair, list for multiple pairs
        return float(score) if not isinstance(score, list) else float(score[0])

    def compute_scores_batch(self, query: str, passages: List[str]) -> List[float]:
        """
        Compute relevance scores for multiple passages against a single query

        Args:
            query: The query text
            passages: List of passage texts to rank

        Returns:
            List of relevance scores (same order as input passages)
        """
        pairs = [[query, passage] for passage in passages]
        scores = self.reranker.compute_score(pairs)
        # FlagReranker returns list for multiple pairs, ensure it's always a list
        return scores if isinstance(scores, list) else [float(scores)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    global reranker_model

    # Load model on startup
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    use_fp16 = os.getenv("USE_FP16", "false").lower() == "true"

    logger.info(f"Initializing reranker model: {model_name}")
    reranker_model = RerankModel(model_name=model_name, use_fp16=use_fp16)
    logger.info("Reranker model ready")

    yield

    # Cleanup
    logger.info("Shutting down reranker model")
    reranker_model = None

# Initialize FastAPI app
app = FastAPI(
    title="BAAI Reranker Service",
    description="Web service for reranking text pairs using BAAI/bge-reranker-v2-m3",
    version="2.0.0",
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
    normalize: Optional[bool] = Field(True, description="Whether to normalize scores to 0-1 range using sigmoid")

class BatchRerankRequest(BaseModel):
    pairs: List[TextPair] = Field(..., description="List of text pairs to rerank", min_items=1, max_items=100)
    normalize: Optional[bool] = Field(True, description="Whether to normalize scores to 0-1 range using sigmoid")

class SingleRerankResponse(BaseModel):
    score: float = Field(..., description="Reranking score")
    normalized: bool = Field(..., description="Whether the score is normalized")
    query_length: int = Field(..., description="Length of the query in characters")
    passage_length: int = Field(..., description="Length of the passage in characters")

class BatchRerankResponse(BaseModel):
    scores: List[float] = Field(..., description="List of reranking scores")
    normalized: bool = Field(..., description="Whether scores are normalized")
    pairs_count: int = Field(..., description="Number of pairs processed")

class QueryRerankRequest(BaseModel):
    query: str = Field(..., description="The query text", min_length=1, max_length=10000)
    passages: List[str] = Field(..., description="List of passages to rerank", min_items=1, max_items=100)
    normalize: Optional[bool] = Field(True, description="Whether to normalize scores to 0-1 range using sigmoid")

    @validator('passages')
    def validate_passages(cls, v):
        for passage in v:
            if not passage or passage.isspace():
                raise ValueError('Passages cannot be empty or only whitespace')
            if len(passage) > 10000:
                raise ValueError('Each passage must be less than 10000 characters')
        return v

class RankedPassage(BaseModel):
    passage: str = Field(..., description="The passage text")
    score: float = Field(..., description="Relevance score (higher = more relevant)")

class QueryRerankResponse(BaseModel):
    re_ranked: List[RankedPassage] = Field(..., description="Passages sorted by relevance score (descending)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    version: str = Field(..., description="API version")

# Utility functions
def normalize_score(score: float) -> float:
    """Apply sigmoid normalization to convert score to 0-1 range"""
    import math
    try:
        return 1 / (1 + math.exp(-score))
    except:
        return 0.5

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "BAAI Reranker Service",
        "model": os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "version": "2.0.0",
        "backend": "FlagEmbedding",
        "endpoints": {
            "/": "Service information",
            "/health": "Health check endpoint",
            "/rerank": "Single text pair reranking",
            "/rerank/batch": "Batch text pairs reranking",
            "/rerank/query": "Query-based passage reranking (sorted by relevance)",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint for model status"""
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model not initialized"
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=reranker_model.model_name,
        device=reranker_model.device,
        version="2.0.0"
    )

@app.post("/rerank", response_model=SingleRerankResponse, tags=["Reranking"])
async def rerank_single(request: SingleRerankRequest):
    """Rerank a single text pair using PyTorch"""
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model not available"
        )

    try:
        score = reranker_model.compute_score(request.query, request.passage)

        if request.normalize:
            score = normalize_score(score)

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
    """Rerank multiple text pairs using PyTorch"""
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model not available"
        )

    try:
        # Group by query to optimize batch processing
        query_groups = {}
        for i, pair in enumerate(request.pairs):
            if pair.query not in query_groups:
                query_groups[pair.query] = []
            query_groups[pair.query].append((i, pair.passage))

        results = [0.0] * len(request.pairs)

        # Process each query group
        for query, passages_with_indices in query_groups.items():
            indices = [idx for idx, _ in passages_with_indices]
            passages = [passage for _, passage in passages_with_indices]

            scores = reranker_model.compute_scores_batch(query, passages)

            # Apply normalization if requested
            if request.normalize:
                scores = [normalize_score(score) for score in scores]

            # Map scores back to original positions
            for idx, score in zip(indices, scores):
                results[idx] = float(score)

        return BatchRerankResponse(
            scores=results,
            normalized=request.normalize,
            pairs_count=len(request.pairs)
        )
    except Exception as e:
        logger.error(f"Error during batch reranking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch reranking failed: {str(e)}"
        )

@app.post("/rerank/query", response_model=QueryRerankResponse, tags=["Reranking"])
async def rerank_query(request: QueryRerankRequest):
    """Rerank multiple passages for a single query, returning sorted results"""
    if reranker_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model not available"
        )

    try:
        # Compute scores for all passages
        scores = reranker_model.compute_scores_batch(request.query, request.passages)

        # Apply normalization if requested
        if request.normalize:
            scores = [normalize_score(score) for score in scores]

        # Combine passages with scores
        ranked_passages = [
            RankedPassage(passage=passage, score=float(score))
            for passage, score in zip(request.passages, scores)
        ]

        # Sort by score descending
        ranked_passages.sort(key=lambda x: x.score, reverse=True)

        return QueryRerankResponse(re_ranked=ranked_passages)
    except Exception as e:
        logger.error(f"Error during query reranking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query reranking failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting PyTorch-based reranker service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
