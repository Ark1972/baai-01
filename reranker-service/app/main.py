from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Union, Optional
import httpx
import logging
import os
import sys
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ollama client
ollama_client = None

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.model_name = os.getenv("MODEL_NAME", "xitao/bge-reranker-v2-m3")
        self.timeout = 60.0
        
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy and model is loaded"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check if Ollama server is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Check if our model is available
                models = response.json()
                model_names = [model["name"] for model in models.get("models", [])]
                
                # More flexible model name matching
                # Check exact match first
                if self.model_name in model_names:
                    return True
                
                # Check if any model contains our base name (handles version tags)
                base_name = self.model_name.split(':')[0]  # Remove tag if present
                for model_name in model_names:
                    if base_name in model_name or model_name.startswith(base_name):
                        logger.info(f"Found model {model_name} matching {base_name}")
                        return True
                
                # If no models found, log available models for debugging
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def rerank(self, query: str, passages: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Rerank query-passage pairs using Ollama
        
        Args:
            query: The query text
            passages: Single passage or list of passages
            
        Returns:
            Single score or list of scores
        """
        try:
            # Prepare input for Ollama
            if isinstance(passages, str):
                # Single passage
                prompt = f"Query: {query}\nPassage: {passages}\nRelevance score:"
                is_single = True
            else:
                # Multiple passages - process as batch
                is_single = False
                prompt = f"Query: {query}\n"
                for i, passage in enumerate(passages):
                    prompt += f"Passage {i+1}: {passage}\n"
                prompt += "Relevance scores:"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_k": 1,
                        "top_p": 0.1
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                raw_response = result.get("response", "")
                
                # Parse response to extract scores
                # Note: This is a simplified parser - in production you'd want more robust parsing
                if is_single:
                    # Extract single score
                    try:
                        # Look for numbers in the response
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', raw_response)
                        if numbers:
                            return float(numbers[0])
                        else:
                            # Fallback: return a default score based on response sentiment
                            return 0.5 if "relevant" in raw_response.lower() else 0.1
                    except:
                        return 0.1
                else:
                    # Extract multiple scores
                    try:
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', raw_response)
                        scores = [float(num) for num in numbers[:len(passages)]]
                        # Pad with default scores if we didn't get enough
                        while len(scores) < len(passages):
                            scores.append(0.1)
                        return scores[:len(passages)]
                    except:
                        return [0.1] * len(passages)
                        
        except Exception as e:
            logger.error(f"Ollama rerank failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reranking failed: {str(e)}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Ollama client lifecycle"""
    global ollama_client
    
    # Initialize Ollama client
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    ollama_client = OllamaClient(f"http://{ollama_host}")
    
    # Wait for Ollama to be ready
    logger.info("Waiting for Ollama server to be ready...")
    for attempt in range(30):  # Wait up to 60 seconds
        if await ollama_client.health_check():
            logger.info("Ollama server is ready")
            
            # Try to preload the model by making a test call
            try:
                logger.info("Testing model with sample query...")
                test_score = await ollama_client.rerank("test query", "test passage")
                logger.info(f"Model test successful, got score: {test_score}")
            except Exception as e:
                logger.warning(f"Model test failed but continuing: {e}")
            
            break
        await asyncio.sleep(2)
        logger.info(f"Waiting for Ollama... attempt {attempt + 1}/30")
    else:
        logger.error("Ollama server failed to become ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Ollama client")
    ollama_client = None

# Initialize FastAPI app
app = FastAPI(
    title="BAAI Reranker Service (Ollama)",
    description="Web service for reranking text pairs using Ollama with xitao/bge-reranker-v2-m3",
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

# Request/Response models (same as before)
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
    ollama_status: bool = Field(..., description="Whether Ollama is available")
    model_name: str = Field(..., description="Name of the loaded model")
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
        "service": "BAAI Reranker Service (Ollama)",
        "model": os.getenv("MODEL_NAME", "xitao/bge-reranker-v2-m3"),
        "version": "1.0.0",
        "backend": "Ollama",
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
    """Health check endpoint for both FastAPI and Ollama"""
    if ollama_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama client not initialized"
        )
    
    ollama_healthy = await ollama_client.health_check()
    
    if not ollama_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama server or model not available"
        )
    
    return HealthResponse(
        status="healthy",
        ollama_status=ollama_healthy,
        model_name=os.getenv("MODEL_NAME", "xitao/bge-reranker-v2-m3"),
        version="1.0.0"
    )

@app.post("/rerank", response_model=SingleRerankResponse, tags=["Reranking"])
async def rerank_single(request: SingleRerankRequest):
    """Rerank a single text pair using Ollama"""
    if ollama_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama client not available"
        )
    
    try:
        score = await ollama_client.rerank(request.query, request.passage)
        
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
    """Rerank multiple text pairs using Ollama"""
    if ollama_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama client not available"
        )
    
    try:
        # Process all pairs with the same query (optimize for common use case)
        # Group by query to optimize Ollama calls
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
            
            scores = await ollama_client.rerank(query, passages)
            if not isinstance(scores, list):
                scores = [scores]
            
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Ollama-based reranker service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)