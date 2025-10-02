#!/usr/bin/env python3
"""
BAAI Reranker Service Client
Example client for interacting with the reranker API
"""

import requests
import json
from typing import List, Dict, Union, Optional
import time
from dataclasses import dataclass


@dataclass
class TextPair:
    """Represents a query-passage text pair"""
    query: str
    passage: str


class RerankerClient:
    """Client for BAAI Reranker Service API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the reranker client
        
        Args:
            base_url: Base URL of the reranker service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict:
        """Check if the service is healthy"""
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def rerank_single(
        self,
        query: str,
        passage: str,
        normalize: bool = False
    ) -> Dict:
        """
        Rerank a single query-passage pair
        
        Args:
            query: The query text
            passage: The passage to rank
            normalize: Whether to normalize scores to 0-1 range
            
        Returns:
            Dict containing score and metadata
        """
        payload = {
            "query": query,
            "passage": passage,
            "normalize": normalize
        }
        
        response = self.session.post(
            f"{self.base_url}/rerank",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def rerank_batch(
        self,
        pairs: List[TextPair],
        normalize: bool = False
    ) -> Dict:
        """
        Rerank multiple query-passage pairs in batch
        
        Args:
            pairs: List of TextPair objects
            normalize: Whether to normalize scores to 0-1 range
            
        Returns:
            Dict containing scores and metadata
        """
        payload = {
            "pairs": [
                {"query": pair.query, "passage": pair.passage}
                for pair in pairs
            ],
            "normalize": normalize
        }
        
        response = self.session.post(
            f"{self.base_url}/rerank/batch",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Example usage of the reranker client"""
    
    # Configure the service URL (use environment variable or default)
    import os
    service_url = os.getenv("RERANKER_SERVICE_URL", "http://localhost:8000")
    
    # Create client
    with RerankerClient(service_url) as client:
        
        # Health check
        print("🔍 Checking service health...")
        health = client.health_check()
        print(f"✅ Service status: {health['status']}")
        print(f"   Model: {health['model_name']}")
        print()
        
        # Example 1: Single text pair reranking
        print("📝 Example 1: Single text pair reranking")
        print("-" * 50)
        
        query = "What is machine learning?"
        passage = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        
        result = client.rerank_single(query, passage)
        print(f"Query: {query}")
        print(f"Passage: {passage[:100]}...")
        print(f"Score: {result['score']:.4f}")
        print(f"Normalized: {result['normalized']}")
        print()
        
        # Example 2: Single pair with normalization
        print("📝 Example 2: Single pair with normalization")
        print("-" * 50)
        
        result_normalized = client.rerank_single(query, passage, normalize=True)
        print(f"Normalized Score: {result_normalized['score']:.4f}")
        print()
        
        # Example 3: Batch reranking
        print("📝 Example 3: Batch reranking")
        print("-" * 50)
        
        pairs = [
            TextPair(
                query="What is Python?",
                passage="Python is a high-level programming language known for its simplicity."
            ),
            TextPair(
                query="What is Python?",
                passage="A python is a large non-venomous snake found in Africa, Asia, and Australia."
            ),
            TextPair(
                query="How to cook pasta?",
                passage="To cook pasta, boil water, add salt, add pasta, and cook for 8-12 minutes."
            ),
            TextPair(
                query="How to cook pasta?",
                passage="Python is a programming language used for web development."
            )
        ]
        
        batch_result = client.rerank_batch(pairs)
        print(f"Processed {batch_result['pairs_count']} pairs:")
        for i, (pair, score) in enumerate(zip(pairs, batch_result['scores'])):
            print(f"{i+1}. Query: {pair.query[:30]}...")
            print(f"   Passage: {pair.passage[:50]}...")
            print(f"   Score: {score:.4f}")
        print()
        
        # Example 4: Batch with normalization
        print("📝 Example 4: Batch reranking with normalization")
        print("-" * 50)
        
        batch_normalized = client.rerank_batch(pairs, normalize=True)
        print("Normalized scores:")
        for i, score in enumerate(batch_normalized['scores']):
            print(f"{i+1}. Score: {score:.4f} (0-1 range)")
        print()
        
        # Performance test
        print("⚡ Performance Test")
        print("-" * 50)
        
        start_time = time.time()
        for _ in range(10):
            client.rerank_single(query, passage)
        elapsed = time.time() - start_time
        
        print(f"10 single requests: {elapsed:.2f}s")
        print(f"Average per request: {elapsed/10:.3f}s")
        
        # Batch performance
        start_time = time.time()
        client.rerank_batch(pairs * 10)  # 40 pairs total
        elapsed = time.time() - start_time
        
        print(f"1 batch request (40 pairs): {elapsed:.2f}s")
        print(f"Average per pair: {elapsed/40:.3f}s")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the reranker service.")
        print("   Make sure the service is running on http://localhost:8000")
        print("   Or set RERANKER_SERVICE_URL environment variable.")
    except Exception as e:
        print(f"❌ Error: {e}")